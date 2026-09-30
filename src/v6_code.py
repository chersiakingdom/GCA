# %% [code] 0. 스레드 설정 (numpy import 전에 실행되어야 병렬 워커가 서로 CPU를 뺏지 않습니다)
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "4")

# %% [code] 1. 구현 (수정할 필요 없음)
import gc, glob, hashlib, json, math, re, sys, time, itertools
import multiprocessing as mp
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from scipy import optimize
from scipy.spatial import cKDTree
import scipy.fft as sfft

if "CFG" not in globals():
    CFG = {}
VERSION = "6.0-relative-graph-matching"

# ----------------------------------------------------------------------------
# 로그 / 진행률 (단계 전환 + 10% 간격 또는 최소 N초 간격으로만 출력)
# ----------------------------------------------------------------------------
_T0 = time.time()


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')} +{(time.time()-_T0)/60:5.1f}m] {msg}", flush=True)


def fmt_sec(s):
    s = max(0, int(s))
    return f"{s//3600}h{(s%3600)//60:02d}m" if s >= 3600 else f"{s//60}m{s%60:02d}s"


class Progress:
    def __init__(self, total, label, every_frac=0.25, min_sec=None):
        self.total = max(int(total), 1); self.label = label; self.done = 0
        self.every = every_frac; self.next = every_frac; self.t0 = time.time()
        self.min_sec = CFG.get("PROGRESS_MIN_SEC", 300) if min_sec is None else min_sec
        self.last = self.t0

    def update(self, n=1):
        self.done += n
        f = self.done / self.total; now = time.time()
        if f >= self.next or (now - self.last) >= self.min_sec:
            el = now - self.t0
            eta = el / max(f, 1e-9) * (1 - f)
            if f < 1.0:
                log(f"    {self.label}: {100*f:3.0f}%  (경과 {fmt_sec(el)}, 이 단계 남은 예상 {fmt_sec(eta)})")
            while self.next <= f:
                self.next += self.every
            self.last = now


def stage(title):
    log("=" * 8 + f" {title} " + "=" * 8)


# ----------------------------------------------------------------------------
# JSON 입출력: [[x,y,z], ...] (추가 열 허용) 또는 {"xyz"|"points"|"coordinates": [...]}
# ----------------------------------------------------------------------------
_TBL = bytes.maketrans(b"[],", b"   ")


class PointFile:
    """원본 JSON의 구조를 기억해 같은 형식으로 다시 쓰기 위한 객체."""

    def __init__(self, path, raw, kind, key=None, container=None, col_is_int=None):
        self.path, self.raw, self.kind, self.key, self.container = str(path), raw, kind, key, container
        self.col_is_int = col_is_int

    @property
    def ncol(self):
        return self.raw.shape[1]


def _parse_list_of_lists(path, chunk_bytes=256 << 20):
    with open(path, "rb") as f:
        head = f.read(1 << 16)
    m = re.search(rb"\[\s*\[([^\[\]]*)\]", head)
    if m is None:
        raise ValueError("첫 행을 읽지 못했습니다: " + str(path))
    ncol = len(m.group(1).split(b","))
    size = os.path.getsize(path); prog = Progress(size, "JSON 읽기 " + Path(path).name, 0.25, 600)
    parts, carry = [], b""
    with open(path, "rb") as f:
        while True:
            buf = f.read(chunk_bytes)
            if not buf:
                break
            prog.update(len(buf))
            buf = carry + buf
            k = buf.rfind(b"]")
            if k < 0:
                carry = buf; continue
            part, carry = buf[:k + 1], buf[k + 1:]
            arr = np.fromstring(part.translate(_TBL).decode("ascii"), dtype=np.float64, sep=" ")
            if arr.size:
                parts.append(arr)
    if carry.strip(b" \t\r\n],"):
        arr = np.fromstring(carry.translate(_TBL).decode("ascii"), dtype=np.float64, sep=" ")
        if arr.size:
            parts.append(arr)
    flat = np.concatenate(parts) if parts else np.empty(0)
    del parts
    if flat.size % ncol:
        raise ValueError(f"숫자 개수({flat.size})가 열 수({ncol})의 배수가 아닙니다: {path}")
    return flat.reshape(-1, ncol)


def load_points_json(path):
    with open(path, "rb") as f:
        head = f.read(1 << 16).lstrip()
    if re.match(rb"\[\s*\[", head):
        raw = _parse_list_of_lists(path); kind, key, cont = "list", None, None
    else:
        try:
            import orjson
            data = orjson.loads(open(path, "rb").read())
        except ImportError:
            data = json.load(open(path))
        if isinstance(data, dict):
            key = next((k for k in ("xyz", "points", "coordinates") if k in data), None)
            if key is None:
                key = max((k for k, v in data.items() if isinstance(v, list)), key=lambda k: len(data[k]))
            raw = np.asarray(data[key], np.float64); cont = {k: v for k, v in data.items() if k != key}
            kind = "dict"
        else:
            raw = np.asarray(data, np.float64); kind, key, cont = "list", None, None
        del data
    if raw.ndim != 2 or raw.shape[1] < 3:
        raise ValueError(f"Nx3 이상의 좌표 배열이 필요합니다: {path}, shape={raw.shape}")
    if not np.isfinite(raw).all():
        raise ValueError(f"NaN/inf 좌표가 있습니다: {path}")
    samp = raw[:: max(1, len(raw) // 2_000_000)]
    col_is_int = [bool(np.all(samp[:, c] == np.round(samp[:, c]))) for c in range(raw.shape[1])]
    return PointFile(path, raw, kind, key, cont, col_is_int)


def write_like(template: PointFile, rows: np.ndarray, path, chunk=1_000_000):
    """template과 같은 구조/열 형식으로 rows(N, ncol)를 JSON으로 저장."""
    path = Path(path); tmp = path.with_suffix(path.suffix + ".tmp")
    fmts = ["%d" if ii else "%.10g" for ii in template.col_is_int]
    row_fmt = "[" + ", ".join(fmts) + "]"
    rows = np.asarray(rows, np.float64)
    for c, ii in enumerate(template.col_is_int):
        if ii:
            rows[:, c] = np.round(rows[:, c])
    with open(tmp, "w") as f:
        if template.kind == "dict":
            f.write("{")
            for k, v in template.container.items():
                f.write(json.dumps(k) + ": " + json.dumps(v) + ", ")
            f.write(json.dumps(template.key) + ": ")
        f.write("[")
        for s in range(0, len(rows), chunk):
            blk = rows[s:s + chunk].tolist()
            f.write((",\n" if s else "\n") + ",\n".join(row_fmt % tuple(r) for r in blk))
        f.write("\n]")
        if template.kind == "dict":
            f.write("}")
    tmp.replace(path)


# ----------------------------------------------------------------------------
# TIFF 정보 / 좌표 순서 확인
# ----------------------------------------------------------------------------
def natural_key(s):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]


def tiff_list(d, natural=False):
    files = sorted(glob.glob(os.path.join(d, "*.tif")) + glob.glob(os.path.join(d, "*.tiff")),
                   key=natural_key if natural else None)
    return files


def tiff_meta(d, natural=False):
    files = tiff_list(d, natural)
    if not files:
        return None
    try:
        import tifffile
        with tifffile.TiffFile(files[0]) as t:
            shp = t.pages[0].shape
    except Exception:
        from PIL import Image
        im = Image.open(files[0]); shp = (im.height, im.width)
    return dict(files=files, nz=len(files), H=int(shp[0]), W=int(shp[1]))


def read_slice(files, z):
    try:
        import tifffile
        return tifffile.imread(files[int(z)])
    except ImportError:
        from PIL import Image
        return np.asarray(Image.open(files[int(z)]))


def coord_columns(pf: PointFile, meta, order):
    """(x,y,z)가 raw의 몇 번째 열인지 반환."""
    if order != "auto":
        return tuple(order.lower().index(c) for c in "xyz")
    mx = pf.raw[:, :3].max(0)
    if meta is None:
        log("  (TIFF 정보 없음) 좌표 순서를 XYZ로 가정합니다.")
        return (0, 1, 2)
    lim = {"x": meta["W"], "y": meta["H"], "z": meta["nz"]}
    for order_try in ("xyz", "zyx", "yxz", "xzy", "yzx", "zxy"):
        cols = tuple(order_try.index(c) for c in "xyz")
        if all(mx[cols[i]] <= lim[a] * 1.001 + 1 for i, a in enumerate("xyz")):
            return cols
    log("  [경고] 어떤 좌표 순서도 TIFF 크기와 맞지 않습니다. XYZ로 가정합니다.")
    return (0, 1, 2)


# ----------------------------------------------------------------------------
# 공간 버킷 (z가 가장 느린 축으로 정렬 → z 구간이 연속 메모리)
# ----------------------------------------------------------------------------
class Buckets:
    def __init__(self, P_um, size, chunk=20_000_000):
        P_um = np.asarray(P_um, np.float32)
        self.size = float(size)
        self.lo = P_um.min(0).astype(np.float64) - 1e-3
        hi = P_um.max(0).astype(np.float64)
        self.shape = tuple((np.floor((hi - self.lo) / size).astype(np.int64) + 1).tolist())
        nx, ny, nz = self.shape
        key = np.empty(len(P_um), np.int64)
        for s in range(0, len(P_um), chunk):
            ijk = np.floor((P_um[s:s + chunk] - self.lo) / size).astype(np.int64)
            key[s:s + chunk] = (ijk[:, 2] * ny + ijk[:, 1]) * nx + ijk[:, 0]
        self.order = np.argsort(key, kind="stable")
        cnt = np.bincount(key, minlength=nx * ny * nz)
        del key
        self.start = np.zeros(nx * ny * nz + 1, np.int64); np.cumsum(cnt, out=self.start[1:])
        self.P = P_um[self.order]

    def _cell(self, x):
        return np.floor((np.asarray(x, np.float64) - self.lo) / self.size).astype(np.int64)

    def query_box(self, lo, hi, exact=True):
        """[lo, hi) 박스 안의 (정렬된 배열 기준) 인덱스."""
        nx, ny, nz = self.shape
        i0 = np.maximum(self._cell(lo), 0); i1 = np.minimum(self._cell(hi), np.array(self.shape) - 1)
        if np.any(i1 < i0):
            return np.empty(0, np.int64)
        iz, iy = np.meshgrid(np.arange(i0[2], i1[2] + 1), np.arange(i0[1], i1[1] + 1), indexing="ij")
        base = (iz.ravel() * ny + iy.ravel()) * nx
        a = self.start[base + i0[0]]; b = self.start[base + i1[0] + 1]
        m = b > a
        a, b = a[m], b[m]
        if not len(a):
            return np.empty(0, np.int64)
        ln = b - a
        idx = np.repeat(a - np.r_[0, np.cumsum(ln)[:-1]], ln) + np.arange(ln.sum())
        if exact:
            p = self.P[idx]
            ok = np.all((p >= lo) & (p < hi), axis=1)
            idx = idx[ok]
        return idx


# ----------------------------------------------------------------------------
# 변환 모델: y = A x + b + Σ_level u_level(x)   (x: R1 µm, y: R2 µm)
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# 뇌 좌표계: 각 라운드의 점을 "그 뇌 자신의" 좌표로 변환 (원점 = 뇌 중심, 축 = 뇌의 주축)
#   x' = (x - c) · E      (c: 중심, E: 주축 3개. 원래 영상 축에 가장 가까운 순서/방향으로 맞춤)
#   회전·이동만 하므로 세포 사이 거리와 이웃 관계는 그대로 유지됨 (크기 정규화는 하지 않음: µm 유지)
# ----------------------------------------------------------------------------
class BrainFrame:
    def __init__(self, c, E):
        self.c = np.asarray(c, np.float64); self.E = np.asarray(E, np.float64)

    @classmethod
    def fit(cls, P_um, n=20_000_000, seed=0):
        rng = np.random.default_rng(seed)
        S = P_um[rng.integers(0, len(P_um), min(n, len(P_um)))].astype(np.float64)
        lo, hi = np.percentile(S, 0.5, axis=0), np.percentile(S, 99.5, axis=0)
        S = S[np.all((S >= lo) & (S <= hi), axis=1)]          # 뇌 밖 이상점 제외
        c = S.mean(0)
        _, V = np.linalg.eigh(np.cov((S - c).T))
        best, bt = None, -np.inf
        for perm in itertools.permutations(range(3)):          # 영상 축에 가장 가까운 순서·부호 선택
            for sg in itertools.product((1.0, -1.0), repeat=3):
                E = V[:, perm] * np.array(sg)
                if np.linalg.det(E) > 0 and np.trace(E) > bt:
                    best, bt = E, np.trace(E)
        return cls(c, best)

    def to(self, X):
        return ((np.asarray(X, np.float64) - self.c) @ self.E).astype(np.float32)

    def back(self, Xp):
        return np.asarray(Xp, np.float64) @ self.E.T + self.c

    def tilt_deg(self):
        return float(np.rad2deg(np.arccos(np.clip(abs(self.E[2, 2]), -1, 1))))


class Warp:
    def __init__(self, A=None, b=None):
        self.A = np.eye(3) if A is None else np.asarray(A, float)
        self.b = np.zeros(3) if b is None else np.asarray(b, float)
        self.levels = []   # dict(origin(3), step, U(nx,ny,nz,3) float32)
        self.base = None   # (원래 좌표계 Warp, BrainFrame R1, BrainFrame R2): 이전 실행 변형장을 뇌 좌표계에서 재사용
        self.base_path = None

    @classmethod
    def from_raw(cls, path, f1, f2):
        """원래 좌표계(µm)에서 만든 변형장(예: v5 warp_final.npz)을 뇌 좌표계에서 쓰도록 감쌈."""
        w = cls(); w.base = (Warp.load(path), f1, f2); w.base_path = str(path); return w

    def copy(self):
        w = Warp(self.A.copy(), self.b.copy()); w.levels = [dict(L) for L in self.levels]; return w

    def apply(self, X, chunk=2_000_000):
        X = np.asarray(X)
        out = np.empty((len(X), 3), np.float32)
        for s in range(0, len(X), chunk):
            x = X[s:s + chunk].astype(np.float64)
            if self.base is not None:
                inner, f1, f2 = self.base
                y = f2.to(inner.apply(f1.back(x))).astype(np.float64)
            else:
                y = x @ self.A.T + self.b
            for L in self.levels:
                g = ((x - L["origin"]) / L["step"]).T
                for c in range(3):
                    y[:, c] += ndi.map_coordinates(L["U"][..., c], g, order=1, mode="nearest")
            out[s:s + chunk] = y
        return out

    def save(self, path):
        d = dict(A=self.A, b=self.b, n_levels=len(self.levels))
        if self.base is not None:
            _, f1, f2 = self.base
            d.update(base_path=str(self.base_path), f1_c=f1.c, f1_E=f1.E, f2_c=f2.c, f2_E=f2.E)
        for i, L in enumerate(self.levels):
            d[f"L{i}_origin"] = L["origin"]; d[f"L{i}_step"] = L["step"]; d[f"L{i}_U"] = L["U"]
        np.savez_compressed(path, **d)

    @classmethod
    def load(cls, path):
        f = np.load(path)
        w = cls(f["A"], f["b"])
        if "base_path" in f:
            w.base = (Warp.load(str(f["base_path"])), BrainFrame(f["f1_c"], f["f1_E"]), BrainFrame(f["f2_c"], f["f2_E"]))
            w.base_path = str(f["base_path"])
        for i in range(int(f["n_levels"])):
            w.levels.append(dict(origin=f[f"L{i}_origin"], step=float(f[f"L{i}_step"]), U=f[f"L{i}_U"]))
        return w


# ----------------------------------------------------------------------------
# 1단계: 전역 3D affine (밀도 볼륨 NCC). "뇌 전체를 감싸는 상자/축 비교" 아이디어의 견고한 형태
# ----------------------------------------------------------------------------
def _grid_for(P, bin_um, pad=6):
    lo = np.percentile(P, 0.02, axis=0) - pad * bin_um
    hi = np.percentile(P, 99.98, axis=0) + pad * bin_um
    shape = np.ceil((hi - lo) / bin_um).astype(int) + 1
    return lo, shape


def _density(P, lo, bin_um, shape, smooth=1.0):
    ijk = np.floor((P - lo) / bin_um).astype(np.int64)
    ok = np.all((ijk >= 0) & (ijk < shape), axis=1)
    ijk = ijk[ok]
    flat = (ijk[:, 0] * shape[1] + ijk[:, 1]) * shape[2] + ijk[:, 2]
    D = np.log1p(np.bincount(flat, minlength=int(np.prod(shape))).reshape(shape).astype(np.float32))
    # 밴드패스(DoG): 조직 외곽 모양뿐 아니라 내부 밀도 패턴(층/뇌실/섬유다발)이 정합을 이끌도록
    return ndi.gaussian_filter(D, smooth) - ndi.gaussian_filter(D, 4 * smooth)


def _density_raw(P, lo, bin_um, shape):
    ijk = np.floor((P - lo) / bin_um).astype(np.int64)
    ok = np.all((ijk >= 0) & (ijk < shape), axis=1); ijk = ijk[ok]
    flat = (ijk[:, 0] * shape[1] + ijk[:, 1]) * shape[2] + ijk[:, 2]
    return ndi.gaussian_filter(np.bincount(flat, minlength=int(np.prod(shape))).reshape(shape).astype(np.float32), 1.0)


def _centers(lo, bin_um, shape, idx):
    return lo + (np.stack(np.unravel_index(idx, shape), 1) + 0.5) * bin_um


def _sample_vals(D, lo, bin_um, Y):
    g = ((Y - lo) / bin_um - 0.5).T
    return ndi.map_coordinates(D, g, order=1, mode="constant", cval=0.0)


def _ncc(a, b):
    a = a - a.mean(); b = b - b.mean()
    return float(np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12))


def _rot_z(deg):
    t = np.deg2rad(deg); c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def _fft_translation(D1, M):
    """D1(x) ≈ M(x+s) 인 s (voxel)."""
    shp = [int(sfft.next_fast_len(2 * n)) for n in D1.shape]
    a = M - M.mean(); b = D1 - D1.mean()
    cc = sfft.irfftn(sfft.rfftn(a, shp) * np.conj(sfft.rfftn(b, shp)), shp)
    k = np.array(np.unravel_index(np.argmax(cc), cc.shape))
    return np.where(k > np.array(shp) // 2, k - np.array(shp), k).astype(float)


def global_affine(P1, P2, cfg):
    rng = np.random.default_rng(0)
    n = cfg["GLOBAL_SUBSAMPLE"]
    S1 = P1[rng.integers(0, len(P1), min(n, len(P1)))].astype(np.float64)
    S2 = P2[rng.integers(0, len(P2), min(n, len(P2)))].astype(np.float64)

    def trimmed(S):
        lo, hi = np.percentile(S, 0.5, axis=0), np.percentile(S, 99.5, axis=0)
        return S[np.all((S >= lo) & (S <= hi), axis=1)]
    T1, T2 = trimmed(S1), trimmed(S2)
    c1, c2 = T1.mean(0), T2.mean(0)
    _, E1 = np.linalg.eigh(np.cov(T1.T)); _, E2 = np.linalg.eigh(np.cov(T2.T))
    E1, E2 = E1[:, ::-1], E2[:, ::-1]
    zext = np.ptp(T1[:, 2])

    # 후보 회전: 동일(기본), 180° 뒤집힘, z축 평면 회전 sweep, PCA 축 정렬(부호 4종)
    cands = [("identity", np.eye(3))]
    for nm, R in (("flipX180", np.diag([1., -1, -1])), ("flipY180", np.diag([-1., 1, -1])),
                  ("flipZ180", np.diag([-1., -1, 1]))):
        cands.append((nm, R))
    for ang in range(15, 360, 15):
        cands.append((f"rotZ{ang}", _rot_z(ang)))
    for s in itertools.product((1., -1.), repeat=3):
        R = E2 @ np.diag(s) @ E1.T
        if np.linalg.det(R) > 0:
            cands.append((f"pca{''.join('+' if v > 0 else '-' for v in s)}", R))
    if cfg["ALLOW_REFLECTION"]:
        cands += [(nm + "_mirX", R @ np.diag([-1., 1, 1])) for nm, R in list(cands)]

    def score_setup(bin_um):
        lo1, sh1 = _grid_for(S1, bin_um); lo2, sh2 = _grid_for(S2, bin_um)
        D1 = _density(S1, lo1, bin_um, sh1); D2 = _density(S2, lo2, bin_um, sh2)
        cnt = _density_raw(S1, lo1, bin_um, sh1)
        tissue = cnt > 0.2 * np.median(cnt[cnt > 0])
        mask = ndi.binary_dilation(tissue, iterations=3)
        idx = np.flatnonzero(mask.ravel())
        if len(idx) > cfg["GLOBAL_NCC_SAMPLES"]:
            idx = rng.choice(idx, cfg["GLOBAL_NCC_SAMPLES"], replace=False)
        X = _centers(lo1, bin_um, sh1, idx); f = D1.ravel()[idx]
        return dict(bin=bin_um, lo1=lo1, sh1=sh1, lo2=lo2, D1=D1, D2=D2, X=X, f=f)

    def translation_for(R, G):
        # M(x1) = D2(c2 + R(x1 - c1)); D1(x) ≈ M(x+s) → b = c2 - R c1 + R s
        allidx = np.arange(int(np.prod(G["sh1"])))
        Xg = _centers(G["lo1"], G["bin"], G["sh1"], allidx)
        M = _sample_vals(G["D2"], G["lo2"], G["bin"], (Xg - c1) @ R.T + c2).reshape(G["D1"].shape)
        s = _fft_translation(G["D1"], M) * G["bin"]
        return c2 - R @ c1 + R @ s

    def ncc_of(A, b, G):
        return _ncc(G["f"], _sample_vals(G["D2"], G["lo2"], G["bin"], G["X"] @ A.T + b))

    bins = list(cfg["GLOBAL_BINS_UM"])
    G = score_setup(bins[0])
    scored = []
    for nm, R in cands:
        b = translation_for(R, G)
        scored.append((ncc_of(R, b, G), nm, R, b))
    scored.sort(key=lambda t: -t[0])
    log("  회전 후보 상위 5개 (NCC): " + ", ".join(f"{nm}={sc:.3f}" for sc, nm, _, _ in scored[:5]))

    freeze_z = zext < cfg["GLOBAL_MIN_Z_EXTENT_UM"]
    if freeze_z:
        log(f"  R1 z 두께 {zext:.0f} µm < {cfg['GLOBAL_MIN_Z_EXTENT_UM']} µm → z 기울기/배율은 고정하고 XY affine+이동만 추정")

    def refine(A0, b0, G, maxfev):
        def unpack(p):
            L = p[:9].reshape(3, 3) * 0.01
            if freeze_z:
                L[2, :] = 0; L[:, 2] = 0
            A = A0 @ (np.eye(3) + L)
            b = b0 + A0 @ c1 - A @ c1 + p[9:] * G["bin"]
            return A, b
        fun = lambda p: -ncc_of(*unpack(p), G)
        r = optimize.minimize(fun, np.zeros(12), method="Powell",
                              options=dict(xtol=1e-2, ftol=1e-5, maxfev=maxfev))
        A, b = unpack(r.x)
        return -r.fun, A, b

    best = None
    for sc, nm, R, b in scored[:3]:
        v, A, bb = refine(R, b, G, cfg["GLOBAL_MAXFEV"])
        log(f"  후보 {nm}: NCC {sc:.3f} → affine 보정 후 {v:.3f}")
        if best is None or v > best[0]:
            best = (v, A, bb, nm)
    v, A, b, nm = best
    for bin_um in bins[1:]:
        G = score_setup(bin_um)
        v, A, b = refine(A, b, G, cfg["GLOBAL_MAXFEV"])
        log(f"  bin {bin_um:g} µm: NCC {v:.3f}")
    U_, sv, Vt = np.linalg.svd(A)
    rot = U_ @ Vt
    ang = np.rad2deg(np.arccos(np.clip((np.trace(rot) - 1) / 2, -1, 1)))
    tilt = np.rad2deg(np.arccos(np.clip(abs(rot[2, 2]), -1, 1)))
    info = dict(candidate=nm, ncc=v, A=A.tolist(), b=b.tolist(), scales=sv.tolist(),
                rotation_deg=float(ang), z_axis_tilt_deg=float(tilt),
                center_shift_um=(A @ c1 + b - c1).tolist())
    log(f"  전역 affine: 후보 {nm}, NCC {v:.3f}, 회전 {ang:.2f}°, z축 기울기 {tilt:.2f}°, "
        f"배율 {np.round(sv, 4).tolist()}, 중심 이동 {np.round(info['center_shift_um'], 1).tolist()} µm")
    if v < cfg["GLOBAL_NCC_WARN"]:
        log(f"  [경고] 전역 NCC {v:.3f}가 낮습니다. global_overlay.png를 확인하세요 "
            f"(좌표 순서/spacing/뒤집힘 문제일 수 있음).")
    return Warp(A, b), info, G


# ----------------------------------------------------------------------------
# 병렬 워커 공용 상태 (fork로 복사 없이 공유)
# ----------------------------------------------------------------------------
_G = {}


def _pool(n):
    ctx = mp.get_context("fork")
    return ctx.Pool(n)


def _splat(P, lo, vox, shape):
    ijk = np.floor((P - lo) / vox + 0.5).astype(np.int64)
    ok = np.all((ijk >= 0) & (ijk < shape), axis=1); ijk = ijk[ok]
    flat = (ijk[:, 0] * shape[1] + ijk[:, 1]) * shape[2] + ijk[:, 2]
    return np.bincount(flat, minlength=int(np.prod(shape))).reshape(shape).astype(np.float32)


def _subpix(win, p):
    off = np.zeros(3)
    for ax in range(3):
        if 0 < p[ax] < win.shape[ax] - 1:
            i = list(p); i[ax] -= 1; a = win[tuple(i)]; i[ax] += 2; c = win[tuple(i)]; b = win[tuple(p)]
            den = a - 2 * b + c
            if den < 0:
                off[ax] = float(np.clip(0.5 * (a - c) / den, -0.5, 0.5))
    return off


# 블록 코드: 0 정상, 1 R1 부족, 2 R2 부족, 3 탐색경계, 4 피크 약함
BLOCK_CODES = {0: "ok", 1: "few_r1", 2: "few_r2", 3: "search_boundary", 4: "weak_peak"}


def _block_task(node_ids):
    g = _G; L = g["level"]; B1, B2, W = g["B1"], g["B2"], g["warp"]
    h = L["block"] / 2; s = L["search"]; vox = L["vox"]; sv = int(np.ceil(s / vox))
    out = np.zeros((len(node_ids), 5), np.float32)   # dx,dy,dz,z,code
    rng = np.random.default_rng(int(node_ids[0]))
    for k, n in enumerate(node_ids):
        c = g["centers"][n]
        i1 = B1.query_box(c - h, c + h)
        if len(i1) < L["min_pts"]:
            out[k, 4] = 1; continue
        if len(i1) > L["max_pts"]:
            i1 = rng.choice(i1, L["max_pts"], replace=False)
        Q = W.apply(B1.P[i1])
        lo = Q.min(0) - s - 2 * vox; hi = Q.max(0) + s + 2 * vox
        i2 = B2.query_box(lo, hi)
        if len(i2) < 0.15 * len(i1):
            out[k, 4] = 2; continue
        shape = np.ceil((hi - lo) / vox).astype(int) + 1
        sig = L["sigma"] / vox
        S1_ = _splat(Q, lo, vox, shape); S2_ = _splat(B2.P[i2], lo, vox, shape)
        # 밴드패스: 넓은 밀도 기울기가 피크를 탐색 경계로 끌고 가는 것을 방지
        I1 = ndi.gaussian_filter(S1_, sig) - ndi.gaussian_filter(S1_, 4 * sig)
        I2 = ndi.gaussian_filter(S2_, sig) - ndi.gaussian_filter(S2_, 4 * sig)
        inner = np.zeros(shape, bool)
        inner[sv:shape[0] - sv, sv:shape[1] - sv, sv:shape[2] - sv] = True
        I1[~inner] = 0
        cc = sfft.irfftn(sfft.rfftn(I2, workers=1) * np.conj(sfft.rfftn(I1, workers=1)), shape, workers=1)
        cc = np.fft.fftshift(cc)
        ct = np.array(shape) // 2
        win = cc[ct[0] - sv:ct[0] + sv + 1, ct[1] - sv:ct[1] + sv + 1, ct[2] - sv:ct[2] + sv + 1]
        p = np.array(np.unravel_index(np.argmax(win), win.shape))
        med = np.median(win); mad = np.median(np.abs(win - med)) * 1.4826 + 1e-12
        z = (win[tuple(p)] - med) / mad
        out[k, 3] = z
        if np.any(p == 0) or np.any(p == np.array(win.shape) - 1):
            out[k, 4] = 3; continue
        if z < L["z_min"]:
            out[k, 4] = 4; continue
        out[k, :3] = (p - sv + _subpix(win, p)) * vox
    return node_ids, out


def _nan_median_neighbors(D, valid):
    """3x3x3 이웃(자기 제외) 중앙값, 성분별."""
    nx, ny, nz, _ = D.shape
    med = np.full(D.shape, np.nan, np.float32); cnt = np.zeros(D.shape[:3], np.int16)
    offs = [o for o in itertools.product((-1, 0, 1), repeat=3) if o != (0, 0, 0)]
    Dp = np.pad(np.where(valid[..., None], D, np.nan), ((1, 1), (1, 1), (1, 1), (0, 0)), constant_values=np.nan)
    for c in range(3):
        stack = np.stack([Dp[1 + o[0]:1 + o[0] + nx, 1 + o[1]:1 + o[1] + ny, 1 + o[2]:1 + o[2] + nz, c] for o in offs], 0)
        with np.errstate(all="ignore"):
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                med[..., c] = np.nanmedian(stack, 0)
        if c == 0:
            cnt = np.isfinite(stack).sum(0)
        del stack
    return med, cnt


def regularize_field(D, W, outlier_um, sigma_nodes=1.0):
    """이상치 제거 + 정규화 컨볼루션으로 결측 블록 채우기 + 평활화. 실패로 멈추지 않음."""
    valid = W > 0
    if valid.sum() >= 5:
        med, cnt = _nan_median_neighbors(D, valid)
        dev = np.linalg.norm(np.nan_to_num(D - med), axis=-1)
        bad = valid & (cnt >= 3) & (dev > outlier_um)
        W = W.copy(); W[bad] = 0; valid = W > 0
    else:
        bad = np.zeros_like(valid)
    U = np.zeros(D.shape, np.float32); filled = np.zeros(D.shape[:3], bool)
    Dz = np.nan_to_num(D)
    for sig in (sigma_nodes, 2 * sigma_nodes, 4 * sigma_nodes, 8 * sigma_nodes):
        den = ndi.gaussian_filter(W, sig, mode="constant")
        ok = (den > 1e-3 * max(W.max(), 1e-9)) & ~filled
        if ok.any():
            for c in range(3):
                num = ndi.gaussian_filter(W * Dz[..., c], sig, mode="constant")
                U[..., c][ok] = num[ok] / den[ok]
            filled |= ok
    return U, int(bad.sum())


def run_block_level(B1, B2, warp, L, cfg):
    lo = B1.P.min(0).astype(np.float64); hi = B1.P.max(0).astype(np.float64)
    step = L["step"]
    gshape = tuple((np.ceil((hi - lo) / step).astype(int) + 1).tolist())
    origin = lo + 0.5 * step
    # 조직이 있는 격자점만 계산
    cell = np.floor((B1.P[:: max(1, len(B1.P) // 20_000_000)] - lo) / step).astype(np.int64)
    cell = np.clip(cell, 0, np.array(gshape) - 1)
    occ = np.zeros(gshape, bool); occ[cell[:, 0], cell[:, 1], cell[:, 2]] = True
    rad = max(1, int(round(L["block"] / step / 2)) - 0)
    occ = ndi.binary_dilation(occ, iterations=rad) if rad > 0 else occ
    nodes = np.flatnonzero(occ.ravel())
    centers = origin + np.stack(np.unravel_index(np.arange(int(np.prod(gshape))), gshape), 1) * step
    _G.update(B1=B1, B2=B2, warp=warp, level=L, centers=centers)
    res = np.zeros((int(np.prod(gshape)), 5), np.float32); res[:, 4] = 1
    tasks = [nodes[i:i + L["batch"]] for i in range(0, len(nodes), L["batch"])]
    prog = Progress(len(nodes), f"{L['name']} 블록 ({len(nodes):,}개)")
    with _pool(cfg["N_WORKERS"]) as pool:
        for ids, out in pool.imap_unordered(_block_task, tasks):
            res[ids] = out; prog.update(len(ids))
    code = res[:, 4].astype(int); zsc = res[:, 3]
    valid = code == 0
    D = res[:, :3].reshape(gshape + (3,)).copy(); D[~valid.reshape(gshape)] = np.nan
    Wt = np.where(valid, np.clip(zsc, 0, 50), 0).astype(np.float32).reshape(gshape)
    U, n_out = regularize_field(D, Wt, max(2.5 * L["vox"], 6.0), 1.0)
    tested = np.isin(np.arange(len(code)), nodes)
    counts = {BLOCK_CODES[c]: int(((code == c) & tested).sum()) for c in BLOCK_CODES}
    dm = np.linalg.norm(res[valid, :3], axis=1)
    summ = dict(level=L["name"], tested=int(tested.sum()), counts=counts, outliers_removed=n_out,
                valid_frac=float(valid.sum() / max(tested.sum(), 1)),
                resid_median_um=float(np.median(dm)) if len(dm) else None,
                resid_p90_um=float(np.percentile(dm, 90)) if len(dm) else None)
    log(f"  {L['name']} 결과: 유효 {100*summ['valid_frac']:.1f}% {counts}, 이상치 제거 {n_out}, "
        f"보정량 중앙값 {summ['resid_median_um'] or 0:.1f} µm / p90 {summ['resid_p90_um'] or 0:.1f} µm")
    if summ["valid_frac"] < 0.2:
        log("  [경고] 유효 블록이 20% 미만입니다. 이 단계의 보정은 주변 유효 블록/이전 단계 값으로 채워집니다.")
    warp.levels.append(dict(origin=origin, step=float(step), U=U.astype(np.float32)))
    rel = dict(origin=origin, step=step, code=code.reshape(gshape).astype(np.int8), z=zsc.reshape(gshape))
    return summ, rel


# ----------------------------------------------------------------------------
# 3단계: 점 단위 매칭 (타일별 독립, 병렬). ICP 보정과 최종 매칭에 공통 사용
# ----------------------------------------------------------------------------
def mutual_nn(Q, Y, r, rounds=3, zw=1.0, free1=None, free2=None):
    sc = np.array([1.0, 1.0, zw], np.float32)
    m1 = np.full(len(Q), -1, np.int64); d1 = np.full(len(Q), np.inf, np.float32)
    free1 = np.arange(len(Q)) if free1 is None else free1
    free2 = np.arange(len(Y)) if free2 is None else free2
    for _ in range(rounds):
        if len(free1) == 0 or len(free2) == 0:
            break
        t2 = cKDTree(Y[free2] * sc)
        dd, jj = t2.query(Q[free1] * sc, k=1, distance_upper_bound=r)
        ok = np.isfinite(dd)
        if not ok.any():
            break
        c1 = free1[ok]; c2 = free2[jj[ok]]; cd = dd[ok]
        t1 = cKDTree(Q[free1] * sc)
        _, back = t1.query(Y[c2] * sc, k=1)
        mut = free1[back] == c1
        m1[c1[mut]] = c2[mut]; d1[c1[mut]] = cd[mut]
        used1 = np.zeros(len(Q), bool); used1[c1[mut]] = True
        used2 = np.zeros(len(Y), bool); used2[c2[mut]] = True
        free1 = free1[~used1[free1]]; free2 = free2[~used2[free2]]
    return m1, d1


def local_median_residual(Xp, R, Xq, k, radius, chunk=500_000, exclude_self=False):
    """Xq 각 점에 대해 가까운 k개 매칭쌍(Xp)의 잔차 중앙값 / MAD / 개수. exclude_self: Xq==Xp일 때 자기 자신 제외."""
    med = np.zeros((len(Xq), 3), np.float32); mad = np.full(len(Xq), np.nan, np.float32)
    nsup = np.zeros(len(Xq), np.int16)
    if len(Xp) < 3:
        return med, mad, nsup
    kk = min(k + (1 if exclude_self else 0), len(Xp)); t = cKDTree(Xp)
    for s in range(0, len(Xq), chunk):
        d, j = t.query(Xq[s:s + chunk], k=kk, distance_upper_bound=radius)
        if kk == 1:
            d, j = d[:, None], j[:, None]
        if exclude_self:
            d, j = d[:, 1:], j[:, 1:]
        ok = np.isfinite(d)
        jj = np.where(ok, j, 0)
        Rn = R[jj].astype(np.float32); Rn[~ok] = np.nan
        with np.errstate(all="ignore"):
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m = np.nanmedian(Rn, axis=1)
                dv = np.nanmedian(np.linalg.norm(Rn - m[:, None, :], axis=2), axis=1)
        n = ok.sum(1)
        m[n == 0] = 0
        med[s:s + chunk] = m; mad[s:s + chunk] = dv; nsup[s:s + chunk] = n
    return med, mad, nsup


def consistent_match(X, Q, Y, r, cfg, rounds=3):
    """상호 최근접 + 이웃 일관성 필터: 주변 매칭쌍들과 다르게 움직인 쌍은 제외 (그래프 관계 보존)."""
    m1, d1 = mutual_nn(Q, Y, r, rounds=rounds, zw=cfg["Z_WEIGHT"])
    mk = m1 >= 0
    R = (Y[m1[mk]] - Q[mk]).astype(np.float32)
    med, _, nsup = local_median_residual(X[mk], R, X[mk], cfg["KNN_LOCAL"], cfg["LOCAL_RADIUS_UM"], exclude_self=True)
    dev = np.linalg.norm(R - med, axis=1)
    bad = (nsup < 4) | (dev > cfg["CONSIST_TOL_UM"])
    idx_mk = np.flatnonzero(mk)
    m1[idx_mk[bad]] = -1; d1[idx_mk[bad]] = np.inf
    return m1, d1


def _tile_task(tid):
    g = _G; B1, B2, W, cfg = g["B1"], g["B2"], g["warp"], g["cfg"]
    lo, hi = g["tiles"][tid]
    i1 = B1.query_box(lo, hi)
    if len(i1) == 0:
        return tid, None
    X = B1.P[i1]; Q = W.apply(X)
    mode = g["mode"]; r = g["r"]; mg = r + 2.0
    i2 = B2.query_box(Q.min(0) - mg, Q.max(0) + mg)
    Y = B2.P[i2]
    if len(Y) == 0:
        if mode == "icp":
            return tid, None
        Qc = Q
        return tid, dict(i1=i1, m2=np.full(len(i1), -1, np.int64), d=np.full(len(i1), np.inf, np.float32),
                         pos=Qc, mad=np.full(len(i1), np.nan, np.float32), nsup=np.zeros(len(i1), np.int16))
    m1, d1 = consistent_match(X, Q, Y, r, cfg)
    mk = m1 >= 0; R = (Y[m1[mk]] - Q[mk]).astype(np.float32)

    if mode == "icp":
        gd = g["icp_grid"]
        ijk = np.floor((X[mk] - gd["origin"]) / gd["step"] + 0.5).astype(np.int64)
        ijk = np.clip(ijk, 0, np.array(gd["shape"]) - 1)
        if len(ijk) == 0:
            return tid, None
        o0 = ijk.min(0); sh = tuple((ijk.max(0) - o0 + 1).tolist())
        q = ijk - o0
        flat = (q[:, 0] * sh[1] + q[:, 1]) * sh[2] + q[:, 2]
        n = int(np.prod(sh))
        sums = np.stack([np.bincount(flat, weights=R[:, c], minlength=n) for c in range(3)]
                        + [np.bincount(flat, minlength=n).astype(np.float64)], 1).astype(np.float32)
        return tid, dict(o0=o0, sh=sh, sums=sums, n=int(mk.sum()),
                         res_med=float(np.median(np.linalg.norm(R, axis=1))) if len(R) else np.nan)

    # final: 국소 변형 보정(이웃 매칭쌍 잔차 중앙값) 후, 남은 점끼리 한 번 더 상호 최근접
    corr, mad, nsup = local_median_residual(X[mk], R, X, cfg["KNN_LOCAL"], cfg["LOCAL_RADIUS_UM"])
    Qc = Q + corr
    f1 = np.flatnonzero(~mk)
    used2 = np.zeros(len(Y), bool); used2[m1[mk]] = True
    f2 = np.flatnonzero(~used2)
    m2b, d2b = mutual_nn(Qc, Y, cfg["R_ACCEPT2_UM"], rounds=2, zw=cfg["Z_WEIGHT"], free1=f1, free2=f2)
    add = m2b >= 0
    m1[add] = m2b[add]; d1[add] = d2b[add]
    out_m2 = np.where(m1 >= 0, i2[np.maximum(m1, 0)], -1)
    pos = np.where((m1 >= 0)[:, None], Y[np.maximum(m1, 0)], Qc).astype(np.float32)
    res = dict(i1=i1, m2=out_m2, d=d1, pos=pos, mad=mad, nsup=nsup)
    return tid, res


def make_tiles(B1, tile_xy, tile_z):
    lo = B1.lo; hi = B1.lo + np.array(B1.shape) * B1.size
    nt = np.ceil((hi - lo) / np.array([tile_xy, tile_xy, tile_z])).astype(int)
    tiles = []
    for iz in range(nt[2]):
        for iy in range(nt[1]):
            for ix in range(nt[0]):
                a = lo + np.array([ix * tile_xy, iy * tile_xy, iz * tile_z])
                b = a + np.array([tile_xy, tile_xy, tile_z])
                tiles.append((a, b))
    # 비어있는 타일 제거 (버킷 카운트로 빠르게)
    keep = []
    for t in tiles:
        if len(B1.query_box(t[0], t[1], exact=False)):
            keep.append(t)
    return keep


def run_icp_round(B1, B2, warp, rnd, cfg, tiles):
    lo = B1.P.min(0).astype(np.float64); hi = B1.P.max(0).astype(np.float64)
    step = rnd["grid"]; shape = tuple((np.ceil((hi - lo) / step).astype(int) + 1).tolist())
    gd = dict(origin=lo, step=step, shape=shape)
    _G.update(B1=B1, B2=B2, warp=warp, cfg=cfg, tiles=tiles, mode="icp", r=rnd["r"], icp_grid=gd)
    S = np.zeros(shape + (4,), np.float32); npairs = 0; meds = []
    prog = Progress(len(tiles), f"ICP r={rnd['r']:g}µm ({len(tiles)} 타일)")
    with _pool(cfg["N_WORKERS_TILE"]) as pool:
        for tid, res in pool.imap_unordered(_tile_task, range(len(tiles))):
            prog.update(1)
            if res is None:
                continue
            o, sh = res["o0"], res["sh"]
            S[o[0]:o[0] + sh[0], o[1]:o[1] + sh[1], o[2]:o[2] + sh[2], :] += res["sums"].reshape(sh + (4,))
            npairs += res["n"]; meds.append(res["res_med"])
    cnt = S[..., 3]; D = np.full(shape + (3,), np.nan, np.float32)
    ok = cnt >= 3
    D[ok] = S[ok, :3] / cnt[ok, None]
    Wt = np.where(ok, np.sqrt(cnt), 0).astype(np.float32)
    U, n_out = regularize_field(D, Wt, max(rnd["r"], 3.0), 1.0)
    warp.levels.append(dict(origin=lo, step=float(step), U=U))
    info = dict(r=rnd["r"], grid=step, pairs=int(npairs),
                residual_median_um=float(np.nanmedian(meds)) if meds else None,
                correction_median_um=float(np.median(np.linalg.norm(D[ok], axis=1))) if ok.any() else None)
    log(f"  ICP 결과: 매칭쌍 {npairs:,}, 매칭 잔차 중앙값 {info['residual_median_um'] or 0:.2f} µm, "
        f"격자 보정 중앙값 {info['correction_median_um'] or 0:.2f} µm")
    return info


# ----------------------------------------------------------------------------
# 4단계 (v6): 상대좌표 그래프 매칭
#   각 세포를 절대좌표가 아니라 "이웃 세포들까지의 상대 벡터 집합"으로 표현하고,
#   R1 세포 i 와 R2 후보 a 의 상대 벡터 집합이 얼마나 겹치는지(그래프 일치도)로 짝을 정함.
#   - 이동(translation)에 무관: 상대 벡터만 비교
#   - 회전/기울기/국소 변형: R1 이웃 벡터를 현재 변형장으로 R2 공간에 옮긴 뒤 비교 (Q_j - Q_i)
#   - 누락/가짜 검출에 강함: 집합 비교(일부 이웃이 없어도 나머지가 맞으면 높은 점수)
#   - 점수의 의미는 null(무작위 위치의 후보) 분포와 비교해 FDR로 보정
# ----------------------------------------------------------------------------
def graph_scores(V1, V2, cj, valid, tol2, chunk=3000):
    """tol2: 스칼라 또는 (n,K1) — 이웃 벡터마다 허용 거리². """
    """V1 (n,K1,3): R1 세포 i의 이웃 상대 벡터, V2 (m,K2,3): R2 세포 a의 이웃 상대 벡터,
    cj (n,Kc): 후보 a 인덱스. 반환 S (n,Kc) = i의 이웃 벡터 중 a의 이웃 벡터와 tol 이내로 일치하는 비율."""
    n, Kc = cj.shape
    S = np.full((n, Kc), -1.0, np.float32)
    tol2 = np.broadcast_to(np.asarray(tol2, np.float32), V1.shape[:2])
    for s in range(0, n, chunk):
        v = valid[s:s + chunk]
        cc = np.where(v, cj[s:s + chunk], 0)
        W2 = V2[cc]                                        # (c,Kc,K2,3)
        A1 = V1[s:s + chunk]                               # (c,K1,3)
        d2 = ((A1[:, None, :, None, :] - W2[:, :, None, :, :]) ** 2).sum(-1)   # (c,Kc,K1,K2)
        hit = d2.min(-1) < tol2[s:s + chunk, None, :]
        S[s:s + chunk] = np.where(v, hit.mean(-1), -1.0)
    return S


def _graph_score_task(tid):
    g = _G; B1, B2, W, cfg, gp = g["B1"], g["B2"], g["warp"], g["cfg"], g["gpass"]
    lo, hi = g["tiles"][tid]; H = cfg["GRAPH_HALO_UM"]
    ih = B1.query_box(lo - H, hi + H)
    if len(ih) == 0:
        return tid, None
    Xh = B1.P[ih]
    ev = np.flatnonzero(np.all((Xh >= lo) & (Xh < hi), axis=1))
    if gp["stride"] > 1:
        ev = ev[(ih[ev] % gp["stride"]) == 0]
    if len(ev) == 0:
        return tid, None
    K1, K2, kc, rc = cfg["GRAPH_K1"], cfg["GRAPH_K2"], gp["kc"], gp["r_cand"]
    # 상대 벡터 비교에서 z는 GRAPH_Z_WEIGHT만큼 가중 (R2의 z 위치 오차가 xy보다 크기 때문)
    zs = np.array([1.0, 1.0, cfg["GRAPH_Z_WEIGHT"]], np.float32)
    Qh = W.apply(Xh)
    Qe = Qh[ev]
    out = dict(i=ih[ev], a=np.full(len(ev), -1, np.int64), s1=np.full(len(ev), -1, np.float32),
               s2=np.full(len(ev), -1, np.float32), r=np.zeros((len(ev), 3), np.float32), null=np.empty(0, np.float32))
    if len(Xh) < K1 + 2:
        return tid, out
    _, nb1 = cKDTree(Xh).query(Xh[ev], K1 + 1)
    V1 = ((Qh[nb1[:, 1:]] - Qe[:, None, :]) * zs).astype(np.float32)
    mg = rc + H
    i2 = B2.query_box(Qe.min(0) - mg, Qe.max(0) + mg)
    if len(i2) < K2 + 2:
        return tid, out
    Y = B2.P[i2]; tY = cKDTree(Y)
    _, nb2 = tY.query(Y, K2 + 1)
    V2 = ((Y[nb2[:, 1:]] - Y[:, None, :]) * zs).astype(np.float32)
    kc = min(kc, len(Y))
    dd, cj = tY.query(Qe, k=kc, distance_upper_bound=rc)
    if kc == 1:
        dd, cj = dd[:, None], cj[:, None]
    valid = np.isfinite(dd)
    # 허용 거리 = 고정값 + 벡터 길이 비례분 (국소 변형(strain)으로 먼 이웃일수록 벡터가 더 달라짐)
    tol2 = (cfg["GRAPH_TOL_UM"] + cfg["GRAPH_TOL_FRAC"] * np.linalg.norm(V1, axis=2)) ** 2
    S = graph_scores(V1, V2, cj, valid, tol2)
    # 동점이면 더 가까운 후보 우선
    key = S - 1e-4 * np.where(valid, dd, 1e3) / max(rc, 1e-6)
    o = np.argsort(-key, axis=1)
    b1 = o[:, 0]; rows = np.arange(len(ev))
    s1 = S[rows, b1]
    s2 = S[rows, o[:, 1]] if kc > 1 else np.full(len(ev), -1, np.float32)
    has = valid[rows, b1] & (s1 >= 0)
    out["a"][has] = i2[cj[rows[has], b1[has]]]
    out["s1"] = np.where(has, s1, -1).astype(np.float32); out["s2"] = s2.astype(np.float32)
    out["r"][has] = Y[cj[rows[has], b1[has]]] - Qe[has]
    # null: 같은 세포를 무작위 방향으로 NULL_SHIFT_UM 옮긴 위치의 후보들로 같은 점수 계산
    ns = np.random.default_rng(tid + 7).choice(len(ev), min(len(ev), 4000), replace=False)
    v = np.random.default_rng(tid + 11).normal(size=(len(ns), 3))
    v *= cfg["NULL_SHIFT_UM"] / np.linalg.norm(v, axis=1, keepdims=True)
    ddn, cjn = tY.query(Qe[ns] + v, k=kc, distance_upper_bound=rc)
    if kc == 1:
        ddn, cjn = ddn[:, None], cjn[:, None]
    Sn = graph_scores(V1[ns], V2, cjn, np.isfinite(ddn), tol2[ns])
    out["null"] = Sn.max(1).astype(np.float32)
    return tid, out


def vote_scores(Dc, validc, nbr, tol2, chunk=400):
    """이웃 합의(consensus) 점수. Dc (m,kc,3): 각 R1 세포의 후보 변위(Y_a - Q_j), nbr (n,KG): 평가 세포 i의 R1 이웃.
    점수(i,a) = i의 이웃 중 변위 d_ia 와 tol 이내로 같은 변위의 후보를 가진 세포의 비율.
    (형률님 spectral matching의 '서로 일관된 대응끼리 강화' 원리를 이웃 단위 투표로 계산)"""
    n, KG = nbr.shape; kc = Dc.shape[1]
    S = np.full((n, kc), -1.0, np.float32)
    for s in range(0, n, chunk):
        idx = np.arange(s, min(s + chunk, n))
        own = Dc[idx]                                   # (c,kc,3)  — i 자신: 평가 대상 순서 = Dc 앞부분
        nb = nbr[idx]                                   # (c,KG)
        Dn = Dc[nb]; Vn = validc[nb]                    # (c,KG,kc,3), (c,KG,kc)
        d2 = ((own[:, :, None, None, :] - Dn[:, None, :, :, :]) ** 2).sum(-1)   # (c,kc,KG,kc)
        d2 = np.where(Vn[:, None, :, :], d2, np.inf)
        agree = (d2.min(-1) < tol2).mean(-1)            # (c,kc)
        S[s:s + len(idx)] = np.where(validc[idx], agree, -1.0)
    return S


def _vote_task(tid):
    g = _G; B1, B2, W, cfg, gp = g["B1"], g["B2"], g["warp"], g["cfg"], g["gpass"]
    lo, hi = g["tiles"][tid]; H = cfg["GRAPH_HALO_UM"]
    ih = B1.query_box(lo - H, hi + H)
    if len(ih) == 0:
        return tid, None
    Xh = B1.P[ih]
    ev = np.flatnonzero(np.all((Xh >= lo) & (Xh < hi), axis=1))
    if gp["stride"] > 1:
        ev = ev[(ih[ev] % gp["stride"]) == 0]
    if len(ev) == 0:
        return tid, None
    KG, kc, rc = cfg["VOTE_K"], gp["kc"], gp["r_cand"]
    out = dict(i=ih[ev], a=np.full(len(ev), -1, np.int64), s1=np.full(len(ev), -1, np.float32),
               s2=np.full(len(ev), -1, np.float32), r=np.zeros((len(ev), 3), np.float32), null=np.empty(0, np.float32))
    if len(Xh) < KG + 2:
        return tid, out
    Qh = W.apply(Xh)
    i2 = B2.query_box(Qh.min(0) - rc - 1, Qh.max(0) + rc + 1)
    if len(i2) < 10:
        return tid, out
    Y = B2.P[i2]; tY = cKDTree(Y)
    kc = min(kc, len(Y))
    _, nbr = cKDTree(Xh).query(Xh[ev], KG + 1); nbr = nbr[:, 1:]
    # 평가 세포 + 그 이웃들의 후보 변위만 계산
    need = np.unique(np.r_[ev, nbr.ravel()])
    pos_of = np.full(len(Xh), -1, np.int64)
    order = np.r_[ev, np.setdiff1d(need, ev, assume_unique=False)]
    pos_of[order] = np.arange(len(order))
    zs = np.array([1.0, 1.0, cfg["GRAPH_Z_WEIGHT"]], np.float32)

    def disp(Qs):
        dd, cj = tY.query(Qs, k=kc, distance_upper_bound=rc)
        if kc == 1:
            dd, cj = dd[:, None], cj[:, None]
        v = np.isfinite(dd); cjs = np.where(v, cj, 0)
        return ((Y[cjs] - Qs[:, None, :]) * zs).astype(np.float32), v, cj
    Dc, validc, cj = disp(Qh[order])
    tol2 = cfg["VOTE_TOL_UM"] ** 2
    S = vote_scores(Dc, validc, pos_of[nbr], tol2)
    rows = np.arange(len(ev))
    key = S - 1e-4 * np.linalg.norm(Dc[:len(ev)], axis=2) / max(rc, 1e-6)
    o = np.argsort(-key, axis=1); b1 = o[:, 0]
    s1 = S[rows, b1]; s2 = S[rows, o[:, 1]] if kc > 1 else np.full(len(ev), -1, np.float32)
    has = validc[rows, b1] & (s1 >= 0)
    out["a"][has] = i2[cj[rows[has], b1[has]]]
    out["s1"] = np.where(has, s1, -1).astype(np.float32); out["s2"] = s2.astype(np.float32)
    out["r"][has] = Y[cj[rows[has], b1[has]]] - Qh[ev][has]
    # null: 타일 전체를 무작위 방향으로 NULL_SHIFT_UM 옮긴 위치에서 같은 투표
    v = np.random.default_rng(tid + 11).normal(size=3); v *= cfg["NULL_SHIFT_UM"] / np.linalg.norm(v)
    Dn, vn, _ = disp(Qh[order] + v.astype(np.float32))
    ns = min(len(ev), 4000)
    Sn = vote_scores(Dn, vn, pos_of[nbr[:ns]], tol2)
    out["null"] = Sn.max(1).astype(np.float32)
    return tid, out


def run_graph_score_pass(B1, B2, warp, cfg, tiles, gp, label):
    _G.update(B1=B1, B2=B2, warp=warp, cfg=cfg, tiles=tiles, gpass=gp)
    parts, nulls = [], []
    prog = Progress(len(tiles), f"{label} ({len(tiles)} 타일)")
    task = _vote_task if gp.get("mode") == "vote" else _graph_score_task
    with _pool(cfg["N_WORKERS_TILE"]) as pool:
        for tid, res in pool.imap_unordered(task, range(len(tiles))):
            prog.update(1)
            if res is None:
                continue
            nulls.append(res.pop("null")); parts.append(res)
    cat = {k: np.concatenate([p[k] for p in parts]) for k in ("i", "a", "s1", "s2", "r")}
    null = np.concatenate(nulls) if nulls else np.empty(0, np.float32)
    return cat, null


def choose_threshold(s_obs, s_null, cfg, K1=None):
    """점수 임계값: null(무작위 위치)에서 그 점수 이상이 나올 비율 / 실제에서 나올 비율 ≤ 목표 FDR."""
    K1 = cfg["GRAPH_K1"] if K1 is None else K1
    levels = np.arange(0, K1 + 1) / K1
    so = s_obs[s_obs >= 0]
    rows = []
    min_score = cfg["GRAPH_MIN_SCORE"] if K1 == cfg["GRAPH_K1"] else cfg["VOTE_MIN_SCORE"]
    for t in levels:
        po = float((so >= t - 1e-6).mean()) if len(so) else 0.0
        pn = float((s_null >= t - 1e-6).mean()) if len(s_null) else 0.0
        rows.append((float(t), po, pn, pn / po if po > 0 else np.inf))
    thr = None
    for t, po, pn, f in rows:
        if t >= min_score - 1e-6 and f <= cfg["GRAPH_FDR"]:
            thr = t; break
    if thr is None:
        thr = 1.0
    # 점수 수준별 신뢰도 (1 - local FDR)
    ho = np.array([((so >= t - 1e-6) & (so < t + 1.0 / K1 - 1e-6)).mean() for t in levels])
    hn = np.array([((s_null >= t - 1e-6) & (s_null < t + 1.0 / K1 - 1e-6)).mean() for t in levels])
    conf = np.clip(1 - np.where(ho > 0, hn / np.maximum(ho, 1e-12), 1), 0, 1)
    return thr, rows, conf


def accept_pairs(cat, thr, cfg):
    ok = (cat["a"] >= 0) & (cat["s1"] >= thr - 1e-6) & ((cat["s1"] - cat["s2"]) >= cfg["GRAPH_MARGIN"] - 1e-6)
    idx = np.flatnonzero(ok)
    # 같은 R2 세포를 여러 R1이 고르면 점수가 가장 높은 하나만 (1:1)
    o = idx[np.lexsort((-cat["s1"][idx], cat["a"][idx]))]
    first = np.r_[True, cat["a"][o][1:] != cat["a"][o][:-1]]
    return o[first]


def graph_field_pass(B1, B2, warp, cfg, tiles, gp, k):
    kind = "이웃 합의 투표" if gp.get("mode") == "vote" else "상대벡터 패턴"
    cat, null = run_graph_score_pass(B1, B2, warp, cfg, tiles, gp, f"그래프 보정 {k} [{kind}] (후보반경 {gp['r_cand']:g}µm, 1/{gp['stride']} 표본)")
    vote = gp.get("mode") == "vote"
    thr, rows, _ = choose_threshold(cat["s1"], null, cfg, K1=cfg["VOTE_K"] if vote else None)
    acc = accept_pairs(cat, thr, dict(cfg, GRAPH_MARGIN=cfg["VOTE_MARGIN"]) if vote else cfg)
    X = B1.P[cat["i"][acc]].astype(np.float64); R = cat["r"][acc]
    lo = B1.P.min(0).astype(np.float64); hi = B1.P.max(0).astype(np.float64)
    step = gp["grid"]; shape = tuple((np.ceil((hi - lo) / step).astype(int) + 1).tolist())
    ijk = np.clip(np.floor((X - lo) / step + 0.5).astype(np.int64), 0, np.array(shape) - 1)
    flat = np.ravel_multi_index(ijk.T, shape); n = int(np.prod(shape))
    cnt = np.bincount(flat, minlength=n).reshape(shape).astype(np.float32)
    D = np.full(shape + (3,), np.nan, np.float32)
    ok = cnt >= 3
    for c in range(3):
        # 격자 칸 안의 중앙값 대신 평균 (이미 FDR로 걸렀고, 아래에서 이웃 기반 이상치 제거)
        sm = np.bincount(flat, weights=R[:, c], minlength=n).reshape(shape)
        D[..., c][ok] = (sm[ok] / cnt[ok]).astype(np.float32)
    U, n_out = regularize_field(D, np.where(ok, np.sqrt(cnt), 0).astype(np.float32), max(gp["r_cand"] / 3, 3.0), 1.0)
    warp.levels.append(dict(origin=lo, step=float(step), U=U))
    fdr_at = next((f for t, po, pn, f in rows if abs(t - thr) < 1e-6), None)
    info = dict(r_cand=gp["r_cand"], stride=gp["stride"], grid=step, evaluated=int(len(cat["i"])),
                threshold=thr, est_fdr=fdr_at, accepted=int(len(acc)),
                accepted_frac=float(len(acc) / max(len(cat["i"]), 1)),
                shift_median_um=float(np.median(np.linalg.norm(R, axis=1))) if len(R) else None,
                shift_p90_um=float(np.percentile(np.linalg.norm(R, axis=1), 90)) if len(R) else None)
    log(f"  그래프 보정 {k}: 평가 {info['evaluated']:,}개 중 채택 {info['accepted']:,} ({100*info['accepted_frac']:.1f}%), "
        f"점수 임계값 {thr:.2f} (추정 FDR {100*(fdr_at or 0):.2f}%), 추가 이동 중앙값 {info['shift_median_um'] or 0:.1f} µm / "
        f"p90 {info['shift_p90_um'] or 0:.1f} µm")
    return info


# ----------------------------------------------------------------------------
# 5단계 (v6): 누락 세포 위치 = 주변 관측 세포들의 국소 affine 변형 (+ 불확실성), hold-out 검증
# ----------------------------------------------------------------------------
def local_affine_predict(Xm, Rm, Xq, cfg, exclude_self=False, chunk=200_000):
    """Xm: 관측 세포 R1 위치, Rm: 그 세포들의 잔차(실제 R2 - 변형장 예측), Xq: 추정할 세포.
    가까운 LOCAL_K개 관측 세포로 가중 국소 affine (잔차 = t + G·(x - x_q))을 맞춰 t를 예측값으로 사용.
    반환: 예측 잔차 (n,3), 가중 RMS 잔차(불확실성, µm), 사용한 이웃 수."""
    n = len(Xq); pred = np.zeros((n, 3), np.float32); rms = np.full(n, np.nan, np.float32)
    nsup = np.zeros(n, np.int16)
    if len(Xm) < 3 or n == 0:
        return pred, rms, nsup
    k = min(cfg["LOCAL_K"] + (1 if exclude_self else 0), len(Xm))
    sig = cfg["LOCAL_SIGMA_UM"]; lam = cfg["LOCAL_RIDGE"]
    t = cKDTree(Xm)
    for s in range(0, n, chunk):
        d, j = t.query(Xq[s:s + chunk], k=k, distance_upper_bound=cfg["LOCAL_RADIUS_UM"])
        if k == 1:
            d, j = d[:, None], j[:, None]
        if exclude_self:
            d, j = d[:, 1:], j[:, 1:]
        ok = np.isfinite(d); jj = np.where(ok, j, 0)
        w = np.where(ok, np.exp(-0.5 * (np.where(ok, d, 0) / sig) ** 2), 0.0)
        dx = (Xm[jj] - Xq[s:s + chunk, None, :]) / sig           # (c,k,3)
        A = np.concatenate([np.ones(dx.shape[:2] + (1,)), dx], -1)  # (c,k,4)
        Rn = Rm[jj].astype(np.float64)
        ATA = np.einsum("nk,nki,nkj->nij", w, A, A)
        ATA[:, 1:, 1:] += lam * np.eye(3)
        ATA[:, 0, 0] += 1e-9
        ATb = np.einsum("nk,nki,nkc->nic", w, A, Rn)
        cnt = ok.sum(1)
        sol = np.linalg.solve(ATA, ATb)                            # (c,4,3)
        p = sol[:, 0, :]
        few = cnt < 6                                              # 이웃이 적으면 가중 평균(이동만)
        if few.any():
            ws = w[few].sum(1, keepdims=True)
            p[few] = np.where(ws > 0, (w[few, :, None] * Rn[few]).sum(1) / np.maximum(ws, 1e-12), 0)
        fit = np.einsum("nki,nic->nkc", A, sol)
        res2 = ((Rn - fit) ** 2).sum(-1)
        rr = np.sqrt((w * res2).sum(1) / np.maximum(w.sum(1), 1e-12))
        rr[cnt == 0] = np.nan; p[cnt == 0] = 0
        pred[s:s + chunk] = p; rms[s:s + chunk] = rr; nsup[s:s + chunk] = cnt
    return pred, rms, nsup


def _complete_task(tid):
    g = _G; B1, B2, W, cfg = g["B1"], g["B2"], g["warp"], g["cfg"]
    lo, hi = g["tiles"][tid]; H = cfg["LOCAL_RADIUS_UM"]
    ih = B1.query_box(lo - H, hi + H)
    if len(ih) == 0:
        return tid, None
    Xh = B1.P[ih]
    core = np.flatnonzero(np.all((Xh >= lo) & (Xh < hi), axis=1))
    if len(core) == 0:
        return tid, None
    Qh = W.apply(Xh)
    m = g["m2"][ih].copy()
    mk = m >= 0
    R = np.zeros((len(ih), 3), np.float32)
    R[mk] = B2.P[m[mk]] - Qh[mk]
    # 같은 R1 z-평면 우선: z 차이를 PLANE_Z_SCALE배로 늘린 좌표에서 이웃을 찾음
    #  → 같은 R1 평면에 있던 세포들이 R2에서도 하나의 매끄러운 면을 이루도록 (절대 z가 아니라 상대 관계)
    Xd = Xh.astype(np.float64) * np.array([1.0, 1.0, cfg["PLANE_Z_SCALE"]])
    # 이웃 일관성: 자기 자신을 제외한 주변 관측 세포들의 국소 affine과 CONSIST_TOL 이상 다르면 제외
    mi = np.flatnonzero(mk)
    ps, _, ns = local_affine_predict(Xd[mi], R[mi], Xd[mi], cfg, exclude_self=True)
    dv = R[mi] - ps
    dn = np.linalg.norm(dv, axis=1); dz = np.abs(dv[:, 2])
    # 허용치는 데이터에 맞춰 자동 조정: 이 타일 편차 중앙값의 3배 (설정값 이상)
    has = ns >= 6
    tol = max(cfg["CONSIST_TOL_UM"], 3 * float(np.median(dn[has]))) if has.any() else cfg["CONSIST_TOL_UM"]
    tolz = max(cfg["PLANE_TOL_UM"], 3 * float(np.median(dz[has]))) if has.any() else cfg["PLANE_TOL_UM"]
    bad = has & ((dn > tol) | (dz > tolz))
    m[mi[bad]] = -1; mk = m >= 0
    method = np.where(mk, 1, 0).astype(np.int8)
    fill_d = np.full(len(ih), np.inf, np.float32); alt = np.zeros((len(ih), 3), np.float32)
    # 2차 매칭: 그래프로 짝이 확정된 주변 세포들의 국소 변형으로 위치를 보정한 뒤,
    #          남은 R1 세포와 아직 안 쓰인 R2 검출을 엄격한 상호 최근접(R_ACCEPT2_UM)으로 연결
    if cfg["NN_FILL"]:
        mi = np.flatnonzero(mk); cu0 = core[~mk[core]]
        if len(mi) >= 6 and len(cu0):
            pu, _, nu = local_affine_predict(Xd[mi], R[mi], Xd[cu0], cfg)
            Qc = (Qh[cu0] + pu).astype(np.float32)
            r2 = cfg["R_ACCEPT2_UM"]
            i2 = B2.query_box(Qc.min(0) - r2 - 1, Qc.max(0) + r2 + 1)
            i2 = i2[~g["used2"][i2]]
            if len(i2):
                mm, dd = mutual_nn(Qc, B2.P[i2], r2, rounds=2, zw=cfg["Z_WEIGHT"])
                okf = (mm >= 0) & (nu >= 6)
                sel = cu0[okf]
                m[sel] = i2[mm[okf]]; method[sel] = 2; fill_d[sel] = dd[okf]; alt[sel] = Qc[okf]
                R[sel] = B2.P[m[sel]] - Qh[sel]
                mk = m >= 0
    # hold-out: 코어의 관측 세포 일부를 가리고 나머지로 위치를 맞혀봄 (보완점 정확도 추정)
    cm = core[mk[core]]
    rng = np.random.default_rng(tid + 3)
    ho = cm[rng.random(len(cm)) < cfg["HOLDOUT_FRAC"]]
    train = mk.copy(); train[ho] = False
    ti = np.flatnonzero(train)
    ph, _, _ = local_affine_predict(Xd[ti], R[ti], Xd[ho], cfg)
    ho_err = np.linalg.norm(ph - R[ho], axis=1).astype(np.float32)
    ho_err_z = np.abs(ph[:, 2] - R[ho, 2]).astype(np.float32)
    ho_err_warp = np.linalg.norm(R[ho], axis=1).astype(np.float32)   # 국소 보정 없이 변형장만 썼을 때
    # 보완: 코어의 미관측 세포
    mi = np.flatnonzero(mk)
    cu = core[~mk[core]]
    pu, ru, nu = local_affine_predict(Xd[mi], R[mi], Xd[cu], cfg)
    pos = np.empty((len(core), 3), np.float32); unc = np.zeros(len(core), np.float32); nsup = np.zeros(len(core), np.int16)
    loc = np.empty(len(ih), np.int64); loc[core] = np.arange(len(core))
    co = core[mk[core]]
    pos[loc[co]] = B2.P[m[co]]
    pos[loc[cu]] = Qh[cu] + pu; unc[loc[cu]] = ru; nsup[loc[cu]] = nu
    return tid, dict(i=ih[core], m2=m[core], pos=pos, unc=unc, nsup=nsup, method=method[core],
                     fill_d=fill_d[core], alt=alt[core],
                     n_rejected=int(bad.sum()), ho_err=ho_err, ho_err_z=ho_err_z, ho_err_warp=ho_err_warp, ho_x=Xh[ho])


def run_completion(B1, B2, warp, cfg, tiles, m2_sorted):
    N1 = len(B1.P)
    used2 = np.zeros(len(B2.P), bool); used2[m2_sorted[m2_sorted >= 0]] = True
    _G.update(B1=B1, B2=B2, warp=warp, cfg=cfg, tiles=tiles, m2=m2_sorted, used2=used2)
    m2 = np.full(N1, -1, np.int64); pos = np.zeros((N1, 3), np.float32)
    method = np.zeros(N1, np.int8); fill_d = np.full(N1, np.inf, np.float32); alt = np.zeros((N1, 3), np.float32)
    unc = np.zeros(N1, np.float32); nsup = np.zeros(N1, np.int16)
    he, hw, hx, hz = [], [], [], []; nrej = 0
    prog = Progress(len(tiles), f"보완 + hold-out 검증 ({len(tiles)} 타일)")
    with _pool(cfg["N_WORKERS_TILE"]) as pool:
        for tid, res in pool.imap_unordered(_complete_task, range(len(tiles))):
            prog.update(1)
            if res is None:
                continue
            i = res["i"]
            m2[i] = res["m2"]; pos[i] = res["pos"]; unc[i] = res["unc"]; nsup[i] = res["nsup"]
            method[i] = res["method"]; fill_d[i] = res["fill_d"]; alt[i] = res["alt"]
            nrej += res["n_rejected"]; he.append(res["ho_err"]); hw.append(res["ho_err_warp"]); hx.append(res["ho_x"]); hz.append(res["ho_err_z"])
    # 타일 경계에서 2차 매칭이 같은 R2 검출을 중복으로 고른 경우: 더 가까운 쪽만 유지
    f = np.flatnonzero(method == 2)
    o = f[np.lexsort((fill_d[f], m2[f]))]
    dup = o[np.r_[False, m2[o][1:] == m2[o][:-1]]]
    m2[dup] = -1; method[dup] = 0; pos[dup] = alt[dup]
    cat = lambda L, d: np.concatenate(L) if L else np.empty((0,) + d, np.float32)
    return dict(m2=m2, pos=pos, unc=unc, nsup=nsup, n_rejected=nrej, method=method, n_fill_conflicts=int(len(dup)),
                ho_err=cat(he, ()), ho_err_z=cat(hz, ()), ho_err_warp=cat(hw, ()), ho_x=cat(hx, (3,)))


def save_holdout_map(ho_x, ho_err, path, bin_um=250.0, thr=5.0):
    import matplotlib.pyplot as plt
    if len(ho_x) == 0:
        return
    ij = np.floor((ho_x - ho_x.min(0)) / bin_um).astype(int); sh = ij.max(0) + 1
    fig, ax = plt.subplots(1, 3, figsize=(20, 6))
    for k, (a0, a1, nm) in enumerate(((0, 1, "XY"), (0, 2, "XZ"))):
        tot = np.zeros((sh[a0], sh[a1])); err = np.zeros_like(tot)
        np.add.at(tot, (ij[:, a0], ij[:, a1]), 1); np.add.at(err, (ij[:, a0], ij[:, a1]), ho_err)
        im = ax[k].imshow(np.where(tot >= 20, err / np.maximum(tot, 1), np.nan).T, origin="lower",
                          vmin=0, vmax=2 * thr, cmap="RdYlGn_r", aspect="auto")
        plt.colorbar(im, ax=ax[k]); ax[k].set_title(f"hold-out mean error (um), {nm}, {bin_um:g} um bins")
    ax[2].hist(np.clip(ho_err, 0, 20), np.linspace(0, 20, 81), color="#555")
    ax[2].axvline(thr, ls="--", c="r"); ax[2].set_xlabel("hold-out error (um)")
    ax[2].set_title(f"median {np.median(ho_err):.2f} um, <{thr:g} um: {100*(ho_err < thr).mean():.1f}%")
    fig.tight_layout(); fig.savefig(path, dpi=100); plt.close(fig)


# ----------------------------------------------------------------------------
# QC
# ----------------------------------------------------------------------------
def edge_preservation(B1, matched, pos, k=8, n_blocks=200, block_um=300.0, min_pairs=40, seed=0):
    """형률님 somaprint_consistency와 같은 원리: 블록 안 매칭 세포들의 kNN 엣지가 R2에서도 유지되는 비율 vs 셔플 null.
    B1: R1 공간 인덱스, matched/pos: B1 정렬 순서 기준 매칭 여부 / R2 위치(µm)."""
    rng = np.random.default_rng(seed)
    cand = np.flatnonzero(matched)
    if len(cand) < min_pairs:
        return dict(n_blocks=0)
    obs, nul = [], []
    h = block_um / 2
    for c in B1.P[rng.choice(cand, min(n_blocks * 3, len(cand)), replace=False)]:
        idx = B1.query_box(c - h, c + h)
        idx = idx[matched[idx]]
        if len(idx) < min_pairs:
            continue
        if len(idx) > 3000:
            idx = rng.choice(idx, 3000, replace=False)
        a, b = B1.P[idx], pos[idx]
        kk = min(k + 1, len(a))
        _, ja = cKDTree(a).query(a, kk); _, jb = cKDTree(b).query(b, kk)
        sa = [set(r[1:]) for r in ja]; sb = [set(r[1:]) for r in jb]
        obs.append(np.mean([len(x & y) / max(len(x), 1) for x, y in zip(sa, sb)]))
        perm = rng.permutation(len(b)); _, jp = cKDTree(b[perm]).query(b[perm], kk)
        sp = [set(perm[r[1:]]) for r in jp]
        nul.append(np.mean([len(x & y) / max(len(x), 1) for x, y in zip(sa, sp)]))
        if len(obs) >= n_blocks:
            break
    return dict(n_blocks=len(obs), preserved_p50=float(np.median(obs)) if obs else None,
                null_p50=float(np.median(nul)) if nul else None)


# ----------------------------------------------------------------------------
# 전체 실행
# ----------------------------------------------------------------------------
def _h(obj):
    return hashlib.sha1(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:10]


def _sig(cfg, keys):
    d = {k: cfg[k] for k in keys}
    for k in ("R1_JSON", "R2_JSON"):
        st = os.stat(cfg[k]); d[k + "_stat"] = [st.st_size, int(st.st_mtime)]
    return _h(d)


def run(cfg):
    global _T0
    _T0 = time.time()
    CFG.update(cfg)
    in_keys = ["R1_JSON", "R2_JSON", "SPACING_R1_XYZ_UM", "SPACING_R2_XYZ_UM", "COORD_ORDER", "R1_Z_RANGE",
               "QUICK_CTX_SLICES", "R2_EXTRA_SLICES", "BUCKET_UM"]
    in_keys = in_keys + ["BRAIN_FRAME"]
    reg_keys = in_keys + (["INIT_WARP"] if cfg.get("INIT_WARP") else
                          ["GLOBAL_BINS_UM", "LEVELS", "ALLOW_REFLECTION", "INITIAL_AFFINE"])
    run_dir = Path(cfg["OUT_ROOT"]) / f"run_{_sig(cfg, reg_keys)}"
    run_dir.mkdir(parents=True, exist_ok=True)
    json.dump({k: v for k, v in cfg.items()}, open(run_dir / "config.json", "w"), indent=1, default=str)
    log(f"SYTO16 R1→R2 canonical matching {VERSION}")
    log(f"결과 폴더: {run_dir}")

    # ---------------- 0. 입력 ----------------
    stage("0/6 입력 읽기")
    m1 = tiff_meta(cfg["R1_TIFF_DIR"], cfg["NATURAL_SORT"]); m2 = tiff_meta(cfg["R2_TIFF_DIR"], cfg["NATURAL_SORT"])
    for nm, m in (("R1", m1), ("R2", m2)):
        log(f"  {nm} TIFF: " + (f"{m['nz']}장, {m['W']}x{m['H']} (W×H)" if m else "찾지 못함"))
    F1 = load_points_json(cfg["R1_JSON"]); F2 = load_points_json(cfg["R2_JSON"])
    c1 = coord_columns(F1, m1, cfg["COORD_ORDER"]); c2 = coord_columns(F2, m2, cfg["COORD_ORDER"])
    V1 = F1.raw[:, list(c1)]; V2 = F2.raw[:, list(c2)]
    log(f"  R1 {len(V1):,}점, R2 {len(V2):,}점; 좌표 열(x,y,z)=R1{c1} R2{c2}")
    sp1 = np.asarray(cfg["SPACING_R1_XYZ_UM"], np.float64); sp2 = np.asarray(cfg["SPACING_R2_XYZ_UM"], np.float64)
    sel1 = np.arange(len(V1)); core_mask = None
    if cfg["R1_Z_RANGE"] is not None:
        z0, z1 = cfg["R1_Z_RANGE"]; ctx = cfg["QUICK_CTX_SLICES"]
        sel1 = np.flatnonzero((V1[:, 2] >= z0 - ctx) & (V1[:, 2] < z1 + ctx))
        core_mask = (V1[sel1, 2] >= z0) & (V1[sel1, 2] < z1)
        zz = V1[sel1, 2]; e = cfg["R2_EXTRA_SLICES"]
        sel2 = np.flatnonzero((V2[:, 2] >= zz.min() - e) & (V2[:, 2] <= zz.max() + e))
        log(f"  빠른 테스트: 출력 R1 z[{z0},{z1}) {core_mask.sum():,}점, 정합 문맥 {len(sel1):,}점, R2 후보 {len(sel2):,}점")
    else:
        sel2 = np.arange(len(V2))
    if cfg["BRAIN_FRAME"]:
        # 각 라운드를 그 뇌 자신의 좌표계로 변환 (전체 점으로 중심·주축 계산; 빠른 테스트여도 전체 기준)
        f1 = BrainFrame.fit(V1 * sp1); f2 = BrainFrame.fit(V2 * sp2)
        rel = f1.E.T @ f2.E
        rel_deg = float(np.rad2deg(np.arccos(np.clip((np.trace(rel) - 1) / 2, -1, 1))))
        log(f"  뇌 좌표계 변환: R1 중심 {np.round(f1.c, 0).tolist()} µm, 주축 z 기울기 {f1.tilt_deg():.2f}° | "
            f"R2 중심 {np.round(f2.c, 0).tolist()} µm, 주축 z 기울기 {f2.tilt_deg():.2f}°")
        log(f"    두 라운드 뇌 축 사이 회전 차이 {rel_deg:.2f}°, 중심 차이 {np.round(f2.c - f1.c, 1).tolist()} µm "
            f"(이후 정렬은 x′,y′,z′ 좌표에서 이 차이를 더 정밀하게 맞춤)")
        np.savez(run_dir / "brain_frames.npz", r1_center_um=f1.c, r1_axes=f1.E, r2_center_um=f2.c, r2_axes=f2.E)
        P1 = f1.to(V1[sel1] * sp1); P2 = f2.to(V2[sel2] * sp2)
    else:
        f1 = f2 = None
        P1 = (V1[sel1] * sp1).astype(np.float32); P2 = (V2[sel2] * sp2).astype(np.float32)
    log("  공간 인덱스 생성 중...")
    B1 = Buckets(P1, cfg["BUCKET_UM"]); B2 = Buckets(P2, cfg["BUCKET_UM"])
    del P1, P2; gc.collect()
    report = dict(version=VERSION, run_dir=str(run_dir))
    tiles = make_tiles(B1, cfg["TILE_XY_UM"], cfg["TILE_Z_UM"])
    log(f"  타일 {len(tiles)}개 (XY {cfg['TILE_XY_UM']:g} µm × Z {cfg['TILE_Z_UM']:g} µm ≈ {cfg['TILE_Z_UM']/sp1[2]:.0f}장)")

    if cfg.get("INIT_WARP"):
        # ---- 이전 실행(v5 등)의 변형장을 출발점으로 사용: 1~3단계 생략 ----
        stage("1-3/6 이전 변형장 재사용 (INIT_WARP)")
        warp = Warp.from_raw(cfg["INIT_WARP"], f1, f2) if f1 is not None else Warp.load(cfg["INIT_WARP"])
        log(f"  {cfg['INIT_WARP']} (원래 좌표계 변형장을 뇌 좌표계로 감싸서 사용)" if f1 is not None else f"  {cfg['INIT_WARP']}")
        report["init_warp"] = cfg["INIT_WARP"]
    else:
        # ---------------- 1. 전역 affine ----------------
        stage("1/6 전역 3D affine 정합 (뇌 전체 밀도, 회전/기울기/배율/이동)")
        wpath = run_dir / "warp_global.npz"
        if cfg["RESUME"] and wpath.exists():
            warp = Warp.load(wpath); report["global"] = json.load(open(run_dir / "global.json"))
            log("  이전 결과 재사용")
        else:
            if cfg["INITIAL_AFFINE"] is not None:
                Hm = np.asarray(cfg["INITIAL_AFFINE"], float)
                warp = Warp(Hm[:3, :3], Hm[:3, 3]); ginfo = dict(candidate="user_initial_affine")
                log("  사용자 지정 초기 affine 사용")
            else:
                warp, ginfo, G = global_affine(B1.P, B2.P, cfg)
                try:
                    save_global_overlay(G, warp, run_dir / "global_overlay.png")
                except Exception as ex:
                    log(f"  (overlay 그림 생략: {ex})")
            warp.save(wpath); json.dump(ginfo, open(run_dir / "global.json", "w"), indent=1)
            report["global"] = ginfo
        # ---------------- 2. 블록 ----------------
        stage("2/6 3D 블록 비선형 보정 (블록마다 독립, 실패 블록은 이웃으로 채움)")
        report["levels"] = []
        for i, L in enumerate(cfg["LEVELS"]):
            p = run_dir / f"warp_L{i+1}.npz"
            if cfg["RESUME"] and p.exists():
                warp = Warp.load(p); report["levels"].append(json.load(open(run_dir / f"level_L{i+1}.json")))
                log(f"  {L['name']} 이전 결과 재사용"); continue
            log(f"  {L['name']}: 블록 {L['block']:g} µm, 간격 {L['step']:g} µm, 탐색 ±{L['search']:g} µm")
            summ, rel = run_block_level(B1, B2, warp, L, cfg)
            warp.save(p); json.dump(summ, open(run_dir / f"level_L{i+1}.json", "w"), indent=1)
            np.savez_compressed(run_dir / f"reliability_L{i+1}.npz", **rel)
            report["levels"].append(summ)
        # ---------------- 3. ICP ----------------
        stage("3/6 점 기반 미세 보정 (상호 최근접 + 이웃 일관성)")
        report["icp"] = []
        hk = _h(cfg["ICP_ROUNDS"])
        for i, rnd in enumerate(cfg["ICP_ROUNDS"]):
            p = run_dir / f"warp_ICP{i+1}_{hk}.npz"
            if cfg["RESUME"] and p.exists():
                warp = Warp.load(p); report["icp"].append(json.load(open(run_dir / f"icp_{i+1}_{hk}.json")))
                log(f"  ICP {i+1} 이전 결과 재사용"); continue
            info = run_icp_round(B1, B2, warp, rnd, cfg, tiles)
            warp.save(p); json.dump(info, open(run_dir / f"icp_{i+1}_{hk}.json", "w"), indent=1)
            report["icp"].append(info)

    # ---------------- 4. 상대좌표 그래프 매칭으로 변형장 보정 ----------------
    stage("4/6 상대좌표 그래프 매칭 (이웃 상대벡터 패턴 비교) → 변형장 보정")
    gkeys = {k: cfg[k] for k in ("GRAPH_K1", "GRAPH_K2", "GRAPH_TOL_UM", "GRAPH_TOL_FRAC", "GRAPH_Z_WEIGHT",
                                 "VOTE_K", "VOTE_TOL_UM", "VOTE_MIN_SCORE", "VOTE_MARGIN", "GRAPH_HALO_UM", "GRAPH_FDR",
                                 "GRAPH_MIN_SCORE", "GRAPH_MARGIN", "Z_WEIGHT", "NULL_SHIFT_UM", "TILE_XY_UM", "TILE_Z_UM")}
    report["graph_passes"] = []
    for k, gp in enumerate(cfg["GRAPH_PASSES"], 1):
        hk = _h([gkeys, cfg["GRAPH_PASSES"][:k], cfg.get("ICP_ROUNDS"), cfg.get("INIT_WARP")])
        p = run_dir / f"warp_G{k}_{hk}.npz"
        if cfg["RESUME"] and p.exists():
            warp = Warp.load(p); report["graph_passes"].append(json.load(open(run_dir / f"graph_{k}_{hk}.json")))
            log(f"  그래프 보정 {k} 이전 결과 재사용"); continue
        info = graph_field_pass(B1, B2, warp, cfg, tiles, gp, k)
        warp.save(p); json.dump(info, open(run_dir / f"graph_{k}_{hk}.json", "w"), indent=1)
        report["graph_passes"].append(info)
    warp.save(run_dir / "warp_final.npz")

    # ---------------- 5. 전체 세포 그래프 매칭 + 보완 ----------------
    stage("5/6 전체 세포 상대좌표 그래프 매칭 (1:1) + 누락 세포 국소 affine 보완 + hold-out 검증")
    gf = cfg["GRAPH_FINAL"]
    cat, null = run_graph_score_pass(B1, B2, warp, cfg, tiles, dict(gf, stride=1), "전체 세포 그래프 점수")
    thr, rows_fdr, conf_lv = choose_threshold(cat["s1"], null, cfg)
    acc = accept_pairs(cat, thr, cfg)
    N1 = len(B1.P)
    m2s = np.full(N1, -1, np.int64); m2s[cat["i"][acc]] = cat["a"][acc]
    score_s = np.full(N1, -1, np.float32); score_s[cat["i"]] = cat["s1"]
    fdr_at = next((f for t, po, pn, f in rows_fdr if abs(t - thr) < 1e-6), None)
    log(f"  그래프 점수 임계값 {thr:.2f} (추정 FDR {100*(fdr_at or 0):.2f}%), 채택 {len(acc):,} / {N1:,}")
    C = run_completion(B1, B2, warp, cfg, tiles, m2s)
    del cat; gc.collect()
    he, hw = C["ho_err"], C["ho_err_warp"]
    if len(he):
        log(f"  hold-out 검증 ({len(he):,}개 관측 세포를 가리고 주변으로 맞힘): 오차 중앙값 {np.median(he):.2f} µm, "
            f"p90 {np.percentile(he, 90):.2f} µm, {cfg['HOLDOUT_OK_UM']:g} µm 이내 {100*(he < cfg['HOLDOUT_OK_UM']).mean():.1f}% "
            f"(z 성분 중앙값 {np.median(C['ho_err_z']):.2f} µm; 국소 보정 없이 변형장만: 중앙값 {np.median(hw):.2f} µm)")
        save_holdout_map(C["ho_x"], he, run_dir / "holdout_error_map.png", thr=cfg["HOLDOUT_OK_UM"])   # 축: 뇌 좌표계

    # 정렬 순서 → 원래 순서
    inv1 = B1.order
    arr = lambda a: _unsort(a, inv1)
    matched = C["m2"] >= 0
    m2_sel = np.full(N1, -1, np.int64); m2_sel[matched] = sel2[B2.order[C["m2"][matched]]]
    m2_o = arr(m2_sel); pos_o = arr(C["pos"]); unc_o = arr(C["unc"]); nsup_o = arr(C["nsup"]); score_o = arr(score_s)
    # 1 관측 / 2 보완(주변 관측 세포 LOCAL_MIN_SUPPORT개 이상으로 국소 변형 추정) / 3 보완(주변 근거 부족: 전역 변형장만)
    status = np.where(m2_o >= 0, 1, np.where(nsup_o >= cfg["LOCAL_MIN_SUPPORT"], 2, 3)).astype(np.uint8)
    method_o = arr(C["method"])
    lv = np.clip(np.round(score_o * cfg["GRAPH_K1"]).astype(int), 0, cfg["GRAPH_K1"])
    conf_o = np.where(method_o == 1, conf_lv[lv], np.nan).astype(np.float32)
    out_idx = np.arange(N1) if core_mask is None else np.flatnonzero(core_mask)
    r1_rows = sel1[out_idx]
    n_obs = int((status[out_idx] == 1).sum()); n_imp = int((status[out_idx] == 2).sum()); n_weak = int((status[out_idx] == 3).sum())
    used2 = np.zeros(len(V2), bool); used2[m2_o[out_idx][m2_o[out_idx] >= 0]] = True
    unused2 = np.flatnonzero(~used2) if core_mask is None else np.array([], np.int64)
    nn_ = max(len(out_idx), 1)
    log(f"  R1 {len(out_idx):,}개 → 관측 {n_obs:,} ({100*n_obs/nn_:.1f}%), 보완(주변 근거 충분) {n_imp:,} ({100*n_imp/nn_:.1f}%), "
        f"보완(주변 근거 부족, 신뢰 낮음) {n_weak:,} ({100*n_weak/nn_:.1f}%); 이웃 불일치로 제외 {C['n_rejected']:,}")
    log(f"    관측 내역: 그래프 매칭 {int((method_o[out_idx]==1).sum()):,}, 국소 보정 후 2차 매칭 {int((method_o[out_idx]==2).sum()):,}")

    # ---------------- 6. 저장 ----------------
    stage("6/6 저장")
    rows = np.empty((len(out_idx), F2.ncol), np.float64)
    ob = status[out_idx] == 1
    rows[ob] = F2.raw[m2_o[out_idx][ob]]
    im = ~ob
    rows[im] = F1.raw[r1_rows[im]] if F1.ncol == F2.ncol else 0
    pos_imp = pos_o[out_idx][im].astype(np.float64)
    if f2 is not None:
        pos_imp = f2.back(pos_imp)          # 뇌 좌표계 → R2 원래 좌표(µm)
    vox_imp = pos_imp / sp2
    for a in range(3):
        rows[im, c2[a]] = vox_imp[:, a]
    final_json = run_dir / "R2_canonical_matched_to_R1.json"
    write_like(F2, rows, final_json)
    log(f"  최종 JSON (행 i = R1 행 {'i' if core_mask is None else 'r1_row[i]'}): {final_json}")
    if core_mask is not None:
        write_like(F1, F1.raw[r1_rows], run_dir / "R1_core_subset.json")
    if len(unused2):
        write_like(F2, F2.raw[unused2], run_dir / "R2_unmatched_detections.json")
    r2_vox = rows[:, list(c2)]
    np.savez_compressed(run_dir / "canonical_nodes.npz",
                        r1_row=r1_rows, r1_xyz_vox=V1[r1_rows], r2_xyz_vox=r2_vox,
                        status=status[out_idx], r2_row=m2_o[out_idx],
                        match_method=method_o[out_idx], graph_score=score_o[out_idx], match_confidence=conf_o[out_idx],
                        imputed_uncertainty_um=unc_o[out_idx], n_support=nsup_o[out_idx],
                        r2_unmatched_rows=unused2,
                        status_legend=np.array(["", "1=observed (matched to an R2 detection)",
                                                "2=imputed, local support (>=LOCAL_MIN_SUPPORT matched neighbours)",
                                                "3=imputed, no local support (global deformation only; low reliability)",
                                                "match_method: 1=relative-graph, 2=mutual-NN after local correction"]))
    ep = edge_preservation(B1, C["m2"] >= 0, C["pos"])
    report.update(n_r1=int(len(out_idx)), n_observed=n_obs, n_imputed_supported=n_imp, n_imputed_unsupported=n_weak,
                  n_r2_input=int(len(V2)),
                  n_r2_unmatched=int(len(unused2)) if core_mask is None else None,
                  graph_threshold=thr, graph_est_fdr=fdr_at,
                  graph_fdr_table=[dict(score=t, obs_frac=po, null_frac=pn, fdr=f) for t, po, pn, f in rows_fdr],
                  n_rejected_inconsistent=C["n_rejected"],
                  n_graph_matched=int((method_o[out_idx] == 1).sum()), n_nn_filled=int((method_o[out_idx] == 2).sum()),
                  holdout=dict(n=int(len(he)), median_um=float(np.median(he)) if len(he) else None,
                               p90_um=float(np.percentile(he, 90)) if len(he) else None,
                               frac_within_ok=float((he < cfg["HOLDOUT_OK_UM"]).mean()) if len(he) else None,
                               z_median_um=float(np.median(C["ho_err_z"])) if len(he) else None,
                               warp_only_median_um=float(np.median(hw)) if len(hw) else None),
                  imputed_uncertainty_um=dict(p50=float(np.nanmedian(unc_o[out_idx][status[out_idx] == 2])) if n_imp else None),
                  holdout_note="hold-out은 관측 세포 주변에서만 측정되므로 status=2의 정확도를 나타냄. status=3에는 해당 없음.",
                  edge_preservation=ep,
                  final_json=str(final_json), nodes_npz=str(run_dir / "canonical_nodes.npz"),
                  coord_cols_r1=list(c1), coord_cols_r2=list(c2), quick_test_range=cfg["R1_Z_RANGE"])
    json.dump(report, open(run_dir / "report.json", "w"), indent=1, default=float)
    assert len(rows) == len(out_idx)
    log(f"완료. R1 {len(out_idx):,}행 = 최종 R2 {len(rows):,}행. 보고서: {run_dir/'report.json'}")
    return dict(report=report, run_dir=run_dir, r1_xyz_vox=V1[r1_rows], r2_xyz_vox=r2_vox,
                status=status[out_idx], r2_unmatched_vox=V2[unused2] if len(unused2) else np.empty((0, 3)),
                confidence=conf_o[out_idx], uncertainty=unc_o[out_idx], warp=RawWarp(warp, f1, f2),
                warp_brain_frame=warp, frames=(f1, f2))


class RawWarp:
    """원래 좌표계용 인터페이스: R1 µm(원래 좌표) → R2 µm(원래 좌표). 진단/다른 채널 점 변환에 사용."""
    def __init__(self, warp, f1, f2):
        self.warp, self.f1, self.f2 = warp, f1, f2

    def apply(self, X):
        if self.f1 is None:
            return self.warp.apply(X)
        return self.f2.back(self.warp.apply(self.f1.to(X))).astype(np.float32)


def _unsort(a_sorted, order):
    out = np.empty_like(a_sorted); out[order] = a_sorted; return out


# ----------------------------------------------------------------------------
# 시각화
# ----------------------------------------------------------------------------
def save_global_overlay(G, warp, path):
    import matplotlib; import matplotlib.pyplot as plt
    D1 = G["D1"]; sh = D1.shape
    idx_all = np.arange(int(np.prod(sh))).reshape(sh)
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    for i, (nm, sl) in enumerate((("XY (mid z)", np.s_[:, :, sh[2] // 2]), ("XZ (mid y)", np.s_[:, sh[1] // 2, :]),
                                  ("YZ (mid x)", np.s_[sh[0] // 2, :, :]))):
        ids = idx_all[sl]
        Xg = _centers(G["lo1"], G["bin"], sh, ids.ravel())
        b = _sample_vals(G["D2"], G["lo2"], G["bin"], warp.apply(Xg).astype(np.float64)).reshape(ids.shape)
        a = D1[sl]
        nz = lambda v: np.clip(v / max(np.percentile(v, 99.5), 1e-9), 0, 1)
        rgb = np.stack([nz(a), nz(b), np.zeros_like(a)], -1)
        ax[i].imshow(np.transpose(rgb, (1, 0, 2)), origin="lower"); ax[i].set_title(nm + "  red=R1, green=registered R2")
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def _contrast(img):
    lo, hi = np.percentile(img[:: 4, :: 4], (1, 99.7))
    return np.clip((img.astype(np.float32) - lo) / max(hi - lo, 1e-9), 0, 1)


def evidence_score(img, xy, r_in=2, r0=5, r1=8):
    """영상 근거 지수: 점 위치 주변 밝기 - 고리 배경, 배경 MAD로 정규화 (보완점의 실제 신호 유무 확인용)."""
    H, W = img.shape; out = np.full(len(xy), np.nan)
    yy, xx = np.mgrid[-r1:r1 + 1, -r1:r1 + 1]; rr = np.hypot(yy, xx)
    ring = (rr >= r0) & (rr <= r1); core = rr <= r_in
    for k, (x, y) in enumerate(np.round(xy).astype(int)):
        if r1 <= x < W - r1 and r1 <= y < H - r1:
            p = img[y - r1:y + r1 + 1, x - r1:x + r1 + 1].astype(np.float32)
            bg = p[ring]; m = np.median(bg); s = 1.4826 * np.median(np.abs(bg - m)) + 1e-6
            out[k] = (p[core].mean() - m) / s
    return out


def patch_ncc(I1, xy1, I2, xy2, r=6):
    """영상 근거 지수 (형률님 evidence index): R1 세포 위치의 patch와 R2 예상/매칭 위치의 patch 사이 NCC.
    이미지를 바꾸거나 만들지 않고, 두 원본 영상의 해당 위치를 비교만 함."""
    out = np.full(len(xy1), np.nan)
    for k, ((x1, y1), (x2, y2)) in enumerate(zip(np.round(xy1).astype(int), np.round(xy2).astype(int))):
        if (r <= x1 < I1.shape[1] - r and r <= y1 < I1.shape[0] - r and
                r <= x2 < I2.shape[1] - r and r <= y2 < I2.shape[0] - r):
            a = I1[y1 - r:y1 + r + 1, x1 - r:x1 + r + 1].astype(np.float32).ravel()
            b = I2[y2 - r:y2 + r + 1, x2 - r:x2 + r + 1].astype(np.float32).ravel()
            a -= a.mean(); b -= b.mean()
            den = np.linalg.norm(a) * np.linalg.norm(b)
            if den > 0:
                out[k] = float(a @ b / den)
    return out


def visualize_slice(res, cfg, z_r1=None, dz=None, crops=3, crop_px=160, seed=0, save=True):
    """R1 z 단면 + 대응 R2 단면 위에 최종 점 표시, 확대 영역에서 같은 ID를 같은 번호로 표시."""
    import matplotlib.pyplot as plt
    z_r1 = cfg["VIS_Z_R1"] if z_r1 is None else z_r1; dz = cfg["VIS_DZ"] if dz is None else dz
    f1 = tiff_list(cfg["R1_TIFF_DIR"], cfg["NATURAL_SORT"]); f2 = tiff_list(cfg["R2_TIFF_DIR"], cfg["NATURAL_SORT"])
    A = res["r1_xyz_vox"]; Bv = res["r2_xyz_vox"]; st = res["status"]
    s1 = np.abs(A[:, 2] - z_r1) <= dz
    if not s1.any():
        print("해당 z에 R1 점이 없습니다."); return
    z_r2 = int(np.clip(np.round(np.median(Bv[s1, 2])), 0, len(f2) - 1))
    s2 = np.abs(Bv[:, 2] - z_r2) <= dz
    U = res["r2_unmatched_vox"]; su = np.abs(U[:, 2] - z_r2) <= dz if len(U) else np.zeros(0, bool)
    I1 = read_slice(f1, z_r1); I2 = read_slice(f2, z_r2)
    zr = Bv[s1, 2]
    print(f"R1 z={z_r1}의 세포들은 R2에서 z 중앙값 {z_r2} (5~95% 범위 {np.percentile(zr,5):.0f}~{np.percentile(zr,95):.0f}; "
          f"범위가 넓으면 라운드 간 기울어짐 때문입니다).")
    print(f"R1 z={z_r1}±{dz}: {s1.sum():,}점 | R2 z={z_r2}±{dz}: {s2.sum():,}점 "
          f"(관측 {(s2&(st==1)).sum():,}, 보완 {(s2&(st==2)).sum():,}, 근거부족 보완 {(s2&(st==3)).sum():,}, 미사용 R2 검출 {int(su.sum()):,})")

    ds = max(1, int(np.ceil(max(I1.shape) / 2200)))
    fig, ax = plt.subplots(1, 2, figsize=(22, 11))
    ax[0].imshow(_contrast(I1)[::ds, ::ds], cmap="gray")
    ax[0].scatter(A[s1, 0] / ds, A[s1, 1] / ds, s=1, c="#00e5ff", lw=0)
    ax[0].set_title(f"R1 z={z_r1}  (R1 points within +-{dz} slice)")
    ax[1].imshow(_contrast(I2)[::ds, ::ds], cmap="gray")
    ob = s2 & (st == 1); im = s2 & (st == 2); wk = s2 & (st == 3)
    ax[1].scatter(Bv[ob, 0] / ds, Bv[ob, 1] / ds, s=1, c="#00ff66", lw=0, label="1 observed (matched R2 detection)")
    ax[1].scatter(Bv[im, 0] / ds, Bv[im, 1] / ds, s=1, c="#ff33cc", lw=0, label="2 imputed, local support")
    if wk.any():
        ax[1].scatter(Bv[wk, 0] / ds, Bv[wk, 1] / ds, s=1, c="#4da6ff", lw=0, label="3 imputed, no local support (low reliability)")
    if su.any():
        ax[1].scatter(U[su, 0] / ds, U[su, 1] / ds, s=1, c="#ffaa00", lw=0, label="unused R2 detection (pruned)")
    ax[1].legend(markerscale=10, loc="upper right"); ax[1].set_title(f"R2 z={z_r2}  (final points within +-{dz} slice)")
    fig.tight_layout()
    if save:
        fig.savefig(Path(res["run_dir"]) / f"slice_R1z{z_r1}_R2z{z_r2}.png", dpi=100)
    plt.show()

    # 확대: 같은 R1 세포 ID를 두 라운드에서 같은 번호로 표시. R2는 그 세포 자신의 z 단면을 읽음.
    rng = np.random.default_rng(seed)
    cand = np.flatnonzero(s1 & (st == 1))
    if len(cand) == 0:
        return
    fig, ax = plt.subplots(crops, 2, figsize=(14, 7 * crops)); ax = np.atleast_2d(ax)
    h = crop_px // 2
    for r in range(crops):
        c = cand[rng.integers(len(cand))]
        x1, y1 = A[c, :2]; x2, y2, zc = Bv[c]
        zc = int(np.clip(np.round(zc), 0, len(f2) - 1))
        I2c = read_slice(f2, zc)
        sub1 = np.flatnonzero(s1 & (np.abs(A[:, 0] - x1) < h * 0.8) & (np.abs(A[:, 1] - y1) < h * 0.8))
        sub1 = sub1[np.argsort(np.hypot(A[sub1, 0] - x1, A[sub1, 1] - y1))][:20]
        for col, (img, cx, cy) in enumerate(((I1, x1, y1), (I2c, x2, y2))):
            X0, Y0 = max(int(cx) - h, 0), max(int(cy) - h, 0)
            crop = img[Y0:Y0 + crop_px, X0:X0 + crop_px]
            ax[r, col].imshow(_contrast(crop), cmap="gray", extent=(X0, X0 + crop.shape[1], Y0 + crop.shape[0], Y0))
            for n_, j in enumerate(sub1):
                if col == 0:
                    px, py, ec = A[j, 0], A[j, 1], "#00e5ff"
                else:
                    if abs(Bv[j, 2] - zc) > 2:   # 이 R2 단면에서 멀리 떨어진 세포는 표시하지 않음
                        continue
                    px, py, ec = Bv[j, 0], Bv[j, 1], {1: "#00ff66", 2: "#ff33cc", 3: "#4da6ff"}[int(st[j])]
                ax[r, col].scatter(px, py, s=60, facecolors="none", edgecolors=ec, linewidths=1.2)
                ax[r, col].text(px + 2, py - 2, str(n_), color="yellow", fontsize=9)
            ax[r, col].set_title(("R1 z=%d" % z_r1 if col == 0 else "R2 z=%d" % zc) +
                                 f"  crop #{r+1} (same number = same R1 cell; green=observed, magenta=imputed, blue=imputed w/o support)", fontsize=8)
    fig.tight_layout()
    if save:
        fig.savefig(Path(res["run_dir"]) / f"slice_crops_R1z{z_r1}.png", dpi=100)
    plt.show()

    # 영상 근거 (두 가지)
    #  (a) 밝기 대비: R2 영상에서 점 위치가 주변보다 밝은가
    #  (b) patch NCC: R1 세포 patch와 R2 위치 patch가 닮았는가 (R1·R2 모두 이 두 단면에 있는 세포만)
    ev_o = evidence_score(I2, Bv[ob, :2]); ev_i = evidence_score(I2, Bv[im, :2])
    rnd = np.c_[rng.uniform(0, I2.shape[1], 5000), rng.uniform(0, I2.shape[0], 5000)]
    ev_r = evidence_score(I2, rnd)
    both = s1 & s2
    nc_o = patch_ncc(I1, A[both & (st == 1), :2], I2, Bv[both & (st == 1), :2])
    nc_i = patch_ncc(I1, A[both & (st == 2), :2], I2, Bv[both & (st == 2), :2])
    off = rng.normal(size=(int(both.sum()), 2)); off *= 15 / np.linalg.norm(off, axis=1, keepdims=True)
    nc_r = patch_ncc(I1, A[both, :2], I2, Bv[both, :2] + off)
    fig, ax = plt.subplots(1, 2, figsize=(16, 4))
    for axx, trip, bins, xl in ((ax[0], ((ev_o, "observed", "#00aa55"), (ev_i, "imputed", "#cc2299"), (ev_r, "random position", "#888888")),
                                 np.linspace(-5, 30, 71), "brightness evidence = (center - ring bg) / bg MAD"),
                                (ax[1], ((nc_o, "observed", "#00aa55"), (nc_i, "imputed", "#cc2299"), (nc_r, "null: R2 patch 15 px off", "#888888")),
                                 np.linspace(-1, 1, 81), "patch NCC (R1 cell patch vs R2 patch)")):
        for v, nm, cl in trip:
            v = v[np.isfinite(v)]
            if len(v):
                axx.hist(np.clip(v, bins[0], bins[-1]), bins, density=True, histtype="step", lw=2, color=cl,
                         label=f"{nm} (median {np.median(v):.2f}, n={len(v):,})")
        axx.set_xlabel(xl); axx.legend(fontsize=8)
    ax[0].set_title(f"R2 z={z_r2}"); ax[1].set_title(f"cells in R1 z={z_r1} and R2 z={z_r2}")
    fig.tight_layout()
    if save:
        fig.savefig(Path(res["run_dir"]) / f"slice_evidence_R1z{z_r1}.png", dpi=100)
    plt.show()
    return dict(z_r2=z_r2, evidence_observed=ev_o, evidence_imputed=ev_i, evidence_random=ev_r,
                ncc_observed=nc_o, ncc_imputed=nc_i, ncc_null=nc_r)
