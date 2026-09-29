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
VERSION = "5.0-global-affine+independent-3d-blocks"

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
class Warp:
    def __init__(self, A=None, b=None):
        self.A = np.eye(3) if A is None else np.asarray(A, float)
        self.b = np.zeros(3) if b is None else np.asarray(b, float)
        self.levels = []   # dict(origin(3), step, U(nx,ny,nz,3) float32)

    def copy(self):
        w = Warp(self.A.copy(), self.b.copy()); w.levels = [dict(L) for L in self.levels]; return w

    def apply(self, X, chunk=2_000_000):
        X = np.asarray(X)
        out = np.empty((len(X), 3), np.float32)
        for s in range(0, len(X), chunk):
            x = X[s:s + chunk].astype(np.float64)
            y = x @ self.A.T + self.b
            for L in self.levels:
                g = ((x - L["origin"]) / L["step"]).T
                for c in range(3):
                    y[:, c] += ndi.map_coordinates(L["U"][..., c], g, order=1, mode="nearest")
            out[s:s + chunk] = y
        return out

    def save(self, path):
        d = dict(A=self.A, b=self.b, n_levels=len(self.levels))
        for i, L in enumerate(self.levels):
            d[f"L{i}_origin"] = L["origin"]; d[f"L{i}_step"] = L["step"]; d[f"L{i}_U"] = L["U"]
        np.savez_compressed(path, **d)

    @classmethod
    def load(cls, path):
        f = np.load(path)
        w = cls(f["A"], f["b"])
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


def local_median_residual(Xp, R, Xq, k, radius, chunk=500_000):
    """Xq 각 점에 대해 가까운 k개 매칭쌍(Xp)의 잔차 중앙값 / MAD / 개수."""
    med = np.zeros((len(Xq), 3), np.float32); mad = np.full(len(Xq), np.nan, np.float32)
    nsup = np.zeros(len(Xq), np.int16)
    if len(Xp) < 3:
        return med, mad, nsup
    kk = min(k, len(Xp)); t = cKDTree(Xp)
    for s in range(0, len(Xq), chunk):
        d, j = t.query(Xq[s:s + chunk], k=kk, distance_upper_bound=radius)
        if kk == 1:
            d, j = d[:, None], j[:, None]
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
    med, _, nsup = local_median_residual(X[mk], R, X[mk], cfg["KNN_LOCAL"] + 1, cfg["LOCAL_RADIUS_UM"])
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


def run_final_matching(B1, B2, warp, cfg, tiles):
    N1 = len(B1.P)
    m2 = np.full(N1, -1, np.int64); dist = np.full(N1, np.inf, np.float32)
    pos = np.zeros((N1, 3), np.float32); mad = np.full(N1, np.nan, np.float32); nsup = np.zeros(N1, np.int16)
    _G.update(B1=B1, B2=B2, warp=warp, cfg=cfg, tiles=tiles, mode="final", r=cfg["R_ACCEPT_UM"])
    prog = Progress(len(tiles), f"최종 매칭 ({len(tiles)} 타일)")
    with _pool(cfg["N_WORKERS_TILE"]) as pool:
        for tid, res in pool.imap_unordered(_tile_task, range(len(tiles))):
            prog.update(1)
            if res is None:
                continue
            i = res["i1"]
            m2[i] = res["m2"]; dist[i] = res["d"]; pos[i] = res["pos"]; mad[i] = res["mad"]; nsup[i] = res["nsup"]
    # 타일 경계에서 같은 R2 점을 두 R1이 가진 경우: 더 가까운 쪽만 유지
    has = np.flatnonzero(m2 >= 0)
    o = has[np.lexsort((dist[has], m2[has]))]
    dup = np.r_[False, m2[o][1:] == m2[o][:-1]]
    lose = o[dup]
    if len(lose):
        m2[lose] = -1; dist[lose] = np.inf
        pos[lose] = warp.apply(B1.P[lose])
    return dict(m2=m2, dist=dist, pos=pos, mad=mad, nsup=nsup, n_conflicts=int(len(lose)))


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
def _sig(cfg, keys):
    d = {k: cfg[k] for k in keys}
    for k in ("R1_JSON", "R2_JSON"):
        st = os.stat(cfg[k]); d[k + "_stat"] = [st.st_size, int(st.st_mtime)]
    return hashlib.sha1(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()[:12]


def run(cfg):
    global _T0
    _T0 = time.time()
    CFG.update(cfg)
    reg_keys = ["R1_JSON", "R2_JSON", "SPACING_R1_XYZ_UM", "SPACING_R2_XYZ_UM", "COORD_ORDER", "R1_Z_RANGE",
                "QUICK_CTX_SLICES", "R2_EXTRA_SLICES", "GLOBAL_BINS_UM", "LEVELS", "ICP_ROUNDS", "ALLOW_REFLECTION",
                "INITIAL_AFFINE"]
    run_dir = Path(cfg["OUT_ROOT"]) / f"run_{_sig(cfg, reg_keys)}"
    run_dir.mkdir(parents=True, exist_ok=True)
    json.dump({k: v for k, v in cfg.items()}, open(run_dir / "config.json", "w"), indent=1, default=str)
    log(f"SYTO16 R1→R2 canonical matching {VERSION}")
    log(f"결과 폴더: {run_dir}")

    # ---------------- 0. 입력 ----------------
    stage("0/5 입력 읽기")
    m1 = tiff_meta(cfg["R1_TIFF_DIR"], cfg["NATURAL_SORT"]); m2 = tiff_meta(cfg["R2_TIFF_DIR"], cfg["NATURAL_SORT"])
    for nm, m in (("R1", m1), ("R2", m2)):
        log(f"  {nm} TIFF: " + (f"{m['nz']}장, {m['W']}x{m['H']} (W×H)" if m else "찾지 못함"))
    F1 = load_points_json(cfg["R1_JSON"]); F2 = load_points_json(cfg["R2_JSON"])
    c1 = coord_columns(F1, m1, cfg["COORD_ORDER"]); c2 = coord_columns(F2, m2, cfg["COORD_ORDER"])
    V1 = F1.raw[:, list(c1)]; V2 = F2.raw[:, list(c2)]
    log(f"  R1 {len(V1):,}점, R2 {len(V2):,}점; 좌표 열(x,y,z)=R1{c1} R2{c2}")
    log(f"  R1 범위 x[{V1[:,0].min():.0f},{V1[:,0].max():.0f}] y[{V1[:,1].min():.0f},{V1[:,1].max():.0f}] "
        f"z[{V1[:,2].min():.0f},{V1[:,2].max():.0f}]")
    log(f"  R2 범위 x[{V2[:,0].min():.0f},{V2[:,0].max():.0f}] y[{V2[:,1].min():.0f},{V2[:,1].max():.0f}] "
        f"z[{V2[:,2].min():.0f},{V2[:,2].max():.0f}]")
    sp1 = np.asarray(cfg["SPACING_R1_XYZ_UM"], np.float64); sp2 = np.asarray(cfg["SPACING_R2_XYZ_UM"], np.float64)

    # 빠른 테스트: R1 z 범위 + 주변 문맥 (문맥은 정합에만 사용, 출력은 핵심 범위만)
    sel1 = np.arange(len(V1)); core_mask = None
    if cfg["R1_Z_RANGE"] is not None:
        z0, z1 = cfg["R1_Z_RANGE"]; ctx = cfg["QUICK_CTX_SLICES"]
        sel1 = np.flatnonzero((V1[:, 2] >= z0 - ctx) & (V1[:, 2] < z1 + ctx))
        core_mask = (V1[sel1, 2] >= z0) & (V1[sel1, 2] < z1)
        zz = V1[sel1, 2]
        e = cfg["R2_EXTRA_SLICES"]
        sel2 = np.flatnonzero((V2[:, 2] >= zz.min() - e) & (V2[:, 2] <= zz.max() + e))
        log(f"  빠른 테스트: 출력 R1 z[{z0},{z1}) {core_mask.sum():,}점, 정합 문맥 {len(sel1):,}점, R2 후보 {len(sel2):,}점")
    else:
        sel2 = np.arange(len(V2))
    P1 = (V1[sel1] * sp1).astype(np.float32); P2 = (V2[sel2] * sp2).astype(np.float32)

    log("  공간 인덱스 생성 중...")
    B1 = Buckets(P1, cfg["BUCKET_UM"]); B2 = Buckets(P2, cfg["BUCKET_UM"])
    del P1, P2; gc.collect()

    # ---------------- 1. 전역 affine ----------------
    stage("1/5 전역 3D affine 정합 (뇌 전체 밀도, 회전/기울기/배율/이동)")
    wpath = run_dir / "warp_global.npz"; report = dict(version=VERSION, run_dir=str(run_dir))
    if cfg["RESUME"] and wpath.exists():
        warp = Warp.load(wpath); report["global"] = json.load(open(run_dir / "global.json"))
        log("  이전 결과 재사용")
    else:
        if cfg["INITIAL_AFFINE"] is not None:
            H = np.asarray(cfg["INITIAL_AFFINE"], float)
            warp = Warp(H[:3, :3], H[:3, 3]); ginfo = dict(candidate="user_initial_affine")
            log("  사용자 지정 초기 affine 사용")
        else:
            warp, ginfo, G = global_affine(B1.P, B2.P, cfg)
            try:
                save_global_overlay(G, warp, run_dir / "global_overlay.png")
            except Exception as ex:
                log(f"  (overlay 그림 생략: {ex})")
        warp.save(wpath); json.dump(ginfo, open(run_dir / "global.json", "w"), indent=1)
        report["global"] = ginfo

    # ---------------- 2. 독립 3D 블록 비선형 보정 ----------------
    stage("2/5 3D 블록 비선형 보정 (블록마다 독립, 실패 블록은 이웃으로 채움)")
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

    # ---------------- 3. 점 기반 ICP 보정 ----------------
    stage("3/5 점 기반 미세 보정 (상호 최근접 + 이웃 일관성, 타일 독립)")
    tiles = make_tiles(B1, cfg["TILE_XY_UM"], cfg["TILE_Z_UM"])
    log(f"  타일 {len(tiles)}개 (XY {cfg['TILE_XY_UM']:g} µm × Z {cfg['TILE_Z_UM']:g} µm ≈ {cfg['TILE_Z_UM']/sp1[2]:.0f}장)")
    report["icp"] = []
    for i, rnd in enumerate(cfg["ICP_ROUNDS"]):
        p = run_dir / f"warp_ICP{i+1}.npz"
        if cfg["RESUME"] and p.exists():
            warp = Warp.load(p); report["icp"].append(json.load(open(run_dir / f"icp_{i+1}.json")))
            log(f"  ICP {i+1} 이전 결과 재사용"); continue
        info = run_icp_round(B1, B2, warp, rnd, cfg, tiles)
        warp.save(p); json.dump(info, open(run_dir / f"icp_{i+1}.json", "w"), indent=1)
        report["icp"].append(info)
    warp.save(run_dir / "warp_final.npz")

    # ---------------- 4. 최종 매칭 + 누락점 보완 ----------------
    stage("4/5 최종 1:1 매칭 + 누락 R1 세포 위치 추정")
    M = run_final_matching(B1, B2, warp, cfg, tiles)
    N1 = len(B1.P)
    # 정렬 순서 → 원래(선택된) 순서
    inv1 = B1.order
    m2_sorted = M["m2"]
    matched = m2_sorted >= 0
    m2_orig_sel = np.full(N1, -1, np.int64)
    m2_orig_sel[matched] = sel2[B2.order[m2_sorted[matched]]]
    arr = lambda a: _unsort(a, inv1)
    m2_o = arr(m2_orig_sel); dist_o = arr(M["dist"]); pos_o = arr(M["pos"]); mad_o = arr(M["mad"]); nsup_o = arr(M["nsup"])
    status = np.where(m2_o >= 0, 1, 2).astype(np.uint8)   # 1 관측(R2 검출과 매칭), 2 추정(보완)

    # 출력 대상 (빠른 테스트면 핵심 z 범위만)
    out_idx = np.arange(N1) if core_mask is None else np.flatnonzero(core_mask)
    r1_rows = sel1[out_idx]
    n_obs = int((status[out_idx] == 1).sum()); n_imp = int((status[out_idx] == 2).sum())
    used2 = np.zeros(len(V2), bool); used2[m2_o[out_idx][m2_o[out_idx] >= 0]] = True
    # 사용되지 않은 R2 검출 (출력 범위에 대응하는 영역만 세기 위해: 전체 모드에서는 전체)
    if core_mask is None:
        unused2 = np.flatnonzero(~used2)
    else:
        unused2 = np.array([], np.int64)
    log(f"  R1 {len(out_idx):,}개 → 관측(R2 검출 매칭) {n_obs:,} ({100*n_obs/max(len(out_idx),1):.1f}%), "
        f"추정 보완 {n_imp:,} ({100*n_imp/max(len(out_idx),1):.1f}%), 경계 중복 해소 {M['n_conflicts']:,}")

    # ---------------- 5. 저장 ----------------
    stage("5/5 저장")
    rows = np.empty((len(out_idx), F2.ncol), np.float64)
    ob = status[out_idx] == 1
    rows[ob] = F2.raw[m2_o[out_idx][ob]]
    im = ~ob
    if F1.ncol == F2.ncol:
        rows[im] = F1.raw[r1_rows[im]]
    else:
        rows[im] = 0
    vox_imp = pos_o[out_idx][im].astype(np.float64) / sp2
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
                        status=status[out_idx], r2_row=m2_o[out_idx], match_dist_um=dist_o[out_idx],
                        local_mad_um=mad_o[out_idx], n_support=nsup_o[out_idx],
                        r2_unmatched_rows=unused2,
                        status_legend=np.array(["", "1=observed(R2 detection)", "2=imputed(from R1 + neighbor deformation)"]))
    # QC
    ep = edge_preservation(B1, M["m2"] >= 0, M["pos"])
    if ep.get("n_blocks"):
        log(f"  그래프 엣지 보존율(k=8): 관측 {ep['preserved_p50']:.3f} vs 셔플 null {ep['null_p50']:.3f} "
            f"({ep['n_blocks']} 블록)")
    dd = dist_o[out_idx][ob]
    report.update(n_r1=int(len(out_idx)), n_observed=n_obs, n_imputed=n_imp, n_r2_input=int(len(V2)),
                  n_r2_unmatched=int(len(unused2)) if core_mask is None else None,
                  match_dist_um=dict(p50=float(np.median(dd)) if len(dd) else None,
                                     p90=float(np.percentile(dd, 90)) if len(dd) else None),
                  edge_preservation=ep, n_tile_conflicts=M["n_conflicts"],
                  final_json=str(final_json), nodes_npz=str(run_dir / "canonical_nodes.npz"),
                  coord_cols_r1=list(c1), coord_cols_r2=list(c2), quick_test_range=cfg["R1_Z_RANGE"])
    json.dump(report, open(run_dir / "report.json", "w"), indent=1, default=float)
    assert len(rows) == len(out_idx)
    log(f"완료. R1 {len(out_idx):,}행 = 최종 R2 {len(rows):,}행. 보고서: {run_dir/'report.json'}")
    return dict(report=report, run_dir=run_dir, r1_xyz_vox=V1[r1_rows], r2_xyz_vox=r2_vox,
                status=status[out_idx], r2_unmatched_vox=V2[unused2] if len(unused2) else np.empty((0, 3)),
                warp=warp)


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
          f"(관측 {(s2&(st==1)).sum():,}, 추정 {(s2&(st==2)).sum():,}, 미사용 R2 검출 {int(su.sum()):,})")

    ds = max(1, int(np.ceil(max(I1.shape) / 2200)))
    fig, ax = plt.subplots(1, 2, figsize=(22, 11))
    ax[0].imshow(_contrast(I1)[::ds, ::ds], cmap="gray")
    ax[0].scatter(A[s1, 0] / ds, A[s1, 1] / ds, s=1, c="#00e5ff", lw=0)
    ax[0].set_title(f"R1 z={z_r1}  (R1 points within +-{dz} slice)")
    ax[1].imshow(_contrast(I2)[::ds, ::ds], cmap="gray")
    ob = s2 & (st == 1); im = s2 & (st == 2)
    ax[1].scatter(Bv[ob, 0] / ds, Bv[ob, 1] / ds, s=1, c="#00ff66", lw=0, label="observed (matched R2 detection)")
    ax[1].scatter(Bv[im, 0] / ds, Bv[im, 1] / ds, s=1, c="#ff33cc", lw=0, label="imputed (R1 cell, predicted)")
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
                    px, py, ec = Bv[j, 0], Bv[j, 1], ("#00ff66" if st[j] == 1 else "#ff33cc")
                ax[r, col].scatter(px, py, s=60, facecolors="none", edgecolors=ec, linewidths=1.2)
                ax[r, col].text(px + 2, py - 2, str(n_), color="yellow", fontsize=9)
            ax[r, col].set_title(("R1 z=%d" % z_r1 if col == 0 else "R2 z=%d" % zc) +
                                 f"  crop #{r+1} (same number = same R1 cell ID; green=observed, magenta=imputed)", fontsize=9)
    fig.tight_layout()
    if save:
        fig.savefig(Path(res["run_dir"]) / f"slice_crops_R1z{z_r1}.png", dpi=100)
    plt.show()

    # 영상 근거 지수(형률님 evidence index의 단순형): 관측 vs 추정 vs 무작위 위치
    ev_o = evidence_score(I2, Bv[ob, :2]); ev_i = evidence_score(I2, Bv[im, :2])
    rnd = np.c_[rng.uniform(0, I2.shape[1], 5000), rng.uniform(0, I2.shape[0], 5000)]
    ev_r = evidence_score(I2, rnd)
    fig, ax = plt.subplots(figsize=(8, 4)); bins = np.linspace(-5, 30, 71)
    for v, nm, cl in ((ev_o, "observed", "#00aa55"), (ev_i, "imputed", "#cc2299"), (ev_r, "random position", "#888888")):
        v = v[np.isfinite(v)]
        if len(v):
            ax.hist(np.clip(v, -5, 30), bins, density=True, histtype="step", lw=2, color=cl,
                    label=f"{nm} (median {np.median(v):.1f}, n={len(v):,})")
    ax.set_xlabel("image evidence = (center - ring background) / background MAD"); ax.legend()
    ax.set_title(f"R2 z={z_r2}")
    fig.tight_layout(); plt.show()
    return dict(z_r2=z_r2, evidence_observed=ev_o, evidence_imputed=ev_i, evidence_random=ev_r)
