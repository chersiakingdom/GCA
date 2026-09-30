# %% [code] 진단 셀: result = run(CFG) 가 끝난 같은 커널에서 실행
# (커널을 재시작했다면 셀 0~2 실행 후 RUN 경로만 지정해도 1, 3번은 동작합니다. 2번은 result가 필요합니다.)
import numpy as np, matplotlib.pyplot as plt, json
from pathlib import Path
from scipy.spatial import cKDTree

RUN = Path(result["run_dir"])          # 또는 Path(".../syto16_R1_canonical_v5/run_799cd3c8930d")
sp1 = np.asarray(CFG["SPACING_R1_XYZ_UM"], float); sp2 = np.asarray(CFG["SPACING_R2_XYZ_UM"], float)
rng = np.random.default_rng(0)
d = np.load(RUN / "canonical_nodes.npz")
A = d["r1_xyz_vox"]; B = d["r2_xyz_vox"]; st = d["status"]
N = len(A)
warp = result["warp"] if "result" in globals() else Warp.load(RUN / "warp_final.npz")

# ---------------------------------------------------------------------------
# 1) 관측(매칭) 비율이 어디서 높고 어디서 낮은가  (XY / XZ 지도, 250 µm 격자, 2천만 점 표본)
# ---------------------------------------------------------------------------
BIN = 250.0
s = rng.integers(0, N, min(N, 20_000_000))
X = A[s] * sp1; ob = st[s] == 1
ij = np.floor((X - X.min(0)) / BIN).astype(int); sh = ij.max(0) + 1
fig, ax = plt.subplots(1, 3, figsize=(20, 6))
for k, (a0, a1, nm) in enumerate(((0, 1, "XY"), (0, 2, "XZ"))):
    tot = np.zeros((sh[a0], sh[a1])); hit = np.zeros_like(tot)
    np.add.at(tot, (ij[:, a0], ij[:, a1]), 1); np.add.at(hit, (ij[ob, a0], ij[ob, a1]), 1)
    r = np.where(tot >= 50, hit / np.maximum(tot, 1), np.nan)
    im = ax[k].imshow(r.T, origin="lower", vmin=0, vmax=1, cmap="viridis", aspect="auto")
    ax[k].set_title(f"observed fraction ({nm}, {BIN:g} um bins)"); plt.colorbar(im, ax=ax[k])
zb = np.floor(A[s, 2] / 50).astype(int)
tz = np.bincount(zb); hz = np.bincount(zb[ob], minlength=len(tz))
ok = tz > 1000
ax[2].plot(np.flatnonzero(ok) * 50, hz[ok] / tz[ok]); ax[2].set_xlabel("R1 z (slice)"); ax[2].set_ylabel("observed fraction")
ax[2].set_ylim(0, 1); ax[2].set_title("observed fraction vs R1 z")
fig.tight_layout(); fig.savefig(RUN / "diag_observed_maps.png", dpi=100); plt.show()

# ---------------------------------------------------------------------------
# 2) 예측 위치에서 가장 가까운 R2 검출까지의 거리/방향 (반경 제한 없음) + 우연 수준(null)
#    - 실제 대응이 있으면 작은 거리에 뚜렷한 초과분이 생김
#    - dz 분포가 dx,dy보다 훨씬 넓으면 z 방향 위치 오차가 커서 5 µm 반경이 너무 좁은 것
# ---------------------------------------------------------------------------
R2all = np.concatenate([B[st == 1], result["r2_unmatched_vox"]]) * sp2
N_REG, HALF = 12, 200.0
cent = A[rng.integers(0, N, N_REG * 20)] * sp1
rs = np.linspace(0.5, 15, 30)
fig, ax = plt.subplots(1, 2, figsize=(15, 5))
comp_all, rows = [], []
for c in cent:
    m = np.all(np.abs(A * sp1 - c) < HALF, axis=1) if N < 5e6 else None
    if m is None:   # 큰 데이터: z 먼저 걸러서 비용 절감
        zi = np.flatnonzero(np.abs(A[:, 2] * sp1[2] - c[2]) < HALF)
        zi = zi[np.all(np.abs(A[zi, :2] * sp1[:2] - c[:2]) < HALF, axis=1)]
    else:
        zi = np.flatnonzero(m)
    if len(zi) < 500:
        continue
    P = warp.apply(A[zi] * sp1).astype(np.float64)
    lo, hi = P.min(0) - 60, P.max(0) + 60
    zz = np.flatnonzero((R2all[:, 2] > lo[2]) & (R2all[:, 2] < hi[2]))
    Y = R2all[zz[np.all((R2all[zz] > lo) & (R2all[zz] < hi), axis=1)]]
    if len(Y) < 100:
        continue
    t = cKDTree(Y)
    dd, jj = t.query(P); comp_all.append((Y[jj] - P)[dd < 15])
    v = rng.normal(size=3); v *= 40 / np.linalg.norm(v)
    dn, _ = t.query(P + v)
    f_obs = [(dd < r).mean() for r in rs]; f_nul = [(dn < r).mean() for r in rs]
    rows.append(dict(center_um=c.round(0).tolist(), n=len(zi), within5=round(float(f_obs[9]), 3), null5=round(float(f_nul[9]), 3),
                     observed_status_frac=round(float((st[zi] == 1).mean()), 3)))
    ax[0].plot(rs, np.array(f_obs) - np.array(f_nul), alpha=.6)
    if len(rows) >= N_REG:
        break
ax[0].axvline(CFG["R_ACCEPT_UM"], ls="--", c="k"); ax[0].set_xlabel("radius (um)")
ax[0].set_ylabel("P(R2 within r) - null(40um shift)"); ax[0].set_title("excess over chance, per region")
C = np.concatenate(comp_all)
for k, nm in enumerate("xyz"):
    ax[1].hist(C[:, k], np.linspace(-15, 15, 61), histtype="step", lw=2, label=f"d{nm}", density=True)
ax[1].legend(); ax[1].set_title("predicted -> nearest R2 detection (components, um)")
fig.tight_layout(); fig.savefig(RUN / "diag_nn_displacement.png", dpi=100); plt.show()
for r in rows:
    print(r)
print("dx,dy,dz 표준편차(µm, |d|<15):", np.round(C.std(0), 2))

# ---------------------------------------------------------------------------
# 3) 블록 정합 신뢰도 지도 (L1~L3): 블록 상태의 XY 분포
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(2, 3, figsize=(20, 11))
for i in range(3):
    p = RUN / f"reliability_L{i+1}.npz"
    if not p.exists():
        continue
    rel = np.load(p); code = rel["code"]
    tested = code != 1
    okf = np.where(tested.sum(2) > 0, (code == 0).sum(2) / np.maximum(tested.sum(2), 1), np.nan)
    bnd = np.where(tested.sum(2) > 0, (code == 3).sum(2) / np.maximum(tested.sum(2), 1), np.nan)
    a = ax[0, i].imshow(okf.T, origin="lower", vmin=0, vmax=1); plt.colorbar(a, ax=ax[0, i])
    ax[0, i].set_title(f"L{i+1}: fraction of valid blocks (XY)")
    b = ax[1, i].imshow(bnd.T, origin="lower", vmin=0, vmax=1, cmap="magma"); plt.colorbar(b, ax=ax[1, i])
    ax[1, i].set_title(f"L{i+1}: fraction hitting search boundary (XY)")
fig.tight_layout(); fig.savefig(RUN / "diag_block_reliability.png", dpi=100); plt.show()
print("저장:", RUN / "diag_observed_maps.png", RUN / "diag_nn_displacement.png", RUN / "diag_block_reliability.png")
