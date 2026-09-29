# %% [code] 2. 사용자 설정 (여기만 확인/수정)
R1_ROOT = "/data7/SBY/20260805_13_54_23_N407_Multi_IHC_R1_SBY_destriped_DONE"
R2_ROOT = "/data7/SBY/20260817_18_35_01_N407_Multi_IHC_R2_SBY_destriped_DONE"
_ncpu = os.cpu_count() or 4
CFG = dict(
    # --- 입력/출력 ---
    R1_JSON=R1_ROOT + "/source/blobs_whole.json",     # v4와 같은 whole-brain 입력. 다른 파일이면 여기만 변경
    R2_JSON=R2_ROOT + "/source/blobs_whole.json",
    R1_TIFF_DIR=R1_ROOT + "/Ex_488_Ch0_stitched",
    R2_TIFF_DIR=R2_ROOT + "/Ex_488_Ch0_stitched",
    OUT_ROOT=R2_ROOT + "/source/syto16_R1_canonical_v5",
    SPACING_R1_XYZ_UM=(1.794, 1.794, 2.0),
    SPACING_R2_XYZ_UM=(1.794, 1.794, 2.0),
    COORD_ORDER="auto",        # "auto" = TIFF 크기로 확인 (v4 입력은 XYZ). 강제하려면 "xyz"
    NATURAL_SORT=False,        # v4와 동일 (TIFF 파일명 정렬)

    # --- 빠른 테스트 (None = whole brain) ---
    R1_Z_RANGE=None,           # 예: (1975, 1986) → 이 z의 R1만 출력. 정합은 앞뒤 QUICK_CTX_SLICES장 포함해서 수행
    QUICK_CTX_SLICES=250,
    R2_EXTRA_SLICES=300,

    # --- 1단계: 전역 3D affine ---
    INITIAL_AFFINE=None,       # 알고 있으면 4x4 (R1 µm → R2 µm). 보통 None
    ALLOW_REFLECTION=False,
    GLOBAL_SUBSAMPLE=20_000_000, GLOBAL_BINS_UM=(160., 80., 40.), GLOBAL_NCC_SAMPLES=300_000,
    GLOBAL_MAXFEV=3000, GLOBAL_MIN_Z_EXTENT_UM=400., GLOBAL_NCC_WARN=0.5,

    # --- 2단계: 독립 3D 블록 (coarse → fine). 실패 블록은 이웃으로 채우고 멈추지 않음 ---
    LEVELS=[
        dict(name="L1", block=800., step=400., vox=12., sigma=12., search=200., min_pts=300, max_pts=300_000, z_min=6., batch=8),
        dict(name="L2", block=400., step=200., vox=6., sigma=6., search=60., min_pts=200, max_pts=150_000, z_min=6., batch=16),
        dict(name="L3", block=200., step=100., vox=4., sigma=4., search=20., min_pts=100, max_pts=60_000, z_min=6., batch=32),
    ],

    # --- 3~4단계: 점 매칭 (타일 = XY 2 mm × Z 200 µm(100장), 모두 독립) ---
    ICP_ROUNDS=[dict(r=7., grid=150.), dict(r=5., grid=80.)],
    BUCKET_UM=100., TILE_XY_UM=2000., TILE_Z_UM=200.,
    R_ACCEPT_UM=5.,            # 핵 간 평균 거리(~10.8 µm)의 절반 이하
    R_ACCEPT2_UM=4.,           # 국소 변형 보정 후 2차 매칭 반경
    CONSIST_TOL_UM=3.,         # 이웃 매칭쌍들과 움직임이 이만큼 다르면 제외
    KNN_LOCAL=10, LOCAL_RADIUS_UM=60., Z_WEIGHT=1.0,

    # --- 실행 ---
    N_WORKERS=max(1, min(32, _ncpu - 2)),       # 블록 단계 프로세스 수
    N_WORKERS_TILE=max(1, min(12, _ncpu - 2)),  # 타일 단계 (메모리 더 씀)
    PROGRESS_MIN_SEC=600,      # 단계 안의 진행률은 25% 단위, 또는 오래 걸리면 10분 간격으로만 출력
    RESUME=True,               # 같은 설정이면 끝난 단계는 재사용

    # --- 시각화 ---
    VIS_Z_R1=1980, VIS_DZ=1,
)
