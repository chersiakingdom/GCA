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
    OUT_ROOT=R2_ROOT + "/source/syto16_R1_canonical_v6",
    SPACING_R1_XYZ_UM=(1.794, 1.794, 2.0),
    SPACING_R2_XYZ_UM=(1.794, 1.794, 2.0),
    COORD_ORDER="auto",        # "auto" = TIFF 크기로 확인 (v4 입력은 XYZ). 강제하려면 "xyz"
    NATURAL_SORT=False,        # v4와 동일 (TIFF 파일명 정렬)

    # --- 빠른 테스트 (None = whole brain) ---
    R1_Z_RANGE=None,           # 예: (1975, 1986) → 이 z의 R1만 출력. 정합은 앞뒤 QUICK_CTX_SLICES장 포함해서 수행
    QUICK_CTX_SLICES=250,
    R2_EXTRA_SLICES=300,

    # --- 이전 실행의 변형장 재사용 (1~3단계 생략, 약 1시간 절약) ---
    # v5 결과를 출발점으로: R2_ROOT + "/source/syto16_R1_canonical_v5/run_799cd3c8930d/warp_final.npz"
    # 처음부터 다시 하려면 None
    INIT_WARP=R2_ROOT + "/source/syto16_R1_canonical_v5/run_799cd3c8930d/warp_final.npz",

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

    # --- 3단계: 점 기반 미세 보정 (INIT_WARP 사용 시 생략) ---
    ICP_ROUNDS=[dict(r=7., grid=150.), dict(r=5., grid=80.)],
    BUCKET_UM=100., TILE_XY_UM=2000., TILE_Z_UM=200.,   # 타일 = XY 2 mm × Z 200 µm(100장), 모두 독립 처리
    R_ACCEPT_UM=5., KNN_LOCAL=10, Z_WEIGHT=1.0,

    # --- 4~5단계: 상대좌표 그래프 매칭 (모든 세포를 이웃 상대벡터 패턴으로 표현해 비교) ---
    GRAPH_K1=10,               # R1 세포마다 비교에 쓰는 이웃 수
    GRAPH_K2=14,               # R2 세포마다 이웃 수 (누락/가짜 검출을 감안해 더 많이)
    GRAPH_TOL_UM=4.0,          # 상대 벡터가 (이 값 + 벡터길이×GRAPH_TOL_FRAC) 안이면 "같은 이웃"으로 인정
    GRAPH_TOL_FRAC=0.0,        # 국소 변형(strain)이 큰 데이터면 0.1 정도로 (거리 10 µm 이웃이면 +1 µm)
    GRAPH_Z_WEIGHT=1.0,        # 상대 벡터 비교에서 z 차이 가중치 (R2 z 오차가 크면 0.5~0.7)
    GRAPH_HALO_UM=40.,         # 타일 경계에서 이웃을 잃지 않도록 여유
    GRAPH_PASSES=[             # 변형장 보정용 (넓은 범위 → 좁은 범위). 표본 세포만 평가
        # 이웃 합의 투표: 세포 i의 후보 변위를 주변 R1 이웃 VOTE_K개도 똑같이 갖는지로 판정 (넓은 범위에서도 견고)
        dict(mode="vote", r_cand=35., kc=24, stride=10, grid=80.),
        dict(mode="vote", r_cand=18., kc=12, stride=4, grid=50.),
        # 상대벡터 패턴 (세포 단위 그래프 일치)
        dict(mode="graph", r_cand=10., kc=8, stride=2, grid=40.),
    ],
    VOTE_K=24, VOTE_TOL_UM=4.0, VOTE_MIN_SCORE=0.25, VOTE_MARGIN=0.08,
    GRAPH_FINAL=dict(r_cand=10., kc=8),   # 전체 세포 최종 매칭
    GRAPH_FDR=0.01,            # null(무작위 위치) 대비 거짓 매칭률 목표 → 점수 임계값 자동 결정
    GRAPH_MIN_SCORE=0.3,       # 이웃 10개 중 최소 3개는 일치
    GRAPH_MARGIN=0.1,          # 1등 후보가 2등보다 이웃 1개 이상 더 일치해야 채택
    NULL_SHIFT_UM=40.,

    # --- 5단계: 누락 세포 보완 (국소 affine) + 검증 ---
    LOCAL_K=12, LOCAL_RADIUS_UM=80., LOCAL_SIGMA_UM=30., LOCAL_RIDGE=1.0,
    LOCAL_MIN_SUPPORT=6,       # 반경 안 관측 세포가 이보다 적으면 status=3 (전역 변형장만으로 추정, 신뢰 낮음)
    PLANE_Z_SCALE=5.0,         # 같은 R1 z-평면 이웃 우선 (z 차이를 5배로 계산) → 같은 평면이 R2에서도 한 면을 이루도록
    PLANE_TOL_UM=3.0,          # 같은 평면 이웃들이 예측하는 z와 이만큼(또는 편차 중앙값×3) 넘게 다르면 짝 제외
    CONSIST_TOL_UM=3.0,        # 주변 관측 세포들의 국소 변형과 이만큼(또는 타일 편차 중앙값×3) 넘게 다르면 짝 제외
    NN_FILL=True, R_ACCEPT2_UM=4.0,   # 그래프 매칭 후 국소 보정 위치에서 남은 세포를 엄격한 상호 최근접으로 2차 매칭
    HOLDOUT_FRAC=0.05, HOLDOUT_OK_UM=5.0,   # 관측 세포 5%를 가리고 맞혀 보는 검증, 5 µm = 핵 간격의 절반

    # --- 실행 ---
    N_WORKERS=max(1, min(32, _ncpu - 2)),       # 블록 단계 프로세스 수
    N_WORKERS_TILE=max(1, min(12, _ncpu - 2)),  # 타일 단계 (메모리 더 씀)
    PROGRESS_MIN_SEC=600,      # 단계 안의 진행률은 25% 단위, 또는 오래 걸리면 10분 간격으로만 출력
    RESUME=True,               # 같은 설정이면 끝난 단계는 재사용

    # --- 시각화 ---
    VIS_Z_R1=1980, VIS_DZ=1,
)
