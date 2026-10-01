# ============================================================
# AAV-Cre M1 (n=4)  vs  intranasal EV reinjection (n=3)
#
# 분석 전략 문서
#   "5. EV network pattern vs intranasal EV injection 시
#    recombination cell pattern"
# 에 해당하는 설정 파일.
#
# 이 파일의 값만 확인/수정하면 됩니다.
# ============================================================

from pathlib import Path


# ============================================================
# 1. REINJECTION SAMPLE (intranasal EV, n=3)
#
# 각 path 아래
#   source/results/tdt_total_cell_count_rh_lvl{1..7}.csv
#   source/results/tdt_total_cell_count_whole_lvl{1..7}.csv
# 를 읽습니다.
#
# lh = whole - rh 로 계산합니다.
# ============================================================

REINJ_ROOTS = {

    "Reinj_1":
        "/data5/20231202_14_23_01_2nd_EV_reinj_#4_IN_M1_1_destriped_DONE",

    "Reinj_2":
        "/data5/20240104_15_44_33_2nd_EV_reinj_#5_Intranasal_M1_2_destriped_DONE",

    "Reinj_3":
        "/data5/20231203_13_15_02_2nd_EV_reinj_#6_IN_M1_3_destriped_DONE",

}


# ============================================================
# 2. FIGURE 1 분석 결과 폴더
#
# 기존 코드들의 RESULTS["output"] 과 동일한 폴더.
#
# 같은 notebook에서 Figure 1 분석 직후 실행하면
# None 그대로 두어도 RESULTS["output"] 을 자동으로 사용합니다.
# ============================================================

FIG1_RUN_DIR = None


# ============================================================
# 3. 출력 폴더
#
# None 이면 <FIG1_RUN_DIR>/../Figure_list_reinjection
# ============================================================

OUTPUT_DIR = None


# ============================================================
# 4. 비교 대상 source
#
# AAV-Cre M1 (n=4) 와 비교합니다.
# ============================================================

MATCHED_SOURCE = "Motor"

SOURCE_ORDER = [
    "Motor",
    "dHP",
    "S1",
    "PFC",
]


SOURCE_LABEL = {
    "Motor": "M1",
    "dHP": "dHP",
    "S1": "S1",
    "PFC": "PFC",
}


SOURCE_COLORS = {
    "Motor": "#B75B45",
    "dHP": "#39846F",
    "S1": "#3979AD",
    "PFC": "#8062A0",
}


# reinjection group 전용 색 (기존 4개 source color와 겹치지 않음)
REINJ_COLOR = "#C8912B"

MATCH_COLOR = "#EF7C3E"


GROUP_LABEL = {
    "Cre": "AAV-Cre (M1)",
    "Reinj": "Intranasal EV reinjection",
}


# ============================================================
# 5. HEMISPHERE 처리
#
# intranasal 투여는 injection hemisphere가 존재하지 않으므로
# ipsilateral / contralateral 을 정의할 수 없습니다.
#
# 따라서 primary analysis 에서는 두 group 모두
# 양쪽 hemisphere를 합한 bilateral density 를 사용합니다.
#
#   "pooled"
#       Cre   : ipsi + contra
#       Reinj : rh + lh
#       -> 양쪽 모두 bilateral. 기본값.
#
#   "rh_as_ipsi"
#       Reinj 의 rh 를 ipsilateral 로 간주.
#       intranasal 에는 생물학적 근거가 없으므로
#       sensitivity analysis 목적으로만 사용하십시오.
# ============================================================

HEMI_MODE = "pooled"


# ============================================================
# 6. REGION SET (feature space)
#
# 분석 전략 문서 1번 / 5-A 에 따라 두 수준으로 수행합니다.
#
#   "PredefinedM1"
#       Figure 1 에서 M1 source 의 recipient region 으로
#       정의된 영역만 사용.
#
#   "Predefined20"
#       Figure 1 predefined recipient region 20개 전체 (union).
#       source-specificity 분석에 필요합니다.
#
#   "AllGrayMatter"
#       Figure 1 run 의
#       primary/AllGrayMatter/<source>/feature_definitions.csv
#       에 정의된 region 을 그대로 사용 (broad analysis).
#       정의 파일을 해석할 수 없으면 자동으로 제외됩니다.
# ============================================================

SCOPES = [
    "PredefinedM1",
    "Predefined20",
    "AllGrayMatter",
]


SCOPE_LABEL = {
    "PredefinedM1": "M1 predefined recipient regions",
    "Predefined20": "Predefined 20 regions",
    "AllGrayMatter": "All gray matter",
}


# ============================================================
# 7. Permutation (분석 전략 문서 5-B)
# ============================================================

N_PERMUTATION = 10000

PERMUTATION_SEED = 20231202


# ============================================================
# 8. QC tolerance
# ============================================================

AREA_ATOL = 1e-6

COUNT_ATOL = 1e-8

# lh = whole - rh 에서 허용하는 음수 크기
# (반올림/저장 과정에서 발생하는 수준만 허용)
NEGATIVE_TOL = 1e-6

# count 는 정수 단위이므로 -0.5 미만의 음수는 실제 불일치로 처리
NEGATIVE_COUNT_TOL = 0.5


SHOW_FIGURES = True


# ============================================================
# 9. 파일 이름 규칙
#
# 다른 이름으로 저장되어 있으면 여기만 수정하십시오.
# ============================================================

RH_FILE_TEMPLATE = "tdt_total_cell_count_rh_lvl{level}.csv"

WHOLE_FILE_TEMPLATE = "tdt_total_cell_count_whole_lvl{level}.csv"

RESULTS_SUBDIR = "source/results"

LEVELS = [1, 2, 3, 4, 5, 6, 7]


def reinj_results_dir(root):
    return Path(root).expanduser() / RESULTS_SUBDIR
