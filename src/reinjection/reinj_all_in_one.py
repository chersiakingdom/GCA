# ============================================================
# AAV-Cre M1 (n = 4)  vs  intranasal EV reinjection (n = 3)
#
# 분석 전략 문서
#   "5. EV network pattern vs intranasal EV injection 시
#    recombination cell pattern"
#
# 이 파일 하나를 Jupyter cell 에 그대로 붙여 넣고 실행하면
# 모든 figure 와 값 CSV 가 생성됩니다.
#
# 설정은 바로 아래 SETTINGS 부분만 확인하시면 됩니다.
#
# ------------------------------------------------------------
# Hemisphere
#
# 분석 전략 문서 4번에 따라 left / right 를 합치지 않고
# 각각 독립적인 anatomical feature 로 유지합니다.
#
#   한 mouse 의 p = (2 x region) 개 feature 전체에서 합이 1
#
# AAV-Cre 는 injection side 기준으로 ipsilateral / contralateral
# 로 정렬하고, intranasal reinjection 은 injection hemisphere 가
# 없으므로 RH / LH 를 그대로 같은 두 자리에 둡니다.
#
#   slot 1 : Cre ipsilateral      <-> reinjection RH
#   slot 2 : Cre contralateral    <-> reinjection LH
#
# 이 대응은 intranasal 에서는 임의 선택이므로, 반대로 뒤집은
# (RH <-> contralateral) 경우의 값도 함께 계산해서 CSV 에
# 남기고 figure 에도 함께 표기합니다.
# ============================================================

from pathlib import Path
from itertools import combinations
import ast
import copy
import json
import re
import traceback

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

try:
    import statsmodels.api as sm
except ImportError:
    sm = None


# ============================================================
# ============================================================
#  SETTINGS
# ============================================================
# ============================================================

# ------------------------------------------------------------
# 1. reinjection sample (intranasal EV, n = 3)
#
# 각 path 아래
#   source/results/tdt_total_cell_count_rh_lvl{1..7}.csv
#   source/results/tdt_total_cell_count_whole_lvl{1..7}.csv
#
# lh = whole - rh 로 계산합니다.
# ------------------------------------------------------------

REINJ_ROOTS = {

    "Reinj_1":
        "/data5/20231202_14_23_01_2nd_EV_reinj_#4_IN_M1_1_destriped_DONE",

    "Reinj_2":
        "/data5/20240104_15_44_33_2nd_EV_reinj_#5_Intranasal_M1_2_destriped_DONE",

    "Reinj_3":
        "/data5/20231203_13_15_02_2nd_EV_reinj_#6_IN_M1_3_destriped_DONE",

}


# ------------------------------------------------------------
# 2. Figure 1 분석 결과 폴더 (run_... 폴더)
#
# None 이면 아래 순서로 자동으로 찾습니다.
#
#   1) 노트북에 남아 있는 기존 코드의 변수
#      RESULTS / OVERVIEW / SCATTER_RESULTS / HELLINGER_RESULTS /
#      RUN_DIR / SCATTER_RUN_DIR / HELLINGER_RUN_DIR
#      (Figure_list 경로를 들고 있어도 옆의 run_... 을 찾습니다)
#
#   2) SEARCH_ROOTS 및 reinjection path, 현재 폴더, home 아래에서
#      SELECT_outputs/run_* 중 완료된 폴더를 자동 탐색
#      (여러 개면 가장 최근 것, 나머지는 후보로 출력)
#
# 특정 폴더를 쓰시려면 경로를 직접 적어 주십시오.
# ------------------------------------------------------------

FIG1_RUN_DIR = None


# run_... 폴더를 찾을 상위 폴더를 추가하고 싶을 때 (선택)
# 예: ["/data5/SELECT_outputs", "/data7/analysis"]

SEARCH_ROOTS = []


# ------------------------------------------------------------
# 3. 출력 폴더
#
# None 이면 <run_dir>/../Figure_list_reinjection
# ------------------------------------------------------------

OUTPUT_DIR = None


# ------------------------------------------------------------
# 4. hemisphere 처리
#
#   "lateralized" : left / right 를 각각 독립 feature 로 유지 (기본)
#   "pooled"      : 양쪽을 합쳐 bilateral 로 비교 (보조 분석용)
# ------------------------------------------------------------

HEMI_MODE = "lateralized"


# reinjection RH 를 Cre 의 어느 hemisphere 자리에 둘 것인가
#   "rh_to_ipsi"   : RH -> ipsilateral slot (기본)
#   "rh_to_contra" : RH -> contralateral slot
ORIENTATION = "rh_to_ipsi"


# ------------------------------------------------------------
# 5. region set
# ------------------------------------------------------------

# Figure 1 분석(run_analysis)이 만들어 둔 feature space 를 그대로 씁니다.
#   primary/Predefined/Motor/
#   primary/AllGrayMatter/Motor/
#
# region 을 새로 고르거나 residual 을 다시 정의하지 않습니다.

SCOPES = [
    "Predefined",
    "AllGrayMatter",
]


# ------------------------------------------------------------
# 5-1. reinjection CSV prefix / 폴더 규칙
#
# Figure 1 코드의 resolve_inputs 와 같은 우선순위를 씁니다.
#   ex_co/results  ->  results_cocheck  ->  results
# ------------------------------------------------------------

REINJ_PREFIX = "tdt_total_cell_count"

RESULTS_RULES = [
    "ex_co/results",
    "results_cocheck",
    "results",
]

# 위 규칙을 적용할 기준 폴더 (root 아래에서 차례로 시도)
RESULTS_BASES = ["source", ""]


# ------------------------------------------------------------
# 6. permutation
# ------------------------------------------------------------

N_PERMUTATION = 10000

PERMUTATION_SEED = 20231202


# ------------------------------------------------------------
# 7. 기타
# ------------------------------------------------------------

SHOW_FIGURES = True

MATCHED_SOURCE = "Motor"

AREA_ATOL = 1e-6

COUNT_ATOL = 1e-8

NEGATIVE_COUNT_TOL = 0.5

LEVELS = [1, 2, 3, 4, 5, 6, 7]

FILE_TEMPLATE = "{prefix}_{hemi}_lvl{level}.csv"


# ============================================================
# 색 / 라벨
# ============================================================

SOURCE_ORDER = ["Motor", "dHP", "S1", "PFC"]

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

REINJ_COLOR = "#C8912B"

MATCH_COLOR = "#EF7C3E"

GROUP_LABEL = {
    "Cre": "AAV-Cre (M1)",
    "Reinj": "Intranasal EV reinjection",
}

SCOPE_LABEL = {
    "Predefined": "Predefined recipient regions (M1)",
    "AllGrayMatter": "All gray matter",
}

# hemisphere slot
SLOTS = ["s1", "s2"]

SLOT_LABEL = {
    "Cre": {"s1": "Ipsilateral", "s2": "Contralateral"},
    "Reinj": {"s1": "Right", "s2": "Left"},
}


# ============================================================
# FONT
#
# 서버에 Arial 이 없으면 Liberation Sans 로 렌더링한 뒤
# 저장된 SVG 내부 font-family 만 Arial 로 바꿉니다.
# ============================================================

_available_fonts = {f.name for f in font_manager.fontManager.ttflist}

if "Arial" in _available_fonts:
    RENDER_FONT = "Arial"
elif "Liberation Sans" in _available_fonts:
    RENDER_FONT = "Liberation Sans"
else:
    RENDER_FONT = "DejaVu Sans"

TARGET_SVG_FONT = "Arial"


def rc():
    return plt.rc_context({
        "font.family": RENDER_FONT,
        "font.size": 9,
        "svg.fonttype": "none",
        "axes.unicode_minus": False,
    })


def save_svg(fig, path, show=None):

    path = Path(path)

    fig.savefig(
        path, format="svg", bbox_inches="tight", facecolor="white"
    )

    if RENDER_FONT != TARGET_SVG_FONT:

        path.write_text(
            path.read_text(encoding="utf-8").replace(
                RENDER_FONT, TARGET_SVG_FONT
            ),
            encoding="utf-8",
        )

    if SHOW_FIGURES if show is None else show:
        plt.show()

    plt.close(fig)

    return path


# ============================================================
# HELPER
# ============================================================

def require(ok, message):
    if not bool(ok):
        raise ValueError(message)


def hellinger(a, b):
    """H(p,q) = ||sqrt(p) - sqrt(q)|| / sqrt(2)"""

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    return float(
        np.linalg.norm(np.sqrt(a) - np.sqrt(b)) / np.sqrt(2)
    )


def spearman_rho(a, b):

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    if len(a) < 3 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return np.nan

    return float(
        np.corrcoef(
            pd.Series(a).rank(method="average"),
            pd.Series(b).rank(method="average"),
        )[0, 1]
    )


def to_p(density):
    """
    mouse 별 sum = 1 normalization.

    Eq6 의 off-source denominator 는 한 mouse 안에서 모든 feature 에
    동일하게 작용하는 상수이므로 p 에서 소거됩니다.
    따라서 두 dataset 의 area 단위가 달라도 p 는 영향받지 않습니다.
    """

    total = density.sum(axis=1)

    require(
        np.isfinite(total).all() and (total > 0).all(),
        "density 합이 0 이거나 유한하지 않은 mouse 가 있습니다: "
        + ", ".join(
            f"{mouse}={value:.4g}" for mouse, value in total.items()
        )
    )

    p = density.div(total, axis=0)

    require(
        np.allclose(p.sum(axis=1), 1, atol=1e-8, rtol=1e-8),
        "p 합이 1 이 아닙니다."
    )

    return p


# ============================================================
# column 인식
#
# 기존 Figure 1 코드와 동일한 규약
#   id / name / acronym / <side>_count / <side>_area
# 를 1순위로 보고, count / area 처럼 side prefix 가 없는 경우도
# 함께 처리합니다.
# ============================================================

def _normalize(name):
    return re.sub(r"[^a-z0-9]", "", str(name).strip().lower())


_SIDE_TOKENS = {
    "rh": ["rh", "right"],
    "whole": ["whole", "total", "both", "bilateral"],
    "lh": ["lh", "left"],
}


def detect_column(frame, base, role, where, side=None, required=True):
    """
    base 예: "count", "area", "id", "acronym", "name"

    side 가 주어지면 그 side 의 column 을 우선 선택하고,
    다른 side 의 column 은 후보에서 제외합니다.
    """

    lookup = {_normalize(col): col for col in frame.columns}

    base_key = _normalize(base)

    prefer = _SIDE_TOKENS.get(side, []) if side else []

    exclude = [
        token
        for key, tokens in _SIDE_TOKENS.items()
        if side is not None and key != side
        for token in tokens
    ]

    # 1. <side>_<base>
    for token in prefer:
        for key in (token + base_key, base_key + token):
            if key in lookup:
                return lookup[key]

    # 2. <base> 단독
    if base_key in lookup:
        return lookup[base_key]

    # 3. base 를 포함하되 다른 side token 은 없는 column
    for token in prefer:
        for key, original in lookup.items():
            if base_key in key and token in key:
                return original

    for key, original in lookup.items():

        if base_key not in key:
            continue

        if any(bad in key for bad in exclude):
            continue

        return original

    if not required:
        return None

    raise ValueError(
        f"{where}\n"
        f"'{role}' column 을 찾지 못했습니다 "
        f"(side = {side}).\n"
        f"실제 column: {list(frame.columns)}"
    )


# ============================================================
# reinjection lvl1~7 읽기
#
# Figure 1 분석 코드(run_analysis)의 load_one_csv / merge_levels /
# load_count_family 와 같은 규칙을 사용합니다.
#
#   - column: id, region, count, area
#   - 14개 파일 (whole/rh × lvl1~7) 모두 필요
#   - 여러 level 에 같은 ID 가 있으면 값이 같을 때만 1회 사용
#   - lh = whole - rh
# ============================================================

def reinj_results_directory(root):
    """
    Figure 1 코드의 resolve_inputs 와 같은 우선순위로 results 폴더를
    찾습니다: ex_co/results -> results_cocheck -> results
    """

    root = Path(root).expanduser()

    tried = []

    for base in RESULTS_BASES:

        parent = root / base if base else root

        for rule in RESULTS_RULES:

            directory = parent / rule

            tried.append(directory)

            if directory.is_dir():
                return directory, tried

    return None, tried


def load_one_csv(path, allow_empty=False):
    """
    Figure 1 코드의 load_one_csv 와 같은 검사.

    allow_empty=True 면 행이 없는 level 파일은 건너뜁니다
    (값을 만들어 넣지는 않습니다).
    """

    frame = pd.read_csv(path, encoding="utf-8-sig")

    frame.columns = [str(c).strip().lower() for c in frame.columns]

    require(
        not frame.columns.duplicated().any(),
        f"{path}: 중복 column 이름"
    )

    require(
        {"id", "region", "count", "area"}.issubset(frame.columns),
        f"{path}: id, region, count, area column 이 필요합니다.\n"
        f"실제 column: {list(frame.columns)}"
    )

    if len(frame) == 0:

        require(allow_empty, f"{path}: 비어 있는 CSV")

        return frame.assign(
            id=pd.Series(dtype="int64"),
            count=pd.Series(dtype=float),
            area=pd.Series(dtype=float),
            region=pd.Series(dtype=object),
        )[["id", "region", "count", "area"]]

    for column in ("id", "count", "area"):

        frame[column] = pd.to_numeric(frame[column], errors="raise")

        require(
            np.isfinite(frame[column]).all(),
            f"{path}: {column} 에 결측/무한값"
        )

    require(
        np.equal(frame.id, np.floor(frame.id)).all(),
        f"{path}: 정수가 아닌 ID"
    )

    frame["id"] = frame.id.astype("int64")

    require(
        not frame.id.duplicated().any(),
        f"{path}: 한 CSV 안에서 ID 중복"
    )

    require(frame.region.notna().all(), f"{path}: region 이름 누락")

    frame["region"] = frame.region.astype(str).str.strip()

    for column in ("count", "area"):
        require((frame[column] >= 0).all(), f"{path}: {column} 음수")

    require(
        not ((frame.area == 0) & (frame["count"] > 0)).any(),
        f"{path}: area=0 인데 count>0"
    )

    return frame[["id", "region", "count", "area"]].copy()


def merge_levels(files, mouse, hemi):
    """lvl1~7 을 합칩니다. 중복 ID 는 값이 일치할 때만 1회 사용."""

    parts = []

    empty = []

    for level, path in enumerate(files, start=1):

        frame = load_one_csv(path, allow_empty=True)

        if len(frame) == 0:
            empty.append(level)
            continue

        frame["level"] = level

        parts.append(frame)

    if empty:
        print(f"  {mouse} / {hemi}: lvl {empty} 는 행이 없어 건너뜁니다.")

    require(
        len(parts) > 0,
        f"{mouse} / {hemi}: lvl1~7 이 모두 비어 있습니다."
    )

    all_rows = pd.concat(parts, ignore_index=True)

    duplicated = all_rows[all_rows.id.duplicated(keep=False)]

    for rid, block in duplicated.groupby("id"):

        require(
            len({_normalize(name) for name in block.region}) == 1,
            f"{mouse} / {hemi}: level 사이 ID-name 불일치: {rid}"
        )

        for column, tolerance in (
            ("count", COUNT_ATOL), ("area", AREA_ATOL)
        ):

            require(
                np.allclose(
                    block[column], block[column].iloc[0],
                    atol=tolerance, rtol=1e-10,
                ),
                f"{mouse} / {hemi}: level 사이 중복 ID 값 불일치: "
                f"{rid} / {column}"
            )

    merged = (
        all_rows
        .drop_duplicates("id")
        .drop(columns="level")
        .set_index("id")
        .sort_index()
    )

    return merged


def _nonnegative(values, tolerance, label):

    array = np.asarray(values, dtype=float)

    require(np.isfinite(array).all(), f"{label}: NaN/inf")

    bad = np.flatnonzero(array < -tolerance)

    require(
        len(bad) == 0,
        f"{label}: 음수입니다. 위치={bad[:10].tolist()}, "
        f"값={array[bad[:10]].tolist()}"
    )

    return np.maximum(array, 0)


def load_reinj_tables(roots=None, prefix=None):
    """
    mouse 별 region 표.

    region / whole_count / rh_count / lh_count / whole_area /
    rh_area / lh_area
    """

    roots = REINJ_ROOTS if roots is None else roots

    prefix = REINJ_PREFIX if prefix is None else prefix

    tables = {}

    for mouse, root in roots.items():

        directory, tried = reinj_results_directory(root)

        require(
            directory is not None,
            f"{mouse}: results 폴더를 찾지 못했습니다.\n"
            + "\n".join(f"  {path}" for path in tried)
        )

        by_hemi = {}

        for hemi in ("whole", "rh"):

            files = [
                directory / FILE_TEMPLATE.format(
                    prefix=prefix, hemi=hemi, level=level
                )
                for level in LEVELS
            ]

            missing = [str(path) for path in files if not path.is_file()]

            require(
                not missing,
                f"{mouse}: {len(missing)}개 CSV 가 없습니다 "
                f"(prefix = {prefix}).\n"
                + "\n".join(f"  {path}" for path in missing)
            )

            by_hemi[hemi] = merge_levels(files, mouse, hemi)

        whole, rh = by_hemi["whole"], by_hemi["rh"]

        require(
            set(whole.index) == set(rh.index),
            f"{mouse}: whole 과 rh 의 region ID 집합이 다릅니다. "
            "결측을 0 으로 대체하지 않습니다."
        )

        rh = rh.reindex(whole.index)

        require(
            all(
                _normalize(a) == _normalize(b)
                for a, b in zip(whole.region, rh.region)
            ),
            f"{mouse}: whole/rh ID-name 불일치"
        )

        frame = pd.DataFrame(index=whole.index)

        frame["region"] = whole.region

        for metric, tolerance in (
            ("count", COUNT_ATOL), ("area", AREA_ATOL)
        ):

            frame["whole_" + metric] = whole[metric].astype(float)
            frame["rh_" + metric] = rh[metric].astype(float)

            frame["lh_" + metric] = _nonnegative(
                whole[metric].values - rh[metric].values,
                tolerance,
                f"{mouse}: whole - rh {metric}",
            )

        for hemi in ("lh", "rh"):

            require(
                not (
                    (frame[hemi + "_area"] <= AREA_ATOL)
                    & (frame[hemi + "_count"] > COUNT_ATOL)
                ).any(),
                f"{mouse}: {hemi} area=0 인데 count>0"
            )

        frame["results_directory"] = str(directory)

        tables[mouse] = frame

    return tables


# ============================================================
# Figure 1 결과 읽기
# ============================================================

def is_run_dir(path):
    """config.json + RUN_STATUS.json 이 있는 완료된 run 폴더인가."""

    try:
        path = Path(path)
    except TypeError:
        return False

    if not (path / "config.json").is_file():
        return False

    status_path = path / "RUN_STATUS.json"

    if not status_path.is_file():
        return False

    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False

    return status.get("status") == "completed"


def _as_run_dir(value):
    """
    변수에서 얻은 경로를 run 폴더로 정규화합니다.

    run 폴더 자체 / Figure_list 같은 형제 폴더 / 그 안의 파일
    어느 쪽을 받아도 됩니다.
    """

    if value is None:
        return None

    if isinstance(value, dict):

        for key in ("output", "input_run", "run", "svg", "figure"):

            found = _as_run_dir(value.get(key))

            if found is not None:
                return found

        return None

    if isinstance(value, (list, tuple)):

        for item in value:

            found = _as_run_dir(item)

            if found is not None:
                return found

        return None

    if not isinstance(value, (str, Path)):
        return None

    path = Path(value).expanduser()

    if path.is_file():
        path = path.parent

    if not path.exists():
        return None

    path = path.resolve()

    # 자기 자신 또는 부모 중 run 폴더
    for candidate in [path, *path.parents]:

        if is_run_dir(candidate):
            return candidate

    # 형제 폴더 (예: Figure_list 옆의 run_...)
    for base in [path, path.parent]:

        runs = sorted(
            (child for child in base.glob("run_*") if is_run_dir(child)),
            key=lambda p: p.stat().st_mtime,
        )

        if runs:
            return runs[-1]

    return None


REQUIRED_FILES = [
    "config.json",
    "RUN_STATUS.json",
    "input_manifest.csv",
    "raw_readout_regions.csv",
    "off_source_counts.csv",
    "predefined_recipient_regions.csv",
    "source_region_definition.csv",
]


def run_summary(path):
    """run 폴더가 어떤 분석인지 요약합니다."""

    path = Path(path)

    summary = dict(
        path=path,
        name=path.name,
        groups={},
        missing_files=[
            name for name in REQUIRED_FILES
            if not (path / name).is_file()
        ],
    )

    try:
        saved = json.loads(
            (path / "config.json").read_text(encoding="utf-8")
        )
        cohorts = saved.get("cohorts", {})
    except (OSError, ValueError, KeyError):
        return summary

    summary["groups"] = {
        group: {
            source: len(mice)
            for source, mice in sources.items()
        }
        for group, sources in cohorts.items()
        if isinstance(sources, dict)
    }

    return summary


def describe_run(summary):

    groups = summary["groups"]

    if not groups:
        return "cohort 정보 없음"

    return " | ".join(
        group
        + ": "
        + ", ".join(
            f"{SOURCE_LABEL.get(source, source)} n={n}"
            for source, n in sources.items()
        )
        for group, sources in groups.items()
    )


def check_run(path):
    """
    이 분석(AAV-Cre M1 vs reinjection)에 쓸 수 있는 run 폴더인지
    내용으로 확인합니다.

    returns (ok, reason, summary)
    """

    path = Path(path)

    if not is_run_dir(path):
        return False, "완료된 run 폴더가 아님", run_summary(path)

    summary = run_summary(path)

    if summary["missing_files"]:
        return (
            False,
            "파일 없음: " + ", ".join(summary["missing_files"]),
            summary,
        )

    cre = summary["groups"].get("Cre")

    if not cre:
        return (
            False,
            "cohort 에 Cre group 이 없음 "
            f"(있는 group: {list(summary['groups'])})",
            summary,
        )

    missing_sources = [
        source for source in SOURCE_ORDER
        if not cre.get(source)
    ]

    if missing_sources:
        return (
            False,
            "Cre cohort 에 source 가 없음: "
            + ", ".join(
                SOURCE_LABEL.get(s, s) for s in missing_sources
            )
            + f" (있는 source: {list(cre)})",
            summary,
        )

    available = [
        scope
        for scope in SCOPES
        if (
            path / "primary" / scope / MATCHED_SOURCE / "p.csv"
        ).is_file()
    ]

    if not available:
        return (
            False,
            "primary/<scope>/"
            + MATCHED_SOURCE
            + "/p.csv 가 없음 (찾은 scope: "
            + str([
                child.name
                for child in (path / "primary").glob("*")
                if child.is_dir()
            ])
            + ")",
            summary,
        )

    summary["scopes"] = available

    try:

        presets = pd.read_csv(path / "predefined_recipient_regions.csv")

        if not (presets.source == MATCHED_SOURCE).any():
            return (
                False,
                "predefined_recipient_regions.csv 에 "
                f"{MATCHED_SOURCE} recipient region 이 없음",
                summary,
            )

        manifest = pd.read_csv(path / "input_manifest.csv")

        if "injection_side" not in manifest.columns:
            return (
                False,
                "input_manifest.csv 에 injection_side column 이 없음",
                summary,
            )

    except (OSError, ValueError) as error:
        return False, f"파일을 읽지 못함: {error}", summary

    return True, "", summary


_NAMESPACE_KEYS = [
    "RESULTS",
    "OVERVIEW",
    "SCATTER_RESULTS",
    "HELLINGER_RESULTS",
    "SOURCE_SPECIFICITY_FIGURE",
    "RUN_DIR",
    "SCATTER_RUN_DIR",
    "HELLINGER_RUN_DIR",
    "FIG1_RUN_DIR",
]


def _namespaces():
    """노트북 / 호출자 / IPython namespace 를 모두 모읍니다."""

    spaces = [globals()]

    try:
        import __main__
        spaces.append(vars(__main__))
    except Exception:  # noqa: BLE001
        pass

    try:
        from IPython import get_ipython
        shell = get_ipython()
        if shell is not None:
            spaces.append(shell.user_ns)
    except Exception:  # noqa: BLE001
        pass

    try:
        import inspect
        frame = inspect.currentframe()
        for _ in range(12):
            frame = frame.f_back
            if frame is None:
                break
            spaces.append(frame.f_globals)
            spaces.append(frame.f_locals)
    except Exception:  # noqa: BLE001
        pass

    return spaces


def _search_roots():
    """run 폴더를 찾아볼 상위 폴더 후보."""

    roots = []

    for value in SEARCH_ROOTS:
        roots.append(Path(value).expanduser())

    for root in REINJ_ROOTS.values():

        path = Path(root).expanduser()

        roots.append(path)
        roots.append(path.parent)

    roots.append(Path.cwd())
    roots.append(Path.home())

    seen = set()
    unique = []

    for root in roots:

        try:
            resolved = root.resolve()
        except OSError:
            continue

        if resolved in seen or not resolved.is_dir():
            continue

        seen.add(resolved)
        unique.append(resolved)

    return unique


_RUN_PATTERNS = [
    "run_*",
    "SELECT_outputs/run_*",
    "*/SELECT_outputs/run_*",
    "*/*/SELECT_outputs/run_*",
    "*/run_*",
]


def find_run_dirs(verbose=False):
    """완료된 run 폴더를 파일 시스템에서 찾습니다 (최신 순)."""

    found = {}

    for root in _search_roots():

        for pattern in _RUN_PATTERNS:

            try:
                matches = list(root.glob(pattern))
            except OSError:
                continue

            for path in matches:

                if not path.is_dir() or not is_run_dir(path):
                    continue

                resolved = path.resolve()

                found[resolved] = resolved.stat().st_mtime

                if verbose:
                    print(f"  발견: {resolved}")

    return [
        path
        for path, _ in sorted(
            found.items(), key=lambda item: item[1], reverse=True
        )
    ]


def resolve_run_dir(run_dir=None):
    """
    Figure 1 run 폴더를 찾습니다.

    순서
      1. 인자 또는 FIG1_RUN_DIR
      2. 노트북에 남아 있는 기존 코드의 변수
         (RESULTS, OVERVIEW, SCATTER_RESULTS, HELLINGER_RESULTS,
          RUN_DIR, ...)
      3. SELECT_outputs/run_* 폴더 자동 탐색
    """

    if run_dir is None:
        run_dir = FIG1_RUN_DIR

    if run_dir is not None:

        path = Path(run_dir).expanduser().resolve()

        resolved = _as_run_dir(path)

        require(
            resolved is not None,
            f"지정하신 폴더가 완료된 Figure 1 run 폴더가 아닙니다:\n{path}\n"
            "config.json / RUN_STATUS.json 이 있는 run_... 폴더를 "
            "지정해 주십시오."
        )

        ok, reason, summary = check_run(resolved)

        require(
            ok,
            f"지정하신 run 폴더는 이 분석에 쓸 수 없습니다:\n{resolved}\n"
            f"  이유: {reason}\n"
            f"  내용: {describe_run(summary)}"
        )

        print("Figure 1 run 폴더:")
        print(f"  {resolved}")
        print(f"  {describe_run(summary)}")

        return resolved

    # ----------------------------------------------------
    # 2. 노트북 변수
    # ----------------------------------------------------

    rejected = []

    for namespace in _namespaces():

        for key in _NAMESPACE_KEYS:

            if key not in namespace:
                continue

            resolved = _as_run_dir(namespace[key])

            if resolved is None:
                continue

            ok, reason, summary = check_run(resolved)

            if ok:

                print(f"Figure 1 run 폴더를 {key} 에서 찾았습니다:")
                print(f"  {resolved}")
                print(f"  {describe_run(summary)}")

                return resolved

            rejected.append((resolved, f"{key} 변수 / {reason}"))

    # ----------------------------------------------------
    # 3. 파일 시스템 탐색
    #
    # 이름이 아니라 내용(cohort 구성, 필요한 파일)으로 거릅니다.
    # ----------------------------------------------------

    valid = []

    for path in find_run_dirs():

        ok, reason, summary = check_run(path)

        if ok:
            valid.append((path, summary))
        else:
            rejected.append((path, reason))

    if len(valid) == 1:

        path, summary = valid[0]

        print("Figure 1 run 폴더를 자동 탐색으로 찾았습니다:")
        print(f"  {path}")
        print(f"  {describe_run(summary)}")

        if rejected:
            print(
                f"  (조건에 맞지 않아 제외한 run {len(rejected)}개는 "
                "diagnose() 로 확인하실 수 있습니다)"
            )

        return path

    if len(valid) > 1:

        lines = [
            "이 분석에 쓸 수 있는 Figure 1 run 폴더가 여러 개입니다.",
            "어느 것을 쓸지 자동으로 고르지 않겠습니다.",
            "",
            "FIG1_RUN_DIR 에 아래 중 하나를 지정해 주십시오 "
            "(최근 순).",
            "",
        ]

        for path, summary in valid:
            lines.append(f'  FIG1_RUN_DIR = "{path}"')
            lines.append(f"      {describe_run(summary)}")
            lines.append("")

        raise ValueError("\n".join(lines))

    lines = [
        "이 분석에 쓸 수 있는 Figure 1 run 폴더를 찾지 못했습니다.",
        "",
        "필요한 조건",
        "  - RUN_STATUS.json 이 completed",
        "  - " + ", ".join(REQUIRED_FILES),
        "  - cohort 의 Cre group 에 "
        + ", ".join(SOURCE_LABEL[s] for s in SOURCE_ORDER)
        + " 가 모두 존재",
        "",
    ]

    if rejected:

        lines.append("확인했지만 제외된 run")

        for path, reason in rejected[:12]:
            lines.append(f"  {path}")
            lines.append(f"      -> {reason}")

        lines.append("")

    lines += [
        "탐색한 폴더",
        *[f"  {root}" for root in _search_roots()],
        "",
        "FIG1_RUN_DIR 에 run_... 폴더를 직접 지정하시거나, "
        "SEARCH_ROOTS 에 상위 폴더를 추가해 주십시오.",
    ]

    raise ValueError("\n".join(lines))


def load_atlas(cfg):

    path = Path(cfg["atlas_path"]).expanduser()

    if path.suffix.lower() in (".xlsx", ".xls"):
        atlas = pd.read_excel(path, sheet_name=cfg.get("atlas_sheet", 0))
    else:
        atlas = pd.read_csv(path)

    atlas.columns = atlas.columns.str.strip().str.lower()

    require(not atlas.id.duplicated().any(), "Atlas id 중복")

    return atlas


def load_fig1(run_dir=None):

    run = resolve_run_dir(run_dir)

    saved = json.loads((run / "config.json").read_text(encoding="utf-8"))

    status = json.loads(
        (run / "RUN_STATUS.json").read_text(encoding="utf-8")
    )

    require(
        status.get("status") == "completed",
        "완료된 Figure 1 분석 결과 폴더가 아닙니다."
    )

    cohorts = saved["cohorts"]

    cre_mice = [
        m for source in SOURCE_ORDER for m in cohorts["Cre"][source]
    ]

    require(
        len(set(cre_mice)) == len(cre_mice),
        "Figure 1 Cre cohort 에 mouse id 중복이 있습니다."
    )

    if OUTPUT_DIR is not None:
        output = Path(OUTPUT_DIR).expanduser()
    else:
        output = run.parent / "Figure_list_reinjection"

    output.mkdir(parents=True, exist_ok=True)

    return dict(
        run=run,
        output=output,
        cfg=saved["config"],
        cohorts=cohorts,
        cre_mice=cre_mice,
        manifest=pd.read_csv(
            run / "input_manifest.csv"
        ).set_index("mouse", verify_integrity=True),
        raw=pd.read_csv(run / "raw_readout_regions.csv"),
        presets=pd.read_csv(run / "predefined_recipient_regions.csv"),
        source_ids=set(
            pd.read_csv(
                run / "source_region_definition.csv"
            ).id.astype(int)
        ),
        totals=pd.read_csv(
            run / "off_source_counts.csv"
        ).set_index("mouse", verify_integrity=True),
    )


def cre_m1_mice(ctx):

    mice = list(ctx["cohorts"]["Cre"][MATCHED_SOURCE])

    require(len(mice) > 0, f"Cre / {MATCHED_SOURCE} cohort 가 비어 있습니다.")

    return mice


# ============================================================
# FEATURE SPACE
#
# Figure 1 분석이 이미 만들어 둔 feature 정의와 p 를 그대로 읽고,
# 같은 정의를 reinjection data 에 적용합니다.
#
#   primary/<scope>/Motor/feature_definitions.csv
#   primary/<scope>/Motor/p.csv
#   source_specificity/shared_feature_definitions.csv
#   source_specificity/p_shared_feature_space.csv
#
# feature 는 "parent - nearest included descendants" (residual) 이며
# count 와 area 에 같은 선형 연산을 적용합니다.
# ============================================================

def parse_id_list(value):
    """subtract_ids 는 JSON 또는 python list 표기로 저장되어 있습니다."""

    if value is None:
        return []

    if isinstance(value, (list, tuple, np.ndarray)):
        return [int(x) for x in value]

    if isinstance(value, float) and np.isnan(value):
        return []

    text = str(value).strip()

    if text in ("", "nan", "none", "None", "[]"):
        return []

    for parser in (json.loads, ast.literal_eval):

        try:
            parsed = parser(text)
        except (ValueError, SyntaxError):
            continue

        if isinstance(parsed, (list, tuple)):
            return [int(x) for x in parsed]

        return [int(parsed)]

    raise ValueError(f"subtract_ids 를 해석하지 못했습니다: {value!r}")


def _read_feature_definitions(path):

    frame = pd.read_csv(path)

    frame.columns = [str(c).strip() for c in frame.columns]

    if "feature" not in frame.columns:
        frame = frame.rename(columns={frame.columns[0]: "feature"})

    require(
        {"feature", "hemisphere", "region_id"}.issubset(frame.columns),
        f"{path}: feature / hemisphere / region_id column 이 필요합니다.\n"
        f"실제 column: {list(frame.columns)}"
    )

    frame["region_id"] = frame["region_id"].astype(int)

    frame["subtract_ids"] = (
        frame["subtract_ids"].map(parse_id_list)
        if "subtract_ids" in frame.columns
        else [[] for _ in range(len(frame))]
    )

    if "region_label" not in frame.columns:
        frame["region_label"] = frame["region_id"].astype(str)

    frame["label"] = frame["region_label"].astype(str)

    return frame.set_index("feature", verify_integrity=True)


SPECIFICITY_SCOPE = "SourceSpecificity"


# (scope, orientation) -> 해당 region set 에 tdT+ 세포가 0 개인 mouse
NO_SIGNAL = {}

NO_SIGNAL_PRINTED = set()


class Skipped(Exception):
    """필수가 아닌 단계를 건너뛸 때 (실패가 아님)."""


def _space_paths(ctx, scope):

    if scope == SPECIFICITY_SCOPE:

        directory = ctx["run"] / "source_specificity"

        return (
            directory / "shared_feature_definitions.csv",
            directory / "p_shared_feature_space.csv",
        )

    directory = ctx["run"] / "primary" / scope / MATCHED_SOURCE

    return (
        directory / "feature_definitions.csv",
        directory / "p.csv",
    )


def load_feature_space(ctx, scope):
    """Figure 1 의 feature 정의와 p 를 읽습니다."""

    definition_path, p_path = _space_paths(ctx, scope)

    if not (definition_path.is_file() and p_path.is_file()):
        return None

    features = _read_feature_definitions(definition_path)

    p = pd.read_csv(p_path, index_col=0).astype(float)

    p.index = [str(i) for i in p.index]

    shared = [column for column in p.columns if column in features.index]

    require(
        len(shared) > 0,
        f"{p_path}: feature 정의와 p 의 column 이 맞지 않습니다."
    )

    return dict(
        scope=scope,
        features=features.loc[shared],
        p=p.loc[:, shared],
    )


def feature_spaces(ctx, scopes=None):

    scopes = SCOPES if scopes is None else scopes

    spaces = {}

    for scope in scopes:

        space = load_feature_space(ctx, scope)

        if space is None:
            print(
                f"  {scope}: Figure 1 run 에 결과 폴더가 없어 건너뜁니다 "
                f"({_space_paths(ctx, scope)[0]})"
            )
            continue

        spaces[scope] = space

    require(
        len(spaces) > 0,
        "Figure 1 run 에서 사용할 수 있는 feature space 가 없습니다.\n"
        "primary/Predefined/<source>/ 또는 "
        "primary/AllGrayMatter/<source>/ 폴더를 확인하십시오."
    )

    return spaces


# ------------------------------------------------------------
# hemisphere slot
#
#   ipsi   slot  <->  reinjection RH   (ORIENTATION 기본값)
#   contra slot  <->  reinjection LH
# ------------------------------------------------------------

def reinj_side_for(hemisphere, orientation=None):

    orientation = ORIENTATION if orientation is None else orientation

    require(
        orientation in ("rh_to_ipsi", "rh_to_contra"),
        "ORIENTATION 은 rh_to_ipsi 또는 rh_to_contra 여야 합니다."
    )

    first, second = (
        ("rh", "lh") if orientation == "rh_to_ipsi" else ("lh", "rh")
    )

    return first if hemisphere == "ipsi" else second


def apply_features(tables, features, orientation=None):
    """
    Figure 1 의 residual 정의를 reinjection data 에 그대로 적용해
    feature 별 count / area 를 만듭니다.
    """

    mice = list(tables)

    count = pd.DataFrame(
        index=mice, columns=features.index, dtype=float
    )

    area = count.copy()

    needed = set()

    for record in features.itertuples():
        needed.add(int(record.region_id))
        needed.update(int(i) for i in record.subtract_ids)

    for mouse, frame in tables.items():

        missing = sorted(needed - set(frame.index.astype(int)))

        require(
            not missing,
            f"{mouse}: Figure 1 feature 에 쓰인 region id 가 "
            f"reinjection CSV 에 없습니다 ({len(missing)}개): "
            f"{missing[:20]}"
        )

        for metric, output, tolerance in (
            ("count", count, COUNT_ATOL),
            ("area", area, AREA_ATOL),
        ):

            values = []

            for record in features.itertuples():

                column = (
                    reinj_side_for(record.hemisphere, orientation)
                    + "_" + metric
                )

                value = float(frame.at[int(record.region_id), column])

                if record.subtract_ids:
                    value -= float(
                        frame.loc[
                            [int(i) for i in record.subtract_ids], column
                        ].sum()
                    )

                values.append(value)

            output.loc[mouse] = _nonnegative(
                values,
                tolerance,
                f"{mouse}: parent-minus-children {metric}",
            )

    return count, area


def coverage_report(tables, features, count, area, orientation=None):
    """reinjection feature 의 count / area 상태를 사람이 읽을 수 있게 정리."""

    lines = []

    for mouse in count.index:

        c = count.loc[mouse].astype(float)
        a = area.loc[mouse].astype(float)

        lines.append(
            f"  {mouse}: feature {len(c)}개 | area>0 {int((a > AREA_ATOL).sum())} "
            f"| count>0 {int((c > COUNT_ATOL).sum())} "
            f"| count 합 {c.sum():.0f} | area 합 {a.sum():.4g}"
        )

    lines.append("")
    lines.append("  예시 feature (raw 값 -> residual 값)")

    for feature in list(features.index[:6]):

        record = features.loc[feature]

        side = reinj_side_for(record["hemisphere"], orientation)

        rid = int(record["region_id"])

        subtract = [int(i) for i in record["subtract_ids"]]

        lines.append(
            f"    {feature} ({record['label']}), {side}, "
            f"subtract {len(subtract)}개"
        )

        for mouse, frame in tables.items():

            raw_count = float(frame.at[rid, side + "_count"])
            raw_area = float(frame.at[rid, side + "_area"])

            lines.append(
                f"      {mouse}: raw count {raw_count:.0f}, "
                f"raw area {raw_area:.4g} -> residual count "
                f"{float(count.at[mouse, feature]):.0f}, "
                f"area {float(area.at[mouse, feature]):.4g}"
            )

    return "\n".join(lines)


def reinj_p(tables, features, orientation=None):

    count, area = apply_features(tables, features, orientation)

    present = area > AREA_ATOL

    bad = (count > COUNT_ATOL) & ~present

    require(
        not bad.to_numpy().any(),
        "reinjection: area=0 인 feature 에 count>0 이 있습니다.\n"
        + coverage_report(tables, features, count, area, orientation)
    )

    valid = present.all(axis=0)

    dropped = [
        feature for feature in features.index if not valid[feature]
    ]

    kept = [feature for feature in features.index if valid[feature]]

    density = count.loc[:, kept] / area.loc[:, kept]

    totals = density.sum(axis=1)

    structural = len(kept) == 0 or not np.isfinite(totals).all()

    if structural:

        raise ValueError(
            "reinjection p 를 계산할 수 없습니다 (area 문제).\n"
            f"  feature {len(features)}개 중 모든 reinjection mouse 에서 "
            f"area>0 인 것: {len(kept)}개 (제외 {len(dropped)}개)\n"
            + "  mouse 별 density 합: "
            + ", ".join(
                f"{mouse}={value:.4g}" for mouse, value in totals.items()
            )
            + "\n\n"
            + coverage_report(tables, features, count, area, orientation)
        )

    # 이 region set 에 tdT+ 세포가 하나도 없는 mouse 는 p 가 정의되지
    # 않습니다. uniform map 으로 채우지 않고 이 scope 에서만 제외합니다.
    no_signal = [mouse for mouse, value in totals.items() if value <= 0]

    require(
        len(no_signal) < len(totals),
        "이 region set 에서 모든 reinjection mouse 의 tdT+ 세포 수가 "
        "0 입니다.\n"
        + coverage_report(tables, features, count, area, orientation)
    )

    density = density.loc[[m for m in density.index if m not in no_signal]]

    return to_p(density), count, area, dropped, no_signal


def common_feature_space(ctx, tables, scope, orientation=None):
    """
    Figure 1 p 와 reinjection p 를 같은 feature 집합으로 맞춥니다.

    returns (p_cre, p_reinj, features)
    """

    space = (
        scope if isinstance(scope, dict) else load_feature_space(ctx, scope)
    )

    require(space is not None, f"{scope}: feature space 를 읽지 못했습니다.")

    features = space["features"]

    p_fig1 = space["p"]

    p_reinj, count, _, dropped, no_signal = reinj_p(
        tables, features, orientation
    )

    key = (space["scope"], orientation or ORIENTATION)

    for mouse in no_signal:

        if (space["scope"], mouse) in NO_SIGNAL_PRINTED:
            continue

        NO_SIGNAL_PRINTED.add((space["scope"], mouse))

        if True:

            print(
                f"  [{space['scope']}] {mouse}: 이 region set 안에 "
                "tdT+ 세포가 0 개라 p 를 정의할 수 없어, 이 scope 의 "
                "p 기반 분석에서만 제외합니다 "
                f"(region set 전체 count = {float(count.loc[mouse].sum()):.0f})."
            )

    NO_SIGNAL[key] = list(no_signal)

    if dropped:
        print(
            f"  {space['scope']}: reinjection 에서 area=0 인 feature "
            f"{len(dropped)}개 제외"
        )

    shared = [
        feature for feature in features.index
        if feature in set(p_fig1.columns) and feature in set(p_reinj.columns)
    ]

    require(
        len(shared) >= 3,
        f"{space['scope']}: 공통 feature 가 3개 미만입니다."
    )

    return (
        to_p(p_fig1.loc[:, shared]),
        to_p(p_reinj.loc[:, shared]),
        features.loc[shared],
    )


def slot_of(features):
    """figure 에서 쓰는 hemisphere slot 이름 (s1 = ipsi, s2 = contra)."""

    return features["hemisphere"].map(
        lambda h: "s1" if h == "ipsi" else "s2"
    )


# ============================================================
# ATLAS HIERARCHY (permutation 의 major division 분류에만 사용)
# ============================================================

DIVISION_ACRONYMS = [
    "Isocortex", "OLF", "HPF", "CTXsp",
    "STR", "PAL", "TH", "HY", "MB", "P", "MY", "CB",
]


def _hierarchy(atlas):

    parents = {}

    for row in atlas.itertuples():

        value = row.parent_structure_id

        parents[int(row.id)] = (
            None if (pd.isna(value) or value < 0) else int(value)
        )

    return parents


def _ancestors(rid, parents):

    chain = []
    seen = {rid}

    parent = parents.get(rid)

    while parent is not None:

        require(parent not in seen, "Atlas hierarchy 에 순환이 있습니다.")

        seen.add(parent)
        chain.append(parent)

        parent = parents.get(parent)

    return chain


def division_map(ctx, region_ids):
    """region id -> major anatomical division acronym"""

    atlas = load_atlas(ctx["cfg"])

    parents = _hierarchy(atlas)

    acronym_by_id = (
        atlas.set_index("id")["acronym"].astype(str).to_dict()
        if "acronym" in atlas.columns else {}
    )

    division_ids = {
        rid: acronym_by_id.get(rid)
        for rid in atlas.id.astype(int)
        if acronym_by_id.get(rid) in DIVISION_ACRONYMS
    }

    out = {}

    for rid in region_ids:

        rid = int(rid)

        if rid in division_ids:
            out[rid] = division_ids[rid]
            continue

        found = "other"

        for ancestor in _ancestors(rid, parents):

            if ancestor in division_ids:
                found = division_ids[ancestor]
                break

        out[rid] = found

    return out, parents


def predefined_m1_ids(ctx):
    """Figure 1 에서 M1 의 recipient 로 고정한 region id."""

    presets = ctx["presets"]

    ids = (
        presets.loc[presets.source == MATCHED_SOURCE, "id"]
        .astype(int).tolist()
    )

    require(
        len(ids) > 0,
        f"predefined_recipient_regions.csv 에 {MATCHED_SOURCE} "
        "recipient region 이 없습니다."
    )

    return ids


def fig1_table(ctx, scope, name):
    """primary/<scope>/Motor/<name>.csv"""

    path = ctx["run"] / "primary" / scope / MATCHED_SOURCE / f"{name}.csv"

    if not path.is_file():
        return None

    return pd.read_csv(path, index_col=0)


# ============================================================
# FIGURE 1  regional distribution heatmap
# ============================================================

def make_overview(ctx, tables, show=None):

    out = ctx["output"]

    cre_mice = cre_m1_mice(ctx)
    reinj_mice = list(tables)

    written = []
    records = []

    for scope, space in feature_spaces(ctx).items():

        try:
            p_cre, p_reinj, features = common_feature_space(
                ctx, tables, space
            )
        except ValueError as error:
            print(f"  [{scope}] 이 scope 는 건너뜁니다:")
            print("    " + str(error).replace("\n", "\n    "))
            continue

        p_cre = p_cre.loc[[m for m in cre_mice if m in p_cre.index]]

        require(
            len(p_cre) > 0,
            f"{scope}: p.csv 에 Cre {SOURCE_LABEL[MATCHED_SOURCE]} "
            "mouse 가 없습니다."
        )

        missing = [m for m in reinj_mice if m not in p_reinj.index]

        # 신호 없는 mouse 는 NaN 열 (회색) 로 남겨서 개체를 숨기지 않음
        p_reinj = p_reinj.reindex(reinj_mice)

        slots = slot_of(features)

        label_by_region = (
            features.groupby("region_id")["label"].first().to_dict()
        )

        reference = p_cre.mean(axis=0)

        region_total = (
            pd.Series(
                reference.to_numpy(),
                index=features["region_id"].to_numpy(),
            )
            .groupby(level=0).sum()
            .sort_values(ascending=False)
        )

        region_order = region_total.index.tolist()

        def matrix_for(p, mice, slot):

            columns = {
                int(features.at[feature, "region_id"]): feature
                for feature in features.index
                if slots[feature] == slot
            }

            table = pd.DataFrame(
                index=region_order, columns=list(mice) + ["Mean"],
                dtype=float,
            )

            for region in region_order:

                feature = columns.get(region)

                if feature is None:
                    continue

                values = p.loc[mice, feature]

                table.loc[region, mice] = values.to_numpy()
                table.at[region, "Mean"] = float(values.mean(skipna=True))

            return table

        panels = {
            (group, slot): matrix_for(p, mice, slot)
            for group, p, mice in (
                ("Cre", p_cre, list(p_cre.index)),
                ("Reinj", p_reinj, reinj_mice),
            )
            for slot in SLOTS
        }

        vmax = float(np.sqrt(max(
            np.nanmax(table.to_numpy(dtype=float))
            for table in panels.values()
        )))

        for (group, slot), table in panels.items():

            for column in table.columns:

                for region in region_order:

                    value = table.at[region, column]

                    records.append(dict(
                        scope=scope,
                        group=group,
                        hemisphere=SLOT_LABEL[group][slot],
                        column=column,
                        region_id=region,
                        label=label_by_region.get(region, str(region)),
                        p=float(value) if pd.notna(value) else np.nan,
                        sqrt_p=(
                            float(np.sqrt(value))
                            if pd.notna(value) else np.nan
                        ),
                    ))

        # ----------------------------------------------------
        # FIGURE
        # ----------------------------------------------------

        n_region = len(region_order)

        n_cre = len(p_cre) + 1
        n_reinj = len(reinj_mice) + 1

        ratios = [n_cre, n_cre, 4.0, n_reinj, n_reinj, 0.55]

        positions = {
            ("Cre", "s1"): 0,
            ("Cre", "s2"): 1,
            ("Reinj", "s1"): 3,
            ("Reinj", "s2"): 4,
        }

        height = max(4.6, 0.19 * n_region + 2.2)

        with rc():

            fig = plt.figure(figsize=(11.2, height))

            grid = fig.add_gridspec(
                1, len(ratios),
                width_ratios=ratios,
                left=0.03, right=0.945, bottom=0.085, top=0.755,
                wspace=0.07,
            )

            cmap = copy.copy(plt.get_cmap("viridis"))
            cmap.set_bad("#E0E0E0")

            axes = {}

            for (group, slot), position in positions.items():

                table = panels[(group, slot)]

                ax = fig.add_subplot(grid[0, position])

                values = np.sqrt(table.to_numpy(dtype=float))

                image = ax.pcolormesh(
                    np.arange(values.shape[1] + 1) - 0.5,
                    np.arange(values.shape[0] + 1) - 0.5,
                    np.ma.masked_invalid(values),
                    cmap=cmap, vmin=0, vmax=vmax, shading="flat",
                )

                ax.set_ylim(values.shape[0] - 0.5, -0.5)

                ax.set_yticks([])

                ax.set_xticks(range(table.shape[1]))

                ax.set_xticklabels(
                    [
                        "Mean" if column == "Mean" else str(index + 1)
                        for index, column in enumerate(table.columns)
                    ],
                    fontsize=7.5, fontfamily=RENDER_FONT, rotation=90,
                )

                ax.tick_params(axis="both", length=0)

                ax.axvline(table.shape[1] - 1.5, color="white", lw=1.3)

                ax.set_title(
                    SLOT_LABEL[group][slot],
                    fontsize=10, fontfamily=RENDER_FONT,
                    fontstyle="italic", pad=16,
                )

                for spine in ax.spines.values():
                    spine.set_linewidth(0.6)

                axes[(group, slot)] = ax

            for group, color in (
                ("Cre", SOURCE_COLORS[MATCHED_SOURCE]),
                ("Reinj", REINJ_COLOR),
            ):

                group_axes = [
                    ax for (g, _), ax in axes.items() if g == group
                ]

                left = min(ax.get_position().x0 for ax in group_axes)
                right = max(ax.get_position().x1 for ax in group_axes)

                n_mouse = (
                    len(p_cre) if group == "Cre" else len(reinj_mice)
                )

                note = ""

                if group == "Reinj" and missing:
                    note = "\n" + ", ".join(
                        f"Mouse {reinj_mice.index(m) + 1}" for m in missing
                    ) + ": no tdT+ cells in this region set (gray)"

                fig.text(
                    (left + right) / 2, 0.885,
                    f"{GROUP_LABEL[group]}  (n = {n_mouse}){note}",
                    ha="center", va="center",
                    fontsize=12, fontfamily=RENDER_FONT,
                    fontweight="bold", color=color,
                )

            label_ax = fig.add_subplot(grid[0, 2])

            label_ax.set(xlim=(0, 1), ylim=(n_region - 0.5, -0.5))

            label_ax.axis("off")

            fontsize = (
                8.5 if n_region <= 40 else max(3.0, 330 / n_region)
            )

            for index, region in enumerate(region_order):

                label_ax.text(
                    0.5, index,
                    label_by_region.get(region, str(region)),
                    ha="center", va="center",
                    fontsize=fontsize, fontfamily=RENDER_FONT,
                    color="#B96524",
                )

            label_ax.set_title(
                "Recipient\nregion",
                fontsize=9, fontfamily=RENDER_FONT,
                color="#B96524", pad=8,
            )

            cax = fig.add_subplot(grid[0, 5])

            cbar = fig.colorbar(image, cax=cax)

            cbar.set_label("√p", fontsize=10, fontfamily=RENDER_FONT)

            cbar.ax.tick_params(labelsize=8)

            for tick in cbar.ax.get_yticklabels():
                tick.set_fontfamily(RENDER_FONT)

            fig.text(
                0.5, 0.955,
                f"{SCOPE_LABEL.get(scope, scope)}   "
                f"({n_region} regions × 2 hemispheres, "
                f"{len(features)} features)",
                ha="center", va="center",
                fontsize=12, fontfamily=RENDER_FONT, fontweight="bold",
            )

            path = save_svg(
                fig, out / f"Reinj_overview_heatmap_{scope}.svg", show
            )

        written.append(path)

        print(f"  {scope}: {len(features)} features -> {path.name}")

    values = pd.DataFrame(records)

    values.to_csv(
        out / "Reinj_overview_heatmap_values.csv", index=False
    )

    return dict(svg=[str(path) for path in written], values=values)


# ============================================================
# FIGURE 2  Spearman scatter
# ============================================================

def make_scatter(ctx, tables, show=None, panel_inch=2.75):

    out = ctx["output"]

    cre_mice = cre_m1_mice(ctx)
    reinj_mice = list(tables)

    spaces = feature_spaces(ctx)

    scopes = []

    panels = {}
    point_records = []
    metrics = []

    for scope, space in spaces.items():

        try:
            p_cre, p_reinj, features = common_feature_space(
                ctx, tables, space
            )
        except ValueError as error:
            print(f"  [{scope}] 이 scope 는 건너뜁니다:")
            print("    " + str(error).replace("\n", "\n    "))
            continue

        p_cre = p_cre.loc[[m for m in cre_mice if m in p_cre.index]]

        scopes.append(scope)

        slots = slot_of(features)

        info = pd.DataFrame({
            "feature": features.index,
            "hemisphere": features["hemisphere"].to_numpy(),
            "slot": slots.to_numpy(),
            "region_id": features["region_id"].to_numpy(),
            "label": features["label"].to_numpy(),
        })

        reference = p_cre.mean(axis=0)

        targets = [
            (
                mouse, "individual", index,
                p_reinj.loc[mouse] if mouse in p_reinj.index else None,
            )
            for index, mouse in enumerate(reinj_mice)
        ]

        targets.append((
            "Mean", "mean", len(reinj_mice),
            p_reinj.mean(axis=0),
        ))

        for label, comparison, row, target in targets:

            if target is None:

                panels[(scope, row)] = dict(
                    points=None,
                    metric=dict(
                        scope=scope, comparison=comparison,
                        reinj_target=label, n_features=len(features),
                        n_shown=0, spearman_rho=np.nan, hellinger=np.nan,
                        no_signal=True,
                    ),
                )

                metrics.append(panels[(scope, row)]["metric"])

                continue

            a = reference.to_numpy(dtype=float)
            b = target.to_numpy(dtype=float)

            keep = (a > 0) | (b > 0)

            points = info.copy()

            points["scope"] = scope
            points["comparison"] = comparison
            points["reinj_target"] = label
            points["p_Cre_M1_reference"] = a
            points["p_reinjection"] = b
            points["sqrt_p_Cre_M1_reference"] = np.sqrt(a)
            points["sqrt_p_reinjection"] = np.sqrt(b)
            points["included"] = keep

            point_records.append(points)

            metric = dict(
                scope=scope,
                comparison=comparison,
                reinj_target=label,
                n_Cre_mice=len(p_cre),
                n_reinj_mice=(
                    len(p_reinj) if comparison == "mean" else 1
                ),
                n_features=len(features),
                n_shown=int(keep.sum()),
                spearman_rho=spearman_rho(a[keep], b[keep]),
                hellinger=hellinger(a, b),
            )

            for slot in SLOTS:

                mask = (info["slot"] == slot).to_numpy() & keep

                metric[f"spearman_rho_{slot}"] = spearman_rho(
                    a[mask], b[mask]
                )

            metrics.append(metric)

            panels[(scope, row)] = dict(points=points, metric=metric)

    require(
        len(scopes) > 0,
        "모든 scope 를 건너뛰었습니다. 위에 출력된 이유를 확인해 주십시오."
    )

    points_df = pd.concat(point_records, ignore_index=True)

    metrics_df = pd.DataFrame(metrics)

    points_df.to_csv(
        out / "Reinj_vs_CreM1_scatter_points.csv", index=False
    )

    metrics_df.to_csv(
        out / "Reinj_vs_CreM1_scatter_metrics.csv", index=False
    )

    upper = {}

    for scope in scopes:

        subset = points_df.loc[points_df.scope == scope]

        upper[scope] = max(float(subset[[
            "sqrt_p_Cre_M1_reference", "sqrt_p_reinjection"
        ]].to_numpy().max()), 0.05) * 1.06

    n_rows = len(reinj_mice) + 1
    n_cols = len(scopes)

    slot_style = {
        "s1": dict(
            facecolor=REINJ_COLOR, edgecolor="white", linewidth=0.35
        ),
        "s2": dict(
            facecolor="white", edgecolor=REINJ_COLOR, linewidth=0.8
        ),
    }

    with rc():

        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(n_cols * panel_inch + 0.6, n_rows * panel_inch + 0.5),
            squeeze=False,
        )

        fig.subplots_adjust(
            left=0.11, right=0.985, top=0.90, bottom=0.115,
            wspace=0.26, hspace=0.36,
        )

        for col, scope in enumerate(scopes):

            position = axes[0, col].get_position()

            fig.text(
                (position.x0 + position.x1) / 2, 0.955,
                SCOPE_LABEL.get(scope, scope),
                ha="center", va="center",
                fontsize=11, fontfamily=RENDER_FONT, fontweight="bold",
            )

            for row in range(n_rows):

                ax = axes[row, col]

                is_mean = row == n_rows - 1

                panel = panels[(scope, row)]

                data = panel["points"]
                metric = panel["metric"]

                ax.set_facecolor("#F7F7F7" if is_mean else "white")

                limit = upper[scope]

                if data is None:

                    ax.set_xlim(0, 1)
                    ax.set_ylim(0, 1)
                    ax.set_xticks([])
                    ax.set_yticks([])

                    ax.set_aspect("equal", adjustable="box")

                    ax.text(
                        0.5, 0.5,
                        "No tdT+ cells\nin this region set\n(p undefined)",
                        ha="center", va="center",
                        fontsize=9, color="#888888",
                        fontfamily=RENDER_FONT,
                    )

                    for spine in ax.spines.values():
                        spine.set_linewidth(0.65)
                        spine.set_color("#C8C8C8")

                    ax.set_title(
                        f"Mouse {row + 1}",
                        fontsize=9.5, fontfamily=RENDER_FONT, pad=5,
                    )

                    continue

                ax.plot(
                    [0, limit], [0, limit], "--",
                    color="#B5B5B5", lw=0.8, zorder=1,
                )

                for slot in SLOTS:

                    selected = data.loc[
                        data.included & (data.slot == slot)
                    ]

                    style = slot_style[slot]

                    ax.scatter(
                        selected["sqrt_p_Cre_M1_reference"],
                        selected["sqrt_p_reinjection"],
                        s=30 if is_mean else 23,
                        alpha=0.85,
                        facecolors=style["facecolor"],
                        edgecolors=style["edgecolor"],
                        linewidths=style["linewidth"],
                        zorder=3,
                    )

                ax.set_xlim(-limit * 0.025, limit)
                ax.set_ylim(-limit * 0.025, limit)

                ax.set_aspect("equal", adjustable="box")

                ax.xaxis.set_major_locator(
                    MaxNLocator(nbins=4, min_n_ticks=3)
                )

                ax.yaxis.set_major_locator(
                    MaxNLocator(nbins=4, min_n_ticks=3)
                )

                ax.tick_params(labelsize=8, length=3, width=0.6, pad=2)

                for spine in ax.spines.values():
                    spine.set_linewidth(0.65)

                rho = metric["spearman_rho"]

                ax.set_title(
                    ("Mean" if is_mean else f"Mouse {row + 1}")
                    + "\n"
                    + (
                        f"ρ = {rho:.2f}" if np.isfinite(rho) else "ρ = NA"
                    )
                    + f"   H = {metric['hellinger']:.2f}",
                    fontsize=9.5, fontfamily=RENDER_FONT,
                    fontweight="bold" if is_mean else "normal",
                    pad=5,
                )

        fig.text(
            0.5, 0.045,
            "AAV-Cre M1 reference (√p)",
            ha="center", va="center",
            fontsize=11, fontfamily=RENDER_FONT,
        )

        fig.text(
            0.025, 0.5,
            "Intranasal EV reinjection (√p)",
            ha="center", va="center", rotation=90,
            fontsize=11, fontfamily=RENDER_FONT,
        )

        fig.legend(
            handles=[
                Line2D(
                    [0], [0], marker="o", linestyle="",
                    markerfacecolor=REINJ_COLOR, markeredgecolor="white",
                    markersize=6,
                    label="Cre ipsilateral  ·  reinjection RH",
                ),
                Line2D(
                    [0], [0], marker="o", linestyle="",
                    markerfacecolor="white", markeredgecolor=REINJ_COLOR,
                    markeredgewidth=1.0, markersize=6,
                    label="Cre contralateral  ·  reinjection LH",
                ),
            ],
            loc="lower center", bbox_to_anchor=(0.5, 0.0),
            ncol=2, frameon=False, fontsize=8.5,
        )

        path = save_svg(fig, out / "Reinj_vs_CreM1_scatter.svg", show)

    print()
    print(
        metrics_df[[
            "scope", "reinj_target", "n_features", "n_shown",
            "spearman_rho", "hellinger",
        ]].to_string(index=False)
    )

    return dict(svg=str(path), points=points_df, metrics=metrics_df)


# ============================================================
# FIGURE 3  Hellinger distance
# ============================================================

HELLINGER_COLUMNS = [
    ("reinj_to_reference", "EV reinjection →\nCre M1 reference"),
    ("cre_loo", "Cre M1 →\nLOO reference"),
    ("cre_pairwise", "Cre M1 ↔\nCre M1"),
]


def make_hellinger(ctx, tables, show=None):

    out = ctx["output"]

    cre_mice = cre_m1_mice(ctx)
    reinj_mice = list(tables)

    spaces = feature_spaces(ctx)

    scopes = []

    records = []
    summary = []

    other_orientation = (
        "rh_to_contra" if ORIENTATION == "rh_to_ipsi" else "rh_to_ipsi"
    )

    for scope, space in spaces.items():

        try:
            p_cre, p_reinj, features = common_feature_space(
                ctx, tables, space
            )
        except ValueError as error:
            print(f"  [{scope}] 이 scope 는 건너뜁니다:")
            print("    " + str(error).replace("\n", "\n    "))
            continue

        p_cre = p_cre.loc[[m for m in cre_mice if m in p_cre.index]]

        scopes.append(scope)

        present_cre = list(p_cre.index)

        reference = p_cre.mean(axis=0)

        for number, mouse in enumerate(reinj_mice, start=1):

            if mouse not in p_reinj.index:
                continue

            records.append(dict(
                scope=scope,
                comparison="reinj_to_reference",
                label=f"Mouse {number}",
                unit=mouse,
                hellinger=hellinger(p_reinj.loc[mouse], reference),
            ))

        for number, mouse in enumerate(present_cre, start=1):

            others = [m for m in present_cre if m != mouse]

            if not others:
                continue

            records.append(dict(
                scope=scope,
                comparison="cre_loo",
                label=f"Mouse {number}",
                unit=mouse,
                hellinger=hellinger(
                    p_cre.loc[mouse], p_cre.loc[others].mean(axis=0)
                ),
            ))

        for first, second in combinations(present_cre, 2):

            records.append(dict(
                scope=scope,
                comparison="cre_pairwise",
                label=f"{first} / {second}",
                unit=f"{first}|{second}",
                hellinger=hellinger(p_cre.loc[first], p_cre.loc[second]),
            ))

        # hemisphere 대응을 뒤집은 경우 (sensitivity)
        _, p_reinj_flipped, _ = common_feature_space(
            ctx, tables, space, orientation=other_orientation
        )

        flipped_mean = float(np.mean([
            hellinger(p_reinj_flipped.loc[mouse], reference)
            for mouse in p_reinj_flipped.index
        ]))

        table = pd.DataFrame([r for r in records if r["scope"] == scope])

        values = {
            key: table.loc[table.comparison == key, "hellinger"].to_numpy()
            for key, _ in HELLINGER_COLUMNS
        }

        summary.append(dict(
            scope=scope,
            n_features=len(features),
            n_reinj=len(values["reinj_to_reference"]),
            n_Cre=len(present_cre),
            reinj_mean=float(np.mean(values["reinj_to_reference"])),
            reinj_min=float(np.min(values["reinj_to_reference"])),
            reinj_max=float(np.max(values["reinj_to_reference"])),
            cre_loo_mean=float(np.mean(values["cre_loo"])),
            cre_pairwise_mean=float(np.mean(values["cre_pairwise"])),
            cre_pairwise_min=float(np.min(values["cre_pairwise"])),
            cre_pairwise_max=float(np.max(values["cre_pairwise"])),
            Delta_H_reinj_minus_CreLOO=float(
                np.mean(values["reinj_to_reference"])
                - np.mean(values["cre_loo"])
            ),
            n_reinj_within_Cre_pairwise_range=int(np.sum(
                values["reinj_to_reference"]
                <= np.max(values["cre_pairwise"])
            )),
            reinj_mean_flipped_hemisphere=flipped_mean,
            reinj_no_signal_mice=";".join(
                m for m in reinj_mice if m not in p_reinj.index
            ),
        ))

    require(
        len(scopes) > 0,
        "모든 scope 를 건너뛰었습니다. 위에 출력된 이유를 확인해 주십시오."
    )

    distance_df = pd.DataFrame(records)
    summary_df = pd.DataFrame(summary)

    distance_df.to_csv(out / "Reinj_Hellinger_distances.csv", index=False)
    summary_df.to_csv(out / "Reinj_Hellinger_summary.csv", index=False)

    cre_color = SOURCE_COLORS[MATCHED_SOURCE]

    styles = {
        "reinj_to_reference": dict(
            facecolors=REINJ_COLOR, edgecolors="white",
            linewidths=0.55, marker="o", s=49,
        ),
        "cre_loo": dict(
            facecolors="white", edgecolors=cre_color,
            linewidths=1.25, marker="o", s=49,
        ),
        "cre_pairwise": dict(
            facecolors="white", edgecolors=cre_color,
            linewidths=1.0, marker="D", s=36,
        ),
    }

    bar_color = {
        "reinj_to_reference": REINJ_COLOR,
        "cre_loo": cre_color,
        "cre_pairwise": cre_color,
    }

    with rc():

        fig, axes = plt.subplots(
            1, len(scopes),
            figsize=(3.8 * len(scopes) + 0.8, 4.8),
            sharey=True, squeeze=False,
        )

        fig.subplots_adjust(
            left=0.10, right=0.985, top=0.84, bottom=0.27, wspace=0.16,
        )

        for col, scope in enumerate(scopes):

            ax = axes[0, col]

            subset = distance_df.loc[distance_df.scope == scope]

            for index, (key, _) in enumerate(HELLINGER_COLUMNS):

                values = subset.loc[
                    subset.comparison == key, "hellinger"
                ].to_numpy()

                if len(values) == 0:
                    continue

                x = (
                    np.array([float(index)])
                    if len(values) == 1
                    else index + np.linspace(-0.11, 0.11, len(values))
                )

                ax.scatter(x, values, zorder=4, **styles[key])

                ax.hlines(
                    float(np.mean(values)),
                    index - 0.19, index + 0.19,
                    color=bar_color[key], linewidth=2.2, zorder=5,
                )

            row = summary_df.loc[summary_df.scope == scope].iloc[0]

            if np.isfinite(row["reinj_mean_flipped_hemisphere"]):

                ax.hlines(
                    row["reinj_mean_flipped_hemisphere"],
                    -0.19, 0.19,
                    color=REINJ_COLOR, linewidth=1.2,
                    linestyle=":", zorder=5,
                )

            ax.set_xticks(range(len(HELLINGER_COLUMNS)))

            ax.set_xticklabels(
                [label for _, label in HELLINGER_COLUMNS],
                fontsize=8.3, fontfamily=RENDER_FONT,
            )

            ax.set_xlim(-0.45, len(HELLINGER_COLUMNS) - 0.55)
            ax.set_ylim(-0.02, 1.02)
            ax.set_yticks(np.arange(0, 1.01, 0.2))

            ax.tick_params(
                axis="both", labelsize=8, length=3, width=0.6,
                color="#8D959A", pad=2,
            )

            ax.grid(axis="y", color="#ECEEEF", linewidth=0.65, zorder=-1)

            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            for name in ("left", "bottom"):
                ax.spines[name].set_color("#A0A5A8")
                ax.spines[name].set_linewidth(0.7)

            no_signal_note = (
                f", reinjection n = {int(row['n_reinj'])}"
                if int(row["n_reinj"]) < len(reinj_mice) else ""
            )

            ax.set_title(
                SCOPE_LABEL.get(scope, scope)
                + f"\n{int(row['n_features'])} features{no_signal_note}",
                fontsize=11, fontfamily=RENDER_FONT,
                fontweight="bold", pad=8,
            )

        fig.text(
            0.022, 0.58,
            "Hellinger distance",
            rotation=90, ha="center", va="center",
            fontsize=11, fontfamily=RENDER_FONT,
        )

        handles = [
            Line2D(
                [0], [0], marker="o", linestyle="",
                markerfacecolor=REINJ_COLOR, markeredgecolor="white",
                markeredgewidth=0.6, markersize=6.5,
                label="EV reinjection → Cre M1 reference",
            ),
            Line2D(
                [0], [0], marker="o", linestyle="",
                markerfacecolor="white", markeredgecolor=cre_color,
                markeredgewidth=1.2, markersize=6.5,
                label="Cre M1 → leave-one-out reference",
            ),
            Line2D(
                [0], [0], marker="D", linestyle="",
                markerfacecolor="white", markeredgecolor=cre_color,
                markeredgewidth=1.0, markersize=5.8,
                label="Cre M1 pairwise (inter-animal variability)",
            ),
            Line2D([0], [0], color="#666666", linewidth=2.2, label="Mean"),
            Line2D(
                [0], [0], color=REINJ_COLOR, linewidth=1.2,
                linestyle=":",
                label="Mean with RH/LH assignment flipped",
            ),
        ]

        fig.legend(
            handles=handles,
            loc="lower center", bbox_to_anchor=(0.53, 0.0),
            ncol=2, frameon=False, fontsize=8.5,
            handletextpad=0.55, columnspacing=1.5,
        )

        path = save_svg(
            fig, out / "Reinj_Hellinger_reference_comparison.svg", show
        )

    print()
    print(
        summary_df[[
            "scope", "n_features", "reinj_mean", "cre_loo_mean",
            "cre_pairwise_mean", "Delta_H_reinj_minus_CreLOO",
            "n_reinj_within_Cre_pairwise_range",
            "reinj_mean_flipped_hemisphere",
        ]].to_string(index=False)
    )

    return dict(
        svg=str(path), distances=distance_df, summary=summary_df
    )


# ============================================================
# FIGURE 4  matched versus mismatched source
#
# Figure 1 분석의 source_specificity/ 결과와 같은 feature space
# (네 source 의 union 을 공통 제외)를 사용합니다.
# ============================================================

def make_source_specificity(ctx, tables, show=None):

    out = ctx["output"]

    reinj_mice = list(tables)

    space = load_feature_space(ctx, SPECIFICITY_SCOPE)

    require(
        space is not None,
        "source_specificity/ 폴더가 없어 matched-vs-mismatched 분석을 "
        "할 수 없습니다. Figure 1 분석을 "
        "run_source_specificity=True 로 실행하셔야 합니다."
    )

    p_cre, p_reinj, features = common_feature_space(ctx, tables, space)

    references = {}

    for source in SOURCE_ORDER:

        source_mice = [
            mouse for mouse in ctx["cohorts"]["Cre"][source]
            if mouse in p_cre.index
        ]

        require(
            len(source_mice) > 0,
            f"source_specificity p 에 Cre / {source} mouse 가 없습니다."
        )

        references[source] = p_cre.loc[source_mice].mean(axis=0)

    defined = [m for m in reinj_mice if m in p_reinj.index]

    matrix = pd.DataFrame(
        index=defined, columns=SOURCE_ORDER, dtype=float
    )

    for mouse in defined:
        for source in SOURCE_ORDER:
            matrix.at[mouse, source] = hellinger(
                p_reinj.loc[mouse], references[source]
            )

    rows = []

    for mouse in defined:

        number = reinj_mice.index(mouse) + 1

        matched = float(matrix.at[mouse, MATCHED_SOURCE])

        mismatched = {
            source: float(matrix.at[mouse, source])
            for source in SOURCE_ORDER if source != MATCHED_SOURCE
        }

        nearest = min(mismatched, key=mismatched.get)

        assigned = matrix.loc[mouse].idxmin()

        rows.append(dict(
            n_features=len(features),
            mouse=mouse,
            mouse_number=number,
            matched_source=MATCHED_SOURCE,
            H_matched=matched,
            nearest_mismatched_source=nearest,
            H_nearest_mismatched=mismatched[nearest],
            margin=mismatched[nearest] - matched,
            assigned_source=assigned,
            correct=bool(assigned == MATCHED_SOURCE),
            **{
                f"H_{source}": float(matrix.at[mouse, source])
                for source in SOURCE_ORDER
            },
        ))

    assignments = pd.DataFrame(rows)

    assignments.to_csv(
        out / "Reinj_source_specificity_assignments.csv", index=False
    )

    matrix.rename_axis("mouse").reset_index().to_csv(
        out / "Reinj_source_specificity_hellinger.csv", index=False
    )

    with rc():

        fig, axes = plt.subplots(
            1, 2,
            figsize=(9.4, 1.15 * len(reinj_mice) + 3.4),
            gridspec_kw={"width_ratios": [1.25, 1.0]},
            squeeze=False,
        )

        fig.subplots_adjust(
            left=0.14, right=0.95, top=0.82, bottom=0.17, wspace=0.30,
        )

        ax = axes[0, 0]

        image = ax.imshow(
            matrix.to_numpy(dtype=float),
            aspect="auto", cmap="viridis_r",
            vmin=0, vmax=1, interpolation="nearest",
        )

        ax.set_xticks(range(len(SOURCE_ORDER)))

        ax.set_xticklabels(
            [SOURCE_LABEL[s] for s in SOURCE_ORDER], fontsize=9
        )

        for tick, source in zip(ax.get_xticklabels(), SOURCE_ORDER):
            tick.set_color(SOURCE_COLORS[source])
            tick.set_fontweight("bold")

        ax.set_xlabel("Cre reference source", fontsize=10, labelpad=7)

        ax.set_yticks(range(len(matrix)))

        ax.set_yticklabels(
            [f"Mouse {reinj_mice.index(m) + 1}" for m in matrix.index],
            fontsize=9,
        )

        ax.set_ylabel("EV reinjection mice", fontsize=10, labelpad=8)

        matched_column = SOURCE_ORDER.index(MATCHED_SOURCE)

        for i in range(len(matrix)):

            ax.add_patch(Rectangle(
                (matched_column - 0.5, i - 0.5), 1, 1,
                fill=False, edgecolor=MATCH_COLOR, linewidth=2,
            ))

            for j in range(len(SOURCE_ORDER)):

                value = float(matrix.iloc[i, j])

                ax.text(
                    j, i, f"{value:.2f}",
                    ha="center", va="center", fontsize=8,
                    color="white" if value > 0.55 else "black",
                )

        ax.set_title(
            "Distance to Cre source references",
            fontsize=11, fontweight="bold", pad=10,
        )

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        cbar = fig.colorbar(image, ax=ax, shrink=0.8, pad=0.03)

        cbar.set_label("Hellinger distance", fontsize=9, labelpad=7)

        cbar.ax.tick_params(labelsize=8)

        ax_bar = axes[0, 1]

        margins = assignments["margin"].to_numpy(dtype=float)

        y = np.arange(len(margins))

        ax_bar.barh(
            y, margins, height=0.5,
            color=REINJ_COLOR, edgecolor="none", alpha=0.92, zorder=3,
        )

        ax_bar.axvline(0, color="#555555", linewidth=1.0, zorder=4)

        half = max(float(np.max(np.abs(margins))) * 1.20, 0.05)

        ax_bar.set_xlim(-half, half)

        ticks = np.linspace(-half, half, 5)

        ax_bar.set_xticks(ticks)

        ax_bar.set_xticklabels([f"{t:.2f}" for t in ticks], fontsize=8)

        ax_bar.set_yticks(y)
        ax_bar.set_yticklabels([])
        ax_bar.tick_params(axis="y", length=0)
        ax_bar.invert_yaxis()

        ax_bar.set_xlabel(
            "ΔH (nearest mismatched − matched)", fontsize=10, labelpad=7
        )

        ax_bar.set_title(
            "Source-reference specificity\n"
            f"matched nearest in {int(assignments['correct'].sum())} / "
            f"{len(assignments)} mice",
            fontsize=11, fontweight="bold", pad=10,
        )

        ax_bar.grid(axis="x", color="#ECEEEF", linewidth=0.65, zorder=0)

        ax_bar.tick_params(
            axis="x", length=3, width=0.6, color="#8D959A", pad=3
        )

        ax_bar.spines["top"].set_visible(False)
        ax_bar.spines["right"].set_visible(False)
        ax_bar.spines["left"].set_visible(False)
        ax_bar.spines["bottom"].set_color("#A0A5A8")
        ax_bar.spines["bottom"].set_linewidth(0.7)

        fig.text(
            0.5, 0.94,
            "Shared feature space: union of all strict sources excluded\n"
            f"{len(features)} features",
            ha="center", va="center",
            fontsize=10, fontfamily=RENDER_FONT,
        )

        path = save_svg(
            fig, out / "Reinj_source_reference_specificity.svg", show
        )

    print()
    print(
        assignments[[
            "mouse_number", "H_matched", "nearest_mismatched_source",
            "H_nearest_mismatched", "margin", "assigned_source", "correct",
        ]].to_string(index=False)
    )

    return dict(
        svg=str(path), hellinger=matrix, assignments=assignments
    )


# ============================================================
# FIGURE 5  recipient-region enrichment + constrained permutation
#
# AllGrayMatter partition 자체가 서로 겹치지 않는 region 집합이므로
# 그대로 permutation pool 로 사용합니다.
# ============================================================

def _candidates_for(pool, feature, log_tolerance, min_candidates):

    log_area = np.log(pool["area"].to_numpy(dtype=float))

    division = pool["division"].to_numpy()

    position = pool.index.get_loc(feature)

    target_log_area = log_area[position]

    same_division = division == division[position]

    tolerance = log_tolerance

    for _ in range(4):

        mask = same_division & (
            np.abs(log_area - target_log_area) <= tolerance
        )

        if mask.sum() >= min_candidates:
            return np.array(pool.index)[mask], "division+volume"

        tolerance *= 2

    tolerance = log_tolerance

    for _ in range(4):

        mask = np.abs(log_area - target_log_area) <= tolerance

        if mask.sum() >= min_candidates:
            return np.array(pool.index)[mask], "volume_only"

        tolerance *= 2

    return np.array(pool.index), "whole_pool"


def _constrained_sets(
    pool, target_features, n_permutation, rng,
    log_tolerance=np.log(2.0), min_candidates=8,
):

    candidates = {}
    relaxation = {}

    for feature in target_features:

        choices, how = _candidates_for(
            pool, feature, log_tolerance, min_candidates
        )

        candidates[feature] = choices
        relaxation[feature] = how

    order = sorted(target_features, key=lambda f: len(candidates[f]))

    everything = list(pool.index)

    draws = []
    forced = 0

    for _ in range(n_permutation):

        used = set()

        for feature in order:

            available = [
                candidate for candidate in candidates[feature]
                if candidate not in used
            ]

            if not available:

                forced += 1

                available = [
                    candidate for candidate in everything
                    if candidate not in used
                ]

            used.add(str(rng.choice(available)))

        draws.append(sorted(used))

    return draws, dict(
        relaxation=relaxation,
        n_candidates={
            feature: int(len(choices))
            for feature, choices in candidates.items()
        },
        forced_draws=forced,
    )


def make_enrichment(ctx, tables, show=None):

    out = ctx["output"]

    rng = np.random.default_rng(PERMUTATION_SEED)

    space = load_feature_space(ctx, "AllGrayMatter")

    require(
        space is not None,
        "primary/AllGrayMatter/ 결과가 없어 enrichment 분석을 할 수 "
        "없습니다."
    )

    features = space["features"]

    count_reinj, area_reinj = apply_features(tables, features)

    cre_count = fig1_table(ctx, "AllGrayMatter", "count")

    require(
        cre_count is not None,
        "primary/AllGrayMatter/Motor/count.csv 가 없습니다."
    )

    cre_mice = [m for m in cre_m1_mice(ctx) if m in cre_count.index]

    divisions, parents = division_map(
        ctx, features["region_id"].unique().tolist()
    )

    m1_ids = set(predefined_m1_ids(ctx))

    records = []
    nulls = {}
    constraint_rows = []

    for slot, hemisphere in (("s1", "ipsi"), ("s2", "contra")):

        subset = features.loc[features["hemisphere"] == hemisphere]

        if subset.empty:
            continue

        area = area_reinj.loc[:, subset.index].mean(axis=0)

        pool = pd.DataFrame({
            "label": subset["label"].to_numpy(),
            "region_id": subset["region_id"].to_numpy(),
            "division": [
                divisions[int(rid)] for rid in subset["region_id"]
            ],
            "area": area.loc[subset.index].to_numpy(dtype=float),
        }, index=subset.index)

        pool = pool.loc[pool["area"] > AREA_ATOL]

        require(
            len(pool) > 10,
            f"{hemisphere}: permutation pool 이 10개 미만입니다."
        )

        target = [
            feature for feature in pool.index
            if int(pool.at[feature, "region_id"]) in m1_ids
            or (
                set(_ancestors(int(pool.at[feature, "region_id"]), parents))
                & m1_ids
            )
        ]

        require(
            len(target) >= 3,
            f"{hemisphere}: pool 안의 Figure 1 M1 recipient region 이 "
            "3개 미만입니다."
        )

        draws, info = _constrained_sets(
            pool, target, N_PERMUTATION, rng
        )

        for feature in target:

            constraint_rows.append(dict(
                hemisphere=hemisphere,
                feature=feature,
                label=pool.at[feature, "label"],
                division=pool.at[feature, "division"],
                area=float(pool.at[feature, "area"]),
                n_candidates=info["n_candidates"][feature],
                constraint=info["relaxation"][feature],
            ))

        units = [
            ("Reinj", mouse, count_reinj.loc[mouse, pool.index])
            for mouse in tables
        ]

        units += [
            ("Cre", mouse, cre_count.loc[mouse, pool.index])
            for mouse in cre_mice
        ]

        for group, mouse, counts in units:

            counts = counts.astype(float)

            total = float(counts.sum())

            if total <= 0:
                print(
                    f"  경고: {mouse} / {hemisphere} 의 count 합이 0 "
                    "이라 enrichment 를 계산하지 않습니다."
                )
                continue

            observed = float(counts.loc[target].sum() / total)

            null = np.array([
                float(counts.loc[draw].sum() / total) for draw in draws
            ])

            records.append(dict(
                group=group,
                mouse=mouse,
                hemisphere=(
                    SLOT_LABEL["Reinj"][slot]
                    if group == "Reinj"
                    else SLOT_LABEL["Cre"][slot]
                ),
                slot=slot,
                n_pool_features=len(pool),
                n_target_features=len(target),
                total_count=total,
                target_count=float(counts.loc[target].sum()),
                enrichment=observed,
                null_mean=float(null.mean()),
                null_sd=float(null.std(ddof=1)),
                null_p95=float(np.quantile(null, 0.95)),
                enrichment_over_null=(
                    observed / null.mean() if null.mean() > 0 else np.nan
                ),
                permutation_p=float(
                    (1 + np.sum(null >= observed)) / (len(null) + 1)
                ),
                n_permutation=len(null),
            ))

            if group == "Reinj":
                nulls[(mouse, slot)] = null

    enrichment_df = pd.DataFrame(records)

    enrichment_df.to_csv(
        out / "Reinj_recipient_region_enrichment.csv", index=False
    )

    pd.DataFrame(constraint_rows).to_csv(
        out / "Reinj_permutation_constraints.csv", index=False
    )

    reinj_mice = list(tables)

    cre_mean = {
        slot: enrichment_df.loc[
            (enrichment_df.group == "Cre")
            & (enrichment_df.slot == slot), "enrichment"
        ].mean()
        for slot in SLOTS
    }

    with rc():

        fig, axes = plt.subplots(
            len(SLOTS), len(reinj_mice),
            figsize=(3.3 * len(reinj_mice) + 0.4, 2.9 * len(SLOTS) + 1.0),
            squeeze=False,
        )

        fig.subplots_adjust(
            left=0.085, right=0.985, top=0.84, bottom=0.17,
            wspace=0.18, hspace=0.45,
        )

        for row, slot in enumerate(SLOTS):

            for col, mouse in enumerate(reinj_mice):

                ax = axes[row, col]

                if (mouse, slot) not in nulls:
                    ax.set_visible(False)
                    continue

                null = nulls[(mouse, slot)]

                record = enrichment_df.loc[
                    (enrichment_df.mouse == mouse)
                    & (enrichment_df.slot == slot)
                ].iloc[0]

                ax.hist(
                    null, bins=40, color="#D8DCDF",
                    edgecolor="white", linewidth=0.3, zorder=2,
                )

                ax.axvline(
                    record["enrichment"],
                    color=REINJ_COLOR, linewidth=2.0, zorder=4,
                )

                if np.isfinite(cre_mean[slot]):

                    ax.axvline(
                        cre_mean[slot],
                        color=SOURCE_COLORS[MATCHED_SOURCE],
                        linewidth=1.6, linestyle="--", zorder=3,
                    )

                ax.set_title(
                    f"Mouse {col + 1} · {SLOT_LABEL['Reinj'][slot]}\n"
                    f"enrichment = {record['enrichment']:.3f}, "
                    f"p = {record['permutation_p']:.4f}",
                    fontsize=9.5, fontfamily=RENDER_FONT, pad=6,
                )

                ax.tick_params(
                    labelsize=8, length=3, width=0.6,
                    color="#8D959A", pad=2,
                )

                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)

                for name in ("left", "bottom"):
                    ax.spines[name].set_color("#A0A5A8")
                    ax.spines[name].set_linewidth(0.7)

                if col == 0:
                    ax.set_ylabel(
                        "Constrained\npermutations",
                        fontsize=9.5, fontfamily=RENDER_FONT,
                    )

        fig.text(
            0.5, 0.085,
            "Fraction of tdTomato+ cells in Figure 1 M1 recipient regions",
            ha="center", va="center",
            fontsize=10.5, fontfamily=RENDER_FONT,
        )

        fig.text(
            0.5, 0.955,
            "Recipient-region enrichment vs constrained region permutation\n"
            f"{N_PERMUTATION} permutations, "
            "major division and region volume preserved",
            ha="center", va="center",
            fontsize=11, fontfamily=RENDER_FONT, fontweight="bold",
        )

        fig.legend(
            handles=[
                Line2D(
                    [0], [0], color=REINJ_COLOR, linewidth=2.0,
                    label="Observed (EV reinjection)",
                ),
                Line2D(
                    [0], [0], color=SOURCE_COLORS[MATCHED_SOURCE],
                    linewidth=1.6, linestyle="--",
                    label="AAV-Cre M1 mean (matched hemisphere)",
                ),
            ],
            loc="lower center", bbox_to_anchor=(0.5, 0.0),
            ncol=2, frameon=False, fontsize=8.5,
        )

        path = save_svg(
            fig, out / "Reinj_recipient_region_enrichment.svg", show
        )

    print()
    print(
        enrichment_df[[
            "group", "mouse", "hemisphere", "enrichment",
            "null_mean", "enrichment_over_null", "permutation_p",
        ]].to_string(index=False)
    )

    return dict(svg=str(path), enrichment=enrichment_df)


# ============================================================
# FIGURE 6  reference density count model
# ============================================================

def make_reference_model(ctx, tables, show=None):

    out = ctx["output"]

    if sm is None:
        raise Skipped(
            "statsmodels 가 설치되어 있지 않아 count model figure 를 "
            "건너뜁니다. 필요하시면 %pip install statsmodels 후 "
            "run_all(only=['reference_model']) 로 이 그림만 다시 그리시면 됩니다."
        )

    space = load_feature_space(ctx, "AllGrayMatter")

    require(
        space is not None,
        "primary/AllGrayMatter/ 결과가 없어 count model 을 실행할 수 "
        "없습니다."
    )

    features = space["features"]

    p_fig1 = space["p"]

    cre_mice = [m for m in cre_m1_mice(ctx) if m in p_fig1.index]

    reference = p_fig1.loc[cre_mice].mean(axis=0)

    count, area = apply_features(tables, features)

    positive = reference[reference > 0]

    require(
        len(positive) >= 10,
        "Cre M1 reference 에서 0 보다 큰 feature 가 10개 미만입니다."
    )

    floor = float(positive.min()) / 2.0

    rows = []

    for mouse in tables:

        rows.append(pd.DataFrame({
            "mouse": mouse,
            "feature": features.index,
            "hemisphere": features["hemisphere"].to_numpy(),
            "label": features["label"].to_numpy(),
            "count": count.loc[mouse].to_numpy(dtype=float),
            "area": area.loc[mouse].to_numpy(dtype=float),
            "reference_p": reference.reindex(
                features.index
            ).to_numpy(dtype=float),
        }))

    data = pd.concat(rows, ignore_index=True)

    data = data.loc[
        (data["area"] > AREA_ATOL) & data["reference_p"].notna()
    ].copy()

    data["log_reference_p"] = np.log(
        data["reference_p"].clip(lower=floor)
    )

    data["observed_density"] = data["count"] / data["area"]

    design = pd.DataFrame({
        "log_reference_p": data["log_reference_p"].to_numpy(),
    })

    mouse_list = list(tables)

    for mouse in mouse_list[1:]:
        design[f"mouse[{mouse}]"] = (
            data["mouse"] == mouse
        ).astype(float).to_numpy()

    design["hemisphere[contra]"] = (
        data["hemisphere"] == "contra"
    ).astype(float).to_numpy()

    design = sm.add_constant(design, has_constant="add")

    response = np.rint(data["count"].to_numpy(dtype=float))

    require((response >= 0).all(), "count 에 음수가 있습니다.")

    exposure = data["area"].to_numpy(dtype=float)

    model_name = "NegativeBinomial(NB2)"

    try:

        fit = sm.NegativeBinomial(
            response, design, loglike_method="nb2", exposure=exposure,
        ).fit(disp=0, maxiter=200)

        require(
            np.isfinite(fit.params).all() and np.isfinite(fit.bse).all(),
            "NB 추정값이 유한하지 않습니다."
        )

    except Exception as error:  # noqa: BLE001

        print()
        print(f"  NB 추정 실패 -> Poisson 으로 대체합니다: {error}")

        model_name = "Poisson"

        fit = sm.GLM(
            response, design,
            family=sm.families.Poisson(), exposure=exposure,
        ).fit()

    coefficient = float(fit.params["log_reference_p"])
    standard_error = float(fit.bse["log_reference_p"])
    p_value = float(fit.pvalues["log_reference_p"])

    ci_low = coefficient - 1.96 * standard_error
    ci_high = coefficient + 1.96 * standard_error

    per_unit = []

    for mouse in mouse_list:

        for hemisphere in ("ipsi", "contra"):

            subset = data.loc[
                (data.mouse == mouse) & (data.hemisphere == hemisphere)
            ]

            if subset.empty:
                continue

            per_unit.append(dict(
                mouse=mouse,
                reinjection_side=reinj_side_for(hemisphere).upper(),
                cre_hemisphere=hemisphere,
                n_features=len(subset),
                spearman_rho=spearman_rho(
                    subset["reference_p"].to_numpy(),
                    subset["observed_density"].to_numpy(),
                ),
                fraction_nonzero=float((subset["count"] > 0).mean()),
            ))

    per_unit_df = pd.DataFrame(per_unit)

    summary = pd.DataFrame([dict(
        model=model_name,
        n_observations=int(len(data)),
        n_features=int(data.feature.nunique()),
        n_mice=len(mouse_list),
        coefficient_log_reference_p=coefficient,
        standard_error=standard_error,
        ci95_low=ci_low,
        ci95_high=ci_high,
        p_value=p_value,
        interpretation=(
            "coefficient > 0 이면 Figure 1 M1 reference density 가 높은 "
            "region 일수록 reinjection 에서도 recipient cell 이 많음"
        ),
    )])

    data.to_csv(
        out / "Reinj_reference_density_model_data.csv", index=False
    )

    summary.to_csv(
        out / "Reinj_reference_density_model_summary.csv", index=False
    )

    per_unit_df.to_csv(
        out / "Reinj_reference_density_model_per_mouse.csv", index=False
    )

    (out / "Reinj_reference_density_model_fit.txt").write_text(
        str(fit.summary()), encoding="utf-8"
    )

    with rc():

        fig, axes = plt.subplots(
            2, len(mouse_list),
            figsize=(3.3 * len(mouse_list) + 0.6, 6.6),
            sharex=True, sharey=True, squeeze=False,
        )

        fig.subplots_adjust(
            left=0.11, right=0.985, top=0.84, bottom=0.12,
            wspace=0.14, hspace=0.32,
        )

        for row, hemisphere in enumerate(("ipsi", "contra")):

            for col, mouse in enumerate(mouse_list):

                ax = axes[row, col]

                subset = data.loc[
                    (data.mouse == mouse)
                    & (data.hemisphere == hemisphere)
                ]

                if subset.empty:
                    ax.set_visible(False)
                    continue

                x = subset["reference_p"].to_numpy(dtype=float)
                y = subset["observed_density"].to_numpy(dtype=float)

                shown = (x > 0) & (y > 0)

                ax.scatter(
                    x[shown], y[shown],
                    s=15, alpha=0.72, color=REINJ_COLOR,
                    edgecolors="white", linewidths=0.3, zorder=3,
                )

                zero = (x > 0) & (y <= 0)

                if zero.any() and shown.any():

                    ax.scatter(
                        x[zero],
                        np.full(int(zero.sum()), y[shown].min() * 0.5),
                        s=9, marker="v", color="#B9BDC0", zorder=2,
                    )

                ax.set_xscale("log")
                ax.set_yscale("log")

                rho = per_unit_df.loc[
                    (per_unit_df.mouse == mouse)
                    & (per_unit_df.cre_hemisphere == hemisphere),
                    "spearman_rho",
                ].iloc[0]

                side = reinj_side_for(hemisphere).upper()

                ax.set_title(
                    f"Mouse {col + 1} · {side}\n"
                    + (
                        f"ρ = {rho:.2f}"
                        if np.isfinite(rho) else "ρ = NA"
                    ),
                    fontsize=9.5, fontfamily=RENDER_FONT, pad=6,
                )

                ax.tick_params(
                    labelsize=8, length=3, width=0.6,
                    color="#8D959A", pad=2,
                )

                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)

                for name in ("left", "bottom"):
                    ax.spines[name].set_color("#A0A5A8")
                    ax.spines[name].set_linewidth(0.7)

                if col == 0:
                    ax.set_ylabel(
                        f"Reinjection density\n({hemisphere} slot)",
                        fontsize=9.5, fontfamily=RENDER_FONT,
                    )

        fig.text(
            0.5, 0.045,
            "AAV-Cre M1 reference density (p, matched hemisphere)",
            ha="center", va="center",
            fontsize=10.5, fontfamily=RENDER_FONT,
        )

        fig.text(
            0.5, 0.945,
            "Does Figure 1 reference density predict reinjection signal?\n"
            f"{model_name}: β(log reference) = {coefficient:.3f} "
            f"[{ci_low:.3f}, {ci_high:.3f}], p = {p_value:.3g}",
            ha="center", va="center",
            fontsize=10.5, fontfamily=RENDER_FONT, fontweight="bold",
        )

        path = save_svg(
            fig, out / "Reinj_reference_density_model.svg", show
        )

    print()
    print(summary.drop(columns=["interpretation"]).to_string(index=False))
    print()
    print(per_unit_df.to_string(index=False))

    return dict(
        svg=str(path), summary=summary,
        per_mouse=per_unit_df, data=data, fit=fit,
    )


# ============================================================
# FIGURE 7  signal / coverage / laterality QC
#
# negative control 데이터가 없으므로 (a) 는
# "signal > negative control" 검정이 아니라 절대량 비교입니다.
# ============================================================

TOP_N = 10


def make_qc(ctx, tables, show=None):

    out = ctx["output"]

    space = load_feature_space(ctx, "AllGrayMatter")

    require(
        space is not None,
        "primary/AllGrayMatter/ 결과가 없어 QC 를 실행할 수 없습니다."
    )

    features = space["features"]

    ipsi = features.index[features["hemisphere"] == "ipsi"]
    contra = features.index[features["hemisphere"] == "contra"]

    count_reinj, area_reinj = apply_features(tables, features)

    cre_count = fig1_table(ctx, "AllGrayMatter", "count")
    cre_area = fig1_table(ctx, "AllGrayMatter", "area")

    require(
        cre_count is not None and cre_area is not None,
        "primary/AllGrayMatter/Motor/count.csv, area.csv 가 없습니다."
    )

    cre_mice = [m for m in cre_m1_mice(ctx) if m in cre_count.index]

    records = []
    curves = {}

    def add_unit(group, label, counts, areas):

        counts = counts.astype(float)
        areas = areas.astype(float)

        total = float(counts.sum())

        require(total > 0, f"{label}: 분석 feature 전체 count 가 0 입니다.")

        present = areas > AREA_ATOL

        density = np.zeros(len(counts), dtype=float)

        density[present.to_numpy()] = (
            counts[present].to_numpy() / areas[present].to_numpy()
        )

        p = density / density.sum()

        order = np.sort(p)[::-1]

        curves[(group, label)] = np.cumsum(order)

        records.append(dict(
            group=group,
            label=label,
            n_features=len(counts),
            total_count=total,
            n_features_with_signal=int((counts > 0).sum()),
            coverage=float((counts > 0).mean()),
            top_feature_share=float(order[0]),
            top_n_share=float(order[:TOP_N].sum()),
            laterality_index=float(
                counts.loc[[c for c in ipsi if c in counts.index]].sum()
                / total
            ),
        ))

    for mouse in tables:

        add_unit(
            "Reinj", mouse,
            count_reinj.loc[mouse], area_reinj.loc[mouse],
        )

    for mouse in cre_mice:

        add_unit(
            "Cre", mouse,
            cre_count.loc[mouse, features.index],
            cre_area.loc[mouse, features.index],
        )

    qc = pd.DataFrame(records)

    qc.to_csv(out / "Reinj_QC_signal_and_coverage.csv", index=False)

    color_of = {
        "Cre": SOURCE_COLORS[MATCHED_SOURCE],
        "Reinj": REINJ_COLOR,
    }

    groups = ["Cre", "Reinj"]

    with rc():

        fig, axes = plt.subplots(1, 4, figsize=(14.0, 3.6), squeeze=False)

        fig.subplots_adjust(
            left=0.055, right=0.99, top=0.78, bottom=0.20, wspace=0.34
        )

        def strip(ax, column, hline=None, log=False, ylabel=""):

            for index, group in enumerate(groups):

                values = qc.loc[qc.group == group, column].to_numpy(float)

                if len(values) == 0:
                    continue

                x = (
                    np.array([float(index)])
                    if len(values) == 1
                    else index + np.linspace(-0.11, 0.11, len(values))
                )

                ax.scatter(
                    x, values, s=46,
                    facecolors=color_of[group], edgecolors="white",
                    linewidths=0.55, zorder=4,
                )

                ax.hlines(
                    float(np.mean(values)),
                    index - 0.2, index + 0.2,
                    color=color_of[group], linewidth=2.2, zorder=5,
                )

            if hline is not None:
                ax.axhline(
                    hline, color="#999999", linewidth=0.9,
                    linestyle="--", zorder=1,
                )

            if log:
                ax.set_yscale("log")

            ax.set_xticks(range(len(groups)))

            ax.set_xticklabels(
                [
                    f"AAV-Cre M1\n(n = {int((qc.group == 'Cre').sum())})",
                    "EV reinjection\n"
                    f"(n = {int((qc.group == 'Reinj').sum())})",
                ],
                fontsize=8.3, fontfamily=RENDER_FONT,
            )

            ax.set_xlim(-0.45, len(groups) - 0.55)

            ax.set_ylabel(ylabel, fontsize=9.5, fontfamily=RENDER_FONT)

            ax.grid(axis="y", color="#ECEEEF", linewidth=0.65, zorder=0)

            ax.tick_params(
                labelsize=8, length=3, width=0.6,
                color="#8D959A", pad=2,
            )

            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            for name in ("left", "bottom"):
                ax.spines[name].set_color("#A0A5A8")
                ax.spines[name].set_linewidth(0.7)

        strip(
            axes[0, 0], "total_count", log=True,
            ylabel="tdTomato+ cells in analyzed features",
        )

        axes[0, 0].set_title(
            "Absolute signal", fontsize=10.5, fontweight="bold", pad=7
        )

        strip(
            axes[0, 1], "coverage",
            ylabel="Fraction of features with ≥ 1 cell",
        )

        axes[0, 1].set_ylim(-0.03, 1.03)

        axes[0, 1].set_title(
            "Spatial coverage", fontsize=10.5, fontweight="bold", pad=7
        )

        ax = axes[0, 2]

        for (group, _), curve in curves.items():

            ax.plot(
                np.arange(1, len(curve) + 1), curve,
                color=color_of[group], linewidth=1.2,
                alpha=0.85, zorder=3,
            )

        ax.set_xscale("log")
        ax.set_ylim(0, 1.03)

        ax.axhline(
            0.5, color="#999999", linewidth=0.9,
            linestyle="--", zorder=1,
        )

        ax.set_xlabel(
            "Number of features (ranked)",
            fontsize=9.5, fontfamily=RENDER_FONT,
        )

        ax.set_ylabel(
            "Cumulative share of p", fontsize=9.5, fontfamily=RENDER_FONT
        )

        ax.set_title(
            "Distribution concentration",
            fontsize=10.5, fontweight="bold", pad=7,
        )

        ax.grid(color="#ECEEEF", linewidth=0.65, zorder=0)

        ax.tick_params(
            labelsize=8, length=3, width=0.6, color="#8D959A", pad=2
        )

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        for name in ("left", "bottom"):
            ax.spines[name].set_color("#A0A5A8")
            ax.spines[name].set_linewidth(0.7)

        strip(
            axes[0, 3], "laterality_index", hline=0.5,
            ylabel="Cre: ipsi / total   ·   Reinj: "
                   f"{reinj_side_for('ipsi').upper()} / total",
        )

        axes[0, 3].set_ylim(0, 1.03)

        axes[0, 3].set_title(
            "Laterality", fontsize=10.5, fontweight="bold", pad=7
        )

        fig.text(
            0.5, 0.955,
            "Signal, coverage and laterality QC   "
            f"({len(features)} all-gray-matter features)",
            ha="center", va="center",
            fontsize=11.5, fontfamily=RENDER_FONT, fontweight="bold",
        )

        path = save_svg(
            fig, out / "Reinj_QC_signal_coverage_laterality.svg", show
        )

    print()
    print(
        qc[[
            "group", "label", "total_count", "coverage",
            "top_n_share", "laterality_index",
        ]].to_string(index=False)
    )

    return dict(svg=str(path), qc=qc)


def diagnose(run_dir=None):
    """
    입력 상태만 점검하고 출력합니다. figure 는 만들지 않습니다.

    오류가 날 때 이 출력을 그대로 보내 주시면 원인을 바로 알 수 있습니다.
    """

    print("==========================================")
    print("DIAGNOSE")
    print("==========================================")

    # ----------------------------------------------------
    # 1. 후보 run 폴더
    # ----------------------------------------------------

    print()
    print("[1] 찾은 run 폴더")

    try:

        found = find_run_dirs()

        if not found:
            print("  없음")

        for path in found:

            ok, reason, summary = check_run(path)

            print(f"  {'[사용 가능]' if ok else '[제외]    '} {path}")
            print(f"      {describe_run(summary)}")

            if not ok:
                print(f"      -> {reason}")

    except Exception:  # noqa: BLE001
        traceback.print_exc()

    # ----------------------------------------------------
    # 2. 선택된 run 폴더
    # ----------------------------------------------------

    print()
    print("[2] 사용할 run 폴더")

    run = None

    try:
        run = resolve_run_dir(run_dir)
    except Exception as error:  # noqa: BLE001
        print(f"  확정 실패: {error}")

    if run is not None:

        for name in REQUIRED_FILES:

            path = run / name

            if not path.is_file():
                print(f"  없음: {name}")
                continue

            if name.endswith(".json"):
                print(f"  {name}: 있음")
                continue

            try:
                frame = pd.read_csv(path, nrows=5)
                print(
                    f"  {name}: {list(frame.columns)}"
                )
            except Exception as error:  # noqa: BLE001
                print(f"  {name}: 읽기 실패 {error}")

        try:

            saved = json.loads(
                (run / "config.json").read_text(encoding="utf-8")
            )

            atlas_path = Path(saved["config"]["atlas_path"]).expanduser()

            print(
                f"  atlas_path: {atlas_path} "
                f"({'있음' if atlas_path.exists() else '없음'})"
            )

            print(
                f"  Cre {SOURCE_LABEL[MATCHED_SOURCE]} mice: "
                f"{saved['cohorts']['Cre'][MATCHED_SOURCE]}"
            )

        except Exception as error:  # noqa: BLE001
            print(f"  config.json 확인 실패: {error}")

        ctx_run = dict(run=run)

        agm = (
            run / "primary" / "AllGrayMatter" / MATCHED_SOURCE
            / "feature_definitions.csv"
        )

        if agm.is_file():
            try:
                frame = pd.read_csv(agm, nrows=5)
                print(f"  feature_definitions.csv: {list(frame.columns)}")
            except Exception as error:  # noqa: BLE001
                print(f"  feature_definitions.csv 읽기 실패: {error}")
        else:
            print(f"  없음: {agm}")

        for scope in SCOPES + [SPECIFICITY_SCOPE]:

            definition_path, p_path = _space_paths(ctx_run, scope)

            if not (definition_path.is_file() and p_path.is_file()):
                print(f"  feature space {scope}: 없음 ({definition_path})")
                continue

            try:

                frame = pd.read_csv(p_path, index_col=0)

                print(
                    f"  feature space {scope}: {frame.shape[1]} features, "
                    f"mice {list(frame.index)}"
                )

            except Exception as error:  # noqa: BLE001
                print(f"  feature space {scope} 읽기 실패: {error}")

    # ----------------------------------------------------
    # 3. reinjection 파일
    # ----------------------------------------------------

    print()
    print("[3] reinjection 파일")

    for mouse, root in REINJ_ROOTS.items():

        print(f"  {mouse}: {root}")

        directory, tried = reinj_results_directory(root)

        if directory is None:
            print("    results 폴더 없음. 확인한 경로:")
            for path in tried:
                print(f"      {path}")
            continue

        print(f"    results 폴더: {directory}")

        for hemi in ("whole", "rh"):

            files = [
                directory / FILE_TEMPLATE.format(
                    prefix=REINJ_PREFIX, hemi=hemi, level=level
                )
                for level in LEVELS
            ]

            missing = [
                level for level, path in zip(LEVELS, files)
                if not path.is_file()
            ]

            if missing:

                print(f"    {hemi}: lvl {missing} 파일 없음")

                others = sorted(
                    {
                        path.name.split("_" + hemi + "_")[0]
                        for path in directory.glob("*_" + hemi + "_lvl*.csv")
                    }
                )

                if others:
                    print(f"      이 폴더에 있는 prefix: {others}")

            if files[0].is_file():

                try:

                    frame = pd.read_csv(files[0], encoding="utf-8-sig")

                    print(
                        f"    {hemi} lvl{LEVELS[0]}: "
                        f"{len(frame)} rows, {list(frame.columns)}"
                    )

                except Exception as error:  # noqa: BLE001
                    print(f"    {hemi} 읽기 실패: {error}")

    # ----------------------------------------------------
    # 4. 실제 로딩
    # ----------------------------------------------------

    print()
    print("[4] 실제 로딩")

    ctx = None
    tables = None

    try:
        ctx = load_fig1(run)
        print(f"  Figure 1: OK, Cre mice {len(ctx['cre_mice'])}")
        print(f"  출력 폴더: {ctx['output']}")
    except Exception:  # noqa: BLE001
        print("  Figure 1 로딩 실패:")
        traceback.print_exc()

    try:
        tables = load_reinj_tables()
        for mouse, frame in tables.items():
            print(
                f"  {mouse}: {len(frame)} regions, "
                f"whole count {frame['whole_count'].sum():.0f} "
                f"(rh {frame['rh_count'].sum():.0f} / "
                f"lh {frame['lh_count'].sum():.0f})"
            )
    except Exception:  # noqa: BLE001
        print("  reinjection 로딩 실패:")
        traceback.print_exc()

    if ctx is not None and tables is not None:

        reinj_ids = set.intersection(*[
            set(table.index.astype(int)) for table in tables.values()
        ])

        for scope in SCOPES + [SPECIFICITY_SCOPE]:

            try:

                space = load_feature_space(ctx, scope)

                if space is None:
                    continue

                features = space["features"]

                needed = set(features["region_id"].astype(int))

                for ids in features["subtract_ids"]:
                    needed.update(int(i) for i in ids)

                missing = sorted(needed - reinj_ids)

                print(
                    f"  {scope}: {len(features)} features, "
                    f"region id {len(needed)}개 중 "
                    f"reinjection 에 없는 것 {len(missing)}개"
                )

                if missing:
                    print(f"      {missing[:20]}")

                else:

                    p_cre, p_reinj, shared = common_feature_space(
                        ctx, tables, space
                    )

                    print(
                        f"      공통 feature {len(shared)}개, "
                        f"Figure 1 mice {len(p_cre)}, "
                        f"reinjection mice {len(p_reinj)}"
                    )

            except Exception:  # noqa: BLE001
                print(f"  {scope} 확인 실패:")
                traceback.print_exc()

    print()
    print("==========================================")

    return dict(run=run, ctx=ctx, tables=tables)


STEPS = [
    ("overview", make_overview),
    ("scatter", make_scatter),
    ("hellinger", make_hellinger),
    ("source_specificity", make_source_specificity),
    ("enrichment", make_enrichment),
    ("reference_model", make_reference_model),
    ("qc", make_qc),
]


def run_all(only=None, stop_on_error=False):

    ctx = load_fig1()

    tables = load_reinj_tables()

    print("==========================================")
    print("INPUT")
    print("==========================================")
    print("Figure 1 run:")
    print(ctx["run"])
    print()
    print("출력 폴더:")
    print(ctx["output"])
    print()
    print(
        f"AAV-Cre {SOURCE_LABEL[MATCHED_SOURCE]} mice "
        f"(n = {len(cre_m1_mice(ctx))}):"
    )
    for mouse in cre_m1_mice(ctx):
        print(f"  {mouse}")
    print()
    print(f"Reinjection mice (n = {len(tables)}):")
    for mouse, frame in tables.items():
        print(f"  {mouse}: {len(frame)} regions")
    print()
    print(f"Hemisphere: {HEMI_MODE} / orientation = {ORIENTATION}")

    results = {}
    errors = {}

    for name, function in STEPS:

        if only is not None and name not in only:
            continue

        print()
        print("##########################################")
        print(f"# {name}")
        print("##########################################")

        try:
            results[name] = function(ctx, tables)
        except Skipped as reason:
            print(f"  SKIPPED: {reason}")
            results[name] = "skipped"
            continue
        except Exception as error:  # noqa: BLE001
            if stop_on_error:
                raise
            print()
            print(f"[{name}] 실패:")
            traceback.print_exc()
            results[name] = None
            errors[name] = error

    print()
    print("==========================================")
    print("DONE")
    print("==========================================")
    print(ctx["output"])
    print()

    for name, value in results.items():
        state = (
            "SKIPPED" if isinstance(value, str) and value == "skipped"
            else "ok" if value is not None
            else "FAILED"
        )
        print(f"{name}: {state}")

    no_signal_rows = [
        dict(scope=scope, orientation=orientation, mouse=mouse)
        for (scope, orientation), mice in NO_SIGNAL.items()
        for mouse in mice
    ]

    if no_signal_rows:

        frame = pd.DataFrame(no_signal_rows).drop_duplicates(
            ["scope", "mouse"]
        )

        frame.to_csv(
            ctx["output"] / "Reinj_no_signal_mice.csv", index=False
        )

        print()
        print("region set 안에 tdT+ 세포가 0 개라 p 기반 분석에서 제외된 경우")

        for row in frame.itertuples():
            print(f"  {row.scope}: {row.mouse}")

    if errors:

        print()
        print("실패 요약")

        for name, error in errors.items():

            first_line = str(error).strip().splitlines()

            print(
                f"  {name}: {type(error).__name__}: "
                + (first_line[0] if first_line else "")
            )

        print()
        print(
            "원인을 확인하시려면 diagnose() 를 실행하고 그 출력을 "
            "보내 주십시오."
        )

    return dict(
        ctx=ctx, tables=tables, results=results, errors=errors
    )


if __name__ == "__main__":
    REINJ = run_all()
