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

import statsmodels.api as sm


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
# 2. Figure 1 분석 결과 폴더
#
# 같은 notebook 에서 기존 분석 직후면 None 그대로 두세요.
# RESULTS["output"] 을 자동으로 사용합니다.
# ------------------------------------------------------------

FIG1_RUN_DIR = None


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

SCOPES = [
    "PredefinedM1",     # Figure 1 에서 M1 의 recipient 로 정의된 region
    "Predefined20",     # Figure 1 predefined recipient region 20개 전체
    "AllGrayMatter",    # Figure 1 broad region set
]


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

RESULTS_SUBDIR = "source/results"

RH_FILE_TEMPLATE = "tdt_total_cell_count_rh_lvl{level}.csv"

WHOLE_FILE_TEMPLATE = "tdt_total_cell_count_whole_lvl{level}.csv"


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
    "PredefinedM1": "M1 predefined recipient regions",
    "Predefined20": "Predefined 20 regions",
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
        "모든 feature 의 density 합이 0 인 mouse 가 있습니다."
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
# ============================================================

def _read_level_file(path, side):

    require(path.is_file(), f"파일을 찾을 수 없습니다:\n{path}")

    frame = pd.read_csv(path)

    frame.columns = [str(col).strip() for col in frame.columns]

    where = f"파일: {path}"

    id_col = detect_column(frame, "id", "region id", where)
    count_col = detect_column(frame, "count", "cell count", where, side)
    area_col = detect_column(frame, "area", "region area", where, side)

    acronym_col = detect_column(
        frame, "acronym", "acronym", where, required=False
    )

    name_col = detect_column(
        frame, "name", "region name", where, required=False
    )

    out = pd.DataFrame({
        "id": pd.to_numeric(frame[id_col], errors="coerce"),
        "count": pd.to_numeric(frame[count_col], errors="coerce"),
        "area": pd.to_numeric(frame[area_col], errors="coerce"),
    })

    out["acronym"] = (
        frame[acronym_col].astype(str) if acronym_col is not None else ""
    )

    out["name"] = (
        frame[name_col].astype(str) if name_col is not None else ""
    )

    out = out.loc[out["id"].notna()].copy()

    out["id"] = out["id"].astype(int)

    require(
        out["count"].notna().all() and out["area"].notna().all(),
        f"{where}\ncount 또는 area 에 숫자가 아닌 값이 있습니다."
    )

    return out


def _combine_levels(root, template, side, label):

    results = Path(root).expanduser() / RESULTS_SUBDIR

    frames = []

    for level in LEVELS:

        frame = _read_level_file(
            results / template.format(level=level), side
        )

        frame["level"] = level

        frames.append(frame)

    merged = pd.concat(frames, ignore_index=True)

    duplicated = merged.loc[merged.duplicated("id", keep=False)]

    if len(duplicated):

        spread = duplicated.groupby("id").agg(
            count_spread=("count", lambda s: float(np.ptp(s))),
            area_spread=("area", lambda s: float(np.ptp(s))),
        )

        bad = spread.loc[
            (spread.count_spread > COUNT_ATOL)
            | (spread.area_spread > AREA_ATOL)
        ]

        require(
            bad.empty,
            f"{label}: 여러 level 파일에서 같은 region id 의 "
            f"count/area 가 다릅니다.\n{bad.head(20)}"
        )

    return (
        merged
        .sort_values(["id", "level"])
        .drop_duplicates("id", keep="first")
        .set_index("id", verify_integrity=True)
        .sort_index()
    )


def load_reinj_tables(roots=None):
    """
    mouse 별
    rh_count / rh_area / lh_count / lh_area /
    whole_count / whole_area
    """

    roots = REINJ_ROOTS if roots is None else roots

    tables = {}

    for mouse, root in roots.items():

        rh = _combine_levels(root, RH_FILE_TEMPLATE, "rh", f"{mouse} rh")

        whole = _combine_levels(
            root, WHOLE_FILE_TEMPLATE, "whole", f"{mouse} whole"
        )

        shared = rh.index.intersection(whole.index)

        require(
            len(shared) > 0,
            f"{mouse}: rh 와 whole 파일에 공통 region id 가 없습니다."
        )

        frame = pd.DataFrame(index=shared.sort_values())

        frame["acronym"] = whole.loc[frame.index, "acronym"]
        frame["name"] = whole.loc[frame.index, "name"]

        frame["rh_count"] = rh.loc[frame.index, "count"].astype(float)
        frame["rh_area"] = rh.loc[frame.index, "area"].astype(float)

        frame["whole_count"] = whole.loc[frame.index, "count"].astype(float)
        frame["whole_area"] = whole.loc[frame.index, "area"].astype(float)

        frame["lh_count"] = frame["whole_count"] - frame["rh_count"]
        frame["lh_area"] = frame["whole_area"] - frame["rh_area"]

        require(
            np.isfinite(frame[[
                "rh_count", "rh_area", "whole_count", "whole_area"
            ]].to_numpy()).all(),
            f"{mouse}: count/area 에 NaN 또는 inf 가 있습니다."
        )

        require(
            (frame[["rh_count", "whole_count"]].to_numpy() >= 0).all()
            and (frame[["rh_area", "whole_area"]].to_numpy() >= 0).all(),
            f"{mouse}: count/area 에 음수가 있습니다."
        )

        bad = frame.loc[frame["lh_count"] < -NEGATIVE_COUNT_TOL]

        require(
            bad.empty,
            f"{mouse}: rh_count > whole_count 인 region 이 있습니다.\n"
            f"{bad[['acronym', 'rh_count', 'whole_count']].head(20)}"
        )

        bad = frame.loc[
            frame["lh_area"] < -frame["whole_area"].abs() * 1e-9 - AREA_ATOL
        ]

        require(
            bad.empty,
            f"{mouse}: rh_area > whole_area 인 region 이 있습니다.\n"
            f"{bad[['acronym', 'rh_area', 'whole_area']].head(20)}"
        )

        frame["lh_count"] = frame["lh_count"].clip(lower=0.0)
        frame["lh_area"] = frame["lh_area"].clip(lower=0.0)

        tables[mouse] = frame

    return tables


# ============================================================
# Figure 1 결과 읽기
# ============================================================

def resolve_run_dir(run_dir=None):

    if run_dir is None:
        run_dir = FIG1_RUN_DIR

    if run_dir is None:
        results = globals().get("RESULTS")
        if isinstance(results, dict):
            run_dir = results.get("output")

    require(
        run_dir is not None,
        "Figure 1 분석 폴더를 찾을 수 없습니다.\n"
        "FIG1_RUN_DIR 를 지정하거나 RESULTS['output'] 이 있는 "
        "상태에서 실행하십시오."
    )

    return Path(run_dir).expanduser().resolve()


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
# region set
# ============================================================

def _allgraymatter_ids(ctx, source):

    path = (
        ctx["run"] / "primary" / "AllGrayMatter" / source
        / "feature_definitions.csv"
    )

    if not path.is_file():
        return None

    meta = pd.read_csv(path)

    meta.columns = [str(col).strip() for col in meta.columns]

    where = f"파일: {path}"

    id_col = detect_column(meta, "region_id", "region id", where,
                           required=False)

    if id_col is None:
        id_col = detect_column(meta, "id", "region id", where)

    residual_like = [
        col
        for col in meta.columns
        if any(
            key in _normalize(col)
            for key in ("residual", "subtract", "minus", "excludedchildren")
        )
    ]

    for col in residual_like:

        values = meta[col]

        has_content = (
            values.fillna(False).astype(bool).any()
            if values.dtype == bool
            else values.astype(str).str.strip().replace(
                {"": None, "nan": None, "None": None, "[]": None}
            ).notna().any()
        )

        require(
            not has_content,
            f"{where}\n'{col}' 에 residual(parent - children) 정의가 "
            "있습니다. 같은 정의를 reinjection data 에도 적용해야 하므로 "
            "AllGrayMatter scope 를 쓰기 전에 형식 확인이 필요합니다."
        )

    hemi_col = detect_column(
        meta, "hemisphere", "hemisphere", where, required=False
    )

    subset = meta

    if hemi_col is not None:

        hemisphere = meta[hemi_col].astype(str).str.strip().str.lower()

        if hemisphere.isin(["ipsi", "rh", "right"]).any():
            subset = meta.loc[hemisphere.isin(["ipsi", "rh", "right"])]

    ids = sorted(set(
        pd.to_numeric(subset[id_col], errors="coerce")
        .dropna().astype(int).tolist()
    ))

    return ids or None


def region_sets(ctx):

    presets = ctx["presets"]

    atlas = load_atlas(ctx["cfg"])

    acronym_by_id = (
        atlas.set_index("id")["acronym"].astype(str).to_dict()
        if "acronym" in atlas.columns else {}
    )

    name_by_id = (
        atlas.set_index("id")["name"].astype(str).to_dict()
        if "name" in atlas.columns else {}
    )

    sets = {}

    m1 = presets.loc[presets.source == MATCHED_SOURCE]

    require(
        not m1.empty,
        f"predefined_recipient_regions.csv 에 source == "
        f"{MATCHED_SOURCE} 인 region 이 없습니다."
    )

    sets["PredefinedM1"] = m1.drop_duplicates("id").copy()

    sets["Predefined20"] = pd.concat(
        [presets.loc[presets.source == s] for s in SOURCE_ORDER],
        ignore_index=True,
    ).drop_duplicates("id").copy()

    agm_ids = _allgraymatter_ids(ctx, MATCHED_SOURCE)

    if agm_ids:

        sets["AllGrayMatter"] = pd.DataFrame({
            "id": agm_ids,
            "acronym": [acronym_by_id.get(i, str(i)) for i in agm_ids],
            "name": [name_by_id.get(i, str(i)) for i in agm_ids],
            "source": "AllGrayMatter",
        })

    # ----------------------------------------------------
    # source region 제외
    #
    # source-specificity 에서 reference 마다 다른 region 을 빼면
    # distance 를 직접 비교할 수 없으므로 네 source 의 union 을
    # 모든 scope 에서 동일하게 제외합니다.
    # ----------------------------------------------------

    cleaned = {}

    for scope, frame in sets.items():

        frame = frame.copy()

        frame["id"] = frame["id"].astype(int)

        kept = frame.loc[~frame["id"].isin(ctx["source_ids"])]

        require(
            not kept.empty,
            f"{scope}: source region 제외 후 남는 region 이 없습니다."
        )

        cleaned[scope] = kept.drop_duplicates("id").reset_index(drop=True)

    return {
        scope: cleaned[scope] for scope in SCOPES if scope in cleaned
    }


def acronym_map(frame):
    return {
        int(row.id): str(row.acronym)
        for row in frame.itertuples(index=False)
    }


# ============================================================
# density -> p
#
# lateralized: columns = MultiIndex (slot, region_id)
# pooled     : columns = region_id
# ============================================================

def _density(count, area):

    count = np.asarray(count, dtype=float)
    area = np.asarray(area, dtype=float)

    present = area > AREA_ATOL

    require(
        not ((count > COUNT_ATOL) & ~present).any(),
        "area = 0 인데 count > 0 인 region 이 있습니다."
    )

    out = np.full(len(count), np.nan, dtype=float)

    out[present] = count[present] / area[present]

    return out


def _finalize(density, label):

    valid = density.notna()

    inconsistent = valid.columns[
        valid.any(axis=0) != valid.all(axis=0)
    ].tolist()

    require(
        not inconsistent,
        f"{label}: 일부 mouse 에서만 area = 0 인 feature 가 있습니다: "
        f"{inconsistent[:20]}"
    )

    kept = valid.columns[valid.all(axis=0)]

    require(len(kept) > 0, f"{label}: 유효한 feature 가 없습니다.")

    return density.loc[:, kept].astype(float)


def fig1_density(ctx, ids, mode=None):

    mode = HEMI_MODE if mode is None else mode

    ids = [int(i) for i in ids]

    raw = ctx["raw"]

    frame = raw.loc[
        raw.mouse.isin(ctx["cre_mice"]) & raw.id.isin(ids)
    ].set_index(["mouse", "id"], verify_integrity=True)

    if mode == "pooled":
        columns = pd.Index(ids, name="region_id")
    else:
        columns = pd.MultiIndex.from_product(
            [SLOTS, ids], names=["slot", "region_id"]
        )

    density = pd.DataFrame(
        index=ctx["cre_mice"], columns=columns, dtype=float
    )

    for mouse in ctx["cre_mice"]:

        side = str(
            ctx["manifest"].at[mouse, "injection_side"]
        ).strip().upper()

        require(side in ("R", "L"), f"{mouse}: injection_side 가 R/L 이 아닙니다.")

        sub = frame.xs(mouse).reindex(ids)

        require(
            sub[["rh_count", "rh_area", "lh_count", "lh_area"]]
            .notna().to_numpy().all(),
            f"{mouse}: raw_readout_regions.csv 에 없는 region 이 있습니다:\n"
            f"{sub.index[sub.rh_count.isna()].tolist()[:20]}"
        )

        ipsi = "rh" if side == "R" else "lh"
        contra = "lh" if side == "R" else "rh"

        if mode == "pooled":

            density.loc[mouse] = _density(
                sub[f"{ipsi}_count"] + sub[f"{contra}_count"],
                sub[f"{ipsi}_area"] + sub[f"{contra}_area"],
            )

        else:

            density.loc[mouse, "s1"] = _density(
                sub[f"{ipsi}_count"], sub[f"{ipsi}_area"]
            )

            density.loc[mouse, "s2"] = _density(
                sub[f"{contra}_count"], sub[f"{contra}_area"]
            )

    return _finalize(density, "Figure 1 Cre")


def reinj_density(tables, ids, mode=None, orientation=None):

    mode = HEMI_MODE if mode is None else mode

    orientation = ORIENTATION if orientation is None else orientation

    require(
        orientation in ("rh_to_ipsi", "rh_to_contra"),
        "ORIENTATION 은 rh_to_ipsi 또는 rh_to_contra 여야 합니다."
    )

    ids = [int(i) for i in ids]

    if mode == "pooled":
        columns = pd.Index(ids, name="region_id")
    else:
        columns = pd.MultiIndex.from_product(
            [SLOTS, ids], names=["slot", "region_id"]
        )

    density = pd.DataFrame(
        index=list(tables), columns=columns, dtype=float
    )

    first, second = (
        ("rh", "lh") if orientation == "rh_to_ipsi" else ("lh", "rh")
    )

    for mouse, frame in tables.items():

        sub = frame.reindex(ids)

        missing = sub.index[sub["whole_count"].isna()].tolist()

        require(
            not missing,
            f"{mouse}: lvl1~7 파일에 없는 region id 가 있습니다: "
            f"{missing[:20]}"
        )

        if mode == "pooled":

            density.loc[mouse] = _density(
                sub["whole_count"], sub["whole_area"]
            )

        else:

            density.loc[mouse, "s1"] = _density(
                sub[f"{first}_count"], sub[f"{first}_area"]
            )

            density.loc[mouse, "s2"] = _density(
                sub[f"{second}_count"], sub[f"{second}_area"]
            )

    return _finalize(density, "Reinjection")


def common_feature_space(
    ctx, tables, ids, mode=None, orientation=None
):
    """
    두 dataset 에서 모두 유효한 feature 만 남기고
    같은 순서로 p 를 반환합니다.
    """

    mode = HEMI_MODE if mode is None else mode

    cre = fig1_density(ctx, ids, mode=mode)

    reinj = reinj_density(
        tables, ids, mode=mode, orientation=orientation
    )

    shared = [
        column for column in cre.columns if column in set(reinj.columns)
    ]

    require(
        len(shared) >= 3,
        "두 dataset 에서 공통으로 유효한 feature 가 3개 미만입니다."
    )

    return (
        to_p(cre.loc[:, shared]),
        to_p(reinj.loc[:, shared]),
        shared,
    )


def feature_frame(shared, acronyms, mode=None):
    """shared feature list -> slot / region_id / acronym 표"""

    mode = HEMI_MODE if mode is None else mode

    if mode == "pooled":

        return pd.DataFrame({
            "slot": "pooled",
            "region_id": [int(c) for c in shared],
            "acronym": [acronyms.get(int(c), str(c)) for c in shared],
        })

    return pd.DataFrame({
        "slot": [c[0] for c in shared],
        "region_id": [int(c[1]) for c in shared],
        "acronym": [acronyms.get(int(c[1]), str(c[1])) for c in shared],
    })


# ============================================================
# atlas hierarchy / broad region pool
# ============================================================

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


DIVISION_ACRONYMS = [
    "Isocortex", "OLF", "HPF", "CTXsp",
    "STR", "PAL", "TH", "HY", "MB", "P", "MY", "CB",
]


def leaf_pool(ctx, tables):
    """
    두 dataset 에 모두 있는 region 중, 자신의 descendant 가 pool 에
    함께 들어있지 않은 region 만 남겨 서로 겹치지 않는 broad pool 을
    만듭니다 (분석 전략 문서 2번).
    """

    atlas = load_atlas(ctx["cfg"])

    parents = _hierarchy(atlas)

    fig1_ids = set(ctx["raw"].id.astype(int))

    reinj_ids = set.intersection(*[
        set(frame.index.astype(int)) for frame in tables.values()
    ])

    available = (fig1_ids & reinj_ids) - set(ctx["source_ids"])

    available = {
        rid for rid in available
        if not (set(_ancestors(rid, parents)) & set(ctx["source_ids"]))
    }

    require(
        len(available) > 20,
        "broad region pool 이 20개 미만입니다. Figure 1 "
        "raw_readout_regions.csv 와 reinjection lvl 파일의 region id 를 "
        "확인하십시오."
    )

    has_descendant = set()

    for rid in available:
        for ancestor in _ancestors(rid, parents):
            if ancestor in available:
                has_descendant.add(ancestor)

    pool_ids = sorted(available - has_descendant)

    acronym_by_id = (
        atlas.set_index("id")["acronym"].astype(str).to_dict()
        if "acronym" in atlas.columns else {}
    )

    division_ids = {
        rid: acronym_by_id.get(rid)
        for rid in atlas.id.astype(int)
        if acronym_by_id.get(rid) in DIVISION_ACRONYMS
    }

    def division_of(rid):

        if rid in division_ids:
            return division_ids[rid]

        for ancestor in _ancestors(rid, parents):
            if ancestor in division_ids:
                return division_ids[ancestor]

        return "other"

    area = [
        float(np.mean([
            frame.at[rid, "whole_area"] for frame in tables.values()
        ]))
        for rid in pool_ids
    ]

    frame = pd.DataFrame({
        "id": pool_ids,
        "acronym": [acronym_by_id.get(rid, str(rid)) for rid in pool_ids],
        "division": [division_of(rid) for rid in pool_ids],
        "area": area,
    }).set_index("id")

    return frame, parents


# ============================================================
# FIGURE 1  regional distribution heatmap
# ============================================================

def make_overview(ctx, tables, show=None):

    out = ctx["output"]

    cre_mice = cre_m1_mice(ctx)
    reinj_mice = list(tables)

    lateralized = HEMI_MODE != "pooled"

    written = []
    records = []

    for scope, frame in region_sets(ctx).items():

        p_cre, p_reinj, shared = common_feature_space(
            ctx, tables, frame["id"].tolist()
        )

        p_cre = p_cre.loc[cre_mice]
        p_reinj = p_reinj.loc[reinj_mice]

        acronyms = acronym_map(frame)

        info = feature_frame(shared, acronyms)

        # region 순서: Cre reference 에서 양쪽 hemisphere 합이 큰 순
        reference = p_cre.mean(axis=0)

        if lateralized:

            by_region = (
                pd.Series(
                    reference.to_numpy(),
                    index=info["region_id"].to_numpy(),
                )
                .groupby(level=0).sum()
                .sort_values(ascending=False)
            )

            region_order = by_region.index.tolist()

        else:

            region_order = (
                reference.sort_values(ascending=False).index.tolist()
            )

        label_by_id = dict(zip(info["region_id"], info["acronym"]))

        slots = SLOTS if lateralized else ["pooled"]

        def matrix_for(p, mice, slot):

            if lateralized:
                sub = p.xs(slot, level="slot", axis=1)
            else:
                sub = p

            available = [
                rid for rid in region_order if rid in set(sub.columns)
            ]

            table = sub.loc[mice, available].T

            table["Mean"] = sub.loc[mice, available].mean(axis=0)

            return table.reindex(region_order)

        panels = {
            (group, slot): matrix_for(p, mice, slot)
            for group, p, mice in (
                ("Cre", p_cre, cre_mice),
                ("Reinj", p_reinj, reinj_mice),
            )
            for slot in slots
        }

        vmax = float(np.sqrt(max(
            np.nanmax(table.to_numpy(dtype=float))
            for table in panels.values()
        )))

        for (group, slot), table in panels.items():

            for column in table.columns:

                for rid in region_order:

                    value = table.at[rid, column]

                    records.append(dict(
                        scope=scope,
                        group=group,
                        hemisphere=SLOT_LABEL.get(
                            group, {}
                        ).get(slot, slot),
                        column=column,
                        region_id=rid,
                        acronym=label_by_id.get(rid, str(rid)),
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

        n_cre = len(cre_mice) + 1
        n_reinj = len(reinj_mice) + 1

        if lateralized:
            ratios = [n_cre, n_cre, 3.4, n_reinj, n_reinj, 0.55]
            positions = {
                ("Cre", "s1"): 0,
                ("Cre", "s2"): 1,
                ("Reinj", "s1"): 3,
                ("Reinj", "s2"): 4,
            }
            label_position = 2
            colorbar_position = 5
            width = 10.6
        else:
            ratios = [n_cre, 3.4, n_reinj, 0.55]
            positions = {("Cre", "pooled"): 0, ("Reinj", "pooled"): 2}
            label_position = 1
            colorbar_position = 3
            width = 8.6

        height = max(4.6, 0.19 * n_region + 2.2)

        with rc():

            fig = plt.figure(figsize=(width, height))

            grid = fig.add_gridspec(
                1, len(ratios),
                width_ratios=ratios,
                left=0.03, right=0.945, bottom=0.085, top=0.78,
                wspace=0.07,
            )

            cmap = plt.get_cmap("viridis").copy()
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
                    fontsize=7.5,
                    fontfamily=RENDER_FONT,
                    rotation=90,
                )

                ax.tick_params(axis="both", length=0)

                ax.axvline(table.shape[1] - 1.5, color="white", lw=1.3)

                ax.set_title(
                    SLOT_LABEL.get(group, {}).get(slot, ""),
                    fontsize=10,
                    fontfamily=RENDER_FONT,
                    fontstyle="italic",
                    pad=24,
                )

                for spine in ax.spines.values():
                    spine.set_linewidth(0.6)

                axes[(group, slot)] = ax

            # group header
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
                    len(cre_mice) if group == "Cre" else len(reinj_mice)
                )

                fig.text(
                    (left + right) / 2, 0.875,
                    f"{GROUP_LABEL[group]}  (n = {n_mouse})",
                    ha="center", va="center",
                    fontsize=12,
                    fontfamily=RENDER_FONT,
                    fontweight="bold",
                    color=color,
                )

            # region label
            label_ax = fig.add_subplot(grid[0, label_position])

            label_ax.set(xlim=(0, 1), ylim=(n_region - 0.5, -0.5))

            label_ax.axis("off")

            fontsize = (
                8.5 if n_region <= 40 else max(3.2, 330 / n_region)
            )

            for index, rid in enumerate(region_order):

                label_ax.text(
                    0.5, index,
                    label_by_id.get(rid, str(rid)),
                    ha="center", va="center",
                    fontsize=fontsize,
                    fontfamily=RENDER_FONT,
                    color="#B96524",
                )

            label_ax.set_title(
                "Recipient\nregion",
                fontsize=9,
                fontfamily=RENDER_FONT,
                color="#B96524",
                pad=8,
            )

            cax = fig.add_subplot(grid[0, colorbar_position])

            cbar = fig.colorbar(image, cax=cax)

            cbar.set_label("√p", fontsize=10, fontfamily=RENDER_FONT)

            cbar.ax.tick_params(labelsize=8)

            for tick in cbar.ax.get_yticklabels():
                tick.set_fontfamily(RENDER_FONT)

            fig.text(
                0.5, 0.965,
                f"{SCOPE_LABEL[scope]}   "
                f"({n_region} regions × "
                f"{'2 hemispheres' if lateralized else 'bilateral'})",
                ha="center", va="center",
                fontsize=12,
                fontfamily=RENDER_FONT,
                fontweight="bold",
            )

            path = save_svg(
                fig, out / f"Reinj_overview_heatmap_{scope}.svg", show
            )

        written.append(path)

        print(f"  {scope}: {n_region} regions -> {path.name}")

    values = pd.DataFrame(records)

    values.to_csv(
        out / "Reinj_overview_heatmap_values.csv", index=False
    )

    return dict(svg=[str(p) for p in written], values=values)


# ============================================================
# FIGURE 2  Spearman scatter
# ============================================================

def make_scatter(ctx, tables, show=None, panel_inch=2.75):

    out = ctx["output"]

    cre_mice = cre_m1_mice(ctx)
    reinj_mice = list(tables)

    lateralized = HEMI_MODE != "pooled"

    sets = region_sets(ctx)

    scopes = list(sets)

    panels = {}
    point_records = []
    metrics = []

    for scope, frame in sets.items():

        p_cre, p_reinj, shared = common_feature_space(
            ctx, tables, frame["id"].tolist()
        )

        info = feature_frame(shared, acronym_map(frame))

        reference = p_cre.loc[cre_mice].mean(axis=0)

        targets = [
            (mouse, "individual", index, p_reinj.loc[mouse])
            for index, mouse in enumerate(reinj_mice)
        ]

        targets.append((
            "Mean", "mean", len(reinj_mice),
            p_reinj.loc[reinj_mice].mean(axis=0),
        ))

        for label, comparison, row, target in targets:

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
                n_Cre_mice=len(cre_mice),
                n_reinj_mice=(
                    len(reinj_mice) if comparison == "mean" else 1
                ),
                n_features=len(shared),
                n_shown=int(keep.sum()),
                spearman_rho=spearman_rho(a[keep], b[keep]),
                hellinger=hellinger(a, b),
            )

            for slot in (SLOTS if lateralized else []):

                mask = (info["slot"] == slot).to_numpy() & keep

                metric[f"spearman_rho_{slot}"] = spearman_rho(
                    a[mask], b[mask]
                )

            metrics.append(metric)

            panels[(scope, row)] = dict(points=points, metric=metric)

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
        "pooled": dict(
            facecolor=REINJ_COLOR, edgecolor="white", linewidth=0.35
        ),
    }

    with rc():

        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(n_cols * panel_inch, n_rows * panel_inch + 0.5),
            squeeze=False,
        )

        fig.subplots_adjust(
            left=0.09, right=0.985, top=0.90, bottom=0.115,
            wspace=0.26, hspace=0.36,
        )

        for col, scope in enumerate(scopes):

            position = axes[0, col].get_position()

            fig.text(
                (position.x0 + position.x1) / 2, 0.955,
                SCOPE_LABEL[scope],
                ha="center", va="center",
                fontsize=11,
                fontfamily=RENDER_FONT,
                fontweight="bold",
            )

            for row in range(n_rows):

                ax = axes[row, col]

                is_mean = row == n_rows - 1

                panel = panels[(scope, row)]

                data = panel["points"]
                metric = panel["metric"]

                ax.set_facecolor("#F7F7F7" if is_mean else "white")

                limit = upper[scope]

                ax.plot(
                    [0, limit], [0, limit], "--",
                    color="#B5B5B5", lw=0.8, zorder=1,
                )

                for slot in (SLOTS if lateralized else ["pooled"]):

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
                    fontsize=9.5,
                    fontfamily=RENDER_FONT,
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
            0.022, 0.5,
            "Intranasal EV reinjection (√p)",
            ha="center", va="center", rotation=90,
            fontsize=11, fontfamily=RENDER_FONT,
        )

        if lateralized:

            fig.legend(
                handles=[
                    Line2D(
                        [0], [0], marker="o", linestyle="",
                        markerfacecolor=REINJ_COLOR,
                        markeredgecolor="white",
                        markersize=6,
                        label="Cre ipsilateral  ·  reinjection RH",
                    ),
                    Line2D(
                        [0], [0], marker="o", linestyle="",
                        markerfacecolor="white",
                        markeredgecolor=REINJ_COLOR,
                        markeredgewidth=1.0,
                        markersize=6,
                        label="Cre contralateral  ·  reinjection LH",
                    ),
                ],
                loc="lower center",
                bbox_to_anchor=(0.5, 0.0),
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

    sets = region_sets(ctx)

    scopes = list(sets)

    records = []
    summary = []

    other_orientation = (
        "rh_to_contra" if ORIENTATION == "rh_to_ipsi" else "rh_to_ipsi"
    )

    for scope, frame in sets.items():

        ids = frame["id"].tolist()

        p_cre, p_reinj, shared = common_feature_space(ctx, tables, ids)

        p_cre = p_cre.loc[cre_mice]

        reference = p_cre.mean(axis=0)

        for number, mouse in enumerate(reinj_mice, start=1):

            records.append(dict(
                scope=scope,
                comparison="reinj_to_reference",
                label=f"Mouse {number}",
                unit=mouse,
                hellinger=hellinger(p_reinj.loc[mouse], reference),
            ))

        for number, mouse in enumerate(cre_mice, start=1):

            others = [m for m in cre_mice if m != mouse]

            records.append(dict(
                scope=scope,
                comparison="cre_loo",
                label=f"Mouse {number}",
                unit=mouse,
                hellinger=hellinger(
                    p_cre.loc[mouse], p_cre.loc[others].mean(axis=0)
                ),
            ))

        for first, second in combinations(cre_mice, 2):

            records.append(dict(
                scope=scope,
                comparison="cre_pairwise",
                label=f"{first} / {second}",
                unit=f"{first}|{second}",
                hellinger=hellinger(p_cre.loc[first], p_cre.loc[second]),
            ))

        # ----------------------------------------------------
        # hemisphere 대응을 뒤집은 경우 (sensitivity)
        # ----------------------------------------------------

        flipped_mean = np.nan

        if HEMI_MODE != "pooled":

            p_cre_f, p_reinj_f, _ = common_feature_space(
                ctx, tables, ids, orientation=other_orientation
            )

            reference_f = p_cre_f.loc[cre_mice].mean(axis=0)

            flipped_mean = float(np.mean([
                hellinger(p_reinj_f.loc[mouse], reference_f)
                for mouse in reinj_mice
            ]))

        table = pd.DataFrame([r for r in records if r["scope"] == scope])

        values = {
            key: table.loc[table.comparison == key, "hellinger"].to_numpy()
            for key, _ in HELLINGER_COLUMNS
        }

        summary.append(dict(
            scope=scope,
            n_features=len(shared),
            n_reinj=len(values["reinj_to_reference"]),
            n_Cre=len(cre_mice),
            reinj_mean=float(np.mean(values["reinj_to_reference"])),
            reinj_min=float(np.min(values["reinj_to_reference"])),
            reinj_max=float(np.max(values["reinj_to_reference"])),
            cre_loo_mean=float(np.mean(values["cre_loo"])),
            cre_pairwise_mean=float(np.mean(values["cre_pairwise"])),
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
        ))

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
            figsize=(3.6 * len(scopes) + 0.6, 4.8),
            sharey=True, squeeze=False,
        )

        fig.subplots_adjust(
            left=0.085, right=0.985, top=0.84, bottom=0.27, wspace=0.16,
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

            ax.set_title(
                SCOPE_LABEL[scope] + f"\n{int(row['n_features'])} features",
                fontsize=11, fontfamily=RENDER_FONT,
                fontweight="bold", pad=8,
            )

        fig.text(
            0.018, 0.58,
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
        ]

        if HEMI_MODE != "pooled":

            handles.append(Line2D(
                [0], [0], color=REINJ_COLOR, linewidth=1.2,
                linestyle=":",
                label="Mean with RH/LH assignment flipped",
            ))

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
# FIGURE 4  matched vs mismatched source
#
# PredefinedM1 은 region 자체가 M1 에서 정의되어 circular 하므로
# 이 분석에서는 제외합니다.
# ============================================================

SPECIFICITY_EXCLUDED_SCOPES = ("PredefinedM1",)


def make_source_specificity(ctx, tables, show=None):

    out = ctx["output"]

    reinj_mice = list(tables)

    sets = {
        scope: frame
        for scope, frame in region_sets(ctx).items()
        if scope not in SPECIFICITY_EXCLUDED_SCOPES
    }

    require(len(sets) > 0, "source-specificity 에 사용할 scope 가 없습니다.")

    matrices = {}
    rows = []

    for scope, frame in sets.items():

        p_cre, p_reinj, shared = common_feature_space(
            ctx, tables, frame["id"].tolist()
        )

        references = {}

        for source in SOURCE_ORDER:

            source_mice = list(ctx["cohorts"]["Cre"][source])

            require(
                len(source_mice) > 0,
                f"Cre / {source} cohort 가 비어 있습니다."
            )

            references[source] = p_cre.loc[source_mice].mean(axis=0)

        matrix = pd.DataFrame(
            index=reinj_mice, columns=SOURCE_ORDER, dtype=float
        )

        for mouse in reinj_mice:
            for source in SOURCE_ORDER:
                matrix.at[mouse, source] = hellinger(
                    p_reinj.loc[mouse], references[source]
                )

        matrices[scope] = matrix

        for number, mouse in enumerate(reinj_mice, start=1):

            matched = float(matrix.at[mouse, MATCHED_SOURCE])

            mismatched = {
                source: float(matrix.at[mouse, source])
                for source in SOURCE_ORDER if source != MATCHED_SOURCE
            }

            nearest = min(mismatched, key=mismatched.get)

            assigned = matrix.loc[mouse].idxmin()

            rows.append(dict(
                scope=scope,
                n_features=len(shared),
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

    pd.concat(
        [
            matrix.assign(scope=scope).reset_index(names="mouse")
            for scope, matrix in matrices.items()
        ],
        ignore_index=True,
    ).to_csv(
        out / "Reinj_source_specificity_hellinger.csv", index=False
    )

    scopes = list(sets)

    with rc():

        fig, axes = plt.subplots(
            len(scopes), 2,
            figsize=(
                9.2,
                len(scopes) * (1.15 * len(reinj_mice) + 1.85),
            ),
            gridspec_kw={"width_ratios": [1.25, 1.0]},
            squeeze=False,
        )

        fig.subplots_adjust(
            left=0.14, right=0.95, top=0.88, bottom=0.12,
            wspace=0.30, hspace=0.45,
        )

        for row, scope in enumerate(scopes):

            matrix = matrices[scope]

            ax = axes[row, 0]

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
                [f"Mouse {i + 1}" for i in range(len(matrix))], fontsize=9
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
                "Distance to Cre source references\n" + SCOPE_LABEL[scope],
                fontsize=11, fontweight="bold", pad=10,
            )

            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            cbar = fig.colorbar(image, ax=ax, shrink=0.8, pad=0.03)

            cbar.set_label("Hellinger distance", fontsize=9, labelpad=7)

            cbar.ax.tick_params(labelsize=8)

            ax_bar = axes[row, 1]

            subset = assignments.loc[
                assignments.scope == scope
            ].set_index("mouse").loc[matrix.index]

            margins = subset["margin"].to_numpy(dtype=float)

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

            ax_bar.set_xticklabels(
                [f"{t:.2f}" for t in ticks], fontsize=8
            )

            ax_bar.set_yticks(y)
            ax_bar.set_yticklabels([])
            ax_bar.tick_params(axis="y", length=0)
            ax_bar.invert_yaxis()

            ax_bar.set_xlabel(
                "ΔH (nearest mismatched − matched)",
                fontsize=10, labelpad=7,
            )

            ax_bar.set_title(
                "Source-reference specificity\n"
                f"matched nearest in {int(subset['correct'].sum())} / "
                f"{len(subset)} mice",
                fontsize=11, fontweight="bold", pad=10,
            )

            ax_bar.grid(
                axis="x", color="#ECEEEF", linewidth=0.65, zorder=0
            )

            ax_bar.tick_params(
                axis="x", length=3, width=0.6, color="#8D959A", pad=3
            )

            ax_bar.spines["top"].set_visible(False)
            ax_bar.spines["right"].set_visible(False)
            ax_bar.spines["left"].set_visible(False)
            ax_bar.spines["bottom"].set_color("#A0A5A8")
            ax_bar.spines["bottom"].set_linewidth(0.7)

        path = save_svg(
            fig, out / "Reinj_source_reference_specificity.svg", show
        )

    print()
    print(
        assignments[[
            "scope", "mouse_number", "H_matched",
            "nearest_mismatched_source", "H_nearest_mismatched",
            "margin", "assigned_source", "correct",
        ]].to_string(index=False)
    )

    return dict(
        svg=str(path), hellinger=matrices, assignments=assignments
    )


# ============================================================
# FIGURE 5  recipient-region enrichment + constrained permutation
# ============================================================

def _candidates_for(pool, rid, log_tolerance, min_candidates):
    """
    한 target region 을 대체할 수 있는 region 후보.

    같은 major division 안에서 log volume 차이가 tolerance 이내인
    region 을 우선 사용하고, 후보가 부족하면 tolerance 를 넓힌 뒤
    division 제약을 풉니다.
    """

    log_area = np.log(pool["area"].to_numpy(dtype=float))

    division = pool["division"].to_numpy()

    position = pool.index.get_loc(rid)

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
    pool, target_ids, n_permutation, rng,
    log_tolerance=np.log(2.0), min_candidates=8,
):

    target_ids = list(target_ids)

    candidates = {}
    relaxation = {}

    for rid in target_ids:

        choices, how = _candidates_for(
            pool, rid, log_tolerance, min_candidates
        )

        candidates[rid] = choices
        relaxation[rid] = how

    order = sorted(target_ids, key=lambda rid: len(candidates[rid]))

    everything = np.array(pool.index)

    draws = []
    forced = 0

    for _ in range(n_permutation):

        used = set()

        for rid in order:

            available = [
                candidate for candidate in candidates[rid]
                if candidate not in used
            ]

            if not available:

                forced += 1

                available = [
                    candidate for candidate in everything
                    if candidate not in used
                ]

            used.add(int(rng.choice(available)))

        draws.append(np.array(sorted(used)))

    return draws, dict(
        relaxation=relaxation,
        n_candidates={
            rid: int(len(choices))
            for rid, choices in candidates.items()
        },
        forced_draws=forced,
    )


def make_enrichment(ctx, tables, show=None):

    out = ctx["output"]

    rng = np.random.default_rng(PERMUTATION_SEED)

    pool, parents = leaf_pool(ctx, tables)

    pool_ids = list(pool.index)

    sets = region_sets(ctx)

    require(
        "PredefinedM1" in sets,
        "PredefinedM1 region set 을 만들 수 없습니다."
    )

    m1_ids = set(sets["PredefinedM1"]["id"].astype(int))

    target_ids = [
        rid for rid in pool_ids
        if rid in m1_ids or (set(_ancestors(rid, parents)) & m1_ids)
    ]

    require(
        len(target_ids) >= 3,
        "broad region pool 안에 Figure 1 M1 recipient region 이 "
        "3개 미만입니다."
    )

    draws, info = _constrained_sets(
        pool, target_ids, N_PERMUTATION, rng
    )

    print()
    print("  permutation 제약:")

    for rid in target_ids:
        print(
            f"    {pool.at[rid, 'acronym']}: "
            f"{info['n_candidates'][rid]} candidates "
            f"({info['relaxation'][rid]})"
        )

    if info["forced_draws"]:
        print(f"    전체 pool 에서 강제로 뽑은 횟수: {info['forced_draws']}")

    pd.DataFrame([
        dict(
            region_id=rid,
            acronym=pool.at[rid, "acronym"],
            division=pool.at[rid, "division"],
            area=float(pool.at[rid, "area"]),
            n_candidates=info["n_candidates"][rid],
            constraint=info["relaxation"][rid],
        )
        for rid in target_ids
    ]).to_csv(out / "Reinj_permutation_constraints.csv", index=False)

    # --------------------------------------------------------
    # hemisphere 별로 따로 계산
    # --------------------------------------------------------

    raw = ctx["raw"].set_index(["mouse", "id"])

    units = []

    for mouse, frame in tables.items():

        sub = frame.loc[pool_ids]

        for hemisphere, column in (("RH", "rh_count"), ("LH", "lh_count")):

            units.append((
                "Reinj", mouse, hemisphere,
                sub[column].to_numpy(dtype=float),
            ))

    for mouse in cre_m1_mice(ctx):

        sub = raw.xs(mouse).reindex(pool_ids)

        side = str(
            ctx["manifest"].at[mouse, "injection_side"]
        ).strip().upper()

        ipsi = "rh" if side == "R" else "lh"
        contra = "lh" if side == "R" else "rh"

        for hemisphere, key in (("ipsi", ipsi), ("contra", contra)):

            units.append((
                "Cre", mouse, hemisphere,
                sub[f"{key}_count"].to_numpy(dtype=float),
            ))

    index_of = {rid: i for i, rid in enumerate(pool_ids)}

    target_index = np.array([index_of[rid] for rid in target_ids])

    draw_index = [
        np.array([index_of[rid] for rid in draw]) for draw in draws
    ]

    records = []
    nulls = {}

    for group, mouse, hemisphere, counts in units:

        total = float(counts.sum())

        if total <= 0:

            print(
                f"  경고: {mouse} / {hemisphere} 의 pool 내 count 가 0 "
                "이라 enrichment 를 계산하지 않습니다."
            )

            continue

        observed = float(counts[target_index].sum() / total)

        null = np.array([
            float(counts[index].sum() / total) for index in draw_index
        ])

        records.append(dict(
            group=group,
            mouse=mouse,
            hemisphere=hemisphere,
            n_pool_regions=len(pool_ids),
            n_target_regions=len(target_ids),
            total_count=total,
            target_count=float(counts[target_index].sum()),
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
            nulls[(mouse, hemisphere)] = null

    enrichment_df = pd.DataFrame(records)

    enrichment_df.to_csv(
        out / "Reinj_recipient_region_enrichment.csv", index=False
    )

    pool.reset_index().to_csv(
        out / "Reinj_permutation_region_pool.csv", index=False
    )

    reinj_mice = list(tables)

    hemispheres = ["RH", "LH"]

    cre_mean = {
        "ipsi": enrichment_df.loc[
            (enrichment_df.group == "Cre")
            & (enrichment_df.hemisphere == "ipsi"), "enrichment"
        ].mean(),
        "contra": enrichment_df.loc[
            (enrichment_df.group == "Cre")
            & (enrichment_df.hemisphere == "contra"), "enrichment"
        ].mean(),
    }

    reference_for = {"RH": "ipsi", "LH": "contra"}

    if ORIENTATION == "rh_to_contra":
        reference_for = {"RH": "contra", "LH": "ipsi"}

    with rc():

        fig, axes = plt.subplots(
            len(hemispheres), len(reinj_mice),
            figsize=(3.3 * len(reinj_mice) + 0.4, 2.9 * len(hemispheres) + 1.0),
            sharey=False, squeeze=False,
        )

        fig.subplots_adjust(
            left=0.085, right=0.985, top=0.84, bottom=0.17,
            wspace=0.18, hspace=0.45,
        )

        for row, hemisphere in enumerate(hemispheres):

            for col, mouse in enumerate(reinj_mice):

                ax = axes[row, col]

                key = (mouse, hemisphere)

                if key not in nulls:
                    ax.set_visible(False)
                    continue

                null = nulls[key]

                record = enrichment_df.loc[
                    (enrichment_df.mouse == mouse)
                    & (enrichment_df.hemisphere == hemisphere)
                ].iloc[0]

                ax.hist(
                    null, bins=40, color="#D8DCDF",
                    edgecolor="white", linewidth=0.3, zorder=2,
                )

                ax.axvline(
                    record["enrichment"],
                    color=REINJ_COLOR, linewidth=2.0, zorder=4,
                )

                cre_value = cre_mean[reference_for[hemisphere]]

                if np.isfinite(cre_value):

                    ax.axvline(
                        cre_value,
                        color=SOURCE_COLORS[MATCHED_SOURCE],
                        linewidth=1.6, linestyle="--", zorder=3,
                    )

                ax.set_title(
                    f"Mouse {col + 1} · {hemisphere}\n"
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
            f"{len(target_ids)} target regions out of {len(pool_ids)} "
            f"non-overlapping regions, {N_PERMUTATION} permutations",
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

    return dict(
        svg=str(path), enrichment=enrichment_df,
        pool=pool, target_ids=target_ids,
    )


# ============================================================
# FIGURE 6  reference density count model
# ============================================================

def make_reference_model(ctx, tables, show=None):

    out = ctx["output"]

    pool, _ = leaf_pool(ctx, tables)

    pool_ids = list(pool.index)

    cre_mice = cre_m1_mice(ctx)

    # Figure 1 reference: mouse 별 p 를 구한 뒤 동일 weight 평균
    p_cre = to_p(fig1_density(ctx, pool_ids, mode="lateralized"))

    reference = p_cre.loc[cre_mice].mean(axis=0)

    positive = reference[reference > 0]

    require(
        len(positive) >= 10,
        "Cre M1 reference 에서 0보다 큰 feature 가 10개 미만입니다."
    )

    floor = float(positive.min()) / 2.0

    slot_for = {"RH": "s1", "LH": "s2"}

    if ORIENTATION == "rh_to_contra":
        slot_for = {"RH": "s2", "LH": "s1"}

    rows = []

    for mouse, frame in tables.items():

        sub = frame.loc[pool_ids]

        for hemisphere in ("RH", "LH"):

            slot = slot_for[hemisphere]

            available = [
                rid for rid in pool_ids
                if (slot, rid) in set(reference.index)
            ]

            values = reference.loc[[(slot, rid) for rid in available]]

            rows.append(pd.DataFrame({
                "mouse": mouse,
                "hemisphere": hemisphere,
                "region_id": available,
                "acronym": pool.loc[available, "acronym"].to_numpy(),
                "count": sub.loc[
                    available, f"{hemisphere.lower()}_count"
                ].to_numpy(dtype=float),
                "area": sub.loc[
                    available, f"{hemisphere.lower()}_area"
                ].to_numpy(dtype=float),
                "reference_p": values.to_numpy(dtype=float),
            }))

    data = pd.concat(rows, ignore_index=True)

    data["log_reference_p"] = np.log(
        data["reference_p"].clip(lower=floor)
    )

    data = data.loc[data["area"] > AREA_ATOL].copy()

    data["observed_density"] = data["count"] / data["area"]

    design = pd.DataFrame({
        "log_reference_p": data["log_reference_p"].to_numpy(),
    })

    mouse_list = list(tables)

    for mouse in mouse_list[1:]:
        design[f"mouse[{mouse}]"] = (
            data["mouse"] == mouse
        ).astype(float).to_numpy()

    design["hemisphere[LH]"] = (
        data["hemisphere"] == "LH"
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

        for hemisphere in ("RH", "LH"):

            sub = data.loc[
                (data.mouse == mouse) & (data.hemisphere == hemisphere)
            ]

            if sub.empty:
                continue

            per_unit.append(dict(
                mouse=mouse,
                hemisphere=hemisphere,
                n_regions=len(sub),
                spearman_rho=spearman_rho(
                    sub["reference_p"].to_numpy(),
                    sub["observed_density"].to_numpy(),
                ),
                fraction_nonzero=float((sub["count"] > 0).mean()),
            ))

    per_unit_df = pd.DataFrame(per_unit)

    summary = pd.DataFrame([dict(
        model=model_name,
        n_observations=int(len(data)),
        n_regions=int(data.region_id.nunique()),
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
            figsize=(3.3 * len(mouse_list) + 0.4, 6.6),
            sharex=True, sharey=True, squeeze=False,
        )

        fig.subplots_adjust(
            left=0.095, right=0.985, top=0.84, bottom=0.12,
            wspace=0.14, hspace=0.32,
        )

        for row, hemisphere in enumerate(("RH", "LH")):

            for col, mouse in enumerate(mouse_list):

                ax = axes[row, col]

                sub = data.loc[
                    (data.mouse == mouse)
                    & (data.hemisphere == hemisphere)
                ]

                if sub.empty:
                    ax.set_visible(False)
                    continue

                x = sub["reference_p"].to_numpy(dtype=float)
                y = sub["observed_density"].to_numpy(dtype=float)

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
                    & (per_unit_df.hemisphere == hemisphere),
                    "spearman_rho",
                ].iloc[0]

                ax.set_title(
                    f"Mouse {col + 1} · {hemisphere}\n"
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
                        "Reinjection density\n(cells / area)",
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
# "signal > negative control" 검정이 아니라 두 group 의 절대량
# 비교입니다.
# ============================================================

TOP_N = 10


def make_qc(ctx, tables, show=None):

    out = ctx["output"]

    pool, _ = leaf_pool(ctx, tables)

    pool_ids = list(pool.index)

    raw = ctx["raw"].set_index(["mouse", "id"])

    records = []
    curves = {}

    def add_unit(group, label, counts, areas, first, second):

        total = float(np.nansum(counts))

        require(total > 0, f"{label}: 분석 region 전체 count 가 0 입니다.")

        density = np.where(
            areas > AREA_ATOL,
            counts / np.where(areas > AREA_ATOL, areas, np.nan),
            np.nan,
        )

        p = np.nan_to_num(density, nan=0.0)

        p = p / p.sum()

        order = np.sort(p)[::-1]

        curves[(group, label)] = np.cumsum(order)

        records.append(dict(
            group=group,
            label=label,
            n_regions=len(pool_ids),
            total_count=total,
            n_regions_with_signal=int(np.sum(counts > 0)),
            coverage=float(np.mean(counts > 0)),
            top_region_share=float(order[0]),
            top_n_share=float(order[:TOP_N].sum()),
            laterality_index=float(np.nansum(first) / total),
            second_side_count=float(np.nansum(second)),
        ))

    for mouse, frame in tables.items():

        sub = frame.loc[pool_ids]

        add_unit(
            "Reinj", mouse,
            (sub["rh_count"] + sub["lh_count"]).to_numpy(float),
            (sub["rh_area"] + sub["lh_area"]).to_numpy(float),
            sub["rh_count"].to_numpy(float),
            sub["lh_count"].to_numpy(float),
        )

    for mouse in cre_m1_mice(ctx):

        sub = raw.xs(mouse).reindex(pool_ids)

        side = str(
            ctx["manifest"].at[mouse, "injection_side"]
        ).strip().upper()

        ipsi = "rh" if side == "R" else "lh"
        contra = "lh" if side == "R" else "rh"

        add_unit(
            "Cre", mouse,
            (sub[f"{ipsi}_count"] + sub[f"{contra}_count"]).to_numpy(float),
            (sub[f"{ipsi}_area"] + sub[f"{contra}_area"]).to_numpy(float),
            sub[f"{ipsi}_count"].to_numpy(float),
            sub[f"{contra}_count"].to_numpy(float),
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
                    f"EV reinjection\n(n = {int((qc.group == 'Reinj').sum())})",
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
            ylabel="tdTomato+ cells in analyzed regions",
        )

        axes[0, 0].set_title(
            "Absolute signal", fontsize=10.5, fontweight="bold", pad=7
        )

        strip(
            axes[0, 1], "coverage",
            ylabel="Fraction of regions with ≥ 1 cell",
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
            "Number of regions (ranked)",
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
            ylabel="Cre: ipsi / total   ·   Reinj: RH / total",
        )

        axes[0, 3].set_ylim(0, 1.03)

        axes[0, 3].set_title(
            "Laterality", fontsize=10.5, fontweight="bold", pad=7
        )

        fig.text(
            0.5, 0.955,
            "Signal, coverage and laterality QC   "
            f"({len(pool_ids)} non-overlapping regions)",
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


# ============================================================
# RUN
# ============================================================

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

    for name, function in STEPS:

        if only is not None and name not in only:
            continue

        print()
        print("##########################################")
        print(f"# {name}")
        print("##########################################")

        try:
            results[name] = function(ctx, tables)
        except Exception:  # noqa: BLE001
            if stop_on_error:
                raise
            print()
            print(f"[{name}] 실패:")
            traceback.print_exc()
            results[name] = None

    print()
    print("==========================================")
    print("ALL DONE")
    print("==========================================")
    print(ctx["output"])
    print()

    for name, value in results.items():
        print(f"{name}: {'ok' if value is not None else 'FAILED'}")

    return dict(ctx=ctx, tables=tables, results=results)


if __name__ == "__main__":
    REINJ = run_all()
