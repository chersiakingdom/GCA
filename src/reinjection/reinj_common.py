# ============================================================
# 공통 모듈
#
# AAV-Cre M1 (n=4) vs intranasal EV reinjection (n=3)
#
# 모든 figure 코드가 이 모듈을 통해 동일한 feature space,
# 동일한 normalization, 동일한 QC 를 사용합니다.
# ============================================================

from pathlib import Path
import json
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager

try:
    from . import reinj_config as C
except ImportError:
    import reinj_config as C


# ============================================================
# FONT
#
# 서버에 Arial 이 없으면 Liberation Sans 로 렌더링한 뒤
# 저장된 SVG 내부 font-family 만 Arial 로 변경합니다.
# ============================================================

_available_fonts = {
    f.name for f in font_manager.fontManager.ttflist
}

if "Arial" in _available_fonts:
    RENDER_FONT = "Arial"
elif "Liberation Sans" in _available_fonts:
    RENDER_FONT = "Liberation Sans"
else:
    RENDER_FONT = "DejaVu Sans"

TARGET_SVG_FONT = "Arial"


def rc():
    """기존 figure 코드들과 동일한 rc context."""

    return plt.rc_context({
        "font.family": RENDER_FONT,
        "font.size": 9,
        "svg.fonttype": "none",
        "axes.unicode_minus": False,
    })


def save_svg(fig, path, show=None):
    """SVG 로만 저장하고 font-family 를 Arial 로 맞춥니다."""

    path = Path(path)

    fig.savefig(
        path,
        format="svg",
        bbox_inches="tight",
        facecolor="white",
    )

    if RENDER_FONT != TARGET_SVG_FONT:

        text = path.read_text(encoding="utf-8")

        path.write_text(
            text.replace(RENDER_FONT, TARGET_SVG_FONT),
            encoding="utf-8",
        )

    if (C.SHOW_FIGURES if show is None else show):
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
        np.linalg.norm(np.sqrt(a) - np.sqrt(b))
        / np.sqrt(2)
    )


def spearman_rho(a, b):
    """Rank correlation. 분산이 0이면 NaN."""

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
    Density (count / area) 를 mouse 별 sum = 1 로 normalize.

    Eq6 의 off-source denominator 는 한 mouse 안에서 모든 region 에
    동일하게 곱해지는 상수이므로 p 계산에서 소거됩니다.
    따라서 area 단위가 두 dataset 에서 달라도 p 는 영향받지 않습니다.
    """

    total = density.sum(axis=1)

    require(
        np.isfinite(total).all() and (total > 0).all(),
        "density 합이 0 또는 비유한값인 mouse 가 있습니다:\n"
        f"{total[~(np.isfinite(total) & (total > 0))]}"
    )

    p = density.div(total, axis=0)

    require(
        np.allclose(p.sum(axis=1), 1, atol=1e-8, rtol=1e-8),
        "p 합이 1이 아닙니다."
    )

    return p


# ============================================================
# COLUMN 자동 인식
# ============================================================

_ID_CANDIDATES = [
    "id", "region_id", "structure_id", "atlas_id",
    "structureid", "regionid",
]

_NAME_CANDIDATES = [
    "name", "region_name", "structure_name", "safe_name",
]

_ACRONYM_CANDIDATES = [
    "acronym", "abbrev", "abbreviation", "region_label", "label",
]

_COUNT_CANDIDATES = [
    "count", "cell_count", "cells", "n_cells", "tdt_count",
    "total_cell_count", "tdtomato_count", "n",
]

_AREA_CANDIDATES = [
    "area", "region_area", "volume", "region_volume",
    "voxels", "voxel_count", "n_voxels", "area_um2",
    "volume_mm3", "volume_um3",
]


def _normalize(name):
    return re.sub(r"[^a-z0-9]", "", str(name).strip().lower())


def detect_column(frame, candidates, role, where, required=True):
    """
    후보 이름과 정확히 일치하는 column 을 먼저 찾고,
    없으면 부분 일치로 찾습니다.

    찾지 못하면 실제 column 목록을 포함한 에러를 냅니다.
    """

    lookup = {
        _normalize(col): col
        for col in frame.columns
    }

    for candidate in candidates:

        key = _normalize(candidate)

        if key in lookup:
            return lookup[key]

    for candidate in candidates:

        key = _normalize(candidate)

        for norm, original in lookup.items():

            if key and key in norm:
                return original

    if not required:
        return None

    raise ValueError(
        f"{where}\n"
        f"'{role}' 에 해당하는 column 을 찾지 못했습니다.\n"
        f"기대한 이름 후보: {candidates}\n"
        f"실제 column: {list(frame.columns)}\n"
        "reinj_common.py 상단의 후보 목록에 실제 이름을 추가하십시오."
    )


# ============================================================
# REINJECTION DATA 읽기
# ============================================================

def _read_level_file(path):
    """
    tdt_total_cell_count_*_lvl{n}.csv 한 개를 읽어
    id / count / area 로 정리합니다.
    """

    require(
        path.is_file(),
        f"파일을 찾을 수 없습니다:\n{path}"
    )

    frame = pd.read_csv(path)

    frame.columns = [
        str(col).strip()
        for col in frame.columns
    ]

    where = f"파일: {path}"

    id_col = detect_column(frame, _ID_CANDIDATES, "region id", where)
    count_col = detect_column(frame, _COUNT_CANDIDATES, "cell count", where)
    area_col = detect_column(frame, _AREA_CANDIDATES, "region area/volume", where)

    acronym_col = detect_column(
        frame, _ACRONYM_CANDIDATES, "acronym", where, required=False
    )

    name_col = detect_column(
        frame, _NAME_CANDIDATES, "region name", where, required=False
    )

    out = pd.DataFrame({
        "id": pd.to_numeric(frame[id_col], errors="coerce"),
        "count": pd.to_numeric(frame[count_col], errors="coerce"),
        "area": pd.to_numeric(frame[area_col], errors="coerce"),
    })

    out["acronym"] = (
        frame[acronym_col].astype(str)
        if acronym_col is not None
        else ""
    )

    out["name"] = (
        frame[name_col].astype(str)
        if name_col is not None
        else ""
    )

    out = out.loc[out["id"].notna()].copy()

    out["id"] = out["id"].astype(int)

    require(
        out["count"].notna().all() and out["area"].notna().all(),
        f"{where}\ncount 또는 area 에 숫자가 아닌 값이 있습니다."
    )

    return out


def _combine_levels(root, template, label):
    """lvl1~7 을 합쳐 region id 단위 표로 만듭니다."""

    results = C.reinj_results_dir(root)

    frames = []

    for level in C.LEVELS:

        path = results / template.format(level=level)

        frame = _read_level_file(path)

        frame["level"] = level

        frames.append(frame)

    merged = pd.concat(frames, ignore_index=True)

    # 같은 region 이 여러 level 파일에 중복되어 있으면
    # 값이 일치하는지 확인한 뒤 하나만 남깁니다.
    duplicated = merged.loc[merged.duplicated("id", keep=False)]

    if len(duplicated):

        spread = duplicated.groupby("id").agg(
            count_spread=("count", lambda s: float(np.ptp(s))),
            area_spread=("area", lambda s: float(np.ptp(s))),
        )

        bad = spread.loc[
            (spread.count_spread > C.COUNT_ATOL)
            | (spread.area_spread > C.AREA_ATOL)
        ]

        require(
            bad.empty,
            f"{label}: 여러 level 파일에서 같은 region id 의 "
            f"count/area 가 다릅니다.\n{bad.head(20)}"
        )

    merged = (
        merged
        .sort_values(["id", "level"])
        .drop_duplicates("id", keep="first")
        .set_index("id", verify_integrity=True)
        .sort_index()
    )

    return merged


def load_reinj_tables(roots=None):
    """
    reinjection mouse 별로
    rh_count / rh_area / whole_count / whole_area / lh_count / lh_area
    표를 만듭니다.

    lh = whole - rh
    """

    roots = C.REINJ_ROOTS if roots is None else roots

    tables = {}

    for mouse, root in roots.items():

        rh = _combine_levels(root, C.RH_FILE_TEMPLATE, f"{mouse} rh")

        whole = _combine_levels(root, C.WHOLE_FILE_TEMPLATE, f"{mouse} whole")

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

        # ----------------------------------------------------
        # QC
        # ----------------------------------------------------

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

        bad_count = frame.loc[
            frame["lh_count"] < -C.NEGATIVE_COUNT_TOL
        ]

        require(
            bad_count.empty,
            f"{mouse}: rh_count > whole_count 인 region 이 있습니다.\n"
            f"{bad_count[['acronym', 'rh_count', 'whole_count']].head(20)}"
        )

        bad_area = frame.loc[
            frame["lh_area"] < -abs(frame["whole_area"]) * 1e-9 - C.NEGATIVE_TOL
        ]

        require(
            bad_area.empty,
            f"{mouse}: rh_area > whole_area 인 region 이 있습니다.\n"
            f"{bad_area[['acronym', 'rh_area', 'whole_area']].head(20)}"
        )

        frame["lh_count"] = frame["lh_count"].clip(lower=0.0)
        frame["lh_area"] = frame["lh_area"].clip(lower=0.0)

        tables[mouse] = frame

    return tables


# ============================================================
# FIGURE 1 결과 읽기
# ============================================================

def resolve_run_dir(run_dir=None):

    if run_dir is None:
        run_dir = C.FIG1_RUN_DIR

    if run_dir is None:
        run_dir = globals().get("RESULTS", {}).get("output")

    if run_dir is None:
        import builtins
        run_dir = getattr(builtins, "RESULTS", {}).get("output") \
            if isinstance(getattr(builtins, "RESULTS", None), dict) else None

    require(
        run_dir is not None,
        "Figure 1 분석 폴더를 찾을 수 없습니다.\n"
        "reinj_config.py 의 FIG1_RUN_DIR 를 지정하거나 "
        "RESULTS['output'] 이 있는 상태에서 실행하십시오."
    )

    return Path(run_dir).expanduser().resolve()


def resolve_output_dir(run):

    if C.OUTPUT_DIR is not None:
        out = Path(C.OUTPUT_DIR).expanduser()
    else:
        out = run.parent / "Figure_list_reinjection"

    out.mkdir(parents=True, exist_ok=True)

    return out


def load_atlas(cfg):

    path = Path(cfg["atlas_path"]).expanduser()

    if path.suffix.lower() in (".xlsx", ".xls"):
        atlas = pd.read_excel(
            path,
            sheet_name=cfg.get("atlas_sheet", 0),
        )
    else:
        atlas = pd.read_csv(path)

    atlas.columns = atlas.columns.str.strip().str.lower()

    require(
        not atlas.id.duplicated().any(),
        "Atlas id 중복"
    )

    return atlas


def load_fig1(run_dir=None):
    """Figure 1 run 폴더에서 필요한 모든 정보를 읽습니다."""

    run = resolve_run_dir(run_dir)

    saved = json.loads(
        (run / "config.json").read_text(encoding="utf-8")
    )

    status = json.loads(
        (run / "RUN_STATUS.json").read_text(encoding="utf-8")
    )

    require(
        status.get("status") == "completed",
        "완료된 Figure 1 분석 결과 폴더가 아닙니다."
    )

    cfg = saved["config"]
    cohorts = saved["cohorts"]

    mice = [
        m
        for source in C.SOURCE_ORDER
        for m in cohorts["Cre"][source]
    ]

    require(
        len(set(mice)) == len(mice),
        "Figure 1 Cre cohort 에 mouse id 중복이 있습니다."
    )

    manifest = pd.read_csv(
        run / "input_manifest.csv"
    ).set_index("mouse", verify_integrity=True)

    raw = pd.read_csv(run / "raw_readout_regions.csv")

    presets = pd.read_csv(run / "predefined_recipient_regions.csv")

    source_ids = set(
        pd.read_csv(run / "source_region_definition.csv").id.astype(int)
    )

    totals = pd.read_csv(
        run / "off_source_counts.csv"
    ).set_index("mouse", verify_integrity=True)

    return dict(
        run=run,
        output=resolve_output_dir(run),
        cfg=cfg,
        cohorts=cohorts,
        cre_mice=mice,
        manifest=manifest,
        raw=raw,
        presets=presets,
        source_ids=source_ids,
        totals=totals,
    )


# ============================================================
# REGION SET
# ============================================================

def _allgraymatter_ids(ctx, source):
    """
    primary/AllGrayMatter/<source>/feature_definitions.csv 에서
    region id 를 읽습니다.

    residual (parent - children) feature 가 정의되어 있고 그 정의를
    해석할 수 없는 경우에는 조용히 넘어가지 않고 에러를 냅니다.
    """

    path = (
        ctx["run"]
        / "primary"
        / "AllGrayMatter"
        / source
        / "feature_definitions.csv"
    )

    if not path.is_file():
        return None

    meta = pd.read_csv(path)

    meta.columns = [str(c).strip() for c in meta.columns]

    where = f"파일: {path}"

    id_col = detect_column(
        meta, ["region_id", "id", "structure_id"],
        "region id", where,
    )

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
            f"{where}\n"
            f"'{col}' column 에 residual(parent - children) 정의가 "
            "들어 있습니다.\n"
            "이 정의를 reinjection data 에도 동일하게 적용해야 하므로, "
            "해당 column 의 형식을 확인한 뒤 "
            "reinj_common.py 의 _allgraymatter_ids 를 보완해야 합니다.\n"
            "확인 전에는 AllGrayMatter scope 를 사용하지 마십시오 "
            "(reinj_config.SCOPES 에서 제거)."
        )

    hemi_col = detect_column(
        meta, ["hemisphere", "hemi", "side"],
        "hemisphere", where, required=False,
    )

    if hemi_col is not None:

        hemisphere = (
            meta[hemi_col].astype(str).str.strip().str.lower()
        )

        subset = meta.loc[hemisphere.isin(["ipsi", "rh", "right"])]

        if subset.empty:
            subset = meta

    else:
        subset = meta

    ids = sorted(
        set(
            pd.to_numeric(subset[id_col], errors="coerce")
            .dropna()
            .astype(int)
            .tolist()
        )
    )

    return ids or None


def region_sets(ctx):
    """
    scope 별 region id / acronym 을 만듭니다.

    AllGrayMatter 는 Figure 1 의 정의를 그대로 사용하며,
    정의 파일이 없거나 해석 불가능하면 제외됩니다.
    """

    presets = ctx["presets"]

    atlas = load_atlas(ctx["cfg"])

    acronym_by_id = (
        atlas.set_index("id")["acronym"].astype(str).to_dict()
        if "acronym" in atlas.columns
        else {}
    )

    name_by_id = (
        atlas.set_index("id")["name"].astype(str).to_dict()
        if "name" in atlas.columns
        else {}
    )

    sets = {}

    # ----------------------------------------------------
    # M1 predefined recipient regions
    # ----------------------------------------------------

    m1 = presets.loc[presets.source == C.MATCHED_SOURCE]

    require(
        not m1.empty,
        f"predefined_recipient_regions.csv 에 source == "
        f"{C.MATCHED_SOURCE} 인 region 이 없습니다."
    )

    sets["PredefinedM1"] = m1.drop_duplicates("id").copy()

    # ----------------------------------------------------
    # predefined 20 regions (union)
    # ----------------------------------------------------

    union = pd.concat(
        [presets.loc[presets.source == s] for s in C.SOURCE_ORDER],
        ignore_index=True,
    ).drop_duplicates("id").copy()

    sets["Predefined20"] = union

    # ----------------------------------------------------
    # AllGrayMatter
    # ----------------------------------------------------

    agm_ids = _allgraymatter_ids(ctx, C.MATCHED_SOURCE)

    if agm_ids:

        sets["AllGrayMatter"] = pd.DataFrame({
            "id": agm_ids,
            "acronym": [
                acronym_by_id.get(i, str(i)) for i in agm_ids
            ],
            "name": [
                name_by_id.get(i, str(i)) for i in agm_ids
            ],
            "source": "AllGrayMatter",
        })

    # ----------------------------------------------------
    # source region 제외 (분석 전략 문서 3번)
    #
    # source-specificity 를 위해 네 source 의 union 을 모두 제외한
    # 동일한 feature space 를 사용합니다.
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

        cleaned[scope] = (
            kept
            .drop_duplicates("id")
            .reset_index(drop=True)
        )

    return {
        scope: cleaned[scope]
        for scope in C.SCOPES
        if scope in cleaned
    }


# ============================================================
# DENSITY 계산
# ============================================================

def fig1_density(ctx, ids, mode=None):
    """
    Figure 1 Cre mouse 의 regional density.

    mode = "pooled"  : (ipsi + contra) count / (ipsi + contra) area
    mode = "ipsi"    : ipsilateral 만
    """

    mode = C.HEMI_MODE if mode is None else mode

    ids = [int(i) for i in ids]

    raw = ctx["raw"]

    frame = raw.loc[
        raw.mouse.isin(ctx["cre_mice"]) & raw.id.isin(ids)
    ].set_index(["mouse", "id"], verify_integrity=True)

    density = pd.DataFrame(
        index=ctx["cre_mice"], columns=ids, dtype=float
    )

    for mouse in ctx["cre_mice"]:

        side = str(
            ctx["manifest"].at[mouse, "injection_side"]
        ).strip().upper()

        require(
            side in ("R", "L"),
            f"{mouse}: injection_side 가 R/L 이 아닙니다."
        )

        sub = frame.xs(mouse).reindex(ids)

        require(
            sub[["rh_count", "rh_area", "lh_count", "lh_area"]]
            .notna().to_numpy().all(),
            f"{mouse}: raw_readout_regions.csv 에 없는 region 이 있습니다.\n"
            f"{sub.index[sub.rh_count.isna()].tolist()}"
        )

        ipsi = "rh" if side == "R" else "lh"
        contra = "lh" if side == "R" else "rh"

        if mode == "pooled":
            count = sub[f"{ipsi}_count"] + sub[f"{contra}_count"]
            area = sub[f"{ipsi}_area"] + sub[f"{contra}_area"]
        elif mode == "ipsi":
            count = sub[f"{ipsi}_count"]
            area = sub[f"{ipsi}_area"]
        elif mode == "contra":
            count = sub[f"{contra}_count"]
            area = sub[f"{contra}_area"]
        else:
            raise ValueError(f"알 수 없는 hemisphere mode: {mode}")

        density.loc[mouse] = _density(count, area)

    return _finalize_density(density, "Figure 1 Cre")


def reinj_density(tables, ids, mode=None):
    """
    Reinjection mouse 의 regional density.

    mode = "pooled"     : whole (rh + lh)
    mode = "rh_as_ipsi" : rh 만
    mode = "lh"         : lh 만
    """

    mode = C.HEMI_MODE if mode is None else mode

    ids = [int(i) for i in ids]

    density = pd.DataFrame(
        index=list(tables), columns=ids, dtype=float
    )

    for mouse, frame in tables.items():

        sub = frame.reindex(ids)

        missing = sub.index[sub["whole_count"].isna()].tolist()

        require(
            not missing,
            f"{mouse}: lvl1~7 파일에 없는 region id 가 있습니다: "
            f"{missing[:20]}"
        )

        if mode in ("pooled", "whole"):
            count = sub["whole_count"]
            area = sub["whole_area"]
        elif mode in ("rh_as_ipsi", "rh"):
            count = sub["rh_count"]
            area = sub["rh_area"]
        elif mode == "lh":
            count = sub["lh_count"]
            area = sub["lh_area"]
        else:
            raise ValueError(f"알 수 없는 hemisphere mode: {mode}")

        density.loc[mouse] = _density(count, area)

    return _finalize_density(density, "Reinjection")


def _density(count, area):

    count = count.astype(float).to_numpy()
    area = area.astype(float).to_numpy()

    present = area > C.AREA_ATOL

    require(
        not ((count > C.COUNT_ATOL) & ~present).any(),
        "area = 0 인데 count > 0 인 region 이 있습니다."
    )

    out = np.full(len(count), np.nan, dtype=float)

    out[present] = count[present] / area[present]

    return out


def _finalize_density(density, label):
    """
    모든 mouse 에서 area = 0 인 region 은 feature 에서 제외하고,
    일부 mouse 에서만 0 인 경우에는 에러를 냅니다.
    """

    valid = density.notna()

    inconsistent = valid.columns[
        valid.any(axis=0) != valid.all(axis=0)
    ].tolist()

    require(
        not inconsistent,
        f"{label}: 일부 mouse 에서만 area = 0 인 region 이 있습니다: "
        f"{inconsistent[:20]}"
    )

    kept = valid.columns[valid.all(axis=0)]

    require(
        len(kept) > 0,
        f"{label}: 유효한 region 이 없습니다."
    )

    return density.loc[:, kept].astype(float)


def common_feature_space(ctx, tables, ids, mode_cre=None, mode_reinj=None):
    """
    두 group 에서 모두 유효한 region 만 남긴 뒤
    동일한 feature 순서로 p 를 반환합니다.
    """

    mode_cre = (
        "pooled" if (mode_cre or C.HEMI_MODE) == "pooled" else "ipsi"
    )

    mode_reinj = mode_reinj or C.HEMI_MODE

    cre = fig1_density(ctx, ids, mode=mode_cre)

    reinj = reinj_density(tables, ids, mode=mode_reinj)

    shared = [
        int(i)
        for i in ids
        if i in set(cre.columns) and i in set(reinj.columns)
    ]

    require(
        len(shared) >= 3,
        "두 dataset 에서 공통으로 유효한 region 이 3개 미만입니다."
    )

    return (
        to_p(cre.loc[:, shared]),
        to_p(reinj.loc[:, shared]),
        shared,
    )


# ============================================================
# 라벨
# ============================================================

def cre_m1_mice(ctx):

    mice = list(ctx["cohorts"]["Cre"][C.MATCHED_SOURCE])

    require(
        len(mice) > 0,
        f"Cre / {C.MATCHED_SOURCE} cohort 가 비어 있습니다."
    )

    return mice


def acronym_map(frame):

    return {
        int(r.id): str(r.acronym)
        for r in frame.itertuples(index=False)
    }
