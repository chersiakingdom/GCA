# ============================================================
# FIGURE 5
#
# Recipient-region enrichment + constrained region permutation
#
# 분석 전략 문서 5-B / 10번
#
#   enrichment =
#       (Figure 1 M1 recipient region 들의 tdT+ cell 수)
#       / (분석에 포함된 전체 tdT+ cell 수)
#
# 비교 대상은 free shuffle 이 아니라
#
#   - major anatomical division
#   - region volume category
#
# 를 보존한 random region set 입니다.
#
# 따라서 p-value 의 의미는
# "실제 recipient region 으로의 집중이 anatomical division 과
#  region size 만 맞춘 random region set 보다 높은가" 입니다.
#
# biological replicate 수준의 동일성을 검정하는 것이 아닙니다.
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from . import reinj_common as K
    from . import reinj_config as C
except ImportError:
    import reinj_common as K
    import reinj_config as C


# Allen CCF major divisions
DIVISION_ACRONYMS = [
    "Isocortex", "OLF", "HPF", "CTXsp",
    "STR", "PAL", "TH", "HY", "MB", "P", "MY", "CB",
]

N_VOLUME_BINS = 3


# ============================================================
# atlas hierarchy
# ============================================================

def _hierarchy(atlas):

    parents = {}

    for row in atlas.itertuples():

        value = row.parent_structure_id

        parents[int(row.id)] = (
            None
            if (pd.isna(value) or value < 0)
            else int(value)
        )

    return parents


def _ancestors(rid, parents):

    chain = []
    seen = {rid}

    parent = parents.get(rid)

    while parent is not None:

        K.require(
            parent not in seen,
            "Atlas hierarchy 에 순환이 있습니다."
        )

        seen.add(parent)
        chain.append(parent)

        parent = parents.get(parent)

    return chain


def leaf_pool(ctx, tables):
    """
    두 dataset 에 모두 존재하는 region 중,
    자기 자신의 descendant 가 함께 포함되지 않은 region 만 남겨
    서로 겹치지 않는 broad region pool 을 만듭니다.

    (분석 전략 문서 2번 parent-child overlap 처리)
    """

    atlas = K.load_atlas(ctx["cfg"])

    parents = _hierarchy(atlas)

    fig1_ids = set(ctx["raw"].id.astype(int))

    reinj_ids = set.intersection(*[
        set(frame.index.astype(int)) for frame in tables.values()
    ])

    available = (fig1_ids & reinj_ids) - set(ctx["source_ids"])

    # source region 의 descendant 도 제외
    available = {
        rid
        for rid in available
        if not (set(_ancestors(rid, parents)) & set(ctx["source_ids"]))
    }

    K.require(
        len(available) > 20,
        "broad region pool 이 20개 미만입니다. "
        "Figure 1 raw_readout_regions.csv 와 reinjection lvl 파일의 "
        "region id 가 맞는지 확인하십시오."
    )

    has_descendant = set()

    for rid in available:

        for ancestor in _ancestors(rid, parents):

            if ancestor in available:
                has_descendant.add(ancestor)

    pool = sorted(available - has_descendant)

    # ----------------------------------------------------
    # major division / volume category
    # ----------------------------------------------------

    acronym_by_id = (
        atlas.set_index("id")["acronym"].astype(str).to_dict()
        if "acronym" in atlas.columns
        else {}
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

    area = pd.Series(
        {
            rid: float(np.mean([
                frame.at[rid, "whole_area"]
                for frame in tables.values()
            ]))
            for rid in pool
        },
        name="area",
    )

    frame = pd.DataFrame({
        "id": pool,
        "acronym": [acronym_by_id.get(rid, str(rid)) for rid in pool],
        "division": [division_of(rid) for rid in pool],
        "area": area.loc[pool].to_numpy(),
    })

    frame["volume_bin"] = pd.qcut(
        frame["area"].rank(method="first"),
        N_VOLUME_BINS,
        labels=False,
    )

    frame["stratum"] = (
        frame["division"].astype(str)
        + "|"
        + frame["volume_bin"].astype(str)
    )

    return frame.set_index("id"), parents


# ============================================================
# permutation
# ============================================================

def _candidates_for(pool, rid, log_tolerance, min_candidates):
    """
    한 target region 을 대체할 수 있는 region 후보.

    같은 major division 안에서 log volume 차이가 tolerance 이내인
    region 을 쓰되, 후보가 너무 적으면 tolerance 를 단계적으로 넓히고
    그래도 부족하면 division 제약을 풉니다.
    """

    log_area = np.log(pool["area"].to_numpy(dtype=float))

    division = pool["division"].to_numpy()

    position = pool.index.get_loc(rid)

    target_log_area = log_area[position]

    target_division = division[position]

    same_division = division == target_division

    tolerance = log_tolerance

    for _ in range(4):

        mask = same_division & (
            np.abs(log_area - target_log_area) <= tolerance
        )

        if mask.sum() >= min_candidates:
            return np.array(pool.index)[mask], "division+volume"

        tolerance *= 2

    # division 제약을 풀고 volume 만 맞춤
    tolerance = log_tolerance

    for _ in range(4):

        mask = np.abs(log_area - target_log_area) <= tolerance

        if mask.sum() >= min_candidates:
            return np.array(pool.index)[mask], "volume_only"

        tolerance *= 2

    return np.array(pool.index), "whole_pool"


def _constrained_sets(
    pool,
    target_ids,
    n_permutation,
    rng,
    log_tolerance=np.log(2.0),
    min_candidates=8,
):
    """
    target region 과
      - major anatomical division
      - 비슷한 region volume
    을 맞춘 random region set 을 생성합니다.

    region 단위로 후보를 만들기 때문에 stratum 단위로 묶을 때보다
    null distribution 이 훨씬 넓어집니다.
    """

    target_ids = list(target_ids)

    candidates = {}
    relaxation = {}

    for rid in target_ids:

        choices, how = _candidates_for(
            pool, rid, log_tolerance, min_candidates
        )

        candidates[rid] = choices
        relaxation[rid] = how

    # 후보가 적은 region 부터 배정
    order = sorted(
        target_ids,
        key=lambda rid: len(candidates[rid]),
    )

    everything = np.array(pool.index)

    draws = []

    forced = 0

    for _ in range(n_permutation):

        used = set()

        for rid in order:

            available = [
                candidate
                for candidate in candidates[rid]
                if candidate not in used
            ]

            if not available:

                forced += 1

                available = [
                    candidate
                    for candidate in everything
                    if candidate not in used
                ]

            used.add(int(rng.choice(available)))

        draws.append(np.array(sorted(used)))

    info = dict(
        relaxation=relaxation,
        n_candidates={
            rid: int(len(choices))
            for rid, choices in candidates.items()
        },
        forced_draws=forced,
    )

    return draws, info


def make_enrichment(ctx=None, tables=None, show=None):

    ctx = ctx or K.load_fig1()
    tables = tables if tables is not None else K.load_reinj_tables()

    out = ctx["output"]

    rng = np.random.default_rng(C.PERMUTATION_SEED)

    pool, parents = leaf_pool(ctx, tables)

    # --------------------------------------------------------
    # target = Figure 1 M1 recipient region (및 그 descendant)
    # --------------------------------------------------------

    sets = K.region_sets(ctx)

    K.require(
        "PredefinedM1" in sets,
        "PredefinedM1 region set 을 만들 수 없습니다."
    )

    m1_ids = set(sets["PredefinedM1"]["id"].astype(int))

    target_ids = [
        rid
        for rid in pool.index
        if rid in m1_ids or (set(_ancestors(rid, parents)) & m1_ids)
    ]

    K.require(
        len(target_ids) >= 3,
        "broad region pool 안에 Figure 1 M1 recipient region 이 "
        "3개 미만입니다."
    )

    draws, info = _constrained_sets(
        pool, target_ids, C.N_PERMUTATION, rng
    )

    print()
    print("permutation 제약 상태:")

    for rid in target_ids:
        print(
            f"  {pool.at[rid, 'acronym']}: "
            f"{info['n_candidates'][rid]} candidate regions "
            f"({info['relaxation'][rid]})"
        )

    if info["forced_draws"]:
        print(
            f"  후보 고갈로 전체 pool 에서 뽑은 횟수: "
            f"{info['forced_draws']}"
        )

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
    ]).to_csv(
        out / "Reinj_permutation_constraints.csv", index=False
    )

    # --------------------------------------------------------
    # enrichment
    # --------------------------------------------------------

    pool_ids = list(pool.index)

    units = []

    for mouse, frame in tables.items():

        units.append((
            "Reinj",
            mouse,
            frame.loc[pool_ids, "whole_count"].to_numpy(dtype=float),
        ))

    cre_mice = K.cre_m1_mice(ctx)

    raw = ctx["raw"].set_index(["mouse", "id"])

    for mouse in cre_mice:

        sub = raw.xs(mouse).reindex(pool_ids)

        units.append((
            "Cre",
            mouse,
            (sub["rh_count"] + sub["lh_count"]).to_numpy(dtype=float),
        ))

    records = []
    nulls = {}

    index_of = {rid: i for i, rid in enumerate(pool_ids)}

    target_index = np.array([index_of[rid] for rid in target_ids])

    draw_index = [
        np.array([index_of[rid] for rid in draw])
        for draw in draws
    ]

    for group, mouse, counts in units:

        total = float(counts.sum())

        K.require(
            total > 0,
            f"{mouse}: broad region pool 안의 전체 count 가 0 입니다."
        )

        observed = float(counts[target_index].sum() / total)

        null = np.array([
            float(counts[index].sum() / total)
            for index in draw_index
        ])

        p_value = float(
            (1 + np.sum(null >= observed)) / (len(null) + 1)
        )

        records.append(dict(
            group=group,
            mouse=mouse,
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
            permutation_p=p_value,
            n_permutation=len(null),
        ))

        if group == "Reinj":
            nulls[mouse] = null

    enrichment_df = pd.DataFrame(records)

    enrichment_df.to_csv(
        out / "Reinj_recipient_region_enrichment.csv", index=False
    )

    pool.reset_index().to_csv(
        out / "Reinj_permutation_region_pool.csv", index=False
    )

    # --------------------------------------------------------
    # FIGURE
    # --------------------------------------------------------

    reinj_mice = list(tables)

    cre_mean = float(
        enrichment_df.loc[
            enrichment_df.group == "Cre", "enrichment"
        ].mean()
    )

    with K.rc():

        fig, axes = plt.subplots(
            1, len(reinj_mice),
            figsize=(3.3 * len(reinj_mice) + 0.4, 3.5),
            sharey=True,
            squeeze=False,
        )

        fig.subplots_adjust(
            left=0.085,
            right=0.985,
            top=0.80,
            bottom=0.27,
            wspace=0.14,
        )

        for col, mouse in enumerate(reinj_mice):

            ax = axes[0, col]

            null = nulls[mouse]

            row = enrichment_df.loc[
                (enrichment_df.group == "Reinj")
                & (enrichment_df.mouse == mouse)
            ].iloc[0]

            ax.hist(
                null,
                bins=40,
                color="#D8DCDF",
                edgecolor="white",
                linewidth=0.3,
                zorder=2,
            )

            ax.axvline(
                row["enrichment"],
                color=C.REINJ_COLOR,
                linewidth=2.0,
                zorder=4,
            )

            ax.axvline(
                cre_mean,
                color=C.SOURCE_COLORS[C.MATCHED_SOURCE],
                linewidth=1.6,
                linestyle="--",
                zorder=3,
            )

            ax.set_title(
                f"Mouse {col + 1}\n"
                f"enrichment = {row['enrichment']:.3f}, "
                f"p = {row['permutation_p']:.4f}",
                fontsize=9.5,
                fontfamily=K.RENDER_FONT,
                pad=6,
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
                    "Constrained permutations",
                    fontsize=10,
                    fontfamily=K.RENDER_FONT,
                )

        fig.text(
            0.5, 0.125,
            "Fraction of tdTomato+ cells in Figure 1 M1 recipient regions",
            ha="center",
            va="center",
            fontsize=10.5,
            fontfamily=K.RENDER_FONT,
        )

        fig.text(
            0.5, 0.955,
            "Recipient-region enrichment vs constrained region permutation\n"
            f"{len(target_ids)} target regions out of "
            f"{len(pool_ids)} non-overlapping regions, "
            f"{C.N_PERMUTATION} permutations",
            ha="center",
            va="center",
            fontsize=11,
            fontfamily=K.RENDER_FONT,
            fontweight="bold",
        )

        from matplotlib.lines import Line2D

        fig.legend(
            handles=[
                Line2D(
                    [0], [0], color=C.REINJ_COLOR, linewidth=2.0,
                    label="Observed (EV reinjection)",
                ),
                Line2D(
                    [0], [0],
                    color=C.SOURCE_COLORS[C.MATCHED_SOURCE],
                    linewidth=1.6, linestyle="--",
                    label="AAV-Cre M1 mean",
                ),
            ],
            loc="lower center",
            bbox_to_anchor=(0.5, 0.005),
            ncol=2,
            frameon=False,
            fontsize=8.5,
        )

        path = K.save_svg(
            fig,
            out / "Reinj_recipient_region_enrichment.svg",
            show=show,
        )

    print()
    print("==========================================")
    print("ENRICHMENT / PERMUTATION COMPLETE")
    print("==========================================")
    print()
    print(
        enrichment_df[[
            "group", "mouse", "enrichment", "null_mean",
            "enrichment_over_null", "permutation_p",
        ]].to_string(index=False)
    )
    print()
    print("SVG:")
    print(path)

    return dict(
        svg=str(path),
        enrichment=enrichment_df,
        pool=pool,
        target_ids=target_ids,
    )


if __name__ == "__main__":
    make_enrichment()
