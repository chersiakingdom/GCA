# ============================================================
# FIGURE 7
#
# QC: signal 및 coverage, 그리고 laterality
#
# 분석 전략 문서 5 Step 0, 4번(hemisphere), IV-2
#
#   (a) 분석에 포함된 region 전체의 tdTomato+ cell 수
#   (b) signal 이 존재하는 region 의 비율 (coverage)
#   (c) 상위 region 누적 점유율 (distribution 이 몇 개
#       hotspot 에 의해 결정되는지 확인)
#   (d) laterality index
#
# (a) 는 absolute amount, (c)/(d) 는 pattern 에 관한 값이므로
# Hellinger 분석과 분리해서 해석합니다.
#
# 주의:
# 이 코드에는 negative control (vehicle / naive plasma /
# EV-depleted plasma 등) 데이터가 포함되어 있지 않습니다.
# 따라서 (a) 는 "signal > negative control" 을 검정하는 것이
# 아니라 두 group 의 절대량을 기술적으로 비교하는 값입니다.
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from . import reinj_common as K
    from . import reinj_config as C
    from .reinj_fig05_enrichment import leaf_pool
except ImportError:
    import reinj_common as K
    import reinj_config as C
    from reinj_fig05_enrichment import leaf_pool


TOP_N = 10


def make_qc(ctx=None, tables=None, show=None):

    ctx = ctx or K.load_fig1()
    tables = tables if tables is not None else K.load_reinj_tables()

    out = ctx["output"]

    pool, _ = leaf_pool(ctx, tables)

    pool_ids = list(pool.index)

    cre_mice = K.cre_m1_mice(ctx)

    raw = ctx["raw"].set_index(["mouse", "id"])

    records = []
    curves = {}

    def add_unit(group, label, count, area, first_side, second_side):

        total = float(np.nansum(count))

        K.require(
            total > 0,
            f"{label}: 분석 region 전체 count 가 0 입니다."
        )

        density = np.where(area > C.AREA_ATOL, count / np.where(
            area > C.AREA_ATOL, area, np.nan
        ), np.nan)

        p = np.nan_to_num(density, nan=0.0)

        p = p / p.sum()

        order = np.sort(p)[::-1]

        curves[(group, label)] = np.cumsum(order)

        records.append(dict(
            group=group,
            label=label,
            n_regions=len(pool_ids),
            total_count=total,
            n_regions_with_signal=int(np.sum(count > 0)),
            coverage=float(np.mean(count > 0)),
            top_region_share=float(order[0]),
            top_n_share=float(order[:TOP_N].sum()),
            laterality_index=(
                float(np.nansum(first_side) / total)
                if total > 0 else np.nan
            ),
            second_side_count=float(np.nansum(second_side)),
        ))

    for mouse, frame in tables.items():

        sub = frame.loc[pool_ids]

        add_unit(
            "Reinj",
            mouse,
            sub["whole_count"].to_numpy(dtype=float),
            sub["whole_area"].to_numpy(dtype=float),
            sub["rh_count"].to_numpy(dtype=float),
            sub["lh_count"].to_numpy(dtype=float),
        )

    for mouse in cre_mice:

        sub = raw.xs(mouse).reindex(pool_ids)

        side = str(
            ctx["manifest"].at[mouse, "injection_side"]
        ).strip().upper()

        ipsi = "rh" if side == "R" else "lh"
        contra = "lh" if side == "R" else "rh"

        add_unit(
            "Cre",
            mouse,
            (sub[f"{ipsi}_count"] + sub[f"{contra}_count"]).to_numpy(float),
            (sub[f"{ipsi}_area"] + sub[f"{contra}_area"]).to_numpy(float),
            sub[f"{ipsi}_count"].to_numpy(float),
            sub[f"{contra}_count"].to_numpy(float),
        )

    qc = pd.DataFrame(records)

    qc.to_csv(out / "Reinj_QC_signal_and_coverage.csv", index=False)

    # --------------------------------------------------------
    # FIGURE
    # --------------------------------------------------------

    color_of = {
        "Cre": C.SOURCE_COLORS[C.MATCHED_SOURCE],
        "Reinj": C.REINJ_COLOR,
    }

    groups = ["Cre", "Reinj"]

    with K.rc():

        fig, axes = plt.subplots(
            1, 4,
            figsize=(14.0, 3.6),
            squeeze=False,
        )

        fig.subplots_adjust(
            left=0.055,
            right=0.99,
            top=0.78,
            bottom=0.20,
            wspace=0.34,
        )

        def strip(ax, column, log=False, hline=None, ylabel=""):

            for index, group in enumerate(groups):

                values = qc.loc[qc.group == group, column].to_numpy(float)

                x = (
                    np.array([float(index)])
                    if len(values) == 1
                    else index + np.linspace(-0.11, 0.11, len(values))
                )

                ax.scatter(
                    x, values,
                    s=46,
                    facecolors=color_of[group],
                    edgecolors="white",
                    linewidths=0.55,
                    zorder=4,
                )

                ax.hlines(
                    float(np.mean(values)),
                    index - 0.2, index + 0.2,
                    color=color_of[group],
                    linewidth=2.2,
                    zorder=5,
                )

            if hline is not None:

                ax.axhline(
                    hline,
                    color="#999999",
                    linewidth=0.9,
                    linestyle="--",
                    zorder=1,
                )

            if log:
                ax.set_yscale("log")

            ax.set_xticks(range(len(groups)))

            ax.set_xticklabels(
                [
                    "AAV-Cre M1\n(n = "
                    + str(int((qc.group == "Cre").sum()))
                    + ")",
                    "EV reinjection\n(n = "
                    + str(int((qc.group == "Reinj").sum()))
                    + ")",
                ],
                fontsize=8.3,
                fontfamily=K.RENDER_FONT,
            )

            ax.set_xlim(-0.45, len(groups) - 0.55)

            ax.set_ylabel(
                ylabel,
                fontsize=9.5,
                fontfamily=K.RENDER_FONT,
            )

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
            axes[0, 0], "total_count",
            log=True,
            ylabel="tdTomato+ cells in analyzed regions",
        )

        axes[0, 0].set_title(
            "Absolute signal",
            fontsize=10.5,
            fontweight="bold",
            pad=7,
        )

        strip(
            axes[0, 1], "coverage",
            ylabel="Fraction of regions with ≥ 1 cell",
        )

        axes[0, 1].set_ylim(-0.03, 1.03)

        axes[0, 1].set_title(
            "Spatial coverage",
            fontsize=10.5,
            fontweight="bold",
            pad=7,
        )

        # ----------------------------------------------------
        # (c) 누적 점유율
        # ----------------------------------------------------

        ax = axes[0, 2]

        for (group, label), curve in curves.items():

            ax.plot(
                np.arange(1, len(curve) + 1),
                curve,
                color=color_of[group],
                linewidth=1.2,
                alpha=0.85,
                zorder=3,
            )

        ax.set_xscale("log")

        ax.set_ylim(0, 1.03)

        ax.axhline(
            0.5, color="#999999", linewidth=0.9,
            linestyle="--", zorder=1,
        )

        ax.set_xlabel(
            "Number of regions (ranked)",
            fontsize=9.5,
            fontfamily=K.RENDER_FONT,
        )

        ax.set_ylabel(
            "Cumulative share of p",
            fontsize=9.5,
            fontfamily=K.RENDER_FONT,
        )

        ax.set_title(
            "Distribution concentration",
            fontsize=10.5,
            fontweight="bold",
            pad=7,
        )

        ax.grid(color="#ECEEEF", linewidth=0.65, zorder=0)

        ax.tick_params(
            labelsize=8, length=3, width=0.6,
            color="#8D959A", pad=2,
        )

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        for name in ("left", "bottom"):
            ax.spines[name].set_color("#A0A5A8")
            ax.spines[name].set_linewidth(0.7)

        # ----------------------------------------------------
        # (d) laterality
        # ----------------------------------------------------

        strip(
            axes[0, 3], "laterality_index",
            hline=0.5,
            ylabel="Cre: ipsi / total   ·   Reinj: RH / total",
        )

        axes[0, 3].set_ylim(0, 1.03)

        axes[0, 3].set_title(
            "Laterality",
            fontsize=10.5,
            fontweight="bold",
            pad=7,
        )

        fig.text(
            0.5, 0.955,
            "Signal, coverage and laterality QC   "
            f"({len(pool_ids)} non-overlapping regions)",
            ha="center",
            va="center",
            fontsize=11.5,
            fontfamily=K.RENDER_FONT,
            fontweight="bold",
        )

        path = K.save_svg(
            fig,
            out / "Reinj_QC_signal_coverage_laterality.svg",
            show=show,
        )

    print()
    print("==========================================")
    print("QC FIGURE COMPLETE")
    print("==========================================")
    print()
    print(
        qc[[
            "group", "label", "total_count", "coverage",
            "top_n_share", "laterality_index",
        ]].to_string(index=False)
    )
    print()
    print("SVG:")
    print(path)

    return dict(svg=str(path), qc=qc)


if __name__ == "__main__":
    make_qc()
