# ============================================================
# FIGURE 3
#
# Hellinger distance plot
#
# 같은 scale 위에 세 가지를 함께 표시합니다.
#
#   ●  reinjection mouse -> AAV-Cre M1 reference (n = 4 mean)
#   ○  Cre M1 mouse      -> leave-one-out Cre M1 reference
#   ◇  Cre M1 mouse      <-> Cre M1 mouse (pairwise)
#
# 뒤의 두 값이 분석 전략 문서 9번의
# natural inter-animal variability benchmark 입니다.
#
# Figure 1 pairwise 값은 mouse 를 서로 공유하므로
# independent biological replicate 로 간주하지 않으며
# t-test 등을 수행하지 않습니다.
# ============================================================

from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

try:
    from . import reinj_common as K
    from . import reinj_config as C
except ImportError:
    import reinj_common as K
    import reinj_config as C


COLUMNS = [
    ("reinj_to_reference", "EV reinjection →\nCre M1 reference"),
    ("cre_loo", "Cre M1 →\nLOO reference"),
    ("cre_pairwise", "Cre M1 ↔\nCre M1"),
]


def make_hellinger(ctx=None, tables=None, show=None):

    ctx = ctx or K.load_fig1()
    tables = tables if tables is not None else K.load_reinj_tables()

    out = ctx["output"]

    cre_mice = K.cre_m1_mice(ctx)
    reinj_mice = list(tables)

    sets = K.region_sets(ctx)
    scopes = list(sets)

    records = []
    summary = []

    for scope, frame in sets.items():

        p_cre, p_reinj, shared = K.common_feature_space(
            ctx, tables, frame["id"].tolist()
        )

        p_cre = p_cre.loc[cre_mice]

        reference = p_cre.mean(axis=0)

        # ----------------------------------------------------
        # reinjection -> Cre M1 reference
        # ----------------------------------------------------

        for number, mouse in enumerate(reinj_mice, start=1):

            records.append(dict(
                scope=scope,
                comparison="reinj_to_reference",
                label=f"Mouse {number}",
                unit=mouse,
                hellinger=K.hellinger(p_reinj.loc[mouse], reference),
            ))

        # ----------------------------------------------------
        # Cre M1 -> leave-one-out reference
        # ----------------------------------------------------

        for number, mouse in enumerate(cre_mice, start=1):

            others = [m for m in cre_mice if m != mouse]

            records.append(dict(
                scope=scope,
                comparison="cre_loo",
                label=f"Mouse {number}",
                unit=mouse,
                hellinger=K.hellinger(
                    p_cre.loc[mouse],
                    p_cre.loc[others].mean(axis=0),
                ),
            ))

        # ----------------------------------------------------
        # Cre M1 pairwise
        # ----------------------------------------------------

        for first, second in combinations(cre_mice, 2):

            records.append(dict(
                scope=scope,
                comparison="cre_pairwise",
                label=f"{first} / {second}",
                unit=f"{first}|{second}",
                hellinger=K.hellinger(
                    p_cre.loc[first], p_cre.loc[second]
                ),
            ))

        # ----------------------------------------------------
        # summary
        # ----------------------------------------------------

        table = pd.DataFrame(
            [r for r in records if r["scope"] == scope]
        )

        values = {
            key: table.loc[
                table.comparison == key, "hellinger"
            ].to_numpy()
            for key, _ in COLUMNS
        }

        summary.append(dict(
            scope=scope,
            n_regions=len(shared),
            n_reinj=len(values["reinj_to_reference"]),
            n_Cre=len(cre_mice),
            reinj_mean=float(np.mean(values["reinj_to_reference"])),
            reinj_min=float(np.min(values["reinj_to_reference"])),
            reinj_max=float(np.max(values["reinj_to_reference"])),
            cre_loo_mean=float(np.mean(values["cre_loo"])),
            cre_loo_min=float(np.min(values["cre_loo"])),
            cre_loo_max=float(np.max(values["cre_loo"])),
            cre_pairwise_mean=float(np.mean(values["cre_pairwise"])),
            cre_pairwise_min=float(np.min(values["cre_pairwise"])),
            cre_pairwise_max=float(np.max(values["cre_pairwise"])),
            Delta_H_reinj_minus_CreLOO=float(
                np.mean(values["reinj_to_reference"])
                - np.mean(values["cre_loo"])
            ),
            n_reinj_within_Cre_pairwise_range=int(
                np.sum(
                    values["reinj_to_reference"]
                    <= np.max(values["cre_pairwise"])
                )
            ),
        ))

    distance_df = pd.DataFrame(records)
    summary_df = pd.DataFrame(summary)

    distance_df.to_csv(
        out / "Reinj_Hellinger_distances.csv", index=False
    )

    summary_df.to_csv(
        out / "Reinj_Hellinger_summary.csv", index=False
    )

    # --------------------------------------------------------
    # FIGURE
    # --------------------------------------------------------

    with K.rc():

        fig, axes = plt.subplots(
            1, len(scopes),
            figsize=(3.6 * len(scopes) + 0.6, 4.6),
            sharey=True,
            squeeze=False,
        )

        fig.subplots_adjust(
            left=0.085,
            right=0.985,
            top=0.86,
            bottom=0.255,
            wspace=0.16,
        )

        color = C.REINJ_COLOR
        cre_color = C.SOURCE_COLORS[C.MATCHED_SOURCE]

        styles = {
            "reinj_to_reference": dict(
                facecolors=color, edgecolors="white",
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
            "reinj_to_reference": color,
            "cre_loo": cre_color,
            "cre_pairwise": cre_color,
        }

        for col, scope in enumerate(scopes):

            ax = axes[0, col]

            subset = distance_df.loc[distance_df.scope == scope]

            for index, (key, _) in enumerate(COLUMNS):

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
                    index - 0.19,
                    index + 0.19,
                    color=bar_color[key],
                    linewidth=2.2,
                    zorder=5,
                )

            ax.set_xticks(range(len(COLUMNS)))

            ax.set_xticklabels(
                [label for _, label in COLUMNS],
                fontsize=8.3,
                fontfamily=K.RENDER_FONT,
            )

            ax.set_xlim(-0.45, len(COLUMNS) - 0.55)

            ax.set_ylim(-0.02, 1.02)

            ax.set_yticks(np.arange(0, 1.01, 0.2))

            ax.tick_params(
                axis="both",
                labelsize=8,
                length=3,
                width=0.6,
                color="#8D959A",
                pad=2,
            )

            ax.grid(
                axis="y",
                color="#ECEEEF",
                linewidth=0.65,
                zorder=-1,
            )

            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            for name in ("left", "bottom"):
                ax.spines[name].set_color("#A0A5A8")
                ax.spines[name].set_linewidth(0.7)

            n_region = int(
                summary_df.loc[
                    summary_df.scope == scope, "n_regions"
                ].iloc[0]
            )

            ax.set_title(
                C.SCOPE_LABEL[scope] + f"\n{n_region} regions",
                fontsize=11,
                fontfamily=K.RENDER_FONT,
                fontweight="bold",
                pad=8,
            )

        fig.text(
            0.018, 0.56,
            "Hellinger distance",
            rotation=90,
            ha="center",
            va="center",
            fontsize=11,
            fontfamily=K.RENDER_FONT,
        )

        handles = [
            Line2D(
                [0], [0], marker="o", linestyle="",
                markerfacecolor=color, markeredgecolor="white",
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
            Line2D(
                [0], [0], color="#666666", linewidth=2.2,
                label="Mean",
            ),
        ]

        fig.legend(
            handles=handles,
            loc="lower center",
            bbox_to_anchor=(0.53, 0.0),
            ncol=2,
            frameon=False,
            fontsize=8.5,
            handletextpad=0.55,
            columnspacing=1.5,
        )

        path = K.save_svg(
            fig,
            out / "Reinj_Hellinger_reference_comparison.svg",
            show=show,
        )

    print()
    print("==========================================")
    print("HELLINGER FIGURE COMPLETE")
    print("==========================================")
    print()
    print(
        summary_df[[
            "scope", "n_regions", "reinj_mean",
            "cre_loo_mean", "cre_pairwise_mean",
            "Delta_H_reinj_minus_CreLOO",
            "n_reinj_within_Cre_pairwise_range",
        ]].to_string(index=False)
    )
    print()
    print("SVG:")
    print(path)

    return dict(
        svg=str(path),
        distances=distance_df,
        summary=summary_df,
    )


if __name__ == "__main__":
    make_hellinger()
