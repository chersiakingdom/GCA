# ============================================================
# FIGURE 4
#
# Matched versus mismatched source analysis
#
#   rows    = intranasal EV reinjection mice
#   columns = Figure 1 Cre source references (M1 / dHP / S1 / PFC)
#
# 오른쪽 bar = margin
#   margin = min(H to mismatched references) - H(matched reference)
#   > 0 이면 자신의 source reference 가 가장 가깝다는 의미입니다.
#
# 분석 전략 문서 11번 / 5-A optional source-specificity analysis
#
# PredefinedM1 scope 는 region 자체가 M1 에서 정의되었으므로
# source 간 비교에 쓰면 circular 하여 제외합니다.
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

try:
    from . import reinj_common as K
    from . import reinj_config as C
except ImportError:
    import reinj_common as K
    import reinj_config as C


EXCLUDED_SCOPES = ("PredefinedM1",)


def make_source_specificity(ctx=None, tables=None, show=None):

    ctx = ctx or K.load_fig1()
    tables = tables if tables is not None else K.load_reinj_tables()

    out = ctx["output"]

    reinj_mice = list(tables)

    sets = {
        scope: frame
        for scope, frame in K.region_sets(ctx).items()
        if scope not in EXCLUDED_SCOPES
    }

    K.require(
        len(sets) > 0,
        "source-specificity 분석에 사용할 scope 가 없습니다."
    )

    matrices = {}
    rows = []

    for scope, frame in sets.items():

        p_cre, p_reinj, shared = K.common_feature_space(
            ctx, tables, frame["id"].tolist()
        )

        references = {}

        for source in C.SOURCE_ORDER:

            source_mice = list(ctx["cohorts"]["Cre"][source])

            K.require(
                len(source_mice) > 0,
                f"Cre / {source} cohort 가 비어 있습니다."
            )

            references[source] = p_cre.loc[source_mice].mean(axis=0)

        matrix = pd.DataFrame(
            index=reinj_mice,
            columns=C.SOURCE_ORDER,
            dtype=float,
        )

        for mouse in reinj_mice:

            for source in C.SOURCE_ORDER:

                matrix.at[mouse, source] = K.hellinger(
                    p_reinj.loc[mouse], references[source]
                )

        matrices[scope] = matrix

        for number, mouse in enumerate(reinj_mice, start=1):

            matched = float(matrix.at[mouse, C.MATCHED_SOURCE])

            mismatched = {
                source: float(matrix.at[mouse, source])
                for source in C.SOURCE_ORDER
                if source != C.MATCHED_SOURCE
            }

            nearest = min(mismatched, key=mismatched.get)

            assigned = matrix.loc[mouse].idxmin()

            rows.append(dict(
                scope=scope,
                n_regions=len(shared),
                mouse=mouse,
                mouse_number=number,
                matched_source=C.MATCHED_SOURCE,
                H_matched=matched,
                nearest_mismatched_source=nearest,
                H_nearest_mismatched=mismatched[nearest],
                margin=mismatched[nearest] - matched,
                assigned_source=assigned,
                correct=bool(assigned == C.MATCHED_SOURCE),
                **{
                    f"H_{source}": float(matrix.at[mouse, source])
                    for source in C.SOURCE_ORDER
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

    # --------------------------------------------------------
    # FIGURE
    # --------------------------------------------------------

    scopes = list(sets)

    with K.rc():

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
            left=0.14,
            right=0.95,
            top=0.88,
            bottom=0.12,
            wspace=0.30,
            hspace=0.45,
        )

        for row, scope in enumerate(scopes):

            matrix = matrices[scope]

            ax = axes[row, 0]

            image = ax.imshow(
                matrix.to_numpy(dtype=float),
                aspect="auto",
                cmap="viridis_r",
                vmin=0,
                vmax=1,
                interpolation="nearest",
            )

            ax.set_xticks(range(len(C.SOURCE_ORDER)))

            ax.set_xticklabels(
                [C.SOURCE_LABEL[s] for s in C.SOURCE_ORDER],
                fontsize=9,
            )

            for tick, source in zip(
                ax.get_xticklabels(), C.SOURCE_ORDER
            ):
                tick.set_color(C.SOURCE_COLORS[source])
                tick.set_fontweight("bold")

            ax.set_xlabel(
                "Cre reference source",
                fontsize=10,
                labelpad=7,
            )

            ax.set_yticks(range(len(matrix)))

            ax.set_yticklabels(
                [f"Mouse {i + 1}" for i in range(len(matrix))],
                fontsize=9,
            )

            ax.set_ylabel(
                "EV reinjection mice",
                fontsize=10,
                labelpad=8,
            )

            matched_column = C.SOURCE_ORDER.index(C.MATCHED_SOURCE)

            for i in range(len(matrix)):

                ax.add_patch(
                    Rectangle(
                        (matched_column - 0.5, i - 0.5),
                        1, 1,
                        fill=False,
                        edgecolor=C.MATCH_COLOR,
                        linewidth=2,
                    )
                )

                for j in range(len(C.SOURCE_ORDER)):

                    value = float(matrix.iloc[i, j])

                    ax.text(
                        j, i,
                        f"{value:.2f}",
                        ha="center",
                        va="center",
                        fontsize=8,
                        color="white" if value > 0.55 else "black",
                    )

            ax.set_title(
                "Distance to Cre source references\n"
                + C.SCOPE_LABEL[scope],
                fontsize=11,
                fontweight="bold",
                pad=10,
            )

            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            cbar = fig.colorbar(image, ax=ax, shrink=0.8, pad=0.03)

            cbar.set_label(
                "Hellinger distance",
                fontsize=9,
                labelpad=7,
            )

            cbar.ax.tick_params(labelsize=8)

            # ------------------------------------------------
            # margin bar
            # ------------------------------------------------

            ax_bar = axes[row, 1]

            subset = assignments.loc[
                assignments.scope == scope
            ].set_index("mouse").loc[matrix.index]

            margins = subset["margin"].to_numpy(dtype=float)

            y = np.arange(len(margins))

            ax_bar.barh(
                y, margins,
                height=0.5,
                color=C.REINJ_COLOR,
                edgecolor="none",
                alpha=0.92,
                zorder=3,
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
                fontsize=10,
                labelpad=7,
            )

            correct = int(subset["correct"].sum())

            ax_bar.set_title(
                "Source-reference specificity\n"
                f"matched nearest in {correct} / {len(subset)} mice",
                fontsize=11,
                fontweight="bold",
                pad=10,
            )

            ax_bar.grid(
                axis="x",
                color="#ECEEEF",
                linewidth=0.65,
                zorder=0,
            )

            ax_bar.tick_params(
                axis="x",
                length=3,
                width=0.6,
                color="#8D959A",
                pad=3,
            )

            ax_bar.spines["top"].set_visible(False)
            ax_bar.spines["right"].set_visible(False)
            ax_bar.spines["left"].set_visible(False)
            ax_bar.spines["bottom"].set_color("#A0A5A8")
            ax_bar.spines["bottom"].set_linewidth(0.7)

        path = K.save_svg(
            fig,
            out / "Reinj_source_reference_specificity.svg",
            show=show,
        )

    print()
    print("==========================================")
    print("SOURCE SPECIFICITY FIGURE COMPLETE")
    print("==========================================")
    print()
    print(
        assignments[[
            "scope", "mouse_number", "H_matched",
            "nearest_mismatched_source", "H_nearest_mismatched",
            "margin", "assigned_source", "correct",
        ]].to_string(index=False)
    )
    print()
    print("SVG:")
    print(path)

    return dict(
        svg=str(path),
        hellinger=matrices,
        assignments=assignments,
    )


if __name__ == "__main__":
    make_source_specificity()
