# ============================================================
# FIGURE 1
#
# Regional distribution heatmap
#
#   left  = AAV-Cre M1 individual mice (n=4) + mean
#   right = intranasal EV reinjection individual mice (n=3) + mean
#
# 값은 기존 figure 와 동일하게 sqrt(p) 로 표시합니다.
#
# 분석 전략 문서 III-1
# "individual biological replicate 의 pattern 도 확인 가능하도록 함"
# ============================================================

from pathlib import Path

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


def make_overview(ctx=None, tables=None, show=None):

    ctx = ctx or K.load_fig1()
    tables = tables if tables is not None else K.load_reinj_tables()

    out = ctx["output"]

    cre_mice = K.cre_m1_mice(ctx)
    reinj_mice = list(tables)

    sets = K.region_sets(ctx)

    written = []
    records = []

    for scope, frame in sets.items():

        p_cre, p_reinj, shared = K.common_feature_space(
            ctx, tables, frame["id"].tolist()
        )

        p_cre = p_cre.loc[cre_mice]
        p_reinj = p_reinj.loc[reinj_mice]

        acronyms = K.acronym_map(frame)

        labels = [acronyms.get(i, str(i)) for i in shared]

        # ----------------------------------------------------
        # 표시 순서
        #
        # Cre M1 reference 의 p 가 큰 region 부터 위에 배치
        # ----------------------------------------------------

        reference = p_cre.mean(axis=0)

        order = list(
            reference.sort_values(ascending=False).index
        )

        label_by_id = dict(zip(shared, labels))

        left = pd.concat(
            [p_cre.T, p_cre.mean(axis=0).rename("Mean").to_frame()],
            axis=1,
        ).loc[order]

        right = pd.concat(
            [p_reinj.T, p_reinj.mean(axis=0).rename("Mean").to_frame()],
            axis=1,
        ).loc[order]

        vmax = float(
            np.sqrt(
                max(
                    left.to_numpy().max(),
                    right.to_numpy().max(),
                )
            )
        )

        # ----------------------------------------------------
        # 값 저장
        # ----------------------------------------------------

        for group, table in (("Cre", left), ("Reinj", right)):

            for column in table.columns:

                for rid in order:

                    records.append(dict(
                        scope=scope,
                        group=group,
                        column=column,
                        region_id=rid,
                        acronym=label_by_id[rid],
                        p=float(table.at[rid, column]),
                        sqrt_p=float(np.sqrt(table.at[rid, column])),
                    ))

        # ----------------------------------------------------
        # FIGURE
        # ----------------------------------------------------

        n_region = len(order)

        height = max(4.2, 0.19 * n_region + 1.9)

        with K.rc():

            fig = plt.figure(figsize=(8.6, height))

            grid = fig.add_gridspec(
                1, 4,
                width_ratios=[
                    left.shape[1],
                    3.1,
                    right.shape[1],
                    0.55,
                ],
                left=0.035,
                right=0.94,
                bottom=0.09,
                top=0.80,
                wspace=0.07,
            )

            cmap = plt.get_cmap("viridis").copy()
            cmap.set_bad("#E0E0E0")

            axes = {}

            for position, (group, table) in zip(
                (0, 2),
                (("Cre", left), ("Reinj", right)),
            ):

                ax = fig.add_subplot(grid[0, position])

                values = np.sqrt(table.to_numpy(dtype=float))

                image = ax.pcolormesh(
                    np.arange(values.shape[1] + 1) - 0.5,
                    np.arange(values.shape[0] + 1) - 0.5,
                    values,
                    cmap=cmap,
                    vmin=0,
                    vmax=vmax,
                    shading="flat",
                )

                ax.set_ylim(values.shape[0] - 0.5, -0.5)

                ax.set_yticks([])

                ax.set_xticks(range(table.shape[1]))

                ax.set_xticklabels(
                    [
                        "Mean" if c == "Mean" else str(i + 1)
                        for i, c in enumerate(table.columns)
                    ],
                    fontsize=8,
                    fontfamily=K.RENDER_FONT,
                    rotation=90 if table.shape[1] > 6 else 0,
                )

                ax.tick_params(axis="both", length=0)

                # Mean column 구분선
                ax.axvline(
                    table.shape[1] - 1.5,
                    color="white",
                    lw=1.3,
                )

                color = (
                    C.SOURCE_COLORS[C.MATCHED_SOURCE]
                    if group == "Cre"
                    else C.REINJ_COLOR
                )

                ax.add_patch(
                    Rectangle(
                        (-0.5, 1.005),
                        table.shape[1],
                        0.022,
                        transform=ax.get_xaxis_transform(),
                        clip_on=False,
                        facecolor=color,
                        edgecolor="none",
                    )
                )

                ax.set_title(
                    C.GROUP_LABEL[group]
                    + f"\nn = {table.shape[1] - 1}",
                    fontsize=10.5,
                    fontfamily=K.RENDER_FONT,
                    fontweight="bold",
                    color=color,
                    pad=16,
                )

                for spine in ax.spines.values():
                    spine.set_linewidth(0.6)

                axes[group] = ax

            # ------------------------------------------------
            # Recipient region label
            # ------------------------------------------------

            label_ax = fig.add_subplot(grid[0, 1])

            label_ax.set(
                xlim=(0, 1),
                ylim=(n_region - 0.5, -0.5),
            )

            label_ax.axis("off")

            fontsize = 8.5 if n_region <= 40 else max(3.2, 330 / n_region)

            for i, rid in enumerate(order):

                label_ax.text(
                    0.5, i,
                    label_by_id[rid],
                    ha="center",
                    va="center",
                    fontsize=fontsize,
                    fontfamily=K.RENDER_FONT,
                    color="#B96524",
                )

            label_ax.set_title(
                "Recipient\nregion",
                fontsize=9,
                fontfamily=K.RENDER_FONT,
                color="#B96524",
                pad=8,
            )

            # ------------------------------------------------
            # Colorbar
            # ------------------------------------------------

            cax = fig.add_subplot(grid[0, 3])

            cbar = fig.colorbar(image, cax=cax)

            cbar.set_label(
                "√p",
                fontsize=10,
                fontfamily=K.RENDER_FONT,
            )

            cbar.ax.tick_params(labelsize=8)

            for tick in cbar.ax.get_yticklabels():
                tick.set_fontfamily(K.RENDER_FONT)

            # ------------------------------------------------
            # Scope title
            # ------------------------------------------------

            fig.text(
                0.5, 0.965,
                C.SCOPE_LABEL[scope]
                + f"  ({n_region} regions, bilateral)",
                ha="center",
                va="center",
                fontsize=11.5,
                fontfamily=K.RENDER_FONT,
                fontweight="bold",
            )

            path = K.save_svg(
                fig,
                out / f"Reinj_overview_heatmap_{scope}.svg",
                show=show,
            )

        written.append(path)

        print(f"{scope}: {n_region} regions -> {path.name}")

    pd.DataFrame(records).to_csv(
        out / "Reinj_overview_heatmap_values.csv",
        index=False,
    )

    print()
    print("==========================================")
    print("OVERVIEW HEATMAP COMPLETE")
    print("==========================================")
    print("저장 폴더:")
    print(out)

    return dict(
        output=str(out),
        svg=[str(p) for p in written],
        values=pd.DataFrame(records),
    )


if __name__ == "__main__":
    make_overview()
