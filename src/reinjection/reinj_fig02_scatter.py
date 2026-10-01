# ============================================================
# FIGURE 2
#
# Spearman scatter plot
#
#   x = AAV-Cre M1 reference (mean p of n = 4)
#   y = intranasal EV reinjection mouse (p)
#
#   rows    = Reinj mouse 1..3 + Mean
#   columns = region scope
#
# 분석 전략 문서 III-3
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

try:
    from . import reinj_common as K
    from . import reinj_config as C
except ImportError:
    import reinj_common as K
    import reinj_config as C


# 둘 중 하나라도 0보다 크면 표시 (기존 코드와 동일)
ZERO_RULE = "either_positive"


def make_scatter(ctx=None, tables=None, show=None, panel_inch=2.75):

    ctx = ctx or K.load_fig1()
    tables = tables if tables is not None else K.load_reinj_tables()

    out = ctx["output"]

    cre_mice = K.cre_m1_mice(ctx)
    reinj_mice = list(tables)

    sets = K.region_sets(ctx)

    scopes = list(sets)

    panels = {}
    point_records = []
    metrics = []

    for scope, frame in sets.items():

        p_cre, p_reinj, shared = K.common_feature_space(
            ctx, tables, frame["id"].tolist()
        )

        acronyms = K.acronym_map(frame)

        reference = p_cre.loc[cre_mice].mean(axis=0)

        targets = [
            (mouse, "individual", index, p_reinj.loc[mouse])
            for index, mouse in enumerate(reinj_mice)
        ]

        targets.append((
            "Mean",
            "mean",
            len(reinj_mice),
            p_reinj.loc[reinj_mice].mean(axis=0),
        ))

        for label, comparison, row, target in targets:

            a = reference.to_numpy(dtype=float)
            b = target.to_numpy(dtype=float)

            keep = (
                (a > 0) | (b > 0)
                if ZERO_RULE == "either_positive"
                else (a > 0) & (b > 0)
            )

            points = pd.DataFrame({
                "scope": scope,
                "region_id": shared,
                "acronym": [acronyms.get(i, str(i)) for i in shared],
                "comparison": comparison,
                "reinj_target": label,
                "p_Cre_M1_reference": a,
                "p_reinjection": b,
                "sqrt_p_Cre_M1_reference": np.sqrt(a),
                "sqrt_p_reinjection": np.sqrt(b),
                "included": keep,
            })

            point_records.append(points)

            metric = dict(
                scope=scope,
                comparison=comparison,
                reinj_target=label,
                n_Cre_mice=len(cre_mice),
                n_reinj_mice=(
                    len(reinj_mice) if comparison == "mean" else 1
                ),
                n_regions=len(shared),
                n_shown=int(keep.sum()),
                spearman_rho=K.spearman_rho(a[keep], b[keep]),
                hellinger=K.hellinger(a, b),
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

    # --------------------------------------------------------
    # scope 별 공통 축 범위
    # --------------------------------------------------------

    upper = {}

    for scope in scopes:

        subset = points_df.loc[points_df.scope == scope]

        maximum = subset[[
            "sqrt_p_Cre_M1_reference",
            "sqrt_p_reinjection",
        ]].to_numpy().max()

        upper[scope] = max(float(maximum), 0.05) * 1.06

    # --------------------------------------------------------
    # FIGURE
    # --------------------------------------------------------

    n_rows = len(reinj_mice) + 1
    n_cols = len(scopes)

    with K.rc():

        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(n_cols * panel_inch, n_rows * panel_inch),
            squeeze=False,
        )

        fig.subplots_adjust(
            left=0.085,
            right=0.985,
            top=0.90,
            bottom=0.085,
            wspace=0.26,
            hspace=0.36,
        )

        for col, scope in enumerate(scopes):

            position = axes[0, col].get_position()

            fig.text(
                (position.x0 + position.x1) / 2,
                0.955,
                C.SCOPE_LABEL[scope],
                ha="center",
                va="center",
                fontsize=11,
                fontfamily=K.RENDER_FONT,
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
                    [0, limit], [0, limit],
                    "--",
                    color="#B5B5B5",
                    lw=0.8,
                    zorder=1,
                )

                selected = data.loc[data.included]

                ax.scatter(
                    selected["sqrt_p_Cre_M1_reference"],
                    selected["sqrt_p_reinjection"],
                    s=32 if is_mean else 25,
                    alpha=0.82,
                    color=C.REINJ_COLOR,
                    edgecolors="white",
                    linewidths=0.35,
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
                        f"ρ = {rho:.2f}"
                        if np.isfinite(rho)
                        else "ρ = NA"
                    )
                    + f"   H = {metric['hellinger']:.2f}",
                    fontsize=9.5,
                    fontfamily=K.RENDER_FONT,
                    fontweight="bold" if is_mean else "normal",
                    pad=5,
                )

        fig.text(
            0.5, 0.022,
            "AAV-Cre M1 reference (√p)",
            ha="center",
            va="center",
            fontsize=11,
            fontfamily=K.RENDER_FONT,
        )

        fig.text(
            0.022, 0.5,
            "Intranasal EV reinjection (√p)",
            ha="center",
            va="center",
            rotation=90,
            fontsize=11,
            fontfamily=K.RENDER_FONT,
        )

        path = K.save_svg(
            fig,
            out / "Reinj_vs_CreM1_scatter.svg",
            show=show,
        )

    print()
    print("==========================================")
    print("SCATTER FIGURE COMPLETE")
    print("==========================================")
    print()
    print(
        metrics_df[[
            "scope", "reinj_target", "n_regions",
            "n_shown", "spearman_rho", "hellinger",
        ]].to_string(index=False)
    )
    print()
    print("SVG:")
    print(path)

    return dict(
        svg=str(path),
        points=points_df,
        metrics=metrics_df,
    )


if __name__ == "__main__":
    make_scatter()
