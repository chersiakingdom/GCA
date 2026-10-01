# ============================================================
# FIGURE 6
#
# Figure 1 reference density 가 reinjection regional count 를
# 예측하는지에 대한 count model
#
# 분석 전략 문서 5-C
#
#   reinjection regional count
#       ~ log(Figure 1 Cre M1 reference density)
#         + mouse effect
#         + offset(log regional area)
#
# Negative binomial (NB2) 로 추정하고, 수렴하지 않으면
# Poisson 으로 대체한 뒤 그 사실을 기록합니다.
#
# 이 분석은 Hellinger 와 다른 질문에 답하므로
# secondary analysis 로 사용합니다.
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import statsmodels.api as sm

try:
    from . import reinj_common as K
    from . import reinj_config as C
    from .reinj_fig05_enrichment import leaf_pool
except ImportError:
    import reinj_common as K
    import reinj_config as C
    from reinj_fig05_enrichment import leaf_pool


def make_reference_model(ctx=None, tables=None, show=None):

    ctx = ctx or K.load_fig1()
    tables = tables if tables is not None else K.load_reinj_tables()

    out = ctx["output"]

    pool, _ = leaf_pool(ctx, tables)

    pool_ids = list(pool.index)

    cre_mice = K.cre_m1_mice(ctx)

    # --------------------------------------------------------
    # Figure 1 Cre M1 reference density
    #
    # mouse 별 p 를 먼저 구한 뒤 동일 weight 로 평균
    # (분석 전략 문서 6번)
    # --------------------------------------------------------

    p_cre = K.to_p(K.fig1_density(ctx, pool_ids, mode="pooled"))

    reference = p_cre.loc[cre_mice].mean(axis=0)

    reference = reference.reindex(pool_ids)

    positive = reference[reference > 0]

    K.require(
        len(positive) >= 10,
        "Cre M1 reference 에서 0보다 큰 region 이 10개 미만입니다."
    )

    # log 변환을 위한 pseudo-count:
    # 0 인 region 은 관측된 최소 양수값의 절반으로 대체
    floor = float(positive.min()) / 2.0

    log_reference = np.log(reference.clip(lower=floor))

    # --------------------------------------------------------
    # long format
    # --------------------------------------------------------

    rows = []

    for mouse, frame in tables.items():

        sub = frame.loc[pool_ids]

        rows.append(pd.DataFrame({
            "mouse": mouse,
            "region_id": pool_ids,
            "acronym": pool.loc[pool_ids, "acronym"].to_numpy(),
            "count": sub["whole_count"].to_numpy(dtype=float),
            "area": sub["whole_area"].to_numpy(dtype=float),
            "reference_p": reference.to_numpy(dtype=float),
            "log_reference_p": log_reference.to_numpy(dtype=float),
        }))

    data = pd.concat(rows, ignore_index=True)

    data = data.loc[data["area"] > C.AREA_ATOL].copy()

    data["observed_density"] = data["count"] / data["area"]

    # --------------------------------------------------------
    # design matrix
    # --------------------------------------------------------

    design = pd.DataFrame({
        "log_reference_p": data["log_reference_p"].to_numpy(),
    })

    mouse_list = list(tables)

    for mouse in mouse_list[1:]:

        design[f"mouse[{mouse}]"] = (
            data["mouse"] == mouse
        ).astype(float).to_numpy()

    design = sm.add_constant(design, has_constant="add")

    response = np.rint(data["count"].to_numpy(dtype=float))

    K.require(
        (response >= 0).all(),
        "count 에 음수가 있습니다."
    )

    exposure = data["area"].to_numpy(dtype=float)

    model_name = "NegativeBinomial(NB2)"

    try:

        fit = sm.NegativeBinomial(
            response,
            design,
            loglike_method="nb2",
            exposure=exposure,
        ).fit(disp=0, maxiter=200)

        K.require(
            np.isfinite(fit.params).all()
            and np.isfinite(fit.bse).all(),
            "NB 추정값이 유한하지 않습니다."
        )

    except Exception as error:  # noqa: BLE001

        print()
        print("Negative binomial 추정 실패 -> Poisson 으로 대체합니다.")
        print(f"  이유: {error}")

        model_name = "Poisson"

        fit = sm.GLM(
            response,
            design,
            family=sm.families.Poisson(),
            exposure=exposure,
        ).fit()

    coefficient = float(fit.params["log_reference_p"])

    standard_error = float(fit.bse["log_reference_p"])

    p_value = float(fit.pvalues["log_reference_p"])

    ci_low = coefficient - 1.96 * standard_error
    ci_high = coefficient + 1.96 * standard_error

    # --------------------------------------------------------
    # per-mouse Spearman (supporting)
    # --------------------------------------------------------

    per_mouse = []

    for mouse in mouse_list:

        sub = data.loc[data.mouse == mouse]

        per_mouse.append(dict(
            mouse=mouse,
            n_regions=len(sub),
            spearman_rho=K.spearman_rho(
                sub["reference_p"].to_numpy(),
                sub["observed_density"].to_numpy(),
            ),
            fraction_nonzero=float(
                (sub["count"] > 0).mean()
            ),
        ))

    per_mouse_df = pd.DataFrame(per_mouse)

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
            "coefficient > 0: Figure 1 M1 reference density 가 높은 "
            "region 일수록 reinjection 에서도 recipient cell 이 많음"
        ),
    )])

    data.to_csv(
        out / "Reinj_reference_density_model_data.csv", index=False
    )

    summary.to_csv(
        out / "Reinj_reference_density_model_summary.csv", index=False
    )

    per_mouse_df.to_csv(
        out / "Reinj_reference_density_model_per_mouse.csv", index=False
    )

    (out / "Reinj_reference_density_model_fit.txt").write_text(
        str(fit.summary()), encoding="utf-8"
    )

    # --------------------------------------------------------
    # FIGURE
    # --------------------------------------------------------

    with K.rc():

        fig, axes = plt.subplots(
            1, len(mouse_list),
            figsize=(3.3 * len(mouse_list) + 0.4, 3.6),
            sharex=True,
            sharey=True,
            squeeze=False,
        )

        fig.subplots_adjust(
            left=0.09,
            right=0.985,
            top=0.78,
            bottom=0.19,
            wspace=0.14,
        )

        for col, mouse in enumerate(mouse_list):

            ax = axes[0, col]

            sub = data.loc[data.mouse == mouse]

            x = sub["reference_p"].to_numpy(dtype=float)
            y = sub["observed_density"].to_numpy(dtype=float)

            shown = (x > 0) & (y > 0)

            ax.scatter(
                x[shown], y[shown],
                s=16,
                alpha=0.72,
                color=C.REINJ_COLOR,
                edgecolors="white",
                linewidths=0.3,
                zorder=3,
            )

            zero_y = (x > 0) & (y <= 0)

            if zero_y.any() and shown.any():

                ax.scatter(
                    x[zero_y],
                    np.full(int(zero_y.sum()), y[shown].min() * 0.5),
                    s=10,
                    marker="v",
                    color="#B9BDC0",
                    zorder=2,
                )

            ax.set_xscale("log")
            ax.set_yscale("log")

            rho = per_mouse_df.loc[
                per_mouse_df.mouse == mouse, "spearman_rho"
            ].iloc[0]

            ax.set_title(
                f"Mouse {col + 1}\n"
                + (
                    f"ρ = {rho:.2f}"
                    if np.isfinite(rho)
                    else "ρ = NA"
                ),
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
                    "Reinjection recipient-cell density\n"
                    "(cells / analyzed area)",
                    fontsize=9.5,
                    fontfamily=K.RENDER_FONT,
                )

        fig.text(
            0.5, 0.045,
            "AAV-Cre M1 reference density (p)",
            ha="center",
            va="center",
            fontsize=10.5,
            fontfamily=K.RENDER_FONT,
        )

        fig.text(
            0.5, 0.945,
            "Does Figure 1 reference density predict reinjection signal?\n"
            f"{model_name}: β(log reference) = {coefficient:.3f} "
            f"[{ci_low:.3f}, {ci_high:.3f}], p = {p_value:.3g}",
            ha="center",
            va="center",
            fontsize=10.5,
            fontfamily=K.RENDER_FONT,
            fontweight="bold",
        )

        path = K.save_svg(
            fig,
            out / "Reinj_reference_density_model.svg",
            show=show,
        )

    print()
    print("==========================================")
    print("REFERENCE DENSITY MODEL COMPLETE")
    print("==========================================")
    print()
    print(summary.drop(columns=["interpretation"]).to_string(index=False))
    print()
    print(per_mouse_df.to_string(index=False))
    print()
    print("SVG:")
    print(path)

    return dict(
        svg=str(path),
        summary=summary,
        per_mouse=per_mouse_df,
        data=data,
        fit=fit,
    )


if __name__ == "__main__":
    make_reference_model()
