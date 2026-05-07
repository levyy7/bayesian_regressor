import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import norm


def plot_evidence_maximization(
    history: pd.DataFrame,
    figsize: tuple[int, int] = (12, 4),
    save_path: str | None = None,
) -> None:
    """
    Plots evidence maximization diagnostics from the optimization history.

    Args:
        history:   DataFrame with columns 'sigma2', 'sigma2_v', 'neg_log_ev',
                   one row per optimizer iteration (captured via callback).
        figsize:   figure size
        save_path: if provided, saves the figure to this path instead of showing
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # --- Plot 1: convergence of log evidence ---
    axes[0].plot(-history["neg_log_ev"], color="steelblue", linewidth=1.5)
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Log marginal likelihood")
    axes[0].set_title("Evidence maximization convergence")
    axes[0].grid(True, alpha=0.3)

    # --- Plot 2: hyperparameter trajectories ---
    axes[1].plot(history["sigma2"],   label=r"$\sigma^2$",   color="steelblue", linewidth=1.5)
    axes[1].plot(history["sigma2_v"], label=r"$\sigma^2_v$", color="tomato",    linewidth=1.5)
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Value")
    axes[1].set_title("Hyperparameter trajectories")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()


def plot_posterior_distributions(
    top: dict,
    n_std: float = 4.0,
    figsize: tuple[int, int] = (14, 8),
    save_path: str | None = None,
) -> None:
    """
    Plots the marginal posterior distributions for the top-k predictors.

    Args:
        top:       dict returned by top_k_predictors()
        n_std:     how many std devs to plot on each side
        figsize:   figure size
        save_path: if provided, saves figure instead of showing
    """
    k = len(top["names"])
    assert k == 5, "This layout assumes exactly 5 predictors (3 top, 2 bottom)"

    # --- compute shared x limits centered at 0 ---
    half_range = max(
        abs(top["means"][i]) + n_std * top["stds"][i]
        for i in range(k)
    )
    x_min, x_max = -half_range, half_range
    x = np.linspace(x_min, x_max, 500)

    # --- layout: 3 top, 2 bottom (bottom 2 centered via GridSpec) ---
    fig = plt.figure(figsize=figsize)
    gs_top = fig.add_gridspec(2, 6, hspace=0.5, wspace=0.3)

    axes = [
        fig.add_subplot(gs_top[0, 0:2]),
        fig.add_subplot(gs_top[0, 2:4]),
        fig.add_subplot(gs_top[0, 4:6]),
        fig.add_subplot(gs_top[1, 1:3]),
        fig.add_subplot(gs_top[1, 3:5]),
    ]

    for i, ax in enumerate(axes):
        mu    = top["means"][i]
        sigma = top["stds"][i]
        lo    = top["ci_lower"][i]
        hi    = top["ci_upper"][i]
        name  = top["names"][i]

        y = norm.pdf(x, loc=mu, scale=sigma)

        ax.plot(x, y, color="steelblue", linewidth=1.8)

        ci_mask = (x >= lo) & (x <= hi)
        ax.fill_between(x, y, where=ci_mask, alpha=0.25, color="steelblue", label="95% CI")

        ax.axvline(0, color="tomato", linewidth=1.2, linestyle="--", label="0")
        ax.axvline(mu, color="steelblue", linewidth=1.0, linestyle=":")

        ax.set_xlim(x_min, x_max)
        ax.set_yticks([])

        ax.set_title(f"{name}", fontsize=12)
        ax.set_xlabel(r"$w_j$", fontsize=12)
        ax.tick_params(axis="x", labelsize=11)

        ax.annotate(
            f"μ={mu:+.4f}\nσ={sigma:.4f}",
            xy=(0.05, 0.97), xycoords="axes fraction",
            va="top", ha="left", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7)
        )

    fig.suptitle(
        f"Marginal posterior distributions — top-{k} predictors",
        fontsize=14
    )

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    else:
        plt.show()

def plot_posterior_predictive(
    y_test: np.ndarray,
    y_pred_mean: np.ndarray,
    y_pred_std: np.ndarray,
    confidence: float = 0.95,
    figsize: tuple[int, int] = (16, 6),
    save_path: str | None = None,
) -> None:
    """
    Plots posterior predictive diagnostics on the test set.

    Left plot:  predictions sorted by true value, with credible band
    Right plot: true vs predicted scatter with error bars

    Args:
        y_test:       true test targets, shape (n,)
        y_pred_mean:  posterior predictive mean, shape (n,)
        y_pred_std:   posterior predictive std, shape (n,)
        confidence:   credible interval level
        figsize:      figure size
        save_path:    if provided, saves figure instead of showing
    """
    alpha = 1 - confidence
    z = norm.ppf(1 - alpha / 2)
    ci_lower = y_pred_mean - z * y_pred_std
    ci_upper = y_pred_mean + z * y_pred_std

    # sort by true value for the left plot
    sort_idx  = np.argsort(y_test)
    y_sorted  = y_test[sort_idx]
    mu_sorted = y_pred_mean[sort_idx]
    lo_sorted = ci_lower[sort_idx]
    hi_sorted = ci_upper[sort_idx]

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    ax = axes[0]
    xs = np.arange(len(y_sorted))

    ax.fill_between(xs, lo_sorted, hi_sorted, alpha=0.25,
                    color="steelblue", label=f"{int(confidence*100)}% credible band")
    ax.plot(xs, mu_sorted, color="steelblue", linewidth=1.5, label="Predicted mean")
    ax.scatter(xs, y_sorted, color="tomato", s=8, alpha=0.6, zorder=3, label="True values")

    coverage = np.mean((y_test >= ci_lower) & (y_test <= ci_upper))
    ax.set_title(
        f"Posterior predictive confidence over sorted true value\n"
        f"Coverage: {coverage:.1%} (nominal {int(confidence*100)}%)",
        fontsize=12
    )
    ax.set_xlabel("Test samples (sorted by true target)", fontsize=12)
    ax.set_ylabel("Target", fontsize=12)
    ax.legend(fontsize=10)
    ax.tick_params(labelsize=10)

    # --- Right: true vs predicted scatter ---
    ax = axes[1]
    ax.errorbar(
        y_test, y_pred_mean,
        yerr=z * y_pred_std,
        fmt="o", markersize=3, alpha=0.4,
        color="steelblue", ecolor="steelblue", elinewidth=0.5,
        label="Predicted ± CI"
    )

    # perfect prediction line
    lims = [min(y_test.min(), y_pred_mean.min()),
            max(y_test.max(), y_pred_mean.max())]
    ax.plot(lims, lims, color="tomato", linewidth=1.5,
            linestyle="--", label="Perfect prediction")

    ax.set_title("True vs predicted", fontsize=12)
    ax.set_xlabel("True target", fontsize=12)
    ax.set_ylabel("Predicted target", fontsize=12)
    ax.legend(fontsize=10)
    ax.tick_params(labelsize=10)

    fig.suptitle("Posterior predictive distribution (test set)", fontsize=14)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    else:
        plt.show()

def plot_model_comparison(
    results: dict,
    blr_train_rmse: float,
    blr_test_rmse: float,
    figsize: tuple[int, int] = (10, 5),
    save_path: str | None = None,
) -> None:
    """
    Grouped bar chart comparing Train/Test RMSE across models.

    Args:
        results:       dict returned by create_comparison_baselines()
        blr_train_rmse:train RMSE of the BLR model
        blr_test_rmse: test RMSE of the BLR model
        figsize:       figure size
        save_path:     if provided, saves figure instead of showing
    """
    methods = ["OLS", "Ridge (CV)", "Ridge (BLR-λ)", "BLR (evidence)"]

    train_rmses = [
        results["ols"]["train_rmse"],
        results["ridge_cv"]["train_rmse"],
        results["ridge_blr"]["train_rmse"],
        blr_train_rmse
    ]
    test_rmses = [
        results["ols"]["test_rmse"],
        results["ridge_cv"]["test_rmse"],
        results["ridge_blr"]["test_rmse"],
        blr_test_rmse,
    ]

    x = np.arange(len(methods))
    width = 0.35

    fig, ax = plt.subplots(figsize=figsize)

    colors = plt.cm.coolwarm(np.linspace(0.1, 0.9, 2))

    # train bars
    train_vals = [v if v is not None else 0 for v in train_rmses]
    bars_train = ax.bar(x - width/2, train_vals, width,
                        label="Train RMSE", color=colors[0], alpha=0.85)

    # test bars
    bars_test = ax.bar(x + width/2, test_rmses, width,
                       label="Test RMSE", color=colors[1], alpha=0.85)

    # annotate exact values on top of each bar
    for bar in bars_train:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.005,
                    f"{h:.4f}", ha="center", va="bottom", fontsize=9)

    for bar in bars_test:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.005,
                f"{h:.4f}", ha="center", va="bottom", fontsize=9)

    alpha_cv  = results["ridge_cv"]["alpha"]
    alpha_blr = results["ridge_blr"]["alpha"]
    subtitles = ["", f"α={alpha_cv:.4f}", f"α={alpha_blr:.4f}", ""]
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{m}\n{s}" for m, s in zip(methods, subtitles)],
        fontsize=11
    )

    ax.set_ylabel("RMSE", fontsize=12)
    ax.set_title("Model comparison — Train vs Test RMSE", fontsize=13)
    ax.legend(fontsize=11)
    ax.set_ylim(0, max(test_rmses) * 1.2)
    ax.grid(axis="y", alpha=0.3)
    ax.tick_params(axis="y", labelsize=10)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    else:
        plt.show()

def plot_weight_comparison(
    weights: dict,
    feature_names: list[str],
    top_indices: list[int],
    figsize: tuple[int, int] = (12, 6),
    save_path: str | None = None,
) -> None:
    """
    Grouped bar chart comparing posterior/fitted weights across models
    for the top-k predictors.

    Args:
        weights:       dict {model_name: coef_array} for each model
        feature_names: full list of feature names (includes intercept)
        top_indices:   indices of predictors to show (e.g. from top_k_predictors)
        figsize:       figure size
        save_path:     if provided, saves figure instead of showing
    """
    model_names = list(weights.keys())
    n_models    = len(model_names)
    n_preds     = len(top_indices)
    pred_names  = [feature_names[i] for i in top_indices]

    x      = np.arange(n_preds)
    width  = 0.8 / n_models
    colors  = plt.cm.coolwarm(np.linspace(0.1, 0.9, len(model_names)))

    fig, ax = plt.subplots(figsize=figsize)

    for k, (model_name, color) in enumerate(zip(model_names, colors)):
        coefs  = np.array(weights[model_name])
        vals   = coefs[top_indices]
        offset = (k - n_models / 2 + 0.5) * width
        bars   = ax.bar(x + offset, vals, width,
                        label=model_name, color=color, alpha=0.85)

        # annotate values
        for bar in bars:
            h = bar.get_height()
            va = "bottom" if h >= 0 else "top"
            ax.text(bar.get_x() + bar.get_width() / 2,
                    h + (0.003 if h >= 0 else -0.003),
                    f"{h:+.3f}", ha="center", va=va, fontsize=8)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="-")
    ax.set_xticks(x)
    ax.set_xticklabels(pred_names, fontsize=11, rotation=15, ha="right")
    ax.set_ylabel("Weight value", fontsize=12)
    ax.set_title("Weight comparison across models — top predictors", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.tick_params(axis="y", labelsize=10)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    else:
        plt.show()

def plot_prior_sensitivity_weights(
    sens: dict,
    top: dict,
    sigma2_v_opt: float,
    sv_values: list[float] | None = None,
    n_show: int = 4,
    figsize: tuple[int, int] = (12, 6),
    save_path: str | None = None,
) -> None:
    """
    Grouped bar chart showing top-k predictor weights at n_show representative
    sigma2_v values (including the optimal), spanning different orders of magnitude.

    Args:
        sens:         dict returned by prior_sensitivity_analysis()
        top:          dict returned by top_k_predictors() at optimal hyperparams
        sigma2_v_opt: optimal sigma2_v (always included)
        n_show:       number of sigma2_v values to show (default 4)
        figsize:      figure size
        save_path:    if provided, saves figure instead of showing
    """
    grid       = sens["sigma2_v_grid"]
    pred_names = top["names"]
    n_preds    = len(pred_names)

    if sv_values is not None:
        chosen_idx = [int(np.argmin(np.abs(grid - sv))) for sv in sv_values]
    else:
        # fallback: auto-select spanning orders of magnitude
        idx_opt = int(np.argmin(np.abs(grid - sigma2_v_opt)))
        n_others = n_show - 1
        candidates = [i for i in np.linspace(0, len(grid) - 1, 20, dtype=int) if i != idx_opt]
        step = max(1, len(candidates) // n_others)
        chosen_idx = sorted(set(candidates[::step][:n_others] + [idx_opt]))[:n_show]

    idx_opt = int(np.argmin(np.abs(grid - sigma2_v_opt)))
    sv_labels = [
        f"σ²_v={grid[i]:.2g}" + (" *" if i == idx_opt else "")
        for i in chosen_idx
    ]

    x       = np.arange(n_preds)
    width   = 0.8 / n_show
    colors  = plt.cm.coolwarm(np.linspace(0.1, 0.9, n_show))

    fig, ax = plt.subplots(figsize=figsize)

    for k, (idx, label, color) in enumerate(zip(chosen_idx, sv_labels, colors)):
        vals   = sens["tracked_means"][idx, :]
        offset = (k - n_show / 2 + 0.5) * width
        bars   = ax.bar(x + offset, vals, width,
                        label=label, color=color, alpha=0.88)

        for bar in bars:
            h  = bar.get_height()
            va = "bottom" if h >= 0 else "top"
            ax.text(bar.get_x() + bar.get_width() / 2,
                    h + (0.003 if h >= 0 else -0.003),
                    f"{h:+.3f}", ha="center", va=va, fontsize=8)

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(pred_names, fontsize=11, rotation=15, ha="right")
    ax.set_ylabel("Posterior mean weight", fontsize=12)
    ax.set_title("Prior sensitivity on top predictor weights", fontsize=13)
    ax.legend(fontsize=10, title="σ²_v  (* = optimal)")
    ax.grid(axis="y", alpha=0.3)
    ax.tick_params(axis="y", labelsize=10)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    else:
        plt.show()


def plot_sensitivity_metrics(
    sens: dict,
    sigma2_v_opt: float,
    figsize: tuple[int, int] = (15, 5),
    save_path: str | None = None,
) -> None:
    """
    Three-panel plot of prior sensitivity metrics vs sigma2_v (log scale).
        Left:   Test RMSE
        Center: ||mu_n||_2
        Right:  Mean predictive standard deviation

    Args:
        sens:         dict returned by prior_sensitivity_analysis()
        sigma2_v_opt: optimal sigma2_v (marked as vertical line in all panels)
        figsize:      figure size
        save_path:    if provided, saves figure instead of showing
    """
    grid = sens["sigma2_v_grid"]

    metrics = [
        (sens["test_rmse"],     "Test RMSE",                   r"RMSE"),
        (sens["norm_mu_n"],     r"$\|\mu_n\|_2$",              r"$\|\mu_n\|_2$"),
        (sens["mean_pred_sd"],  "Mean predictive std",         r"Mean $\sigma_*$"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    for ax, (values, title, ylabel) in zip(axes, metrics):
        ax.plot(grid, values, color="steelblue", linewidth=1.8,
                marker="o", markersize=4)

        ax.axvline(sigma2_v_opt, color="tomato", linewidth=1.4,
                   linestyle="--", label=fr"$\sigma^2_v$ optimal ({sigma2_v_opt:.4g})")

        ax.set_xscale("log")
        ax.set_xlabel(r"$\sigma^2_v$", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=13)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=10)

    fig.suptitle("Prior sensitivity analysis", fontsize=14)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    else:
        plt.show()