"""CreditGuard dashboard — interactive walkthrough of a real credit-risk model.

Reads from artifacts/ (produced by training + evaluation pipelines).
Run:    streamlit run streamlit_app.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

ARTIFACTS = Path(__file__).parent / "artifacts"

BASE_RATE = 0.0807
N_ROWS = 307_511
FP_COST = 600
FN_COST = 6_000


st.set_page_config(
    page_title="CreditGuard — Credit Risk Decisioning",
    page_icon=":bar_chart:",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data
def load_eval() -> dict:
    return json.loads((ARTIFACTS / "evaluation_results.json").read_text())


@st.cache_data
def load_shap() -> pd.DataFrame:
    return pd.read_csv(ARTIFACTS / "shap_mean_abs.csv")


@st.cache_data
def load_fairness() -> pd.DataFrame:
    return pd.read_csv(ARTIFACTS / "fairness_threshold_sweep.csv")


@st.cache_data
def load_ablation() -> dict:
    return json.loads((ARTIFACTS / "ablation_no_bureau_summary.json").read_text())


@st.cache_data
def load_oof() -> pd.DataFrame:
    return pd.read_parquet(ARTIFACTS / "oof_predictions_scored.parquet")


@st.cache_data
def compute_roc(y_true: np.ndarray, y_score: np.ndarray) -> pd.DataFrame:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return pd.DataFrame({"FPR": fpr, "TPR": tpr})


@st.cache_data
def compute_pr(y_true: np.ndarray, y_score: np.ndarray) -> pd.DataFrame:
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    return pd.DataFrame({"Recall": recall, "Precision": precision})


@st.cache_data
def compute_lift_curve(y_true: np.ndarray, y_score: np.ndarray, n_bins: int = 100) -> pd.DataFrame:
    order = np.argsort(-y_score)
    sorted_y = y_true[order]
    base = sorted_y.mean()
    rows = []
    n = len(sorted_y)
    for k in range(1, n_bins + 1):
        cutoff = int(np.ceil(n * k / n_bins))
        topk = sorted_y[:cutoff]
        lift = topk.mean() / base if base else 0.0
        rows.append({"Top k%": k, "Cumulative lift": float(lift)})
    return pd.DataFrame(rows)


@st.cache_data
def compute_cost_curve(
    y_true: np.ndarray, y_score: np.ndarray, fp_cost: int, fn_cost: int, n_bins: int = 80
) -> pd.DataFrame:
    thresholds = np.linspace(0.05, 0.95, n_bins)
    rows = []
    for t in thresholds:
        y_pred = (y_score >= t).astype(int)
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())
        cost = fp * fp_cost + fn * fn_cost
        rows.append({"Threshold": float(t), "Expected cost": float(cost)})
    return pd.DataFrame(rows)


@st.cache_data
def compute_calibration_curve(
    y_true: np.ndarray, y_score: np.ndarray, n_bins: int = 10
) -> pd.DataFrame:
    bins = np.linspace(0, 1, n_bins + 1)
    rows = []
    for i in range(n_bins):
        mask = (y_score >= bins[i]) & (y_score < bins[i + 1])
        if mask.sum() == 0:
            continue
        rows.append(
            {
                "Predicted": float(y_score[mask].mean()),
                "Observed": float(y_true[mask].mean()),
                "n": int(mask.sum()),
            }
        )
    return pd.DataFrame(rows)


eval_data = load_eval()
shap_df = load_shap()
fairness_df = load_fairness()
ablation = load_ablation()
oof = load_oof()
y_true = oof["y_true"].values
y_score = oof["y_score"].values


st.sidebar.title("CreditGuard")
st.sidebar.caption("Credit default early-warning + decisioning")

st.sidebar.markdown(
    """
**Champion:** LightGBM (investigate — see Overview)
**Challenger:** XGBoost (rejected)

See the Overview tab for the full side-by-side comparison and governance gate outcomes.
"""
)

st.sidebar.divider()
with st.sidebar.expander("Pipeline at a glance", expanded=True):
    st.markdown(
        f"""
- **Dataset:** Home Credit Default Risk
- **Rows:** {N_ROWS:,} applicants
- **Default rate:** {BASE_RATE:.2%}
- **Features:** ~80 engineered -> 15 selected
- **Selection:** Boruta (200 iter) -> VIF (<=10) (RF ranks survivors; no filtering)
- **CV:** 5-fold StratifiedGroupKFold
- **HPO:** Optuna, 50 trials, PR-AUC
- **Cost matrix:** FP=\\${FP_COST}, FN=\\${FN_COST}
"""
    )


# All downstream tabs focus on the champion; Overview tab shows champion vs challenger side-by-side.
model_key = "champion"
model = eval_data[model_key]


st.title("CreditGuard — Credit Risk Decisioning")
st.caption(
    "A reference implementation on Home Credit Default Risk (Kaggle, 307,511 applicants). "
    "The Overview tab compares champion (LightGBM) and challenger (XGBoost); the rest of the "
    "dashboard drills into the deployed champion using honest out-of-fold predictions."
)

tab_overview, tab_ranking, tab_calib, tab_threshold, tab_fair, tab_shap, tab_ablation = st.tabs(
    [
        "Overview",
        "Ranking",
        "Calibration",
        "Threshold simulator",
        "Fairness",
        "Feature importance",
        "Ablation",
    ]
)


with tab_overview:
    st.subheader("Governance decision")
    decision = eval_data["comparison"]["decision"]
    # Three documented gate outcomes:
    #   promote        — challenger clears all four gates -> green
    #   keep_champion  — challenger fails on a hard gate; champion is fine -> blue
    #   investigate    — mixed result (e.g. the champion itself fails a gate) -> orange
    # Anything else falls through to red so a new value surfaces visibly.
    pill_color = {
        "promote": "green",
        "keep_champion": "blue",
        "investigate": "orange",
    }.get(decision, "red")
    st.markdown(f"## :{pill_color}[{decision.replace('_', ' ').upper()}]")
    st.caption(
        "Automated promotion outcome. Four gates must all PASS for the challenger to replace the champion."
    )

    gate_cols = st.columns(4)
    comp = eval_data["comparison"]
    gate_cols[0].metric(
        "PR-AUC gate",
        "FAIL" if comp["pr_auc_improvement"] <= 0 else "PASS",
        f"{comp['pr_auc_improvement']:+.4f}",
        delta_color="inverse" if comp["pr_auc_improvement"] <= 0 else "normal",
        help="Challenger PR-AUC must improve on champion. Primary metric under 8% base rate.",
    )
    gate_cols[1].metric(
        "Calibration gate",
        "FAIL" if comp["calibration_degradation"] > 0.02 else "PASS",
        f"{comp['calibration_degradation']:+.4f} delta",
        delta_color="inverse",
        help="Raw Brier must not degrade by more than 0.02.",
    )
    gate_cols[2].metric(
        "Fairness gate",
        "PASS" if comp["fairness_within_tolerance"] else "FAIL",
        help="Adverse Impact Ratio must stay within tolerance across sensitive attributes.",
    )
    gate_cols[3].metric(
        "Interpretability gate",
        "PASS" if comp["interpretability_acceptable"] else "FAIL",
        help="Both models must remain in 'medium' interpretability tier (tree-based, SHAP-explained).",
    )

    with st.expander("Promotion notes", expanded=False):
        for note in eval_data["comparison"]["notes"]:
            st.markdown(f"- {note}")

    st.divider()
    st.subheader("Champion vs Challenger side-by-side")

    def _rows(c: dict, ch: dict) -> list[dict]:
        return [
            {
                "Metric": "ROC-AUC",
                "LightGBM": c["metrics"]["roc_auc"],
                "XGBoost": ch["metrics"]["roc_auc"],
            },
            {
                "Metric": "PR-AUC",
                "LightGBM": c["metrics"]["pr_auc"],
                "XGBoost": ch["metrics"]["pr_auc"],
            },
            {
                "Metric": "KS",
                "LightGBM": c["metrics"]["ks_statistic"],
                "XGBoost": ch["metrics"]["ks_statistic"],
            },
            {"Metric": "Gini", "LightGBM": c["metrics"]["gini"], "XGBoost": ch["metrics"]["gini"]},
            {
                "Metric": "Brier (raw)",
                "LightGBM": c["calibration"]["brier_score"],
                "XGBoost": ch["calibration"]["brier_score"],
            },
            {
                "Metric": "Brier (Platt)",
                "LightGBM": c["calibration"]["platt_brier"],
                "XGBoost": ch["calibration"]["platt_brier"],
            },
            {
                "Metric": "Brier (Isotonic)",
                "LightGBM": c["calibration"]["isotonic_brier"],
                "XGBoost": ch["calibration"]["isotonic_brier"],
            },
            {
                "Metric": "Lift @ top 1%",
                "LightGBM": c["discrimination"]["lift_at_k"]["top_1pct"],
                "XGBoost": ch["discrimination"]["lift_at_k"]["top_1pct"],
            },
            {
                "Metric": "Lift @ top 5%",
                "LightGBM": c["discrimination"]["lift_at_k"]["top_5pct"],
                "XGBoost": ch["discrimination"]["lift_at_k"]["top_5pct"],
            },
            {
                "Metric": "Optimal threshold",
                "LightGBM": c["threshold"]["optimal_threshold"],
                "XGBoost": ch["threshold"]["optimal_threshold"],
            },
        ]

    comp_df = pd.DataFrame(_rows(eval_data["champion"], eval_data["challenger"]))
    comp_df["Delta"] = comp_df["XGBoost"] - comp_df["LightGBM"]
    st.dataframe(
        comp_df.style.format({"LightGBM": "{:.4f}", "XGBoost": "{:.4f}", "Delta": "{:+.4f}"}),
        hide_index=True,
        use_container_width=True,
    )

    st.info(
        "**Read:** LightGBM edges XGBoost on every ranking metric and has dramatically better raw "
        "Brier (0.177 vs 0.210). Both converge to ~0.068 after Platt/Isotonic, but the governance "
        "framework flags this raw-calibration gap as evidence the XGBoost probabilities are "
        "miscalibrated out of the box."
    )


with tab_ranking:
    st.subheader("Discrimination curves (OOF, N = 307,511)")
    st.caption(
        "Computed live from `artifacts/oof_predictions_scored.parquet`. "
        "OOF means every prediction here was made by a model that did NOT see that applicant in training."
    )

    roc_df = compute_roc(y_true, y_score)
    pr_df = compute_pr(y_true, y_score)
    auc_value = roc_auc_score(y_true, y_score)

    r1, r2 = st.columns(2)
    with r1:
        fig_roc = px.line(roc_df, x="FPR", y="TPR", title=f"ROC curve — AUC = {auc_value:.4f}")
        fig_roc.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dash", color="gray"))
        fig_roc.update_layout(
            height=360, xaxis_title="False positive rate", yaxis_title="True positive rate"
        )
        st.plotly_chart(fig_roc, use_container_width=True)
    with r2:
        fig_pr = px.line(
            pr_df, x="Recall", y="Precision", title=f"PR curve — base rate = {BASE_RATE:.2%}"
        )
        fig_pr.add_hline(
            y=BASE_RATE,
            line_dash="dash",
            line_color="gray",
            annotation_text="random baseline",
            annotation_position="top left",
        )
        fig_pr.update_layout(height=360)
        st.plotly_chart(fig_pr, use_container_width=True)

    st.divider()
    st.subheader("Lift curve")
    st.caption(
        "Ranking the riskiest applicants first, what's the default rate in the top-k% vs overall?"
    )
    lift_curve = compute_lift_curve(y_true, y_score, n_bins=100)
    fig_lift = px.line(lift_curve, x="Top k%", y="Cumulative lift")
    fig_lift.add_hline(y=1.0, line_dash="dash", line_color="gray", annotation_text="random = 1.0x")
    fig_lift.update_layout(
        height=320, xaxis_title="Top k% of applicants (ranked by score, highest first)"
    )
    st.plotly_chart(fig_lift, use_container_width=True)

    _top1_lift = float(lift_curve.loc[lift_curve["Top k%"] == 1, "Cumulative lift"].iloc[0])
    _top1_default_rate = _top1_lift * BASE_RATE
    st.caption(
        f"The top-1% lift of {_top1_lift:.2f}x means: if you review only the 1% riskiest applicants "
        f"the model flags, ~{_top1_default_rate:.0%} of them will default versus ~{BASE_RATE:.0%} in "
        "the overall population. This is what drives capital allocation for review-queue staffing."
    )


with tab_calib:
    st.subheader("Calibration — do predicted probabilities match observed rates?")
    calib = model["calibration"]
    c1, c2 = st.columns([1, 1])

    with c1:
        calib_df = pd.DataFrame(
            {
                "Method": ["Raw", "Platt scaling", "Isotonic regression"],
                "Brier score": [
                    calib["brier_score"],
                    calib["platt_brier"],
                    calib["isotonic_brier"],
                ],
            }
        )
        fig_calib = px.bar(
            calib_df,
            x="Method",
            y="Brier score",
            title="Brier score (lower is better)",
            text_auto=".4f",
        )
        fig_calib.update_layout(height=320)
        st.plotly_chart(fig_calib, use_container_width=True)
        st.caption(
            f"ECE (raw): {calib['ece']:.4f}. Both Platt and Isotonic recover similar quality here — "
            "in production you'd deploy with Isotonic for minor robustness."
        )

    with c2:
        calibration_curve = compute_calibration_curve(y_true, y_score, n_bins=10)
        fig_rel = go.Figure()
        fig_rel.add_trace(
            go.Scatter(
                x=calibration_curve["Predicted"],
                y=calibration_curve["Observed"],
                mode="lines+markers",
                name="Model",
                marker=dict(size=calibration_curve["n"].clip(upper=50_000) / 1200),
            )
        )
        fig_rel.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dash", color="gray"))
        fig_rel.update_layout(
            title="Reliability diagram (live, 10 equal-width bins)",
            xaxis_title="Mean predicted probability",
            yaxis_title="Observed default rate",
            height=320,
        )
        st.plotly_chart(fig_rel, use_container_width=True)
        st.caption("Marker size scales with bin count. Perfect calibration follows the diagonal.")

    with st.expander("Reference reliability diagram (from training run)"):
        reliability_png = ARTIFACTS / "reliability_lightgbm.png"
        if reliability_png.exists():
            st.image(str(reliability_png), use_container_width=True)


with tab_threshold:
    st.subheader("Three-band decisioning simulator")
    st.caption(
        "A production credit engine rarely picks one threshold — it picks TWO. An approve gate for "
        "auto-approval, a decline gate for auto-decline, and the middle band goes to human review. "
        "Slide the gates to see the implications across volume, realised default rate, expected cost, "
        "and fairness."
    )

    c1, c2 = st.columns(2)
    approve_t = c1.slider("Approve gate (score <)", 0.01, 0.50, 0.15, 0.01)
    decline_t = c2.slider("Decline gate (score >=)", 0.10, 0.95, 0.40, 0.01)

    if approve_t >= decline_t:
        st.error("Approve gate must be strictly below decline gate.")
    else:
        approve_mask = y_score < approve_t
        decline_mask = y_score >= decline_t
        review_mask = ~(approve_mask | decline_mask)
        n = len(y_score)

        m = st.columns(3)
        m[0].metric(
            "Auto-approve",
            f"{approve_mask.sum():,}",
            f"{approve_mask.mean():.1%}",
            help="Low-risk applicants auto-approved — no human review cost.",
        )
        m[1].metric(
            "Review queue",
            f"{review_mask.sum():,}",
            f"{review_mask.mean():.1%}",
            help="Sent to an analyst for manual decision.",
        )
        m[2].metric(
            "Auto-decline",
            f"{decline_mask.sum():,}",
            f"{decline_mask.mean():.1%}",
            help="High-risk applicants auto-declined.",
        )

        st.divider()
        dr1, dr2 = st.columns([1, 1])

        with dr1:
            st.markdown("**Realised default rate by band** (OOF)")
            def_df = pd.DataFrame(
                {
                    "Band": ["Auto-approve", "Review queue", "Auto-decline"],
                    "Default rate": [
                        float(y_true[approve_mask].mean()) if approve_mask.any() else 0.0,
                        float(y_true[review_mask].mean()) if review_mask.any() else 0.0,
                        float(y_true[decline_mask].mean()) if decline_mask.any() else 0.0,
                    ],
                }
            )
            fig_band = px.bar(def_df, x="Band", y="Default rate", text_auto=".2%")
            fig_band.add_hline(
                y=BASE_RATE,
                line_dash="dash",
                line_color="gray",
                annotation_text=f"base rate {BASE_RATE:.2%}",
                annotation_position="top right",
            )
            fig_band.update_layout(height=320, yaxis_tickformat=".0%")
            st.plotly_chart(fig_band, use_container_width=True)

        with dr2:
            st.markdown(f"**Confusion matrix at decline gate (t = {decline_t:.2f})**")
            y_pred = (y_score >= decline_t).astype(int)
            cm = confusion_matrix(y_true, y_pred)
            cm_df = pd.DataFrame(
                cm,
                index=["Actual good (0)", "Actual default (1)"],
                columns=["Predicted good (0)", "Predicted decline (1)"],
            )
            fig_cm = px.imshow(
                cm_df.values,
                x=cm_df.columns,
                y=cm_df.index,
                text_auto=True,
                color_continuous_scale="Blues",
                aspect="auto",
            )
            fig_cm.update_layout(height=320, margin=dict(t=20))
            st.plotly_chart(fig_cm, use_container_width=True)

        st.divider()
        st.markdown("**Expected cost vs decline threshold** (holds approve gate fixed)")
        cost_curve = compute_cost_curve(y_true, y_score, FP_COST, FN_COST, n_bins=80)
        fig_cost = px.line(cost_curve, x="Threshold", y="Expected cost")
        fig_cost.add_vline(
            x=model["threshold"]["optimal_threshold"],
            line_dash="dash",
            line_color="green",
            annotation_text=f"optimum t = {model['threshold']['optimal_threshold']}",
        )
        fig_cost.add_vline(
            x=decline_t,
            line_dash="dot",
            line_color="orange",
            annotation_text=f"your gate = {decline_t:.2f}",
        )
        fig_cost.update_layout(height=320, yaxis_tickprefix="$", yaxis_tickformat=",.0f")
        st.plotly_chart(fig_cost, use_container_width=True)

        current_cost = (
            int(((y_score >= decline_t) & (y_true == 0)).sum()) * FP_COST
            + int(((y_score < decline_t) & (y_true == 1)).sum()) * FN_COST
        )
        opt_cost = int(model["threshold"]["expected_cost_at_optimal"])
        st.caption(
            f"At your gate of {decline_t:.2f}: expected cost **USD {current_cost:,.0f}**. "
            f"Optimum is at t={model['threshold']['optimal_threshold']:.2f} with cost "
            f"**USD {opt_cost:,.0f}**. "
            f"Cost matrix: FP = USD {FP_COST} (lost NIM on a wrongly-declined good applicant), "
            f"FN = USD {FN_COST} (EAD x LGD on an approved defaulter)."
        )


with tab_fair:
    st.subheader("Adverse Impact Ratio sweep")
    st.caption(
        "AIR = approval rate of the smallest group divided by the largest. "
        "The 80% (4/5) rule is the common fairness heuristic: AIR >= 0.80 passes. "
        "This sweep uses the OOF predictions and the sensitive columns joined into the artifact."
    )

    plot_df = fairness_df[["threshold", "AIR_CODE_GENDER", "AIR_NAME_EDUCATION_TYPE"]].copy()
    plot_df = plot_df.rename(
        columns={"AIR_CODE_GENDER": "Gender", "AIR_NAME_EDUCATION_TYPE": "Education"}
    )
    fig_air = go.Figure()
    fig_air.add_trace(
        go.Scatter(x=plot_df["threshold"], y=plot_df["Gender"], mode="lines+markers", name="Gender")
    )
    fig_air.add_trace(
        go.Scatter(
            x=plot_df["threshold"], y=plot_df["Education"], mode="lines+markers", name="Education"
        )
    )
    fig_air.add_hline(
        y=0.80,
        line_dash="dash",
        line_color="red",
        annotation_text="80% rule",
        annotation_position="bottom right",
    )
    fig_air.update_layout(
        height=360,
        xaxis_title="Decline threshold",
        yaxis_title="Adverse Impact Ratio",
        yaxis_range=[0.0, 1.0],
    )
    st.plotly_chart(fig_air, use_container_width=True)

    st.warning(
        "**Headline finding:** Education AIR fails the 80% rule across every evaluated threshold. "
        "Gender passes from t=0.40 onwards. In a real deployment this would trigger: (1) a feature "
        "review for education-correlated variables, (2) a threshold calibration by segment if "
        "legally defensible, (3) adverse-action reason codes audited for disparate-impact signals. "
        "See `governance/FAIRNESS_REPORT.md`."
    )

    with st.expander("Raw sweep table"):
        st.dataframe(fairness_df, hide_index=True, use_container_width=True)


with tab_shap:
    st.subheader("Global feature importance (mean |SHAP|)")
    st.caption(
        "Tree SHAP on the full training set. The same attributions power per-applicant "
        "adverse-action reason codes (see `governance/ADVERSE_ACTION.md`)."
    )

    shap_sorted = shap_df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    shap_sorted.index = shap_sorted.index + 1
    st.dataframe(
        shap_sorted,
        column_config={
            "feature_name": "Feature",
            "mean_abs_shap": st.column_config.ProgressColumn(
                "Mean |SHAP| (log-odds)",
                format="%.3f",
                min_value=0.0,
                max_value=float(shap_sorted["mean_abs_shap"].max()),
            ),
        },
        use_container_width=True,
        height=560,
    )
    total = float(shap_sorted["mean_abs_shap"].sum())
    ext_share = (
        float(shap_sorted[shap_sorted["feature_name"].str.contains("ext_")]["mean_abs_shap"].sum())
        / total
    )
    st.caption(
        f"External-bureau features concentrate **{ext_share:.0%}** of total attribution. "
        "`feat_ext_source_mean` alone contributes the single largest share — it's an average of "
        "three external-source scores. Affordability ratio (`feat_annuity_to_credit`) and interaction "
        "terms follow."
    )


with tab_ablation:
    st.subheader("No-bureau ablation: what if the external bureau features disappear?")
    st.caption(
        "Retrained LightGBM after dropping the five `ext_source_*` features from the champion's 15. "
        "Stress-tests sensitivity to the most predictive data source (e.g., if a bureau contract "
        "lapses or a region lacks bureau coverage)."
    )

    champion_metrics = eval_data["champion"]["metrics"]
    ab_cols = st.columns(3)
    ab_cols[0].metric(
        "ROC-AUC",
        f"{ablation['oof']['roc_auc']:.3f}",
        f"{ablation['oof']['roc_auc'] - champion_metrics['roc_auc']:+.3f}",
    )
    ab_cols[1].metric(
        "PR-AUC",
        f"{ablation['oof']['pr_auc']:.3f}",
        f"{ablation['oof']['pr_auc'] - champion_metrics['pr_auc']:+.3f}",
    )
    ab_cols[2].metric(
        "KS",
        f"{ablation['oof']['ks']:.3f}",
        f"{ablation['oof']['ks'] - champion_metrics['ks_statistic']:+.3f}",
    )

    st.markdown("**Excluded features**")
    st.code("\n".join(ablation["excluded_features"]), language="text")
    st.caption(
        f"Ablation trained in {ablation['elapsed_seconds']:.0f}s. ROC-AUC drops by "
        f"~{champion_metrics['roc_auc'] - ablation['oof']['roc_auc']:.3f} — this is the magnitude of "
        "degradation a lender would face if external bureau scores became unavailable. "
        "It scopes the capacity/bureau asymmetry discussion in `features/FEATURE_IMPORTANCE.md`."
    )

st.divider()
st.caption(
    "Artifacts: `artifacts/evaluation_results.json`, `shap_mean_abs.csv`, `fairness_threshold_sweep.csv`, "
    "`oof_predictions_scored.parquet`, `ablation_no_bureau_summary.json`, `reliability_lightgbm.png`. "
    "All curves on this page are recomputed live from the OOF parquet."
)
