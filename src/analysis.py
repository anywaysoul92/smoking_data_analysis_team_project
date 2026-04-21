# analysis.py
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
from scipy import stats

import statsmodels.formula.api as smf
import statsmodels.api as sm
from src.viz import plot_stacked_ratio_bar

# 오즈비 계산

def odds_ratio_with_ci(result) -> pd.DataFrame:
    params = result.params
    conf = result.conf_int()
    conf.columns = ["2.5%", "97.5%"]
    or_df = np.exp(pd.concat([params.rename("OR"), conf], axis=1))
    return or_df.rename(columns={"2.5%": "CI 2.5%", "97.5%": "CI 97.5%"})

# 이상치 탐지

def analyze_caries_by_smoking(health_data2: pd.DataFrame) -> dict:
    # 가설 1: 충치
    df = health_data2.copy()

    SMOKE_COL = "label"  # 흡연 여부 (0=비흡연, 1=흡연)
    CARIES_COL = "dental_caries"   # 충치 발생 여부 (0=없음, 1=있음)

    # -----------------------------
    # 0) 숫자형 변환 + 필요한 값만 남기기
    # -----------------------------
    df[SMOKE_COL] = pd.to_numeric(df[SMOKE_COL], errors="coerce")
    df[CARIES_COL] = pd.to_numeric(df[CARIES_COL], errors="coerce")

    # 결측 제거 (카이제곱/교차표에 NaN 섞이면 결과 꼬임)
    df = df.dropna(subset=[SMOKE_COL, CARIES_COL])

    # 0/1만 남기기 (혹시 2, 9 같은 이상값 있으면 제거)
    df = df[df[SMOKE_COL].isin([0, 1]) & df[CARIES_COL].isin([0, 1])]

    # 정수로 고정 (0.0/1.0 → 0/1)
    df[SMOKE_COL] = df[SMOKE_COL].astype(int)
    df[CARIES_COL] = df[CARIES_COL].astype(int)

    print("분석 데이터 크기:", df.shape)

    # -----------------------------
    # ① 교차표 (빈도) - 2x2 강제 (범주 누락 방지)
    # -----------------------------
    ct = pd.crosstab(df[SMOKE_COL], df[CARIES_COL]).reindex(
        index=[0, 1], columns=[0, 1], fill_value=0
    )
    ct.index = ["비흡연", "흡연"]
    ct.columns = ["충치 없음", "충치 있음"]

    print("\n[교차표 - 빈도]")
    print(ct)

    # -----------------------------
    # ② 비율 교차표 (행 기준)
    # -----------------------------
    ct_ratio = ct.div(ct.sum(axis=1).replace(0, pd.NA), axis=0)
    print("\n[교차표 - 비율]")
    print(ct_ratio)

    # -----------------------------
    # ③ 시각화 (비율 막대그래프)
    # -----------------------------
    plot_stacked_ratio_bar(
        ct_ratio,
        title="흡연 여부에 따른 충치 발생 비율",
        show=True,
        save_path="outputs/figures/caries_by_smoking_ratio.png", 
)

    # -----------------------------
    # ④ 카이제곱 독립성 검정
    # -----------------------------
    chi2, p, dof, expected = chi2_contingency(ct, correction=False)

    print("\n[카이제곱 검정 결과]")
    print(f"Chi-square 통계량: {chi2:.4f}")
    print(f"자유도(dof): {dof}")
    print(f"p-value: {p:.3e}")

    expected_df = pd.DataFrame(expected, index=ct.index, columns=ct.columns)
    print("\n[기대빈도]")
    print(expected_df.round(3))

    return {
        "ct": ct,
        "ct_ratio": ct_ratio,
        "chi2": chi2,
        "p": p,
        "dof": dof,
        "expected": expected_df,
        "n": len(df),
    }

# 가설 2 ====================================



def _welch_df(x: np.ndarray, y: np.ndarray) -> float:
    """
    Welch-Satterthwaite df
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]

    nx, ny = len(x), len(y)
    vx, vy = np.var(x, ddof=1), np.var(y, ddof=1)

    num = (vx / nx + vy / ny) ** 2
    den = (vx**2) / (nx**2 * (nx - 1)) + (vy**2) / (ny**2 * (ny - 1))
    return float(num / den)


def _make_age_group_bins(df: pd.DataFrame, age_col: str = "age") -> pd.Series:
    """
    (20s, 30s, ..., over 80)
    preprocess.py의 age_group(10단위 정수)와는 별개로, 시각화용.
    """
    bins = [0, 29, 39, 49, 59, 69, 79, np.inf]
    labels = ["20s", "30s", "40s", "50s", "60s", "70s", "over 80"]
    return pd.cut(df[age_col], bins=bins, labels=labels, right=True)


def analyze_hemoglobin_by_smoking(df: pd.DataFrame) -> dict:
    """
    가설2: 흡연 여부(label)가 hemoglobin에 영향을 주는가?
    - Welch t-test (label 0 vs 1)
    - 평균차 95% CI (mean0 - mean1)
    - OLS: hemoglobin ~ label + age
    - age_group x label count table
    """
    d = df[["label", "hemoglobin", "age"]].copy()
    d["label"] = pd.to_numeric(d["label"], errors="coerce")
    d["hemoglobin"] = pd.to_numeric(d["hemoglobin"], errors="coerce")
    d["age"] = pd.to_numeric(d["age"], errors="coerce")
    d = d.dropna(subset=["label", "hemoglobin", "age"])
    d = d[d["label"].isin([0, 1])].copy()
    d["label"] = d["label"].astype(int)

    g0 = d.loc[d["label"] == 0, "hemoglobin"].to_numpy()
    g1 = d.loc[d["label"] == 1, "hemoglobin"].to_numpy()

    # Welch t-test
    t_stat, p_value = stats.ttest_ind(g0, g1, equal_var=False, nan_policy="omit")
    mean0, mean1 = float(np.nanmean(g0)), float(np.nanmean(g1))
    diff = mean0 - mean1

    # Welch df + CI
    df_welch = _welch_df(g0, g1)
    se = np.sqrt(np.nanvar(g0, ddof=1) / len(g0) + np.nanvar(g1, ddof=1) / len(g1))
    t_crit = stats.t.ppf(0.975, df_welch)
    ci_low, ci_high = diff - t_crit * se, diff + t_crit * se

    # OLS (age control)
    ols_model = smf.ols("hemoglobin ~ label + age", data=d).fit()

    # age_group for visualization + count check
    d["age_group2"] = _make_age_group_bins(d, "age")
    count_table = d.groupby(["age_group2", "label"]).size().unstack(fill_value=0)
    count_table.columns = ["non-smokers(0)", "smokers(1)"]

    return {
        "df_used": d, 
        "t_stat": float(t_stat),
        "p_value": float(p_value),
        "df_welch": float(df_welch),
        "mean0": mean0,
        "mean1": mean1,
        "diff_mean0_minus_mean1": float(diff),
        "ci_95_low": float(ci_low),
        "ci_95_high": float(ci_high),
        "ols_model": ols_model,
        "count_table": count_table,
        "n0": int(len(g0)),
        "n1": int(len(g1)),
    }


# 가설 3 ====================================

def add_tg_hdl_features(
    df: pd.DataFrame,
    tg_col: str = "triglycerides",
    hdl_col: str = "hdl",
) -> pd.DataFrame:
    d = df.copy()
    d[tg_col] = pd.to_numeric(d[tg_col], errors="coerce")
    d[hdl_col] = pd.to_numeric(d[hdl_col], errors="coerce")

    d["tg_hdl_ratio"] = d[tg_col] / d[hdl_col]
    d["log_tg_hdl_ratio"] = np.where(d["tg_hdl_ratio"] > 0, np.log(d["tg_hdl_ratio"]), np.nan)
    return d


def fit_ols_log_tg_hdl(
    df: pd.DataFrame,
    y_col: str = "log_tg_hdl_ratio",
    label_col: str = "label",
    age_col: str = "age",
    bmi_col: str = "bmi",
):
    model_df = df[[y_col, label_col, age_col, bmi_col]].copy()
    model_df[label_col] = pd.to_numeric(model_df[label_col], errors="coerce")
    model_df[age_col] = pd.to_numeric(model_df[age_col], errors="coerce")
    model_df[bmi_col] = pd.to_numeric(model_df[bmi_col], errors="coerce")
    model_df[y_col] = pd.to_numeric(model_df[y_col], errors="coerce")
    model_df = model_df.dropna()

    # label 0/1 고정
    model_df = model_df[model_df[label_col].isin([0, 1])].copy()
    model_df[label_col] = model_df[label_col].astype(int)

    y = model_df[y_col]
    X = sm.add_constant(model_df[[label_col, age_col, bmi_col]], has_constant="add")
    ols_model = sm.OLS(y, X).fit()
    return ols_model, model_df


def summarize_expB_table(ols_model) -> pd.DataFrame:
    params = ols_model.params
    conf = ols_model.conf_int()
    conf.columns = ["2.5%", "97.5%"]

    exp_table = pd.DataFrame({
        "beta": params,
        "2.5%": conf["2.5%"],
        "97.5%": conf["97.5%"],
    })
    exp_table["Exp(B) [Ratio]"] = np.exp(exp_table["beta"])
    exp_table["95% CI Lower"] = np.exp(exp_table["2.5%"])
    exp_table["95% CI Upper"] = np.exp(exp_table["97.5%"])
    exp_table["Percent change (%)"] = (exp_table["Exp(B) [Ratio]"] - 1) * 100

    return exp_table[["Exp(B) [Ratio]", "95% CI Lower", "95% CI Upper", "Percent change (%)"]]


def analyze_log_tg_hdl_by_smoking(
    df: pd.DataFrame,
    label_col: str = "label",
    age_col: str = "age",
    bmi_col: str = "bmi",
    tg_col: str = "triglycerides",
    hdl_col: str = "hdl",
) -> dict:
    d = add_tg_hdl_features(df, tg_col=tg_col, hdl_col=hdl_col)
    ols_model, model_df = fit_ols_log_tg_hdl(
        d,
        y_col="log_tg_hdl_ratio",
        label_col=label_col,
        age_col=age_col,
        bmi_col=bmi_col,
    )
    exp_table = summarize_expB_table(ols_model)

    # 흡연(label) 해석용 값
    smoke_interp = None
    if label_col in exp_table.index:
        smoke_interp = {
            "ratio": float(exp_table.loc[label_col, "Exp(B) [Ratio]"]),
            "ci_low": float(exp_table.loc[label_col, "95% CI Lower"]),
            "ci_high": float(exp_table.loc[label_col, "95% CI Upper"]),
            "pct": float(exp_table.loc[label_col, "Percent change (%)"]),
        }

    return {
        "df_used": d,
        "model_df": model_df,
        "ols_model": ols_model,
        "exp_table": exp_table,
        "smoke_interp": smoke_interp,
    }





# 가설 4 ====================================
def fit_logit_smoking_by_bmi(df: pd.DataFrame, label_col: str = "label", bmi_col: str = "bmi"):
    """
    Logit: label(흡연=1) ~ bmi
    반환: (result, or_table)
    """
    d = df[[label_col, bmi_col]].copy()
    d[label_col] = pd.to_numeric(d[label_col], errors="coerce")
    d[bmi_col] = pd.to_numeric(d[bmi_col], errors="coerce")
    d = d.dropna(subset=[label_col, bmi_col])
    d = d[d[label_col].isin([0, 1])].copy()
    d[label_col] = d[label_col].astype(int)

    y = d[label_col]
    X = sm.add_constant(d[[bmi_col]], has_constant="add")

    model = sm.Logit(y, X)
    result = model.fit(disp=False)

    params = result.params
    conf = result.conf_int()
    or_table = np.exp(pd.concat([params, conf], axis=1))
    or_table.columns = ["OR", "2.5%", "97.5%"]
    return result, or_table


def make_bmi_pred_curve(df: pd.DataFrame, result, bmi_col: str = "bmi", n_points: int = 200) -> pd.DataFrame:
    bmi_min = float(pd.to_numeric(df[bmi_col], errors="coerce").min())
    bmi_max = float(pd.to_numeric(df[bmi_col], errors="coerce").max())
    bmi_range = np.linspace(bmi_min, bmi_max, n_points)

    X_pred = pd.DataFrame({"const": 1.0, bmi_col: bmi_range})
    pred_prob = result.predict(X_pred)

    return pd.DataFrame({bmi_col: bmi_range, "pred_prob": pred_prob})


def make_bmi_binned_observed_rate(
    df: pd.DataFrame,
    label_col: str = "label",
    bmi_col: str = "bmi",
    n_bins: int = 10,
) -> pd.DataFrame:
    d = df[[label_col, bmi_col]].copy()
    d[label_col] = pd.to_numeric(d[label_col], errors="coerce")
    d[bmi_col] = pd.to_numeric(d[bmi_col], errors="coerce")
    d = d.dropna(subset=[label_col, bmi_col])
    d = d[d[label_col].isin([0, 1])].copy()
    d[label_col] = d[label_col].astype(int)

    d["bmi_bin"] = pd.cut(d[bmi_col], bins=n_bins)
    obs_rate = d.groupby("bmi_bin", observed=True)[label_col].mean()
    bin_mid = np.array([interval.mid for interval in obs_rate.index])

    return pd.DataFrame({"bmi_mid": bin_mid, "obs_rate": obs_rate.values})


def analyze_smoking_by_bmi_logit(df: pd.DataFrame, n_bins: int = 10) -> dict:
    """
    최종 통합 분석 함수:
    - result, or_table
    - pred_df (곡선)
    - obs_df (구간 관측 비율)
    """
    result, or_table = fit_logit_smoking_by_bmi(df)
    pred_df = make_bmi_pred_curve(df, result)
    obs_df = make_bmi_binned_observed_rate(df, n_bins=n_bins)

    or_val = float(or_table.loc["bmi", "OR"])
    ci_low = float(or_table.loc["bmi", "2.5%"])
    ci_high = float(or_table.loc["bmi", "97.5%"])

    return {
        "result": result,
        "or_table": or_table,
        "pred_df": pred_df,
        "obs_df": obs_df,
        "or_val": or_val,
        "ci_low": ci_low,
        "ci_high": ci_high,
    }

# 가설 5 ===============================

def analyze_kidney_by_smoking(df: pd.DataFrame, protein_threshold: int = 1) -> dict:
    """
    가설5:
    - 단백뇨(urine_protein > 1): 교차표/비율 + 카이제곱
    - 단백뇨(0/1): 로지스틱(label + age + systolic_bp + fasting_glucose)
    - 크레아티닌: OLS(label + age + systolic_bp + fasting_glucose + height_cm + weight_kg)
    """

    required = [
        "label", "age", "urine_protein",
        "systolic_bp", "fasting_glucose",
        "serum_creatinine", "height_cm", "weight_kg",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"필수 컬럼 누락: {missing}")

    d = df[required].copy()

    d = d.dropna(subset=["label", "urine_protein"])
    d = d[d["label"].isin([0, 1])].copy()
    d["label"] = d["label"].astype(int)

    # 1) 단백뇨 파생 (범주형/이진)
    d["Proteinuria_Status"] = np.where(
        d["urine_protein"] > protein_threshold,
        "urine protein(Positive)",
        "Normal"
    )
    d["Smoking_Label"] = d["label"].map({0: "non-smokers", 1: "smokers"}).astype("category")
    d["Proteinuria_Binary"] = np.where(d["urine_protein"] > protein_threshold, 1, 0)

    # 2) 교차표 + 비율(행 기준 %)
    cross_table = pd.crosstab(d["Smoking_Label"], d["Proteinuria_Status"])
    prop_table = cross_table.div(cross_table.sum(axis=1), axis=0) * 100

    # 3) 카이제곱
    chi2, p, dof, expected = chi2_contingency(cross_table)
    expected_df = pd.DataFrame(expected, index=cross_table.index, columns=cross_table.columns)

    # 4) 이항 로지스틱
    logit_cols = ["Proteinuria_Binary", "label", "age", "systolic_bp", "fasting_glucose"]
    logit_df = d[logit_cols].dropna()

    y = logit_df["Proteinuria_Binary"]
    X = logit_df[["label", "age", "systolic_bp", "fasting_glucose"]]
    X = sm.add_constant(X, has_constant="add")
    model_kidney = sm.Logit(y, X).fit(disp=False)

    or_table = odds_ratio_with_ci(model_kidney)

    # 5) 크레아티닌 OLS
    ols_cols = ["serum_creatinine", "label", "age", "systolic_bp", "fasting_glucose", "height_cm", "weight_kg"]
    ols_df = d[ols_cols].dropna()

    y2 = ols_df["serum_creatinine"]
    X2 = ols_df[["label", "age", "systolic_bp", "fasting_glucose", "height_cm", "weight_kg"]]
    X2 = sm.add_constant(X2, has_constant="add")
    model_creat = sm.OLS(y2, X2).fit()

    return {
        "df_used": d,
        "cross_table": cross_table,
        "prop_table": prop_table,
        "chi2": float(chi2),
        "p_value": float(p),
        "dof": int(dof),
        "expected_df": expected_df,
        "logit_model": model_kidney,
        "or_table": or_table,
        "ols_model": model_creat,
    }
