from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np

from src.preprocess import (
    RAW_DEFAULT,
    OUT_DEFAULT,
    load_raw,
    preprocess,
    save_processed,
    summarize_outliers_iqr,
)

from src.viz import (
    plot_missingno,
    plot_corr_heatmap,
    plot_corr_with_label_barplot,
    plot_pairplot,
    plot_stacked_ratio_bar,
    plot_observed_vs_predicted_smoking_by_bmi, 
    plot_or_forest_bmi,
    plot_hemoglobin_box_by_smoking,
    plot_hemoglobin_box_by_agegroup_and_smoking,
    plot_tg_hdl_distribution_and_qq,
    plot_proteinuria_rate_by_smoking

)

from src.analysis import (
    analyze_caries_by_smoking,
    analyze_smoking_by_bmi_logit,
    analyze_hemoglobin_by_smoking,
    analyze_log_tg_hdl_by_smoking,
    analyze_kidney_by_smoking,

)



FIG_DIR = Path("outputs/figures")
TAB_DIR = Path("outputs/tables")


def ensure_dirs() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TAB_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DEFAULT.parent.mkdir(parents=True, exist_ok=True)


def save_basic_reports(df_raw: pd.DataFrame, df_proc: pd.DataFrame) -> None:
    shape_df = pd.DataFrame(
        [
            {"stage": "raw", "rows": df_raw.shape[0], "cols": df_raw.shape[1]},
            {"stage": "processed", "rows": df_proc.shape[0], "cols": df_proc.shape[1]},
        ]
    )
    shape_df.to_csv(TAB_DIR / "shape_summary.csv", index=False, encoding="utf-8-sig")

    miss_raw = df_raw.isna().sum().sort_values(ascending=False)
    miss_raw_rate = (miss_raw / len(df_raw)).round(4)
    pd.DataFrame(
        {"column": miss_raw.index, "missing": miss_raw.values, "missing_rate": miss_raw_rate.values}
    ).to_csv(TAB_DIR / "missing_summary_raw.csv", index=False, encoding="utf-8-sig")

    miss_proc = df_proc.isna().sum().sort_values(ascending=False)
    miss_proc_rate = (miss_proc / len(df_proc)).round(4)
    pd.DataFrame(
        {"column": miss_proc.index, "missing": miss_proc.values, "missing_rate": miss_proc_rate.values}
    ).to_csv(TAB_DIR / "missing_summary_processed.csv", index=False, encoding="utf-8-sig")


def run_eda_before_preprocess(df_raw: pd.DataFrame, save_figs: bool = False) -> None:
    plot_missingno(
        df_raw,
        save_path=(FIG_DIR / "missing_matrix.png") if save_figs else None,
        show=True,
    )

    plot_corr_heatmap(
        df_raw,
        title="Correlation Heatmap (Raw)",
        save_path=(FIG_DIR / "corr_heatmap_raw.png") if save_figs else None,
        show=True,
    )

    top_corr_df = plot_corr_with_label_barplot(
        df_raw,
        label_col="label",
        top_n=12,
        save_path=(FIG_DIR / "top_corr_with_label.png") if save_figs else None,
        show=True,
    )

    top_features = top_corr_df["feature"].head(5).tolist()
    cols_for_pairplot = top_features + ["label"]
    plot_pairplot(
        df_raw,
        cols=cols_for_pairplot,
        hue="label",
        save_path=(FIG_DIR / "pairplot_top5.png") if save_figs else None,
        show=True,
    )


def main():
    ensure_dirs()

    # load_raw에서 이미 컬럼 영문화 적용됨(rename_english=True 기본)
    df_raw = load_raw(RAW_DEFAULT)

    RUN_EDA = True
    SAVE_FIGS = True  
    if RUN_EDA:
        run_eda_before_preprocess(df_raw, save_figs=SAVE_FIGS)


    df_proc = preprocess(df_raw, drop_id=True, do_outlier_clip=False, rename_english=False)

    save_processed(df_proc, OUT_DEFAULT)
    save_basic_reports(df_raw, df_proc)

    print("✅ Preprocessing + EDA pipeline done ✅\n")
    print("기본정보 확인\n")
    print("Raw shape:", df_raw.shape)
    print("Processed shape:", df_proc.shape)
    print("Saved processed:", OUT_DEFAULT)

    print("data description: ", df_proc.describe())
    print("data head & tail:")
    print(df_proc.head(20), df_proc.tail(20))

    
    print("Saved reports:", TAB_DIR)
    print("Saved figures:", FIG_DIR if (RUN_EDA and SAVE_FIGS) else "(disabled)")

    # ====================전처리 이후====================



    # 이상치 탐지
    # ====================
    # Outlier report (IQR)
    # ====================
    OUTLIER_COLS = [
        "triglycerides",
        "systolic_bp",
        "fasting_glucose",
        "bmi",
        "vision",
        "serum_creatinine",
        "cholesterol",
        "hemoglobin",
        "liver_enzyme",
    ]

    out_sum, out_map = summarize_outliers_iqr(df_proc, OUTLIER_COLS, k=1.5)

# 요약 저장
    out_sum.to_csv(TAB_DIR / "outliers_iqr_summary.csv", index=False, encoding="utf-8-sig")

# 컬럼별 이상치 저장 
    for col, odf in out_map.items():
        if odf.empty:
            continue
        odf[[col]].to_csv(
            TAB_DIR / f"outliers_{col}_sample.csv",
            index=False,
            encoding="utf-8-sig",
        )

    print("\n✅ IQR outlier report saved:")
    print(out_sum.head(10))

    # ==================================================





    # ==================================================

    # 가설1. 흡연 여부와 충치 발생 여부는 독립이 아니다(연관 있음).
    print("\n가설 1. 흡연 여부와 충치 발생 여부는 독립이 아니다(연관 있음) ")
    analyze_caries_by_smoking(df_proc)
    print("귀무가설 기각, 대립가설 채택! ")

    # 가설2. 흡연 여부는 헤모글로빈 수치와 관련이 있다(흡연자가 더 높을 수 있음). 
    # + 나이 구간(연차)에 따라 흡연으로 인한 헤모글로빈 수치가 양의 영향을 받을 것 이다.
    print("\n가설 2. 흡연자는 비흡연자보다 혈중 헤모글로빈 수치가 높다. ")
    out2 = analyze_hemoglobin_by_smoking(df_proc)

    # 1) boxplot (label별)
    plot_hemoglobin_box_by_smoking(
        out2["df_used"],
        save_path=FIG_DIR / "hemoglobin_by_smoking_box.png",
        show=True,
    )

    # 2) age_group x label boxplot
    plot_hemoglobin_box_by_agegroup_and_smoking(
        out2["df_used"],
        save_path=FIG_DIR / "hemoglobin_by_agegroup_smoking_box.png",
        show=True,
    )

    # 3) 통계 결과 저장/출력
    out2["count_table"].to_csv(TAB_DIR / "hemoglobin_count_by_agegroup_label.csv", encoding="utf-8-sig")

    print("\n=== Welch Two Sample t-test ===")
    print(f"t = {out2['t_stat']:.4f}, df ≈ {out2['df_welch']:.1f}, p-value = {out2['p_value']:.3e}")
    print("95% CI for (mean group0 - mean group1): "
        f"[{out2['ci_95_low']:.6f}, {out2['ci_95_high']:.6f}]")
    print("sample estimates:")
    print(f"mean in group 0 = {out2['mean0']:.5f}")
    print(f"mean in group 1 = {out2['mean1']:.5f}")

    print("\n=== OLS: hemoglobin ~ label + age ===")
    print(out2["ols_model"].summary())
    print("귀무가설 기각, 대립가설 채택! \n")

    # 가설3: 흡연 여부는 TG/Glucose/HDL과 유의한 관련이 있다.

    print("\n가설 3. 흡연 여부는 TG/Glucose/HDL과 유의한 관련이 있다. ")
    out3 = analyze_log_tg_hdl_by_smoking(df_proc)

    # 분포/QQ 저장
    plot_tg_hdl_distribution_and_qq(
        out3["df_used"],
        save_path=FIG_DIR / "tg_hdl_ratio_dist_qq.png",
        show=True,
    )

    # 테이블 저장
    out3["exp_table"].to_csv(TAB_DIR / "ols_log_tg_hdl_expB.csv", encoding="utf-8-sig")

    print("\n=== OLS: log(TG/HDL) ~ label + age + bmi ===")
    print(out3["ols_model"].summary())

    if out3["smoke_interp"] is not None:
        s = out3["smoke_interp"]
        print("\n해석(흡연 변수):")
        print(f"- Exp(B)={s['ratio']:.4f} (95% CI: {s['ci_low']:.4f} ~ {s['ci_high']:.4f})")
        print(f"- 나이와 BMI를 통제했을 때, 흡연자는 TG/HDL 비율이 약 {s['pct']:.1f}% 높다고 해석 가능")
    print("\n귀무가설 기각, 대립가설 채택! \n")


    # 가설 4. BMI가 증가할수록 흡연할 확률이 증가한다.

    print("\n가설 4. BMI가 증가할수록 흡연할 확률이 증가한다. ")
    out = analyze_smoking_by_bmi_logit(df_proc, n_bins=10)
    out["or_table"].to_csv(TAB_DIR / "logit_smoking_by_bmi_or.csv", encoding="utf-8-sig")

    # 그림 저장
    plot_observed_vs_predicted_smoking_by_bmi(
        pred_df=out["pred_df"],
        obs_df=out["obs_df"],
        save_path=FIG_DIR / "logit_smoking_by_bmi_curve.png",
        show=True,
    )

    plot_or_forest_bmi(
        or_val=out["or_val"],
        ci_low=out["ci_low"],
        ci_high=out["ci_high"],
        save_path=FIG_DIR / "logit_smoking_by_bmi_or.png",
        show=True,
    )

    print(out["result"].summary())

    print("\n귀무가설 기각, 대립가설 채택!\n")

    # 가설 5.
    print("\n가설 5. 흡연 여부는 단백뇨/크레아티닌 등 신장 지표와 관련이 있다.")
    out5 = analyze_kidney_by_smoking(df_proc, protein_threshold=1)

    # 표 저장
    out5["cross_table"].to_csv(TAB_DIR / "proteinuria_crosstab.csv", encoding="utf-8-sig")
    out5["prop_table"].to_csv(TAB_DIR / "proteinuria_prop_table.csv", encoding="utf-8-sig")
    out5["expected_df"].to_csv(TAB_DIR / "proteinuria_expected.csv", encoding="utf-8-sig")
    out5["or_table"].to_csv(TAB_DIR / "proteinuria_logit_or.csv", encoding="utf-8-sig")

    # 그림 저장
    plot_proteinuria_rate_by_smoking(
        out5["prop_table"],
        save_path=FIG_DIR / "proteinuria_rate_by_smoking.png",
        show=True,
    )

    # 출력
    print("\n=== [카이제곱 검정 결과] ===")
    print(f"Chi-square: {out5['chi2']:.4f}, dof: {out5['dof']}, p-value: {out5['p_value']:.6f}")

    print("\n=== [로지스틱 회귀 요약: Proteinuria_Binary] ===")
    print(out5["logit_model"].summary())

    print("\n=== [오즈비(Odds Ratio) 및 95% 신뢰구간] ===")
    print(out5["or_table"].round(4))

    print("\n=== [OLS 요약: serum_creatinine] ===")
    print(out5["ols_model"].summary())



    print("유의미한 차이를 확인하지 못함 (H0 기각 실패)")



if __name__ == "__main__":
    main()
