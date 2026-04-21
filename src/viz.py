from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy import stats


# 색상 팔레트 설정 (일관된 색상 사용을 위해 따로 설정함함) 
COLOR_PALETTES = {
    # 흡연 여부 (0=비흡연, 1=흡연)
    "smoking": ["#4A90E2", "#E94B3C"],  # 파랑(비흡연), 빨강(흡연)

    # 나이 그룹 (10대, 20대, 30대, 40대, 50대, 60대 이상)
    "age_groups": ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7", "#DDA0DD"],

    # 건강 지표 상태 (정상, 주의, 위험)
    "health_status": ["#2ECC71", "#F39C12", "#E74C3C"],

    # 성별 (추후 확장용)
    "gender": ["#3498DB", "#E91E63"],

    # 일반 카테고리 색상
    "categorical": sns.color_palette("Set2", 8),
}


def _set_korean_font() -> None:
    # 그래프 제목/출력 한글 깨짐 방지(환경 동일하면 유지 - window로 설정해놓음음)
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False


def _get_smoking_palette() -> list:
    """흡연 관련 그래프용 일관된 색상 팔레트"""
    return COLOR_PALETTES["smoking"]


def _get_age_group_palette() -> list:
    """나이 그룹 관련 그래프용 일관된 색상 팔레트"""
    return COLOR_PALETTES["age_groups"]


_set_korean_font()


def _maybe_savefig(save_path: str | Path | None, dpi: int = 150) -> None:
    if save_path is None:
        return
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")


def plot_missingno(df: pd.DataFrame, save_path: str | Path | None = None, show: bool = True) -> None:
    try:
        import missingno as msno
    except ImportError as e:
        raise ImportError("missingno가 설치되어 있지 않습니다. `pip install missingno` 후 실행하세요.") from e

    plt.figure(figsize=(12, 6))
    msno.matrix(df, fontsize=10)
    plt.title("Missing Value Matrix")
    plt.tight_layout()
    _maybe_savefig(save_path)
    if show:
        plt.show()
    else:
        plt.close()


def plot_corr_heatmap(
    df: pd.DataFrame,
    title: str = "Correlation Heatmap",
    save_path: str | Path | None = None,
    show: bool = True,
) -> None:
    num_df = df.select_dtypes(include="number")
    corr = num_df.corr()

    plt.figure(figsize=(14, 10))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
    )
    plt.title(title)
    plt.tight_layout()
    _maybe_savefig(save_path)
    if show:
        plt.show()
    else:
        plt.close()

# 선택한 여러 수치 컬럼들 사이의 관계(산점도) + 각 변수의 분포를 한 번에 보여주는 pairplot(페어플롯) 시각화 함수
def plot_pairplot(
    df: pd.DataFrame,
    cols: Iterable[str],
    hue: str = "label",
    save_path: str | Path | None = None,
    show: bool = True,
) -> None:
    cols = list(cols)
    # pairplot에 hue를 쓰려면, hue 컬럼도 plot_df 안에 있어야 함.
    # 예를들어 cols=["age","BMI"]만 줬으면 label이 빠져있게 됨.
    # 그래서 자동으로 label을 컬럼 목록에 추가하는 코드드
    if hue not in cols and hue in df.columns:
        cols = cols + [hue]

    plot_df = df[cols].dropna()

    g = sns.pairplot(
        plot_df,
        hue=hue,
        palette=_get_smoking_palette(),
        diag_kind="kde",
        corner=True,
        plot_kws={"alpha": 0.5, "s": 20},
    )

    g.fig.suptitle("Pairplot (Selected Features)", y=1.02)

# 저장하고 화면에 띄움
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        g.fig.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(g.fig)


def plot_corr_with_label_barplot(
    df: pd.DataFrame,
    label_col: str = "label",
    top_n: int = 12,
    method: str = "pearson",
    save_path: str | Path | None = None,
    show: bool = True,
) -> pd.DataFrame:
    num_df = df.select_dtypes(include="number").copy()
    if label_col not in num_df.columns:
        raise ValueError(f"'{label_col}' column not found in numeric columns.")

    corr = num_df.corr(method=method)[label_col].drop(label_col).dropna()

    plot_df = (
        corr.sort_values(key=lambda s: s.abs(), ascending=False)
        .head(top_n)
        .reset_index()
        .rename(columns={"index": "feature", label_col: "corr"})
    )

    plt.figure(figsize=(10, 6))
    sns.barplot(data=plot_df, x="corr", y="feature", orient="h")
    plt.axvline(0, linewidth=1)
    plt.title(f"Top {top_n} Correlation with '{label_col}' ({method})")
    plt.xlabel("Correlation")
    plt.ylabel("")
    plt.tight_layout()
    _maybe_savefig(save_path)
    if show:
        plt.show()
    else:
        plt.close()

    return plot_df

# 가설 1 충치 - 흡연자
def plot_stacked_ratio_bar(
    ratio_df: pd.DataFrame,
    title: str,
    ylabel: str = "비율",
    figsize=(7, 4),
    legend_loc: str = "upper right",
    show: bool = True,
    save_path: str | None = None,
):
    ax = ratio_df.plot(kind="bar", stacked=True, figsize=figsize)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.legend(loc=legend_loc)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()

    plt.close()
    return ax


# 가설 2

def plot_hemoglobin_box_by_smoking(
    df: pd.DataFrame,
    label_col: str = "label",
    y_col: str = "hemoglobin",
    save_path: str | Path | None = None,
    show: bool = True,
) -> None:
    plt.figure(figsize=(9, 5))
    sns.boxplot(x=label_col, y=y_col, data=df, palette=_get_smoking_palette())
    plt.title("흡연 여부에 따른 헤모글로빈 수치 비교")
    plt.xlabel("")
    plt.xticks([0, 1], ["비흡연자(0)", "흡연자(1)"])
    plt.ylabel("헤모글로빈")
    plt.tight_layout()
    _maybe_savefig(save_path)
    if show:
        plt.show()
    else:
        plt.close()


def plot_hemoglobin_box_by_agegroup_and_smoking(
    df: pd.DataFrame,
    x_col: str = "age_group2",
    y_col: str = "hemoglobin",
    hue_col: str = "label",
    save_path: str | Path | None = None,
    show: bool = True,
) -> None:

    plt.figure(figsize=(12, 6))
    ax = sns.boxplot(x=x_col, y=y_col, hue=hue_col, data=df, palette=_get_smoking_palette())
    plt.title("나이를 통제한 후 연령대별 차이")
    plt.xlabel("나이대 그룹")
    plt.ylabel("헤모글로빈")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, ["비흡연자", "흡연자"], title="흡연 여부")

    plt.tight_layout()
    _maybe_savefig(save_path)
    if show:
        plt.show()
    else:
        plt.close()

# 가설 3

def plot_tg_hdl_distribution_and_qq(
    df: pd.DataFrame,
    ratio_col: str = "tg_hdl_ratio",
    log_col: str = "log_tg_hdl_ratio",
    save_path: str | Path | None = None,
    show: bool = True,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 7))

    # (1) before hist
    axes[0, 0].hist(df[ratio_col].dropna(), bins=30, edgecolor="black", alpha=0.7)
    axes[0, 0].set_title("로그 변환 전: TG/HDL 분포")
    axes[0, 0].set_xlabel("Ratio")
    axes[0, 0].set_ylabel("Frequency")

    # (2) before QQ
    stats.probplot(df[ratio_col].dropna(), dist="norm", plot=axes[0, 1])
    axes[0, 1].set_title("로그 변환 전: Q-Q Plot")

    # (3) after hist
    axes[1, 0].hist(df[log_col].dropna(), bins=30, edgecolor="black", alpha=0.7)
    axes[1, 0].set_title("로그 변환 후: Log(TG/HDL) 분포")
    axes[1, 0].set_xlabel("Log(Ratio)")
    axes[1, 0].set_ylabel("Frequency")

    # (4) after QQ
    stats.probplot(df[log_col].dropna(), dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title("로그 변환 후: Q-Q Plot")

    plt.tight_layout()
    _maybe_savefig(save_path)
    if show:
        plt.show()
    else:
        plt.close(fig)


# 가설 4
def plot_observed_vs_predicted_smoking_by_bmi(
    pred_df: pd.DataFrame,
    obs_df: pd.DataFrame,
    save_path: str | Path | None = None,
    show: bool = True,
) -> None:
    plt.figure(figsize=(7, 4))
    plt.scatter(obs_df["bmi_mid"], obs_df["obs_rate"], alpha=0.8, label="Observed (binned mean)")
    plt.plot(pred_df["bmi"], pred_df["pred_prob"], label="Logistic fit")
    plt.xlabel("BMI")
    plt.ylabel("흡연 확률")
    plt.title("BMI에 따른 실제 vs 예측 흡연 확률")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    _maybe_savefig(save_path)
    if show:
        plt.show()
    else:
        plt.close()


def plot_or_forest_bmi(
    or_val: float,
    ci_low: float,
    ci_high: float,
    save_path: str | Path | None = None,
    show: bool = True,
) -> None:
    plt.figure(figsize=(6, 2))
    plt.errorbar(
        x=or_val,
        y=0,
        xerr=[[or_val - ci_low], [ci_high - or_val]],
        fmt="o",
        capsize=4,
    )
    plt.axvline(1, linestyle="--")
    plt.yticks([])
    plt.xlabel("오즈비 (BMI +1 단위당)")
    plt.title("효과 크기: BMI의 오즈비 (95% 신뢰구간)")
    plt.grid(True, axis="x")
    plt.tight_layout()
    _maybe_savefig(save_path)
    if show:
        plt.show()
    else:
        plt.close()


# 가설 5
def plot_proteinuria_rate_by_smoking(
    prop_table: pd.DataFrame,
    status_order=("Normal", "urine protein(Positive)"),
    save_path: str | Path | None = None,
    show: bool = True,
) -> None:
    cols = [c for c in status_order if c in prop_table.columns]
    prop_plot = prop_table[cols]

    x = np.arange(len(prop_plot.columns))
    width = 0.35

    smoking_colors = _get_smoking_palette()
    plt.figure(figsize=(8, 4))
    plt.bar(x - width/2, prop_plot.loc["non-smokers"].values, width, label="non-smokers", color=smoking_colors[0])
    plt.bar(x + width/2, prop_plot.loc["smokers"].values, width, label="smokers", color=smoking_colors[1])

    plt.xticks(x, prop_plot.columns)
    plt.ylabel("비율(%)")
    plt.title("흡연 여부에 따른 단백뇨 양성 비율 비교")
    plt.legend(loc="upper right")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()

    _maybe_savefig(save_path)
    if show:
        plt.show()
    else:
        plt.close()