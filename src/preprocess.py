from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from typing import Iterable, Tuple, Dict
import pandas as pd


RAW_DEFAULT = Path("data/raw/smoking_health_data.csv")
OUT_DEFAULT = Path("data/processed/smoking_health_processed.csv")

# ---------------------------------------------------------------------
KOR2ENG = {
    "ID": "id",
    "label": "label", 

    "나이": "age",
    "키(cm)": "height_cm",
    "몸무게(kg)": "weight_kg",
    "BMI": "bmi",

    "시력": "vision",
    "공복 혈당": "fasting_glucose",
    "혈압": "systolic_bp",
    "중성 지방": "triglycerides",

    "혈청 크레아티닌": "serum_creatinine",
    "콜레스테롤": "cholesterol",
    "고밀도지단백": "hdl",
    "저밀도지단백": "ldl",
    "헤모글로빈": "hemoglobin",
    "간 효소율": "liver_enzyme",

    "충치": "dental_caries",
    "요 단백": "urine_protein",
}

# 영문화 후 표준 컬럼명
ID_COL = "id"
LABEL_COL = "label"

MISSING_COLS = ("vision", "fasting_glucose", "systolic_bp", "triglycerides")

NUMERIC_COLS = (
    "age",
    "height_cm",
    "weight_kg",
    "bmi",
    "vision",
    "fasting_glucose",
    "systolic_bp",
    "triglycerides",
    "serum_creatinine",
    "cholesterol",
    "hdl",
    "ldl",
    "hemoglobin",
    "liver_enzyme",
)

# 범주/순서형(정수형 유지)
ORDINAL_COLS = ("dental_caries", "urine_protein")


def rename_columns_to_english(df: pd.DataFrame) -> pd.DataFrame:
    """
    모든 컬럼명을 영어로 변경
    - KOR2ENG에 있는 컬럼만 rename
    - 나머지 컬럼은 그대로 유지
    """
    df = df.copy()
    df.columns = [c.strip() if isinstance(c, str) else c for c in df.columns] # df.columns(컬럼명들)을 한 번씩 돌면서 문자열이면 양쪽 공백을 제거(strip) 하고, 문자열이 아니면 그대로 둠둠
    return df.rename(columns=KOR2ENG)

# CSV 원본(raw) 데이터를 불러오는 함수
# 파일을 읽고 컬럼명을 영어로 바꿔서 반환
def load_raw(path: str | Path = RAW_DEFAULT, rename_english: bool = True) -> pd.DataFrame:
    df = pd.read_csv(Path(path))
    return rename_columns_to_english(df) if rename_english else df

# NUMERIC_COLS / LABEL_COL / ORDINAL_COLS에 속한 컬럼들을 숫자로 바꾸고, 숫자로 못 바꾸는 값은 NaN
def _to_numeric(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for c in (*NUMERIC_COLS, LABEL_COL):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    for c in ORDINAL_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def add_age_group(df: pd.DataFrame, age_col: str = "age") -> pd.DataFrame:
    df = df.copy()
    if age_col not in df.columns:
        return df
    df["age_group"] = (df[age_col] // 10 * 10).astype("Int64")
    return df


def impute_missing_custom(df: pd.DataFrame) -> pd.DataFrame:
    """
    결측치 대체 규칙:
    - vision: 최빈값(mode)
    - triglycerides: age_group별 평균(mean)
    - systolic_bp: 중앙값(median)
    - fasting_glucose: 평균(mean)
    """
    df = df.copy()
    if "age_group" not in df.columns:
        df = add_age_group(df)

    # 1) vision -> mode
    if "vision" in df.columns:
        s = df["vision"].dropna()
        if not s.empty:
            mode_vals = s.mode()
            fill_val = mode_vals.iloc[0] if not mode_vals.empty else s.median()
            df["vision"] = df["vision"].fillna(fill_val)

    # 2) triglycerides -> age_group mean
    if "triglycerides" in df.columns:
        if "age_group" in df.columns:
            group_mean = df.groupby("age_group")["triglycerides"].transform("mean")
            df["triglycerides"] = df["triglycerides"].fillna(group_mean)

        overall_mean = df["triglycerides"].mean()
        df["triglycerides"] = df["triglycerides"].fillna(overall_mean)

    # 3) systolic_bp -> median
    if "systolic_bp" in df.columns:
        med = df["systolic_bp"].median()
        df["systolic_bp"] = df["systolic_bp"].fillna(med)

    # 4) fasting_glucose -> mean
    if "fasting_glucose" in df.columns:
        mean = df["fasting_glucose"].mean()
        df["fasting_glucose"] = df["fasting_glucose"].fillna(mean)

    return df


def preprocess(
    df: pd.DataFrame,
    drop_id: bool = True,
    do_outlier_clip: bool = True,
    rename_english: bool = True,
) -> pd.DataFrame:
    """
    최종 전처리(기존 맥락 유지 + 컬럼 영문화 옵션 추가):
    0) (옵션) 컬럼 영문화
    1) 타입 정리(수치형/범주형)
    2) 중복 제거(id 기준)
    3) age_group 생성
    4) 결측치 사용자 규칙대로 대체
    5) 라벨/순서형 타입 정리
    """
    df = df.copy()

    # 0) 컬럼 영문화
    if rename_english:
        df = rename_columns_to_english(df)

    # 1) 타입 정리
    df = _to_numeric(df)

    # 2) 중복 제거
    if ID_COL in df.columns:
        df = df.drop_duplicates(subset=[ID_COL], keep="first").reset_index(drop=True)

    # 3) age_group
    df = add_age_group(df)

    # 4) 결측치 대체
    df = impute_missing_custom(df)

    # 5) 라벨 / 순서형 정리
    if LABEL_COL in df.columns:
        df[LABEL_COL] = df[LABEL_COL].astype(int)

    for c in ORDINAL_COLS:
        if c in df.columns:
            df[c] = df[c].round().astype(int)

    # id 제거
    if drop_id and ID_COL in df.columns:
        df = df.drop(columns=[ID_COL])

    return df


def save_processed(df: pd.DataFrame, out_path: str | Path = OUT_DEFAULT) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, encoding="utf-8-sig")



# 이상치 탐지
def detect_outliers_iqr(
    df: pd.DataFrame,
    column: str,
    k: float = 1.5,
) -> tuple[pd.DataFrame, float, float]:
    """
    IQR 기반 이상치 탐지.
    반환:
      - outliers_df: column이 lower~upper 밖인 행들
      - lower_bound, upper_bound
    """
    if column not in df.columns:
        raise KeyError(f"Column not found: {column}")

    s = pd.to_numeric(df[column], errors="coerce")
    s_non_na = s.dropna()

    q1 = float(s_non_na.quantile(0.25))
    q3 = float(s_non_na.quantile(0.75))
    iqr = q3 - q1

    lower = q1 - k * iqr
    upper = q3 + k * iqr

    mask = (s < lower) | (s > upper)
    outliers_df = df.loc[mask].copy()
    return outliers_df, float(lower), float(upper)


def summarize_outliers_iqr(
    df: pd.DataFrame,
    columns: Iterable[str],
    k: float = 1.5,
) -> tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    여러 컬럼에 대해 이상치 요약 테이블 + 컬럼별 이상치 행 dict 반환
    """
    rows = []
    outlier_map: Dict[str, pd.DataFrame] = {}

    n = len(df)
    for col in columns:
        out_df, lower, upper = detect_outliers_iqr(df, col, k=k)
        cnt = len(out_df)
        rate = (cnt / n) if n else 0.0

        s = pd.to_numeric(df[col], errors="coerce")
        rows.append(
            {
                "column": col,
                "n": n,
                "outliers": cnt,
                "outlier_rate": round(rate, 6),
                "lower": lower,
                "upper": upper,
                "min": float(np.nanmin(s)) if np.isfinite(np.nanmin(s)) else float("nan"),
                "max": float(np.nanmax(s)) if np.isfinite(np.nanmax(s)) else float("nan"),
            }
        )
        outlier_map[col] = out_df

    summary = pd.DataFrame(rows).sort_values("outliers", ascending=False).reset_index(drop=True)
    return summary, outlier_map