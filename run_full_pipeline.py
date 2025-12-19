"""
End-to-end Cash Flow Forecasting Pipeline
==========================================
1) Clean raw transactions
2) Build weekly features per entity
3) Train ML models with backtest and short/long forecasts
4) Generate interactive HTML dashboard with AstraZeneca theme

Usage:
    python run_full_pipeline.py

The script is self-contained and safe to run multiple times; it regenerates
processed data and outputs under the existing folders.
"""
from __future__ import annotations

import json
import os
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# AstraZeneca Color Palette
AZ_COLORS = {
    "mulberry": "#830051",      # Primary - Color 1
    "lime_green": "#C4D600",    # Accent - Color 2
    "navy": "#003865",          # Color 3
    "graphite": "#3F4444",      # Color 4
    "light_blue": "#68D2DF",    # Color 5
    "magenta": "#D0006F",       # Color 6
    "purple": "#3C1053",        # Color 7
    "gold": "#F0AB00",          # Color 8
    "positive": "#C4D600",      # Lime green for positive
    "negative": "#D0006F",      # Magenta for negative
    "actual": "#003865",        # Navy for actual
    "forecast_short": "#68D2DF", # Light blue for 1-month
    "forecast_long": "#830051",  # Mulberry for 6-month
    "backtest": "#F0AB00",      # Gold for backtest predictions
}

# Optional boosters
try:  # pragma: no cover - optional dependency
    import xgboost as xgb  # type: ignore

    HAS_XGB = True
except Exception:  # pragma: no cover - optional dependency
    HAS_XGB = False

try:  # pragma: no cover - optional dependency
    import lightgbm as lgb  # type: ignore

    HAS_LGB = True
except Exception:  # pragma: no cover - optional dependency
    HAS_LGB = False


FEATURE_COLS = [
    "Week_of_Month",
    "Is_Month_End",
    "Month",
    "Quarter",
    "Net_Lag1",
    "Net_Lag2",
    "Net_Lag4",
    "Net_Rolling4_Mean",
    "Net_Rolling4_Std",
    "Transaction_Count",
    "Inflow_Count",
    "Outflow_Count",
    "Inflow_Lag1",
    "Outflow_Lag1",
]


class PathConfig:
    def __init__(self) -> None:
        self.base = Path(__file__).resolve().parent
        self.data_dir = self.base / "Data"
        self.raw_main = self.data_dir / "Datathon Dataset.xlsx - Data - Main.csv"
        self.processed = self.base / "processed_data"
        self.outputs = self.base / "outputs"
        self.processed.mkdir(parents=True, exist_ok=True)
        self.outputs.mkdir(parents=True, exist_ok=True)
        self.forecast_dir = self.outputs / "ml_pipeline"
        self.dashboard_dir = self.outputs / "dashboards"
        self.forecast_dir.mkdir(parents=True, exist_ok=True)
        self.dashboard_dir.mkdir(parents=True, exist_ok=True)


class DataCleaner:
    KEEP_COLS = [
        "Name",
        "Period",
        "Account",
        "PK",
        "Offst.acct",
        "Name of offsetting account",
        "Pstng Date",
        "Doc..Date",
        "Amount in USD",
        "LCurr",
        "Category",
    ]

    def __init__(self, paths: PathConfig) -> None:
        self.paths = paths

    def run(self) -> pd.DataFrame:
        if not self.paths.raw_main.exists():
            raise FileNotFoundError(f"Raw file missing: {self.paths.raw_main}")

        df = pd.read_csv(self.paths.raw_main)
        missing = [c for c in self.KEEP_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        df = df[self.KEEP_COLS].copy()
        df["Amount in USD"] = (
            df["Amount in USD"].astype(str).str.replace(",", "", regex=False)
        )
        df["Amount in USD"] = pd.to_numeric(df["Amount in USD"], errors="coerce").fillna(0)
        df["Pstng Date"] = pd.to_datetime(df["Pstng Date"], format="mixed")
        df["Doc..Date"] = pd.to_datetime(df["Doc..Date"], format="mixed")

        output_path = self.paths.processed / "clean_transactions.csv"
        df.to_csv(output_path, index=False)
        print(f"✅ Clean data saved to {output_path} ({len(df):,} rows)")
        return df


class WeeklyAggregator:
    def __init__(self, paths: PathConfig) -> None:
        self.paths = paths

    def _clean_category(self, value: str) -> str:
        return value.replace(" ", "_").replace("/", "_")

    def run(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
        year_start = datetime(df["Pstng Date"].dt.year.min(), 1, 1)
        df = df.copy()
        df["Days_From_Start"] = (df["Pstng Date"] - year_start).dt.days
        df["Week_Num"] = (df["Days_From_Start"] // 7) + 1
        df["Week_Start"] = year_start + pd.to_timedelta((df["Week_Num"] - 1) * 7, unit="D")
        df["Week_End"] = df["Week_Start"] + timedelta(days=6)

        entities = df["Name"].unique()
        weeks = sorted(df["Week_Num"].unique())
        categories = df["Category"].unique()

        weekly_rows: List[Dict[str, object]] = []
        for entity in entities:
            entity_df = df[df["Name"] == entity]
            for week in weeks:
                week_df = entity_df[entity_df["Week_Num"] == int(week)]
                row: Dict[str, object] = {
                    "Entity": entity,
                    "Week_Num": int(week),
                    "Week_Start": year_start + timedelta(days=(int(week) - 1) * 7),
                    "Week_End": year_start + timedelta(days=int(week) * 7 - 1),
                    "Total_Net": week_df["Amount in USD"].sum(),
                    "Total_Inflow": week_df[week_df["Amount in USD"] > 0]["Amount in USD"].sum(),
                    "Total_Outflow": week_df[week_df["Amount in USD"] < 0]["Amount in USD"].sum(),
                }
                row["Outflow_Abs"] = abs(row["Total_Outflow"])
                row["Transaction_Count"] = len(week_df)
                row["Inflow_Count"] = len(week_df[week_df["Amount in USD"] > 0])
                row["Outflow_Count"] = len(week_df[week_df["Amount in USD"] < 0])
                row["Avg_Transaction"] = week_df["Amount in USD"].mean() if len(week_df) else 0
                row["Avg_Inflow"] = week_df[week_df["Amount in USD"] > 0]["Amount in USD"].mean() if row["Inflow_Count"] else 0
                row["Avg_Outflow"] = week_df[week_df["Amount in USD"] < 0]["Amount in USD"].mean() if row["Outflow_Count"] else 0
                row["Max_Transaction"] = week_df["Amount in USD"].max() if len(week_df) else 0
                row["Min_Transaction"] = week_df["Amount in USD"].min() if len(week_df) else 0
                for cat in categories:
                    cat_df = week_df[week_df["Category"] == cat]
                    cat_clean = self._clean_category(str(cat))
                    row[f"Cat_{cat_clean}_Net"] = cat_df["Amount in USD"].sum()
                    row[f"Cat_{cat_clean}_Count"] = len(cat_df)
                row["PK40_Amount"] = week_df[week_df["PK"] == 40]["Amount in USD"].sum()
                row["PK50_Amount"] = week_df[week_df["PK"] == 50]["Amount in USD"].sum()
                row["PK40_Count"] = len(week_df[week_df["PK"] == 40])
                row["PK50_Count"] = len(week_df[week_df["PK"] == 50])
                weekly_rows.append(row)

        weekly_df = pd.DataFrame(weekly_rows).fillna(0)
        weekly_df["Month"] = weekly_df["Week_Start"].dt.month
        weekly_df["Week_of_Month"] = ((weekly_df["Week_Start"].dt.day - 1) // 7) + 1
        weekly_df["Is_Month_End"] = weekly_df["Week_Start"].dt.day > 21
        weekly_df["Quarter"] = weekly_df["Week_Start"].dt.quarter
        weekly_df = weekly_df.sort_values(["Entity", "Week_Num"]).reset_index(drop=True)

        for lag in [1, 2, 4]:
            weekly_df[f"Net_Lag{lag}"] = weekly_df.groupby("Entity")["Total_Net"].shift(lag)
            weekly_df[f"Inflow_Lag{lag}"] = weekly_df.groupby("Entity")["Total_Inflow"].shift(lag)
            weekly_df[f"Outflow_Lag{lag}"] = weekly_df.groupby("Entity")["Total_Outflow"].shift(lag)

        weekly_df["Net_Rolling4_Mean"] = weekly_df.groupby("Entity")["Total_Net"].transform(
            lambda x: x.rolling(window=4, min_periods=1).mean()
        )
        weekly_df["Net_Rolling4_Std"] = weekly_df.groupby("Entity")["Total_Net"].transform(
            lambda x: x.rolling(window=4, min_periods=1).std()
        )
        weekly_df["Cumulative_Net"] = weekly_df.groupby("Entity")["Total_Net"].cumsum()

        weekly_df.to_csv(self.paths.processed / "weekly_entity_features.csv", index=False)
        entity_frames: Dict[str, pd.DataFrame] = {}
        for entity in entities:
            entity_df = weekly_df[weekly_df["Entity"] == entity].copy()
            entity_path = self.paths.processed / f"weekly_{entity}.csv"
            entity_df.to_csv(entity_path, index=False)
            entity_frames[entity] = entity_df
        print(f"✅ Weekly features saved to {self.paths.processed}")
        return weekly_df, entity_frames


@dataclass
class MonthlyForecast:
    """Stores forecast data for a single month."""
    month_num: int  # 1-6
    dates: List[datetime]
    predictions: List[float]
    cumulative_net: float  # Running total up to this month


@dataclass
class ModelResult:
    name: str
    model: object
    scaler: Optional[StandardScaler]
    backtest_pred: np.ndarray
    metrics: Dict[str, float]


@dataclass
class ForecastArtifacts:
    best_model: str
    backtest_dates: pd.Series
    backtest_actual: np.ndarray
    backtest_pred: np.ndarray
    future_short_dates: List[datetime]
    future_short_pred: np.ndarray
    future_long_dates: List[datetime]
    future_long_pred: np.ndarray
    monthly_forecasts: List[MonthlyForecast] = field(default_factory=list)  # Month-by-month breakdown
    metrics: Dict[str, float] = field(default_factory=dict)
    all_model_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)


class MLForecaster:
    def __init__(
        self,
        paths: PathConfig,
        backtest_weeks: int = 4,
        forecast_weeks: int = 4,
        long_horizon_weeks: int = 26,
    ) -> None:
        self.paths = paths
        self.backtest_weeks = backtest_weeks
        self.forecast_weeks = forecast_weeks
        self.long_horizon_weeks = long_horizon_weeks

    def _metrics(self, actual: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
        mae = mean_absolute_error(actual, pred)
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mape_mask = actual != 0
        mape = (
            np.mean(np.abs((actual[mape_mask] - pred[mape_mask]) / actual[mape_mask])) * 100
            if mape_mask.any()
            else np.nan
        )
        direction_acc = (
            np.mean(np.sign(np.diff(actual)) == np.sign(np.diff(pred))) * 100
            if len(actual) > 1
            else np.nan
        )
        sign_acc = np.mean(np.sign(actual) == np.sign(pred)) * 100
        return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "Direction_Accuracy": direction_acc, "Sign_Accuracy": sign_acc}

    def _train_models(
        self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray
    ) -> Dict[str, ModelResult]:
        results: Dict[str, ModelResult] = {}

        def add_model(name: str, estimator, use_scaler: bool = False) -> None:
            scaler = StandardScaler() if use_scaler else None
            Xtr = scaler.fit_transform(X_train) if scaler is not None else X_train
            Xte = scaler.transform(X_test) if scaler is not None else X_test
            estimator.fit(Xtr, y_train)
            pred = estimator.predict(Xte)
            results[name] = ModelResult(
                name=name,
                model=estimator,
                scaler=scaler,
                backtest_pred=pred,
                metrics=self._metrics(y_test, pred),
            )

        if HAS_XGB:
            add_model(
                "XGBoost",
                xgb.XGBRegressor(
                    n_estimators=200,
                    max_depth=4,
                    learning_rate=0.08,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    objective="reg:squarederror",
                    random_state=42,
                    verbosity=0,
                ),
            )

        if HAS_LGB:
            add_model(
                "LightGBM",
                lgb.LGBMRegressor(
                    n_estimators=200,
                    max_depth=5,
                    learning_rate=0.08,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    random_state=42,
                    verbose=-1,
                ),
            )

        add_model(
            "RandomForest",
            RandomForestRegressor(
                n_estimators=300,
                max_depth=8,
                min_samples_split=3,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1,
            ),
        )
        add_model(
            "GradientBoosting",
            GradientBoostingRegressor(n_estimators=250, max_depth=4, learning_rate=0.05, subsample=0.9, random_state=42),
        )
        add_model("Ridge", Ridge(alpha=1.0), use_scaler=True)

        # Ensemble weighted by inverse RMSE
        if results:
            inv_weights = {k: 1 / max(v.metrics["RMSE"], 1e-6) for k, v in results.items()}
            total = sum(inv_weights.values())
            ensemble_pred = np.zeros_like(next(iter(results.values())).backtest_pred)
            for name, res in results.items():
                ensemble_pred += (inv_weights[name] / total) * res.backtest_pred
            results["Ensemble"] = ModelResult(
                name="Ensemble",
                model=None,
                scaler=None,
                backtest_pred=ensemble_pred,
                metrics=self._metrics(y_test, ensemble_pred),
            )
        return results

    def _future_feature_row(
        self,
        target_date: datetime,
        net_history: List[float],
        last_row: pd.Series,
        inflow_history: List[float],
        outflow_history: List[float],
    ) -> Dict[str, float]:
        row = {
            "Week_of_Month": ((target_date.day - 1) // 7) + 1,
            "Is_Month_End": int(target_date.day > 21),
            "Month": target_date.month,
            "Quarter": (target_date.month - 1) // 3 + 1,
            "Net_Lag1": net_history[-1],
            "Net_Lag2": net_history[-2] if len(net_history) > 1 else net_history[-1],
            "Net_Lag4": net_history[-4] if len(net_history) > 3 else net_history[0],
            "Net_Rolling4_Mean": float(np.mean(net_history[-4:])),
            "Net_Rolling4_Std": float(np.std(net_history[-4:])),
            "Transaction_Count": float(last_row.get("Transaction_Count", 0)),
            "Inflow_Count": float(last_row.get("Inflow_Count", 0)),
            "Outflow_Count": float(last_row.get("Outflow_Count", 0)),
            "Inflow_Lag1": inflow_history[-1] if inflow_history else float(last_row.get("Total_Inflow", 0)),
            "Outflow_Lag1": outflow_history[-1] if outflow_history else float(last_row.get("Total_Outflow", 0)),
        }
        return row

    def _iterative_forecast(
        self,
        model_name: str,
        available_features: List[str],
        base_df: pd.DataFrame,
        horizon: int,
    ) -> Tuple[List[datetime], np.ndarray]:
        full_X = base_df[available_features].values
        full_y = base_df["Total_Net"].values

        def build_estimator() -> Tuple[object, Optional[StandardScaler]]:
            if model_name == "XGBoost" and HAS_XGB:
                return (
                    xgb.XGBRegressor(
                        n_estimators=200,
                        max_depth=4,
                        learning_rate=0.08,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        objective="reg:squarederror",
                        random_state=42,
                        verbosity=0,
                    ),
                    None,
                )
            if model_name == "LightGBM" and HAS_LGB:
                return (
                    lgb.LGBMRegressor(
                        n_estimators=200,
                        max_depth=5,
                        learning_rate=0.08,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        random_state=42,
                        verbose=-1,
                    ),
                    None,
                )
            if model_name == "RandomForest":
                return (
                    RandomForestRegressor(
                        n_estimators=300,
                        max_depth=8,
                        min_samples_split=3,
                        min_samples_leaf=2,
                        random_state=42,
                        n_jobs=-1,
                    ),
                    None,
                )
            if model_name == "GradientBoosting":
                return (
                    GradientBoostingRegressor(n_estimators=250, max_depth=4, learning_rate=0.05, subsample=0.9, random_state=42),
                    None,
                )
            # Ridge or fallback
            return Ridge(alpha=1.0), StandardScaler()

        estimator, scaler = build_estimator()
        X_fit = scaler.fit_transform(full_X) if scaler else full_X
        estimator.fit(X_fit, full_y)

        history_net = list(full_y)
        inflow_history = list(base_df.get("Total_Inflow", pd.Series([0] * len(base_df))).values)
        outflow_history = list(base_df.get("Total_Outflow", pd.Series([0] * len(base_df))).values)
        last_row = base_df.iloc[-1]
        cursor_date = pd.to_datetime(last_row["Week_Start"]).to_pydatetime()

        preds: List[float] = []
        dates: List[datetime] = []
        for _ in range(horizon):
            cursor_date = cursor_date + timedelta(days=7)
            feat_row = self._future_feature_row(cursor_date, history_net, last_row, inflow_history, outflow_history)
            X_next = np.array([[feat_row.get(col, 0) for col in available_features]])
            X_next = scaler.transform(X_next) if scaler else X_next
            y_hat = float(estimator.predict(X_next)[0])
            preds.append(y_hat)
            dates.append(cursor_date)
            history_net.append(y_hat)

        return dates, np.array(preds)

    def _iterative_forecast_monthly(
        self,
        model_name: str,
        available_features: List[str],
        base_df: pd.DataFrame,
        num_months: int = 6,
        weeks_per_month: int = 4,
    ) -> Tuple[List[datetime], np.ndarray, List[MonthlyForecast]]:
        """
        Perform iterative month-by-month forecasting.
        Each month's forecast uses predictions from previous months as input.
        """
        full_X = base_df[available_features].values
        full_y = base_df["Total_Net"].values

        def build_estimator() -> Tuple[object, Optional[StandardScaler]]:
            if model_name == "XGBoost" and HAS_XGB:
                return (
                    xgb.XGBRegressor(
                        n_estimators=200, max_depth=4, learning_rate=0.08,
                        subsample=0.9, colsample_bytree=0.9,
                        objective="reg:squarederror", random_state=42, verbosity=0,
                    ),
                    None,
                )
            if model_name == "LightGBM" and HAS_LGB:
                return (
                    lgb.LGBMRegressor(
                        n_estimators=200, max_depth=5, learning_rate=0.08,
                        subsample=0.9, colsample_bytree=0.9, random_state=42, verbose=-1,
                    ),
                    None,
                )
            if model_name == "RandomForest":
                return (
                    RandomForestRegressor(
                        n_estimators=300, max_depth=8, min_samples_split=3,
                        min_samples_leaf=2, random_state=42, n_jobs=-1,
                    ),
                    None,
                )
            if model_name == "GradientBoosting":
                return (
                    GradientBoostingRegressor(
                        n_estimators=250, max_depth=4, learning_rate=0.05,
                        subsample=0.9, random_state=42
                    ),
                    None,
                )
            return Ridge(alpha=1.0), StandardScaler()

        estimator, scaler = build_estimator()
        X_fit = scaler.fit_transform(full_X) if scaler else full_X
        estimator.fit(X_fit, full_y)

        # Initialize history with actual data
        history_net = list(full_y)
        inflow_history = list(base_df.get("Total_Inflow", pd.Series([0] * len(base_df))).values)
        outflow_history = list(base_df.get("Total_Outflow", pd.Series([0] * len(base_df))).values)
        last_row = base_df.iloc[-1]
        cursor_date = pd.to_datetime(last_row["Week_Start"]).to_pydatetime()

        all_dates: List[datetime] = []
        all_preds: List[float] = []
        monthly_forecasts: List[MonthlyForecast] = []
        cumulative_total = 0.0

        # Forecast month by month
        for month_idx in range(num_months):
            month_dates: List[datetime] = []
            month_preds: List[float] = []

            # Forecast 4 weeks for this month
            for week_idx in range(weeks_per_month):
                cursor_date = cursor_date + timedelta(days=7)
                
                # Build features using history (which includes previous months' predictions)
                feat_row = self._future_feature_row(
                    cursor_date, history_net, last_row, inflow_history, outflow_history
                )
                X_next = np.array([[feat_row.get(col, 0) for col in available_features]])
                X_next = scaler.transform(X_next) if scaler else X_next
                
                # Predict
                y_hat = float(estimator.predict(X_next)[0])
                
                # Store prediction
                month_dates.append(cursor_date)
                month_preds.append(y_hat)
                all_dates.append(cursor_date)
                all_preds.append(y_hat)
                
                # Update history with this prediction for next iteration
                history_net.append(y_hat)

            # Calculate cumulative total up to this month
            cumulative_total += sum(month_preds)
            
            # Store monthly forecast
            monthly_forecasts.append(MonthlyForecast(
                month_num=month_idx + 1,
                dates=month_dates,
                predictions=month_preds,
                cumulative_net=cumulative_total,
            ))

        return all_dates, np.array(all_preds), monthly_forecasts

    def forecast_entity(self, entity: str, df: pd.DataFrame) -> Optional[ForecastArtifacts]:
        df = df.copy().sort_values("Week_Start")
        df["Week_Start"] = pd.to_datetime(df["Week_Start"])
        available_features = [c for c in FEATURE_COLS if c in df.columns]
        df = df.dropna(subset=available_features)
        if len(df) < self.backtest_weeks + 8:
            print(f"  Skipping {entity}: not enough history after cleaning.")
            return None

        train_df = df.iloc[:-self.backtest_weeks]
        test_df = df.iloc[-self.backtest_weeks :]
        X_train, y_train = train_df[available_features].values, train_df["Total_Net"].values
        X_test, y_test = test_df[available_features].values, test_df["Total_Net"].values

        results = self._train_models(X_train, y_train, X_test, y_test)
        best_name = min(results.keys(), key=lambda k: results[k].metrics["RMSE"])
        best = results[best_name]

        # 1-month forecast (4 weeks)
        short_dates, short_pred = self._iterative_forecast(best_name, available_features, df, self.forecast_weeks)
        
        # 6-month forecast with month-by-month iteration
        long_dates, long_pred, monthly_forecasts = self._iterative_forecast_monthly(
            best_name, available_features, df, num_months=6, weeks_per_month=4
        )

        print(
            f"  {entity}: {best_name} | RMSE={best.metrics['RMSE']:.2f} | MAE={best.metrics['MAE']:.2f}"
        )
        print(f"    → 6-month forecast generated iteratively (month-by-month)")

        return ForecastArtifacts(
            best_model=best_name,
            backtest_dates=test_df["Week_Start"],
            backtest_actual=test_df["Total_Net"].values,
            backtest_pred=best.backtest_pred,
            future_short_dates=short_dates,
            future_short_pred=short_pred,
            future_long_dates=long_dates,
            future_long_pred=long_pred,
            monthly_forecasts=monthly_forecasts,
            metrics=best.metrics,
            all_model_metrics={name: res.metrics for name, res in results.items()},
        )

    def run(self, entity_frames: Dict[str, pd.DataFrame]) -> Dict[str, ForecastArtifacts]:
        all_results: Dict[str, ForecastArtifacts] = {}
        print("\n=== Training & forecasting ===")
        for entity, frame in entity_frames.items():
            print(f"\nEntity: {entity}")
            art = self.forecast_entity(entity, frame)
            if art:
                all_results[entity] = art
        return all_results


class InteractiveDashboardBuilder:
    """Build interactive HTML dashboard using Plotly with AstraZeneca theme."""

    def __init__(self, paths: PathConfig) -> None:
        self.paths = paths

    def _format_currency(self, value: float) -> str:
        if abs(value) >= 1e6:
            return f"${value/1e6:,.2f}M"
        elif abs(value) >= 1e3:
            return f"${value/1e3:,.1f}K"
        return f"${value:,.0f}"

    def _get_category_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze top cash flow categories."""
        cat_cols = [c for c in df.columns if c.startswith("Cat_") and c.endswith("_Net")]
        if not cat_cols:
            return {"top_inflow": [], "top_outflow": []}

        cat_totals = {}
        for col in cat_cols:
            cat_name = col.replace("Cat_", "").replace("_Net", "").replace("_", " ")
            cat_totals[cat_name] = df[col].sum()

        sorted_cats = sorted(cat_totals.items(), key=lambda x: x[1], reverse=True)
        top_inflow = [(k, v) for k, v in sorted_cats if v > 0][:5]
        top_outflow = [(k, abs(v)) for k, v in sorted_cats if v < 0][:5]

        return {"top_inflow": top_inflow, "top_outflow": top_outflow, "all": cat_totals}

    def _generate_entity_html(
        self,
        entity: str,
        df: pd.DataFrame,
        art: ForecastArtifacts,
    ) -> str:
        """Generate HTML section for one entity."""
        df = df.copy()
        df["Week_Start"] = pd.to_datetime(df["Week_Start"])

        # Calculate key metrics
        total_inflow = df["Total_Inflow"].sum()
        total_outflow = abs(df["Total_Outflow"].sum())
        net_position = df["Total_Net"].sum()
        avg_weekly_net = df["Total_Net"].mean()
        volatility = df["Total_Net"].std()

        # Short-term forecast metrics
        short_forecast_total = float(np.sum(art.future_short_pred))
        short_forecast_avg = float(np.mean(art.future_short_pred))

        # Long-term forecast metrics
        long_forecast_total = float(np.sum(art.future_long_pred))
        long_forecast_avg = float(np.mean(art.future_long_pred))

        # Category analysis
        cat_analysis = self._get_category_analysis(df)

        # Backtest accuracy
        mae = art.metrics.get("MAE", 0)
        rmse = art.metrics.get("RMSE", 0)
        mape = art.metrics.get("MAPE", 0)
        direction_acc = art.metrics.get("Direction_Accuracy", 0)

        # Prepare data for charts (convert to JSON-safe format)
        history_dates = [d.strftime("%Y-%m-%d") for d in df["Week_Start"]]
        history_net = df["Total_Net"].tolist()
        history_inflow = df["Total_Inflow"].tolist()
        history_outflow = [abs(x) for x in df["Total_Outflow"].tolist()]

        backtest_dates = [d.strftime("%Y-%m-%d") for d in art.backtest_dates]
        backtest_actual = art.backtest_actual.tolist()
        backtest_pred = art.backtest_pred.tolist()

        short_dates = [d.strftime("%Y-%m-%d") for d in art.future_short_dates]
        short_pred = art.future_short_pred.tolist()

        long_dates = [d.strftime("%Y-%m-%d") for d in art.future_long_dates]
        long_pred = art.future_long_pred.tolist()

        # Prepare monthly forecast data for visualization
        monthly_data = []
        month_colors = [
            AZ_COLORS["light_blue"],   # Month 1
            AZ_COLORS["lime_green"],   # Month 2
            AZ_COLORS["gold"],         # Month 3
            AZ_COLORS["magenta"],      # Month 4
            AZ_COLORS["purple"],       # Month 5
            AZ_COLORS["mulberry"],     # Month 6
        ]
        for mf in art.monthly_forecasts:
            monthly_data.append({
                "month_num": mf.month_num,
                "dates": [d.strftime("%Y-%m-%d") for d in mf.dates],
                "predictions": mf.predictions,
                "total": sum(mf.predictions),
                "cumulative": mf.cumulative_net,
                "color": month_colors[mf.month_num - 1] if mf.month_num <= len(month_colors) else AZ_COLORS["graphite"],
            })

        # Top categories for display
        top_inflow_cats = cat_analysis.get("top_inflow", [])[:3]
        top_outflow_cats = cat_analysis.get("top_outflow", [])[:3]

        # Liquidity risk assessment
        if short_forecast_total < 0:
            liquidity_status = "⚠️ At Risk"
            liquidity_color = AZ_COLORS["magenta"]
            liquidity_msg = f"Expected net outflow of {self._format_currency(abs(short_forecast_total))} over the next month. Monitor closely."
        elif short_forecast_avg < avg_weekly_net * 0.5:
            liquidity_status = "⚡ Moderate"
            liquidity_color = AZ_COLORS["gold"]
            liquidity_msg = f"Cash flow expected to decrease. Forecasted weekly average: {self._format_currency(short_forecast_avg)}"
        else:
            liquidity_status = "✅ Stable"
            liquidity_color = AZ_COLORS["lime_green"]
            liquidity_msg = f"Healthy cash position expected. Net inflow forecast: {self._format_currency(short_forecast_total)}"

        # Long-term sustainability
        if long_forecast_total < 0:
            sustainability = "Needs Attention"
            sust_color = AZ_COLORS["magenta"]
        elif long_forecast_total < net_position * 0.3:
            sustainability = "Monitor"
            sust_color = AZ_COLORS["gold"]
        else:
            sustainability = "Sustainable"
            sust_color = AZ_COLORS["lime_green"]

        return f'''
        <div class="entity-section" id="{entity}">
            <div class="entity-header">
                <h2><span class="entity-badge">{entity}</span> Cash Flow Analysis</h2>
                <div class="model-badge">Best Model: <strong>{art.best_model}</strong></div>
            </div>

            <!-- KPI Cards Row -->
            <div class="kpi-row">
                <div class="kpi-card">
                    <div class="kpi-icon">💰</div>
                    <div class="kpi-value" style="color: {AZ_COLORS['lime_green']}">{self._format_currency(total_inflow)}</div>
                    <div class="kpi-label">Total Inflow</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-icon">💸</div>
                    <div class="kpi-value" style="color: {AZ_COLORS['magenta']}">{self._format_currency(total_outflow)}</div>
                    <div class="kpi-label">Total Outflow</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-icon">📊</div>
                    <div class="kpi-value" style="color: {AZ_COLORS['navy']}">{self._format_currency(net_position)}</div>
                    <div class="kpi-label">Net Position</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-icon">📈</div>
                    <div class="kpi-value" style="color: {AZ_COLORS['light_blue']}">{self._format_currency(short_forecast_total)}</div>
                    <div class="kpi-label">1-Month Forecast</div>
                </div>
            </div>

            <!-- Main Charts Section -->
            <div class="charts-grid">
                <!-- Backtest Chart -->
                <div class="chart-container full-width">
                    <h3>🔍 Backtest: Actual vs Forecast (Last 4 Weeks)</h3>
                    <p class="chart-description">Comparing model predictions against actual cash flows to evaluate forecast accuracy.</p>
                    <div id="backtest-{entity}" class="chart"></div>
                    <div class="metrics-row">
                        <div class="metric">
                            <span class="metric-label">MAE</span>
                            <span class="metric-value">{self._format_currency(mae)}</span>
                        </div>
                        <div class="metric">
                            <span class="metric-label">RMSE</span>
                            <span class="metric-value">{self._format_currency(rmse)}</span>
                        </div>
                        <div class="metric">
                            <span class="metric-label">MAPE</span>
                            <span class="metric-value">{mape:.1f}%</span>
                        </div>
                        <div class="metric">
                            <span class="metric-label">Direction Accuracy</span>
                            <span class="metric-value">{direction_acc:.1f}%</span>
                        </div>
                    </div>
                </div>

                <!-- Inflow/Outflow Chart -->
                <div class="chart-container full-width">
                    <h3>💹 Cash Flow Overview: Inflow vs Outflow</h3>
                    <p class="chart-description">Weekly breakdown of cash inflows and outflows across the historical period.</p>
                    <div id="inflow-outflow-{entity}" class="chart"></div>
                </div>

                <!-- Short-term Forecast -->
                <div class="chart-container">
                    <h3>📅 1-Month Ahead Forecast</h3>
                    <p class="chart-description">Weekly cash flow predictions for the next 4 weeks based on {art.best_model} model.</p>
                    <div id="short-forecast-{entity}" class="chart"></div>
                    <div class="insight-box" style="border-left-color: {liquidity_color}">
                        <div class="insight-header">
                            <span class="status-badge" style="background: {liquidity_color}">{liquidity_status}</span>
                            <strong>Short-term Liquidity Assessment</strong>
                        </div>
                        <p>{liquidity_msg}</p>
                    </div>
                </div>

                <!-- Long-term Forecast -->
                <div class="chart-container">
                    <h3>📆 6-Month Ahead Forecast (Iterative Month-by-Month)</h3>
                    <p class="chart-description">Each month's forecast uses previous months' predictions as input features, building iteratively.</p>
                    <div id="long-forecast-{entity}" class="chart"></div>
                    <div class="insight-box" style="border-left-color: {sust_color}">
                        <div class="insight-header">
                            <span class="status-badge" style="background: {sust_color}">{sustainability}</span>
                            <strong>Long-term Sustainability</strong>
                        </div>
                        <p>6-month forecasted total: {self._format_currency(long_forecast_total)}. {"Consider strategic adjustments." if sustainability != "Sustainable" else "Position appears sustainable."}</p>
                    </div>
                </div>
            </div>

            <!-- Month-by-Month Forecast Breakdown -->
            <div class="monthly-breakdown">
                <h3>📊 Month-by-Month Forecast Breakdown</h3>
                <p class="chart-description">Iterative forecasting: each month's prediction feeds into the next month's features, creating a cascading forecast.</p>
                <div class="monthly-grid">
                    {"".join([f'''
                    <div class="monthly-card" style="border-top: 4px solid {md['color']}">
                        <div class="monthly-header">
                            <span class="month-badge" style="background: {md['color']}">Month {md['month_num']}</span>
                            <span class="month-dates">{md['dates'][0]} → {md['dates'][-1]}</span>
                        </div>
                        <div class="monthly-stats">
                            <div class="monthly-stat">
                                <span class="stat-value">{self._format_currency(md['total'])}</span>
                                <span class="stat-label">Monthly Net</span>
                            </div>
                            <div class="monthly-stat">
                                <span class="stat-value">{self._format_currency(md['cumulative'])}</span>
                                <span class="stat-label">Cumulative</span>
                            </div>
                        </div>
                        <div class="weekly-detail">
                            {"".join([f'<span class="week-val" style="color: {"#22c55e" if v >= 0 else "#ef4444"}">{self._format_currency(v)}</span>' for v in md['predictions']])}
                        </div>
                    </div>
                    ''' for md in monthly_data])}
                </div>
                <div id="monthly-chart-{entity}" class="chart" style="margin-top: 1.5rem;"></div>
            </div>

            <!-- Category Analysis -->
            <div class="category-analysis">
                <h3>🏷️ Key Cash Flow Drivers</h3>
                <div class="category-grid">
                    <div class="category-box inflow">
                        <h4>Top Inflow Categories</h4>
                        <ul>
                            {"".join([f'<li><span class="cat-name">{cat}</span><span class="cat-value">{self._format_currency(val)}</span></li>' for cat, val in top_inflow_cats]) if top_inflow_cats else '<li>No significant inflows</li>'}
                        </ul>
                    </div>
                    <div class="category-box outflow">
                        <h4>Top Outflow Categories</h4>
                        <ul>
                            {"".join([f'<li><span class="cat-name">{cat}</span><span class="cat-value">-{self._format_currency(val)}</span></li>' for cat, val in top_outflow_cats]) if top_outflow_cats else '<li>No significant outflows</li>'}
                        </ul>
                    </div>
                </div>
            </div>

            <!-- Model Performance Table -->
            <div class="model-comparison">
                <h3>🤖 Model Performance Comparison</h3>
                <table class="model-table">
                    <thead>
                        <tr>
                            <th>Model</th>
                            <th>MAE</th>
                            <th>RMSE</th>
                            <th>MAPE (%)</th>
                            <th>Direction Acc (%)</th>
                        </tr>
                    </thead>
                    <tbody>
                        {"".join([f'''<tr class="{'best-model' if name == art.best_model else ''}">
                            <td>{name} {'⭐' if name == art.best_model else ''}</td>
                            <td>{self._format_currency(m.get('MAE', 0))}</td>
                            <td>{self._format_currency(m.get('RMSE', 0))}</td>
                            <td>{m.get('MAPE', 0):.1f}</td>
                            <td>{m.get('Direction_Accuracy', 0):.1f}</td>
                        </tr>''' for name, m in art.all_model_metrics.items()])}
                    </tbody>
                </table>
            </div>
        </div>

        <script>
        // Backtest Chart
        Plotly.newPlot('backtest-{entity}', [
            {{
                x: {json.dumps(backtest_dates)},
                y: {json.dumps(backtest_actual)},
                type: 'scatter',
                mode: 'lines+markers',
                name: 'Actual',
                line: {{color: '{AZ_COLORS["navy"]}', width: 4}},
                marker: {{size: 14, symbol: 'circle'}}
            }},
            {{
                x: {json.dumps(backtest_dates)},
                y: {json.dumps(backtest_pred)},
                type: 'scatter',
                mode: 'lines+markers',
                name: 'Forecast',
                line: {{color: '{AZ_COLORS["gold"]}', width: 4, dash: 'dash'}},
                marker: {{size: 12, symbol: 'diamond'}}
            }}
        ], {{
            height: 350,
            margin: {{t: 30, b: 50, l: 80, r: 30}},
            xaxis: {{title: 'Week', tickangle: -30}},
            yaxis: {{title: 'Net Cash Flow (USD)', tickformat: ',.0f'}},
            legend: {{x: 0, y: 1.15, orientation: 'h'}},
            hovermode: 'x unified',
            shapes: [{{type: 'line', x0: '{backtest_dates[0]}', x1: '{backtest_dates[-1]}', y0: 0, y1: 0, line: {{color: '#888', width: 1, dash: 'dot'}}}}]
        }}, {{responsive: true}});

        // Inflow/Outflow Chart
        Plotly.newPlot('inflow-outflow-{entity}', [
            {{
                x: {json.dumps(history_dates)},
                y: {json.dumps(history_inflow)},
                type: 'bar',
                name: 'Inflow',
                marker: {{color: '{AZ_COLORS["lime_green"]}'}}
            }},
            {{
                x: {json.dumps(history_dates)},
                y: {json.dumps([-x for x in history_outflow])},
                type: 'bar',
                name: 'Outflow',
                marker: {{color: '{AZ_COLORS["magenta"]}'}}
            }},
            {{
                x: {json.dumps(history_dates)},
                y: {json.dumps(history_net)},
                type: 'scatter',
                mode: 'lines+markers',
                name: 'Net Cash Flow',
                line: {{color: '{AZ_COLORS["navy"]}', width: 3}},
                marker: {{size: 6}}
            }}
        ], {{
            height: 400,
            barmode: 'relative',
            margin: {{t: 30, b: 60, l: 80, r: 30}},
            xaxis: {{title: 'Week', tickangle: -45}},
            yaxis: {{title: 'Amount (USD)', tickformat: ',.0f'}},
            legend: {{x: 0, y: 1.15, orientation: 'h'}},
            hovermode: 'x unified'
        }}, {{responsive: true}});

        // Short-term Forecast Chart
        var lastHistDate = '{history_dates[-1]}';
        var lastHistVal = {history_net[-1]};
        Plotly.newPlot('short-forecast-{entity}', [
            {{
                x: {json.dumps(history_dates[-8:])},
                y: {json.dumps(history_net[-8:])},
                type: 'scatter',
                mode: 'lines+markers',
                name: 'Historical',
                line: {{color: '{AZ_COLORS["navy"]}', width: 3}},
                marker: {{size: 10}}
            }},
            {{
                x: [lastHistDate].concat({json.dumps(short_dates)}),
                y: [lastHistVal].concat({json.dumps(short_pred)}),
                type: 'scatter',
                mode: 'lines+markers',
                name: '1-Month Forecast',
                line: {{color: '{AZ_COLORS["light_blue"]}', width: 4}},
                marker: {{size: 12, symbol: 'diamond'}},
                fill: 'tozeroy',
                fillcolor: 'rgba(104, 210, 223, 0.2)'
            }}
        ], {{
            height: 320,
            margin: {{t: 30, b: 50, l: 70, r: 30}},
            xaxis: {{title: 'Week', tickangle: -30}},
            yaxis: {{title: 'Net Cash Flow (USD)', tickformat: ',.0f'}},
            legend: {{x: 0, y: 1.15, orientation: 'h'}},
            hovermode: 'x unified',
            shapes: [{{type: 'line', x0: lastHistDate, x1: lastHistDate, y0: 0, y1: 1, yref: 'paper', line: {{color: '#888', width: 2, dash: 'dash'}}}}]
        }}, {{responsive: true}});

        // Long-term Forecast Chart (Month-by-Month with different colors)
        var longForecastTraces = [
            {{
                x: {json.dumps(history_dates[-12:])},
                y: {json.dumps(history_net[-12:])},
                type: 'scatter',
                mode: 'lines+markers',
                name: 'Historical',
                line: {{color: '{AZ_COLORS["navy"]}', width: 3}},
                marker: {{size: 8}}
            }}
        ];
        
        // Add each month as a separate trace with different colors
        var monthColors = ['{AZ_COLORS["light_blue"]}', '{AZ_COLORS["lime_green"]}', '{AZ_COLORS["gold"]}', '{AZ_COLORS["magenta"]}', '{AZ_COLORS["purple"]}', '{AZ_COLORS["mulberry"]}'];
        var monthlyData = {json.dumps(monthly_data)};
        var prevEndDate = lastHistDate;
        var prevEndVal = lastHistVal;
        
        monthlyData.forEach(function(month, idx) {{
            var xVals = [prevEndDate].concat(month.dates);
            var yVals = [prevEndVal].concat(month.predictions);
            longForecastTraces.push({{
                x: xVals,
                y: yVals,
                type: 'scatter',
                mode: 'lines+markers',
                name: 'Month ' + month.month_num,
                line: {{color: monthColors[idx], width: 3}},
                marker: {{size: 10, symbol: idx % 2 === 0 ? 'diamond' : 'circle'}},
                fill: 'tozeroy',
                fillcolor: monthColors[idx] + '20'
            }});
            prevEndDate = month.dates[month.dates.length - 1];
            prevEndVal = month.predictions[month.predictions.length - 1];
        }});
        
        Plotly.newPlot('long-forecast-{entity}', longForecastTraces, {{
            height: 350,
            margin: {{t: 30, b: 50, l: 70, r: 30}},
            xaxis: {{title: 'Week', tickangle: -30}},
            yaxis: {{title: 'Net Cash Flow (USD)', tickformat: ',.0f'}},
            legend: {{x: 0, y: 1.2, orientation: 'h'}},
            hovermode: 'x unified',
            shapes: [{{type: 'line', x0: lastHistDate, x1: lastHistDate, y0: 0, y1: 1, yref: 'paper', line: {{color: '#888', width: 2, dash: 'dash'}}}}]
        }}, {{responsive: true}});

        // Monthly Summary Bar Chart
        var monthlyTotals = monthlyData.map(m => m.total);
        var monthlyLabels = monthlyData.map(m => 'Month ' + m.month_num);
        var monthlyBarColors = monthlyTotals.map((v, i) => v >= 0 ? monthColors[i] : '{AZ_COLORS["magenta"]}');
        
        Plotly.newPlot('monthly-chart-{entity}', [
            {{
                x: monthlyLabels,
                y: monthlyTotals,
                type: 'bar',
                marker: {{color: monthlyBarColors}},
                text: monthlyTotals.map(v => v >= 0 ? '+' + (v/1000).toFixed(0) + 'K' : (v/1000).toFixed(0) + 'K'),
                textposition: 'outside',
                hovertemplate: 'Month %{{x}}<br>Net: %{{y:,.0f}}<extra></extra>'
            }},
            {{
                x: monthlyLabels,
                y: monthlyData.map(m => m.cumulative),
                type: 'scatter',
                mode: 'lines+markers',
                name: 'Cumulative',
                yaxis: 'y2',
                line: {{color: '{AZ_COLORS["navy"]}', width: 3}},
                marker: {{size: 10}}
            }}
        ], {{
            height: 300,
            margin: {{t: 30, b: 50, l: 70, r: 70}},
            xaxis: {{title: ''}},
            yaxis: {{title: 'Monthly Net (USD)', tickformat: ',.0f'}},
            yaxis2: {{title: 'Cumulative (USD)', overlaying: 'y', side: 'right', tickformat: ',.0f'}},
            showlegend: true,
            legend: {{x: 0.5, y: 1.15, orientation: 'h', xanchor: 'center'}},
            hovermode: 'x unified'
        }}, {{responsive: true}});
        </script>
        '''

    def build(
        self,
        entity_frames: Dict[str, pd.DataFrame],
        forecast_results: Dict[str, ForecastArtifacts],
        weekly_df: pd.DataFrame,
    ) -> Path:
        """Build complete interactive dashboard."""
        # Aggregate metrics
        total_entities = len(forecast_results)
        total_inflow = weekly_df["Total_Inflow"].sum()
        total_outflow = abs(weekly_df["Total_Outflow"].sum())
        total_net = weekly_df["Total_Net"].sum()

        entity_links = "".join([
            f'<a href="#{entity}" class="nav-link">{entity}</a>'
            for entity in forecast_results.keys()
        ])

        entity_sections = "".join([
            self._generate_entity_html(entity, entity_frames[entity], art)
            for entity, art in forecast_results.items()
        ])

        html_content = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Cash Flow Forecasting Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {{
            --az-mulberry: {AZ_COLORS["mulberry"]};
            --az-lime: {AZ_COLORS["lime_green"]};
            --az-navy: {AZ_COLORS["navy"]};
            --az-graphite: {AZ_COLORS["graphite"]};
            --az-light-blue: {AZ_COLORS["light_blue"]};
            --az-magenta: {AZ_COLORS["magenta"]};
            --az-purple: {AZ_COLORS["purple"]};
            --az-gold: {AZ_COLORS["gold"]};
        }}

        * {{ box-sizing: border-box; margin: 0; padding: 0; }}

        body {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            color: var(--az-graphite);
            line-height: 1.6;
        }}

        .dashboard-header {{
            background: linear-gradient(135deg, var(--az-mulberry) 0%, var(--az-purple) 100%);
            color: white;
            padding: 2rem 3rem;
            box-shadow: 0 4px 20px rgba(0,0,0,0.15);
        }}

        .header-content {{
            max-width: 1600px;
            margin: 0 auto;
        }}

        .dashboard-title {{
            font-size: 2.5rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
        }}

        .dashboard-subtitle {{
            font-size: 1.1rem;
            opacity: 0.9;
            font-weight: 300;
        }}

        .nav-bar {{
            background: white;
            padding: 1rem 3rem;
            border-bottom: 3px solid var(--az-mulberry);
            position: sticky;
            top: 0;
            z-index: 100;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}

        .nav-content {{
            max-width: 1600px;
            margin: 0 auto;
            display: flex;
            gap: 1rem;
            flex-wrap: wrap;
            align-items: center;
        }}

        .nav-label {{
            font-weight: 600;
            color: var(--az-mulberry);
            margin-right: 1rem;
        }}

        .nav-link {{
            padding: 0.5rem 1.2rem;
            background: #f1f3f4;
            border-radius: 25px;
            text-decoration: none;
            color: var(--az-graphite);
            font-weight: 500;
            transition: all 0.3s ease;
        }}

        .nav-link:hover {{
            background: var(--az-mulberry);
            color: white;
            transform: translateY(-2px);
        }}

        .main-content {{
            max-width: 1600px;
            margin: 0 auto;
            padding: 2rem;
        }}

        /* Executive Summary */
        .executive-summary {{
            background: white;
            border-radius: 16px;
            padding: 2rem;
            margin-bottom: 2rem;
            box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        }}

        .summary-title {{
            font-size: 1.5rem;
            color: var(--az-mulberry);
            margin-bottom: 1.5rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }}

        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1.5rem;
        }}

        .summary-card {{
            text-align: center;
            padding: 1.5rem;
            border-radius: 12px;
            background: linear-gradient(145deg, #f8f9fa, #ffffff);
            border: 1px solid #e9ecef;
        }}

        .summary-value {{
            font-size: 2rem;
            font-weight: 700;
            margin-bottom: 0.25rem;
        }}

        .summary-label {{
            font-size: 0.9rem;
            color: #6c757d;
            font-weight: 500;
        }}

        /* Methodology Section */
        .methodology {{
            background: white;
            border-radius: 16px;
            padding: 2rem;
            margin-bottom: 2rem;
            box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        }}

        .methodology h2 {{
            color: var(--az-mulberry);
            margin-bottom: 1rem;
        }}

        .methodology-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 1.5rem;
        }}

        .method-card {{
            padding: 1.5rem;
            border-radius: 12px;
            border-left: 4px solid var(--az-mulberry);
            background: #f8f9fa;
        }}

        .method-card h4 {{
            color: var(--az-navy);
            margin-bottom: 0.5rem;
        }}

        .method-card p {{
            font-size: 0.95rem;
            color: #555;
        }}

        /* Entity Section */
        .entity-section {{
            background: white;
            border-radius: 16px;
            padding: 2rem;
            margin-bottom: 2rem;
            box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        }}

        .entity-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1.5rem;
            flex-wrap: wrap;
            gap: 1rem;
        }}

        .entity-header h2 {{
            font-size: 1.75rem;
            color: var(--az-graphite);
        }}

        .entity-badge {{
            background: var(--az-mulberry);
            color: white;
            padding: 0.3rem 0.8rem;
            border-radius: 8px;
            font-size: 1.2rem;
        }}

        .model-badge {{
            background: var(--az-light-blue);
            padding: 0.5rem 1rem;
            border-radius: 20px;
            font-size: 0.9rem;
        }}

        /* KPI Cards */
        .kpi-row {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 1rem;
            margin-bottom: 2rem;
        }}

        .kpi-card {{
            background: linear-gradient(145deg, #ffffff, #f8f9fa);
            border-radius: 12px;
            padding: 1.25rem;
            text-align: center;
            border: 1px solid #e9ecef;
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }}

        .kpi-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 8px 25px rgba(0,0,0,0.1);
        }}

        .kpi-icon {{
            font-size: 1.5rem;
            margin-bottom: 0.5rem;
        }}

        .kpi-value {{
            font-size: 1.5rem;
            font-weight: 700;
        }}

        .kpi-label {{
            font-size: 0.85rem;
            color: #6c757d;
            margin-top: 0.25rem;
        }}

        /* Charts */
        .charts-grid {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 1.5rem;
            margin-bottom: 2rem;
        }}

        .chart-container {{
            background: #fafbfc;
            border-radius: 12px;
            padding: 1.5rem;
            border: 1px solid #e9ecef;
        }}

        .chart-container.full-width {{
            grid-column: 1 / -1;
        }}

        .chart-container h3 {{
            font-size: 1.1rem;
            color: var(--az-navy);
            margin-bottom: 0.5rem;
        }}

        .chart-description {{
            font-size: 0.9rem;
            color: #6c757d;
            margin-bottom: 1rem;
        }}

        .chart {{
            width: 100%;
            min-height: 300px;
        }}

        .metrics-row {{
            display: flex;
            gap: 1.5rem;
            margin-top: 1rem;
            padding-top: 1rem;
            border-top: 1px solid #e9ecef;
            flex-wrap: wrap;
        }}

        .metric {{
            display: flex;
            flex-direction: column;
        }}

        .metric-label {{
            font-size: 0.8rem;
            color: #6c757d;
            text-transform: uppercase;
        }}

        .metric-value {{
            font-size: 1.1rem;
            font-weight: 600;
            color: var(--az-navy);
        }}

        /* Insight Box */
        .insight-box {{
            background: #f8f9fa;
            border-left: 4px solid var(--az-mulberry);
            padding: 1rem;
            border-radius: 0 8px 8px 0;
            margin-top: 1rem;
        }}

        .insight-header {{
            display: flex;
            align-items: center;
            gap: 0.75rem;
            margin-bottom: 0.5rem;
        }}

        .status-badge {{
            padding: 0.25rem 0.75rem;
            border-radius: 15px;
            font-size: 0.8rem;
            color: white;
            font-weight: 600;
        }}

        /* Category Analysis */
        .category-analysis {{
            margin-bottom: 2rem;
        }}

        .category-analysis h3 {{
            color: var(--az-navy);
            margin-bottom: 1rem;
        }}

        .category-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 1.5rem;
        }}

        .category-box {{
            padding: 1.5rem;
            border-radius: 12px;
            border: 1px solid #e9ecef;
        }}

        .category-box.inflow {{
            background: linear-gradient(145deg, rgba(196, 214, 0, 0.1), rgba(196, 214, 0, 0.05));
            border-left: 4px solid var(--az-lime);
        }}

        .category-box.outflow {{
            background: linear-gradient(145deg, rgba(208, 0, 111, 0.1), rgba(208, 0, 111, 0.05));
            border-left: 4px solid var(--az-magenta);
        }}

        .category-box h4 {{
            margin-bottom: 1rem;
            color: var(--az-graphite);
        }}

        .category-box ul {{
            list-style: none;
        }}

        .category-box li {{
            display: flex;
            justify-content: space-between;
            padding: 0.5rem 0;
            border-bottom: 1px dashed #e9ecef;
        }}

        .cat-name {{
            font-weight: 500;
        }}

        .cat-value {{
            font-weight: 600;
        }}

        /* Monthly Breakdown */
        .monthly-breakdown {{
            background: #fafbfc;
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 2rem;
            border: 1px solid #e9ecef;
        }}

        .monthly-breakdown h3 {{
            color: var(--az-navy);
            margin-bottom: 0.5rem;
        }}

        .monthly-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 1rem;
            margin-top: 1rem;
        }}

        .monthly-card {{
            background: white;
            border-radius: 10px;
            padding: 1rem;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }}

        .monthly-card:hover {{
            transform: translateY(-3px);
            box-shadow: 0 6px 20px rgba(0,0,0,0.1);
        }}

        .monthly-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 0.75rem;
            flex-wrap: wrap;
            gap: 0.5rem;
        }}

        .month-badge {{
            padding: 0.25rem 0.6rem;
            border-radius: 12px;
            font-size: 0.75rem;
            color: white;
            font-weight: 600;
        }}

        .month-dates {{
            font-size: 0.7rem;
            color: #6c757d;
        }}

        .monthly-stats {{
            display: flex;
            justify-content: space-between;
            gap: 0.5rem;
            margin-bottom: 0.75rem;
        }}

        .monthly-stat {{
            text-align: center;
            flex: 1;
        }}

        .stat-value {{
            display: block;
            font-size: 1rem;
            font-weight: 700;
            color: var(--az-navy);
        }}

        .stat-label {{
            font-size: 0.65rem;
            color: #6c757d;
            text-transform: uppercase;
        }}

        .weekly-detail {{
            display: flex;
            gap: 0.25rem;
            flex-wrap: wrap;
            justify-content: center;
        }}

        .week-val {{
            font-size: 0.65rem;
            padding: 0.15rem 0.4rem;
            background: #f1f3f4;
            border-radius: 4px;
            font-weight: 500;
        }}

        /* Model Table */
        .model-comparison {{
            margin-bottom: 1rem;
        }}

        .model-comparison h3 {{
            color: var(--az-navy);
            margin-bottom: 1rem;
        }}

        .model-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.95rem;
        }}

        .model-table th {{
            background: var(--az-navy);
            color: white;
            padding: 0.75rem;
            text-align: left;
        }}

        .model-table td {{
            padding: 0.75rem;
            border-bottom: 1px solid #e9ecef;
        }}

        .model-table tr:hover {{
            background: #f8f9fa;
        }}

        .model-table .best-model {{
            background: rgba(196, 214, 0, 0.2);
            font-weight: 600;
        }}

        /* Responsive */
        @media (max-width: 768px) {{
            .charts-grid {{
                grid-template-columns: 1fr;
            }}
            .dashboard-header {{
                padding: 1.5rem;
            }}
            .dashboard-title {{
                font-size: 1.75rem;
            }}
        }}

        /* Print styles */
        @media print {{
            .nav-bar {{ display: none; }}
            .entity-section {{ break-inside: avoid; }}
        }}
    </style>
</head>
<body>
    <header class="dashboard-header">
        <div class="header-content">
            <h1 class="dashboard-title">💰 Cash Flow Forecasting Dashboard</h1>
            <p class="dashboard-subtitle">Time-Series Analysis & ML-Based Predictions | Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}</p>
        </div>
    </header>

    <nav class="nav-bar">
        <div class="nav-content">
            <span class="nav-label">Jump to Entity:</span>
            {entity_links}
        </div>
    </nav>

    <main class="main-content">
        <!-- Executive Summary -->
        <section class="executive-summary">
            <h2 class="summary-title">📊 Executive Summary</h2>
            <div class="summary-grid">
                <div class="summary-card">
                    <div class="summary-value" style="color: var(--az-navy)">{total_entities}</div>
                    <div class="summary-label">Entities Analyzed</div>
                </div>
                <div class="summary-card">
                    <div class="summary-value" style="color: var(--az-lime)">{self._format_currency(total_inflow)}</div>
                    <div class="summary-label">Total Inflows</div>
                </div>
                <div class="summary-card">
                    <div class="summary-value" style="color: var(--az-magenta)">{self._format_currency(total_outflow)}</div>
                    <div class="summary-label">Total Outflows</div>
                </div>
                <div class="summary-card">
                    <div class="summary-value" style="color: var(--az-mulberry)">{self._format_currency(total_net)}</div>
                    <div class="summary-label">Net Position</div>
                </div>
            </div>
        </section>

        <!-- Methodology Section -->
        <section class="methodology">
            <h2>📚 Forecasting Methodology</h2>
            <div class="methodology-grid">
                <div class="method-card">
                    <h4>🎯 Model Selection</h4>
                    <p>We employ an ensemble of ML models including <strong>XGBoost</strong>, <strong>LightGBM</strong>, <strong>Random Forest</strong>, <strong>Gradient Boosting</strong>, and <strong>Ridge Regression</strong>. The best model is selected based on lowest RMSE during backtesting.</p>
                </div>
                <div class="method-card">
                    <h4>📈 Key Features</h4>
                    <p>Models use <strong>lag features</strong> (1, 2, 4 weeks), <strong>rolling statistics</strong> (4-week mean/std), <strong>temporal patterns</strong> (week of month, quarter), and <strong>transaction counts</strong> to capture cash flow dynamics.</p>
                </div>
                <div class="method-card">
                    <h4>🔍 Backtest Validation</h4>
                    <p>The last <strong>4 weeks</strong> are held out for backtesting. We measure accuracy using MAE, RMSE, MAPE, and direction accuracy to ensure reliable forecasts.</p>
                </div>
                <div class="method-card">
                    <h4>⚠️ Limitations</h4>
                    <p>Forecasts assume historical patterns continue. External shocks, policy changes, or market disruptions may cause deviations. Longer horizons have higher uncertainty.</p>
                </div>
            </div>
        </section>

        <!-- Entity Sections -->
        {entity_sections}

    </main>

    <footer style="text-align: center; padding: 2rem; color: #6c757d; font-size: 0.9rem;">
        <p>🏢 Global Finance Services & Group Controllers | Cash Flow Analytics Platform</p>
        <p style="margin-top: 0.5rem;">Dashboard generated using Python ML Pipeline with AstraZeneca Theme</p>
    </footer>
</body>
</html>'''

        output_path = self.paths.dashboard_dir / "interactive_dashboard.html"
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        return output_path


def main() -> None:
    paths = PathConfig()
    print("=" * 70)
    print("  CASH FLOW FORECASTING PIPELINE")
    print("  Generating Interactive Dashboard with AstraZeneca Theme")
    print("=" * 70)

    print("\n=== Step 1: Cleaning raw data ===")
    cleaner = DataCleaner(paths)
    clean_df = cleaner.run()

    print("\n=== Step 2: Building weekly features ===")
    aggregator = WeeklyAggregator(paths)
    weekly_df, entity_frames = aggregator.run(clean_df)

    print("\n=== Step 3: Training ML models & generating forecasts ===")
    forecaster = MLForecaster(paths)
    forecast_results = forecaster.run(entity_frames)

    print("\n=== Step 4: Building interactive HTML dashboard ===")
    dashboard = InteractiveDashboardBuilder(paths)
    dash_path = dashboard.build(entity_frames, forecast_results, weekly_df)
    print(f"  ✅ Interactive dashboard saved to: {dash_path}")

    print("\n" + "=" * 70)
    print("  ✅ PIPELINE COMPLETE!")
    print("=" * 70)
    print(f"  📁 Clean data:      {paths.processed / 'clean_transactions.csv'}")
    print(f"  📁 Weekly features: {paths.processed / 'weekly_entity_features.csv'}")
    print(f"  📊 Dashboard:       {dash_path}")
    print(f"\n  Open the dashboard in your browser to explore the interactive visualizations!")


if __name__ == "__main__":
    main()
