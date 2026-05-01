import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import re

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from pathlib import Path
from sklearn.base import clone

PM25_FILE = Path("weighted-pm2.5.xlsx")
WEATHER_FILE = Path("Weather data(2015-2025).xlsx")
VEHICLE_FILE = Path("vehicle_regression.xlsx")

OUTPUT_DIR = Path("pm25_project_outputs")
PLOT_DIR = OUTPUT_DIR / "plots"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TEST_RATIO = 0.2


def clean_column_names(df):
    df = df.copy()
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"[^\w]+", "_", regex=True)
        .str.replace(r"_+", "_", regex=True)
        .str.strip("_")
    )
    return df

def evaluate_regression(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred)
    }

def chronological_split(df, test_ratio=0.2):
    df = df.sort_values("date").reset_index(drop=True)
    split_idx = int(len(df) * (1 - test_ratio))
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    return train_df, test_df

def make_pipeline(model, scale=False):
    steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", model))
    return Pipeline(steps)

def plot_actual_vs_pred(dates, y_true, y_pred, title, save_path=None):
    plt.figure(figsize=(10, 5))
    plt.plot(dates, y_true, marker="o", label="Actual")
    plt.plot(dates, y_pred, marker="o", label="Predicted")
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("PM2.5")
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

def plot_bar(df_plot, x, y, title, save_path=None):
    plt.figure(figsize=(10, 5))
    sns.barplot(data=df_plot, x=x, y=y)
    plt.title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


pm25_wide = pd.read_excel(PM25_FILE)
pm25_wide = clean_column_names(pm25_wide)

pm25_long = pm25_wide.melt(
    id_vars=["year"],
    var_name="month_col",
    value_name="pm25_target"
)

pm25_long["month"] = pm25_long["month_col"].str.extract(r"(\d+)").astype(int)
pm25_long["year"] = pd.to_numeric(pm25_long["year"], errors="coerce")
pm25_long["pm25_target"] = pd.to_numeric(pm25_long["pm25_target"], errors="coerce")

pm25_long["date"] = pd.to_datetime(
    dict(year=pm25_long["year"], month=pm25_long["month"], day=1),
    errors="coerce"
)

pm25_df = (
    pm25_long[["date", "year", "month", "pm25_target"]]
    .dropna(subset=["date", "pm25_target"])
    .sort_values("date")
    .reset_index(drop=True)
)




xls = pd.ExcelFile(WEATHER_FILE)
weather_sheet_names = xls.sheet_names

weather_frames = []

for sheet_name in weather_sheet_names:
    if re.fullmatch(r"\d{4}", str(sheet_name).strip()):
        year = int(sheet_name)

        temp_df = pd.read_excel(WEATHER_FILE, sheet_name=sheet_name)
        temp_df = clean_column_names(temp_df)

        col_map = {}

        for col in temp_df.columns:
            if col == "month":
                col_map[col] = "month"
            elif "temperature" in col:
                col_map[col] = "mean_temp"
            elif "humidity" in col:
                col_map[col] = "mean_humidity"
            elif "rainfall" in col:
                col_map[col] = "rainfall"
            elif "pm2" in col or "particulates" in col:
                col_map[col] = "weather_pm25_station"

        temp_df = temp_df.rename(columns=col_map)

        needed_cols = ["month", "mean_temp", "mean_humidity", "rainfall", "weather_pm25_station"]
        existing_cols = [c for c in needed_cols if c in temp_df.columns]
        temp_df = temp_df[existing_cols].copy()

        for col in existing_cols:
            temp_df[col] = pd.to_numeric(temp_df[col], errors="coerce")

        temp_df = temp_df[temp_df["month"].between(1, 12, inclusive="both")].copy()
        temp_df["month"] = temp_df["month"].astype(int)
        temp_df["year"] = year
        temp_df["date"] = pd.to_datetime(
            dict(year=temp_df["year"], month=temp_df["month"], day=1),
            errors="coerce"
        )

        weather_frames.append(temp_df)

weather_df = pd.concat(weather_frames, ignore_index=True)
weather_df = weather_df.sort_values("date").reset_index(drop=True)


vehicle_df = pd.read_excel(VEHICLE_FILE)
vehicle_df = clean_column_names(vehicle_df)

vehicle_df["year"] = pd.to_numeric(vehicle_df["year"], errors="coerce")
vehicle_df["month"] = pd.to_numeric(vehicle_df["month"], errors="coerce")
vehicle_df["total_vehicles"] = pd.to_numeric(vehicle_df["total_vehicles"], errors="coerce")
vehicle_df["diesel_share_pct"] = pd.to_numeric(vehicle_df["diesel_share_pct"], errors="coerce")
vehicle_df["electric_share_pct"] = pd.to_numeric(vehicle_df["electric_share_pct"], errors="coerce")

vehicle_df["date"] = pd.to_datetime(
    dict(year=vehicle_df["year"], month=vehicle_df["month"], day=1),
    errors="coerce"
)

vehicle_df["diesel_share"] = vehicle_df["diesel_share_pct"]
vehicle_df["ev_share"] = vehicle_df["electric_share_pct"]

vehicle_df = vehicle_df[[
    "date", "year", "month", "total_vehicles", "diesel_share", "ev_share"
]].dropna(subset=["date"]).sort_values("date").reset_index(drop=True)



df = (
    pm25_df
    .merge(
        weather_df[["date", "mean_temp", "mean_humidity", "rainfall", "weather_pm25_station"]],
        on="date",
        how="inner"
    )
    .merge(
        vehicle_df[["date", "total_vehicles", "diesel_share", "ev_share"]],
        on="date",
        how="inner"
    )
    .sort_values("date")
    .reset_index(drop=True)
)


df.to_csv(OUTPUT_DIR / "merged_model_data.csv", index=False)




plt.figure(figsize=(10, 5))
plt.plot(df["date"], df["pm25_target"], marker="o")
plt.title("Weighted Monthly PM2.5 Trend")
plt.xlabel("Date")
plt.ylabel("PM2.5")
plt.tight_layout()
plt.savefig(PLOT_DIR / "pm25_trend.png", dpi=300, bbox_inches="tight")
plt.show()

plt.figure(figsize=(10, 5))
for col in ["mean_temp", "mean_humidity", "rainfall"]:
    plt.plot(df["date"], df[col], marker="o", label=col)
plt.title("Weather Trends")
plt.xlabel("Date")
plt.ylabel("Value")
plt.legend()
plt.tight_layout()
plt.savefig(PLOT_DIR / "weather_trends.png", dpi=300, bbox_inches="tight")
plt.show()

plt.figure(figsize=(10, 5))
for col in ["total_vehicles", "diesel_share", "ev_share"]:
    plt.plot(df["date"], df[col], marker="o", label=col)
plt.title("Vehicle Trends")
plt.xlabel("Date")
plt.ylabel("Value")
plt.legend()
plt.tight_layout()
plt.savefig(PLOT_DIR / "vehicle_trends.png", dpi=300, bbox_inches="tight")
plt.show()

df.select_dtypes(include=[np.number]).hist(figsize=(12, 10), bins=20)
plt.suptitle("Histograms of Numeric Variables")
plt.tight_layout()
plt.savefig(PLOT_DIR / "histograms.png", dpi=300, bbox_inches="tight")
plt.show()

corr_cols = ["pm25_target", "mean_temp", "mean_humidity", "rainfall", "total_vehicles", "diesel_share", "ev_share"]
plt.figure(figsize=(12, 8))
sns.heatmap(df[corr_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig(PLOT_DIR / "correlation_heatmap.png", dpi=300, bbox_inches="tight")
plt.show()


target_col = "pm25_target"

features_A = ["mean_temp", "mean_humidity", "rainfall"]
features_B = ["mean_temp", "mean_humidity", "rainfall", "total_vehicles", "ev_share"]
features_C = ["mean_temp", "mean_humidity", "rainfall", "total_vehicles", "diesel_share", "ev_share"]

feature_sets = {
    "WeatherOnly": features_A,
    "WeatherPlusVehicles": features_B,
    "WeatherPlusVehiclesPlusDiesel": features_C
}


train_df, test_df = chronological_split(df, test_ratio=TEST_RATIO)



models = {
    "LinearRegression": make_pipeline(LinearRegression(), scale=True),
    "Ridge": make_pipeline(Ridge(alpha=1.0), scale=True),
    "RandomForest": make_pipeline(
        RandomForestRegressor(
            n_estimators=300,
            random_state=RANDOM_STATE
        ),
        scale=False
    )
}

results = []
prediction_store = {}

for feature_set_name, feature_cols in feature_sets.items():
    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_test = test_df[feature_cols]
    y_test = test_df[target_col]

    for model_name, model in models.items():
        fitted_model = clone(model)
        fitted_model.fit(X_train, y_train)
        y_pred = fitted_model.predict(X_test)

        metrics = evaluate_regression(y_test, y_pred)

        results.append({
            "Model": model_name,
            "Feature_Set": feature_set_name,
            "Features": ", ".join(feature_cols),
            "MAE": metrics["MAE"],
            "RMSE": metrics["RMSE"],
            "R2": metrics["R2"]
        })

        prediction_store[(model_name, feature_set_name)] = {
            "model": fitted_model,
            "features": feature_cols,
            "dates": test_df["date"],
            "y_true": y_test,
            "y_pred": y_pred
        }

results_df = pd.DataFrame(results).sort_values(["RMSE", "MAE"], ascending=True)


results_df.to_csv(OUTPUT_DIR / "model_results.csv", index=False)


for (model_name, feature_set_name), obj in prediction_store.items():
    plot_actual_vs_pred(
        obj["dates"],
        obj["y_true"],
        obj["y_pred"],
        title=f"Actual vs Predicted - {model_name} - {feature_set_name}",
        save_path=PLOT_DIR / f"actual_vs_pred_{model_name}_{feature_set_name}.png"
    )


best_row = results_df.iloc[0]
best_model_name = best_row["Model"]
best_feature_set = best_row["Feature_Set"]

best_obj = prediction_store[(best_model_name, best_feature_set)]
best_model = best_obj["model"]
best_features = list(best_obj["features"])

print("\nBest model:")
print(best_row)

if best_model_name in ["LinearRegression", "Ridge"]:
    final_model = best_model.named_steps["model"]

    coefs = np.ravel(final_model.coef_)   

    print("\nNumber of features:", len(best_features))
    print("Number of coefficients:", len(coefs))

    min_len = min(len(best_features), len(coefs))

    coef_df = pd.DataFrame({
        "feature": best_features[:min_len],
        "coefficient": coefs[:min_len]
    }).sort_values("coefficient", key=np.abs, ascending=False)

    print("\nCoefficients:")
    print(coef_df)

    coef_df.to_csv(
        OUTPUT_DIR / f"coefficients_{best_model_name}_{best_feature_set}.csv",
        index=False
    )

    plt.figure(figsize=(8, 5))
    sns.barplot(data=coef_df, x="coefficient", y="feature")
    plt.title(f"Coefficients - {best_model_name} - {best_feature_set}")
    plt.tight_layout()
    plt.savefig(
        PLOT_DIR / f"coefficients_{best_model_name}_{best_feature_set}.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.show()

elif best_model_name == "RandomForest":
    final_model = best_model.named_steps["model"]

    importances = np.ravel(final_model.feature_importances_)  

    print("\nNumber of features:", len(best_features))
    print("Number of importances:", len(importances))

    min_len = min(len(best_features), len(importances))

    imp_df = pd.DataFrame({
        "feature": best_features[:min_len],
        "importance": importances[:min_len]
    }).sort_values("importance", ascending=False)

    print("\nFeature importances:")
    print(imp_df)

    imp_df.to_csv(
        OUTPUT_DIR / f"rf_importance_{best_model_name}_{best_feature_set}.csv",
        index=False
    )

    plt.figure(figsize=(8, 5))
    sns.barplot(data=imp_df, x="importance", y="feature")
    plt.title(f"Feature Importance - {best_model_name} - {best_feature_set}")
    plt.tight_layout()
    plt.savefig(
        PLOT_DIR / f"rf_importance_{best_model_name}_{best_feature_set}.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.show()

best_y_true = np.array(best_obj["y_true"])
best_y_pred = np.array(best_obj["y_pred"])
residuals = best_y_true - best_y_pred

plt.figure(figsize=(10, 5))
plt.scatter(best_y_pred, residuals)
plt.axhline(0, color="red", linestyle="--")
plt.title(f"Residual Plot - {best_model_name} - {best_feature_set}")
plt.xlabel("Predicted PM2.5")
plt.ylabel("Residual")
plt.tight_layout()
plt.savefig(
    PLOT_DIR / f"residual_plot_{best_model_name}_{best_feature_set}.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()

