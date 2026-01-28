import pandas as pd
import hashlib
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, accuracy_score
import shap
import warnings
warnings.filterwarnings("ignore")


def load_clean_data():
    retail = pd.read_csv("data/retail_data_clean.csv", parse_dates=["InvoiceDate"])
    month = pd.read_csv("data/Month_metrics.csv")
    product = pd.read_csv("data/Product_metrics.csv")
    country = pd.read_csv("data/Country_metrics.csv")
    rfm = pd.read_csv("data/RFM_analysis.csv")

    (
        rfm,
        clv_model,
        churn_model,
        clv_importance,
        churn_importance,
        clv_mae,
        churn_acc,
        scaler_clv,
        scaler_churn,
    ) = build_clv_churn_models(rfm)

    return (
        retail,
        month,
        product,
        country,
        rfm,
        clv_model,
        churn_model,
        clv_importance,
        churn_importance,
        clv_mae,
        churn_acc,
        scaler_clv,
        scaler_churn,
    )


def build_clv_churn_models(rfm):
    df = rfm.copy()

    # Lifespan
    df["Lifespan"] = df["Tenure"] if "Tenure" in df.columns else df["Recency"]

    # CLV target
    if "CLV" not in df.columns:
        df["CLV"] = df["Monetary"] * df["Frequency"]

    clv_features = ["Frequency", "AvgOrderValue", "Lifespan"]
    X = df[clv_features].fillna(0)
    y = df["CLV"].fillna(0)

    q99 = y.quantile(0.99)
    X = X[y <= q99]
    y = y[y <= q99]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler_clv = StandardScaler()
    X_train_scaled = scaler_clv.fit_transform(X_train)
    X_test_scaled = scaler_clv.transform(X_test)

    clv_model = RandomForestRegressor(
        n_estimators=100, random_state=42, max_depth=10
    )
    clv_model.fit(X_train_scaled, y_train)

    y_pred = clv_model.predict(X_test_scaled)
    clv_mae = mean_absolute_error(y_test, y_pred)

    clv_importance = pd.DataFrame({
        "Feature": clv_features,
        "Importance": clv_model.feature_importances_
    }).sort_values("Importance", ascending=False)

    df["Predicted_CLV"] = clv_model.predict(
        scaler_clv.transform(df[clv_features].fillna(0))
    )

    # Churn model
    df["ChurnLabel"] = (df["Recency"] > df["Recency"].median()).astype(int)

    churn_features = ["Recency", "Frequency", "Monetary", "AvgOrderValue", "Lifespan"]
    Xc = df[churn_features].fillna(0)
    yc = df["ChurnLabel"]

    Xc_train, Xc_test, yc_train, yc_test = train_test_split(
        Xc, yc, test_size=0.2, random_state=42, stratify=yc
    )

    scaler_churn = StandardScaler()
    Xc_train_scaled = scaler_churn.fit_transform(Xc_train)
    Xc_test_scaled = scaler_churn.transform(Xc_test)

    churn_model = RandomForestClassifier(
        n_estimators=100, random_state=42, max_depth=10, class_weight="balanced"
    )
    churn_model.fit(Xc_train_scaled, yc_train)

    churn_pred = churn_model.predict(Xc_test_scaled)
    churn_acc = accuracy_score(yc_test, churn_pred)

    churn_importance = pd.DataFrame({
        "Feature": churn_features,
        "Importance": churn_model.feature_importances_
    }).sort_values("Importance", ascending=False)

    df["Churn_Proba"] = churn_model.predict_proba(
        scaler_churn.transform(Xc)
    )[:, 1]

    df["ChurnRisk"] = pd.cut(
        df["Churn_Proba"],
        bins=[0, 0.33, 0.66, 1.0],
        labels=["Low", "Medium", "High"],
        include_lowest=True,
    )

    return (
        df,
        clv_model,
        churn_model,
        clv_importance,
        churn_importance,
        clv_mae,
        churn_acc,
        scaler_clv,
        scaler_churn,
    )


def hash_df(df):
    return hashlib.md5(pd.util.hash_pandas_object(df, index=True).values).hexdigest()


def compute_shap_values(model, X):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    return shap_values
