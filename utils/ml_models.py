import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def forecast_sales(df, periods=12):
    """
    Simple time series forecasting using moving averages and trend
    For production, use Prophet or ARIMA
    """
    # Aggregate monthly sales
    monthly_sales = df.groupby(df['InvoiceDate'].dt.to_period('M'))['TotalAmount'].sum()
    monthly_sales.index = monthly_sales.index.to_timestamp()
    
    # Calculate moving average and trend
    ma_3 = monthly_sales.rolling(window=3).mean()
    ma_6 = monthly_sales.rolling(window=6).mean()
    
    # Simple linear trend
    from scipy import stats
    x = np.arange(len(monthly_sales))
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, monthly_sales.values)
    
    # Forecast
    last_date = monthly_sales.index[-1]
    forecast_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), 
                                   periods=periods, freq='MS')
    
    forecast_values = []
    for i in range(1, periods + 1):
        # Combine trend and last moving average
        trend_value = slope * (len(monthly_sales) + i) + intercept
        ma_value = ma_3.iloc[-1] if not pd.isna(ma_3.iloc[-1]) else monthly_sales.iloc[-1]
        forecast_value = 0.6 * trend_value + 0.4 * ma_value
        forecast_values.append(max(0, forecast_value))  # Ensure non-negative
    
    forecast_df = pd.DataFrame({
        'Date': forecast_dates,
        'Forecast': forecast_values,
        'Lower_Bound': np.array(forecast_values) * 0.85,
        'Upper_Bound': np.array(forecast_values) * 1.15
    })
    
    return monthly_sales, forecast_df

def predict_clv(clv_data):
    """
    Predict Customer Lifetime Value using Random Forest
    """
    # Prepare features
    features = ['Frequency', 'AvgOrderValue', 'Lifespan']
    X = clv_data[features].fillna(0)
    y = clv_data['CLV'].fillna(0)
    
    # Remove extreme outliers for training only
    q99 = y.quantile(0.99)
    X_train_filtered = X[y <= q99]
    y_train_filtered = y[y <= q99]
    
    # Split data (using filtered data for training)
    X_train, X_test, y_train, y_test = train_test_split(
        X_train_filtered, y_train_filtered, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
    model.fit(X_train_scaled, y_train)
    
    # Predictions on test set
    y_pred = model.predict(X_test_scaled)
    mae = mean_absolute_error(y_test, y_pred)
    
    # Feature importance
    importance = pd.DataFrame({
        'Feature': features,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    # Predict for all customers (including outliers) using the original X
    X_all_scaled = scaler.transform(X)
    clv_data_copy = clv_data.copy()
    clv_data_copy['Predicted_CLV'] = model.predict(X_all_scaled)
    
    return model, scaler, clv_data_copy, mae, importance

def predict_churn(X, y):
    """
    Predict customer churn using Random Forest Classifier
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                        random_state=42, stratify=y)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10,
                                   class_weight='balanced')
    model.fit(X_train_scaled, y_train)
    
    # Predictions
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    # Feature importance
    importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    # Predict for all customers
    X_all_scaled = scaler.transform(X)
    churn_proba = model.predict_proba(X_all_scaled)[:, 1]
    
    return model, scaler, churn_proba, accuracy, importance, report

def market_basket_analysis(basket_binary, min_support=0.01):
    """
    Perform market basket analysis using Apriori-like approach
    Note: For production, use mlxtend library
    """
    # Calculate support for individual items
    item_support = basket_binary.sum() / len(basket_binary)
    frequent_items = item_support[item_support >= min_support].sort_values(ascending=False)
    
    # Find frequent pairs (simplified version)
    pairs = []
    items = frequent_items.index.tolist()[:20]  # Limit to top 20 items for performance
    
    for i, item1 in enumerate(items):
        for item2 in items[i+1:]:
            # Count co-occurrences
            both = ((basket_binary[item1] == 1) & (basket_binary[item2] == 1)).sum()
            support = both / len(basket_binary)
            
            if support >= min_support:
                # Calculate confidence and lift
                item1_count = basket_binary[item1].sum()
                item2_count = basket_binary[item2].sum()
                
                confidence_1_to_2 = both / item1_count if item1_count > 0 else 0
                confidence_2_to_1 = both / item2_count if item2_count > 0 else 0
                
                expected = (item1_count * item2_count) / len(basket_binary)
                lift = both / expected if expected > 0 else 0
                
                pairs.append({
                    'Item_A': item1,
                    'Item_B': item2,
                    'Support': support,
                    'Confidence_A_to_B': confidence_1_to_2,
                    'Confidence_B_to_A': confidence_2_to_1,
                    'Lift': lift
                })
    
    pairs_df = pd.DataFrame(pairs).sort_values('Lift', ascending=False)
    
    return frequent_items, pairs_df
