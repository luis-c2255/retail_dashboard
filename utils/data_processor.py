import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# ============================================================================
# MAIN DATA LOADING FUNCTION - ENHANCED WITH ALL DERIVED COLUMNS
# ============================================================================
def load_and_prepare_data(filepath='data/retail_data_cleaned.csv'):
    """
    Load and prepare the main dataset with all required columns
    This prevents KeyErrors across all dashboard pages
    """
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        # Try alternative path
        # try:
            # df = pd.read_csv('data/original_data.csv')
        # except FileNotFoundError:
            raise FileNotFoundError(
                # "Data file not found. Please ensure data/retail_data_cleaned.csv"
                # "or data/original_data.csv exists. Run prepare_data.py first"
            )
    # Parse dates with explicit format handling
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce')

    # Ensure TotalAmount exists
    if 'TotalAmount' not in df.columns:
        df['TotalAmount'] = df['Quantity'] * df['UnitPrice']

    # Create all derived temporal columns (prevents KeyErrors)
    if 'DayOfWeek' not in df.columns:
        df['DayOfWeek'] = df['InvoiceDate'].dt.day_name()

    if 'Hour' not in df.columns:
        df['Hour'] = df['InvoiceDate'].dt.hour

    if 'Year' not in df.columns:
        df['Year'] = df['InvoiceDate'].dt.year

    if 'Month' not in df.columns:
        df['Month'] = df['InvoiceDate'].dt.month

    if 'Day' not in df.columns:
        df['Day'] = df['InvoiceDate'].dt.day

    if 'YearMonth' not in df.columns:
        df['YearMonth'] = df['InvoiceDate'].dt.to_period('M')

    # Ensure CustomerID is numeric
    df['CustomerID'] = pd.to_numeric(df['CustomerID'], errors='coerce')

    # Remove any rows with NaN CustomerID or InvoiceDate
    df = df.dropna(subset=['CustomerID', 'InvoiceDate'])
    
    # Remove negative quantities and prices (returns/errors)
    df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]

    return df


# ============================================================================
# RFM ANALYSIS
# ============================================================================

def create_rfm_features(df):
    """
    Create RFM analysis with enhanced features
    Returns: DataFrame with RFM scores and segments
    """
    reference_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)

    # Calculate RFM metrics
    rfm = df.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (reference_date - x.max()).days,
        'InvoiceNo': 'nunique',
        'TotalAmount': ['sum', 'mean']
    }).reset_index()

    rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary', 'AvgOrderValue']

    # Add additional features
    customer_stats = df.groupby('CustomerID').agg({
        'InvoiceDate': ['min', 'max'],
        'Quantity': 'sum',
        'Country': lambda x: x.mode()[0] if len(x) > 0 else 'Unknown'
    }).reset_index()

    customer_stats.columns = ['CustomerID', 'FirstPurchase', 'LastPurchase',
                              'TotalItems', 'Country']

    rfm = rfm.merge(customer_stats, on='CustomerID')

    # Calculate customer tenure in days
    rfm['Tenure'] = (rfm['LastPurchase'] - rfm['FirstPurchase']).dt.days

    # Create RFM scores (1-5 scale)
    try:
        rfm['R_Score'] = pd.qcut(rfm['Recency'], 5, labels=[5, 4, 3, 2, 1], duplicates='drop')
    except:
        # If qcut fails, use a simpler approach
        rfm['R_Score'] = pd.cut(rfm['Recency'], 5, labels=[5, 4, 3, 2, 1])

    try:
        rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method='first'), 5,
                                  labels=[1, 2, 3, 4, 5], duplicates='drop')
    except:
        rfm['F_Score'] = pd.cut(rfm['Frequency'], 5, labels=[1, 2, 3, 4, 5])

    try:
        rfm['M_Score'] = pd.qcut(rfm['Monetary'], 5, labels=[1, 2, 3, 4, 5], duplicates='drop')
    except:
        rfm['M_Score'] = pd.cut(rfm['Monetary'], 5, labels=[1, 2, 3, 4, 5])

    # Segment customers
    def segment_customers(row):
        try:
            r, f, m = int(row['R_Score']), int(row['F_Score']), int(row['M_Score'])
        except:
            return 'Others'

        if r >= 4 and f >= 4 and m >= 4:
            return 'Champions'
        elif r >= 4 and f >= 3:
            return 'Loyal Customers'
        elif r >= 4 and f < 3:
            return 'Potential Loyalists'
        elif r >= 3 and f >= 3 and m >= 3:
            return 'Promising'
        elif r < 3 and f >= 4:
            return 'At Risk'
        elif r < 3 and f < 3 and m >= 3:
            return 'Hibernating'
        elif r < 2:
            return 'Lost'
        else:
            return 'New Customers'

    rfm['Segment'] = rfm.apply(segment_customers, axis=1)

    # Convert scores to numeric for calculations
    rfm['R_Score'] = pd.to_numeric(rfm['R_Score'], errors='coerce')
    rfm['F_Score'] = pd.to_numeric(rfm['F_Score'], errors='coerce')
    rfm['M_Score'] = pd.to_numeric(rfm['M_Score'], errors='coerce')

    # Fill any NaN values
    rfm = rfm.fillna(0)

    return rfm


# ============================================================================
# COHORT ANALYSIS
# ============================================================================

def create_cohort_data(df):
    """
    Create cohort retention analysis
    Returns: cohort_counts, retention_rates
    """
    # Create a copy to avoid modifying original
    df_cohort = df.copy()

    # Get cohort month (first purchase)
    df_cohort['CohortMonth'] = df_cohort.groupby('CustomerID')['InvoiceDate'].transform('min').dt.to_period('M')

    # Get current month
    df_cohort['OrderMonth'] = df_cohort['InvoiceDate'].dt.to_period('M')

    # Calculate cohort index (months since first purchase)
    def get_month_diff(row):
        return (row['OrderMonth'] - row['CohortMonth']).n

    df_cohort['CohortIndex'] = df_cohort.apply(get_month_diff, axis=1)

    # Count unique customers per cohort per month
    cohort_data = df_cohort.groupby(['CohortMonth', 'CohortIndex'])['CustomerID'].nunique().reset_index()
    cohort_data.columns = ['CohortMonth', 'CohortIndex', 'CustomerCount']

    # Pivot for heatmap
    cohort_pivot = cohort_data.pivot(
        index='CohortMonth',
        columns='CohortIndex',
        values='CustomerCount'
    )

    # Fill NaN with 0
    cohort_pivot = cohort_pivot.fillna(0)

    # Calculate retention rates
    cohort_size = cohort_pivot.iloc[:, 0]

    # Avoid division by zero
    retention = cohort_pivot.divide(cohort_size, axis=0).replace([np.inf, -np.inf], 0).fillna(0) * 100

    return cohort_pivot, retention


# ============================================================================
# BASKET ANALYSIS
# ============================================================================

def prepare_basket_analysis(df):
    """
    Prepare data for market basket analysis
    Returns: Binary transaction matrix
    """
    # Group products by invoice
    basket = df.groupby(['InvoiceNo', 'Description'])['Quantity'].sum().unstack().fillna(0)

    # Convert to binary (1 if purchased, 0 otherwise)
    basket_binary = basket.map(lambda x: 1 if x > 0 else 0)

    return basket_binary


# ============================================================================
# CLV CALCULATION
# ============================================================================

def calculate_clv_features(df, rfm):
    """
    Calculate Customer Lifetime Value features
    Returns: DataFrame with CLV metrics
    """
    # Average purchase frequency
    customer_frequency = df.groupby('CustomerID')['InvoiceNo'].nunique()

    # Average order value
    customer_avg_value = df.groupby('CustomerID')['TotalAmount'].mean()

    # Customer lifespan in months
    customer_lifespan = rfm.set_index('CustomerID')['Tenure'] / 30
    customer_lifespan = customer_lifespan.fillna(1)  # Avoid division by zero

    # Simple CLV calculation: Avg Order Value * Purchase Frequency * Lifespan
    clv = pd.DataFrame({
        'CustomerID': customer_frequency.index,
        'Frequency': customer_frequency.values,
        'AvgOrderValue': customer_avg_value.values,
        'Lifespan': customer_lifespan.values,
    })

    # Calculate CLV (annualized)
    clv['CLV'] = (clv['AvgOrderValue'] * clv['Frequency'] * (clv['Lifespan'] / 12)).fillna(0)

    # Remove negative or invalid values
    clv['CLV'] = clv['CLV'].clip(lower=0)

    return clv


# ============================================================================
# CHURN PREPARATION
# ============================================================================

def prepare_churn_features(df, rfm):
    """
    Prepare features for churn prediction
    Returns: X (features), y (labels), rfm with churn flag
    """
    # Define churn: customers who haven't purchased in last 90 days
    churn_threshold = 90
    rfm['Churned'] = (rfm['Recency'] > churn_threshold).astype(int)

    # Features for prediction
    feature_cols = ['Recency', 'Frequency', 'Monetary', 'AvgOrderValue', 'Tenure']

    # Ensure all feature columns are numeric
    for col in feature_cols:
        rfm[col] = pd.to_numeric(rfm[col], errors='coerce')

    # Fill NaN values with 0
    X = rfm[feature_cols].fillna(0)
    y = rfm['Churned']

    return X, y, rfm


# ============================================================================
# PRODUCT METRICS
# ============================================================================

def calculate_product_metrics(df):
    """Calculate detailed product performance metrics"""
    product_metrics = df.groupby('Description').agg({
        'Quantity': 'sum',
        'TotalAmount': 'sum',
        'InvoiceNo': 'nunique',
        'CustomerID': 'nunique'
    }).reset_index()

    product_metrics.columns = ['Product', 'TotalQuantity', 'Revenue', 'NumOrders', 'NumCustomers']
    product_metrics = product_metrics.sort_values('Revenue', ascending=False)

    return product_metrics


# ============================================================================
# COUNTRY METRICS
# ============================================================================

def calculate_country_metrics(df):
    """Calculate detailed country performance metrics"""
    country_metrics = df.groupby('Country').agg({
        'TotalAmount': 'sum',
        'InvoiceNo': 'nunique',
        'CustomerID': 'nunique'
    }).reset_index()

    country_metrics.columns = ['Country', 'Revenue', 'NumOrders', 'NumCustomers']
    country_metrics = country_metrics.sort_values('Revenue', ascending=False)

    return country_metrics


# ============================================================================
# MONTHLY METRICS
# ============================================================================

def calculate_monthly_metrics(df):
    """Calculate monthly performance metrics"""
    monthly_metrics = df.groupby(df['InvoiceDate'].dt.to_period('M')).agg({
        'TotalAmount': 'sum',
        'InvoiceNo': 'nunique',
        'CustomerID': 'nunique'
    }).reset_index()

    monthly_metrics.columns = ['Month', 'Revenue', 'NumOrders', 'NumCustomers']
    monthly_metrics['Month'] = monthly_metrics['Month'].astype(str)

    return monthly_metrics


# ============================================================================
# DATA VALIDATION
# ============================================================================

def validate_data(df):
    """
    Validate data quality and return summary
    Returns: Dictionary with validation results
    """
    validation = {
        'total_rows': len(df),
        'missing_values': df.isnull().sum().to_dict(),
        'date_range': {
            'min': df['InvoiceDate'].min(),
            'max': df['InvoiceDate'].max()
        },
        'unique_customers': df['CustomerID'].nunique(),
        'unique_products': df['Description'].nunique(),
        'total_revenue': df['TotalAmount'].sum(),
        'avg_order_value': df.groupby('InvoiceNo')['TotalAmount'].sum().mean()
    }

    return validation


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_date_range(df):
    """Get the date range of the dataset"""
    return df['InvoiceDate'].min(), df['InvoiceDate'].max()


def get_top_products(df, n=10):
    """Get top N products by revenue"""
    return df.groupby('Description')['TotalAmount'].sum().nlargest(n)


def get_top_customers(df, n=10):
    """Get top N customers by revenue"""
    return df.groupby('CustomerID')['TotalAmount'].sum().nlargest(n)


def get_top_countries(df, n=10):
    """Get top N countries by revenue"""
    return df.groupby('Country')['TotalAmount'].sum().nlargest(n)
