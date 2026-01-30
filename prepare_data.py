import pandas as pd
import numpy as np
from datetime import datetime
import os
import sys
# Add utils to path
sys.path.append('.')
from utils.data_processor import (
        load_and_prepare_data,
        create_rfm_features,
        calculate_product_metrics,
        calculate_country_metrics,
        calculate_monthly_metrics
)

def prepare_all_data():
    """Prepare all data files needed for the dashboard"""
    print("=" * 80)
    print("RETAIL ANALYTICS DASHBOARD - DATA PREPARATION")
    print("=" * 80)

    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Step 1: Load and cleaning raw data
    print("\n[1/6] Loading and cleaning raw data...")
    try:
        df_raw = pd.read_csv('data/retail_data_cleaned.csv')
        print(f"  ✓ Loaded {len(df_raw):,} rows")

        # Clean the data
        df = df_raw.copy()
        
        # Parse dates
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce')
        
        # Create TotalAmount
        df['TotalAmount'] = df['Quantity'] * df['UnitPrice']
        
        # Create temporal columns
        df['DayOfWeek'] = df['InvoiceDate'].dt.day_name()
        df['Hour'] = df['InvoiceDate'].dt.hour
        df['Year'] = df['InvoiceDate'].dt.year
        df['Month'] = df['InvoiceDate'].dt.month
        
        # Clean CustomerID
        df['CustomerID'] = pd.to_numeric(df['CustomerID'], errors='coerce')
        df = df.dropna(subset=['CustomerID', 'InvoiceDate'])
        
        # Remove negative values
        df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]
        
        # Save cleaned data
        df.to_csv('data/retail_data_cleaned.csv', index=False)
        print(f"   ✓ Cleaned data saved: {len(df):,} rows retained")
    except Exception as e:
        print(f" ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 2: Create RFM analysis
    print("\n[2/6] Creating RFM analysis...")
    try:
        rfm = create_rfm_features(df)
        rfm.to_csv('data/rfm_analysis.csv', index=False)
        print(f"  ✓ RFM analysis saved: {len(rfm):,} customers")
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 3: Product metrics
    print("\n[3/6] Calculating product metrics...")
    try:
        product_metrics = calculate_product_metrics(df)
        product_metrics.to_csv('data/product_metrics.csv', index=False)
        print(f"  ✓ Product metrics saved: {len(product_metrics):,} products")
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 4: Country metrics
    print("\n[4/6] Calculating country metrics...")
    try:
        country_metrics = calculate_country_metrics(df)
        country_metrics.to_csv('data/country_metrics.csv', index=False)
        print(f"  ✓ Country metrics saved: {len(country_metrics):,} countries")
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 5: Monthly metrics
    print("\n[5/6] Calculating monthly metrics...")
    try:
        monthly_metrics = calculate_monthly_metrics(df)
        monthly_metrics.to_csv('data/monthly_metrics.csv', index=False)
        print(f"  ✓ Monthly metrics saved: {len(monthly_metrics):,} months")
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 6: Summary
    print("\n[6/6] Summary...")
    print("\n" + "=" * 80)
    print("DATA PREPARATION COMPLETE!")
    print("=" * 80)
    print(f"\nFiles created in 'data/' folder:")
    print(f"  • retail_data_cleaned.csv ({len(df):,} rows)")
    print(f"  • rfm_analysis.csv ({len(rfm):,} customers)")
    print(f"  • product_metrics.csv ({len(product_metrics):,} products)")
    print(f"  • country_metrics.csv ({len(country_metrics):,} countries)")
    print(f"  • monthly_metrics.csv ({len(monthly_metrics):,} months)")
    print("\n✅ You're ready to run the dashboard!")
    print("Run: streamlit run app.py")
    print("=" * 80 + "\n")
    
    return True


if __name__ == "__main__":
    success = prepare_all_data()
    if not success:
        print("\n⚠️  Data preparation failed. Please check the errors above.")
        sys.exit(1)
    else:
        sys.exit(0)
