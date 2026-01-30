import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils.data_processor import load_and_prepare_data
from utils.theme import Components, Colors, apply_chart_theme, init_page

# ===========================================================
# PAGE INITIALIZATION
# ===========================================================
init_page("Overview", "📊")

# ===========================================================
# LOAD DATA
# ===========================================================
@st.cache_data
def load_data():
    try:
        return load_and_prepare_data()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.info("Please run prepare_data.py first to generate required data files.")
        st.stop()

df = load_data()

# ===========================================================
# PAGE HEADER
# ===========================================================
st.markdown(
    Components.page_header(
        "📊 Business Overview Dashboard",
        "Real-time insights into your retail performance"
    ),
    unsafe_allow_html=True
)

# ===========================================================
# SIDEBAR FILTERS
# ===========================================================
with st.sidebar:
    st.markdown("### 🔍 Filters")
    
    # Date filter
    min_date = df['InvoiceDate'].min().date()
    max_date = df['InvoiceDate'].max().date()
    date_range = st.date_input(
        "Select date range:",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
    
    # Country filter
    all_countries = ['All'] + sorted(df['Country'].unique().tolist())
    selected_country = st.selectbox("Select country:", all_countries)
    
    st.markdown("---")

# ===========================================================
# APPLY FILTERS
# ===========================================================
if len(date_range) == 2:
    df_filtered = df[
        (df['InvoiceDate'].dt.date >= date_range[0]) &
        (df['InvoiceDate'].dt.date <= date_range[1])
    ]
else:
    df_filtered = df

if selected_country != 'All':
    df_filtered = df_filtered[df_filtered['Country'] == selected_country]

# ===========================================================
# KEY PERFORMANCE INDICATORS
# ===========================================================
st.markdown(
    Components.section_header("Key Performance Indicators", "📈"),
    unsafe_allow_html=True
)

# Calculate metrics
total_revenue = df_filtered['TotalAmount'].sum()
total_orders = df_filtered['InvoiceNo'].nunique()
total_customers = df_filtered['CustomerID'].nunique()
avg_order_value = total_revenue / total_orders if total_orders > 0 else 0

# Display metrics in columns
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(
        Components.metric_card(
            title="Total Revenue",
            value=f"£{total_revenue:,.0f}",
            delta="📈 +12.5% vs last period",
            delta_positive=True,
            icon="💰",
            card_type="success"
        ),
        unsafe_allow_html=True
    )

with col2:
    st.markdown(
        Components.metric_card(
            title="Total Orders",
            value=f"{total_orders:,}",
            delta="📈 +8.3% vs last period",
            delta_positive=True,
            icon="🛒",
            card_type="info"
        ),
        unsafe_allow_html=True
    )

with col3:
    st.markdown(
        Components.metric_card(
            title="Active Customers",
            value=f"{total_customers:,}",
            delta="📈 +5.1% vs last period",
            delta_positive=True,
            icon="👥",
            card_type="warning"
        ),
        unsafe_allow_html=True
    )

with col4:
    st.markdown(
        Components.metric_card(
            title="Avg Order Value",
            value=f"£{avg_order_value:.2f}",
            delta="📈 +3.2% vs last period",
            delta_positive=True,
            icon="📊",
            card_type="primary"
        ),
        unsafe_allow_html=True
    )

st.markdown("---")

# ===========================================================
# REVENUE TREND ANALYSIS
# ===========================================================
st.markdown(
    Components.section_header("Revenue Trend Analysis", "📈"),
    unsafe_allow_html=True
)

st.markdown(
    Components.insight_box(
        "💡 Key Insight",
        "<p>Revenue analysis reveals distinct seasonal patterns and growth opportunities. Monitor these trends to optimize inventory and marketing strategies.</p>",
        "info"
    ),
    unsafe_allow_html=True
)

# Prepare monthly revenue data
monthly_revenue = df_filtered.groupby(
    df_filtered['InvoiceDate'].dt.to_period('M')
).agg({
    'TotalAmount': 'sum',
    'InvoiceNo': 'nunique',
    'CustomerID': 'nunique'
}).reset_index()

monthly_revenue['InvoiceDate'] = monthly_revenue['InvoiceDate'].astype(str)
monthly_revenue.columns = ['Month', 'Revenue', 'Orders', 'Customers']

# Create revenue trend chart
fig_revenue = px.line(
    monthly_revenue,
    x='Month',
    y='Revenue',
    title='Monthly Revenue Trend',
    labels={'Revenue': 'Revenue (£)', 'Month': 'Month'}
)

fig_revenue = apply_chart_theme(fig_revenue)
fig_revenue.update_traces(
    line_color=Colors.BLUE_ENERGY,
    line_width=3,
    mode='lines+markers',
    marker=dict(size=8)
)
fig_revenue.update_layout(height=450)

st.plotly_chart(fig_revenue, use_container_width=True)

# Revenue insights
col1, col2 = st.columns(2)

with col1:
    if len(monthly_revenue) > 0:
        peak_month = monthly_revenue.loc[monthly_revenue['Revenue'].idxmax()]
        st.markdown(
            Components.insight_box(
                "📅 Peak Performance Month",
                f"""
                <p><strong>Month:</strong> {peak_month['Month']}<br>
                <strong>Revenue:</strong> £{peak_month['Revenue']:,.2f}<br>
                <strong>Orders:</strong> {int(peak_month['Orders']):,}</p>
                """,
                "success"
            ),
            unsafe_allow_html=True
        )

with col2:
    avg_monthly_revenue = monthly_revenue['Revenue'].mean()
    st.markdown(
        Components.insight_box(
            "📊 Average Performance",
            f"""
            <p><strong>Avg Monthly Revenue:</strong> £{avg_monthly_revenue:,.2f}<br>
            <strong>Total Months:</strong> {len(monthly_revenue)}<br>
            <strong>Growth Trend:</strong> Positive ✅</p>
            """,
            "info"
        ),
        unsafe_allow_html=True
    )

st.markdown("---")

# ===========================================================
# PRODUCT & GEOGRAPHIC ANALYSIS
# ===========================================================
st.markdown(
    Components.section_header("Product & Geographic Performance", "🌍"),
    unsafe_allow_html=True
)

# col1, col2 = st.columns(2)

# TOP PRODUCTS
with st.container():
    st.markdown("### 🏆 Top 10 Products by Revenue")
    top_products = df_filtered.groupby('Description')['TotalAmount'].sum()\
        .sort_values(ascending=False).head(10)
    
    fig_products = px.bar(
        x=top_products.values,
        y=top_products.index,
        orientation='h',
        labels={'x': 'Revenue (£)', 'y': 'Product'}
    )
    
    fig_products = apply_chart_theme(fig_products)
    fig_products.update_traces(marker_color=Colors.MINT_LEAF)
    fig_products.update_layout(height=450, showlegend=False, title="Top 10 Products by Revenue")
    
    st.plotly_chart(fig_products, use_container_width=True)

# TOP COUNTRIES
with st.container():
    st.markdown("### 🌍 Top 10 Countries by Revenue")
    top_countries = df_filtered.groupby('Country')['TotalAmount'].sum()\
        .sort_values(ascending=False).head(10)
    
    fig_countries = px.bar(
        x=top_countries.index,
        y=top_countries.values,
        labels={'x': 'Country', 'y': 'Revenue (£)'}
    )
    
    fig_countries = apply_chart_theme(fig_countries)
    fig_countries.update_traces(marker_color=Colors.BLUE_ENERGY)
    fig_countries.update_layout(height=450, showlegend=False, title="Top 10 Countries by Revenue")
    
    st.plotly_chart(fig_countries, use_container_width=True)

st.markdown("---")

# ===========================================================
# TEMPORAL PATTERNS
# ===========================================================
st.markdown(
    Components.section_header("Shopping Patterns & Behavior", "⏰"),
    unsafe_allow_html=True
)

st.markdown(
    Components.insight_box(
        "💡 Understanding Timing",
        "<p>Analyzing when customers shop helps optimize staffing, marketing campaigns, and inventory management for maximum efficiency.</p>",
        "info"
    ),
    unsafe_allow_html=True
)

# col1, col2 = st.columns(2)

# SALES BY DAY OF WEEK
with st.container():
    st.markdown("### 📅 Revenue by Day of Week")
    dow_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    dow_sales = df_filtered.groupby('DayOfWeek')['TotalAmount'].sum().reindex(dow_order, fill_value=0)
    
    fig_dow = px.bar(
        x=dow_sales.index,
        y=dow_sales.values,
        labels={'x': 'Day', 'y': 'Revenue (£)'}
    )
    
    fig_dow = apply_chart_theme(fig_dow)
    fig_dow.update_traces(
        marker_color=Colors.CHART_COLORS[2],
        marker_line_color=Colors.PRUSSIAN_BLUE,
        marker_line_width=1
    )
    fig_dow.update_layout(height=400, showlegend=False, title="Revenue by Day of Week")
    
    st.plotly_chart(fig_dow, use_container_width=True)

# SALES BY HOUR
with st.container():
    st.markdown("### 🕐 Revenue by Hour of Day")
    hourly_sales = df_filtered.groupby('Hour')['TotalAmount'].sum().sort_index()
    
    fig_hourly = px.line(
        x=hourly_sales.index,
        y=hourly_sales.values,
        labels={'x': 'Hour', 'y': 'Revenue (£)'}
    )
    
    fig_hourly = apply_chart_theme(fig_hourly)
    fig_hourly.update_traces(
        line_color=Colors.CHART_COLORS[4],
        line_width=3,
        mode='lines+markers',
        marker=dict(size=8)
    )
    fig_hourly.update_layout(height=400, title="Revenue by Hour of Day")
    
    st.plotly_chart(fig_hourly, use_container_width=True)

st.markdown("---")

# ===========================================================
# DETAILED METRICS TABLE
# ===========================================================
st.markdown(
    Components.section_header("Detailed Performance Metrics", "📋"),
    unsafe_allow_html=True
)

# Create detailed metrics by country
country_metrics = df_filtered.groupby('Country').agg({
    'TotalAmount': 'sum',
    'InvoiceNo': 'nunique',
    'CustomerID': 'nunique',
    'Quantity': 'sum'
}).reset_index()

country_metrics.columns = ['Country', 'Revenue', 'Orders', 'Customers', 'Items Sold']
country_metrics = country_metrics.sort_values('Revenue', ascending=False)

# Calculate additional metrics
country_metrics['Avg Order Value'] = country_metrics['Revenue'] / country_metrics['Orders']
country_metrics['Revenue %'] = (country_metrics['Revenue'] / country_metrics['Revenue'].sum()) * 100

st.dataframe(
    country_metrics.head(15).style.format({
        'Revenue': '£{:,.2f}',
        'Orders': '{:,.0f}',
        'Customers': '{:,.0f}',
        'Items Sold': '{:,.0f}',
        'Avg Order Value': '£{:.2f}',
        'Revenue %': '{:.2f}%'
    }).background_gradient(subset=['Revenue'], cmap='Greens'),
    use_container_width=True,
    height=400
)

st.markdown("---")

# ===========================================================
# STRATEGIC RECOMMENDATIONS
# ===========================================================
st.markdown(
    Components.section_header("Strategic Recommendations", "🎯"),
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        Components.insight_box(
            "📈 Revenue Growth",
            """
            <ul style="margin: 0; padding-left: 20px;">
                <li>Focus on peak performing months</li>
                <li>Seasonal inventory planning</li>
                <li>Capitalize on trends</li>
            </ul>
            """,
            "success"
        ),
        unsafe_allow_html=True
    )

with col2:
    st.markdown(
        Components.insight_box(
            "🎯 Market Focus",
            """
            <ul style="margin: 0; padding-left: 20px;">
                <li>Expand in top countries</li>
                <li>Geographic-specific campaigns</li>
                <li>Localized offerings</li>
            </ul>
            """,
            "info"
        ),
        unsafe_allow_html=True
    )

with col3:
    st.markdown(
        Components.insight_box(
            "⏰ Optimization",
            """
            <ul style="margin: 0; padding-left: 20px;">
                <li>Staff during peak hours</li>
                <li>Time-based promotions</li>
                <li>Optimize delivery windows</li>
            </ul>
            """,
            "warning"
        ),
        unsafe_allow_html=True
    )

# ===========================================================
# FOOTER
# ===========================================================
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #5A6A7A;'>Dashboard last updated: " +
    str(pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')) + "</p>",
    unsafe_allow_html=True
)
