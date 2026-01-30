"""
Cohort Analysis - Customer Retention Tracking
Track retention patterns and optimize customer lifecycle
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils.data_processor import load_and_prepare_data, create_cohort_data
from utils.theme import Components, Colors, apply_chart_theme, init_page

init_page("Cohort Analysis", "📈")

@st.cache_data
def load_cohort_data():
    try:
        df = load_and_prepare_data()
        cohort_counts, retention_rates = create_cohort_data(df)
        return df, cohort_counts, retention_rates
    except Exception as e:
        st.error(f"Error: {e}")
        st.stop()

df, cohort_counts, retention_rates = load_cohort_data()

st.markdown(Components.page_header("📈 Cohort Analysis - Customer Retention",
    "Track and optimize customer retention patterns across cohorts"), unsafe_allow_html=True)

st.markdown(Components.insight_box("🎯 What is Cohort Analysis?",
    """<p>Cohort analysis tracks groups of customers who started in the same month to understand:</p>
    <ul><li>📊 Retention: How many customers come back?</li>
    <li>🔄 Behavior Patterns: Do newer cohorts behave differently?</li>
    <li>💰 LTV Predictions: Estimate long-term customer value</li></ul>""",
    "info"), unsafe_allow_html=True)

st.markdown(Components.section_header("Retention Metrics Overview", "📊"), unsafe_allow_html=True)

total_cohorts = len(cohort_counts)
avg_ret_1m = retention_rates.iloc[:, 1].mean() if len(retention_rates.columns) > 1 else 0
avg_ret_3m = retention_rates.iloc[:, 3].mean() if len(retention_rates.columns) > 3 else 0
avg_ret_6m = retention_rates.iloc[:, 6].mean() if len(retention_rates.columns) > 6 else 0

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(Components.metric_card("Total Cohorts", f"{total_cohorts}",
                "📅 Monthly cohorts tracked", True, "📅", "primary"), unsafe_allow_html=True)
with col2:
    st.markdown(Components.metric_card("1-Month Retention", f"{avg_ret_1m:.1f}%",
                "🔄 Average across cohorts", avg_ret_1m > 30, "📊",
                "success" if avg_ret_1m > 40 else "warning"), unsafe_allow_html=True)
with col3:
    st.markdown(Components.metric_card("3-Month Retention", f"{avg_ret_3m:.1f}%",
                "📈 Mid-term retention", avg_ret_3m > 20, "📈", "info"), unsafe_allow_html=True)
with col4:
    st.markdown(Components.metric_card("6-Month Retention", f"{avg_ret_6m:.1f}%",
                "💎 Long-term loyalty", avg_ret_6m > 15, "💎",
                "success" if avg_ret_6m > 20 else "warning"), unsafe_allow_html=True)

st.markdown("---")
st.markdown(Components.section_header("Customer Retention Heatmap", "🔥"), unsafe_allow_html=True)

st.markdown(Components.insight_box("📖 How to Read This Heatmap",
    """<ul><li><strong>Each row:</strong> Cohort of customers who joined that month</li>
    <li><strong>Each column:</strong> Retention after N months</li>
    <li><strong>Green:</strong> Good retention, <strong>Red:</strong> Poor retention</li>
    <li><strong>🎯 Goal:</strong> Newer cohorts should be greener than older ones</li></ul>""",
    "info"), unsafe_allow_html=True)

fig_heatmap = go.Figure(data=go.Heatmap(
    z=retention_rates.values,
    x=['Month ' + str(i) for i in range(len(retention_rates.columns))],
    y=[str(idx) for idx in retention_rates.index],
    colorscale=[[0, '#FF6B6B'], [0.3, '#FFB84D'], [0.5, '#FFE66D'],
                [0.7, Colors.MINT_LEAF], [1, '#2ECC71']],
    text=retention_rates.values.round(1),
    texttemplate='%{text}%',
    textfont={"size": 10, "color": "white"},
    hovertemplate='<b>Cohort: %{y}</b><br>Month: %{x}<br>Retention: %{z:.1f}%<extra></extra>',
    colorbar=dict(title="Retention %", ticksuffix="%")
))

fig_heatmap = apply_chart_theme(fig_heatmap)
fig_heatmap.update_layout(title='Customer Retention Rates by Cohort', height=600,
    xaxis_title='Months Since First Purchase', yaxis_title='Cohort (First Purchase Month)')
st.plotly_chart(fig_heatmap, use_container_width=True)

col1, col2 = st.columns(2)
with col1:
    if len(retention_rates.columns) > 3:
        cohort_3m = retention_rates.iloc[:, 3]
        best_cohort = cohort_3m.idxmax()
        best_ret = cohort_3m.max()
        st.markdown(Components.insight_box("🏆 Best Performing Cohort",
            f"<p><strong>Cohort Month:</strong> {best_cohort}<br></p>"
            f"<p><strong>3-Month Retention:</strong> {best_ret:.1f}%</p>", "success"),
            unsafe_allow_html=True)

with col2:
    if avg_ret_1m > 0 and avg_ret_3m > 0:
        drop_rate = avg_ret_1m - avg_ret_3m
        st.markdown(Components.insight_box("📉 Retention Drop-Off",
            f"<p><strong>Month 1 → Month 3 Drop:</strong> {drop_rate:.1f}%</p>"
            "<p>Critical period for retention efforts!</p>",
            "success" if drop_rate < 20 else "warning"), unsafe_allow_html=True)

st.markdown("---")
st.markdown(Components.section_header("Retention Curves by Cohort", "📉"), unsafe_allow_html=True)

available_cohorts = [str(idx) for idx in retention_rates.index]
default_cohorts = available_cohorts[-5:] if len(available_cohorts) >= 5 else available_cohorts
selected_cohorts = st.multiselect("🔍 Select cohorts to compare:", available_cohorts,
                                   default=default_cohorts)

if selected_cohorts:
    fig_curves = go.Figure()
    colors = Colors.CHART_COLORS
    for idx, cohort in enumerate(selected_cohorts):
        cohort_data = retention_rates.loc[cohort]
        fig_curves.add_trace(go.Scatter(
            x=list(range(len(cohort_data))), y=cohort_data.values,
            mode='lines+markers', name=str(cohort),
            line=dict(width=3, color=colors[idx % len(colors)]),
            marker=dict(size=8, color=colors[idx % len(colors)])
        ))
    
    fig_curves = apply_chart_theme(fig_curves)
    fig_curves.update_layout(title='Retention Rate Comparison', height=500,
        xaxis_title='Months Since First Purchase', yaxis_title='Retention Rate (%)')
    st.plotly_chart(fig_curves, use_container_width=True)

st.markdown("---")
st.markdown(Components.section_header("Key Insights & Recommendations", "💡"), unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(Components.insight_box("📊 Month 1 Retention",
        f"<p><strong>Current: {avg_ret_1m:.1f}%</strong></p>"
        "<p><strong>Actions:</strong> Onboarding email series, welcome discount</p>",
        "success" if avg_ret_1m > 40 else "warning"), unsafe_allow_html=True)
with col2:
    drop = avg_ret_1m - avg_ret_3m if avg_ret_3m > 0 else 0
    st.markdown(Components.insight_box("⚠️ Critical Period",
        f"<p><strong>Months 2-3 Drop: {drop:.1f}%</strong></p>"
        "<p><strong>Actions:</strong> Re-engagement campaigns, personalized offers</p>",
        "success" if drop < 20 else "warning"), unsafe_allow_html=True)
with col3:
    st.markdown(Components.insight_box("💎 Long-term Loyalty",
        f"<p><strong>6-Month: {avg_ret_6m:.1f}%</strong></p>"
        "<p><strong>Actions:</strong> VIP programs, exclusive access, loyalty rewards</p>",
        "success" if avg_ret_6m > 20 else "warning"), unsafe_allow_html=True)

# ===========================================================
# FOOTER
# ===========================================================
st.markdown("---")
st.markdown(f"<p style='text-align: center; color: #5A6A7A;'>Cohort Analysis | "
            f"{total_cohorts} cohorts tracked | Avg 6-month: {avg_ret_6m:.1f}%</p>",
            unsafe_allow_html=True)
