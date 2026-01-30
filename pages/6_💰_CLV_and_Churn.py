"""
CLV & Churn - Customer Lifetime Value & Churn Prediction
Maximize customer value and prevent churn with AI-powered insights
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils.data_processor import (load_and_prepare_data, create_rfm_features,
                                   calculate_clv_features, prepare_churn_features)
from utils.ml_models import predict_clv, predict_churn
from utils.theme import Components, Colors, apply_chart_theme, init_page

init_page("CLV & Churn", "💰")

@st.cache_data
def load_clv_churn_data():
    try:
        df = load_and_prepare_data()
        rfm = create_rfm_features(df)
        clv_data = calculate_clv_features(df, rfm)
        X_churn, y_churn, rfm_churn = prepare_churn_features(df, rfm)
        
        # CLV predictions
        clv_model, clv_scaler, clv_pred, clv_mae, clv_imp = predict_clv(clv_data)
        # Churn predictions - returns predictions for all customers in X_churn
        churn_model, churn_scaler, churn_proba, churn_acc, churn_imp, churn_rep = predict_churn(X_churn, y_churn)
        
        # Add churn predictions to rfm dataframe
        #Reset index to ensure alignment
        rfm_churn_reset = rfm_churn.reset_index(drop=True)
        
        # Verify lengths match after reset
        if len(churn_proba) == len(rfm_churn_reset):
            rfm_churn_reset['ChurnProbability'] = churn_proba
        else:
            # If there's still a mismatch, create a mapping using CustomerID
            # This shouldn't happen, but let's be defensive
            churn_df = pd.DataFrame({
                'CustomerID': rfm_churn_reset['CustomerID'].values,
                'ChurnProbability': churn_proba[:len(rfm_churn_reset)] if len(churn_proba) > len(rfm_churn_reset) else list(churn_proba) + [0.5] * (len(rfm_churn_reset) - len(churn_proba))
            })
            rfm_churn_reset = rfm_churn_reset.merge(churn_df, on='CustomerID', how='left')
            rfm_churn_reset['ChurnProbability'] = rfm_churn_reset['ChurnProbability'].fillna(0.5)  # Default to 0.5 if missing
            
        # Create risk cactegories
        rfm_churn_reset['ChurnRisk'] = pd.cut(
            rfm_churn_reset['ChurnProbability'],
            bins=[0, 0.3, 0.7, 1.0],
            labels=['Loww', 'Medium', 'High']
        )
        # Merge CLV predictions
        rfm_final = rfm_churn_reset.merge(
            clv_pred[['CustomerID', 'Predicted_CLV']],
            on='CustomerID',
            how='left'
        )
        # Fill any missing CLV values
        rfm_final['Predicted_CLV'] = rfm_final['Predicted_CLV'].fillna(rfm_final['Monetary'])
        
        return df, rfm_final, clv_mae, clv_imp, churn_acc, churn_imp, churn_rep
    except Exception as e:
        st.error(f" ❌ Error loading CLV & Churn data: {str(e)}")
        st.error("Please check that the data has been prepared correctly.")
        st.info("💡 Try running: `python prepare_data.py`")
        import traceback
        with st.expander("🔍 Show detailed error"):
            st.code(traceback.format_exc())
        st.stop()
        
df, rfm, clv_mae, clv_imp, churn_acc, churn_imp, churn_rep = load_clv_churn_data()
       

st.markdown(Components.page_header("💰 Customer Lifetime Value & Churn Prediction",
    "Maximize customer value and prevent churn with AI-powered insights"), unsafe_allow_html=True)

st.markdown(Components.insight_box("🎯 Why CLV and Churn Matter",
    """<p><strong>Customer Lifetime Value (CLV)</strong> predicts total revenue from a customer.</p>
    <p><strong>Churn Prediction</strong> identifies customers likely to stop purchasing.</p>
    <p><strong>Together, they help you:</strong> Focus resources, prevent loss, optimize spend, increase revenue</p>""",
    "info"), unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["💰 Customer Lifetime Value", "🚨 Churn Prediction", "🎯 Strategic Action Plan"])

with tab1:
    st.markdown(Components.section_header("Customer Lifetime Value Analysis", "💰"), unsafe_allow_html=True)
    
    avg_clv = rfm['Predicted_CLV'].mean()
    median_clv = rfm['Predicted_CLV'].median()
    total_clv = rfm['Predicted_CLV'].sum()
    top_10_clv = rfm.nlargest(int(len(rfm) * 0.1), 'Predicted_CLV')['Predicted_CLV'].sum()
    top_10_share = (top_10_clv / total_clv) * 100
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(Components.metric_card("Average CLV", f"£{avg_clv:,.0f}",
                    "📊 Per customer", True, "💰", "success"), unsafe_allow_html=True)
    with col2:
        st.markdown(Components.metric_card("Median CLV", f"£{median_clv:,.0f}",
                    "📊 Middle value", True, "📊", "info"), unsafe_allow_html=True)
    with col3:
        st.markdown(Components.metric_card("Total CLV", f"£{total_clv:,.0f}",
                    "💎 Total value", True, "💎", "primary"), unsafe_allow_html=True)
    with col4:
        st.markdown(Components.metric_card("Top 10% Share", f"{top_10_share:.1f}%",
                    "🏆 Concentration", True, "🏆", "warning"), unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### 📊 Customer Lifetime Value Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Distribution Histogram")
        fig_clv = px.histogram(rfm, x='Predicted_CLV', nbins=50)
        fig_clv = apply_chart_theme(fig_clv)
        fig_clv.update_traces(marker_color=Colors.BLUE_ENERGY)
        fig_clv.update_layout(height=400, showlegend=False, title="Distribution Histogram")
        st.plotly_chart(fig_clv, use_container_width=True)
    
    with col2:
        st.markdown("#### Average CLV by Segment")
        seg_clv = rfm.groupby('Segment')['Predicted_CLV'].mean().sort_values(ascending=False)
        fig_seg = px.bar(x=seg_clv.values, y=seg_clv.index, orientation='h')
        fig_seg = apply_chart_theme(fig_seg)
        fig_seg.update_traces(marker_color=Colors.MINT_LEAF)
        fig_seg.update_layout(height=400, showlegend=False, title="Average CLV by Segment")
        st.plotly_chart(fig_seg, use_container_width=True)
    
    st.markdown("---")
    st.markdown("### 🏆 Top 20 Customers by Predicted CLV")
    
    top_clv = rfm.nlargest(20, 'Predicted_CLV')[
        ['CustomerID', 'Predicted_CLV', 'Monetary', 'Frequency', 'Recency', 'Segment', 'Country']
    ]
    
    st.dataframe(
        top_clv.style.format({'Predicted_CLV': '£{:,.2f}', 'Monetary': '£{:,.2f}',
                              'Frequency': '{:.0f}', 'Recency': '{:.0f}'})
        .background_gradient(subset=['Predicted_CLV'], cmap='Greens'),
        use_container_width=True, height=500
    )

with tab2:
    st.markdown(Components.section_header("Churn Risk Analysis", "🚨"), unsafe_allow_html=True)
    
    total_cust = len(rfm)
    high_risk = len(rfm[rfm['ChurnRisk'] == 'High'])
    med_risk = len(rfm[rfm['ChurnRisk'] == 'Medium'])
    low_risk = len(rfm[rfm['ChurnRisk'] == 'Low'])
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(Components.metric_card("Total Customers", f"{total_cust:,}",
                    "👥 Active base", True, "👥", "primary"), unsafe_allow_html=True)
    with col2:
        st.markdown(Components.metric_card("High Risk", f"{high_risk:,}",
                    f"🚨 {(high_risk/total_cust)*100:.1f}%", False, "🚨", "error"), unsafe_allow_html=True)
    with col3:
        st.markdown(Components.metric_card("Medium Risk", f"{med_risk:,}",
                    f"⚠️ {(med_risk/total_cust)*100:.1f}%", False, "⚠️", "warning"), unsafe_allow_html=True)
    with col4:
        st.markdown(Components.metric_card("Low Risk", f"{low_risk:,}",
                    f"✅ {(low_risk/total_cust)*100:.1f}%", True, "✅", "success"), unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### 📊 Churn Risk Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Risk Level Breakdown")
        risk_counts = rfm['ChurnRisk'].value_counts().reset_index()
        risk_counts.columns = ['ChurnRisk', 'Count']
        color_map = {'Low': Colors.MINT_LEAF, 'Medium': '#FFB84D', 'High': '#FF6B6B'}
        
        fig_risk = px.pie(risk_counts, values='Count', names='ChurnRisk', color='ChurnRisk',
                          color_discrete_map=color_map)
        fig_risk = apply_chart_theme(fig_risk)
        fig_risk.update_traces(textposition='inside', textinfo='percent+label')
        fig_risk.update_layout(height=400, title="Risk Level Breakdown")
        st.plotly_chart(fig_risk, use_container_width=True)
    
    with col2:
        st.markdown("#### Churn Probability Distribution")
        fig_prob = px.histogram(rfm, x='ChurnProbability', nbins=50)
        fig_prob = apply_chart_theme(fig_prob)
        fig_prob.update_traces(marker_color='#FF6B6B')
        fig_prob.update_layout(height=400, showlegend=False, title="Churn Probability Distribution")
        fig_prob.update_xaxes(tickformat='.0%')
        st.plotly_chart(fig_prob, use_container_width=True)
    
    st.markdown("---")
    st.markdown("### ⚠️ Top 20 High-Risk Customers")
    
    high_risk_val = rfm[rfm['ChurnRisk'] == 'High']['Predicted_CLV'].sum()
    st.markdown(Components.insight_box("🚨 Critical Alert",
        f"<p><strong>{high_risk} high-risk customers</strong> with value of £{high_risk_val:,.0f}</p>",
        "error"), unsafe_allow_html=True)
    
    high_risk_cust = rfm[rfm['ChurnRisk'] == 'High'].nlargest(20, 'ChurnProbability')[
        ['CustomerID', 'ChurnProbability', 'Predicted_CLV', 'Recency', 'Frequency', 'Monetary', 'Segment']
    ]
    
    st.dataframe(
        high_risk_cust.style.format({'ChurnProbability': '{:.1%}', 'Predicted_CLV': '£{:,.2f}',
                                     'Recency': '{:.0f}', 'Frequency': '{:.0f}', 'Monetary': '£{:,.2f}'})
        .background_gradient(subset=['ChurnProbability'], cmap='Reds'),
        use_container_width=True, height=500
    )

with tab3:
    st.markdown(Components.section_header("Strategic Customer Matrix", "🎯"), unsafe_allow_html=True)
    
    st.markdown(Components.insight_box("🎯 Understanding the Matrix",
        """<p>Four strategic quadrants:</p>
        <ul><li><strong>High CLV + High Risk:</strong> 🚨 URGENT - Save these VIPs</li>
        <li><strong>High CLV + Low Risk:</strong> 💎 MAINTAIN - Your champions</li>
        <li><strong>Low CLV + High Risk:</strong> ⚖️ EVALUATE - Cost-benefit analysis</li>
        <li><strong>Low CLV + Low Risk:</strong> 📈 GROW - Upsell opportunities</li></ul>""",
        "info"), unsafe_allow_html=True)
    
    rfm['CLV_Category'] = pd.qcut(rfm['Predicted_CLV'], q=3, labels=['Low CLV', 'Medium CLV', 'High CLV'], duplicates='drop')
    
    fig_matrix = px.scatter(rfm, x='Predicted_CLV', y='ChurnProbability', color='ChurnRisk',
        size='Frequency', hover_data=['CustomerID', 'Segment'],
        color_discrete_map={'Low': Colors.MINT_LEAF, 'Medium': '#FFB84D', 'High': '#FF6B6B'})
    fig_matrix = apply_chart_theme(fig_matrix)
    fig_matrix.update_layout(height=600, title="Customer CLV vs. Churn Probability Matrix")
    fig_matrix.update_yaxes(tickformat='.0%')
    st.plotly_chart(fig_matrix, use_container_width=True)
    
    st.markdown("---")
    st.markdown("### 💡 Recommended Actions by Segment")
    
    col1, col2 = st.columns(2)
    
    with col1:
        high_val_risk = rfm[(rfm['CLV_Category'] == 'High CLV') & (rfm['ChurnRisk'] == 'High')]
        st.markdown(Components.insight_box("🚨 High CLV + High Risk",
            f"<p><strong>{len(high_val_risk):,} customers</strong> | "
            f"£{high_val_risk['Predicted_CLV'].sum():,.0f} at risk</p>"
            "<p><strong>Actions:</strong> Personal outreach, VIP offers (20-30%), immediate resolution</p>",
            "error"), unsafe_allow_html=True)
        
        growth = rfm[(rfm['CLV_Category'].isin(['Low CLV', 'Medium CLV'])) & (rfm['ChurnRisk'] == 'Low')]
        st.markdown(Components.insight_box("📈 Growth Opportunity",
            f"<p><strong>{len(growth):,} customers</strong> | Potential: £{growth['Predicted_CLV'].sum():,.0f}</p>"
            "<p><strong>Actions:</strong> Cross-sell, upsell, frequency campaigns, product bundles</p>",
            "success"), unsafe_allow_html=True)
    
    with col2:
        champions = rfm[(rfm['CLV_Category'] == 'High CLV') & (rfm['ChurnRisk'] == 'Low')]
        st.markdown(Components.insight_box("💎 Champions",
            f"<p><strong>{len(champions):,} customers</strong> | Value: £{champions['Predicted_CLV'].sum():,.0f}</p>"
            "<p><strong>Actions:</strong> VIP treatment, early access, referral programs, upsell premium</p>",
            "success"), unsafe_allow_html=True)
        
        low_val_risk = rfm[(rfm['CLV_Category'] == 'Low CLV') & (rfm['ChurnRisk'] == 'High')]
        st.markdown(Components.insight_box("⚖️ Evaluate",
            f"<p><strong>{len(low_val_risk):,} customers</strong> | Marginal value</p>"
            "<p><strong>Actions:</strong> Automated win-back, cost-benefit analysis, survey feedback</p>",
            "warning"), unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### 📥 Export Customer Lists")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        high_priority = high_val_risk[['CustomerID', 'ChurnProbability', 'Predicted_CLV',
                                       'Recency', 'Frequency', 'Monetary', 'Segment']]
        csv_hp = high_priority.to_csv(index=False)
        st.download_button("📥 Download High-Risk VIPs", csv_hp, "high_risk_vips.csv",
                          "text/csv", use_container_width=True)
        st.markdown(Components.insight_box("🚨 High Priority",
            f"<p>{len(high_priority):,} customers<br>£{high_priority['Predicted_CLV'].sum():,.0f}</p>",
            "error"), unsafe_allow_html=True)
    
    with col2:
        champs = champions[['CustomerID', 'ChurnProbability', 'Predicted_CLV',
                           'Recency', 'Frequency', 'Monetary', 'Segment']]
        csv_ch = champs.to_csv(index=False)
        st.download_button("📥 Download Champions", csv_ch, "champions.csv",
                          "text/csv", use_container_width=True)
        st.markdown(Components.insight_box("💎 Champions",
            f"<p>{len(champs):,} customers<br>£{champs['Predicted_CLV'].sum():,.0f}</p>",
            "success"), unsafe_allow_html=True)
    
    with col3:
        grow = growth[['CustomerID', 'ChurnProbability', 'Predicted_CLV',
                      'Recency', 'Frequency', 'Monetary', 'Segment']]
        csv_gr = grow.to_csv(index=False)
        st.download_button("📥 Download Growth Targets", csv_gr, "growth_targets.csv",
                          "text/csv", use_container_width=True)
        st.markdown(Components.insight_box("📈 Growth",
            f"<p>{len(grow):,} customers<br>£{grow['Predicted_CLV'].sum():,.0f}</p>",
            "info"), unsafe_allow_html=True)

# ===========================================================
# FOOTER
# ===========================================================
st.markdown("---")
st.markdown(f"<p style='text-align: center; color: #5A6A7A;'>CLV & Churn Analysis | "
            f"{total_cust:,} customers | {high_risk:,} high-risk | Accuracy: {churn_acc:.1%}</p>",
            unsafe_allow_html=True)

