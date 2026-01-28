import panel as pn
import hvplot.pandas  # noqa
import pandas as pd
import numpy as np

from components.kpi_cards import kpi
from components.utils import load_clean_data
from components.skeleton import skeleton_card

pn.extension()

retail, month_metrics, product_metrics, country_metrics, rfm = load_clean_data()

def view(country_filter, date_range, customer_filter):

    # -----------------------------
    # 1. FILTERED RETAIL DATA
    # -----------------------------
    @pn.depends(country_filter, date_range, customer_filter)
    def filtered_retail():
        d = retail.copy()

        if country_filter.value:
            d = d[d["Country"].isin(country_filter.value)]

        if customer_filter.value:
            d = d[d["CustomerID"].isin(customer_filter.value)]

        start, end = date_range.value
        d = d[(d["InvoiceDate"] >= start) & (d["InvoiceDate"] <= end)]

        return d

    # -----------------------------
    # 2. HISTORICAL SERIES + FORECAST
    # -----------------------------
    def build_forecast(d: pd.DataFrame, horizon_months: int = 12):
        if d.empty:
            return None, None, None

        m = (
            d.groupby(d["InvoiceDate"].dt.to_period("M"))["TotalAmount"]
            .sum()
            .reset_index()
        )
        m["Month"] = m["InvoiceDate"].dt.to_timestamp()
        m = m.sort_values("Month")

        if len(m) < 3:
            return m, None, None

        # estimate average monthly growth rate
        first = m["TotalAmount"].iloc[0]
        last = m["TotalAmount"].iloc[-1]
        n = len(m)
        if first <= 0 or n <= 1:
            growth_rate = 0.0
        else:
            growth_rate = (last / first) ** (1 / (n - 1)) - 1

        last_month = m["Month"].iloc[-1]
        forecast_months = pd.period_range(
            last_month.to_period("M") + 1, periods=horizon_months, freq="M"
        ).to_timestamp()

        forecast_values = []
        current = last
        for _ in range(horizon_months):
            current = current * (1 + growth_rate)
            forecast_values.append(current)

        f = pd.DataFrame(
            {"Month": forecast_months, "Forecast": forecast_values}
        )

        # confidence band ±15%
        f["Lower"] = f["Forecast"] * 0.85
        f["Upper"] = f["Forecast"] * 1.15

        return m, f, growth_rate

    @pn.depends(filtered_retail)
    def forecast_data(d: pd.DataFrame):
        return build_forecast(d)

    # -----------------------------
    # 3. KPIS
    # -----------------------------
    @pn.depends(forecast_data)
    def kpi_row(values):
        hist, fcast, growth_rate = values
        if hist is None or fcast is None:
            return pn.Row("Not enough data to compute forecast.")

        hist_avg = hist["TotalAmount"].mean()
        forecast_avg = fcast["Forecast"].mean()
        total_forecast = fcast["Forecast"].sum()

        return pn.Row(
            kpi("Historical Avg", f"£{hist_avg:,.0f}", icon="bar_chart"),
            kpi("Forecast Avg", f"£{forecast_avg:,.0f}", icon="show_chart"),
            kpi("Growth Rate", f"{growth_rate*100:.1f}%", icon="trending_up"),
            kpi("Total Forecast", f"£{total_forecast:,.0f}", icon="stacked_line_chart"),
            sizing_mode="stretch_width",
        )

    # -----------------------------
    # 4. MAIN FORECAST CHART
    # -----------------------------
    @pn.depends(forecast_data)
    def forecast_chart(values):
        hist, fcast, growth_rate = values
        if hist is None or fcast is None:
            return pn.pane.Markdown("_Not enough data to display forecast chart._")

        hist_line = hist.hvplot.line(
            x="Month",
            y="TotalAmount",
            color="#64B5F6",
            line_width=3,
            label="Historical",
            title="Sales Forecast (Next 12 Months)",
            height=400,
        )

        fcast_line = fcast.hvplot.line(
            x="Month",
            y="Forecast",
            color="#FFB74D",
            line_width=3,
            line_dash="dashed",
            label="Forecast",
        )

        band = fcast.hvplot.area(
            x="Month",
            y="Upper",
            y2="Lower",
            color="#FFB74D",
            alpha=0.15,
            label="Confidence Band",
        )

        return (hist_line * band * fcast_line).opts(transition=200)

    # -----------------------------
    # 5. DETAILED FORECAST TABLE
    # -----------------------------
    @pn.depends(forecast_data)
    def forecast_table(values):
        hist, fcast, growth_rate = values
        if hist is None or fcast is None:
            return pn.pane.Markdown("_No forecast table available._")

        df = pd.merge(
            hist[["Month", "TotalAmount"]],
            fcast[["Month", "Forecast"]],
            on="Month",
            how="outer",
        ).sort_values("Month")

        df["MoM_Growth"] = df["TotalAmount"].pct_change() * 100

        table = pn.widgets.DataFrame(
            df,
            autosize_mode="fit_viewport",
            height=350,
        )

        return table

    # -----------------------------
    # 6. GROWTH ANALYSIS
    # -----------------------------
    @pn.depends(forecast_data)
    def growth_analysis(values):
        hist, fcast, growth_rate = values
        if hist is None or fcast is None:
            return pn.pane.Markdown("_No growth analysis available._")

        hist_growth = hist["TotalAmount"].pct_change().mean() * 100
        fcast_growth = (fcast["Forecast"].pct_change().mean() * 100) if len(fcast) > 1 else 0

        md = f"""
### 📈 Growth Analysis

- **Historical average MoM growth:** {hist_growth:.2f}%
- **Forecasted average MoM growth:** {fcast_growth:.2f}%
- **Estimated long‑term growth rate:** {growth_rate*100:.2f}%

Use this to align expectations for revenue planning and to detect whether the forecast is optimistic or conservative.
"""
        return pn.pane.Markdown(md)

    # -----------------------------
    # 7. SEASONAL PATTERNS
    # -----------------------------
    @pn.depends(filtered_retail)
    def seasonal_patterns(d: pd.DataFrame):
        if d.empty:
            return pn.pane.Markdown("_No data available for seasonality._")

        d = d.copy()
        d["MonthNum"] = d["InvoiceDate"].dt.month
        season = (
            d.groupby("MonthNum")["TotalAmount"]
            .sum()
            .reset_index()
            .rename(columns={"TotalAmount": "Revenue"})
        )

        season["MonthName"] = season["MonthNum"].apply(
            lambda x: pd.Timestamp(2000, x, 1).strftime("%b")
        )

        return season.hvplot.bar(
            x="MonthName",
            y="Revenue",
            title="Average Revenue by Calendar Month (Seasonality)",
            color="#26A69A",
            height=350,
        ).opts(transition=200)

    # -----------------------------
    # 8. STRATEGIC RECOMMENDATIONS
    # -----------------------------
    @pn.depends(forecast_data, filtered_retail)
    def strategic_recommendations(values, d: pd.DataFrame):
        hist, fcast, growth_rate = values
        if hist is None or fcast is None or d.empty:
            return pn.pane.Markdown("_No data available for recommendations._")

        peak_month = fcast.loc[fcast["Forecast"].idxmax(), "Month"]
        peak_label = peak_month.strftime("%Y-%m")

        md = f"""
### 📌 Strategic Recommendations

**Peak Month Forecast**
- Highest forecasted revenue in: **{peak_label}**
- Prepare inventory, staffing, and marketing for this period.

**Inventory Management**
- Align stock levels with forecasted peaks and troughs.
- Use the confidence band (±15%) as a buffer for uncertainty.

**Budget Planning**
- Allocate marketing and operational budgets in line with forecasted growth.
- Monitor actual vs forecast monthly and adjust spend accordingly.

**Performance Monitoring**
- Set variance alerts (e.g., ±10%) between actual and forecast.
- Investigate large deviations to refine assumptions and improve the model.
"""
        return pn.pane.Markdown(md)

    # -----------------------------
    # PAGE LAYOUT
    # -----------------------------
    return pn.Column(
        "## Sales Forecasting",
        pn.panel(kpi_row, loading_indicator=True, placeholder=skeleton_card()),
        pn.Spacer(height=10),
        pn.panel(forecast_chart, loading_indicator=True, placeholder=skeleton_card(height="420px")),
        pn.Spacer(height=10),
        pn.panel(forecast_table, loading_indicator=True, placeholder=skeleton_card(height="360px")),
        pn.Spacer(height=10),
        pn.panel(growth_analysis, loading_indicator=True, placeholder=skeleton_card(height="220px")),
        pn.Spacer(height=10),
        pn.panel(seasonal_patterns, loading_indicator=True, placeholder=skeleton_card(height="360px")),
        pn.Spacer(height=10),
        pn.panel(strategic_recommendations, loading_indicator=True, placeholder=skeleton_card(height="260px")),
        sizing_mode="stretch_width",
    )
