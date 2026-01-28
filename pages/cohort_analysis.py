import io
import panel as pn
import hvplot.pandas  # noqa
import pandas as pd

from components.kpi_cards import kpi
from components.utils import load_clean_data
from components.skeleton import skeleton_card

pn.extension()

retail, month_metrics, product_metrics, country_metrics, rfm = load_clean_data()

def view(country_filter, date_range, customer_filter):

    # ---------------------------------------------------------
    # 1. FILTERED RETAIL DATA
    # ---------------------------------------------------------
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

    # ---------------------------------------------------------
    # 2. COHORT ASSIGNMENT + RETENTION MATRIX
    # ---------------------------------------------------------
    @pn.depends(filtered_retail)
    def cohort_data(d: pd.DataFrame):
        if d.empty:
            return None, None, None

        df = d.copy()
        df["InvoiceMonth"] = df["InvoiceDate"].dt.to_period("M")
        df["CohortMonth"] = df.groupby("CustomerID")["InvoiceMonth"].transform("min")

        df["CohortIndex"] = (
            (df["InvoiceMonth"].dt.year - df["CohortMonth"].dt.year) * 12 +
            (df["InvoiceMonth"].dt.month - df["CohortMonth"].dt.month)
        )

        cohort_sizes = df.groupby("CohortMonth")["CustomerID"].nunique()

        retention = (
            df.groupby(["CohortMonth", "CohortIndex"])["CustomerID"]
            .nunique()
            .unstack(fill_value=0)
            .div(cohort_sizes, axis=0)
            .round(3)
        )

        return df, retention, cohort_sizes

    # ---------------------------------------------------------
    # 3. KPI ROW
    # ---------------------------------------------------------
    @pn.depends(cohort_data)
    def kpi_row(values):
        df, retention, cohort_sizes = values
        if df is None:
            return pn.Row("No cohort data for current filters.")

        total_cohorts = retention.shape[0]
        avg_ret_1 = retention[1].mean() if 1 in retention.columns else 0
        avg_ret_3 = retention[3].mean() if 3 in retention.columns else 0
        avg_ret_6 = retention[6].mean() if 6 in retention.columns else 0

        return pn.Row(
            kpi("Total Cohorts", f"{total_cohorts}", icon="calendar_month"),
            kpi("Avg Retention (1M)", f"{avg_ret_1:.2%}", icon="trending_up"),
            kpi("Avg Retention (3M)", f"{avg_ret_3:.2%}", icon="timeline"),
            kpi("Avg Retention (6M)", f"{avg_ret_6:.2%}", icon="insights"),
            sizing_mode="stretch_width",
        )

    # ---------------------------------------------------------
    # 4. RETENTION HEATMAP
    # ---------------------------------------------------------
    @pn.depends(cohort_data)
    def retention_heatmap(values):
        df, retention, cohort_sizes = values
        if df is None:
            return pn.pane.Markdown("_No retention data available._")

        heatmap = retention.hvplot.heatmap(
            x="CohortIndex",
            y="CohortMonth",
            C=retention,
            cmap="RdYlGn",
            title="Cohort Retention Heatmap",
            height=450,
        ).opts(transition=200)

        return heatmap

    # ---------------------------------------------------------
    # 5. RETENTION CURVES
    # ---------------------------------------------------------
    @pn.depends(cohort_data)
    def retention_curves(values):
        df, retention, cohort_sizes = values
        if df is None:
            return pn.pane.Markdown("_No retention curves available._")

        curves = retention.T.hvplot(
            title="Retention Curves by Cohort",
            height=350,
            alpha=0.6,
            line_width=2,
            cmap="Category20",
        ).opts(transition=200)

        return curves

    # ---------------------------------------------------------
    # 6. COHORT SIZE ANALYSIS
    # ---------------------------------------------------------
    @pn.depends(cohort_data)
    def cohort_size_plot(values):
        df, retention, cohort_sizes = values
        if df is None:
            return pn.pane.Markdown("_No cohort size data available._")

        size_df = cohort_sizes.reset_index()
        size_df.columns = ["CohortMonth", "Size"]

        return size_df.hvplot.bar(
            x="CohortMonth",
            y="Size",
            title="Initial Cohort Sizes",
            color="#64B5F6",
            height=350,
        ).opts(transition=200)

    # ---------------------------------------------------------
    # 7. DETAILED COHORT TABLES
    # ---------------------------------------------------------
    @pn.depends(cohort_data)
    def cohort_tables(values):
        df, retention, cohort_sizes = values
        if df is None:
            return pn.pane.Markdown("_No cohort tables available._")

        retention_pct = retention.copy()
        retention_pct.index = retention_pct.index.astype(str)

        table = pn.widgets.DataFrame(
            retention_pct,
            autosize_mode="fit_viewport",
            height=350,
        )

        def _download(event):
            buf = io.StringIO()
            retention_pct.to_csv(buf)
            buf.seek(0)
            download.file = ("cohort_retention.csv", buf.read())

        download = pn.widgets.FileDownload(
            label="Download Cohort Table CSV",
            button_type="primary",
            filename="cohort_retention.csv",
        )
        download.on_click(_download)

        return pn.Column(download, table)

    # ---------------------------------------------------------
    # 8. STRATEGIC INSIGHTS
    # ---------------------------------------------------------
    @pn.depends(cohort_data)
    def strategic_insights(values):
        df, retention, cohort_sizes = values
        if df is None:
            return pn.pane.Markdown("_No insights available._")

        best_cohort = retention[1].idxmax() if 1 in retention.columns else None
        dropoff = retention.diff(axis=1).mean().mean()

        md = f"""
### 📌 Key Insights

**Best Cohort:** {best_cohort}  
These customers show the strongest month‑1 retention.

**Average Drop‑off:** {dropoff:.2%}  
Indicates how quickly customers disengage after their first purchase.

**Recommendations**
- Strengthen onboarding for new cohorts.
- Improve month‑1 engagement (emails, offers, reminders).
- Monitor retention curves for early warning signals.
"""

        return pn.pane.Markdown(md)

    # ---------------------------------------------------------
    # PAGE LAYOUT
    # ---------------------------------------------------------
    return pn.Column(
        "## Cohort Analysis",
        pn.panel(kpi_row, loading_indicator=True, placeholder=skeleton_card()),
        pn.Spacer(height=10),
        pn.panel(retention_heatmap, loading_indicator=True, placeholder=skeleton_card(height="450px")),
        pn.Spacer(height=10),
        pn.panel(retention_curves, loading_indicator=True, placeholder=skeleton_card(height="350px")),
        pn.Spacer(height=10),
        pn.panel(cohort_size_plot, loading_indicator=True, placeholder=skeleton_card(height="350px")),
        pn.Spacer(height=10),
        pn.panel(cohort_tables, loading_indicator=True, placeholder=skeleton_card(height="350px")),
        pn.Spacer(height=10),
        pn.panel(strategic_insights, loading_indicator=True, placeholder=skeleton_card(height="200px")),
        sizing_mode="stretch_width",
    )
