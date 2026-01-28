import panel as pn
from components.utils import load_clean_data
from components.filters import global_filters
from components.theme import palette
from pages import basket_analysis, clv_churn, cohort_analysis, customer_analysis, overview, forecasting
from components.chart_theme import apply_dark_theme

pn.extension()
apply_dark_theme()

# Load datasets and model artifacts
(
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
) = load_clean_data()

df = retail  # global filters use the retail dataset

colors = palette()
country_filter, date_range, customer_filter = global_filters(df)

template = pn.template.MaterialTemplate(
    title="Retail Analytics Dashboard",
    theme="dark",
    header_background=colors["primary"],
    header_color="white",
    sidebar_width=280,
    collapsed_sidebar=False,
)

template.sidebar[:] = [
    pn.pane.Markdown("### Filters", css_classes=["sidebar-section-title"]),
    pn.Accordion(
        ("Geography", pn.Column(country_filter)),
        ("Date Range", pn.Column(date_range)),
        ("Customers", pn.Column(customer_filter)),
        active=[0],
        sizing_mode="stretch_width",
    )
]

template.main.append(
    pn.Tabs(
        ("Overview", overview.view(country_filter, date_range, customer_filter)),
        ("Customer Analysis", customer_analysis.view(country_filter, date_range, customer_filter)),
        ("Basket Analysis", basket_analysis.view(country_filter, date_range, customer_filter)),
        ("Cohort Analysis", cohort_analysis.view(country_filter, date_range, customer_filter)),
        ("Forecasting", forecasting.view(country_filter, date_range, customer_filter)),
        ("CLV & Churn", clv_churn.view(country_filter, date_range, customer_filter)),
    )
)

template.servable()
