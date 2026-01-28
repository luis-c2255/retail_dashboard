import panel as pn

def kpi(title, value, icon="trending_up"):
    return pn.pane.HTML(
        f"""
        <div class="kpi-wrapper">
            <div class="kpi-card" style="display:flex; align-items:center; gap:12px;">
                <div class="kpi-icon">
                    <span class="material-icons">{icon}</span>
                </div>
                <div class="kpi-content">
                    <div class="kpi-title">{title}</div>
                    <div class="kpi-value">{value}</div>
                </div>        
            </div>
        </div>
        """,
        sizing_mode="stretch_width",
        escape=False,
    )
