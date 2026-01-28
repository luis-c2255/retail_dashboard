import panel as pn

def skeleton_card(width="100%", height="140px"):
    return pn.pane.HTML(
        f"""
        <div class="skeleton-wrapper" style="width:{width};">
            <div class="skeleton-card" style="width:100%; height:{height};"></div>
        </div>
        """,
        sizing_mode="stretch_width",
        escape=False,
    )
