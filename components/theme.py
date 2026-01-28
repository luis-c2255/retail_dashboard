# components/theme.py

PRIMARY = "#0A1A2F"
PRIMARY_LIGHT = "#0D47A1"
ACCENT = "#26A69A"

SURFACE = "#121212"
CARD_BG = "#1E1E1E"
BORDER = "#2A2A2A"

TEXT_PRIMARY = "#E3EAF2"
TEXT_SECONDARY = "#A0A7B4"
TEXT_MUTED = "#6C7480"

CHART_PRIMARY = "#64B5F6"
CHART_SECONDARY = "#4DD0E1"
CHART_TERTIARY = "#81C784"
CHART_WARNING = "#FFB74D"
CHART_ERROR = "#EF5350"

SUCCESS = CHART_TERTIARY
INFO = CHART_PRIMARY

HOVER = "#1A2A3F"
ACTIVE = "#0F1F33"

def palette():
    return {
        "primary": PRIMARY,
        "primary_light": PRIMARY_LIGHT,
        "accent": ACCENT,
        "surface": SURFACE,
        "card": CARD_BG,
        "border": BORDER,
        "text": TEXT_PRIMARY,
        "text_secondary": TEXT_SECONDARY,
        "text_muted": TEXT_MUTED,
        "chart_primary": CHART_PRIMARY,
        "chart_secondary": CHART_SECONDARY,
        "chart_tertiary": CHART_TERTIARY,
        "chart_warning": CHART_WARNING,
        "chart_error": CHART_ERROR,
        "success": SUCCESS,
        "info": INFO,
        "hover": HOVER,
        "active": ACTIVE,
    }
