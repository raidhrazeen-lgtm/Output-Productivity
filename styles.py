"""
Theme constants for the UK Productivity Puzzle dashboard.
Teal blue background theme with black text.
"""

# Colors
TEAL_BACKGROUND = '#008080'  # Teal blue from user specification
COLOR_PRIMARY = '#1f77b4'
COLOR_SECONDARY = '#ff7f0e'
COLOR_GAP = 'rgba(255, 0, 0, 0.2)'
COLOR_ACTUAL = '#1f77b4'
COLOR_COUNTERFACTUAL = '#2ca02c'
COLOR_GFC = '#d62728'
COLOR_COVID = '#9467bd'
TEXT_COLOR = '#000000'  # Black text

# Chart layout defaults
CHART_LAYOUT = {
    'plot_bgcolor': '#008080',  # Teal background
    'paper_bgcolor': '#008080',  # Teal background
    'font': {'family': 'Arial, sans-serif', 'size': 12, 'color': '#000000'},  # Arial font, black text
    'xaxis': {
        'showgrid': True,
        'gridcolor': 'rgba(0,0,0,0.2)',
        'gridwidth': 1,
        'showline': True,
        'linecolor': 'rgba(0,0,0,0.3)',
        'title': {'font': {'family': 'Arial, sans-serif', 'color': '#000000'}},
        'tickfont': {'family': 'Arial, sans-serif', 'color': '#000000'},
    },
    'yaxis': {
        'showgrid': True,
        'gridcolor': 'rgba(0,0,0,0.2)',
        'gridwidth': 1,
        'showline': True,
        'linecolor': 'rgba(0,0,0,0.3)',
        'title': {'font': {'family': 'Arial, sans-serif', 'color': '#000000'}},
        'tickfont': {'family': 'Arial, sans-serif', 'color': '#000000'},
    },
    'margin': {'l': 60, 'r': 20, 't': 40, 'b': 60},
}

# KPI card styles
KPI_CARD_STYLE = {
    'backgroundColor': '#ffffff',  # White cards on teal background
    'padding': '20px',
    'borderRadius': '5px',
    'border': '1px solid #000000',
    'textAlign': 'center',
    'color': '#000000',  # Black text
    'fontFamily': 'Arial, sans-serif',
}

