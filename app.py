"""
UK Productivity Puzzle Dashboard
Production-ready Plotly Dash application for analyzing UK productivity trends.
"""

import dash
from dash import dcc, html, Input, Output, State
import pandas as pd
import numpy as np
import warnings
from datetime import datetime

# Import data loading and chart functions
from data_loader import (
    load_gdp_qoq,
    load_production_index,
    load_productivity,
    load_income_deciles,
    load_sector_productivity,
    index_to_base,
    create_counterfactual,
)
from charts import (
    chart_productivity_gap,
    chart_growth_comparison,
    chart_gdp_vs_productivity,
    chart_production_output,
    chart_income_by_nation,
    chart_sector_heatmap,
    chart_sector_productivity_with_gdp,
    get_sector_category_mapping,
    chart_sector_category,
)
from styles import KPI_CARD_STYLE, TEAL_BACKGROUND, TEXT_COLOR, APP_OWNER
from utils import (
    apply_rolling_mean,
    apply_indexing,
    prepare_productivity_series,
    prepare_counterfactual_series,
)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Measure name standardization
def standardize_measure_name(raw_name: str) -> str:
    """
    Standardize productivity measure names from various ONS formats.
    
    Args:
        raw_name: Raw measure name from data
        
    Returns:
        Standardized measure name
    """
    if pd.isna(raw_name):
        return 'Output per hour'
    
    raw_lower = str(raw_name).lower()
    
    # Map various formats to standard names
    if 'hour' in raw_lower or 'hour worked' in raw_lower:
        return 'Output per hour'
    elif 'worker' in raw_lower:
        return 'Output per worker'
    elif 'job' in raw_lower or 'filled job' in raw_lower:
        return 'Output per job'
    else:
        # Return as-is if no match
        return str(raw_name).strip()

# Initialize Dash app
app = dash.Dash(__name__)
app.title = "UK Productivity Puzzle"

# Load all data
print("Loading data...")
try:
    gdp_df = load_gdp_qoq()
    print(f"✓ GDP data loaded: {len(gdp_df)} records from {gdp_df['date'].min()} to {gdp_df['date'].max()}")
except Exception as e:
    print(f"✗ Error loading GDP data: {e}")
    gdp_df = pd.DataFrame(columns=['date', 'gdp_qoq'])

try:
    production_df = load_production_index()
    print(f"✓ Production data loaded: {len(production_df)} records from {production_df['date'].min()} to {production_df['date'].max()}")
except Exception as e:
    print(f"✗ Error loading production data: {e}")
    production_df = pd.DataFrame(columns=['date', 'prod_index'])

productivity_error_message = None
try:
    productivity_df, sector_df, productivity_error_message = load_productivity()
    if productivity_df is not None and len(productivity_df) > 0:
        # Standardize measure names
        productivity_df['measure'] = productivity_df['measure'].apply(standardize_measure_name)
        measures = sorted(productivity_df['measure'].unique().tolist())
        date_range = (productivity_df['date'].min(), productivity_df['date'].max())
        print(f"✓ Productivity data loaded: {len(productivity_df)} records")
        print(f"  Available measures: {measures}")
        print(f"  Unique measure values in dataframe: {productivity_df['measure'].unique().tolist()}")
        print(f"  Date range: {date_range[0]} to {date_range[1]}")
        productivity_error_message = None  # Clear error if successful
    else:
        if productivity_error_message is None:
            productivity_error_message = "No productivity data extracted from lprod01.xls"
        print(f"✗ No productivity data loaded: {productivity_error_message}")
        measures = ['Output per hour']
        productivity_df = pd.DataFrame(columns=['date', 'measure', 'value'])
    
    if sector_df is not None and len(sector_df) > 0:
        sectors = sector_df['sector'].unique()
        print(f"✓ Sector productivity loaded: {len(sectors)} sectors")
    else:
        print("⚠ Sector productivity not available")
        sector_df = None
except Exception as e:
    import traceback
    productivity_error_message = f"{type(e).__name__}: {str(e)}"
    print(f"✗ Error loading productivity data:")
    print(traceback.format_exc())
    productivity_df = pd.DataFrame(columns=['date', 'measure', 'value'])
    sector_df = None
    measures = ['Output per hour']

try:
    income_df = load_income_deciles()
    if len(income_df) > 0:
        regions = income_df['region'].unique()
        year_range = (income_df['year'].min(), income_df['year'].max())
        print(f"✓ Income data loaded: {len(income_df)} records")
        print(f"  Regions: {', '.join(regions)}")
        print(f"  Year range: {year_range[0].year} to {year_range[1].year}")
    else:
        print("✗ No income data loaded")
except Exception as e:
    print(f"✗ Error loading income data: {e}")
    income_df = pd.DataFrame(columns=['year', 'region', 'median_income'])

try:
    sector_productivity_df = load_sector_productivity()
    if len(sector_productivity_df) > 0:
        sector_list = sector_productivity_df['sector'].unique().tolist()
        print(f"✓ Sector productivity data loaded: {len(sector_productivity_df)} records")
        print(f"  Sectors: {len(sector_list)} unique sectors")
    else:
        print("⚠ No sector productivity data loaded from lprod01.xls")
        sector_list = []
except Exception as e:
    print(f"✗ Error loading sector productivity data: {e}")
    sector_productivity_df = pd.DataFrame(columns=['date', 'sector', 'value'])
    sector_list = []

print("\nData loading complete!\n")

# Prepare productivity counterfactual data
productivity_counterfactual = {}
if len(productivity_df) > 0:
    for measure in measures:
        measure_df = productivity_df[productivity_df['measure'] == measure].copy()
        if len(measure_df) > 0:
            try:
                counterfactual = create_counterfactual(measure_df, 'date', 'value')
                productivity_counterfactual[measure] = counterfactual
            except Exception as e:
                print(f"Warning: Could not create counterfactual for {measure}: {e}")

# Get date ranges for slider
all_dates = []
if len(productivity_df) > 0:
    all_dates.extend(productivity_df['date'].tolist())
if len(gdp_df) > 0:
    all_dates.extend(gdp_df['date'].tolist())
if len(production_df) > 0:
    all_dates.extend(production_df['date'].tolist())

if all_dates:
    min_date = min(all_dates)
    max_date = max(all_dates)
else:
    min_date = pd.Timestamp('1997-01-01')
    max_date = pd.Timestamp('2024-12-31')

# App layout
app.layout = html.Div([
    # Header
    html.Div([
        html.Div([
            html.Div([
                html.H1("UK Productivity Puzzle", style={
                    'margin': '0',
                    'marginBottom': '5px',
                    'color': TEXT_COLOR,
                    'fontFamily': 'Arial, sans-serif',
                    'fontSize': '32px'
                }),
                html.P("Why UK productivity stalled after 2008", 
                       style={
                           'margin': '0',
                           'color': TEXT_COLOR, 
                           'fontSize': '16px',
                           'fontFamily': 'Arial, sans-serif'
                       })
            ], style={'flex': '1'}),
            html.Div([
                html.Div(APP_OWNER['name'], style={
                    'fontWeight': 'bold',
                    'fontSize': '12px',
                    'marginBottom': '3px',
                    'lineHeight': '1.4',
                    'color': TEXT_COLOR,
                    'fontFamily': 'Arial, sans-serif'
                }),
                html.Div([
                    html.A(APP_OWNER['email'], 
                           href=f'mailto:{APP_OWNER["email"]}',
                           style={'color': TEXT_COLOR, 'textDecoration': 'none', 'fontSize': '11px', 'lineHeight': '1.4'},
                           target='_self'),
                ], style={'marginBottom': '2px'}),
                html.Div([
                    html.A(APP_OWNER['linkedin'],
                           href=f'https://{APP_OWNER["linkedin"]}' if not APP_OWNER['linkedin'].startswith('http') else APP_OWNER['linkedin'],
                           style={'color': TEXT_COLOR, 'textDecoration': 'none', 'fontSize': '11px', 'lineHeight': '1.4'},
                           target='_blank'),
                ])
            ], style={
                'textAlign': 'right',
                'fontFamily': 'Arial, sans-serif',
                'color': TEXT_COLOR,
                'lineHeight': '1.4'
            })
        ], style={
            'display': 'flex',
            'justifyContent': 'space-between',
            'alignItems': 'flex-start',
            'maxWidth': '100%'
        })
    ], style={
        'padding': '20px', 
        'backgroundColor': TEAL_BACKGROUND, 
        'borderBottom': '2px solid #000000'
    }),
    
    # Controls
    html.Div([
        html.Div([
            html.Label("Productivity Measure:", style={
                'fontWeight': 'bold', 
                'marginRight': '10px',
                'color': TEXT_COLOR,
                'fontFamily': 'Arial, sans-serif'
            }),
            dcc.Dropdown(
                id='measure-dropdown',
                options=[{'label': m, 'value': m} for m in measures],
                value=measures[0] if len(measures) > 0 else 'Output per hour',
                style={'width': '200px', 'display': 'inline-block'}
            )
        ], style={'display': 'inline-block', 'marginRight': '30px'}),
        
        html.Div([
            html.Label("Index to 2007Q4=100:", style={
                'fontWeight': 'bold', 
                'marginRight': '10px',
                'color': TEXT_COLOR,
                'fontFamily': 'Arial, sans-serif'
            }),
            dcc.Checklist(
                id='index-toggle',
                options=[{'label': 'Yes', 'value': 'index'}],
                value=[],
                style={'display': 'inline-block'}
            )
        ], style={'display': 'inline-block', 'marginRight': '30px'}),
        
        html.Div([
            html.Label("Rolling 4Q Avg:", style={
                'fontWeight': 'bold', 
                'marginRight': '10px',
                'color': TEXT_COLOR,
                'fontFamily': 'Arial, sans-serif'
            }),
            dcc.Checklist(
                id='rolling-toggle',
                options=[{'label': 'Yes', 'value': 'rolling'}],
                value=[],
                style={'display': 'inline-block'}
            )
        ], style={'display': 'inline-block'})
    ], style={
        'padding': '20px', 
        'backgroundColor': TEAL_BACKGROUND, 
        'borderBottom': '1px solid #000000'
    }),
    
    # Visual sanity check for selected measure
    html.Div(id='measure-info', style={
        'padding': '10px 20px',
        'backgroundColor': TEAL_BACKGROUND,
        'borderBottom': '1px solid #000000',
        'fontSize': '12px',
        'color': TEXT_COLOR,
        'fontFamily': 'Arial, sans-serif'
    }),
    
    # Debug info for toggles (Rolling 4Q Avg and Index status)
    html.Div(id='toggle-debug-info', style={
        'padding': '8px 20px',
        'backgroundColor': TEAL_BACKGROUND,
        'borderBottom': '1px solid #000000',
        'fontSize': '11px',
        'color': TEXT_COLOR,
        'fontFamily': 'Arial, sans-serif',
        'fontStyle': 'italic'
    }),
    
    # Tabs
    dcc.Tabs(id='main-tabs', value='overview', children=[
        dcc.Tab(label='Overview', value='overview'),
        dcc.Tab(label='Output Context', value='output'),
        dcc.Tab(label='Industries', value='industries'),
        dcc.Tab(label='Living Standards', value='income'),
        dcc.Tab(label='Data Notes', value='notes')
    ]),
    
    # Tab content
    html.Div(id='tab-content', style={
        'padding': '20px',
        'backgroundColor': TEAL_BACKGROUND,
        'color': TEXT_COLOR,
        'fontFamily': 'Arial, sans-serif',
        'maxWidth': '100%',
        'boxSizing': 'border-box',
        'overflowX': 'hidden'
    })
], style={
    'backgroundColor': TEAL_BACKGROUND,
    'minHeight': '100vh',
    'fontFamily': 'Arial, sans-serif',
    'color': TEXT_COLOR
})

# Measure info callback (visual sanity check)
@app.callback(
    Output('measure-info', 'children'),
    Input('measure-dropdown', 'value')
)
def update_measure_info(measure):
    """Update the visual sanity check showing selected measure details."""
    if not measure or len(productivity_df) == 0:
        return html.Span("No measure selected or data not available.", style={'fontStyle': 'italic'})
    
    filtered_prod = productivity_df[productivity_df['measure'] == measure]
    if len(filtered_prod) == 0:
        # Debug: print available measures
        available = productivity_df['measure'].unique().tolist()
        print(f"[DEBUG] Selected measure '{measure}' not found in data.")
        print(f"[DEBUG] Available measures: {available}")
        print(f"[DEBUG] DataFrame measure column values: {productivity_df['measure'].unique()}")
        return html.Span(f"Selected measure: {measure} (no data available)", style={'fontStyle': 'italic', 'color': '#d62728'})
    
    # Format date range
    min_date = filtered_prod['date'].min()
    max_date = filtered_prod['date'].max()
    
    def format_quarter(date):
        """Format date as YYYYQ#"""
        if pd.isna(date):
            return "N/A"
        date_ts = pd.Timestamp(date)
        quarter = (date_ts.month - 1) // 3 + 1
        return f"{date_ts.year}Q{quarter}"
    
    min_q = format_quarter(min_date)
    max_q = format_quarter(max_date)
    n_obs = len(filtered_prod)
    
    return html.Span([
        html.Strong("Selected measure: "),
        f"{measure} ",
        f"(n={n_obs} observations, from {min_q} to {max_q})"
    ])


# Toggle debug info callback (shows Rolling 4Q Avg and Index status with values)
@app.callback(
    Output('toggle-debug-info', 'children'),
    [Input('measure-dropdown', 'value'),
     Input('index-toggle', 'value'),
     Input('rolling-toggle', 'value')]
)
def update_toggle_debug(measure, index_toggle, rolling_toggle):
    """
    Display debug info showing the effect of Rolling 4Q Avg and Index toggles.
    This helps verify that the toggles actually change the data.
    """
    index_enabled = 'index' in index_toggle
    rolling_enabled = 'rolling' in rolling_toggle
    
    if not measure or len(productivity_df) == 0:
        return html.Span("No data to display toggle effects.", style={'fontStyle': 'italic'})
    
    # Get the transformed series using utility functions
    transformed_df, metadata = prepare_productivity_series(
        productivity_df, measure,
        rolling_enabled=rolling_enabled,
        index_enabled=index_enabled
    )
    
    rolling_status = "ON" if rolling_enabled else "OFF"
    index_status = "ON" if index_enabled else "OFF"
    
    first_val = f"{metadata['first_value']:.2f}" if metadata['first_value'] is not None else "N/A"
    last_val = f"{metadata['last_value']:.2f}" if metadata['last_value'] is not None else "N/A"
    
    return html.Span([
        html.Strong("Rolling 4Q Avg: "), f"{rolling_status} | ",
        html.Strong("Index to 2007Q4=100: "), f"{index_status} | ",
        html.Strong("First value: "), f"{first_val} | ",
        html.Strong("Last value: "), f"{last_val} | ",
        f"(n={metadata['final_rows']} after transformations)"
    ])

# Tab content callback
@app.callback(
    Output('tab-content', 'children'),
    [Input('main-tabs', 'value'),
     Input('measure-dropdown', 'value'),
     Input('index-toggle', 'value'),
     Input('rolling-toggle', 'value')]
)
def update_tab_content(tab, measure, index_toggle, rolling_toggle):
    """Update tab content based on selections."""
    # Debugging: print selected measure
    print(f"\n[DEBUG] Callback triggered - Tab: {tab}, Measure: {measure}")
    
    index_enabled = 'index' in index_toggle
    rolling_enabled = 'rolling' in rolling_toggle
    
    # Filter productivity data by selected measure for debugging
    if len(productivity_df) > 0 and measure:
        filtered_prod = productivity_df[productivity_df['measure'] == measure]
        if len(filtered_prod) > 0:
            print(f"[DEBUG] Filtered productivity data for '{measure}':")
            print(f"  - Rows: {len(filtered_prod)}")
            print(f"  - Date range: {filtered_prod['date'].min()} to {filtered_prod['date'].max()}")
            print(f"  - First 3 rows:\n{filtered_prod.head(3)}")
        else:
            print(f"[DEBUG] WARNING: No data found for measure '{measure}'")
            print(f"  Available measures: {productivity_df['measure'].unique().tolist()}")
    
    if tab == 'overview':
        content = []
        
        # Error banner if productivity data is missing
        if productivity_error_message or len(productivity_df) == 0:
            error_details = productivity_error_message if productivity_error_message else "No productivity data available"
            # Truncate error message to ~300 chars
            if len(error_details) > 300:
                error_details = error_details[:300] + "..."
            
            content.append(html.Div([
                html.Strong("Error: Productivity dataset failed to load"),
                html.Br(),
                html.Br(),
                html.Span("Error details: ", style={'fontWeight': 'bold'}),
                html.Span(error_details, style={'fontFamily': 'monospace', 'fontSize': '12px'})
            ], style={
                'backgroundColor': '#ffebee',
                'color': '#c62828',
                'padding': '15px',
                'borderRadius': '4px',
                'border': '1px solid #c62828',
                'marginBottom': '20px',
                'fontFamily': 'Arial, sans-serif'
            }))
        
        # A) 2-column row: Intro (left) + KPI cards (right)
        intro_kpi_row = []
        
        # Left: Intro + "What this dashboard does"
        intro_content = html.Div([
            html.H2("The UK Productivity Puzzle", style={
                'color': TEXT_COLOR,
                'fontFamily': 'Arial, sans-serif',
                'marginBottom': '8px',
                'fontSize': '22px',
                'fontWeight': 'bold'
            }),
            html.P("UK productivity growth slowed sharply after 2008 — this dashboard shows where the slowdown came from and why it matters.", style={
                'color': TEXT_COLOR,
                'fontFamily': 'Arial, sans-serif',
                'fontSize': '14px',
                'fontStyle': 'italic',
                'marginBottom': '20px',
                'color': '#666666'
            }),
            html.Div([
                html.P([
                    html.Strong("What productivity means", style={'fontSize': '15px'})
                ], style={'marginBottom': '8px', 'color': TEXT_COLOR, 'fontFamily': 'Arial, sans-serif'}),
                html.Ul([
                    html.Li(f"Productivity is output produced per hour worked (or per worker/job, depending on selection).", style={'fontSize': '13px', 'marginBottom': '4px'}),
                    html.Li("Over time, productivity is the main driver of rising wages and living standards.", style={'fontSize': '13px', 'marginBottom': '4px'}),
                    html.Li("When productivity stalls, GDP can still grow, but income growth tends to slow.", style={'fontSize': '13px'})
                ], style={'paddingLeft': '20px', 'marginBottom': '16px', 'color': TEXT_COLOR, 'fontFamily': 'Arial, sans-serif'}),
                html.P([
                    html.Strong("What changed after 2008", style={'fontSize': '15px'})
                ], style={'marginBottom': '8px', 'color': TEXT_COLOR, 'fontFamily': 'Arial, sans-serif'}),
                html.Ul([
                    html.Li("The pre-2008 trend was steady; post-2008 growth became much weaker and flatter.", style={'fontSize': '13px', 'marginBottom': '4px'}),
                    html.Li("The 'productivity gap' measures how far the economy sits below its earlier trajectory.", style={'fontSize': '13px', 'marginBottom': '4px'}),
                    html.Li("The slowdown is visible across multiple industries, not just one sector.", style={'fontSize': '13px'})
                ], style={'paddingLeft': '20px', 'marginBottom': '16px', 'color': TEXT_COLOR, 'fontFamily': 'Arial, sans-serif'}),
                html.P([
                    html.Strong("How to use this dashboard", style={'fontSize': '15px'})
                ], style={'marginBottom': '8px', 'color': TEXT_COLOR, 'fontFamily': 'Arial, sans-serif'}),
                html.Ul([
                    html.Li("Use the dropdown to switch measure (per hour / per worker / per job).", style={'fontSize': '13px', 'marginBottom': '4px'}),
                    html.Li("Use 'Index to 2007Q4=100' to compare relative performance.", style={'fontSize': '13px', 'marginBottom': '4px'}),
                    html.Li("Use the Industries tab to see which sectors drive the divergence.", style={'fontSize': '13px'})
                ], style={'paddingLeft': '20px', 'marginBottom': '0', 'color': TEXT_COLOR, 'fontFamily': 'Arial, sans-serif'})
            ])
        ], style={
            'backgroundColor': '#ffffff',
            'padding': '24px',
            'borderRadius': '5px',
            'border': '1px solid #e0e0e0',
            'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
            'flex': '1',
            'marginRight': '20px'
        })
        intro_kpi_row.append(intro_content)
        
        # Right: KPI Cards (4 cards in 2x2 grid)
        kpi_cards = []
        
        # ============================================================
        # TRANSFORMATION ORDER (applied via utility functions):
        # 1. raw data -> 2. rolling average (if enabled) -> 
        # 3. indexing (if enabled) -> 4. growth rates -> 5. KPIs/charts
        # ============================================================
        
        # Get TRANSFORMED productivity data using utility functions
        # This applies: raw -> rolling -> indexing in correct order
        measure_data, transform_meta = prepare_productivity_series(
            productivity_df, measure,
            rolling_enabled=rolling_enabled,
            index_enabled=index_enabled
        )
        
        # Get counterfactual data (also needs transformations)
        cf_data = None
        if measure in productivity_counterfactual:
            cf_data = prepare_counterfactual_series(
                productivity_counterfactual[measure].copy(),
                rolling_enabled=rolling_enabled,
                index_enabled=index_enabled
            )
        
        # KPI 1: Latest productivity index/level (from TRANSFORMED data)
        if measure_data is not None and len(measure_data) > 0:
            latest_val = measure_data['value'].iloc[-1]
            label = "Latest Index" if index_enabled else "Latest Value"
            
            kpi_cards.append(html.Div([
                html.H3(f"{latest_val:.1f}", style={
                    'margin': '0',
                    'color': TEXT_COLOR,
                    'fontFamily': 'Arial, sans-serif',
                    'fontSize': '32px',
                    'fontWeight': 'bold'
                }),
                html.P(label, style={
                    'margin': '6px 0 0 0',
                    'color': TEXT_COLOR,
                    'fontFamily': 'Arial, sans-serif',
                    'fontSize': '11px'
                }),
                html.P("based on selected measure", style={
                    'margin': '4px 0 0 0',
                    'color': '#666666',
                    'fontFamily': 'Arial, sans-serif',
                    'fontSize': '9px',
                    'fontStyle': 'italic'
                })
            ], style={**KPI_CARD_STYLE, 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)', 'padding': '16px'}))
        else:
            kpi_cards.append(html.Div([
                html.H3("—", style={'margin': '0', 'color': '#999', 'fontSize': '32px'}),
                html.P("No data", style={'margin': '6px 0 0 0', 'color': '#999', 'fontSize': '11px'})
            ], style={**KPI_CARD_STYLE, 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)', 'padding': '16px'}))
        
        # KPI 2: Post-2008 annualised growth (computed from TRANSFORMED levels)
        # Note: Growth rates are calculated from the transformed (smoothed/indexed) levels
        if measure_data is not None and len(measure_data) > 0:
            post_df = measure_data[measure_data['date'] >= pd.Timestamp('2008-01-01')]
            if len(post_df) > 1:
                # Log growth annualised from transformed levels
                years = (post_df['date'].iloc[-1] - post_df['date'].iloc[0]).days / 365.25
                if years > 0 and post_df['value'].iloc[0] > 0:
                    post_growth = (np.log(post_df['value'].iloc[-1] / post_df['value'].iloc[0]) / years) * 100
                else:
                    post_growth = 0
            else:
                post_growth = 0
            
            kpi_cards.append(html.Div([
                html.H3(f"{post_growth:.2f}%", style={
                    'margin': '0',
                    'color': TEXT_COLOR,
                    'fontFamily': 'Arial, sans-serif',
                    'fontSize': '32px',
                    'fontWeight': 'bold'
                }),
                html.P("Post-2008 Growth", style={
                    'margin': '6px 0 0 0',
                    'color': TEXT_COLOR,
                    'fontFamily': 'Arial, sans-serif',
                    'fontSize': '11px'
                }),
                html.P("annualised", style={
                    'margin': '4px 0 0 0',
                    'color': '#666666',
                    'fontFamily': 'Arial, sans-serif',
                    'fontSize': '9px',
                    'fontStyle': 'italic'
                })
            ], style={**KPI_CARD_STYLE, 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)', 'padding': '16px'}))
        else:
            kpi_cards.append(html.Div([
                html.H3("—", style={'margin': '0', 'color': '#999', 'fontSize': '32px'}),
                html.P("No data", style={'margin': '6px 0 0 0', 'color': '#999', 'fontSize': '11px'})
            ], style={**KPI_CARD_STYLE, 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)', 'padding': '16px'}))
        
        # KPI 3: Pre-2008 annualised growth (computed from TRANSFORMED levels)
        if measure_data is not None and len(measure_data) > 0:
            pre_df = measure_data[measure_data['date'] <= pd.Timestamp('2007-12-31')]
            if len(pre_df) == 0:
                pre_df = measure_data[measure_data['date'] < pd.Timestamp('2008-01-01')]
            if len(pre_df) > 1:
                years = (pre_df['date'].iloc[-1] - pre_df['date'].iloc[0]).days / 365.25
                if years > 0 and pre_df['value'].iloc[0] > 0:
                    pre_growth = (np.log(pre_df['value'].iloc[-1] / pre_df['value'].iloc[0]) / years) * 100
                else:
                    pre_growth = 0
            else:
                pre_growth = 0
            
            kpi_cards.append(html.Div([
                html.H3(f"{pre_growth:.2f}%", style={
                    'margin': '0',
                    'color': TEXT_COLOR,
                    'fontFamily': 'Arial, sans-serif',
                    'fontSize': '32px',
                    'fontWeight': 'bold'
                }),
                html.P("Pre-2008 Growth", style={
                    'margin': '6px 0 0 0',
                    'color': TEXT_COLOR,
                    'fontFamily': 'Arial, sans-serif',
                    'fontSize': '11px'
                }),
                html.P("annualised", style={
                    'margin': '4px 0 0 0',
                    'color': '#666666',
                    'fontFamily': 'Arial, sans-serif',
                    'fontSize': '9px',
                    'fontStyle': 'italic'
                })
            ], style={**KPI_CARD_STYLE, 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)', 'padding': '16px'}))
        else:
            kpi_cards.append(html.Div([
                html.H3("—", style={'margin': '0', 'color': '#999', 'fontSize': '32px'}),
                html.P("No data", style={'margin': '6px 0 0 0', 'color': '#999', 'fontSize': '11px'})
            ], style={**KPI_CARD_STYLE, 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)', 'padding': '16px'}))
        
        # KPI 4: Productivity gap (latest) - from TRANSFORMED counterfactual
        gap_pct = None
        if cf_data is not None and len(cf_data) > 0:
            latest_cf = cf_data.iloc[-1]
            if latest_cf['actual'] != 0 and not pd.isna(latest_cf['actual']):
                gap_pct = ((latest_cf['counterfactual'] / latest_cf['actual']) - 1) * 100
            
        if gap_pct is not None:
            kpi_cards.append(html.Div([
                html.H3(f"{gap_pct:.1f}%", style={
                    'margin': '0',
                    'color': TEXT_COLOR,
                    'fontFamily': 'Arial, sans-serif',
                    'fontSize': '32px',
                    'fontWeight': 'bold'
                }),
                html.P("below pre-2008 trend", style={
                    'margin': '6px 0 0 0',
                    'color': TEXT_COLOR,
                    'fontFamily': 'Arial, sans-serif',
                    'fontSize': '11px'
                }),
                html.P("latest gap", style={
                    'margin': '4px 0 0 0',
                    'color': '#666666',
                    'fontFamily': 'Arial, sans-serif',
                    'fontSize': '9px',
                    'fontStyle': 'italic'
                })
            ], style={**KPI_CARD_STYLE, 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)', 'padding': '16px'}))
        else:
            kpi_cards.append(html.Div([
                html.H3("—", style={'margin': '0', 'color': '#999', 'fontSize': '32px'}),
                html.P("No data", style={'margin': '6px 0 0 0', 'color': '#999', 'fontSize': '11px'})
            ], style={**KPI_CARD_STYLE, 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)', 'padding': '16px'}))
        
        # Store KPI values for explanation text
        kpi1_val = None
        kpi2_val = None
        kpi3_val = None
        kpi4_val = None
        
        if measure_data is not None and len(measure_data) > 0:
            kpi1_val = measure_data['value'].iloc[-1]
            post_df = measure_data[measure_data['date'] >= pd.Timestamp('2008-01-01')]
            if len(post_df) > 1:
                years = (post_df['date'].iloc[-1] - post_df['date'].iloc[0]).days / 365.25
                if years > 0 and post_df['value'].iloc[0] > 0:
                    kpi2_val = (np.log(post_df['value'].iloc[-1] / post_df['value'].iloc[0]) / years) * 100
            pre_df = measure_data[measure_data['date'] <= pd.Timestamp('2007-12-31')]
            if len(pre_df) == 0:
                pre_df = measure_data[measure_data['date'] < pd.Timestamp('2008-01-01')]
            if len(pre_df) > 1:
                years = (pre_df['date'].iloc[-1] - pre_df['date'].iloc[0]).days / 365.25
                if years > 0 and pre_df['value'].iloc[0] > 0:
                    kpi3_val = (np.log(pre_df['value'].iloc[-1] / pre_df['value'].iloc[0]) / years) * 100
        
        if gap_pct is not None:
            kpi4_val = gap_pct
        
        # Determine KPI labels and measure-specific phrasing
        kpi1_label = "Latest Index" if index_enabled else "Latest Value"
        kpi2_label = "Post-2008 Growth"
        kpi3_label = "Pre-2008 Growth"
        kpi4_label = "Below pre-2008 trend"
        
        # Measure-specific phrasing
        if measure == "Output per hour":
            measure_phrase = "efficiency per hour worked"
            measure_unit = "per hour"
        elif measure == "Output per worker":
            measure_phrase = "output per employed person"
            measure_unit = "per worker"
        elif measure == "Output per job":
            measure_phrase = "output per filled job"
            measure_unit = "per job"
        else:
            measure_phrase = "productivity"
            measure_unit = ""
        
        # Generate interpretation text based on KPI values
        kpi1_interpretation = ""
        if kpi1_val is not None:
            if index_enabled:
                if kpi1_val > 110:
                    kpi1_interpretation = f"At {kpi1_val:.1f}, this shows strong improvement since 2007Q4."
                elif kpi1_val > 100:
                    kpi1_interpretation = f"At {kpi1_val:.1f}, this indicates modest improvement since 2007Q4."
                elif kpi1_val > 95:
                    kpi1_interpretation = f"At {kpi1_val:.1f}, this shows productivity is close to 2007Q4 levels."
                else:
                    kpi1_interpretation = f"At {kpi1_val:.1f}, this indicates productivity remains below 2007Q4 levels."
            else:
                kpi1_interpretation = f"At {kpi1_val:.1f}, this represents the current level of {measure_phrase}."
        else:
            kpi1_interpretation = "This shows the current level of " + measure_phrase + "."
        
        kpi2_interpretation = ""
        if kpi2_val is not None:
            if kpi2_val < 0.5:
                kpi2_interpretation = f"At {kpi2_val:.2f}% per year, this is very weak growth—well below historical norms and insufficient to sustain rising living standards."
            elif kpi2_val < 1.0:
                kpi2_interpretation = f"At {kpi2_val:.2f}% per year, this is weak growth that limits wage increases and economic expansion."
            else:
                kpi2_interpretation = f"At {kpi2_val:.2f}% per year, this represents moderate growth, though still below pre-crisis rates."
        else:
            kpi2_interpretation = "This shows the average annual growth rate since 2008, which has been weak compared to pre-crisis levels."
        
        kpi3_interpretation = ""
        if kpi3_val is not None:
            if kpi3_val > 2.0:
                kpi3_interpretation = f"At {kpi3_val:.2f}% per year, this was strong pre-crisis growth that supported rising wages and GDP expansion."
            else:
                kpi3_interpretation = f"At {kpi3_val:.2f}% per year, this was the pre-crisis growth rate that provided a benchmark for post-2008 performance."
        else:
            kpi3_interpretation = "This provides a benchmark for comparing post-2008 performance."
        
        kpi4_interpretation = ""
        if kpi4_val is not None:
            if kpi4_val > 20:
                kpi4_interpretation = f"At {kpi4_val:.1f}% below trend, this is a large gap indicating the economy is producing significantly less {measure_phrase} than it could have if pre-2008 trends had continued."
            elif kpi4_val > 10:
                kpi4_interpretation = f"At {kpi4_val:.1f}% below trend, this gap shows meaningful lost potential in {measure_phrase}."
            else:
                kpi4_interpretation = f"At {kpi4_val:.1f}% below trend, this gap indicates some deviation from pre-2008 growth patterns."
        else:
            kpi4_interpretation = "This measures how far current productivity sits below where it would be if pre-2008 growth trends had continued."
        
        # Add KPI cards to right column (2x2 grid) with explanatory text below
        kpi_container = html.Div([
            html.Div(kpi_cards, style={
                'display': 'grid',
                'gridTemplateColumns': 'repeat(2, 1fr)',
                'gap': '12px',
                'marginBottom': '20px'
            }),
            # White card explanatory text block
            html.Div([
                html.H4("What these numbers mean", style={
                    'color': TEXT_COLOR,
                    'fontFamily': 'Arial, sans-serif',
                    'fontSize': '14px',
                    'fontWeight': 'bold',
                    'marginBottom': '12px',
                    'marginTop': '0'
                }),
                html.Div([
                    html.P([
                        html.Strong(f"{kpi1_label}:", style={'fontSize': '12px', 'color': TEXT_COLOR, 'fontWeight': 'bold'}),
                        f" Shows the current {measure_phrase}. ",
                        kpi1_interpretation
                    ], style={'marginBottom': '10px', 'fontSize': '11px', 'lineHeight': '1.5', 'color': TEXT_COLOR}),
                    html.P([
                        html.Strong(f"{kpi2_label}:", style={'fontSize': '12px', 'color': TEXT_COLOR, 'fontWeight': 'bold'}),
                        f" The average annual rate at which {measure_phrase} has grown since 2008. ",
                        kpi2_interpretation
                    ], style={'marginBottom': '10px', 'fontSize': '11px', 'lineHeight': '1.5', 'color': TEXT_COLOR}),
                    html.P([
                        html.Strong(f"{kpi3_label}:", style={'fontSize': '12px', 'color': TEXT_COLOR, 'fontWeight': 'bold'}),
                        f" The average annual {measure_phrase} growth rate before the financial crisis. ",
                        kpi3_interpretation
                    ], style={'marginBottom': '10px', 'fontSize': '11px', 'lineHeight': '1.5', 'color': TEXT_COLOR}),
                    html.P([
                        html.Strong(f"{kpi4_label}:", style={'fontSize': '12px', 'color': TEXT_COLOR, 'fontWeight': 'bold'}),
                        f" Measures how far current {measure_phrase} sits below where it would be if pre-2008 growth trends had continued. ",
                        kpi4_interpretation
                    ], style={'marginBottom': '12px', 'fontSize': '11px', 'lineHeight': '1.5', 'color': TEXT_COLOR}),
                    html.P([
                        "Weak productivity growth limits the economy's ability to raise wages, expand GDP sustainably, and improve living standards over time. ",
                        "When productivity stagnates, income gains become harder to achieve and economic growth becomes more dependent on working longer hours or employing more people, rather than genuine efficiency improvements."
                    ], style={'marginBottom': '8px', 'fontSize': '11px', 'lineHeight': '1.5', 'color': TEXT_COLOR}),
                    html.P([
                        html.I("All figures are based on the selected productivity measure.", style={'fontSize': '10px', 'color': '#666666', 'fontStyle': 'italic'})
                    ], style={'marginBottom': '0', 'fontSize': '10px', 'color': '#666666'})
                ])
            ], style={
                **KPI_CARD_STYLE,
                'padding': '16px',
                'marginTop': '0'
            })
        ], style={
            'flex': '1',
            'maxWidth': '400px'
        })
        intro_kpi_row.append(kpi_container)
        
        # Add the 2-column row to content
        content.append(html.Div(intro_kpi_row, style={
            'display': 'flex',
            'maxWidth': '1200px',
            'margin': '0 auto 30px auto',
            'gap': '20px'
        }))
        
        # A.5) Understanding productivity measures (below KPI cards)
        productivity_measures_explanation = []
        
        # Helper function to style measure name based on selection
        def style_measure_name(measure_name, is_selected):
            if is_selected:
                return html.Strong(measure_name, style={
                    'fontWeight': 'bold',
                    'color': '#1f77b4',  # Blue accent color
                    'fontSize': '14px'
                })
            else:
                return html.Strong(measure_name, style={'fontSize': '14px', 'color': TEXT_COLOR})
        
        # Build the explanation content
        productivity_measures_explanation.append(
            html.Div([
                html.H3("Understanding productivity measures", style={
                    'color': TEXT_COLOR,
                    'fontFamily': 'Arial, sans-serif',
                    'fontSize': '18px',
                    'fontWeight': 'bold',
                    'marginBottom': '16px'
                }),
                
                html.Div([
                    html.P([
                        style_measure_name("Output per hour worked", measure == "Output per hour"),
                        html.Br(),
                        "This is the preferred measure of productivity. It shows how much value the economy produces for each hour of labour supplied. It captures changes in efficiency, technology, skills, and capital intensity, and is less affected by shifts between full-time and part-time work."
                    ], style={'marginBottom': '14px', 'fontSize': '13px', 'lineHeight': '1.6', 'color': TEXT_COLOR}),
                    
                    html.P([
                        style_measure_name("Output per worker", measure == "Output per worker"),
                        html.Br(),
                        "This measures output relative to the number of people employed. It is influenced not only by efficiency, but also by changes in hours worked per employee. For example, if average hours fall while efficiency is unchanged, output per worker may decline even if output per hour does not."
                    ], style={'marginBottom': '14px', 'fontSize': '13px', 'lineHeight': '1.6', 'color': TEXT_COLOR}),
                    
                    html.P([
                        style_measure_name("Output per job", measure == "Output per job"),
                        html.Br(),
                        "This measures output per filled job, regardless of whether jobs are full-time or part-time. It is useful for analysing labour market structure, but can be distorted when employment growth is concentrated in lower-hours or lower-productivity roles."
                    ], style={'marginBottom': '16px', 'fontSize': '13px', 'lineHeight': '1.6', 'color': TEXT_COLOR}),
                    
                    html.P([
                        html.Strong("Why the distinction matters", style={'fontSize': '14px', 'color': TEXT_COLOR, 'fontWeight': 'bold'}),
                        html.Br(),
                        "After 2008, UK employment recovered relatively quickly, but productivity growth did not. Comparing these measures helps distinguish between: growth driven by more people working, growth driven by longer hours, and growth driven by genuine efficiency improvements. The persistent weakness in ",
                        html.Span("output per hour", style={
                            'fontWeight': 'bold' if measure == "Output per hour" else 'normal',
                            'color': '#1f77b4' if measure == "Output per hour" else TEXT_COLOR,
                            'fontSize': '13px'
                        }) if measure else html.Span("output per hour", style={'fontSize': '13px', 'color': TEXT_COLOR}),
                        " indicates that the post-crisis slowdown reflects a deeper efficiency problem, not just labour market composition."
                    ], style={'fontSize': '13px', 'lineHeight': '1.6', 'color': TEXT_COLOR, 'marginTop': '8px'})
                ])
            ], style={
                'backgroundColor': '#ffffff',
                'padding': '24px',
                'borderRadius': '5px',
                'border': '1px solid #e0e0e0',
                'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
                'maxWidth': '1200px',
                'margin': '0 auto 30px auto'
            })
        )
        
        content.append(productivity_measures_explanation[0])
        
        # B) Hero chart: Productivity gap
        if measure and cf_data is not None and len(cf_data) > 0:
            cf_data_hero = cf_data.copy()
            if index_enabled:
                base_row = cf_data_hero[cf_data_hero['date'] <= pd.Timestamp('2007-12-31')]
                if len(base_row) > 0:
                    base_value = base_row['actual'].iloc[-1]
                    cf_data_hero['actual'] = (cf_data_hero['actual'] / base_value) * 100
                    cf_data_hero['counterfactual'] = (cf_data_hero['counterfactual'] / base_value) * 100
                    cf_data_hero['gap'] = cf_data_hero['counterfactual'] - cf_data_hero['actual']
            
            fig_hero = chart_productivity_gap(cf_data_hero, measure)
            content.append(html.Div([
                dcc.Graph(figure=fig_hero)
            ], style={'maxWidth': '1200px', 'margin': '0 auto 30px auto'}))
        
        # C) Two charts stacked vertically (to allow full titles)
        # Chart 1: Pre vs Post growth
        if len(productivity_df) > 0 and measure:
            fig_growth = chart_growth_comparison(productivity_df, measure)
            content.append(html.Div([
                dcc.Graph(figure=fig_growth)
            ], style={'maxWidth': '1200px', 'margin': '0 auto 30px auto'}))
        
        # Chart 2: GDP vs Productivity
        if len(gdp_df) > 0 and len(productivity_df) > 0 and measure:
            fig_gdp_prod = chart_gdp_vs_productivity(gdp_df, productivity_df, measure, rolling_enabled)
            content.append(html.Div([
                dcc.Graph(figure=fig_gdp_prod)
            ], style={'maxWidth': '1200px', 'margin': '0 auto 30px auto'}))
        
        # D) Key takeaways box
        takeaways_bullets = []
        
        # Get values for dynamic bullets
        pre_growth_val = None
        post_growth_val = None
        if measure_data is not None and len(measure_data) > 0:
            post_df_temp = measure_data[measure_data['date'] >= pd.Timestamp('2008-01-01')]
            pre_df_temp = measure_data[measure_data['date'] <= pd.Timestamp('2007-12-31')]
            if len(pre_df_temp) == 0:
                pre_df_temp = measure_data[measure_data['date'] < pd.Timestamp('2008-01-01')]
            
            if len(post_df_temp) > 1:
                years = (post_df_temp['date'].iloc[-1] - post_df_temp['date'].iloc[0]).days / 365.25
                if years > 0:
                    post_growth_val = (np.log(post_df_temp['value'].iloc[-1] / post_df_temp['value'].iloc[0]) / years) * 100
            
            if len(pre_df_temp) > 1:
                years = (pre_df_temp['date'].iloc[-1] - pre_df_temp['date'].iloc[0]).days / 365.25
                if years > 0:
                    pre_growth_val = (np.log(pre_df_temp['value'].iloc[-1] / pre_df_temp['value'].iloc[0]) / years) * 100
        
        # Bullet 1: Pre vs post growth difference
        if pre_growth_val is not None and post_growth_val is not None:
            growth_diff = pre_growth_val - post_growth_val
            takeaways_bullets.append(html.Li(
                f"Productivity growth is materially lower after 2008 than before (see KPI cards: {pre_growth_val:.2f}% pre-2008 vs {post_growth_val:.2f}% post-2008).",
                style={'color': TEXT_COLOR, 'fontFamily': 'Arial, sans-serif', 'marginBottom': '8px', 'fontSize': '14px'}
            ))
        else:
            takeaways_bullets.append(html.Li(
                "Productivity growth is materially lower after 2008 than before (see KPI cards).",
                style={'color': TEXT_COLOR, 'fontFamily': 'Arial, sans-serif', 'marginBottom': '8px', 'fontSize': '14px'}
            ))
        
        # Bullet 2: Current gap size (dynamic)
        if gap_pct is not None:
            takeaways_bullets.append(html.Li(
                f"The current productivity level remains below the pre-crisis trend by {gap_pct:.1f}% (latest gap).",
                style={'color': TEXT_COLOR, 'fontFamily': 'Arial, sans-serif', 'marginBottom': '8px', 'fontSize': '14px'}
            ))
        else:
            takeaways_bullets.append(html.Li(
                "The current productivity level remains below the pre-crisis trend (see productivity gap chart).",
                style={'color': TEXT_COLOR, 'fontFamily': 'Arial, sans-serif', 'marginBottom': '8px', 'fontSize': '14px'}
            ))
        
        # Bullet 3: Sector dispersion
        if len(sector_productivity_df) > 0:
            sectors_count = len(sector_productivity_df['sector'].unique())
            takeaways_bullets.append(html.Li(
                f"Sector performance is uneven: analysis across {sectors_count} industries shows some sectors recover, but broad-based gains are limited.",
                style={'color': TEXT_COLOR, 'fontFamily': 'Arial, sans-serif', 'marginBottom': '8px', 'fontSize': '14px'}
            ))
        else:
            takeaways_bullets.append(html.Li(
                "Sector performance is uneven: some industries recover, but broad-based gains are limited.",
                style={'color': TEXT_COLOR, 'fontFamily': 'Arial, sans-serif', 'marginBottom': '8px', 'fontSize': '14px'}
            ))
        
        # Bullet 4: Living standards link
        takeaways_bullets.append(html.Li(
            "Weak productivity growth helps explain slower improvements in household incomes over time.",
            style={'color': TEXT_COLOR, 'fontFamily': 'Arial, sans-serif', 'marginBottom': '8px', 'fontSize': '14px'}
        ))
        
        # Bullet 5: Output Context reference
        takeaways_bullets.append(html.Li(
            "Use Output Context to compare GDP recovery vs productivity recovery.",
            style={'color': TEXT_COLOR, 'fontFamily': 'Arial, sans-serif', 'fontSize': '14px'}
        ))
        
        content.append(html.Div([
            html.H4("Key Takeaways", style={
                'color': TEXT_COLOR,
                'fontFamily': 'Arial, sans-serif',
                'marginBottom': '15px',
                'fontSize': '18px',
                'fontWeight': 'bold'
            }),
            html.Ul(takeaways_bullets, style={
                'paddingLeft': '20px',
                'margin': '0'
            })
        ], style={
            'backgroundColor': '#ffffff',
            'padding': '20px',
            'borderRadius': '5px',
            'border': '1px solid #e0e0e0',
            'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
            'maxWidth': '1200px',
            'margin': '0 auto'
        }))
        
        return html.Div(content, style={'maxWidth': '1200px', 'margin': '0 auto', 'padding': '0 20px'})
    
    elif tab == 'output':
        tiles = []
        
        # Tile 1: GDP vs Productivity
        if len(gdp_df) > 0 and len(productivity_df) > 0:
            fig_c = chart_gdp_vs_productivity(gdp_df, productivity_df, measure, rolling_enabled)
            tiles.append(dcc.Graph(figure=fig_c, style={'marginBottom': '30px'}))
        
        # Tile 2: Production output with GDP
        if len(production_df) > 0:
            overlay = measure in productivity_df['measure'].values if len(productivity_df) > 0 else False
            fig_d = chart_production_output(production_df, gdp_df, productivity_df, measure, overlay)
            tiles.append(dcc.Graph(figure=fig_d))
            
            # Plain-language explanation for general audience
            tiles.append(
                html.Div(
                    [
                        html.H4(
                            "How to read this chart",
                            style={
                                "color": TEXT_COLOR,
                                "fontFamily": "Arial, sans-serif",
                                "marginTop": "12px",
                                "marginBottom": "10px",
                            },
                        ),
                        html.Ul(
                            [
                                html.Li(
                                    [
                                        html.Strong("Blue line = Production Output: "),
                                        "This shows real output in production industries (like manufacturing, mining, and utilities). It's measured as an index, not in pounds."
                                    ],
                                    style={
                                        "color": TEXT_COLOR,
                                        "fontFamily": "Arial, sans-serif",
                                        "marginBottom": "6px",
                                    },
                                ),
                                html.Li(
                                    [
                                        html.Strong("Green line = GDP: "),
                                        "This shows economy-wide output across all industries. The original data comes as growth rates, which we've converted into a level index for comparison."
                                    ],
                                    style={
                                        "color": TEXT_COLOR,
                                        "fontFamily": "Arial, sans-serif",
                                        "marginBottom": "6px",
                                    },
                                ),
                                html.Li(
                                    [
                                        html.Strong("What an index means: "),
                                        "An index shows relative change over time. For example, an index of 120 means output is about 20% higher than at the starting point. All lines start at the same value at the first common quarter, making them directly comparable."
                                    ],
                                    style={
                                        "color": TEXT_COLOR,
                                        "fontFamily": "Arial, sans-serif",
                                        "marginBottom": "6px",
                                    },
                                ),
                                html.Li(
                                    [
                                        html.Strong("Why the lines can be compared: "),
                                        "GDP starts as quarter-on-quarter growth rates (like 'GDP grew 0.5% this quarter'). We convert these into a level index starting at 100, then rescale GDP and Production to line up at their first common quarter. This preserves each line's shape while allowing direct comparison."
                                    ],
                                    style={
                                        "color": TEXT_COLOR,
                                        "fontFamily": "Arial, sans-serif",
                                        "marginBottom": "6px",
                                    },
                                ),
                                html.Li(
                                    [
                                        html.Strong("How to interpret gaps between lines: "),
                                        "If GDP rises faster than Production, growth is coming from other parts of the economy (like services). If Production falls or stagnates while GDP rises, output growth is becoming less broad-based. Long-lasting gaps signal structural change, not just short-term shocks."
                                    ],
                                    style={
                                        "color": TEXT_COLOR,
                                        "fontFamily": "Arial, sans-serif",
                                        "marginBottom": "6px",
                                    },
                                ),
                                html.Li(
                                    [
                                        html.Strong("Example: "),
                                        "If from 2010 to 2020 the GDP line sits below the Production line and flattens out while Production keeps rising, that tells you GDP has grown more slowly than Production over that decade."
                                    ],
                                    style={
                                        "color": TEXT_COLOR,
                                        "fontFamily": "Arial, sans-serif",
                                        "marginBottom": "6px",
                                    },
                                ),
                                html.Li(
                                    [
                                        html.Strong("Big-picture takeaway: "),
                                        "Before the financial crisis, GDP and production broadly moved together. After the crisis, GDP recovers more strongly while production remains weaker. This divergence helps explain the UK's productivity puzzle."
                                    ],
                                    style={
                                        "color": TEXT_COLOR,
                                        "fontFamily": "Arial, sans-serif",
                                    },
                                ),
                            ],
                            style={
                                "paddingLeft": "18px",
                                "margin": 0,
                            },
                        ),
                    ],
                    style={
                        "backgroundColor": "#f8f9fa",
                        "padding": "14px 16px",
                        "borderRadius": "4px",
                        "border": "1px solid #cccccc",
                        "marginTop": "16px",
                    },
                )
            )
        
        return html.Div(tiles)
    
    elif tab == 'industries':
        # Industries tab: Multiple charts grouped by industry category with explanations
        content = []
        
        if len(sector_productivity_df) > 0:
            # Get category mapping
            category_mapping = get_sector_category_mapping()
            
            # Define category order and their explanations
            category_configs = {
                'Market Services': {
                    'title': "What we're seeing in this chart",
                    'bullets': [
                        'Most market service industries show weak or flat productivity growth after 2008, compared with earlier decades.',
                        'Many lines converge around the 2019 base level, indicating limited long-run efficiency gains.',
                        'Short-term jumps and dips appear, but these do not translate into sustained upward trends.',
                        'The GDP benchmark rises more smoothly than most service-sector productivity lines, suggesting growth has not been driven by widespread service-sector efficiency improvements.'
                    ],
                    'takeaway': 'Market services employ a large share of the workforce, so slow productivity growth here drags down overall UK productivity.'
                },
                'Professional & Financial Services': {
                    'title': "What we're seeing in this chart",
                    'bullets': [
                        'These sectors begin at higher productivity levels than most others.',
                        'Strong gains before 2008 are followed by much flatter or declining trends afterwards.',
                        'Real estate shows a distinct pattern: high early productivity, then sustained weakening.',
                        'GDP continues to grow faster than several of these sectors, indicating that high-value services did not continue to lift aggregate productivity post-crisis.'
                    ],
                    'takeaway': 'Even traditionally high-productivity sectors experienced a post-2008 slowdown, limiting their contribution to growth.'
                },
                'Manufacturing': {
                    'title': "What we're seeing in this chart",
                    'bullets': [
                        'Manufacturing displays large differences across sub-industries, rather than a single common trend.',
                        'Many industries improved into the early 2000s, then flattened or became volatile.',
                        'Sharp swings reflect exposure to global demand, investment cycles, and supply shocks.',
                        'GDP grows more smoothly than manufacturing productivity, showing that aggregate growth can occur despite uneven industrial performance.'
                    ],
                    'takeaway': 'Manufacturing contains pockets of strong productivity, but they are too uneven to drive economy-wide gains on their own.'
                },
                'Production & Utilities': {
                    'title': "What we're seeing in this chart",
                    'bullets': [
                        'With fewer industries, trends are clearer and more stable.',
                        'Productivity rises gradually before levelling off, with visible disruptions around major shocks.',
                        'These industries do not show sustained post-2008 acceleration relative to GDP.',
                        'GDP growth outpaces productivity here, indicating limited contribution to long-run productivity growth.'
                    ],
                    'takeaway': "Production and utilities provide stability, but they do not explain the UK's long-run productivity slowdown."
                }
            }
            
            # Define category order
            category_order = [
                'Market Services',
                'Professional & Financial Services',
                'Manufacturing',
                'Production & Utilities'
            ]
            
            # Add brief instruction at top
            content.append(html.P(
                "Click on legend items to show/hide individual industries. Double-click to isolate a single industry.",
                style={
                    'color': TEXT_COLOR,
                    'fontFamily': 'Arial, sans-serif',
                    'fontSize': '12px',
                    'marginBottom': '30px',
                    'fontStyle': 'italic'
                }
            ))
            
            # Create a chart and explanation for each category
            for category_name in category_order:
                if category_name in category_mapping:
                    category_patterns = category_mapping[category_name]
                    config = category_configs.get(category_name, {})
                    
                    # Section header
                    content.append(html.H3(
                        category_name,
                        style={
                            'color': TEXT_COLOR,
                            'fontFamily': 'Arial, sans-serif',
                            'marginTop': '30px' if len(content) > 1 else '0',
                            'marginBottom': '15px',
                            'fontSize': '20px',
                            'fontWeight': 'bold'
                        }
                    ))
                    
                    # Create chart for this category
                    fig_category = chart_sector_category(
                        sector_productivity_df,
                        gdp_df,
                        category_name,
                        category_patterns
                    )
                    content.append(html.Div([
                        dcc.Graph(
                            figure=fig_category,
                            style={'marginBottom': '20px', 'width': '100%', 'maxWidth': '100%'}
                        )
                    ], style={'width': '100%', 'maxWidth': '100%', 'overflowX': 'auto'}))
                    
                    # Add explanation block immediately below the chart
                    explanation_bullets = []
                    for bullet in config.get('bullets', []):
                        explanation_bullets.append(html.Li(
                            bullet,
                            style={
                                'color': TEXT_COLOR,
                                'fontFamily': 'Arial, sans-serif',
                                'marginBottom': '6px'
                            }
                        ))
                    
                    content.append(html.Div([
                        html.H4(
                            config.get('title', "What we're seeing in this chart"),
                            style={
                                'color': TEXT_COLOR,
                                'fontFamily': 'Arial, sans-serif',
                                'marginTop': '0',
                                'marginBottom': '12px',
                                'fontSize': '16px',
                                'fontWeight': 'bold'
                            }
                        ),
                        html.Ul(
                            explanation_bullets,
                            style={
                                'paddingLeft': '20px',
                                'margin': '0 0 12px 0'
                            }
                        ),
                        html.P([
                            html.Strong('Takeaway: '),
                            config.get('takeaway', '')
                        ], style={
                            'color': TEXT_COLOR,
                            'fontFamily': 'Arial, sans-serif',
                            'fontWeight': 'bold',
                            'marginBottom': '8px',
                            'marginTop': '8px'
                        }),
                        html.P(
                            'Note: All series are indexed to a common base year to compare relative changes over time, not absolute productivity levels.',
                            style={
                                'color': TEXT_COLOR,
                                'fontFamily': 'Arial, sans-serif',
                                'fontSize': '11px',
                                'fontStyle': 'italic',
                                'marginTop': '12px',
                                'marginBottom': '0',
                                'color': '#666666'
                            }
                        )
                    ], style={
                        'backgroundColor': '#ffffff',
                        'padding': '16px 20px',
                        'borderRadius': '4px',
                        'border': '1px solid #e0e0e0',
                        'boxShadow': '0 1px 3px rgba(0,0,0,0.1)',
                        'marginBottom': '40px',
                        'width': '100%',
                        'maxWidth': '100%',
                        'boxSizing': 'border-box',
                        'wordWrap': 'break-word',
                        'overflowWrap': 'break-word'
                    }))
        else:
            content.append(html.Div(
                "Sector productivity data not available. Please ensure lprod01.xls is in the data folder.",
                style={
                    'padding': '20px',
                    'color': TEXT_COLOR,
                    'fontFamily': 'Arial, sans-serif'
                }
            ))
        
        return html.Div(content, style={
            'width': '100%',
            'maxWidth': '100%',
            'boxSizing': 'border-box',
            'overflowX': 'hidden'
        })
    
    elif tab == 'income':
        if len(income_df) > 0:
            fig_e = chart_income_by_nation(income_df, index_enabled)
            return html.Div([
                dcc.Graph(figure=fig_e, style={'marginBottom': '30px'}),
                html.Div([
                    html.H4(
                        "How productivity links to income growth",
                        style={
                            'color': TEXT_COLOR,
                            'fontFamily': 'Arial, sans-serif',
                            'marginTop': '20px',
                            'marginBottom': '12px'
                        }
                    ),
                    html.Ul([
                        html.Li([
                            html.Strong("What this chart shows: "),
                            "Median equivalised disposable household income across UK nations and regions. Values are adjusted for household size, so they reflect living standards rather than raw earnings. Lines show long-run trends, not short-term volatility."
                        ], style={'color': TEXT_COLOR, 'fontFamily': 'Arial, sans-serif', 'marginBottom': '8px'}),
                        html.Li([
                            html.Strong("Why productivity matters: "),
                            "Productivity measures how much output the economy produces per hour worked. Higher productivity allows firms to pay higher wages, invest more, and lower costs without cutting pay. Over time, these gains feed into higher household incomes."
                        ], style={'color': TEXT_COLOR, 'fontFamily': 'Arial, sans-serif', 'marginBottom': '8px'}),
                        html.Li([
                            html.Strong("Link from productivity → GDP → incomes: "),
                            "Productivity growth raises total output (GDP). Sustained GDP growth expands the overall economic 'pie'. Higher GDP makes it easier for wages to rise, taxes to fund public services, and households' disposable incomes to grow. When productivity stalls, GDP can still grow — but income growth slows and becomes uneven."
                        ], style={'color': TEXT_COLOR, 'fontFamily': 'Arial, sans-serif', 'marginBottom': '8px'}),
                        html.Li([
                            html.Strong("Interpreting the post-2008 period: "),
                            "After the financial crisis, UK productivity growth weakened sharply. This helps explain why income growth slowed for many years and regional income gaps persisted. Recent increases reflect inflation, labour shortages, and partial recovery — not a full return to pre-2008 productivity trends."
                        ], style={'color': TEXT_COLOR, 'fontFamily': 'Arial, sans-serif', 'marginBottom': '8px'}),
                        html.Li([
                            html.Strong("Key takeaway: "),
                            "Long-run improvements in living standards depend on sustained productivity growth, not short-term economic rebounds."
                        ], style={'color': TEXT_COLOR, 'fontFamily': 'Arial, sans-serif', 'fontWeight': 'bold', 'marginBottom': '8px'}),
                    ], style={'paddingLeft': '18px', 'margin': 0}),
                ], style={
                    'backgroundColor': '#f8f9fa',
                    'padding': '14px 16px',
                    'borderRadius': '4px',
                    'border': '1px solid #cccccc',
                    'marginTop': '16px'
                })
            ])
        else:
            return html.Div("Income data not available", style={
                'padding': '20px',
                'color': TEXT_COLOR,
                'fontFamily': 'Arial, sans-serif'
            })
    
    elif tab == 'notes':
        return html.Div([
            html.H3("Data Sources and Notes", style={
                'color': TEXT_COLOR,
                'fontFamily': 'Arial, sans-serif'
            }),
            html.Hr(style={'borderColor': '#000000'}),
            html.H4("GDP Data", style={
                'color': TEXT_COLOR,
                'fontFamily': 'Arial, sans-serif'
            }),
            html.P("GDP q/q growth % - Quarterly growth rate of Gross Domestic Product.", style={
                'color': TEXT_COLOR,
                'fontFamily': 'Arial, sans-serif'
            }),
            html.H4("Production Data", style={
                'color': TEXT_COLOR,
                'fontFamily': 'Arial, sans-serif'
            }),
            html.P("Total Production output index CVM (Chained Volume Measure) - Index of total production output.", style={
                'color': TEXT_COLOR,
                'fontFamily': 'Arial, sans-serif'
            }),
            html.H4("Productivity Data", style={
                'color': TEXT_COLOR,
                'fontFamily': 'Arial, sans-serif'
            }),
            html.P("ONS labour productivity tables - Output per hour, per worker, and per job measures.", style={
                'color': TEXT_COLOR,
                'fontFamily': 'Arial, sans-serif'
            }),
            html.H4("Income Data", style={
                'color': TEXT_COLOR,
                'fontFamily': 'Arial, sans-serif'
            }),
            html.P("Median household income (equivalised disposable income) by nation - Annual data from housing affordability tables.", style={
                'color': TEXT_COLOR,
                'fontFamily': 'Arial, sans-serif'
            }),
            html.Hr(style={'borderColor': '#000000'}),
            html.P("Note: All quarterly data is aligned to quarter-end dates. The productivity gap is calculated as the difference between the pre-2008 trend projection and actual productivity.", 
                   style={
                       'fontStyle': 'italic', 
                       'color': TEXT_COLOR,
                       'fontFamily': 'Arial, sans-serif'
                   })
        ], style={
            'maxWidth': '800px',
            'color': TEXT_COLOR,
            'fontFamily': 'Arial, sans-serif'
        })
    
    return html.Div("Tab content not implemented")

if __name__ == '__main__':
    print("\n" + "="*60)
    print("UK Productivity Puzzle Dashboard")
    print("="*60)
    print(f"\nStarting server...")
    print(f"Open http://127.0.0.1:8050 in your browser\n")
    app.run(debug=True, port=8050)

