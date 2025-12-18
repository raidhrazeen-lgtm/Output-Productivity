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
)
from styles import KPI_CARD_STYLE, TEAL_BACKGROUND, TEXT_COLOR

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

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

try:
    productivity_df, sector_df = load_productivity()
    if len(productivity_df) > 0:
        measures = productivity_df['measure'].unique()
        date_range = (productivity_df['date'].min(), productivity_df['date'].max())
        print(f"✓ Productivity data loaded: {len(productivity_df)} records")
        print(f"  Measures: {', '.join(measures)}")
        print(f"  Date range: {date_range[0]} to {date_range[1]}")
    else:
        print("✗ No productivity data loaded")
        measures = ['Output per hour']
    
    if sector_df is not None and len(sector_df) > 0:
        sectors = sector_df['sector'].unique()
        print(f"✓ Sector productivity loaded: {len(sectors)} sectors")
    else:
        print("⚠ Sector productivity not available")
        sector_df = None
except Exception as e:
    print(f"✗ Error loading productivity data: {e}")
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
        html.H1("UK Productivity Puzzle", style={
            'textAlign': 'center', 
            'marginBottom': '10px',
            'color': TEXT_COLOR,
            'fontFamily': 'Arial, sans-serif'
        }),
        html.P("Why UK productivity stalled after 2008", 
               style={
                   'textAlign': 'center', 
                   'color': TEXT_COLOR, 
                   'fontSize': '18px', 
                   'marginBottom': '30px',
                   'fontFamily': 'Arial, sans-serif'
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
        'fontFamily': 'Arial, sans-serif'
    })
], style={
    'backgroundColor': TEAL_BACKGROUND,
    'minHeight': '100vh',
    'fontFamily': 'Arial, sans-serif',
    'color': TEXT_COLOR
})

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
    index_enabled = 'index' in index_toggle
    rolling_enabled = 'rolling' in rolling_toggle
    
    if tab == 'overview':
        # KPI Cards
        kpi_cards = []
        
        if measure in productivity_counterfactual and len(productivity_counterfactual[measure]) > 0:
            cf_df = productivity_counterfactual[measure]
            latest = cf_df.iloc[-1]
            
            # Latest productivity index
            if index_enabled:
                base_value = cf_df[cf_df['date'] <= pd.Timestamp('2007-12-31')]['actual']
                if len(base_value) > 0:
                    latest_index = (latest['actual'] / base_value.iloc[-1]) * 100
                else:
                    latest_index = latest['actual']
            else:
                latest_index = latest['actual']
            
            kpi_cards.append(html.Div([
                html.H3(f"{latest_index:.1f}", style={
                    'margin': '0', 
                    'color': TEXT_COLOR,
                    'fontFamily': 'Arial, sans-serif'
                }),
                html.P(f"Latest {measure} Index", style={
                    'margin': '5px 0 0 0', 
                    'color': TEXT_COLOR,
                    'fontFamily': 'Arial, sans-serif'
                })
            ], style=KPI_CARD_STYLE))
            
            # Productivity gap
            gap_pct = (latest['gap'] / latest['counterfactual']) * 100 if latest['counterfactual'] != 0 else 0
            kpi_cards.append(html.Div([
                html.H3(f"{gap_pct:.1f}%", style={
                    'margin': '0', 
                    'color': TEXT_COLOR,
                    'fontFamily': 'Arial, sans-serif'
                }),
                html.P("Productivity Gap", style={
                    'margin': '5px 0 0 0', 
                    'color': TEXT_COLOR,
                    'fontFamily': 'Arial, sans-serif'
                })
            ], style=KPI_CARD_STYLE))
            
            # Post-2008 growth rate
            post_df = cf_df[cf_df['date'] >= pd.Timestamp('2008-01-01')]
            if len(post_df) > 1:
                post_growth = ((post_df['actual'].iloc[-1] / post_df['actual'].iloc[0]) ** (4 / len(post_df)) - 1) * 100
            else:
                post_growth = 0
            
            kpi_cards.append(html.Div([
                html.H3(f"{post_growth:.2f}%", style={
                    'margin': '0', 
                    'color': TEXT_COLOR,
                    'fontFamily': 'Arial, sans-serif'
                }),
                html.P("Post-2008 Annual Growth", style={
                    'margin': '5px 0 0 0', 
                    'color': TEXT_COLOR,
                    'fontFamily': 'Arial, sans-serif'
                })
            ], style=KPI_CARD_STYLE))
        else:
            kpi_cards = [html.Div("KPI data not available", style={
                'padding': '20px',
                'color': TEXT_COLOR,
                'fontFamily': 'Arial, sans-serif'
            })]
        
        # Charts
        charts = []
        
        # Chart A: Productivity gap
        if measure in productivity_counterfactual:
            cf_data = productivity_counterfactual[measure].copy()
            if index_enabled:
                # Index the data
                base_row = cf_data[cf_data['date'] <= pd.Timestamp('2007-12-31')]
                if len(base_row) > 0:
                    base_value = base_row['actual'].iloc[-1]
                    cf_data['actual'] = (cf_data['actual'] / base_value) * 100
                    cf_data['counterfactual'] = (cf_data['counterfactual'] / base_value) * 100
                    cf_data['gap'] = cf_data['counterfactual'] - cf_data['actual']
            
            fig_a = chart_productivity_gap(cf_data, measure)
            charts.append(dcc.Graph(figure=fig_a, style={'marginBottom': '30px'}))
        
        # Chart B: Growth comparison
        if len(productivity_df) > 0:
            fig_b = chart_growth_comparison(productivity_df)
            charts.append(dcc.Graph(figure=fig_b, style={'marginBottom': '30px'}))
        
        # Sector heatmap (bonus)
        if sector_df is not None and len(sector_df) > 0:
            fig_sector = chart_sector_heatmap(sector_df)
            charts.append(dcc.Graph(figure=fig_sector))
        
        return html.Div([
            html.Div(kpi_cards, style={'display': 'grid', 'gridTemplateColumns': 'repeat(3, 1fr)', 'gap': '20px', 'marginBottom': '30px'}),
            *charts
        ])
    
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
            
            # Short, clear explanation of GDP indexing and scaling
            tiles.append(
                html.Div(
                    [
                        html.H4(
                            "How to read this chart",
                            style={
                                "color": TEXT_COLOR,
                                "fontFamily": "Arial, sans-serif",
                                "marginTop": "12px",
                                "marginBottom": "6px",
                            },
                        ),
                        html.Ul(
                            [
                                html.Li(
                                    "Each line is an index: Production, GDP and (optionally) Productivity are all shown on the same vertical scale.",
                                    style={
                                        "color": TEXT_COLOR,
                                        "fontFamily": "Arial, sans-serif",
                                        "marginBottom": "4px",
                                    },
                                ),
                                html.Li(
                                    "GDP starts as quarter‑on‑quarter growth rates (percent). We convert these into a level index by starting at 100 and, for each quarter, multiplying the previous index by (1 + growth/100).",
                                    style={
                                        "color": TEXT_COLOR,
                                        "fontFamily": "Arial, sans-serif",
                                        "marginBottom": "4px",
                                    },
                                ),
                                html.Li(
                                    "We then rescale the GDP index so that, at the first quarter where both GDP and Production are observed, they take the same value. This keeps the units comparable while preserving the shape of the GDP path.",
                                    style={
                                        "color": TEXT_COLOR,
                                        "fontFamily": "Arial, sans-serif",
                                        "marginBottom": "4px",
                                    },
                                ),
                                html.Li(
                                    "The Productivity line is treated in the same way: it is scaled so that it lines up with Production at the first overlapping quarter.",
                                    style={
                                        "color": TEXT_COLOR,
                                        "fontFamily": "Arial, sans-serif",
                                        "marginBottom": "4px",
                                    },
                                ),
                                html.Li(
                                    "We only plot quarters where all relevant series have data, so you are always comparing like‑for‑like periods.",
                                    style={
                                        "color": TEXT_COLOR,
                                        "fontFamily": "Arial, sans-serif",
                                        "marginBottom": "4px",
                                    },
                                ),
                                html.Li(
                                    "Example: if, from 2010 to 2020, the GDP line sits below the Production line and flattens out while Production keeps rising, that tells you GDP has grown more slowly than Production over that decade.",
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
        # Industries tab: Sector productivity with GDP, interactive sector selection
        content = []
        
        if len(sector_productivity_df) > 0:
            # Sector selector dropdown (multi-select)
            content.append(html.Div([
                html.Label("Select Industries to Display:", style={
                    'fontWeight': 'bold',
                    'color': TEXT_COLOR,
                    'fontFamily': 'Arial, sans-serif',
                    'marginBottom': '10px',
                    'display': 'block'
                }),
                html.P("Click on legend items to show/hide individual sectors. Double-click to isolate a single sector.", style={
                    'color': TEXT_COLOR,
                    'fontFamily': 'Arial, sans-serif',
                    'fontSize': '12px',
                    'marginBottom': '15px'
                }),
            ], style={'marginBottom': '20px'}))
            
            # Create the sector + GDP chart
            fig_sectors = chart_sector_productivity_with_gdp(sector_productivity_df, gdp_df)
            content.append(dcc.Graph(figure=fig_sectors, style={'marginBottom': '30px'}))
            
            # Detailed explanation
            content.append(html.Div([
                html.H4("Understanding This Chart", style={
                    'color': TEXT_COLOR,
                    'fontFamily': 'Arial, sans-serif',
                    'marginBottom': '15px',
                    'marginTop': '0'
                }),
                html.P("This chart shows productivity trends across different UK industries alongside GDP, allowing you to compare how different sectors have performed over time.", style={
                    'color': TEXT_COLOR,
                    'fontFamily': 'Arial, sans-serif',
                    'marginBottom': '15px'
                }),
                html.H5("What the lines represent:", style={
                    'color': TEXT_COLOR,
                    'fontFamily': 'Arial, sans-serif',
                    'marginBottom': '10px'
                }),
                html.Ul([
                    html.Li([
                        html.Strong("GDP (black line): "),
                        "The overall economic output of the UK, converted from quarterly growth rates to an index (base = 100) and scaled to match the productivity indices."
                    ], style={'color': TEXT_COLOR, 'fontFamily': 'Arial, sans-serif', 'marginBottom': '8px'}),
                    html.Li([
                        html.Strong("Sector lines (coloured): "),
                        "Productivity indices for each industry sector, measured as output per job or output per hour worked. The base year is 2019 = 100."
                    ], style={'color': TEXT_COLOR, 'fontFamily': 'Arial, sans-serif', 'marginBottom': '8px'}),
                ], style={'paddingLeft': '20px', 'marginBottom': '15px'}),
                html.H5("How to read the chart:", style={
                    'color': TEXT_COLOR,
                    'fontFamily': 'Arial, sans-serif',
                    'marginBottom': '10px'
                }),
                html.Ul([
                    html.Li("Lines trending upward indicate productivity growth; downward means declining productivity.", style={'color': TEXT_COLOR, 'fontFamily': 'Arial, sans-serif', 'marginBottom': '8px'}),
                    html.Li("Compare sector lines to GDP: sectors above GDP have grown faster than the overall economy; sectors below have lagged.", style={'color': TEXT_COLOR, 'fontFamily': 'Arial, sans-serif', 'marginBottom': '8px'}),
                    html.Li("Steeper slopes indicate faster productivity growth rates.", style={'color': TEXT_COLOR, 'fontFamily': 'Arial, sans-serif', 'marginBottom': '8px'}),
                    html.Li("Convergence of lines suggests sectors are growing at similar rates; divergence indicates widening productivity gaps.", style={'color': TEXT_COLOR, 'fontFamily': 'Arial, sans-serif', 'marginBottom': '8px'}),
                ], style={'paddingLeft': '20px', 'marginBottom': '15px'}),
                html.H5("Interacting with the chart:", style={
                    'color': TEXT_COLOR,
                    'fontFamily': 'Arial, sans-serif',
                    'marginBottom': '10px'
                }),
                html.Ul([
                    html.Li([html.Strong("Single click "), "on a legend item to hide/show that sector."], style={'color': TEXT_COLOR, 'fontFamily': 'Arial, sans-serif', 'marginBottom': '8px'}),
                    html.Li([html.Strong("Double click "), "on a legend item to isolate just that sector (hides all others)."], style={'color': TEXT_COLOR, 'fontFamily': 'Arial, sans-serif', 'marginBottom': '8px'}),
                    html.Li([html.Strong("Double click "), "again to restore all sectors."], style={'color': TEXT_COLOR, 'fontFamily': 'Arial, sans-serif', 'marginBottom': '8px'}),
                    html.Li("Hover over lines to see exact values at each point in time.", style={'color': TEXT_COLOR, 'fontFamily': 'Arial, sans-serif', 'marginBottom': '8px'}),
                ], style={'paddingLeft': '20px', 'marginBottom': '15px'}),
                html.H5("Data sources:", style={
                    'color': TEXT_COLOR,
                    'fontFamily': 'Arial, sans-serif',
                    'marginBottom': '10px'
                }),
                html.Ul([
                    html.Li("Sector productivity data: ONS Labour Productivity tables (lprod01.xls) - Tables 2, 3, 4", style={'color': TEXT_COLOR, 'fontFamily': 'Arial, sans-serif', 'marginBottom': '8px'}),
                    html.Li("Manufacturing sectors: Food & beverages, Textiles, Chemicals, Machinery, Transport equipment, etc.", style={'color': TEXT_COLOR, 'fontFamily': 'Arial, sans-serif', 'marginBottom': '8px'}),
                    html.Li("Services sectors: Wholesale & retail, Transport, Accommodation, Finance, Real estate, Professional services, etc.", style={'color': TEXT_COLOR, 'fontFamily': 'Arial, sans-serif', 'marginBottom': '8px'}),
                    html.Li("GDP data: ONS quarterly GDP growth rates, converted to cumulative index.", style={'color': TEXT_COLOR, 'fontFamily': 'Arial, sans-serif', 'marginBottom': '8px'}),
                ], style={'paddingLeft': '20px', 'marginBottom': '15px'}),
                html.H5("Key insights to look for:", style={
                    'color': TEXT_COLOR,
                    'fontFamily': 'Arial, sans-serif',
                    'marginBottom': '10px'
                }),
                html.Ul([
                    html.Li("Which sectors have seen the strongest productivity growth since the 1990s?", style={'color': TEXT_COLOR, 'fontFamily': 'Arial, sans-serif', 'marginBottom': '8px'}),
                    html.Li("How did different sectors respond to the 2008 financial crisis and 2020 COVID pandemic?", style={'color': TEXT_COLOR, 'fontFamily': 'Arial, sans-serif', 'marginBottom': '8px'}),
                    html.Li("Are manufacturing or services sectors driving overall productivity growth?", style={'color': TEXT_COLOR, 'fontFamily': 'Arial, sans-serif', 'marginBottom': '8px'}),
                    html.Li("Which sectors have stagnated or declined in productivity?", style={'color': TEXT_COLOR, 'fontFamily': 'Arial, sans-serif', 'marginBottom': '8px'}),
                ], style={'paddingLeft': '20px'}),
            ], style={
                'backgroundColor': '#ffffff',
                'padding': '20px',
                'borderRadius': '5px',
                'border': '1px solid #000000',
                'marginTop': '20px'
            }))
        else:
            content.append(html.Div("Sector productivity data not available. Please ensure lprod01.xls is in the data folder.", style={
                'padding': '20px',
                'color': TEXT_COLOR,
                'fontFamily': 'Arial, sans-serif'
            }))
        
        return html.Div(content)
    
    elif tab == 'income':
        if len(income_df) > 0:
            fig_e = chart_income_by_nation(income_df, index_enabled)
            return dcc.Graph(figure=fig_e)
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

