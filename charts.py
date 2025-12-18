"""
Chart building functions for UK Productivity Puzzle dashboard.
All charts use Plotly with minimalist theme.
"""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from styles import CHART_LAYOUT, COLOR_ACTUAL, COLOR_COUNTERFACTUAL, COLOR_GAP, COLOR_GFC, COLOR_COVID


def chart_productivity_gap(counterfactual_df: pd.DataFrame, measure: str = 'Output per hour',
                          date_range: tuple = None) -> go.Figure:
    """
    Chart A: Productivity gap (hero chart).
    Shows actual vs counterfactual productivity with gap shading.
    
    Args:
        counterfactual_df: DataFrame with columns: date, actual, counterfactual, gap
        measure: Productivity measure name
        date_range: Tuple of (start_date, end_date) for filtering
        
    Returns:
        Plotly figure
    """
    df = counterfactual_df.copy()
    
    if date_range:
        df = df[(df['date'] >= date_range[0]) & (df['date'] <= date_range[1])]
    
    fig = go.Figure()
    
    # Add gap shading
    fig.add_trace(go.Scatter(
        x=pd.concat([df['date'], df['date'][::-1]]),
        y=pd.concat([df['counterfactual'], df['actual'][::-1]]),
        fill='toself',
        fillcolor=COLOR_GAP,
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo='skip',
        showlegend=False,
        name='Gap'
    ))
    
    # Actual line
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['actual'],
        mode='lines',
        name='Actual',
        line=dict(color=COLOR_ACTUAL, width=2),
        hovertemplate='<b>Actual</b><br>Date: %{x}<br>Value: %{y:.2f}<extra></extra>'
    ))
    
    # Counterfactual line
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['counterfactual'],
        mode='lines',
        name='Pre-2008 Trend',
        line=dict(color=COLOR_COUNTERFACTUAL, width=2, dash='dash'),
        hovertemplate='<b>Pre-2008 Trend</b><br>Date: %{x}<br>Value: %{y:.2f}<extra></extra>'
    ))
    
    # Vertical lines for GFC and COVID
    gfc_date = pd.Timestamp('2008-01-01')
    covid_date = pd.Timestamp('2020-01-01')
    
    if df['date'].min() <= gfc_date <= df['date'].max():
        fig.add_vline(
            x=gfc_date,
            line_dash="dot",
            line_color=COLOR_GFC,
            annotation_text="GFC",
            annotation_position="top"
        )
    
    if df['date'].min() <= covid_date <= df['date'].max():
        fig.add_vline(
            x=covid_date,
            line_dash="dot",
            line_color=COLOR_COVID,
            annotation_text="COVID",
            annotation_position="top"
        )
    
    fig.update_layout(
        **CHART_LAYOUT,
        title={
            'text': 'UK Productivity: Actual vs Pre-2008 Trend',
            'font': {'family': 'Arial, sans-serif', 'color': '#000000'}
        },
        xaxis_title='Date',
        yaxis_title=f'{measure} (Indexed)',
        hovermode='x unified'
    )
    
    return fig


def chart_growth_comparison(productivity_df: pd.DataFrame, pre_start: str = '1997Q1', 
                           pre_end: str = '2007Q4', post_start: str = '2008Q1') -> go.Figure:
    """
    Chart B: Pre vs Post growth comparison.
    Bar chart showing average annualised growth rates.
    
    Args:
        productivity_df: DataFrame with columns: date, measure, value
        pre_start: Pre-crisis start
        pre_end: Pre-crisis end
        post_start: Post-crisis start
        
    Returns:
        Plotly figure
    """
    def parse_quarter(q_str):
        year, q = q_str.split('Q')
        month = int(q) * 3
        return pd.Timestamp(int(year), month, 1) + pd.offsets.QuarterEnd()
    
    pre_start_date = parse_quarter(pre_start)
    pre_end_date = parse_quarter(pre_end)
    post_start_date = parse_quarter(post_start)
    
    measures = productivity_df['measure'].unique()
    data = []
    
    for measure in measures:
        measure_df = productivity_df[productivity_df['measure'] == measure].copy()
        measure_df = measure_df.sort_values('date')
        
        # Pre-crisis
        pre_df = measure_df[(measure_df['date'] >= pre_start_date) & (measure_df['date'] <= pre_end_date)]
        if len(pre_df) > 1:
            pre_growth = ((pre_df['value'].iloc[-1] / pre_df['value'].iloc[0]) ** (4 / len(pre_df)) - 1) * 100
        else:
            pre_growth = 0
        
        # Post-crisis
        post_df = measure_df[measure_df['date'] >= post_start_date]
        if len(post_df) > 1:
            post_growth = ((post_df['value'].iloc[-1] / post_df['value'].iloc[0]) ** (4 / len(post_df)) - 1) * 100
        else:
            post_growth = 0
        
        data.append({
            'Measure': measure,
            'Period': 'Pre-2008',
            'Growth Rate (%)': pre_growth
        })
        data.append({
            'Measure': measure,
            'Period': 'Post-2008',
            'Growth Rate (%)': post_growth
        })
    
    bar_df = pd.DataFrame(data)
    
    fig = go.Figure()
    
    for period in ['Pre-2008', 'Post-2008']:
        period_df = bar_df[bar_df['Period'] == period]
        fig.add_trace(go.Bar(
            x=period_df['Measure'],
            y=period_df['Growth Rate (%)'],
            name=period,
            hovertemplate=f'<b>{period}</b><br>%{{x}}<br>Growth: %{{y:.2f}}%<extra></extra>'
        ))
    
    fig.update_layout(
        **CHART_LAYOUT,
        title={
            'text': 'Productivity Growth: Pre vs Post-2008',
            'font': {'family': 'Arial, sans-serif', 'color': '#000000'}
        },
        xaxis_title='Productivity Measure',
        yaxis_title='Annualised Growth Rate (%)',
        barmode='group',
        hovermode='x unified'
    )
    
    return fig


def chart_gdp_vs_productivity(gdp_df: pd.DataFrame, productivity_df: pd.DataFrame,
                              measure: str = 'Output per hour', rolling_avg: bool = False) -> go.Figure:
    """
    Chart C: GDP growth vs productivity growth.
    Dual y-axis chart with optional rolling average.
    
    Args:
        gdp_df: DataFrame with columns: date, gdp_qoq
        productivity_df: DataFrame with columns: date, measure, value
        measure: Selected productivity measure
        rolling_avg: Whether to show 4-quarter rolling average
        
    Returns:
        Plotly figure
    """
    # Filter productivity by measure
    prod_df = productivity_df[productivity_df['measure'] == measure].copy()
    prod_df = prod_df.sort_values('date')
    
    # Calculate q/q growth for productivity
    prod_df['prod_qoq'] = prod_df['value'].pct_change(1) * 100
    
    # Merge on date (quarterly alignment)
    gdp_df = gdp_df.copy()
    gdp_df['date'] = pd.to_datetime(gdp_df['date'])
    prod_df['date'] = pd.to_datetime(prod_df['date'])
    
    # Align to quarter end
    gdp_df['date'] = gdp_df['date'] + pd.offsets.QuarterEnd(0) - gdp_df['date']
    prod_df['date'] = prod_df['date'] + pd.offsets.QuarterEnd(0) - prod_df['date']
    
    merged = pd.merge(gdp_df, prod_df[['date', 'prod_qoq']], on='date', how='inner')
    merged = merged.sort_values('date')
    
    if rolling_avg:
        merged['gdp_qoq_rolling'] = merged['gdp_qoq'].rolling(4, min_periods=1).mean()
        merged['prod_qoq_rolling'] = merged['prod_qoq'].rolling(4, min_periods=1).mean()
        gdp_col = 'gdp_qoq_rolling'
        prod_col = 'prod_qoq_rolling'
    else:
        gdp_col = 'gdp_qoq'
        prod_col = 'prod_qoq'
    
    fig = go.Figure()
    
    # GDP growth (left axis)
    fig.add_trace(go.Scatter(
        x=merged['date'],
        y=merged[gdp_col],
        mode='lines',
        name='GDP q/q Growth',
        line=dict(color='#1f77b4', width=2),
        yaxis='y',
        hovertemplate='<b>GDP</b><br>Date: %{x}<br>Growth: %{y:.2f}%<extra></extra>'
    ))
    
    # Productivity growth (right axis)
    fig.add_trace(go.Scatter(
        x=merged['date'],
        y=merged[prod_col],
        mode='lines',
        name=f'{measure} q/q Growth',
        line=dict(color='#ff7f0e', width=2),
        yaxis='y2',
        hovertemplate=f'<b>{measure}</b><br>Date: %{x}<br>Growth: %{{y:.2f}}%<extra></extra>'
    ))
    
    fig.update_layout(
        **CHART_LAYOUT,
        title={
            'text': 'Output vs Productivity Momentum',
            'font': {'family': 'Arial, sans-serif', 'color': '#000000'}
        },
        xaxis_title='Date',
        yaxis=dict(
            title='GDP q/q Growth (%)',
            side='left'
        ),
        yaxis2=dict(
            title=f'{measure} q/q Growth (%)',
            side='right',
            overlaying='y'
        ),
        hovermode='x unified'
    )
    
    return fig


def chart_production_output(production_df: pd.DataFrame, gdp_df: pd.DataFrame = None,
                           productivity_df: pd.DataFrame = None, measure: str = 'Output per hour', 
                           overlay_productivity: bool = False) -> go.Figure:
    """
    Chart D: Production output index with GDP overlay.
    Shows production output, GDP, and optional productivity overlay.
    
    Args:
        production_df: DataFrame with columns: date, prod_index
        gdp_df: Optional DataFrame with columns: date, gdp_qoq (for GDP level/index)
        productivity_df: Optional DataFrame with columns: date, measure, value
        measure: Productivity measure for overlay
        overlay_productivity: Whether to overlay productivity
        
    Returns:
        Plotly figure with white background for better legibility
    """
    fig = go.Figure()
    
    # Prepare all datasets
    prod_df = production_df.copy()
    prod_df = prod_df.sort_values('date')
    
    # Collect all date ranges to find overlapping period
    date_ranges = []
    if len(prod_df) > 0:
        date_ranges.append((prod_df['date'].min(), prod_df['date'].max()))
    
    gdp_data = None
    if gdp_df is not None and len(gdp_df) > 0:
        gdp_data = gdp_df.copy()
        gdp_data = gdp_data.sort_values('date')
        date_ranges.append((gdp_data['date'].min(), gdp_data['date'].max()))
    
    prod_measure_df = None
    if overlay_productivity and productivity_df is not None:
        prod_measure_df = productivity_df[productivity_df['measure'] == measure].copy()
        if len(prod_measure_df) > 0:
            prod_measure_df = prod_measure_df.sort_values('date')
            date_ranges.append((prod_measure_df['date'].min(), prod_measure_df['date'].max()))
    
    # Find overlapping date range (intersection of all date ranges)
    if len(date_ranges) > 0:
        overlap_start = max([dr[0] for dr in date_ranges])
        overlap_end = min([dr[1] for dr in date_ranges])
    else:
        overlap_start = None
        overlap_end = None
    
    # Filter all datasets to overlapping period only
    if overlap_start is not None and overlap_end is not None:
        prod_df = prod_df[(prod_df['date'] >= overlap_start) & (prod_df['date'] <= overlap_end)]
        
        if gdp_data is not None:
            gdp_data = gdp_data[(gdp_data['date'] >= overlap_start) & (gdp_data['date'] <= overlap_end)]
        
        if prod_measure_df is not None:
            prod_measure_df = prod_measure_df[(prod_measure_df['date'] >= overlap_start) & (prod_measure_df['date'] <= overlap_end)]
    
    # Production line (already filtered to overlap)
    fig.add_trace(go.Scatter(
        x=prod_df['date'],
        y=prod_df['prod_index'],
        mode='lines',
        name='Production Output Index',
        line=dict(color='#1f77b4', width=2),
        hovertemplate='<b>Production</b><br>Date: %{x}<br>Index: %{y:.2f}<extra></extra>'
    ))
    
    # Add GDP line (only in overlapping period)
    if gdp_data is not None and len(gdp_data) > 0:
        # If GDP is q/q growth, convert to cumulative index
        if 'gdp_qoq' in gdp_data.columns:
            # Convert q/q growth to cumulative index (base 100 at first date)
            gdp_data['gdp_index'] = 100
            for i in range(1, len(gdp_data)):
                gdp_data.iloc[i, gdp_data.columns.get_loc('gdp_index')] = (
                    gdp_data.iloc[i-1, gdp_data.columns.get_loc('gdp_index')] * 
                    (1 + gdp_data.iloc[i, gdp_data.columns.get_loc('gdp_qoq')] / 100)
                )
            
            # Normalize to match production scale (align first overlapping date)
            if len(prod_df) > 0 and len(gdp_data) > 0:
                merged = pd.merge(prod_df, gdp_data, on='date', how='inner')
                if len(merged) > 0:
                    scale_factor = merged['prod_index'].iloc[0] / merged['gdp_index'].iloc[0] if merged['gdp_index'].iloc[0] != 0 else 1
                    gdp_data['gdp_index_scaled'] = gdp_data['gdp_index'] * scale_factor
                else:
                    # If no overlap after filtering, scale to production range
                    gdp_mean = gdp_data['gdp_index'].mean()
                    prod_mean = prod_df['prod_index'].mean()
                    scale_factor = prod_mean / gdp_mean if gdp_mean != 0 else 1
                    gdp_data['gdp_index_scaled'] = gdp_data['gdp_index'] * scale_factor
            else:
                gdp_data['gdp_index_scaled'] = gdp_data['gdp_index']
            
            fig.add_trace(go.Scatter(
                x=gdp_data['date'],
                y=gdp_data['gdp_index_scaled'],
                mode='lines',
                name='GDP (Index, scaled)',
                line=dict(color='#2ca02c', width=2),
                hovertemplate='<b>GDP</b><br>Date: %{x}<br>Index: %{y:.2f}<extra></extra>'
            ))
    
    # Optional productivity overlay (only in overlapping period)
    if prod_measure_df is not None and len(prod_measure_df) > 0 and len(prod_df) > 0:
        # Find overlapping dates for scaling
        merged = pd.merge(prod_df, prod_measure_df, on='date', how='inner', suffixes=('_prod', '_prod_measure'))
        if len(merged) > 0:
            # Normalize productivity to production scale
            scale_factor = merged['prod_index'].iloc[0] / merged['value'].iloc[0] if merged['value'].iloc[0] != 0 else 1
            prod_measure_df['value_scaled'] = prod_measure_df['value'] * scale_factor
            
            fig.add_trace(go.Scatter(
                x=prod_measure_df['date'],
                y=prod_measure_df['value_scaled'],
                mode='lines',
                name=f'{measure} (scaled)',
                line=dict(color='#ff7f0e', width=2, dash='dot'),
                hovertemplate=f'<b>{measure}</b><br>Date: %{{x}}<br>Value: %{{y:.2f}}<extra></extra>'
            ))
    
    # Use white background for better legibility on teal page background
    # White background provides better contrast for the chart elements and makes text/data easier to read
    fig.update_layout(
        plot_bgcolor='white',  # White plot background for legibility
        paper_bgcolor='white',  # White paper background
        font={'family': 'Arial, sans-serif', 'size': 12, 'color': '#000000'},
        title={
            'text': 'Production Output, GDP, and Productivity',
            'font': {'family': 'Arial, sans-serif', 'color': '#000000'}
        },
        xaxis={
            'showgrid': True,
            'gridcolor': 'rgba(0,0,0,0.1)',
            'gridwidth': 1,
            'showline': True,
            'linecolor': 'rgba(0,0,0,0.2)',
            'title': {'text': 'Date', 'font': {'family': 'Arial, sans-serif', 'color': '#000000'}},
            'tickfont': {'family': 'Arial, sans-serif', 'color': '#000000'},
        },
        yaxis={
            'showgrid': True,
            'gridcolor': 'rgba(0,0,0,0.1)',
            'gridwidth': 1,
            'showline': True,
            'linecolor': 'rgba(0,0,0,0.2)',
            'title': {'text': 'Index', 'font': {'family': 'Arial, sans-serif', 'color': '#000000'}},
            'tickfont': {'family': 'Arial, sans-serif', 'color': '#000000'},
        },
        margin={'l': 60, 'r': 20, 't': 40, 'b': 60},
        hovermode='x unified'
    )
    
    return fig


def chart_income_by_nation(income_df: pd.DataFrame, index_to_base: bool = False,
                          base_year: int = 2007) -> go.Figure:
    """
    Chart E: Living standards (income).
    Line chart showing median income by nation.
    
    Args:
        income_df: DataFrame with columns: year, region, median_income
        index_to_base: Whether to index to base year
        base_year: Base year for indexing
        
    Returns:
        Plotly figure
    """
    df = income_df.copy()
    df = df.sort_values(['region', 'year'])
    
    if index_to_base:
        # Index each region to base year
        def index_region(group):
            base_mask = group['year'].dt.year == base_year
            if base_mask.any():
                base_value = group.loc[base_mask, 'median_income'].iloc[0]
                if base_value != 0 and pd.notna(base_value):
                    return (group['median_income'] / base_value) * 100
            return group['median_income']
        
        df['median_income_indexed'] = df.groupby('region', group_keys=False).apply(index_region).reset_index(drop=True)
        value_col = 'median_income_indexed'
        y_title = f'Median Income (Indexed, {base_year}=100)'
    else:
        value_col = 'median_income'
        y_title = 'Median Household Income (£)'
    
    fig = go.Figure()
    
    for region in df['region'].unique():
        region_df = df[df['region'] == region].sort_values('year')
        fig.add_trace(go.Scatter(
            x=region_df['year'],
            y=region_df[value_col],
            mode='lines+markers',
            name=region,
            line=dict(width=2),
            hovertemplate=f'<b>{region}</b><br>Year: %{{x|%Y}}<br>Income: %{{y:,.0f}}<extra></extra>'
        ))
    
    fig.update_layout(
        **CHART_LAYOUT,
        title={
            'text': 'Median Household Income (Equivalised Disposable)',
            'font': {'family': 'Arial, sans-serif', 'color': '#000000'}
        },
        xaxis_title='Year',
        yaxis_title=y_title,
        hovermode='x unified'
    )
    
    return fig


def chart_sector_heatmap(sector_df: pd.DataFrame) -> go.Figure:
    """
    Bonus chart: Sector productivity heatmap.
    Shows sector vs time (indexed).
    
    Args:
        sector_df: DataFrame with columns: date, sector, value
        
    Returns:
        Plotly figure
    """
    # Index each sector to first available date
    sector_df = sector_df.copy()
    sector_df = sector_df.sort_values(['sector', 'date'])
    
    # Index to first date for each sector
    first_values = sector_df.groupby('sector')['value'].first()
    sector_df = sector_df.merge(first_values, on='sector', suffixes=('', '_first'))
    sector_df['value_indexed'] = (sector_df['value'] / sector_df['value_first']) * 100
    
    # Pivot for heatmap
    pivot_df = sector_df.pivot_table(
        index='sector',
        columns='date',
        values='value_indexed',
        aggfunc='mean'
    )
    
    fig = go.Figure(data=go.Heatmap(
        z=pivot_df.values,
        x=[str(d) for d in pivot_df.columns],
        y=pivot_df.index,
        colorscale='RdYlGn',
        hovertemplate='<b>%{y}</b><br>Date: %{x}<br>Index: %{z:.1f}<extra></extra>'
    ))
    
    fig.update_layout(
        **CHART_LAYOUT,
        title={
            'text': 'Sector Productivity (Indexed)',
            'font': {'family': 'Arial, sans-serif', 'color': '#000000'}
        },
        xaxis_title='Date',
        yaxis_title='Sector',
        height=400
    )
    
    return fig


def chart_sector_productivity_with_gdp(sector_df: pd.DataFrame, gdp_df: pd.DataFrame = None,
                                        selected_sectors: list = None) -> go.Figure:
    """
    Interactive chart showing sector productivity and GDP over time.
    User can select/deselect sectors via the legend.
    
    Args:
        sector_df: DataFrame with columns: date, sector, value (productivity index)
        gdp_df: DataFrame with columns: date, gdp_qoq (GDP growth rates)
        selected_sectors: List of sectors to show (None = show all)
        
    Returns:
        Plotly figure with white background
    """
    fig = go.Figure()
    
    df = sector_df.copy()
    df = df.sort_values(['sector', 'date'])
    
    # Get unique sectors
    all_sectors = df['sector'].unique().tolist()
    
    # If selected_sectors provided, filter
    if selected_sectors is not None:
        sectors_to_plot = [s for s in selected_sectors if s in all_sectors]
    else:
        sectors_to_plot = all_sectors
    
    # Find overlapping date range across all data
    min_date = df['date'].min()
    max_date = df['date'].max()
    
    if gdp_df is not None and len(gdp_df) > 0:
        gdp_min = gdp_df['date'].min()
        gdp_max = gdp_df['date'].max()
        min_date = max(min_date, gdp_min)
        max_date = min(max_date, gdp_max)
    
    # Filter to overlapping period
    df = df[(df['date'] >= min_date) & (df['date'] <= max_date)]
    
    # Add GDP line first (if available)
    if gdp_df is not None and len(gdp_df) > 0:
        gdp_data = gdp_df.copy()
        gdp_data = gdp_data.sort_values('date')
        gdp_data = gdp_data[(gdp_data['date'] >= min_date) & (gdp_data['date'] <= max_date)]
        
        if len(gdp_data) > 0 and 'gdp_qoq' in gdp_data.columns:
            # Convert q/q growth to cumulative index (base 100)
            gdp_data['gdp_index'] = 100.0
            for i in range(1, len(gdp_data)):
                gdp_data.iloc[i, gdp_data.columns.get_loc('gdp_index')] = (
                    gdp_data.iloc[i-1, gdp_data.columns.get_loc('gdp_index')] * 
                    (1 + gdp_data.iloc[i, gdp_data.columns.get_loc('gdp_qoq')] / 100)
                )
            
            # Scale GDP to roughly match sector productivity levels
            # Use median of first values across sectors
            first_values = df.groupby('sector')['value'].first()
            if len(first_values) > 0:
                median_first = first_values.median()
                gdp_first = gdp_data['gdp_index'].iloc[0]
                scale_factor = median_first / gdp_first if gdp_first != 0 else 1
                gdp_data['gdp_index_scaled'] = gdp_data['gdp_index'] * scale_factor
            else:
                gdp_data['gdp_index_scaled'] = gdp_data['gdp_index']
            
            fig.add_trace(go.Scatter(
                x=gdp_data['date'],
                y=gdp_data['gdp_index_scaled'],
                mode='lines',
                name='GDP (scaled index)',
                line=dict(color='#000000', width=3, dash='solid'),
                hovertemplate='<b>GDP</b><br>Date: %{x}<br>Index: %{y:.1f}<extra></extra>',
                visible=True
            ))
    
    # Add sector lines
    colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
        '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5',
        '#c49c94', '#f7b6d2', '#c7c7c7', '#dbdb8d', '#9edae5'
    ]
    
    for i, sector in enumerate(sectors_to_plot):
        sector_data = df[df['sector'] == sector]
        if len(sector_data) > 0:
            # Clean up sector name for display
            display_name = sector.replace('\n', ' ').replace('  ', ' ').strip()
            
            fig.add_trace(go.Scatter(
                x=sector_data['date'],
                y=sector_data['value'],
                mode='lines',
                name=display_name,
                line=dict(color=colors[i % len(colors)], width=2),
                hovertemplate=f'<b>{display_name}</b><br>Date: %{{x}}<br>Index: %{{y:.1f}}<extra></extra>',
                visible=True
            ))
    
    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        font={'family': 'Arial, sans-serif', 'size': 12, 'color': '#000000'},
        title={
            'text': 'Sector Productivity and GDP Over Time',
            'font': {'family': 'Arial, sans-serif', 'color': '#000000', 'size': 16}
        },
        xaxis={
            'title': {'text': 'Year', 'font': {'family': 'Arial, sans-serif', 'color': '#000000'}},
            'showgrid': True,
            'gridcolor': 'rgba(0,0,0,0.1)',
            'tickfont': {'family': 'Arial, sans-serif', 'color': '#000000'},
        },
        yaxis={
            'title': {'text': 'Productivity Index (2019=100)', 'font': {'family': 'Arial, sans-serif', 'color': '#000000'}},
            'showgrid': True,
            'gridcolor': 'rgba(0,0,0,0.1)',
            'tickfont': {'family': 'Arial, sans-serif', 'color': '#000000'},
        },
        hovermode='x unified',
        margin={'l': 60, 'r': 20, 't': 60, 'b': 60},
        legend={
            'orientation': 'v',
            'yanchor': 'top',
            'y': 1,
            'xanchor': 'left',
            'x': 1.02,
            'font': {'family': 'Arial, sans-serif', 'size': 10, 'color': '#000000'},
            'itemclick': 'toggle',  # Click to toggle visibility
            'itemdoubleclick': 'toggleothers'  # Double-click to isolate
        },
        height=600
    )
    
    return fig


def chart_sector_output_lines(sector_df: pd.DataFrame, max_sectors: int = 8) -> go.Figure:
    """
    Line chart for sector-level production output indices.

    Shows up to max_sectors sectors over time.
    """
    df = sector_df.copy()
    df = df.sort_values(["sector", "date"])

    # Choose top sectors by data availability (or variance)
    sector_order = (
        df.groupby("sector")["value"]
        .count()
        .sort_values(ascending=False)
        .head(max_sectors)
        .index
        .tolist()
    )

    fig = go.Figure()

    for sector in sector_order:
        s_df = df[df["sector"] == sector]
        fig.add_trace(
            go.Scatter(
                x=s_df["date"],
                y=s_df["value"],
                mode="lines",
                name=str(sector),
                hovertemplate="<b>%{fullData.name}</b><br>Date: %{x}<br>Index: %{y:.1f}<extra></extra>",
            )
        )

    fig.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        font={"family": "Arial, sans-serif", "size": 12, "color": "#000000"},
        title={
            "text": "Sector Production Output Indices",
            "font": {"family": "Arial, sans-serif", "color": "#000000"},
        },
        xaxis={
            "title": {"text": "Date", "font": {"family": "Arial, sans-serif", "color": "#000000"}},
            "showgrid": True,
            "gridcolor": "rgba(0,0,0,0.1)",
            "tickfont": {"family": "Arial, sans-serif", "color": "#000000"},
        },
        yaxis={
            "title": {"text": "Index", "font": {"family": "Arial, sans-serif", "color": "#000000"}},
            "showgrid": True,
            "gridcolor": "rgba(0,0,0,0.1)",
            "tickfont": {"family": "Arial, sans-serif", "color": "#000000"},
        },
        hovermode="x unified",
        margin={"l": 60, "r": 20, "t": 40, "b": 60},
    )

    return fig

