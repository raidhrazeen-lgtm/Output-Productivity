"""
Utility functions for UK Productivity Puzzle dashboard.
Contains data transformation helpers for rolling averages and indexing.

Transformation order (must be applied in this sequence):
1. raw data -> 
2. rolling average (if enabled) -> 
3. indexing to base period (if enabled) -> 
4. growth rate calculation (from smoothed/indexed levels) -> 
5. charts/KPIs
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple


def apply_rolling_mean(df: pd.DataFrame, value_col: str = "value", 
                       window: int = 4, date_col: str = "date") -> pd.DataFrame:
    """
    Apply rolling average to smooth a time series.
    
    This should be applied to LEVEL values (not growth rates).
    After smoothing levels, growth rates should be computed from the smoothed series.
    
    Args:
        df: DataFrame with date and value columns
        value_col: Name of the value column to smooth
        window: Rolling window size (default 4 for quarterly data = 1 year)
        date_col: Name of the date column
        
    Returns:
        DataFrame with smoothed values (rows with NaN dropped)
    """
    if df is None or len(df) == 0:
        return df
    
    df = df.sort_values(date_col).copy()
    df[value_col] = df[value_col].rolling(window=window, min_periods=window).mean()
    return df.dropna(subset=[value_col]).reset_index(drop=True)


def apply_indexing(df: pd.DataFrame, value_col: str = "value", 
                   date_col: str = "date", base_period: str = "2007Q4",
                   base_value: float = 100.0) -> Tuple[pd.DataFrame, Optional[float]]:
    """
    Rebase a series to a base period (e.g., 2007Q4 = 100).
    
    This should be applied AFTER rolling average (if rolling is enabled).
    
    Args:
        df: DataFrame with date and value columns
        value_col: Name of the value column to index
        date_col: Name of the date column
        base_period: Base period string (e.g., "2007Q4")
        base_value: Value to set at base period (default 100)
        
    Returns:
        Tuple of (indexed DataFrame, original base value used for indexing)
    """
    if df is None or len(df) == 0:
        return df, None
    
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    
    # Parse base period (e.g., "2007Q4" -> end of Q4 2007)
    try:
        year = int(base_period[:4])
        quarter = int(base_period[-1])
        # Create date at end of quarter
        base_date = pd.Timestamp(year=year, month=quarter * 3, day=1) + pd.offsets.MonthEnd(0)
    except:
        # Fallback: use 2007-12-31
        base_date = pd.Timestamp('2007-12-31')
    
    # Find the value at or before the base date
    pre_base = df[df[date_col] <= base_date]
    if len(pre_base) == 0:
        # No data before base date, use first available
        original_base_value = df[value_col].iloc[0]
    else:
        # Use the last value at or before base date
        original_base_value = pre_base[value_col].iloc[-1]
    
    if original_base_value == 0 or pd.isna(original_base_value):
        return df, None
    
    # Rebase: index = base_value * (value / original_base_value)
    df[value_col] = base_value * (df[value_col] / original_base_value)
    
    return df, original_base_value


def compute_growth_rate(df: pd.DataFrame, value_col: str = "value",
                        date_col: str = "date", periods: int = 1) -> pd.DataFrame:
    """
    Compute period-over-period growth rate from level values.
    
    This should be applied AFTER rolling average and indexing (if enabled).
    Growth rates are computed from the transformed levels.
    
    Args:
        df: DataFrame with date and value columns (already smoothed/indexed if applicable)
        value_col: Name of the value column
        date_col: Name of the date column
        periods: Number of periods for growth calculation (1 = q/q, 4 = y/y)
        
    Returns:
        DataFrame with additional 'growth' column (in %)
    """
    if df is None or len(df) == 0:
        return df
    
    df = df.sort_values(date_col).copy()
    df['growth'] = df[value_col].pct_change(periods) * 100
    return df


def prepare_productivity_series(productivity_df: pd.DataFrame, measure: str,
                                 rolling_enabled: bool = False, 
                                 index_enabled: bool = False,
                                 rolling_window: int = 4,
                                 base_period: str = "2007Q4") -> Tuple[pd.DataFrame, dict]:
    """
    Prepare productivity series with all transformations applied in correct order.
    
    Transformation order:
    1. Filter by measure
    2. Apply rolling average (if enabled) - smooths the raw levels
    3. Apply indexing (if enabled) - rebases to 2007Q4=100
    
    Args:
        productivity_df: Raw productivity DataFrame
        measure: Selected measure (e.g., "Output per hour")
        rolling_enabled: Whether to apply 4-quarter rolling average
        index_enabled: Whether to index to 2007Q4=100
        rolling_window: Window size for rolling average
        base_period: Base period for indexing
        
    Returns:
        Tuple of (transformed DataFrame, metadata dict with debug info)
    """
    metadata = {
        'measure': measure,
        'rolling_enabled': rolling_enabled,
        'index_enabled': index_enabled,
        'original_rows': 0,
        'final_rows': 0,
        'first_value': None,
        'last_value': None,
        'base_value_used': None
    }
    
    if productivity_df is None or len(productivity_df) == 0:
        return pd.DataFrame(), metadata
    
    # Step 1: Filter by measure
    df = productivity_df[productivity_df['measure'] == measure].copy()
    df = df.sort_values('date').reset_index(drop=True)
    metadata['original_rows'] = len(df)
    
    if len(df) == 0:
        return df, metadata
    
    # Step 2: Apply rolling average (if enabled) - on RAW LEVELS
    if rolling_enabled:
        df = apply_rolling_mean(df, value_col='value', window=rolling_window)
    
    # Step 3: Apply indexing (if enabled) - AFTER rolling
    if index_enabled:
        df, base_val = apply_indexing(df, value_col='value', base_period=base_period)
        metadata['base_value_used'] = base_val
    
    metadata['final_rows'] = len(df)
    if len(df) > 0:
        metadata['first_value'] = df['value'].iloc[0]
        metadata['last_value'] = df['value'].iloc[-1]
    
    return df, metadata


def prepare_counterfactual_series(counterfactual_df: pd.DataFrame,
                                   rolling_enabled: bool = False,
                                   index_enabled: bool = False,
                                   rolling_window: int = 4,
                                   base_period: str = "2007Q4") -> pd.DataFrame:
    """
    Prepare counterfactual series with transformations.
    
    The counterfactual has columns: date, actual, counterfactual, gap
    We need to apply transformations to both actual and counterfactual columns.
    
    Args:
        counterfactual_df: DataFrame with counterfactual data
        rolling_enabled: Whether to apply rolling average
        index_enabled: Whether to index to base period
        rolling_window: Window size for rolling
        base_period: Base period for indexing
        
    Returns:
        Transformed counterfactual DataFrame
    """
    if counterfactual_df is None or len(counterfactual_df) == 0:
        return counterfactual_df
    
    df = counterfactual_df.copy()
    df = df.sort_values('date').reset_index(drop=True)
    
    # Step 1: Apply rolling average to both actual and counterfactual
    if rolling_enabled:
        df['actual'] = df['actual'].rolling(window=rolling_window, min_periods=rolling_window).mean()
        df['counterfactual'] = df['counterfactual'].rolling(window=rolling_window, min_periods=rolling_window).mean()
        df = df.dropna(subset=['actual', 'counterfactual']).reset_index(drop=True)
    
    if len(df) == 0:
        return df
    
    # Step 2: Apply indexing (if enabled)
    if index_enabled:
        # Find base value from actual series
        base_date = pd.Timestamp('2007-12-31')
        pre_base = df[df['date'] <= base_date]
        if len(pre_base) > 0:
            base_actual = pre_base['actual'].iloc[-1]
            if base_actual != 0 and not pd.isna(base_actual):
                df['actual'] = 100 * df['actual'] / base_actual
                df['counterfactual'] = 100 * df['counterfactual'] / base_actual
    
    # Recalculate gap
    df['gap'] = df['counterfactual'] - df['actual']
    
    return df

