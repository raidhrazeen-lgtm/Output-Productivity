"""
Data loading and cleaning functions for UK Productivity Puzzle dashboard.
Handles ONS series files, productivity tables, and income deciles.
"""

import pandas as pd
import numpy as np
import os
import warnings
import traceback
from pathlib import Path
from typing import Optional, Tuple, List, Dict


def get_data_path(filename: str) -> str:
    """
    Robust path detection for data files.
    Searches in order:
    1. ./data/<filename>
    2. ./<filename>
    3. ../data/<filename>
    
    Args:
        filename: Name of the file to find
        
    Returns:
        Full absolute path to the file
        
    Raises:
        FileNotFoundError: If file not found in any location, with paths tried
    """
    paths_tried = []
    
    # Try ./data/<filename>
    data_path = Path('data') / filename
    paths_tried.append(str(data_path.resolve()))
    if data_path.exists():
        return str(data_path.resolve())
    
    # Try ./<filename>
    current_path = Path(filename)
    paths_tried.append(str(current_path.resolve()))
    if current_path.exists():
        return str(current_path.resolve())
    
    # Try ../data/<filename>
    parent_data_path = Path('..') / 'data' / filename
    paths_tried.append(str(parent_data_path.resolve()))
    if parent_data_path.exists():
        return str(parent_data_path.resolve())
    
    # If none found, raise with all paths tried
    raise FileNotFoundError(
        f"Could not find {filename}. Paths tried:\n" + 
        "\n".join(f"  - {p}" for p in paths_tried)
    )


def find_data_file(filename: str) -> str:
    """
    Legacy function name - calls get_data_path for backward compatibility.
    """
    return get_data_path(filename)


def read_excel_any(path: str, sheet: Optional[str] = None) -> pd.DataFrame:
    """
    Read Excel file safely, using openpyxl for xlsx and xlrd for xls.
    Handles engine issues gracefully.
    
    Args:
        path: Path to Excel file (can be filename or full path)
        sheet: Sheet name or index (None for first sheet)
        
    Returns:
        DataFrame with the sheet data
    """
    # If path doesn't exist as-is, try to find it using get_data_path
    if not Path(path).exists():
        try:
            path = get_data_path(Path(path).name)  # Extract just filename
        except FileNotFoundError:
            # If get_data_path fails, try the original path anyway
            pass
    
    try:
        # Try openpyxl for xlsx files
        if path.endswith('.xlsx'):
            return pd.read_excel(path, sheet_name=sheet, engine='openpyxl')
        # Use xlrd for xls files - REQUIRED for .xls
        elif path.endswith('.xls'):
            return pd.read_excel(path, sheet_name=sheet, engine='xlrd')
        else:
            # Default attempt
            return pd.read_excel(path, sheet_name=sheet)
    except Exception as e:
        # Fallback: try without specifying engine
        try:
            return pd.read_excel(path, sheet_name=sheet)
        except Exception as e2:
            raise Exception(f"Failed to read {path} with openpyxl/xlrd and default engine. "
                          f"Original error: {str(e)}, Fallback error: {str(e2)}")


def parse_ons_series(path: str, date_col_pattern: str = "Time period|Quarter") -> pd.DataFrame:
    """
    Parse ONS series file, finding header row and extracting time series.
    
    Args:
        path: Path to ONS series file
        date_col_pattern: Regex pattern to find date column
        
    Returns:
        DataFrame with columns: date, value
    """
    df = read_excel_any(path)
    
    # Find header row by searching for date column pattern
    header_row = None
    date_col_idx = None
    
    for idx, row in df.iterrows():
        row_str = ' '.join([str(val) for val in row.values if pd.notna(val)])
        if pd.notna(row_str):
            # Check if any column contains the date pattern
            for col_idx, val in enumerate(row.values):
                if pd.notna(val) and isinstance(val, str):
                    import re
                    if re.search(date_col_pattern, val, re.IGNORECASE):
                        header_row = idx
                        date_col_idx = col_idx
                        break
        if header_row is not None:
            break
    
    if header_row is None:
        # Try first row as header
        header_row = 0
        date_col_idx = 0
    
    # Extract table from header row downward
    df_clean = df.iloc[header_row:].copy()
    df_clean.columns = df_clean.iloc[0]
    df_clean = df_clean.iloc[1:].reset_index(drop=True)
    
    # Find date and value columns
    date_col = None
    value_col = None
    
    for col in df_clean.columns:
        col_str = str(col).lower()
        if 'time' in col_str or 'period' in col_str or 'quarter' in col_str or 'date' in col_str:
            date_col = col
        elif 'value' in col_str or 'index' in col_str or 'growth' in col_str or '%' in col_str:
            if value_col is None:  # Take first numeric column
                value_col = col
    
    # If value column not found, use first numeric column after date
    if value_col is None:
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            value_col = numeric_cols[0]
        else:
            # Try to find any column with numeric data
            for col in df_clean.columns:
                if col != date_col:
                    try:
                        pd.to_numeric(df_clean[col].astype(str).str.replace(',', '').str.replace('..', ''))
                        value_col = col
                        break
                    except:
                        continue
    
    if date_col is None or value_col is None:
        raise ValueError(f"Could not identify date and value columns in {path}")
    
    # Extract and clean
    result = pd.DataFrame({
        'date': df_clean[date_col],
        'value': df_clean[value_col]
    })
    
    # Convert date to datetime/period
    result['date'] = pd.to_datetime(result['date'], errors='coerce', format='%Y Q%q', infer_datetime_format=True)
    
    # Handle quarterly format like "2020 Q1"
    if result['date'].isna().any():
        def parse_quarter(date_str):
            if pd.isna(date_str):
                return pd.NaT
            date_str = str(date_str).strip()
            try:
                if 'Q' in date_str.upper():
                    year, q = date_str.upper().split('Q')
                    year = int(year.strip())
                    q = int(q.strip())
                    # Convert to quarter end
                    month = q * 3
                    return pd.Timestamp(year, month, 1) + pd.offsets.QuarterEnd()
                else:
                    return pd.to_datetime(date_str)
            except:
                return pd.NaT
        
        result['date'] = result['date'].apply(parse_quarter)
    
    # Clean value column
    result['value'] = result['value'].astype(str).str.replace(',', '').str.replace('..', '').str.replace('-', '')
    result['value'] = pd.to_numeric(result['value'], errors='coerce')
    
    # Remove rows with missing dates or values
    result = result.dropna(subset=['date', 'value'])
    result = result.sort_values('date').reset_index(drop=True)
    
    return result


def load_gdp_qoq() -> pd.DataFrame:
    """
    Load GDP q/q growth % from GDP-series file.
    
    Returns:
        DataFrame with columns: date, gdp_qoq
    """
    try:
        path = find_data_file("GDP-series-161225 copy.csv")
        # Read without header first to find data start
        df_raw = pd.read_csv(path, header=None)
        
        # Find where data starts (look for quarter format like "1955 Q2")
        data_start_row = None
        for idx, row in df_raw.iterrows():
            first_val = str(row.iloc[0]).strip()
            if 'Q' in first_val.upper() and any(char.isdigit() for char in first_val):
                data_start_row = idx
                break
        
        if data_start_row is None:
            # Try ONS series parser as fallback
            return parse_ons_series(path)
        
        # Extract data starting from data_start_row
        df = df_raw.iloc[data_start_row:].copy()
        df.columns = ['date', 'gdp_qoq']
        df = df.reset_index(drop=True)
        
        # Parse dates (format: "1955 Q2")
        def parse_quarter(date_str):
            if pd.isna(date_str):
                return pd.NaT
            date_str = str(date_str).strip()
            try:
                if 'Q' in date_str.upper():
                    year, q = date_str.upper().split('Q')
                    year = int(year.strip())
                    q = int(q.strip())
                    month = q * 3
                    return pd.Timestamp(year, month, 1) + pd.offsets.QuarterEnd()
                else:
                    return pd.to_datetime(date_str)
            except:
                return pd.NaT
        
        result = pd.DataFrame({
            'date': df['date'].apply(parse_quarter),
            'gdp_qoq': pd.to_numeric(df['gdp_qoq'].astype(str).str.replace(',', '').str.replace('..', ''), errors='coerce')
        })
        
        result = result.dropna().sort_values('date').reset_index(drop=True)
        return result
        
    except Exception as e:
        warnings.warn(f"Failed to load GDP data: {str(e)}")
        return pd.DataFrame(columns=['date', 'gdp_qoq'])


def load_production_index() -> pd.DataFrame:
    """
    Load production output index from Prod-series file.
    
    Returns:
        DataFrame with columns: date, prod_index
    """
    try:
        path = find_data_file("Prod-series-161225.csv")
        # Read without header first to find data start
        df_raw = pd.read_csv(path, header=None)
        
        # Find where data starts (look for quarter format like "1955 Q2")
        data_start_row = None
        for idx, row in df_raw.iterrows():
            first_val = str(row.iloc[0]).strip()
            if 'Q' in first_val.upper() and any(char.isdigit() for char in first_val):
                data_start_row = idx
                break
        
        if data_start_row is None:
            # Try ONS series parser as fallback
            return parse_ons_series(path)
        
        # Extract data starting from data_start_row
        df = df_raw.iloc[data_start_row:].copy()
        df.columns = ['date', 'prod_index']
        df = df.reset_index(drop=True)
        
        # Parse dates (format: "1955 Q2")
        def parse_quarter(date_str):
            if pd.isna(date_str):
                return pd.NaT
            date_str = str(date_str).strip()
            try:
                if 'Q' in date_str.upper():
                    year, q = date_str.upper().split('Q')
                    year = int(year.strip())
                    q = int(q.strip())
                    month = q * 3
                    return pd.Timestamp(year, month, 1) + pd.offsets.QuarterEnd()
                else:
                    return pd.to_datetime(date_str)
            except:
                return pd.NaT
        
        result = pd.DataFrame({
            'date': df['date'].apply(parse_quarter),
            'prod_index': pd.to_numeric(df['prod_index'].astype(str).str.replace(',', '').str.replace('..', ''), errors='coerce')
        })
        
        result = result.dropna().sort_values('date').reset_index(drop=True)
        return result
        
    except Exception as e:
        warnings.warn(f"Failed to load production data: {str(e)}")
        return pd.DataFrame(columns=['date', 'prod_index'])


def load_sector_output_from_prod_series_251125() -> pd.DataFrame:
    """
    Load sector-level production output indices from Prod-series-251125.xls.

    The file typically has:
    - A metadata block at the top
    - A table where the first column is a time period (e.g. '1990 Q1')
      and subsequent columns are sector indices (e.g. Manufacturing, Services).

    Returns:
        Long-format DataFrame with columns: date, sector, value
    """
    try:
        path = find_data_file("Prod-series-251125.xls")
        # Read without header first to detect where the table starts
        df_raw = read_excel_any(path)

        # Find header row by searching for a cell that looks like 'Time period' or 'Quarter'
        header_row = None
        for idx, row in df_raw.iterrows():
            row_str = " ".join(str(v) for v in row.values if pd.notna(v))
            if isinstance(row_str, str) and ("Time period" in row_str or "Quarter" in row_str):
                header_row = idx
                break

        if header_row is None:
            # Fallback: assume row 0 is header
            header_row = 0

        table_df = df_raw.iloc[header_row:].copy()
        table_df.columns = table_df.iloc[0]
        table_df = table_df.iloc[1:].reset_index(drop=True)

        # Identify date column (first column that looks like a period)
        date_col = None
        for col in table_df.columns:
            col_str = str(col).lower()
            if "time" in col_str or "period" in col_str or "quarter" in col_str or "date" in col_str:
                date_col = col
                break
        if date_col is None:
            date_col = table_df.columns[0]

        # Convert date column to quarter-end timestamps
        def parse_quarter(date_str):
            if pd.isna(date_str):
                return pd.NaT
            s = str(date_str).strip()
            try:
                if "Q" in s.upper():
                    year, q = s.upper().split("Q")
                    year = int(year.strip())
                    q = int(q.strip())
                    month = q * 3
                    return pd.Timestamp(year, month, 1) + pd.offsets.QuarterEnd()
                return pd.to_datetime(s)
            except Exception:
                return pd.NaT

        table_df[date_col] = table_df[date_col].apply(parse_quarter)
        table_df = table_df.dropna(subset=[date_col])

        # All other numeric columns are sectors
        sector_cols = [
            c
            for c in table_df.columns
            if c != date_col and table_df[c].notna().any()
        ]

        if not sector_cols:
            warnings.warn("No sector columns found in Prod-series-251125.xls")
            return pd.DataFrame(columns=["date", "sector", "value"])

        # Melt to long format
        long_df = table_df.melt(
            id_vars=[date_col],
            value_vars=sector_cols,
            var_name="sector",
            value_name="value",
        )

        long_df["value"] = pd.to_numeric(
            long_df["value"].astype(str).str.replace(",", "").str.replace("..", ""),
            errors="coerce",
        )
        long_df = long_df.dropna(subset=["value"])
        long_df = long_df.rename(columns={date_col: "date"})
        long_df = long_df.sort_values(["sector", "date"]).reset_index(drop=True)

        return long_df

    except Exception as e:
        warnings.warn(f"Failed to load sector output data from Prod-series-251125.xls: {e}")
        return pd.DataFrame(columns=["date", "sector", "value"])


def load_productivity() -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[str]]:
    """
    Load productivity data from lprod01.xls.
    Extracts headline measures from Table 1 (whole economy productivity).
    
    Based on workbook inspection:
    - Table 1 contains "Labour productivity key measures"
    - Row 3: Sector headers (Whole economy, Production, Manufacturing, Services)
    - Rows 4-5: Measure names (Output per worker, Output per job, Output per hour)
    - Row 7: Section codes (A-U, B-E, C, G-U)
    - Row 9: ONS codes
    - Row 10+: Years in column 1, values in columns 2-10
    
    Returns:
        Tuple of (headline_df, sector_df, error_message)
        headline_df: columns: date, measure, value (or None if error)
        sector_df: columns: date, sector, value (or None if not found or error)
        error_message: Error message string if loading failed, None if successful
    """
    headline_data = []
    
    try:
        # Step 1: Find file path
        path = get_data_path("lprod01.xls")
        abs_path = str(Path(path).resolve())
        print(f"\n[PRODUCTIVITY LOADER] Resolved file path: {abs_path}")
        
        # Step 2: Read Table 1 (the main productivity table)
        print(f"[PRODUCTIVITY LOADER] Reading Table 1...")
        df = pd.read_excel(path, sheet_name='Table 1', engine='xlrd', header=None)
        print(f"[PRODUCTIVITY LOADER] Table 1 shape: {df.shape}")
        
        # Step 3: Parse the known structure
        # Column 1 has years starting around row 10
        # Columns 2-10 have productivity values
        # The measure names are in rows 4-5
        
        # Find the header rows by looking for "Output per" pattern
        measure_row = None
        for idx in range(10):
            row_str = ' '.join([str(v).lower() for v in df.iloc[idx].values if pd.notna(v)])
            if 'output per' in row_str:
                measure_row = idx
                break
        
        if measure_row is None:
            # Fallback: assume row 4-5 based on inspection
            measure_row = 4
        
        print(f"[PRODUCTIVITY LOADER] Measure row detected at: {measure_row}")
        
        # Find where data starts (first row with a year in column 1)
        data_start = None
        year_col = 1  # Years are in column 1
        
        for idx in range(10, min(50, len(df))):
            cell_val = df.iloc[idx, year_col]
            if pd.notna(cell_val):
                try:
                    year = int(float(cell_val))
                    if 1900 <= year <= 2100:
                        data_start = idx
                        print(f"[PRODUCTIVITY LOADER] Data starts at row {data_start} with year {year}")
                        break
                except (ValueError, TypeError):
                    continue
        
        if data_start is None:
            raise ValueError("Could not find data start row with years")
        
        # Extract column headers (measure types)
        # Based on inspection: columns 2-10 contain different measures
        # Row 4: "Output per" labels
        # Row 5: "worker", "job", "hour" continuations
        
        # Build column mapping from rows 4-5
        col_measures = {}
        for col in range(2, min(11, df.shape[1])):
            # Get the sector from row 3
            sector = str(df.iloc[3, col]).strip() if pd.notna(df.iloc[3, col]) else ''
            # Get measure type from rows 4-5
            measure_part1 = str(df.iloc[4, col]).strip() if pd.notna(df.iloc[4, col]) else ''
            measure_part2 = str(df.iloc[5, col]).strip() if pd.notna(df.iloc[5, col]) else ''
            
            full_measure = f"{measure_part1} {measure_part2}".strip()
            
            # Map to standard measure names
            full_lower = full_measure.lower()
            if 'hour' in full_lower:
                measure_key = 'Output per hour'
            elif 'worker' in full_lower:
                measure_key = 'Output per worker'
            elif 'job' in full_lower:
                measure_key = 'Output per job'
            else:
                continue  # Skip unknown measures
            
            # Only take "Whole economy" columns (first set of each measure)
            sector_lower = sector.lower()
            if 'whole' in sector_lower or sector == '' or col <= 4:
                if measure_key not in col_measures:  # Take first occurrence
                    col_measures[col] = measure_key
                    print(f"[PRODUCTIVITY LOADER] Column {col}: {measure_key} (sector: {sector})")
        
        if not col_measures:
            raise ValueError("Could not identify any measure columns")
        
        # Extract data
        for col, measure in col_measures.items():
            series_data = []
            for idx in range(data_start, len(df)):
                year_val = df.iloc[idx, year_col]
                data_val = df.iloc[idx, col]
                
                # Parse year
                if pd.isna(year_val):
                    continue
                try:
                    year = int(float(year_val))
                    if not (1900 <= year <= 2100):
                        continue
                except (ValueError, TypeError):
                    continue
                
                # Parse value
                if pd.isna(data_val):
                    continue
                try:
                    # Handle "..", "-", blanks
                    val_str = str(data_val).strip()
                    if val_str in ['..', '-', '', 'nan']:
                        continue
                    val_str = val_str.replace(',', '')
                    value = float(val_str)
                except (ValueError, TypeError):
                    continue
                
                # Create date (use Jan 1 for annual data)
                date = pd.Timestamp(year=year, month=1, day=1)
                series_data.append({'date': date, 'measure': measure, 'value': value})
            
            if series_data:
                headline_data.append(pd.DataFrame(series_data))
                print(f"[PRODUCTIVITY LOADER] Extracted {len(series_data)} rows for {measure}")
        
        # Step 4: If no data from Table 1, try fallback
        if not headline_data:
            print(f"[PRODUCTIVITY LOADER] Table 1 parsing failed, trying fallback...")
            headline_data = _fallback_productivity_extraction(path)
        
        # Step 5: Combine and return
        if headline_data:
            headline_df = pd.concat(headline_data, ignore_index=True)
            headline_df = headline_df.sort_values(['measure', 'date']).reset_index(drop=True)
            
            # Remove duplicates (keep first)
            headline_df = headline_df.drop_duplicates(subset=['date', 'measure'], keep='first')
            
            print(f"\n[PRODUCTIVITY LOADER] ✓ Successfully loaded productivity data:")
            print(f"  - Total rows: {len(headline_df)}")
            print(f"  - Unique measures: {headline_df['measure'].unique().tolist()}")
            print(f"  - Date range: {headline_df['date'].min()} to {headline_df['date'].max()}")
            print(f"  - Sample data:\n{headline_df.head(10)}")
            
            return headline_df, None, None
        else:
            error_msg = "Could not extract any productivity measures from lprod01.xls"
            print(f"[PRODUCTIVITY LOADER] ERROR: {error_msg}")
            return None, None, error_msg
        
    except FileNotFoundError as e:
        error_msg = str(e)
        print(f"[PRODUCTIVITY LOADER] FILE NOT FOUND ERROR:")
        print(traceback.format_exc())
        return None, None, error_msg
    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}"
        print(f"[PRODUCTIVITY LOADER] ERROR loading productivity data:")
        print(traceback.format_exc())
        return None, None, error_msg


def _fallback_productivity_extraction(path: str) -> List[pd.DataFrame]:
    """
    Fallback extraction: scan all sheets for any time series with years.
    Returns a list of DataFrames with columns: date, measure, value
    """
    print(f"[PRODUCTIVITY LOADER] Running fallback extraction...")
    results = []
    
    try:
        excel_file = pd.ExcelFile(path, engine='xlrd')
        
        for sheet_name in excel_file.sheet_names:
            if sheet_name in ['Index', 'Revisions']:
                continue
            
            try:
                df = pd.read_excel(path, sheet_name=sheet_name, engine='xlrd', header=None)
                
                # Find a column with years
                year_col = None
                for col in range(min(3, df.shape[1])):
                    year_count = 0
                    for idx in range(min(100, len(df))):
                        try:
                            val = df.iloc[idx, col]
                            if pd.notna(val):
                                year = int(float(val))
                                if 1900 <= year <= 2100:
                                    year_count += 1
                        except:
                            pass
                    if year_count > 10:
                        year_col = col
                        break
                
                if year_col is None:
                    continue
                
                # Find first numeric column after year column
                value_col = None
                for col in range(year_col + 1, min(year_col + 5, df.shape[1])):
                    numeric_count = 0
                    for idx in range(10, min(100, len(df))):
                        try:
                            val = df.iloc[idx, col]
                            if pd.notna(val):
                                float(str(val).replace(',', ''))
                                numeric_count += 1
                        except:
                            pass
                    if numeric_count > 10:
                        value_col = col
                        break
                
                if value_col is None:
                    continue
                
                # Extract series
                series_data = []
                for idx in range(len(df)):
                    year_val = df.iloc[idx, year_col]
                    data_val = df.iloc[idx, value_col]
                    
                    try:
                        year = int(float(year_val))
                        if not (1900 <= year <= 2100):
                            continue
                        val_str = str(data_val).replace(',', '')
                        if val_str in ['..', '-', '', 'nan', 'None']:
                            continue
                        value = float(val_str)
                        date = pd.Timestamp(year=year, month=1, day=1)
                        series_data.append({'date': date, 'measure': 'Output per hour (fallback)', 'value': value})
                    except:
                        continue
                
                if len(series_data) > 10:
                    print(f"[PRODUCTIVITY LOADER] Fallback: Extracted {len(series_data)} rows from {sheet_name}")
                    results.append(pd.DataFrame(series_data))
                    break  # Use first valid sheet
                    
            except Exception as e:
                continue
    
    except Exception as e:
        print(f"[PRODUCTIVITY LOADER] Fallback failed: {e}")
    
    return results


def load_sector_productivity() -> pd.DataFrame:
    """
    Load sector-level productivity data from lprod01.xls.
    Extracts data from Tables 2, 3, 4 which contain manufacturing subsectors and services sectors.
    
    Returns:
        DataFrame with columns: date, sector, value (productivity index)
    """
    try:
        path = find_data_file("lprod01.xls")
        
        all_sector_data = []
        
        # Table 2: Manufacturing subsectors - Output per job
        # Table 3: Manufacturing subsectors - Output per hour
        # Table 4: Services sectors - Output per job
        
        tables_config = [
            ('Table 2', 'Manufacturing - Output per job', 12),  # data starts at row 12
            ('Table 3', 'Manufacturing - Output per hour', 12),
            ('Table 4', 'Services - Output per job', 12),
        ]
        
        for sheet_name, table_type, data_start_row in tables_config:
            try:
                df = pd.read_excel(path, sheet_name=sheet_name, engine='xlrd', header=None)
                
                # Get sector names from rows 3-7 (combined)
                # Row 7 has the section codes like "10-12", "13-15", etc.
                sector_row = 7
                sectors = []
                for col_idx in range(2, df.shape[1]):
                    # Combine rows 3-6 for full sector name
                    name_parts = []
                    for row_idx in range(3, 7):
                        val = df.iloc[row_idx, col_idx]
                        if pd.notna(val) and str(val).strip():
                            name_parts.append(str(val).strip())
                    sector_name = ' '.join(name_parts) if name_parts else f'Sector {col_idx}'
                    
                    # Add section code if available
                    section_code = df.iloc[sector_row, col_idx]
                    if pd.notna(section_code):
                        sector_name = f"{sector_name} ({section_code})"
                    
                    sectors.append(sector_name)
                
                # Extract data rows (year in column 1, values in columns 2+)
                for row_idx in range(data_start_row, df.shape[0]):
                    year_val = df.iloc[row_idx, 1]
                    if pd.isna(year_val):
                        continue
                    
                    # Parse year
                    try:
                        year = int(float(year_val))
                        if year < 1900 or year > 2100:
                            continue
                        date = pd.Timestamp(f'{year}-12-31')
                    except:
                        continue
                    
                    # Extract values for each sector
                    for col_idx, sector_name in enumerate(sectors, start=2):
                        if col_idx >= df.shape[1]:
                            break
                        val = df.iloc[row_idx, col_idx]
                        if pd.notna(val):
                            try:
                                value = float(val)
                                all_sector_data.append({
                                    'date': date,
                                    'sector': sector_name,
                                    'value': value,
                                    'table': table_type
                                })
                            except:
                                pass
                
            except Exception as e:
                warnings.warn(f"Error reading {sheet_name}: {str(e)}")
                continue
        
        if len(all_sector_data) == 0:
            warnings.warn("Could not extract sector productivity data from lprod01.xls")
            return pd.DataFrame(columns=['date', 'sector', 'value'])
        
        result_df = pd.DataFrame(all_sector_data)
        result_df = result_df.sort_values(['sector', 'date']).reset_index(drop=True)
        
        return result_df
        
    except Exception as e:
        warnings.warn(f"Error loading sector productivity data: {str(e)}")
        return pd.DataFrame(columns=['date', 'sector', 'value'])


def load_income_deciles() -> pd.DataFrame:
    """
    Load median income by nation from housing affordability workbook.
    Extracts from Table 2 sheet (sheet '2').
    Structure: Row 1 = header, Row 2+ = data with region code, region name, decile, then year columns.
    
    Returns:
        DataFrame with columns: year, region, median_income
    """
    try:
        path = find_data_file("Copy of housingpurchaseaffordabilityukbycountryandenglishregion2024.xlsx")
        
        # Read sheet '2' which is Table 2
        df = read_excel_any(path, sheet="2")
        
        # Region code mapping for UK nations
        region_mapping = {
            'E92000001': 'England',
            'W92000004': 'Wales', 
            'S92000003': 'Scotland',
            'N92000002': 'Northern Ireland'
        }
        
        result_data = []
        
        # Header is row 1 (index 1): 'Country/Region code', 'Country/Region name', 'Income decile', then years
        # Data starts at row 2 (index 2)
        header = df.iloc[1]
        data_df = df.iloc[2:].copy()
        data_df.columns = df.iloc[1].values  # Set proper column names
        
        # Find year columns (format: "1998/99", "1999/00", etc.)
        year_cols = []
        for col in data_df.columns:
            col_str = str(col)
            if '/' in col_str:
                try:
                    year_part = col_str.split('/')[0].strip()
                    if year_part.isdigit() and len(year_part) == 4:
                        year = int(year_part)
                        if 1990 <= year <= 2030:
                            year_cols.append((col, year))
                except:
                    pass
        
        # Filter for decile 5 (median = 50th percentile) and our target regions
        for region_code, region_name in region_mapping.items():
            # Find rows for this region with decile = "50th percentile"
            region_decile5 = data_df[
                (data_df['Country/Region code'].astype(str).str.strip() == region_code) &
                (data_df['Income decile'].astype(str).str.strip() == '50th percentile')
            ]
            
            if len(region_decile5) > 0:
                row = region_decile5.iloc[0]
                
                # Extract income for each year
                for col_name, year in year_cols:
                    if col_name in row.index:
                        income_val = pd.to_numeric(str(row[col_name]).replace(',', ''), errors='coerce')
                        if pd.notna(income_val) and income_val > 0:
                            result_data.append({
                                'year': pd.Timestamp(f'{year}-12-31'),
                                'region': region_name,
                                'median_income': income_val
                            })
        
        if len(result_data) == 0:
            warnings.warn("Could not extract income data from Table 2.")
            return pd.DataFrame(columns=['year', 'region', 'median_income'])
        
        result_df = pd.DataFrame(result_data)
        result_df = result_df.sort_values(['region', 'year']).reset_index(drop=True)
        
        return result_df
        
    except Exception as e:
        warnings.warn(f"Error loading income data: {str(e)}")
        return pd.DataFrame(columns=['year', 'region', 'median_income'])


def index_to_base(df: pd.DataFrame, date_col: str, value_col: str, base_period: str = '2007Q4') -> pd.DataFrame:
    """
    Index values to base period (base=100).
    
    Args:
        df: DataFrame with date and value columns
        date_col: Name of date column
        value_col: Name of value column
        base_period: Base period (e.g., '2007Q4')
        
    Returns:
        DataFrame with indexed values
    """
    df = df.copy()
    
    # Convert base_period to timestamp
    if isinstance(base_period, str):
        if 'Q' in base_period:
            year, q = base_period.split('Q')
            year = int(year)
            q = int(q)
            month = q * 3
            base_date = pd.Timestamp(year, month, 1) + pd.offsets.QuarterEnd()
        else:
            base_date = pd.Timestamp(base_period)
    else:
        base_date = pd.Timestamp(base_period)
    
    # Find base value
    df[date_col] = pd.to_datetime(df[date_col])
    base_row = df[df[date_col] == base_date]
    
    if len(base_row) == 0:
        # Find closest date
        base_row = df.iloc[(df[date_col] - base_date).abs().argsort()[:1]]
    
    if len(base_row) == 0:
        warnings.warn(f"Base period {base_period} not found. Using first available date.")
        base_value = df[value_col].iloc[0]
    else:
        base_value = base_row[value_col].iloc[0]
    
    if base_value == 0 or pd.isna(base_value):
        warnings.warn("Base value is zero or NaN. Cannot index.")
        df[f'{value_col}_indexed'] = df[value_col]
    else:
        df[f'{value_col}_indexed'] = (df[value_col] / base_value) * 100
    
    return df


def create_counterfactual(df: pd.DataFrame, date_col: str, value_col: str, 
                         pre_crisis_start: str = '1997Q1', pre_crisis_end: str = '2007Q4') -> pd.DataFrame:
    """
    Create counterfactual productivity series based on pre-crisis growth trend.
    
    Args:
        df: DataFrame with productivity data
        date_col: Name of date column
        value_col: Name of value column
        pre_crisis_start: Start of pre-crisis window
        pre_crisis_end: End of pre-crisis window (projection start)
        
    Returns:
        DataFrame with columns: date, actual, counterfactual, gap
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)
    
    # Detect if data is annual (all dates are Jan 1) or quarterly
    is_annual = all(df[date_col].dt.month == 1) and all(df[date_col].dt.day == 1)
    
    # Convert pre-crisis dates - handle both annual and quarterly
    def parse_quarter(q_str):
        year, q = q_str.split('Q')
        if is_annual:
            # For annual data, just use the year
            return pd.Timestamp(int(year), 1, 1)
        else:
            # For quarterly data, use quarter end
            month = int(q) * 3
            return pd.Timestamp(int(year), month, 1) + pd.offsets.QuarterEnd()
    
    start_date = parse_quarter(pre_crisis_start)
    end_date = parse_quarter(pre_crisis_end)
    
    # For annual data, use year boundaries
    if is_annual:
        start_year = int(pre_crisis_start.split('Q')[0])
        end_year = int(pre_crisis_end.split('Q')[0])
        start_date = pd.Timestamp(start_year, 1, 1)
        end_date = pd.Timestamp(end_year, 12, 31)  # Include full year
    
    # Extract pre-crisis data
    pre_crisis = df[(df[date_col] >= start_date) & (df[date_col] <= end_date)].copy()
    
    if len(pre_crisis) < 2:
        warnings.warn("Insufficient pre-crisis data. Using all available data before 2008.")
        # Use all data before 2008
        pre_crisis = df[df[date_col] < pd.Timestamp('2008-01-01')].copy()
        if len(pre_crisis) < 2:
            pre_crisis = df.copy()
        end_date = pre_crisis[date_col].max()
    
    # Calculate growth rate (in logs) - use years for annual data
    pre_crisis = pre_crisis.sort_values(date_col)
    pre_crisis['log_value'] = np.log(pre_crisis[value_col])
    
    if is_annual:
        # For annual data, use years as the time unit
        pre_crisis['time_diff'] = (pre_crisis[date_col] - pre_crisis[date_col].iloc[0]).dt.days / 365.25
    else:
        # For quarterly data, use quarters
        pre_crisis['time_diff'] = (pre_crisis[date_col] - pre_crisis[date_col].iloc[0]).dt.days / 90.25
    
    if len(pre_crisis) > 1:
        # Linear regression in log space
        from scipy import stats
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            pre_crisis['time_diff'], pre_crisis['log_value']
        )
        growth_rate = slope  # Annual or quarterly growth rate depending on data
    else:
        growth_rate = 0
    
    # Find projection start point (last pre-crisis date)
    projection_start_idx = df[df[date_col] <= end_date].index
    if len(projection_start_idx) == 0:
        projection_start_idx = 0
    else:
        projection_start_idx = projection_start_idx[-1]
    
    projection_start_value = df[value_col].iloc[projection_start_idx]
    projection_start_date = df[date_col].iloc[projection_start_idx]
    
    # Create counterfactual
    result = df.copy()
    result['actual'] = result[value_col]
    result['counterfactual'] = np.nan
    result['gap'] = np.nan
    
    # Fill pre-crisis with actual values
    result.loc[result[date_col] <= projection_start_date, 'counterfactual'] = result.loc[result[date_col] <= projection_start_date, 'actual']
    
    # Project forward using the appropriate time unit
    for idx in range(projection_start_idx + 1, len(result)):
        if is_annual:
            time_ahead = (result[date_col].iloc[idx] - projection_start_date).days / 365.25
        else:
            time_ahead = (result[date_col].iloc[idx] - projection_start_date).days / 90.25
        result.loc[result.index[idx], 'counterfactual'] = projection_start_value * np.exp(growth_rate * time_ahead)
    
    # Calculate gap
    result['gap'] = result['counterfactual'] - result['actual']
    
    # Rename date column to 'date' for consistency
    result = result.rename(columns={date_col: 'date'})
    return result[['date', 'actual', 'counterfactual', 'gap']]

