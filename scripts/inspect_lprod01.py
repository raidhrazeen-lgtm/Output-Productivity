#!/usr/bin/env python3
"""
Inspector script for lprod01.xls
Analyzes the workbook structure to understand the real format for parsing.
"""

import pandas as pd
import re
import os
from pathlib import Path

def find_file(filename):
    """Find the data file in common locations."""
    paths = [
        Path('data') / filename,
        Path('.') / filename,
        Path('..') / 'data' / filename,
    ]
    for p in paths:
        if p.exists():
            return str(p.resolve())
    raise FileNotFoundError(f"Could not find {filename}")

def main():
    print("=" * 80)
    print("LPROD01.XLS WORKBOOK INSPECTOR")
    print("=" * 80)
    
    # Find and load file
    try:
        filepath = find_file("lprod01.xls")
        print(f"\n✓ File found: {filepath}")
    except FileNotFoundError as e:
        print(f"\n✗ {e}")
        return
    
    # Load all sheets
    print("\n" + "-" * 40)
    print("LOADING WORKBOOK")
    print("-" * 40)
    
    try:
        excel_file = pd.ExcelFile(filepath, engine='xlrd')
        sheet_names = excel_file.sheet_names
        print(f"✓ Loaded with xlrd engine")
        print(f"✓ Found {len(sheet_names)} sheets: {sheet_names}")
    except Exception as e:
        print(f"✗ Failed to load: {e}")
        return
    
    # Keywords to search for
    measure_keywords = [
        'output per hour', 'output per worker', 'output per job', 
        'output per filled job', 'productivity', 'whole economy'
    ]
    
    # Patterns for time periods
    quarter_pattern = re.compile(r'\b(19|20)\d{2}\s*Q[1-4]\b', re.IGNORECASE)
    year_pattern = re.compile(r'\b(19|20)\d{2}\b')
    time_period_keywords = ['time period', 'quarter', 'year', 'period']
    
    # Store all keyword matches
    all_matches = []
    time_matches = []
    
    # Inspect each sheet
    for sheet_name in sheet_names:
        print(f"\n" + "=" * 80)
        print(f"SHEET: {sheet_name}")
        print("=" * 80)
        
        try:
            # Read without header to get raw data
            df = pd.read_excel(filepath, sheet_name=sheet_name, engine='xlrd', header=None)
            print(f"Shape: {df.shape[0]} rows × {df.shape[1]} cols")
            
            # Show first 25 rows × first 12 cols
            print(f"\nFirst 25 rows × 12 cols:")
            print("-" * 60)
            display_df = df.iloc[:25, :12].copy()
            # Convert to strings for display
            for col in display_df.columns:
                display_df[col] = display_df[col].astype(str).str[:30]  # Truncate long values
            print(display_df.to_string(index=True, header=True))
            
            # Search for measure keywords
            print(f"\nKeyword matches in this sheet:")
            print("-" * 40)
            found_in_sheet = False
            for idx, row in df.iterrows():
                for col_idx, cell in enumerate(row):
                    cell_str = str(cell).lower() if pd.notna(cell) else ''
                    for keyword in measure_keywords:
                        if keyword in cell_str:
                            match_info = {
                                'sheet': sheet_name,
                                'row': idx,
                                'col': col_idx,
                                'keyword': keyword,
                                'cell_text': str(cell)[:80]
                            }
                            all_matches.append(match_info)
                            print(f"  [{idx}, {col_idx}] '{keyword}' → \"{str(cell)[:60]}\"")
                            found_in_sheet = True
            
            if not found_in_sheet:
                print("  (no measure keywords found)")
            
            # Search for time period patterns
            print(f"\nTime period patterns:")
            print("-" * 40)
            time_found = False
            for idx, row in df.iterrows():
                for col_idx, cell in enumerate(row):
                    cell_str = str(cell) if pd.notna(cell) else ''
                    cell_lower = cell_str.lower()
                    
                    # Check for time period keywords
                    for tp_kw in time_period_keywords:
                        if tp_kw in cell_lower:
                            time_matches.append({
                                'sheet': sheet_name,
                                'row': idx,
                                'col': col_idx,
                                'type': 'keyword',
                                'text': cell_str[:50]
                            })
                            if not time_found:
                                print(f"  [{idx}, {col_idx}] Time keyword: \"{cell_str[:50]}\"")
                            time_found = True
                    
                    # Check for quarter pattern
                    if quarter_pattern.search(cell_str):
                        time_matches.append({
                            'sheet': sheet_name,
                            'row': idx,
                            'col': col_idx,
                            'type': 'quarter',
                            'text': cell_str[:50]
                        })
                        if idx < 30:  # Only print first few
                            print(f"  [{idx}, {col_idx}] Quarter: \"{cell_str[:50]}\"")
                        time_found = True
                    
                    # Check for year pattern (but not if it's part of a quarter)
                    elif year_pattern.search(cell_str) and 'Q' not in cell_str.upper():
                        if len(cell_str.strip()) == 4:  # Standalone year
                            time_matches.append({
                                'sheet': sheet_name,
                                'row': idx,
                                'col': col_idx,
                                'type': 'year',
                                'text': cell_str[:50]
                            })
                            if idx < 30:
                                print(f"  [{idx}, {col_idx}] Year: \"{cell_str[:50]}\"")
                            time_found = True
            
            if not time_found:
                print("  (no time period patterns found)")
                
        except Exception as e:
            print(f"✗ Error reading sheet: {e}")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    print(f"\nTotal measure keyword matches: {len(all_matches)}")
    if all_matches:
        print("Matches by sheet:")
        sheets_with_matches = {}
        for m in all_matches:
            sheet = m['sheet']
            if sheet not in sheets_with_matches:
                sheets_with_matches[sheet] = []
            sheets_with_matches[sheet].append(m)
        
        for sheet, matches in sheets_with_matches.items():
            print(f"  {sheet}: {len(matches)} matches")
            for m in matches[:5]:  # Show first 5
                print(f"    - [{m['row']},{m['col']}] '{m['keyword']}': {m['cell_text'][:50]}")
    
    print(f"\nTotal time period matches: {len(time_matches)}")
    time_sheets = set(m['sheet'] for m in time_matches)
    print(f"Sheets with time periods: {list(time_sheets)}")
    
    # Recommendation
    print("\n" + "-" * 40)
    print("RECOMMENDED PARSING STRATEGY:")
    print("-" * 40)
    
    if all_matches:
        # Find sheets with both measures and time periods
        measure_sheets = set(m['sheet'] for m in all_matches)
        best_sheets = measure_sheets.intersection(time_sheets)
        if best_sheets:
            print(f"Best candidate sheets (have both measures + time): {list(best_sheets)}")
        else:
            print(f"Sheets with measures: {list(measure_sheets)}")
            print(f"Sheets with time: {list(time_sheets)}")
    else:
        print("No measure keywords found. May need to inspect raw numeric data.")

if __name__ == '__main__':
    main()

