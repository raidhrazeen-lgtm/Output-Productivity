# UK Productivity Puzzle Dashboard

A production-ready Plotly Dash dashboard for analyzing UK productivity trends and the "productivity puzzle" - why UK productivity stalled after 2008.

## Features

- **Productivity Gap Analysis**: Visualizes actual productivity vs. pre-2008 trend projections
- **Growth Comparisons**: Pre vs. post-2008 productivity growth rates
- **Output Context**: GDP growth vs. productivity growth momentum
- **Living Standards**: Median household income by UK nation
- **Sector Analysis**: Heatmap of sector productivity (if available)

## Installation

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure data files are in the `/data` folder (or project root):
   - `lprod01.xls` (ONS labour productivity tables)
   - `GDP-series-161225 copy.csv` (GDP q/q growth %)
   - `Prod-series-161225.csv` (Total Production output index)
   - `Copy of housingpurchaseaffordabilityukbycountryandenglishregion2024.xlsx` (income deciles)

## Running the Dashboard

```bash
python app.py
```

The dashboard will be available at `http://127.0.0.1:8050`

## Project Structure

```
.
├── app.py              # Main Dash application
├── data_loader.py      # Data ingestion and cleaning
├── charts.py           # Chart building functions
├── styles.py           # Theme constants
├── requirements.txt    # Python dependencies
└── data/               # Data files directory
```

## Data Sources

- **GDP Data**: Quarterly GDP growth rates
- **Production Data**: Total production output index (CVM)
- **Productivity Data**: ONS labour productivity (output per hour/worker/job)
- **Income Data**: Median household income by UK nation (annual)

## Dashboard Tabs

1. **Overview**: Productivity gap chart, growth comparison, KPI cards, and sector heatmap
2. **Output Context**: GDP vs. productivity growth, production output index
3. **Living Standards**: Median income trends by nation
4. **Data Notes**: Data sources and methodology notes

## Controls

- **Productivity Measure**: Switch between output per hour/worker/job
- **Index to 2007Q4=100**: Toggle indexing of productivity data
- **Rolling 4Q Avg**: Toggle 4-quarter rolling average for growth charts
- **Date Range Slider**: Filter all quarterly charts by date range

## Notes

- The dashboard handles missing data gracefully with error banners
- All date alignment is to quarter-end timestamps
- Counterfactual productivity is calculated using pre-2008 trend (1997Q1-2007Q4)


