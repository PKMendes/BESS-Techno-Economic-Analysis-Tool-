# BESS Dispatch Optimizer & Financial Feasibility Analysis

Comprehensive battery energy storage system (BESS) dispatch optimization and financial modeling for German energy markets (FCR, aFRR, Day-Ahead, Intraday).

**Version:** 1.3  
**Last Updated:** April 2026

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [Installation](#installation)
4. [Quick Start](#quick-start)
5. [File Structure](#file-structure)
6. [Configuration](#configuration)
7. [Running the Analysis](#running-the-analysis)
8. [Output Files](#output-files)
9. [Key Concepts](#key-concepts)
10. [Troubleshooting](#troubleshooting)
11. [Technical Details](#technical-details)

---

##  Overview

This program performs **multi-year battery dispatch optimization** and **financial feasibility analysis** for utility-scale battery energy storage systems operating in German electricity markets.

### What It Does

1. **Optimizes** battery dispatch across 4 markets (FCR, aFRR, Day-Ahead, Intraday)
2. **Accounts for** battery degradation over warranty period (RTE, SOH, capacity fade)
3. **Calculates** financial metrics (NPV, IRR, payback period, profitability)
4. **Generates** comprehensive Excel reports with 15-minute granularity

### Key Innovations

- **Multi-year simulation** with year-specific degradation curves
- **Accurate cycle counting** using degraded capacity (not nominal)
- **Economic degradation model** (DOD-based with Peukert constant)
- **Buffer constraints** for frequency regulation services
- **Integrated financial modeling** with cash flow projections

---

## Features

### Optimization Features

- **Multi-market participation**: FCR, aFRR positive, aFRR negative, energy arbitrage
- **Block market allocation**: 4-hour FCR blocks, hourly aFRR capacity
- **Linear programming**: PuLP with CBC solver for daily optimization
- **Degradation tracking**: DOD-based degradation with economic penalties
- **SOH constraints**: Battery capacity limits adapt to degradation
- **Buffer management**: Ensures reserve delivery capability

### Financial Features

- **NPV calculation**: Net Present Value with configurable discount rate
- **IRR calculation**: Internal Rate of Return (using numpy-financial)
- **Payback period**: Simple and discounted payback calculations
- **Cash flow projection**: Year-by-year EBITDA, tax, depreciation
- **Multi-year comparison**: Operational metrics across warranty years

### Technical Features

- **Fast bulk loading**: Load all market years at once (~30 seconds)
- **Comprehensive validation**: Data quality checks, NaN detection, range validation
- **Detailed outputs**: 35 Excel sheets with summary and granular data
- **Error handling**: Graceful file handling, permission checks, auto-overwrite

---

## Installation

### Prerequisites

- **Python 3.8+** (tested on 3.9, 3.10, 3.11)
- **Operating System**: Linux, macOS, or Windows

### Step 1: Install Python Dependencies

```bash
pip install -r requirements.txt --break-system-packages
```

**Required packages:**
- `pandas>=2.0.0` - Data manipulation
- `numpy>=1.24.0` - Numerical computations
- `openpyxl>=3.1.0` - Excel file handling
- `PuLP>=2.7.0` - Linear programming
- `numpy-financial` - IRR calculation (optional)

### Step 2: Install CBC Solver (Optional but Recommended)

**Linux:**
```bash
sudo apt-get install coinor-cbc
```

**macOS:**
```bash
brew install coin-or-tools/coinor/cbc
```

**Windows:** CBC is included with PuLP.

### Step 3: Verify Installation

```bash
python -c "import pandas, numpy, pulp, openpyxl; print('✓ All dependencies installed')"
```

---

## Quick Start

### 1. Prepare Input Files

Place files in `input data/` directory:

- `Battery_Manufacturer_Data_Template_rev_pvm.xlsx` - Battery specifications
- `Germany_priceforecast_25years.xlsx` - Market price forecasts

### 2. Run Financial Feasibility Analysis

```bash
python bess_financial_feasibility.py
```

**Output:** `output/BESS_Financial_Feasibility.xlsx`

### 3. Review Results

Open Excel file to see:
- Executive Summary (NPV, IRR, payback)
- Cash Flow Projection
- Multi-Year Operational Summary
- Year-by-year details

---

## File Structure

```
project/
├── bess_dispatch_optimizer.py          # Core optimization engine
├── bess_financial_feasibility.py       # Financial analysis
├── bess_specifications.py              # Config loader
├── load_market_data.py                 # Market data loader
├── requirements.txt                    # Dependencies
├── README.md                           # This file
│
├── input data/
│   ├── Battery_Manufacturer_Data_Template_rev_pvm.xlsx
│   └── Germany_priceforecast_25years.xlsx
│
└── output/
    └── BESS_Financial_Feasibility.xlsx
```

---

## ⚙Configuration

### Battery Configuration

**File:** `Battery_Manufacturer_Data_Template_rev_pvm.xlsx`

**Key parameters:**

| Parameter | Sheet | Typical Value |
|-----------|-------|---------------|
| Nominal Capacity | BESS_specs | 100 MWh |
| Nominal Power | BESS_specs | 50 MW |
| SOC Min | BESS_specs | 10% |
| SOC Max | BESS_specs | 90% |
| CAPEX | BESS_specs | €40,000,000 |
| Max Cycles/Day | Lifetime_Parameters | 2.5 |
| Discount Rate | Financial_Parameters | 7% |

### Market Data

**File:** `Germany_priceforecast_25years.xlsx`

**Required sheets:** `All_Forecasts_2026`, `All_Forecasts_2027`, etc.

**Required columns:**
- `timestamp` - 15-minute intervals
- `da_price_forecast` - Day-ahead (€/MWh)
- `idc_mid_forecast` - Intraday mid (€/MWh)
- `fcr_p50_block` - FCR prices (€/MW/h)
- `afrr_cap_pos_p50` - aFRR pos (€/MW/h)
- `rebap_p50` - Energy settlement (€/MWh)

---

## Running the Analysis

### Full Financial Analysis

```bash
python bess_financial_feasibility.py
```

Duration: ~5-10 minutes for 16-year simulation

### Custom Analysis

```python
from bess_financial_feasibility import BESSFinancialFeasibility

analysis = BESSFinancialFeasibility(
    battery_config_file='input data/Battery_Manufacturer_Data_Template_rev_pvm.xlsx',
    market_data_file='input data/Germany_priceforecast_25years.xlsx'
)

analysis.run_multiyear_simulation(start_year=2026, num_warranty_years=16)
analysis.export_to_excel('output/results.xlsx')
```

---

##  Output Files

### Excel Report: BESS_Financial_Feasibility.xlsx

**Total sheets:** 35 (for 16-year simulation)

#### 1. Executive Summary

- NPV (Net Present Value)
- IRR (Internal Rate of Return)
- Payback Period
- Profitability Index

#### 2. Cash Flow Projection

Year-by-year:
- Gross Revenue
- OPEX
- EBITDA
- Depreciation
- Tax
- Free Cash Flow

#### 3. Multi-Year Operational Summary

Comparison across years:
- RTE, SOH, Capacity
- Revenue by market
- Cycles (FEC)
- Degradation costs

#### 4-35. Year_N_Summary & Details

- Summary: Aggregated metrics
- Details: 15-minute dispatch data

---

## Key Concepts

### Battery Degradation

**State of Health (SOH):**
- Remaining capacity as % of nominal
- Year 0: ~100%, Year 15: ~60%
- All calculations use: `capacity = nominal × SOH`

**Round-Trip Efficiency (RTE):**
- Energy loss during cycling
- Year 0: ~86%, Year 15: ~78%

### Cycle Counting

**Full Equivalent Cycles (FEC):**

## Troubleshooting

### File Not Found

**Solution:** Check files are in `input data/` directory with exact names

### Sheet Not Found

**Solution:** Verify market data file has `All_Forecasts_YYYY` sheets

### LP Solver Error

**Solution:** Install CBC solver:
```bash
sudo apt-get install coinor-cbc  # Linux
brew install coin-or-tools/coinor/cbc  # macOS
```

### Excel File Locked

**Solution:** Close Excel before running

### Negative NPV

**Not an error** - Project is unprofitable. Check:
- CAPEX too high?
- Market prices too low?
- OPEX too high?

---

## Technical Details

### Optimization Algorithm

- **Method:** Mixed-Integer Linear Programming
- **Solver:** CBC (COIN-OR)
- **Time horizon:** 24 hours
- **Variables/day:** ~480
- **Constraints/day:** ~770

### Performance

- Single year: ~30 seconds
- 16 years: ~8-10 minutes
- Memory: ~500 MB

### Data Validation

- No NaN values
- Realistic price ranges
- Positive bid-ask spreads
- Continuous timestamps
- Complete date coverage

---

##  Version History

### Version 1.3 (Current)
- Fixed FEC to use degraded capacity
- Removed sensitivity analysis
- Comprehensive README

### Version 1.2
- Auto file overwrite
- Better error handling

### Version 1.1
- Bulk market loading (16× faster)
- Fixed FCR prices

### Version 1.0
- Initial release

---

## Tips

### For Accurate Results

1. Use realistic degradation curves
2. Validate market forecasts
3. Conservative CAPEX estimates
4. Account for all OPEX
5. Appropriate discount rate (6-10%)

### For Better Performance

1. Use bulk loading: `load_all_market_years()`
2. Close Excel files before running
3. Install CBC solver
4. Sufficient RAM (4GB recommended)

---

## Quick Reference

### Essential Commands

```bash
# Install
pip install -r requirements.txt --break-system-packages

# Run
python bess_financial_feasibility.py

# Check
python -c "import pandas, numpy, pulp; print('✓')"
```

### Key Files

- `bess_financial_feasibility.py` - Main program
- `Battery_Manufacturer_Data_Template_rev_pvm.xlsx` - Config
- `Germany_priceforecast_25years.xlsx` - Market data

---

**End of README** 🎉
