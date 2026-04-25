
import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import dataclass
import warnings


@dataclass
class MarketData:
    """Container for all market price data (variable periods)"""
    
    timestamps: pd.DatetimeIndex
    da_prices: np.ndarray  # €/MWh
    ida_bid: np.ndarray    # €/MWh
    ida_ask: np.ndarray    # €/MWh
    ida_mid: np.ndarray    # €/MWh
    fcr_prices: np.ndarray  # €/MW/h (variable elements)
    afrr_cap_pos: np.ndarray  # €/MW/h
    afrr_cap_neg: np.ndarray  # €/MW/h
    afrr_energy: np.ndarray   # €/MWh
    
    def __len__(self):
        return len(self.timestamps)
    
    def summary(self) -> str:
        spread = self.ida_ask - self.ida_bid
        return f"""
Market Data Summary
Days: {len(self) / 96:.1f} | Periods: {len(self):,}

Day-Ahead:   {np.mean(self.da_prices):6.2f} €/MWh  (range: {np.min(self.da_prices):.0f}-{np.max(self.da_prices):.0f})
Intraday:    {np.mean(self.ida_mid):6.2f} €/MWh  (spread: {np.mean(spread):.2f})
FCR:         {np.mean(self.fcr_prices):6.2f} €/MW/h (blocks: {len(np.unique(self.fcr_prices))})
aFRR Pos:    {np.mean(self.afrr_cap_pos):6.2f} €/MW/h
aFRR Neg:    {np.mean(self.afrr_cap_neg):6.2f} €/MW/h
REBAP:       {np.mean(self.afrr_energy):6.2f} €/MWh
"""


def load_market_data_year(forecast_file: str = "input data/Germany_priceforecast_25years.xlsx",
                         year: int = 2026,
                         verbose: bool = True) -> MarketData:

    forecast_path = Path(forecast_file)
    if not forecast_path.exists():
        raise FileNotFoundError(f"File not found: {forecast_file}")
    
    sheet_name = f'All_Forecasts_{year}'
    
    if verbose:
        print(f"Loading market data from {forecast_file} (sheet: {sheet_name})...")
    
    try:
        df = pd.read_excel(forecast_file, sheet_name=sheet_name)
    except ValueError as e:
        raise ValueError(f"Sheet '{sheet_name}' not found in {forecast_file}. "
                        f"Available sheets may include All_Forecasts_2026, All_Forecasts_2027, etc. "
                        f"Error: {e}")
    
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    timestamps = pd.DatetimeIndex(df['timestamp'])
    
    # Validate date range: must start at Jan 1 00:00:00 and end at Dec 31 23:45:00
    first_ts = timestamps[0]
    last_ts = timestamps[-1]
    
    if (first_ts.month != 1 or first_ts.day != 1 or 
        first_ts.hour != 0 or first_ts.minute != 0):
        raise ValueError(
            f"Data must start at YYYY-01-01 00:00:00, got {first_ts}"
        )
    
    if (last_ts.month != 12 or last_ts.day != 31 or 
        last_ts.hour != 23 or last_ts.minute != 45):
        raise ValueError(
            f"Data must end at YYYY-12-31 23:45:00, got {last_ts}"
        )
    
    # Extract prices
    da_prices = df['da_price_forecast'].values
    ida_mid = df['idc_mid_forecast'].values
    ida_bid = df['idc_bid'].values
    ida_ask = df['idc_ask'].values
    fcr_prices = df['fcr_p50_block'].values  # Variable length
    afrr_cap_pos = df['afrr_cap_pos_p50'].values
    afrr_cap_neg = df['afrr_cap_neg_p50'].values
    afrr_energy = df['rebap_p50'].values
    
    # Check for NaN
    for name, arr in [('DA', da_prices), ('IDA', ida_mid), ('FCR', fcr_prices)]:
        if np.any(np.isnan(arr)):
            raise ValueError(f"{name} contains NaN values")
    
    # Validate bid-ask spread
    spread = ida_ask - ida_bid
    if np.any(spread < 0):
        raise ValueError(f"Invalid IDA: ask < bid in {np.sum(spread < 0)} periods")
    
    if verbose:
        print(f"✓ Loaded {len(timestamps):,} periods ({timestamps[0]} to {timestamps[-1]})")
        print(f"✓ FCR array: {len(fcr_prices):,} elements (€/MW/h)")
        print(f"✓ Avg bid-ask spread: {np.mean(spread):.2f} €/MWh")
    
    return MarketData(
        timestamps=timestamps,
        da_prices=da_prices,
        ida_bid=ida_bid,
        ida_ask=ida_ask,
        ida_mid=ida_mid,
        fcr_prices=fcr_prices,
        afrr_cap_pos=afrr_cap_pos,
        afrr_cap_neg=afrr_cap_neg,
        afrr_energy=afrr_energy
    )


def load_all_market_years(forecast_file: str = "input data/Germany_priceforecast_25years.xlsx",
                         start_year: int = 2026,
                         num_years: int = 16,
                         verbose: bool = True) -> dict:

    forecast_path = Path(forecast_file)
    if not forecast_path.exists():
        raise FileNotFoundError(f"File not found: {forecast_file}")
    
    if verbose:
        print(f"\nLoading market data for {num_years} years from {forecast_file}...")
        print(f"  Years: {start_year} to {start_year + num_years - 1}")
    
    # Check which sheets are available
    xl = pd.ExcelFile(forecast_file)
    available_sheets = xl.sheet_names
    
    market_data_dict = {}
    
    for i in range(num_years):
        year = start_year + i
        sheet_name = f'All_Forecasts_{year}'
        
        if sheet_name not in available_sheets:
            if verbose:
                print(f"  ⚠️  Sheet '{sheet_name}' not found - skipping year {year}")
            continue
        
        try:
            if verbose:
                print(f"  Loading {sheet_name}...", end=' ')
            
            market_data = load_market_data_year(
                forecast_file=forecast_file,
                year=year,
                verbose=False  # Suppress individual year messages
            )
            
            market_data_dict[year] = market_data
            
            if verbose:
                print(f"✓ ({len(market_data):,} periods)")
                
        except Exception as e:
            if verbose:
                print(f"✗ Error: {e}")
            continue
    
    if verbose:
        print(f"\n✓ Successfully loaded {len(market_data_dict)} years of market data")
        print(f"  Available years: {sorted(market_data_dict.keys())}")
    
    return market_data_dict


def load_market_data_2026(forecast_file: str = "input data/Germany_priceforecast_25years.xlsx",
                          verbose: bool = True) -> MarketData:

    return load_market_data_year(forecast_file, year=2026, verbose=verbose)
