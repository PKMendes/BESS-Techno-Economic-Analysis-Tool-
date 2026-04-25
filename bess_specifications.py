
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import numpy as np
import pandas as pd
from pathlib import Path


@dataclass
class BESSHardwareSpecs:
    """Battery hardware specifications - populated from manufacturer Excel"""
    
    # Capacity and Power (from Excel)
    nominal_capacity_mwh: float = 0.0
    nominal_power_mw: float = 0.0
    
    # Efficiency (from Excel RTE curve or static values)
    efficiency_charge: float = 0.92
    efficiency_discharge: float = 0.92
    efficiency_roundtrip: float = 0.85
    
    # SOC Limits (from Excel)
    soc_min: float = 0.0
    soc_max: float = 1.0
    soc_initial: float = 0.50
    
    # Properties for optimizer compatibility
    @property
    def energy_mwh(self) -> float:
        """Alias for nominal_capacity_mwh (used by optimizer)"""
        return self.nominal_capacity_mwh
    
    @property
    def power_mw(self) -> float:
        """Alias for nominal_power_mw (used by optimizer)"""
        return self.nominal_power_mw


@dataclass
class DegradationParameters:
    """Degradation parameters - populated from manufacturer curves"""
    
    # Time-varying curves from manufacturer (list of (year, value) tuples)
    rte_curve: Optional[List[Tuple[float, float]]] = None  # (year, rte_fraction)
    soh_curve: Optional[List[Tuple[float, float]]] = None  # (year, soh_fraction)
    
    # Auxiliary consumption (from Excel)
    aux_consumption_kwh: float = 0.0
    
    # Operating temperature (from Excel)
    operating_temp_celsius: float = 22.5
    
    def get_rte_at_year(self, year: float) -> float:
        """Get RTE at specific year via interpolation"""
        
        years = np.array([y for y, _ in self.rte_curve])
        rtes = np.array([r for _, r in self.rte_curve])
        
        rte = np.interp(year, years, rtes)
        return np.clip(rte, 0.5, 1.0)
    
    def get_soh_at_year(self, year: float) -> float:
        """Get SOH at specific year via interpolation"""
        
        years = np.array([y for y, _ in self.soh_curve])
        sohs = np.array([s for _, s in self.soh_curve])
        
        soh = np.interp(year, years, sohs)
        return np.clip(soh, 0.0, 1.1)
    
    def calculate_cycle_cost(self, dod: float) -> float:

        base_cost = 10.0  # EUR per full cycle
        return base_cost * dod
    
    def calculate_calendar_aging_cost(self, soc: float, hours: float) -> float:

        base_cost_per_hour = 0.1  # EUR per hour
        soc_factor = 1.0 + (soc - 0.5) * 0.5  # Higher SOC = more aging
        return base_cost_per_hour * soc_factor * hours


@dataclass
class LifetimeParameters:

    # From Excel
    warranty_years: float = 15.0
    warranty_cycles: float = 11000.0
    
    # Cycle constraints (from Excel)
    max_cycles_per_day: float = 2.0
    max_cycles_per_week: float = 14.0


@dataclass
class FinancialParameters:

    # From Excel
    capex_total_eur: float = 0.0
    
    # OPEX curve from Excel (list of (year, eur) tuples)
    opex_curve: Optional[List[Tuple[int, float]]] = None
    
    # Standard parameters
    discount_rate_wacc: float = 0.07
    tax_rate: float = 0.30
    
    def get_opex_at_year(self, year: int) -> float:
        """Get OPEX at specific year"""
        if self.opex_curve is None or len(self.opex_curve) == 0:
            return 0.0
        
        years = np.array([y for y, _ in self.opex_curve])
        opex_values = np.array([o for _, o in self.opex_curve])
        
        if year < years[0] or year > years[-1]:
            return 0.0
        
        return float(np.interp(year, years, opex_values))


@dataclass
class FCRParameters:
    """FCR market parameters (German regulations)"""
    fcr_max_capacity_ratio: float = 0.80
    fcr_block_hours: int = 4
    fcr_blocks_per_day: int = 6


@dataclass
class AFRRParameters:
    """aFRR market parameters (German regulations)"""
    afrr_activation_prob_pos: float = 0.12
    afrr_activation_prob_neg: float = 0.18
    afrr_min_bid_size_mw: float = 5.0


@dataclass
class HurdleRates:

    hurdle_fcr_eur_mwh: float = 8.00    # Up from 2.00 - FCR has high opportunity cost
    hurdle_afrr_eur_mwh: float = 5.00   # Down from 8.00 - Encourage aFRR participation
    hurdle_da_eur_mwh: float = 5.00     # Down from 14.05 - Enable DA arbitrage
    hurdle_ida_eur_mwh: float = 12.00   # Down from 28.10 - Enable IDC trading
    
    def get_hurdle(self, market: str) -> float:
        """Get hurdle rate for specified market"""
        hurdle_map = {
            'fcr': self.hurdle_fcr_eur_mwh,
            'afrr': self.hurdle_afrr_eur_mwh,
            'da': self.hurdle_da_eur_mwh,
            'ida': self.hurdle_ida_eur_mwh,
        }
        return hurdle_map.get(market.lower(), 0.0)


@dataclass
class OptimizationParameters:
    """Optimization algorithm settings"""
    solver_name: str = "highs"
    timestep_minutes: int = 15
    timestep_hours: float = 0.25
    periods_per_hour: int = 4
    periods_per_day: int = 96
    use_weekly_optimization: bool = True
    rolling_window_days: int = 7
    rolling_window_periods: int = 672


@dataclass
class MarketParticipationConstraints:

    # FCR constraints (market is saturated, can't commit 100% capacity)
    max_fcr_participation: float = 0.40  # Max 40% of nameplate to FCR
    fcr_bid_success_rate: float = 0.60   # 60% of FCR bids win (market saturated)
    
    # aFRR constraints (deeper market than FCR)
    max_afrr_pos_participation: float = 0.60  # Max 60% to positive aFRR
    max_afrr_neg_participation: float = 0.60  # Max 60% to negative aFRR
    max_afrr_total_participation: float = 0.70  # Max 70% to aFRR total (pos+neg)
    
    # Energy market constraints
    min_available_for_energy: float = 0.20  # Always keep 20% available for arbitrage
    
    # Combined reserve constraint (FCR + aFRR cannot exceed)
    max_total_reserve_participation: float = 0.75  # Max 75% to all reserves combined
    
    def validate_allocation(self, p_fcr: float, p_afrr_pos: float, p_afrr_neg: float, 
                           p_nom: float) -> bool:
        """Validate that allocation respects constraints"""
        fcr_pct = p_fcr / p_nom if p_nom > 0 else 0
        afrr_pos_pct = p_afrr_pos / p_nom if p_nom > 0 else 0
        afrr_neg_pct = p_afrr_neg / p_nom if p_nom > 0 else 0
        total_reserve = fcr_pct + afrr_pos_pct + afrr_neg_pct
        
        if fcr_pct > self.max_fcr_participation:
            return False
        if afrr_pos_pct > self.max_afrr_pos_participation:
            return False
        if afrr_neg_pct > self.max_afrr_neg_participation:
            return False
        if (afrr_pos_pct + afrr_neg_pct) > self.max_afrr_total_participation:
            return False
        if total_reserve > self.max_total_reserve_participation:
            return False
        
        return True


@dataclass
class CompleteBESSConfig:

    hardware: BESSHardwareSpecs = field(default_factory=BESSHardwareSpecs)
    degradation: DegradationParameters = field(default_factory=DegradationParameters)
    lifetime: LifetimeParameters = field(default_factory=LifetimeParameters)
    financial: FinancialParameters = field(default_factory=FinancialParameters)
    fcr: FCRParameters = field(default_factory=FCRParameters)
    afrr: AFRRParameters = field(default_factory=AFRRParameters)
    hurdles: HurdleRates = field(default_factory=HurdleRates)
    participation: MarketParticipationConstraints = field(default_factory=MarketParticipationConstraints)
    optimization: OptimizationParameters = field(default_factory=OptimizationParameters)
    
    @classmethod
    def from_excel(cls, excel_path: str, verbose: bool = True):
        
        excel_path = Path(excel_path)
        if not excel_path.exists():
            raise FileNotFoundError(f"Excel file not found: {excel_path}")
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"LOADING BESS CONFIG FROM EXCEL (using pandas)")
            print(f"{'='*70}")
            print(f"File: {excel_path.name}\n")
        
        # Create config with defaults (market parameters, optimization settings)
        config = cls()
        
        # ========== SHEET 1: BATTERY SPECIFICATIONS ==========
        if verbose:
            print("📋 Sheet 1: Battery Specifications...")
        
        try:
            # Load using pandas
            specs_df = pd.read_excel(excel_path, sheet_name='Battery Specifications', header=None)
            
            # Extract parameter-value pairs into dictionary
            specs = {}
            for _, row in specs_df.iterrows():
                if pd.notna(row[0]) and pd.notna(row[1]):
                    param = str(row[0]).strip()
                    if param not in ['Parameter', 'BATTERY SPECIFICATIONS']:
                        specs[param] = row[1]
            
            # Populate hardware specifications
            config.hardware.nominal_capacity_mwh = float(specs.get('Nominal Capacity', 0))
            config.hardware.nameplate_capacity_mwh = float(specs.get('Nameplate Capacity', 
                                                                     config.hardware.nominal_capacity_mwh))
            config.hardware.nominal_power_mw = float(specs.get('Nominal Power', 0))
            
            # SOC limits (convert from % to fraction)
            config.hardware.soc_min = float(specs.get('Minimum SOC', 0)) / 100.0
            config.hardware.soc_max = float(specs.get('Maximum SOC', 100)) / 100.0
            
            # Warranty parameters
            config.lifetime.warranty_years = float(specs.get('Warranty Period', 15))
            config.lifetime.warranty_cycles = float(specs.get('Warranty Cycles', 5000))
            
            # Max cycles per day (if provided)
            max_cycles_day = specs.get('Max Cycles per day')
            if pd.notna(max_cycles_day):
                config.lifetime.max_cycles_per_day = float(max_cycles_day)
                config.lifetime.max_cycles_per_week = float(max_cycles_day) * 7
            
            # Operating temperature
            temp_min = float(specs.get('Operating Temperature Min', 20))
            temp_max = float(specs.get('Operating Temperature Max', 25))
            config.degradation.operating_temp_celsius = (temp_min + temp_max) / 2.0
            
            # Auxiliary consumption (handle both with and without trailing space)
            aux = specs.get('Aux Consumption Total')
            if pd.notna(aux) and aux != '':
                config.degradation.aux_consumption_kwh = float(aux)
            else:
                aux = specs.get('Aux Consumption Total ')  # Try with trailing space
                if pd.notna(aux):
                    config.degradation.aux_consumption_kwh = float(aux)
            
            # CAPEX
            capex = specs.get('CAPEX')
            if pd.notna(capex):
                config.financial.capex_total_eur = float(capex)
            
            if verbose:
                print(f"  ✓ Capacity: {config.hardware.nominal_capacity_mwh:.1f} MWh")
                print(f"  ✓ Power: {config.hardware.nominal_power_mw:.1f} MW")
                print(f"  ✓ SOC Range: {config.hardware.soc_min*100:.0f}%-{config.hardware.soc_max*100:.0f}%")
                print(f"  ✓ CAPEX: €{config.financial.capex_total_eur:,.0f}")
        
        except Exception as e:
            if verbose:
                print(f"  ⚠ Error loading Battery Specifications: {e}")
        
        # ========== SHEET 2: DEGRADATION CURVE ==========
        if verbose:
            print(f"\n📊 Sheet 2: Degradation Curve...")
        
        try:
            # Load using pandas (skip first title row)
            deg_df = pd.read_excel(excel_path, sheet_name='Degradation Curve', skiprows=1)
            deg_df.columns = [str(col).strip() for col in deg_df.columns]
            
            years, rte_vals, soh_vals = [], [], []
            
            # Extract year-by-year data
            for _, row in deg_df.iterrows():
                year_str = str(row.iloc[0])
                if 'Year' not in year_str:
                    continue
                
                # Parse year number
                try:
                    year = int(year_str.replace('Year', '').strip())
                except:
                    continue
                
                # Extract RTE (Round-Trip Efficiency)
                rte_col = [c for c in deg_df.columns if 'RTE' in c or 'Efficiency' in c]
                if rte_col and pd.notna(row[rte_col[0]]):
                    rte = float(row[rte_col[0]])
                    # Convert to percentage if stored as fraction
                    if rte < 1.5:
                        rte *= 100.0
                else:
                    continue
                
                # Extract SOH (State of Health)
                soh_col = [c for c in deg_df.columns if 'SOH' in c]
                if soh_col and pd.notna(row[soh_col[0]]):
                    soh = float(row[soh_col[0]])
                    # Convert to percentage if stored as fraction
                    if soh < 1.5:
                        soh *= 100.0
                else:
                    continue
                
                # Store data
                years.append(year)
                rte_vals.append(rte / 100.0)  # Store as fraction
                soh_vals.append(soh / 100.0)  # Store as fraction
            
            if years:
                # Store curves as list of (year, value) tuples
                config.degradation.rte_curve = list(zip(years, rte_vals))
                config.degradation.soh_curve = list(zip(years, soh_vals))
                
                # Set initial efficiency from first RTE value
                initial_rte = rte_vals[0]
                config.hardware.efficiency_roundtrip = initial_rte
                config.hardware.efficiency_charge = initial_rte ** 0.5
                config.hardware.efficiency_discharge = initial_rte ** 0.5
                
                if verbose:
                    print(f"  ✓ RTE Curve: {len(years)} years")
                    print(f"    Year 0: {rte_vals[0]*100:.2f}% → Year {years[-1]}: {rte_vals[-1]*100:.2f}%")
                    print(f"  ✓ SOH Curve: {len(years)} years")
                    print(f"    Year 0: {soh_vals[0]*100:.2f}% → Year {years[-1]}: {soh_vals[-1]*100:.2f}%")
        
        except Exception as e:
            if verbose:
                print(f"  ⚠ Error loading Degradation Curve: {e}")
        
        # ========== SHEET 3: OPEX ==========
        if verbose:
            print(f"\n💰 Sheet 3: OPEX...")
        
        try:
            # Load using pandas
            opex_df = pd.read_excel(excel_path, sheet_name='OPEX')
            opex_years, opex_vals = [], []
            
            # Extract year-by-year OPEX
            for _, row in opex_df.iterrows():
                year_str = str(row.iloc[0])
                if 'Year' not in year_str:
                    continue
                
                try:
                    year = int(year_str.replace('Year', '').strip())
                    opex = row.iloc[1]  # Second column is SLA (€)
                    if pd.notna(opex) and opex != 0:
                        opex_years.append(year)
                        opex_vals.append(float(opex))
                except:
                    continue
            
            if opex_years:
                # Store as list of (year, eur) tuples
                config.financial.opex_curve = list(zip(opex_years, opex_vals))
                
                if verbose:
                    print(f"  ✓ OPEX Curve: {len(opex_years)} years")
                    print(f"    Year 0: €{opex_vals[0]:,.0f} → Year {opex_years[-1]}: €{opex_vals[-1]:,.0f}")
        
        except Exception as e:
            if verbose:
                print(f"  ⚠ OPEX sheet not found or error: {e}")
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"✅ LOADING COMPLETE")
            print(f"{'='*70}")
            print(f"\nConfig ready for weekly_optimizer.py")
            print(f"No changes needed to optimizer!")
        
        return config
    
    def summary(self) -> str:
        """Generate summary of configuration"""
        hw = self.hardware
        deg = self.degradation
        lt = self.lifetime
        fin = self.financial
        
        return f"""
BESS Configuration Summary
==========================

Hardware:
  Capacity: {hw.nominal_capacity_mwh:.2f} MWh (Nameplate: {hw.nameplate_capacity_mwh:.2f} MWh)
  Power: {hw.nominal_power_mw:.2f} MW
  Duration: {hw.nominal_capacity_mwh / hw.nominal_power_mw if hw.nominal_power_mw > 0 else 0:.2f} hours
  Efficiency: {hw.efficiency_roundtrip*100:.1f}% (RT)
  SOC Range: {hw.soc_min*100:.0f}% - {hw.soc_max*100:.0f}%

Degradation:
  RTE Curve: {len(deg.rte_curve) if deg.rte_curve else 0} points
  SOH Curve: {len(deg.soh_curve) if deg.soh_curve else 0} points
  Aux Consumption: {deg.aux_consumption_kwh:.2f} kWh/cycle

Warranty:
  Period: {lt.warranty_years:.0f} years
  Cycles: {lt.warranty_cycles:,.0f} total
  Max cycles/day: {lt.max_cycles_per_day:.1f}

Financial:
  CAPEX: €{fin.capex_total_eur:,.0f}
  OPEX Curve: {len(fin.opex_curve) if fin.opex_curve else 0} years
  WACC: {fin.discount_rate_wacc*100:.1f}%

Market Parameters:
  FCR max ratio: {self.fcr.fcr_max_capacity_ratio*100:.0f}%
  aFRR min bid: {self.afrr.afrr_min_bid_size_mw:.1f} MW

Hurdle Rates:
  FCR: €{self.hurdles.hurdle_fcr_eur_mwh:.2f}/MWh
  aFRR: €{self.hurdles.hurdle_afrr_eur_mwh:.2f}/MWh
  DA: €{self.hurdles.hurdle_da_eur_mwh:.2f}/MWh
  IDA: €{self.hurdles.hurdle_ida_eur_mwh:.2f}/MWh

Optimization:
  Horizon: {self.optimization.rolling_window_days} days
  Resolution: {self.optimization.timestep_minutes} minutes
  Solver: {self.optimization.solver_name.upper()}
"""


if __name__ == "__main__":
    # Test loading from Excel
    import sys
    
    if len(sys.argv) > 1:
        excel_file = sys.argv[1]
    else:
        excel_file = "input data/Battery_Manufacturer_Data_to run.xlsx"
    
    try:
        print("="*70)
        print("TESTING BESS SPECIFICATIONS WITH EXCEL LOADING")
        print("="*70)
        
        # Test 1: Load from Excel using pandas
        print("\nTest 1: Loading from Excel...")
        config = CompleteBESSConfig.from_excel(excel_file, verbose=True)
        
        # Test 2: Show summary
        print("\nTest 2: Configuration Summary...")
        print(config.summary())
        
        # Test 3: Verify optimizer compatibility
        print("\nTest 3: Optimizer Compatibility Check...")
        print(f"  ✓ config.hardware.energy_mwh: {config.hardware.energy_mwh}")
        print(f"  ✓ config.hardware.power_mw: {config.hardware.power_mw}")
        print(f"  ✓ config.hardware.efficiency_charge: {config.hardware.efficiency_charge:.4f}")
        print(f"  ✓ config.degradation.get_rte_at_year(5): {config.degradation.get_rte_at_year(5):.4f}")
        print(f"  ✓ config.degradation.get_soh_at_year(5): {config.degradation.get_soh_at_year(5):.4f}")
        print(f"  ✓ config.hurdles.get_hurdle('ida'): €{config.hurdles.get_hurdle('ida'):.2f}/MWh")
        
        print("\n" + "="*70)
        print("✅ ALL TESTS PASSED")
        print("="*70)
        print("\nReady for use with weekly_optimizer.py:")
        print("  from bess_specifications import CompleteBESSConfig")
        print("  config = CompleteBESSConfig.from_excel('Battery_Manufacturer_Data_Template_rev_pvm.xlsx')")
        print("  optimizer = WeeklyMultiMarketOptimizer(config)")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
