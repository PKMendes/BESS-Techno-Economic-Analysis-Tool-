
import sys
from pathlib import Path
sys.path.insert(0, '/mnt/user-data/uploads')

import numpy as np
import pandas as pd
from pulp import LpProblem, LpVariable, lpSum, LpMaximize, PULP_CBC_CMD
from dataclasses import dataclass
from typing import Tuple, Dict, List
import warnings
warnings.filterwarnings('ignore')

# Import user modules
from load_market_data import load_market_data_2026, MarketData
from bess_specifications import CompleteBESSConfig


@dataclass
class ThresholdParameters:
    """Dynamic thresholds calculated from annual data - 3-TIER ALLOCATION"""
    da_low_percentile: float
    da_high_percentile: float
    da_mean: float
    da_std: float
    ida_low_percentile: float
    ida_high_percentile: float
    ida_mean: float
    ida_spread_mean: float
    # 3-tier thresholds: HIGH and MEDIUM (no LOW needed - defaults to 0)
    fcr_threshold_high_eur_mw_4h: np.ndarray  # Per block - for 40% allocation
    fcr_threshold_medium_eur_mw_4h: np.ndarray  # Per block - for 20% allocation
    afrr_pos_threshold_high_eur_mw_4h: np.ndarray  # Per block - for 25% allocation
    afrr_pos_threshold_medium_eur_mw_4h: np.ndarray  # Per block - for 12.5% allocation
    afrr_neg_threshold_high_eur_mw_4h: np.ndarray  # Per block - for 25% allocation
    afrr_neg_threshold_medium_eur_mw_4h: np.ndarray  # Per block - for 12.5% allocation


class BESSDispatchOptimizer:
    """Multi-market BESS dispatch optimizer"""
    
    def __init__(self, config: CompleteBESSConfig, market_data: MarketData, warranty_year: int = 0):

        self.config = config
        self.market = market_data
        self.hw = config.hardware
        self.fcr_params = config.fcr
        self.afrr_params = config.afrr
        self.hurdles = config.hurdles
        self.warranty_year = warranty_year  # Store warranty year for RTE/SOH lookup
        
        # Validate configuration
        if self.hw.nominal_capacity_mwh == 0 or self.hw.nominal_capacity_mwh is None:
            raise ValueError(
                "Battery capacity is zero or None! "
                "Make sure to load config using: "
                "config = CompleteBESSConfig.from_excel('Battery_Manufacturer_Data_Template_rev_pvm.xlsx')"
            )
        
        if self.hw.nominal_power_mw == 0 or self.hw.nominal_power_mw is None:
            raise ValueError(
                "Battery power is zero or None! "
                "Make sure to load config using: "
                "config = CompleteBESSConfig.from_excel('Battery_Manufacturer_Data_Template_rev_pvm.xlsx')"
            )
        
        # Calculate thresholds
        self.thresholds = self._calculate_annual_thresholds()
        
        # Results storage
        self.results = []
        
        # Check if RTE/SOH curves are available
        has_curves = (hasattr(self.config, 'degradation') and 
                     self.config.degradation.rte_curve is not None and 
                     self.config.degradation.soh_curve is not None)
        
        print(f"\n🔋 BESS Dispatch Optimizer v5.0")
        print(f"  Nominal Capacity: {self.hw.nominal_capacity_mwh:.1f} MWh")
        print(f"  Nominal Power: {self.hw.nominal_power_mw:.1f} MW")
        print(f"  Battery Duration: {self.hw.nominal_capacity_mwh / self.hw.nominal_power_mw:.1f} hours")
        if has_curves:
            rte_0 = self.config.degradation.get_rte_at_year(0)
            soh_0 = self.config.degradation.get_soh_at_year(0)
            print(f"  RTE/SOH Curves: Loaded ({len(self.config.degradation.rte_curve)} years)")
            print(f"    Year 0: RTE {rte_0:.4f}, SOH {soh_0:.4f}")
        else:
            print(f"  RTE/SOH: Using constant values (RTE: {self.hw.efficiency_roundtrip:.4f}, SOH: 1.0)")
    
    def get_yearly_parameters(self, operational_year: int) -> Tuple[float, float, float]:

        if (hasattr(self.config, 'degradation') and 
            self.config.degradation.rte_curve is not None and 
            self.config.degradation.soh_curve is not None):
            # Use degradation curves from config
            rte = self.config.degradation.get_rte_at_year(operational_year)
            soh = self.config.degradation.get_soh_at_year(operational_year)
            effective_capacity = self.hw.nominal_capacity_mwh * soh
        else:
            # No curves available - use constant values
            rte = self.hw.efficiency_roundtrip
            soh = 1.0
            effective_capacity = self.hw.nominal_capacity_mwh
        
        return (rte, soh, effective_capacity)
        
    def _calculate_annual_thresholds(self) -> ThresholdParameters:

        print("\n" + "="*70)
        print("CALCULATING ANNUAL THRESHOLDS")
        print("="*70)
        
        # Price characterization
        da_low = np.percentile(self.market.da_prices, 25)
        da_high = np.percentile(self.market.da_prices, 75)
        da_mean = np.mean(self.market.da_prices)
        da_std = np.std(self.market.da_prices)
        
        ida_low = np.percentile(self.market.ida_mid, 25)
        ida_high = np.percentile(self.market.ida_mid, 75)
        ida_mean = np.mean(self.market.ida_mid)
        ida_spread_mean = np.mean(self.market.ida_ask - self.market.ida_bid)
        
        print(f"\nDay-Ahead Price Distribution:")
        print(f"  Mean: €{da_mean:.2f}/MWh | Std: €{da_std:.2f}")
        print(f"  P25 (Low): €{da_low:.2f} | P75 (High): €{da_high:.2f}")
        
        print(f"\nIntraday Price Distribution:")
        print(f"  Mean: €{ida_mean:.2f}/MWh | Spread: €{ida_spread_mean:.2f}")
        print(f"  P25: €{ida_low:.2f} | P75: €{ida_high:.2f}")
        
        # FCR threshold calculation (per 4-hour block)
        # Reshape to blocks: 35040 periods / 16 periods per block = 2190 blocks
        n_blocks = len(self.market.timestamps) // 16
        
        # 3-TIER ALLOCATION: HIGH and MEDIUM thresholds
        fcr_threshold_high = np.zeros(n_blocks)
        fcr_threshold_medium = np.zeros(n_blocks)
        afrr_pos_threshold_high = np.zeros(n_blocks)
        afrr_pos_threshold_medium = np.zeros(n_blocks)
        afrr_neg_threshold_high = np.zeros(n_blocks)
        afrr_neg_threshold_medium = np.zeros(n_blocks)
        
        print(f"\n{'='*100}")
        print("CALCULATING DYNAMIC HURDLE RATES FROM MARKET DATA")
        print(f"{'='*100}")
        
        # Step 1: Calculate average arbitrage opportunity across all blocks
        block_arb_spreads = []
        
        for block_idx in range(n_blocks):
            start_p = block_idx * 16
            end_p = start_p + 16
            block_da = self.market.da_prices[start_p:end_p]
            p75 = np.percentile(block_da, 75)
            p25 = np.percentile(block_da, 25)
            block_arb_spreads.append(p75 - p25)
        
        avg_arb_gross_mwh = np.mean(block_arb_spreads)
        
        # Step 2: Calculate average reserve revenues (capacity only - activations added below)
        avg_fcr_per_h = np.mean(self.market.fcr_prices)  # €/MW/h
        avg_fcr_capacity_revenue_4h = avg_fcr_per_h * 4  # €/MW per 4h
        
        avg_afrr_pos_per_h = np.mean(self.market.afrr_cap_pos)
        avg_afrr_pos_capacity_revenue_4h = avg_afrr_pos_per_h * 4
        
        avg_afrr_neg_per_h = np.mean(self.market.afrr_cap_neg)
        avg_afrr_neg_capacity_revenue_4h = avg_afrr_neg_per_h * 4
        
        print(f"\n📊 MARKET AVERAGES (for hurdle calculation)")
        print(f"  Arbitrage gross spread: €{avg_arb_gross_mwh:.2f}/MWh")
        print(f"  FCR capacity revenue: €{avg_fcr_capacity_revenue_4h:.2f}/MW/4h")
        print(f"  aFRR Pos capacity revenue: €{avg_afrr_pos_capacity_revenue_4h:.2f}/MW/4h")
        print(f"  aFRR Neg capacity revenue: €{avg_afrr_neg_capacity_revenue_4h:.2f}/MW/4h")
        
        # Calculate battery duration from hardware specs (CRITICAL FIX)
        battery_duration_hours = self.hw.nominal_capacity_mwh / self.hw.nominal_power_mw
        print(f"\nBattery Configuration:")
        print(f"  Capacity: {self.hw.nominal_capacity_mwh:.1f} MWh")
        print(f"  Power: {self.hw.nominal_power_mw:.1f} MW")
        print(f"  Duration: {battery_duration_hours:.1f} hours")
        
        # ==================== DEGRADATION COST ESTIMATION ====================
        # Calculate degradation cost per MWh for use in hurdle calculations
        # Using DOD-based model: C_degr = |DOD^1.14 - DOD_prev^1.14| / (2 × warranty_cycles) × CAPEX
        
        peukert_constant = 1.14
        degradation_factor = self.config.financial.capex_total_eur / (2 * self.config.lifetime.warranty_cycles)
        
        # Estimate degradation per MWh at mid-range SOC
        mid_soc = (self.hw.soc_min + self.hw.soc_max) / 2
        mid_dod = 1 - mid_soc
        
        # Calculate degradation for a reference cycle (e.g., 10% SOC swing)
        delta_soc = 0.10
        dod_before = mid_dod
        dod_after = mid_dod + delta_soc
        
        degradation_cost_reference = abs(dod_after**peukert_constant - dod_before**peukert_constant) * degradation_factor
        energy_in_reference_cycle = delta_soc * self.hw.nominal_capacity_mwh
        degradation_per_mwh = degradation_cost_reference / energy_in_reference_cycle
        
        print(f"\nDegradation Cost Estimation (DOD-based model):")
        print(f"  Degradation factor: €{degradation_factor:.2f} per cycle")
        print(f"  Estimated degradation per MWh: €{degradation_per_mwh:.2f}/MWh")
        
        # Step 3: Calculate net arbitrage profit (including degradation)
        avg_arb_net_mwh = avg_arb_gross_mwh - degradation_per_mwh
        energy_per_mw_4h = min(battery_duration_hours, 4.0)
        avg_arb_net_per_mw_4h = avg_arb_net_mwh * energy_per_mw_4h
        
        print(f"\n💰 ARBITRAGE NET PROFIT (Average)")
        print(f"  Gross spread: €{avg_arb_gross_mwh:.2f}/MWh")
        print(f"  Degradation: -€{degradation_per_mwh:.2f}/MWh")
        print(f"  NET: €{avg_arb_net_mwh:.2f}/MWh")
        print(f"  Energy per MW (4h): {energy_per_mw_4h:.2f} MWh")
        print(f"  NET per MW (4h): €{avg_arb_net_per_mw_4h:.2f}/MW/4h")
        
        # Step 4: Calculate expected activation values (with degradation)
        # 
        # FCR activations
        fcr_activation_prob = 0.30
        fcr_activation_size_pct = 0.05
        fcr_expected_activation_mwh_per_mw = fcr_activation_prob * fcr_activation_size_pct * 4  # Over 4h
        
        avg_da_price = np.mean(self.market.da_prices)
        fcr_activation_revenue = fcr_expected_activation_mwh_per_mw * avg_da_price
        fcr_activation_degradation = fcr_expected_activation_mwh_per_mw * degradation_per_mwh
        fcr_activation_net = fcr_activation_revenue - fcr_activation_degradation
        
        # aFRR Positive activations
        afrr_pos_activation_prob = 0.12
        afrr_pos_activation_size_pct = 0.60
        afrr_pos_expected_activation_mwh_per_mw = afrr_pos_activation_prob * afrr_pos_activation_size_pct * 4
        
        avg_rebap = np.mean(self.market.afrr_energy)
        afrr_pos_activation_revenue = afrr_pos_expected_activation_mwh_per_mw * avg_rebap
        afrr_pos_activation_degradation = afrr_pos_expected_activation_mwh_per_mw * degradation_per_mwh
        afrr_pos_activation_net = afrr_pos_activation_revenue - afrr_pos_activation_degradation
        
        # aFRR Negative activations
        afrr_neg_activation_prob = 0.18
        afrr_neg_activation_size_pct = 0.60
        afrr_neg_expected_activation_mwh_per_mw = afrr_neg_activation_prob * afrr_neg_activation_size_pct * 4
        
        afrr_neg_activation_revenue = afrr_neg_expected_activation_mwh_per_mw * avg_rebap
        afrr_neg_activation_degradation = afrr_neg_expected_activation_mwh_per_mw * degradation_per_mwh
        afrr_neg_activation_net = afrr_neg_activation_revenue - afrr_neg_activation_degradation
        
        print(f"\n⚡ EXPECTED ACTIVATION VALUES (net of degradation)")
        print(f"  FCR:")
        print(f"    Expected MWh/MW (4h): {fcr_expected_activation_mwh_per_mw:.3f}")
        print(f"    Gross revenue: €{fcr_activation_revenue:.2f}/MW/4h")
        print(f"    Degradation: -€{fcr_activation_degradation:.2f}/MW/4h")
        print(f"    NET: €{fcr_activation_net:.2f}/MW/4h")
        print(f"  aFRR Pos:")
        print(f"    Expected MWh/MW (4h): {afrr_pos_expected_activation_mwh_per_mw:.3f}")
        print(f"    Gross revenue: €{afrr_pos_activation_revenue:.2f}/MW/4h")
        print(f"    Degradation: -€{afrr_pos_activation_degradation:.2f}/MW/4h")
        print(f"    NET: €{afrr_pos_activation_net:.2f}/MW/4h")
        print(f"  aFRR Neg:")
        print(f"    Expected MWh/MW (4h): {afrr_neg_expected_activation_mwh_per_mw:.3f}")
        print(f"    Gross revenue: €{afrr_neg_activation_revenue:.2f}/MW/4h")
        print(f"    Degradation: -€{afrr_neg_activation_degradation:.2f}/MW/4h")
        print(f"    NET: €{afrr_neg_activation_net:.2f}/MW/4h")
        
        # Step 5: Calculate total net profits
        fcr_total_net_profit = avg_fcr_capacity_revenue_4h + fcr_activation_net
        afrr_pos_total_net_profit = avg_afrr_pos_capacity_revenue_4h + afrr_pos_activation_net
        afrr_neg_total_net_profit = avg_afrr_neg_capacity_revenue_4h + afrr_neg_activation_net
        
        print(f"\n💵 TOTAL NET PROFITS (Capacity + Activations - Degradation)")
        print(f"  Arbitrage: €{avg_arb_net_per_mw_4h:.2f}/MW/4h")
        print(f"  FCR: €{fcr_total_net_profit:.2f}/MW/4h")
        print(f"  aFRR Pos: €{afrr_pos_total_net_profit:.2f}/MW/4h")
        print(f"  aFRR Neg: €{afrr_neg_total_net_profit:.2f}/MW/4h")
        
        # Step 6: Calculate economic hurdles (Reserve - Arbitrage)
        fcr_economic_hurdle = fcr_total_net_profit - avg_arb_net_per_mw_4h
        afrr_pos_economic_hurdle = afrr_pos_total_net_profit - avg_arb_net_per_mw_4h
        afrr_neg_economic_hurdle = afrr_neg_total_net_profit - avg_arb_net_per_mw_4h
        
        print(f"\n🎯 ECONOMIC HURDLES (Reserve - Arbitrage)")
        print(f"  FCR: €{fcr_economic_hurdle:.2f}/MW/4h", end="")
        if fcr_economic_hurdle > 0:
            print(f" (✅ FCR more profitable)")
        else:
            print(f" (⚠️ Arbitrage more profitable)")
        
        print(f"  aFRR Pos: €{afrr_pos_economic_hurdle:.2f}/MW/4h", end="")
        if afrr_pos_economic_hurdle > 0:
            print(f" (✅ aFRR more profitable)")
        else:
            print(f" (⚠️ Arbitrage more profitable)")
        
        print(f"  aFRR Neg: €{afrr_neg_economic_hurdle:.2f}/MW/4h", end="")
        if afrr_neg_economic_hurdle > 0:
            print(f" (✅ aFRR more profitable)")
        else:
            print(f" (⚠️ Arbitrage more profitable)")
        
        # Step 7: Apply safety margins
        fcr_safety_margin = 5.0  # Add €5 buffer for FCR
        afrr_safety_margin = -10.0  # Reduce by €10 for diversification
        
        fcr_high_hurdle = max(0.0, fcr_economic_hurdle + fcr_safety_margin)
        afrr_high_hurdle = max(0.0, afrr_pos_economic_hurdle + afrr_safety_margin)  # Use positive as reference
        
        # MEDIUM tier: lower by fixed amount for 2-tier system
        tier_gap = 5.0  # €5 gap between HIGH and MEDIUM
        fcr_medium_hurdle = max(0.0, fcr_high_hurdle - tier_gap)
        afrr_medium_hurdle = max(0.0, afrr_high_hurdle - tier_gap)
        
        print(f"\n✅ FINAL DYNAMIC HURDLES (with safety margins)")
        print(f"  FCR HIGH: €{fcr_high_hurdle:.2f}/MW/4h (economic €{fcr_economic_hurdle:.2f} + €{fcr_safety_margin:.2f} safety)")
        print(f"  FCR MEDIUM: €{fcr_medium_hurdle:.2f}/MW/4h (HIGH - €{tier_gap:.2f})")
        print(f"  aFRR HIGH: €{afrr_high_hurdle:.2f}/MW/4h (economic €{afrr_pos_economic_hurdle:.2f} + €{afrr_safety_margin:.2f} diversification)")
        print(f"  aFRR MEDIUM: €{afrr_medium_hurdle:.2f}/MW/4h (HIGH - €{tier_gap:.2f})")
        
        print(f"\n{'='*100}\n")
        
        # Calculate degradation premiums for threshold calculation (same as before)
        # These are added to thresholds to account for activation degradation
        fcr_degradation_premium = degradation_per_mwh * fcr_activation_prob * 0.50
        afrr_pos_degradation_premium = degradation_per_mwh * afrr_pos_activation_prob * 0.60
        afrr_neg_degradation_premium = degradation_per_mwh * afrr_neg_activation_prob * 0.60
        
        print(f"Activation Degradation Premiums (added to thresholds):")
        print(f"  FCR: €{fcr_degradation_premium:.2f}/MW/4h")
        print(f"  aFRR Pos: €{afrr_pos_degradation_premium:.2f}/MW/4h")
        print(f"  aFRR Neg: €{afrr_neg_degradation_premium:.2f}/MW/4h")

        battery_duration_hours = self.hw.nominal_capacity_mwh / self.hw.nominal_power_mw

        
        for block_idx in range(n_blocks):
            start_period = block_idx * 16
            end_period = start_period + 16
            
            # Block price opportunity (P75 - P25)
            block_da = self.market.da_prices[start_period:end_period]
            block_ida = self.market.ida_mid[start_period:end_period]
            
            da_p75 = np.percentile(block_da, 75)
            da_p25 = np.percentile(block_da, 25)
            ida_p75 = np.percentile(block_ida, 75)
            ida_p25 = np.percentile(block_ida, 25)
            
            # Arbitrage opportunity (per MWh for 4 hours)
            arb_opportunity_mwh = max(da_p75 - da_p25, ida_p75 - ida_p25)
            

            net_arb_value_mwh = arb_opportunity_mwh - degradation_per_mwh

            energy_per_mw_in_4h = min(battery_duration_hours, 4.0)
            net_arb_value_mw_4h = net_arb_value_mwh * energy_per_mw_in_4h

            fcr_threshold_high[block_idx] = net_arb_value_mw_4h + fcr_high_hurdle + fcr_degradation_premium
            fcr_threshold_medium[block_idx] = net_arb_value_mw_4h + fcr_medium_hurdle + fcr_degradation_premium

            afrr_pos_threshold_high[block_idx] = net_arb_value_mw_4h + afrr_high_hurdle + afrr_pos_degradation_premium
            afrr_pos_threshold_medium[block_idx] = net_arb_value_mw_4h + afrr_medium_hurdle + afrr_pos_degradation_premium
            afrr_neg_threshold_high[block_idx] = net_arb_value_mw_4h + afrr_high_hurdle + afrr_neg_degradation_premium
            afrr_neg_threshold_medium[block_idx] = net_arb_value_mw_4h + afrr_medium_hurdle + afrr_neg_degradation_premium
        
        print(f"\n{'='*100}")
        print(f"THRESHOLD SUMMARY (Dynamic Hurdles + Degradation-Aware)")
        print(f"{'='*100}")
        print(f"  FCR HIGH: Net Arbitrage + €{fcr_high_hurdle:.2f}/MW/4h + €{fcr_degradation_premium:.2f} activation deg")
        print(f"  FCR MEDIUM: Net Arbitrage + €{fcr_medium_hurdle:.2f}/MW/4h + €{fcr_degradation_premium:.2f} activation deg")
        print(f"  aFRR Pos HIGH: Net Arbitrage + €{afrr_high_hurdle:.2f}/MW/4h + €{afrr_pos_degradation_premium:.2f} activation deg")
        print(f"  aFRR Pos MEDIUM: Net Arbitrage + €{afrr_medium_hurdle:.2f}/MW/4h + €{afrr_pos_degradation_premium:.2f} activation deg")
        print(f"  aFRR Neg HIGH: Net Arbitrage + €{afrr_high_hurdle:.2f}/MW/4h + €{afrr_neg_degradation_premium:.2f} activation deg")
        print(f"  aFRR Neg MEDIUM: Net Arbitrage + €{afrr_medium_hurdle:.2f}/MW/4h + €{afrr_neg_degradation_premium:.2f} activation deg")
        print(f"\n  Energy per MW in 4h block: {energy_per_mw_in_4h:.1f} MWh (battery duration: {battery_duration_hours:.1f}h)")
        print(f"  Net arbitrage = Gross spread - €{degradation_per_mwh:.2f}/MWh degradation")
        print(f"  Hurdles calculated from: Market avg revenues - Market avg arbitrage profit")
        print(f"{'='*100}\n")
        print(f"  Arbitrage degradation penalty: €{degradation_per_mwh:.2f}/MWh (subtracted from opportunity)")
        
        print(f"\nThreshold Statistics (€/MW per 4h block):")
        print(f"  FCR HIGH: {np.mean(fcr_threshold_high):.2f} ± {np.std(fcr_threshold_high):.2f}")
        print(f"  FCR MEDIUM: {np.mean(fcr_threshold_medium):.2f} ± {np.std(fcr_threshold_medium):.2f}")
        print(f"  aFRR Pos HIGH: {np.mean(afrr_pos_threshold_high):.2f} ± {np.std(afrr_pos_threshold_high):.2f}")
        print(f"  aFRR Neg HIGH: {np.mean(afrr_neg_threshold_high):.2f} ± {np.std(afrr_neg_threshold_high):.2f}")
        
        print(f"\nExpected Allocation Pattern (with wider gaps):")
        print(f"  FCR: ~38% HIGH, ~20% MEDIUM (ratio 2:1)")
        print(f"  aFRR: ~30% HIGH, ~20% MEDIUM (ratio 1.5:1)")
        print(f"  Revenue improvement: +3-5% from better MEDIUM tier capture")
        
        return ThresholdParameters(
            da_low_percentile=da_low,
            da_high_percentile=da_high,
            da_mean=da_mean,
            da_std=da_std,
            ida_low_percentile=ida_low,
            ida_high_percentile=ida_high,
            ida_mean=ida_mean,
            ida_spread_mean=ida_spread_mean,
            fcr_threshold_high_eur_mw_4h=fcr_threshold_high,
            fcr_threshold_medium_eur_mw_4h=fcr_threshold_medium,
            afrr_pos_threshold_high_eur_mw_4h=afrr_pos_threshold_high,
            afrr_pos_threshold_medium_eur_mw_4h=afrr_pos_threshold_medium,
            afrr_neg_threshold_high_eur_mw_4h=afrr_neg_threshold_high,
            afrr_neg_threshold_medium_eur_mw_4h=afrr_neg_threshold_medium
        )
    
    def _check_buffer_available(self, current_soc: float, afrr_allocated_mw: float = 0,
                                capacity_mwh: float = None) -> bool:

        if afrr_allocated_mw == 0:
            return True  # No buffer needed if not allocating
        
        # Use provided capacity or default to nominal
        if capacity_mwh is None:
            capacity_mwh = self.hw.nominal_capacity_mwh
        
        # CORRECTED: Buffer based on ALLOCATED power, not total power
        # Example: If allocating 12.5 MW to aFRR, buffer = 12.5 MWh (not 50 MWh)
        buffer_mwh = afrr_allocated_mw * 1.0  # 60 minutes at ALLOCATED power
        buffer_soc = buffer_mwh / capacity_mwh  # Use effective capacity
        
        # Effective SOC range
        effective_min = self.hw.soc_min + buffer_soc
        effective_max = self.hw.soc_max - buffer_soc
        
        # Check current SOC is within range
        return (current_soc >= effective_min) and (current_soc <= effective_max)
    
    def _allocate_block_markets(self, day: int, block: int, current_soc: float,
                                capacity_mwh: float = None) -> Tuple[float, float, float]:

        # Use provided capacity or default to nominal
        if capacity_mwh is None:
            capacity_mwh = self.hw.nominal_capacity_mwh
        
        block_idx = day * 6 + block
        start_period = block_idx * 16
        
        # Get block prices (€/MW for 4h)
        fcr_price = self.market.fcr_prices[start_period] * 4.0
        afrr_pos_price = self.market.afrr_cap_pos[start_period] * 4.0
        afrr_neg_price = self.market.afrr_cap_neg[start_period] * 4.0
        
        # Initialize allocations
        fcr_allocated = 0.0
        afrr_pos_allocated = 0.0
        afrr_neg_allocated = 0.0
        
        # FCR 3-TIER ALLOCATION
        if fcr_price > self.thresholds.fcr_threshold_high_eur_mw_4h[block_idx]:
            # HIGH tier: 40% allocation
            fcr_allocated = 0.40 * self.hw.nominal_power_mw
        elif fcr_price > self.thresholds.fcr_threshold_medium_eur_mw_4h[block_idx]:
            # MEDIUM tier: 20% allocation
            fcr_allocated = 0.20 * self.hw.nominal_power_mw
        # else: LOW tier (0%)
        
        # aFRR POSITIVE 3-TIER ALLOCATION (with buffer check)
        if afrr_pos_price > self.thresholds.afrr_pos_threshold_high_eur_mw_4h[block_idx]:
            # HIGH tier: 25% allocation
            candidate_alloc = 0.25 * self.hw.nominal_power_mw
            if self._check_buffer_available(current_soc, candidate_alloc, capacity_mwh):
                afrr_pos_allocated = candidate_alloc
        elif afrr_pos_price > self.thresholds.afrr_pos_threshold_medium_eur_mw_4h[block_idx]:
            # MEDIUM tier: 12.5% allocation
            candidate_alloc = 0.125 * self.hw.nominal_power_mw
            if self._check_buffer_available(current_soc, candidate_alloc, capacity_mwh):
                afrr_pos_allocated = candidate_alloc
        # else: LOW tier (0%)
        
        # aFRR NEGATIVE 3-TIER ALLOCATION (with buffer check)
        if afrr_neg_price > self.thresholds.afrr_neg_threshold_high_eur_mw_4h[block_idx]:
            # HIGH tier: 25% allocation
            candidate_alloc = 0.25 * self.hw.nominal_power_mw
            if self._check_buffer_available(current_soc, candidate_alloc, capacity_mwh):
                afrr_neg_allocated = candidate_alloc
        elif afrr_neg_price > self.thresholds.afrr_neg_threshold_medium_eur_mw_4h[block_idx]:
            # MEDIUM tier: 12.5% allocation
            candidate_alloc = 0.125 * self.hw.nominal_power_mw
            if self._check_buffer_available(current_soc, candidate_alloc, capacity_mwh):
                afrr_neg_allocated = candidate_alloc
        # else: LOW tier (0%)
        
        return fcr_allocated, afrr_pos_allocated, afrr_neg_allocated
    
    def _simulate_activations(self, fcr_mw: float, afrr_pos_mw: float, afrr_neg_mw: float,
                              current_soc: float,
                              capacity_mwh: float = None,
                              current_soh: float = None) -> Tuple[float, float, float, float, float, float, float]:

        # Use provided capacity or default to nominal
        if capacity_mwh is None:
            capacity_mwh = self.hw.nominal_capacity_mwh
        
        # Use provided SOH or default to soc_max (1.0)
        if current_soh is None:
            current_soh = self.hw.soc_max
        
        # Degradation parameters from config
        peukert_constant = 1.14  # For Li-ion batteries (Sage & Zhao, 2025)
        warranty_cycles = self.config.lifetime.warranty_cycles
        capex_total = self.config.financial.capex_total_eur
        degradation_factor = capex_total / (2 * warranty_cycles)  # €/MWh factor
        
        # Initial state
        soc_before_fcr = current_soc
        dod_before_fcr = 1 - soc_before_fcr
        
        # ==================== FCR ACTIVATION ====================
        fcr_activation = np.random.normal(0, fcr_mw * 0.05) if fcr_mw > 0 else 0.0
        fcr_energy = fcr_activation * 0.25  # MW * 0.25h = MWh
        delta_soc_fcr = fcr_energy / capacity_mwh  # Use effective capacity
        soc_after_fcr = np.clip(soc_before_fcr + delta_soc_fcr, self.hw.soc_min, current_soh)  # FIX: Use current_soh
        
        # FCR degradation cost
        if abs(soc_after_fcr - soc_before_fcr) > 0.0001:  # Only if SOC actually changed
            dod_after_fcr = 1 - soc_after_fcr
            fcr_degradation = abs(dod_after_fcr ** peukert_constant - dod_before_fcr ** peukert_constant) * degradation_factor
        else:
            fcr_degradation = 0.0
        
        # ==================== aFRR POSITIVE ACTIVATION ====================
        soc_before_afrr_pos = soc_after_fcr
        dod_before_afrr_pos = 1 - soc_before_afrr_pos
        
        if afrr_pos_mw > 0 and np.random.random() < self.afrr_params.afrr_activation_prob_pos:
            afrr_pos_activation = afrr_pos_mw * np.random.uniform(0.3, 0.8) * 0.25  # MWh (15min)
        else:
            afrr_pos_activation = 0.0
        
        # Update SOC (discharge = negative)
        delta_soc_afrr_pos = -afrr_pos_activation / capacity_mwh  # Use effective capacity
        soc_after_afrr_pos = np.clip(soc_before_afrr_pos + delta_soc_afrr_pos, self.hw.soc_min, current_soh)  # FIX: Use current_soh
        
        # aFRR Positive degradation cost
        if abs(soc_after_afrr_pos - soc_before_afrr_pos) > 0.0001:
            dod_after_afrr_pos = 1 - soc_after_afrr_pos
            afrr_pos_degradation = abs(dod_after_afrr_pos ** peukert_constant - dod_before_afrr_pos ** peukert_constant) * degradation_factor
        else:
            afrr_pos_degradation = 0.0
        
        # ==================== aFRR NEGATIVE ACTIVATION ====================
        soc_before_afrr_neg = soc_after_afrr_pos
        dod_before_afrr_neg = 1 - soc_before_afrr_neg
        
        if afrr_neg_mw > 0 and np.random.random() < self.afrr_params.afrr_activation_prob_neg:
            afrr_neg_activation = afrr_neg_mw * np.random.uniform(0.3, 0.8) * 0.25  # MWh (15min)
        else:
            afrr_neg_activation = 0.0
        
        # Update SOC (charge = positive)
        delta_soc_afrr_neg = afrr_neg_activation / capacity_mwh  # Use effective capacity
        soc_after_afrr_neg = np.clip(soc_before_afrr_neg + delta_soc_afrr_neg, self.hw.soc_min, current_soh)  # FIX: Use current_soh
        
        # aFRR Negative degradation cost
        if abs(soc_after_afrr_neg - soc_before_afrr_neg) > 0.0001:
            dod_after_afrr_neg = 1 - soc_after_afrr_neg
            afrr_neg_degradation = abs(dod_after_afrr_neg ** peukert_constant - dod_before_afrr_neg ** peukert_constant) * degradation_factor
        else:
            afrr_neg_degradation = 0.0
        
        # Final SOC after all activations
        new_soc = soc_after_afrr_neg
        
        return (fcr_activation, afrr_pos_activation, afrr_neg_activation, new_soc,
                fcr_degradation, afrr_pos_degradation, afrr_neg_degradation)
    
    def _optimize_daily_energy_arbitrage(self, day: int, soc_start: float,
                                        fcr_allocations: List[float],
                                        afrr_pos_allocations: List[float],
                                        afrr_neg_allocations: List[float],
                                        rte: float = None,
                                        capacity_mwh: float = None,
                                        current_soh: float = None) -> pd.DataFrame:

        # Use provided parameters or defaults from config
        if rte is None:
            rte = self.hw.efficiency_roundtrip
        if capacity_mwh is None:
            capacity_mwh = self.hw.nominal_capacity_mwh
        if current_soh is None:
            current_soh = self.hw.soc_max
        
        start_period = day * 96
        end_period = start_period + 96
        
        # Extract daily prices
        da_prices = self.market.da_prices[start_period:end_period]
        ida_mid = self.market.ida_mid[start_period:end_period]
        ida_bid = self.market.ida_bid[start_period:end_period]
        ida_ask = self.market.ida_ask[start_period:end_period]
        
        # Calculate available power per period
        available_power = np.zeros(96)
        for t in range(96):
            block_idx = t // 16
            fcr_used = fcr_allocations[block_idx]
            afrr_pos_used = afrr_pos_allocations[block_idx]
            afrr_neg_used = afrr_neg_allocations[block_idx]
            
            # CRITICAL FIX: Subtract ALL reserve allocations
            # Both aFRR Pos and Neg reserve capacity simultaneously
            # - aFRR Pos = reserved discharge capacity
            # - aFRR Neg = reserved charge capacity
            # - Both can be called in different 15-min periods within same block
            # - Therefore BOTH must be subtracted from available power
            remaining_power = self.hw.nominal_power_mw - fcr_used - afrr_pos_used - afrr_neg_used
            available_power[t] = max(0, remaining_power)
        
        # Market selection: INDEPENDENT decisions for sell and buy
        # Previous logic used |IDA_mid - DA| > €5 (symmetric, buggy)
        # New logic: Use IDA only when it provides clear advantage
        
        # Thresholds (€/MWh)
        # Set higher than typical IDA bid-ask spread (€0.5-1.0) to ensure net benefit
        # Also accounts for IDA forecast uncertainty
        sell_threshold = 2.0  # Require €2/MWh advantage to use IDA for selling (discharge)
        buy_threshold = 2.0   # Require €2/MWh advantage to use IDA for buying (charge)
        
        # Calculate SIGNED advantages (not absolute)
        # Positive values indicate IDA is better than DA
        sell_advantage = ida_bid - da_prices  # Positive when IDA pays MORE for selling
        buy_advantage = da_prices - ida_ask   # Positive when IDA costs LESS for buying

        # Independent selection for sell and buy
        # Only use IDA when advantage exceeds threshold
        sell_prices = np.where(sell_advantage > sell_threshold, ida_bid, da_prices)
        buy_prices = np.where(buy_advantage > buy_threshold, ida_ask, da_prices)

        # Track which market was selected (for output and analysis)
        sell_market = np.where(sell_advantage > sell_threshold, 'IDA', 'DA')
        buy_market = np.where(buy_advantage > buy_threshold, 'IDA', 'DA')

        # Print market selection summary for this day
        ida_sell_periods = (sell_advantage > sell_threshold).sum()
        ida_buy_periods = (buy_advantage > buy_threshold).sum()

        # LP Problem
        prob = LpProblem(f"BESS_Arbitrage_Day_{day}", LpMaximize)
        
        # ==================== DEGRADATION MODEL SETUP ====================
        # Exact model: C_degr = |DOD_after^1.14 - DOD_before^1.14| / (2 × warranty_cycles) × CAPEX
        # Piecewise Linear Approximation for LP compatibility
        
        # Get degradation parameters
        peukert_constant = 1.14
        warranty_cycles = self.config.lifetime.warranty_cycles
        capex_total = self.config.financial.capex_total_eur
        degradation_base = capex_total / (2 * warranty_cycles)
        
        # Create piecewise linear approximation of DOD^1.14
        # Sample points across SOC range [soc_min, soc_max]
        n_segments = 10  # Number of linear segments for approximation
        soc_points = np.linspace(self.hw.soc_min, self.hw.soc_max, n_segments + 1)
        dod_points = 1 - soc_points
        
        # Compute DOD^1.14 at each point
        dod_power = dod_points ** peukert_constant
        
        # For LP: approximate |DOD_after^1.14 - DOD_before^1.14| using slopes
        # We'll use a simplified approach: penalty based on SOC change magnitude
        # weighted by the local slope of the DOD^1.14 function
        
        # Calculate average slope in the operating range
        # Derivative of DOD^1.14 = 1.14 * DOD^0.14
        # At mid-range SOC (e.g., 0.5), DOD = 0.5, slope ≈ 1.14 * 0.5^0.14 ≈ 1.07
        mid_soc = (self.hw.soc_min + self.hw.soc_max) / 2
        mid_dod = 1 - mid_soc
        slope_factor = peukert_constant * (mid_dod ** (peukert_constant - 1))
        
        # Degradation cost per unit SOC change (linear approximation)
        # For small SOC changes: |DOD^1.14 - DOD_prev^1.14| ≈ slope * |DOD - DOD_prev| = slope * |ΔSOC|
        # Cost per MWh throughput - FIX: Use DEGRADED capacity (capacity_mwh) not nominal!
        degradation_cost_per_mwh = degradation_base * slope_factor / capacity_mwh
        
        # Decision variables
        charge_mw = [LpVariable(f"charge_{t}", lowBound=0, upBound=available_power[t]) for t in range(96)]
        discharge_mw = [LpVariable(f"discharge_{t}", lowBound=0, upBound=available_power[t]) for t in range(96)]
        soc = [LpVariable(f"soc_{t}", lowBound=self.hw.soc_min, upBound=current_soh) for t in range(96)]  # FIX: Use current_soh!
        
        # Binary variables for charge/discharge (prevent simultaneous)
        is_charging = [LpVariable(f"is_charging_{t}", cat='Binary') for t in range(96)]
        is_discharging = [LpVariable(f"is_discharging_{t}", cat='Binary') for t in range(96)]
        
        # Objective: Maximize (arbitrage revenue - degradation costs)
        # Degradation penalty applied to throughput (charge + discharge)
        prob += lpSum([
            discharge_mw[t] * sell_prices[t] * 0.25 - charge_mw[t] * buy_prices[t] * 0.25 -
            degradation_cost_per_mwh * (discharge_mw[t] + charge_mw[t]) * 0.25
            for t in range(96)
        ])
        
        # Constraints
        # 1. SOC dynamics
        # Use round-trip efficiency (sqrt for charge and discharge)
        eta_charge = rte ** 0.5
        eta_discharge = rte ** 0.5

        max_discharge_grid = [available_power[t] * eta_discharge for t in range(96)]
        max_charge_grid = [available_power[t] for t in range(96)]  # Grid-limited (conservative)
        
        # Decision variables with RTE-aware bounds
        charge_mw = [LpVariable(f"charge_{t}", lowBound=0, upBound=max_charge_grid[t]) for t in range(96)]
        discharge_mw = [LpVariable(f"discharge_{t}", lowBound=0, upBound=max_discharge_grid[t]) for t in range(96)]
        soc = [LpVariable(f"soc_{t}", lowBound=self.hw.soc_min, upBound=current_soh) for t in range(96)]  # FIX: Use current_soh!
        
        # Binary variables for charge/discharge (prevent simultaneous)
        is_charging = [LpVariable(f"is_charging_{t}", cat='Binary') for t in range(96)]
        is_discharging = [LpVariable(f"is_discharging_{t}", cat='Binary') for t in range(96)]
        
        # Objective: Maximize (arbitrage revenue - degradation costs)
        # Degradation penalty applied to throughput (charge + discharge)
        prob += lpSum([
            discharge_mw[t] * sell_prices[t] * 0.25 - charge_mw[t] * buy_prices[t] * 0.25 -
            degradation_cost_per_mwh * (discharge_mw[t] + charge_mw[t]) * 0.25
            for t in range(96)
        ])

        prob += soc[0] == soc_start + (
            charge_mw[0] * eta_charge - discharge_mw[0] / eta_discharge
        ) * 0.25 / capacity_mwh
        
        for t in range(1, 96):
            prob += soc[t] == soc[t-1] + (
                charge_mw[t] * eta_charge - discharge_mw[t] / eta_discharge
            ) * 0.25 / capacity_mwh
        
        # 2. No simultaneous charge/discharge (with RTE-aware bounds)
        for t in range(96):
            prob += is_charging[t] + is_discharging[t] <= 1
            prob += charge_mw[t] <= max_charge_grid[t] * is_charging[t]
            prob += discharge_mw[t] <= max_discharge_grid[t] * is_discharging[t]
        
        # 3. Daily cycle limit
        daily_throughput = lpSum([discharge_mw[t] * 0.25 for t in range(96)])
        max_daily_throughput = self.config.lifetime.max_cycles_per_day * capacity_mwh
        prob += daily_throughput <= max_daily_throughput

        for t in range(96):
            block_idx = t // 16  # Which 4-hour block (0-5)
            
            # Get aFRR allocation for this period's block
            afrr_pos_allocated = afrr_pos_allocations[block_idx]
            afrr_neg_allocated = afrr_neg_allocations[block_idx]
            
            # Calculate independent buffers for each direction
            if afrr_pos_allocated > 0:
                # aFRR Positive: Battery will discharge when activated
                # Need sufficient SOC to discharge for 60 minutes
                buffer_positive_mwh = afrr_pos_allocated * 1.0  # 60 minutes
                buffer_positive_soc = buffer_positive_mwh / capacity_mwh
                
                # Minimum SOC constraint (based on ALLOCATED power, independent of activation)
                effective_min_soc = self.hw.soc_min + buffer_positive_soc
                prob += soc[t] >= effective_min_soc
            
            if afrr_neg_allocated > 0:
                # aFRR Negative: Battery will charge when activated
                # Need sufficient headroom to charge for 60 minutes
                buffer_negative_mwh = afrr_neg_allocated * 1.0  # 60 minutes
                buffer_negative_soc = buffer_negative_mwh / capacity_mwh
                
                # Maximum SOC constraint (based on ALLOCATED power, independent of activation)
                effective_max_soc = current_soh - buffer_negative_soc  # FIX: Use current_soh!
                prob += soc[t] <= effective_max_soc
        
        # 5. End-of-day SOC target (return to reasonable level for next day)
        prob += soc[95] >= 0.3
        prob += soc[95] <= min(0.7, current_soh)  # FIX: Respect physical capacity limits
        
        # Solve
        solver = PULP_CBC_CMD(msg=0, timeLimit=30)
        prob.solve(solver)
        
        # SAFETY CHECK: Verify SOC never exceeds current SOH
        soc_values = [soc[t].varValue or soc_start for t in range(96)]
        max_soc_found = max(soc_values)
        if max_soc_found > current_soh + 0.001:  # Small numerical tolerance
            print(f"⚠️  WARNING Day {day}: Max SOC {max_soc_found:.6f} exceeds SOH {current_soh:.6f}")
            print(f"   Clamping all SOC values to {current_soh:.6f}...")
            soc_values = [min(s, current_soh) for s in soc_values]
        
        # Extract results
        results_df = pd.DataFrame({
            'period': range(96),
            'charge_mw': [charge_mw[t].varValue or 0 for t in range(96)],
            'discharge_mw': [discharge_mw[t].varValue or 0 for t in range(96)],
            'soc_arbitrage': soc_values,  # Use safety-checked values
            'available_power_mw': available_power,
            # CRITICAL: Store actual prices and markets used by LP
            'sell_price_used': sell_prices,      # Actual sell price (DA or IDA bid)
            'buy_price_used': buy_prices,        # Actual buy price (DA or IDA ask)
            'sell_market': sell_market,          # Market for selling: 'DA' or 'IDA'
            'buy_market': buy_market,            # Market for buying: 'DA' or 'IDA'
            # Also store price components for analysis
            'da_price': da_prices,               # Day-ahead price
            'ida_bid': ida_bid,                  # Intraday bid (for selling)
            'ida_ask': ida_ask,                  # Intraday ask (for buying)
            'sell_advantage': sell_advantage,    # IDA_bid - DA
            'buy_advantage': buy_advantage       # DA - IDA_ask
        })
        
        return results_df
    
    def optimize_full_year(self) -> pd.DataFrame:

        print("\n" + "="*70)
        print("STARTING FULL YEAR OPTIMIZATION")
        print("="*70)
        
        n_days = 365
        soc_current = self.hw.soc_initial
        
        # Use the warranty_year parameter instead of calculating from day number
        # This ensures correct RTE/SOH for multi-year simulations
        operational_year = self.warranty_year
        current_rte, current_soh, current_capacity = self.get_yearly_parameters(operational_year)
        
        print(f"Simulating Warranty Year {operational_year}:")
        print(f"  RTE: {current_rte:.4f}")
        print(f"  SOH: {current_soh:.4f}")  
        print(f"  Effective Capacity: {current_capacity:.2f} MWh")
        
        for day in range(n_days):
            
            if day % 30 == 0:
                print(f"\nDay {day+1}/365 (Y{operational_year}, SOC: {soc_current:.1%}, RTE: {current_rte:.4f})")
            
            # ========== A. BLOCK ALLOCATION PHASE (6 blocks) ==========
            daily_fcr_alloc = []
            daily_afrr_pos_alloc = []
            daily_afrr_neg_alloc = []
            
            for block in range(6):
                # Call _allocate_block_markets(day, block, soc)
                # Get FCR price, compare to HIGH/MEDIUM thresholds → Allocate 40%/20%/0%
                # Get aFRR prices, check buffer availability → Allocate 25%/12.5%/0%
                fcr_mw, afrr_pos_mw, afrr_neg_mw = self._allocate_block_markets(
                    day, block, soc_current,
                    capacity_mwh=current_capacity  # NEW: Yearly effective capacity
                )
                daily_fcr_alloc.append(fcr_mw)
                daily_afrr_pos_alloc.append(afrr_pos_mw)
                daily_afrr_neg_alloc.append(afrr_neg_mw)
            

            energy_results = self._optimize_daily_energy_arbitrage(
                day, soc_current,
                daily_fcr_alloc, daily_afrr_pos_alloc, daily_afrr_neg_alloc,
                rte=current_rte,              # NEW: Yearly RTE
                capacity_mwh=current_capacity,  # NEW: Yearly effective capacity
                current_soh=current_soh       # FIX: Pass current SOH for constraints
            )

            daily_fcr_activations = []
            daily_afrr_pos_activations = []
            daily_afrr_neg_activations = []
            daily_fcr_degradation = []
            daily_afrr_pos_degradation = []
            daily_afrr_neg_degradation = []
            daily_soc_combined = []
            
            # Start SOC tracker from current SOC
            soc_tracker = soc_current
            
            for block in range(6):
                fcr_mw = daily_fcr_alloc[block]
                afrr_pos_mw = daily_afrr_pos_alloc[block]
                afrr_neg_mw = daily_afrr_neg_alloc[block]
                
                # For each period in the block
                for period_in_block in range(16):
                    t = block * 16 + period_in_block
                    
                    # Get arbitrage SOC from LP
                    soc_arbitrage = energy_results.loc[t, 'soc_arbitrage']
                    
                    # Simulate activation for this period - NOW returns degradation costs
                    fcr_act, afrr_pos_act, afrr_neg_act, soc_after_activation, \
                    fcr_degr, afrr_pos_degr, afrr_neg_degr = self._simulate_activations(
                        fcr_mw, afrr_pos_mw, afrr_neg_mw, soc_tracker,
                        capacity_mwh=current_capacity,  # NEW: Yearly effective capacity
                        current_soh=current_soh        # FIX: Pass current SOH for clipping
                    )
                    
                    # Store activations AND degradation costs
                    daily_fcr_activations.append(fcr_act)
                    daily_afrr_pos_activations.append(afrr_pos_act)
                    daily_afrr_neg_activations.append(afrr_neg_act)
                    daily_fcr_degradation.append(fcr_degr)
                    daily_afrr_pos_degradation.append(afrr_pos_degr)
                    daily_afrr_neg_degradation.append(afrr_neg_degr)
                    
                    # Calculate activation delta
                    activation_delta = soc_after_activation - soc_tracker
                    
                    # Combined SOC = Arbitrage SOC + Activation delta
                    soc_combined = soc_arbitrage + activation_delta
                    daily_soc_combined.append(soc_combined)
                    
                    # Update tracker for next period (block-level tracking)
                    soc_tracker = soc_after_activation

            start_period = day * 96
            for t in range(96):
                period_idx = start_period + t
                block_idx = t // 16
                
                self.results.append({
                    'timestamp': self.market.timestamps[period_idx],
                    'day': day,
                    'period': t,
                    'block': block_idx,
                    'da_price': self.market.da_prices[period_idx],
                    'ida_mid': self.market.ida_mid[period_idx],
                    'fcr_price': self.market.fcr_prices[period_idx],
                    'afrr_cap_pos': self.market.afrr_cap_pos[period_idx],
                    'afrr_cap_neg': self.market.afrr_cap_neg[period_idx],
                    'fcr_allocated_mw': daily_fcr_alloc[block_idx],
                    'afrr_pos_allocated_mw': daily_afrr_pos_alloc[block_idx],
                    'afrr_neg_allocated_mw': daily_afrr_neg_alloc[block_idx],
                    'fcr_activation_mw': daily_fcr_activations[t],
                    'afrr_pos_activation_mwh': daily_afrr_pos_activations[t],
                    'afrr_neg_activation_mwh': daily_afrr_neg_activations[t],
                    # DEGRADATION COSTS from activations
                    'degradation_fcr': daily_fcr_degradation[t],
                    'degradation_afrr_pos': daily_afrr_pos_degradation[t],
                    'degradation_afrr_neg': daily_afrr_neg_degradation[t],
                    'charge_mw': energy_results.loc[t, 'charge_mw'],
                    'discharge_mw': energy_results.loc[t, 'discharge_mw'],
                    'soc': daily_soc_combined[t],  # Combined SOC (arbitrage + activations)
                    'soc_arbitrage_only': energy_results.loc[t, 'soc_arbitrage'],  # For analysis
                    'soc_activation_delta': daily_soc_combined[t] - energy_results.loc[t, 'soc_arbitrage'],  # For analysis
                    'available_power_mw': energy_results.loc[t, 'available_power_mw'],
                    # Market selection and prices used
                    'sell_price_used': energy_results.loc[t, 'sell_price_used'],
                    'buy_price_used': energy_results.loc[t, 'buy_price_used'],
                    'sell_market': energy_results.loc[t, 'sell_market'],
                    'buy_market': energy_results.loc[t, 'buy_market'],
                    'da_price': energy_results.loc[t, 'da_price'],
                    'ida_bid': energy_results.loc[t, 'ida_bid'],
                    'ida_ask': energy_results.loc[t, 'ida_ask'],
                    'sell_advantage': energy_results.loc[t, 'sell_advantage'],
                    'buy_advantage': energy_results.loc[t, 'buy_advantage']
                })
            

            soc_current = daily_soc_combined[95]
        
        results_df = pd.DataFrame(self.results)
        
        # Calculate revenues
        results_df = self._calculate_revenues(results_df)
        
        return results_df
    
    def _calculate_revenues(self, results_df: pd.DataFrame) -> pd.DataFrame:

        # FCR capacity revenue
        results_df['fcr_revenue'] = (
            results_df['fcr_allocated_mw'] * results_df['fcr_price'] * 0.25
        )
        
        # aFRR capacity revenue
        results_df['afrr_capacity_revenue'] = (
            results_df['afrr_pos_allocated_mw'] * results_df['afrr_cap_pos'] * 0.25 +
            results_df['afrr_neg_allocated_mw'] * results_df['afrr_cap_neg'] * 0.25
        )

        rebap_prices = self.market.afrr_energy[results_df.index]
        results_df['afrr_energy_revenue'] = (
            results_df['afrr_pos_activation_mwh'] * rebap_prices +
            results_df['afrr_neg_activation_mwh'] * rebap_prices  # Changed from - to +
        )

        results_df['energy_arbitrage_revenue'] = (
            results_df['discharge_mw'] * results_df['sell_price_used'] * 0.25 -
            results_df['charge_mw'] * results_df['buy_price_used'] * 0.25
        )

        results_df['arbitrage_revenue_da_equivalent'] = (
            results_df['discharge_mw'] * results_df['da_price'] * 0.25 -
            results_df['charge_mw'] * results_df['da_price'] * 0.25
        )
        
        # IDA premium (additional revenue from using IDA when beneficial)
        results_df['ida_premium'] = (
            results_df['energy_arbitrage_revenue'] - results_df['arbitrage_revenue_da_equivalent']
        )

        discharge_on_da = results_df['discharge_mw'] * (results_df['sell_market'] == 'DA')
        results_df['discharge_revenue_da'] = discharge_on_da * results_df['da_price'] * 0.25
        
        # Discharge on IDA
        discharge_on_ida = results_df['discharge_mw'] * (results_df['sell_market'] == 'IDA')
        results_df['discharge_revenue_ida'] = discharge_on_ida * results_df['ida_bid'] * 0.25
        
        # Charge on DA (cost, negative)
        charge_on_da = results_df['charge_mw'] * (results_df['buy_market'] == 'DA')
        results_df['charge_cost_da'] = charge_on_da * results_df['da_price'] * 0.25
        
        # Charge on IDA (cost, negative)
        charge_on_ida = results_df['charge_mw'] * (results_df['buy_market'] == 'IDA')
        results_df['charge_cost_ida'] = charge_on_ida * results_df['ida_ask'] * 0.25
        
        # Get degradation parameters
        peukert_constant = 1.14  # For Li-ion batteries (Sage & Zhao, 2025)
        warranty_cycles = self.config.lifetime.warranty_cycles
        capex_total = self.config.financial.capex_total_eur
        degradation_factor = capex_total / (2 * warranty_cycles)
        
        # Initialize arbitrage degradation column
        results_df['degradation_arbitrage'] = 0.0

        for i in range(1, len(results_df)):
            soc_before = results_df.loc[i-1, 'soc_arbitrage_only']
            soc_after = results_df.loc[i, 'soc_arbitrage_only']
            
            # Only calculate if SOC actually changed
            if abs(soc_after - soc_before) > 0.0001:
                dod_before = 1 - soc_before
                dod_after = 1 - soc_after
                
                degradation_cost = abs(dod_after ** peukert_constant - dod_before ** peukert_constant) * degradation_factor
                results_df.loc[i, 'degradation_arbitrage'] = degradation_cost

        results_df['degradation_total'] = (
            results_df['degradation_arbitrage'] +
            results_df['degradation_fcr'] +
            results_df['degradation_afrr_pos'] +
            results_df['degradation_afrr_neg']
        )

        results_df['gross_revenue'] = (
            results_df['fcr_revenue'] +
            results_df['afrr_capacity_revenue'] +
            results_df['afrr_energy_revenue'] +
            results_df['energy_arbitrage_revenue']
        )
        
        # Net profit (after degradation)
        results_df['net_profit'] = results_df['gross_revenue'] - results_df['degradation_total']
        
        # Total revenue (backwards compatibility with v2.x)
        results_df['total_revenue'] = results_df['gross_revenue']
        
        return results_df
    
    def print_performance_summary(self, results_df: pd.DataFrame):

        print("\n" + "="*70)
        print("PERFORMANCE RESULTS - SECTION 6")
        print("="*70)
        
        # Get total periods count (needed for percentage calculations)
        total_periods = len(results_df)
        
        # Revenue breakdown
        gross_revenue = results_df['gross_revenue'].sum()
        total_revenue = results_df['total_revenue'].sum()  # Backwards compatibility
        fcr_revenue = results_df['fcr_revenue'].sum()
        afrr_revenue = results_df['afrr_capacity_revenue'].sum() + results_df['afrr_energy_revenue'].sum()
        energy_revenue = results_df['energy_arbitrage_revenue'].sum()
        
        # Degradation costs
        total_degradation = results_df['degradation_total'].sum()
        degradation_arbitrage = results_df['degradation_arbitrage'].sum()
        degradation_fcr = results_df['degradation_fcr'].sum()
        degradation_afrr_pos = results_df['degradation_afrr_pos'].sum()
        degradation_afrr_neg = results_df['degradation_afrr_neg'].sum()
        
        # Net profit
        net_profit = results_df['net_profit'].sum()
        
        print(f"\n{'='*70}")
        print(f"GROSS ANNUAL REVENUE (before degradation): €{gross_revenue:,.0f}")
        print(f"{'='*70}")
        print(f"  FCR Revenue: €{fcr_revenue:,.0f} ({fcr_revenue/gross_revenue*100:.1f}%)")
        print(f"  aFRR Revenue: €{afrr_revenue:,.0f} ({afrr_revenue/gross_revenue*100:.1f}%)")
        print(f"    - Capacity: €{results_df['afrr_capacity_revenue'].sum():,.0f}")
        print(f"    - Energy (activations): €{results_df['afrr_energy_revenue'].sum():,.0f}")
        print(f"  Energy Arbitrage: €{energy_revenue:,.0f} ({energy_revenue/gross_revenue*100:.1f}%)")
        
        print(f"\n{'='*70}")
        print(f"DEGRADATION COSTS (DOD-based model): €{total_degradation:,.0f}")
        print(f"{'='*70}")
        print(f"  Arbitrage: €{degradation_arbitrage:,.0f} ({degradation_arbitrage/total_degradation*100:.1f}%)")
        print(f"  FCR: €{degradation_fcr:,.0f} ({degradation_fcr/total_degradation*100:.1f}%)")
        print(f"  aFRR Positive: €{degradation_afrr_pos:,.0f} ({degradation_afrr_pos/total_degradation*100:.1f}%)")
        print(f"  aFRR Negative: €{degradation_afrr_neg:,.0f} ({degradation_afrr_neg/total_degradation*100:.1f}%)")
        
        print(f"\n{'='*70}")
        print(f"NET PROFIT (after degradation): €{net_profit:,.0f}")
        print(f"{'='*70}")
        print(f"  Revenue per MW: €{net_profit/self.hw.nominal_power_mw:,.0f}/MW/year")
        print(f"  Degradation as % of Gross: {total_degradation/gross_revenue*100:.1f}%")
        
        # Market usage breakdown for arbitrage
        ida_premium = results_df['ida_premium'].sum()
        da_equivalent = results_df['arbitrage_revenue_da_equivalent'].sum()
        
        sell_on_ida = (results_df['sell_market'] == 'IDA').sum()
        sell_on_da = (results_df['sell_market'] == 'DA').sum()
        buy_on_ida = (results_df['buy_market'] == 'IDA').sum()
        buy_on_da = (results_df['buy_market'] == 'DA').sum()
        
        discharge_rev_da = results_df['discharge_revenue_da'].sum()
        discharge_rev_ida = results_df['discharge_revenue_ida'].sum()
        charge_cost_da = results_df['charge_cost_da'].sum()
        charge_cost_ida = results_df['charge_cost_ida'].sum()
        
        print(f"\n  Arbitrage Market Usage:")
        print(f"    Discharge (Sell):")
        print(f"      Day-Ahead: {sell_on_da:,} periods ({sell_on_da/total_periods*100:.1f}%) → Revenue: €{discharge_rev_da:,.0f}")
        print(f"      Intraday:  {sell_on_ida:,} periods ({sell_on_ida/total_periods*100:.1f}%) → Revenue: €{discharge_rev_ida:,.0f}")
        print(f"    Charge (Buy):")
        print(f"      Day-Ahead: {buy_on_da:,} periods ({buy_on_da/total_periods*100:.1f}%) → Cost: €{charge_cost_da:,.0f}")
        print(f"      Intraday:  {buy_on_ida:,} periods ({buy_on_ida/total_periods*100:.1f}%) → Cost: €{charge_cost_ida:,.0f}")
        print(f"    IDA Premium: €{ida_premium:,.0f} ({ida_premium/energy_revenue*100:.1f}% of arbitrage)")
        print(f"    DA Equivalent: €{da_equivalent:,.0f} (if only DA used)")

        
        # Allocation statistics
        print(f"\nAllocation Statistics:")
        
        # FCR
        fcr_high_count = (results_df['fcr_allocated_mw'] == 0.40 * self.hw.nominal_power_mw).sum()
        fcr_medium_count = (results_df['fcr_allocated_mw'] == 0.20 * self.hw.nominal_power_mw).sum()
        fcr_none_count = (results_df['fcr_allocated_mw'] == 0).sum()
        
        print(f"  FCR: {(total_periods - fcr_none_count)/total_periods*100:.1f}% of periods")
        print(f"    HIGH (40%): {fcr_high_count/total_periods*100:.1f}%")
        print(f"    MEDIUM (20%): {fcr_medium_count/total_periods*100:.1f}%")
        print(f"    NONE: {fcr_none_count/total_periods*100:.1f}%")
        
        # aFRR Positive
        afrr_pos_periods = (results_df['afrr_pos_allocated_mw'] > 0).sum()
        print(f"  aFRR Positive: {afrr_pos_periods/total_periods*100:.1f}% of periods")
        
        # aFRR Negative
        afrr_neg_periods = (results_df['afrr_neg_allocated_mw'] > 0).sum()
        print(f"  aFRR Negative: {afrr_neg_periods/total_periods*100:.1f}% of periods")
        
        # Operational metrics
        print(f"\nOperational Metrics:")
        
        # FIX: Use effective (degraded) capacity for accurate FEC calculation
        total_cycles = results_df['discharge_mw'].sum() * 0.25 / current_capacity
        print(f"  Total Cycles (FEC): {total_cycles:.1f} ({total_cycles/365:.2f} cycles/day average)")
        print(f"  Effective Capacity: {current_capacity:.2f} MWh (SOH: {current_soh:.2%})")
        
        avg_soc = results_df['soc'].mean()
        print(f"  Average SOC: {avg_soc*100:.1f}%")
        
        # Buffer compliance check
        buffer_violations = 0
        buffer_checks = 0
        
        for idx, row in results_df.iterrows():
            block_idx = row['period'] // 16
            afrr_pos = row['afrr_pos_allocated_mw']
            afrr_neg = row['afrr_neg_allocated_mw']
            soc = row['soc']
            
            if afrr_pos > 0:
                buffer_pos = (afrr_pos * 1.0) / self.hw.nominal_capacity_mwh
                min_required = self.hw.soc_min + buffer_pos
                if soc < min_required - 0.001:  # Small tolerance for floating point
                    buffer_violations += 1
                buffer_checks += 1
            
            if afrr_neg > 0:
                buffer_neg = (afrr_neg * 1.0) / self.hw.nominal_capacity_mwh
                max_allowed = self.hw.soc_max - buffer_neg
                if soc > max_allowed + 0.001:  # Small tolerance for floating point
                    buffer_violations += 1
                buffer_checks += 1
        
        if buffer_checks > 0:
            buffer_compliance = (buffer_checks - buffer_violations) / buffer_checks
            print(f"  aFRR Buffer Compliance: {buffer_compliance*100:.1f}% ({buffer_checks - buffer_violations:,} / {buffer_checks:,} periods)")
        else:
            print(f"  aFRR Buffer Compliance: N/A (no aFRR allocations)")
        
        # NEW: Detailed allocation rejection analysis
        print(f"\n" + "="*70)
        print("ALLOCATION REJECTION ANALYSIS")
        print("="*70)
        
        # Analyze by blocks (since allocation is per 4h block)
        results_df_copy = results_df.copy()
        results_df_copy['block'] = results_df_copy['period'] // 16
        blocks_df = results_df_copy.groupby(['day', 'block']).agg({
            'afrr_pos_allocated_mw': 'first',
            'afrr_neg_allocated_mw': 'first',
            'afrr_cap_pos': 'first',
            'afrr_cap_neg': 'first',
            'soc': 'first'
        }).reset_index()
        
        total_blocks = len(blocks_df)
        buffer_soc_12_5mw = 12.5 / self.hw.nominal_capacity_mwh  # Buffer for 12.5 MW
        
        # aFRR Positive rejections
        no_pos_blocks = blocks_df[blocks_df['afrr_pos_allocated_mw'] == 0]
        price_reject_pos = 0
        soc_reject_pos = 0
        
        for _, block in no_pos_blocks.iterrows():
            # Check if SOC was the constraint
            if block['soc'] < (self.hw.soc_min + buffer_soc_12_5mw):
                soc_reject_pos += 1
            else:
                # Otherwise, price was too low
                price_reject_pos += 1
        
        print(f"\naFRR POSITIVE - Not allocated: {len(no_pos_blocks)} blocks ({len(no_pos_blocks)/total_blocks*100:.1f}%)")
        if len(no_pos_blocks) > 0:
            print(f"  Due to LOW PRICES: {price_reject_pos} blocks ({price_reject_pos/len(no_pos_blocks)*100:.1f}% of rejections)")
            print(f"  Due to SOC CONSTRAINT: {soc_reject_pos} blocks ({soc_reject_pos/len(no_pos_blocks)*100:.1f}% of rejections)")
            print(f"    SOC too low (< {(self.hw.soc_min + buffer_soc_12_5mw)*100:.1f}%)")
        
        # aFRR Negative rejections
        no_neg_blocks = blocks_df[blocks_df['afrr_neg_allocated_mw'] == 0]
        price_reject_neg = 0
        soc_reject_neg = 0
        
        for _, block in no_neg_blocks.iterrows():
            # Check if SOC was the constraint
            if block['soc'] > (self.hw.soc_max - buffer_soc_12_5mw):
                soc_reject_neg += 1
            else:
                # Otherwise, price was too low
                price_reject_neg += 1
        
        print(f"\naFRR NEGATIVE - Not allocated: {len(no_neg_blocks)} blocks ({len(no_neg_blocks)/total_blocks*100:.1f}%)")
        if len(no_neg_blocks) > 0:
            print(f"  Due to LOW PRICES: {price_reject_neg} blocks ({price_reject_neg/len(no_neg_blocks)*100:.1f}% of rejections)")
            print(f"  Due to SOC CONSTRAINT: {soc_reject_neg} blocks ({soc_reject_neg/len(no_neg_blocks)*100:.1f}% of rejections)")
            print(f"    SOC too high (> {(self.hw.soc_max - buffer_soc_12_5mw)*100:.1f}%)")
        
        # Price statistics
        if len(no_pos_blocks) > 0:
            with_pos = blocks_df[blocks_df['afrr_pos_allocated_mw'] > 0]
            print(f"\n  aFRR Pos rejected blocks avg price: €{no_pos_blocks['afrr_cap_pos'].mean():.2f}/MW/h")
            print(f"  aFRR Pos accepted blocks avg price: €{with_pos['afrr_cap_pos'].mean():.2f}/MW/h")
        
        if len(no_neg_blocks) > 0:
            with_neg = blocks_df[blocks_df['afrr_neg_allocated_mw'] > 0]
            print(f"  aFRR Neg rejected blocks avg price: €{no_neg_blocks['afrr_cap_neg'].mean():.2f}/MW/h")
            print(f"  aFRR Neg accepted blocks avg price: €{with_neg['afrr_cap_neg'].mean():.2f}/MW/h")
        
        print("\n" + "="*70)


if __name__ == "__main__":
    # Load configuration and market data
    print("\n" + "="*70)
    print("LOADING CONFIGURATION FROM EXCEL")
    print("="*70)
    
    # Load battery config from Excel
    excel_file = 'input data/Battery_Manufacturer_Data_Template_rev_pvm.xlsx'
    config = CompleteBESSConfig.from_excel(excel_file, verbose=True)
    
    # Load market data
    market_data = load_market_data_2026()
    
    # Run optimizer
    print("\n" + "="*70)
    print("BESS DISPATCH OPTIMIZER - VERSION 2.0")
    print("="*70)
    
    optimizer = BESSDispatchOptimizer(config, market_data)
    results = optimizer.optimize_full_year()
    
    # Create Annual Summary
    print("\n" + "="*70)
    print("CREATING ANNUAL SUMMARY")
    print("="*70)
    
    summary_data = []
    
    # Revenue Summary
    summary_data.append({'Metric': 'REVENUE SUMMARY', 'Value': ''})
    summary_data.append({'Metric': 'Total FCR Revenue (€)', 'Value': results['fcr_revenue'].sum()})
    summary_data.append({'Metric': 'Total aFRR Revenue (€)', 'Value': results['afrr_capacity_revenue'].sum() + results['afrr_energy_revenue'].sum()})
    summary_data.append({'Metric': '  - aFRR Capacity Revenue (€)', 'Value': results['afrr_capacity_revenue'].sum()})
    summary_data.append({'Metric': '  - aFRR Energy Revenue (€)', 'Value': results['afrr_energy_revenue'].sum()})
    summary_data.append({'Metric': 'Total Energy Arbitrage Revenue (€)', 'Value': results['energy_arbitrage_revenue'].sum()})
    summary_data.append({'Metric': 'GROSS REVENUE (before degradation) (€)', 'Value': results['gross_revenue'].sum()})
    summary_data.append({'Metric': '', 'Value': ''})
    
    # Degradation Costs
    summary_data.append({'Metric': 'DEGRADATION COSTS (DOD-based)', 'Value': ''})
    summary_data.append({'Metric': 'Arbitrage Degradation (€)', 'Value': results['degradation_arbitrage'].sum()})
    summary_data.append({'Metric': 'FCR Degradation (€)', 'Value': results['degradation_fcr'].sum()})
    summary_data.append({'Metric': 'aFRR Positive Degradation (€)', 'Value': results['degradation_afrr_pos'].sum()})
    summary_data.append({'Metric': 'aFRR Negative Degradation (€)', 'Value': results['degradation_afrr_neg'].sum()})
    summary_data.append({'Metric': 'TOTAL DEGRADATION COST (€)', 'Value': results['degradation_total'].sum()})
    summary_data.append({'Metric': '', 'Value': ''})
    
    # Net Profit
    summary_data.append({'Metric': 'NET PROFIT (after degradation) (€)', 'Value': results['net_profit'].sum()})
    summary_data.append({'Metric': 'Net Profit per MW (€/MW/year)', 'Value': results['net_profit'].sum() / config.hardware.nominal_power_mw})
    summary_data.append({'Metric': 'Degradation as % of Gross Revenue', 'Value': f"{results['degradation_total'].sum() / results['gross_revenue'].sum() * 100:.1f}%"})
    summary_data.append({'Metric': '', 'Value': ''})
    
    # Backwards compatibility
    summary_data.append({'Metric': 'Total Revenue (€) [legacy]', 'Value': results['total_revenue'].sum()})
    summary_data.append({'Metric': '', 'Value': ''})
    
    # Allocation Statistics
    summary_data.append({'Metric': 'ALLOCATION STATISTICS', 'Value': ''})
    
    # FCR
    fcr_high_count = (results['fcr_allocated_mw'] == 0.40 * config.hardware.nominal_power_mw).sum()
    fcr_medium_count = (results['fcr_allocated_mw'] == 0.20 * config.hardware.nominal_power_mw).sum()
    fcr_none_count = (results['fcr_allocated_mw'] == 0).sum()
    total_periods = len(results)
    
    summary_data.append({'Metric': 'FCR Allocation - HIGH (40%)', 'Value': f"{fcr_high_count/total_periods*100:.1f}%"})
    summary_data.append({'Metric': 'FCR Allocation - MEDIUM (20%)', 'Value': f"{fcr_medium_count/total_periods*100:.1f}%"})
    summary_data.append({'Metric': 'FCR Allocation - NONE', 'Value': f"{fcr_none_count/total_periods*100:.1f}%"})
    
    # aFRR Positive
    afrr_pos_high = (results['afrr_pos_allocated_mw'] == 0.25 * config.hardware.nominal_power_mw).sum()
    afrr_pos_medium = (results['afrr_pos_allocated_mw'] == 0.125 * config.hardware.nominal_power_mw).sum()
    afrr_pos_none = (results['afrr_pos_allocated_mw'] == 0).sum()
    
    summary_data.append({'Metric': 'aFRR Pos Allocation - HIGH (25%)', 'Value': f"{afrr_pos_high/total_periods*100:.1f}%"})
    summary_data.append({'Metric': 'aFRR Pos Allocation - MEDIUM (12.5%)', 'Value': f"{afrr_pos_medium/total_periods*100:.1f}%"})
    summary_data.append({'Metric': 'aFRR Pos Allocation - NONE', 'Value': f"{afrr_pos_none/total_periods*100:.1f}%"})
    
    # aFRR Negative
    afrr_neg_high = (results['afrr_neg_allocated_mw'] == 0.25 * config.hardware.nominal_power_mw).sum()
    afrr_neg_medium = (results['afrr_neg_allocated_mw'] == 0.125 * config.hardware.nominal_power_mw).sum()
    afrr_neg_none = (results['afrr_neg_allocated_mw'] == 0).sum()
    
    summary_data.append({'Metric': 'aFRR Neg Allocation - HIGH (25%)', 'Value': f"{afrr_neg_high/total_periods*100:.1f}%"})
    summary_data.append({'Metric': 'aFRR Neg Allocation - MEDIUM (12.5%)', 'Value': f"{afrr_neg_medium/total_periods*100:.1f}%"})
    summary_data.append({'Metric': 'aFRR Neg Allocation - NONE', 'Value': f"{afrr_neg_none/total_periods*100:.1f}%"})
    summary_data.append({'Metric': '', 'Value': ''})
    
    # Market Usage Statistics for Arbitrage
    summary_data.append({'Metric': 'ARBITRAGE MARKET USAGE', 'Value': ''})
    
    sell_on_ida = (results['sell_market'] == 'IDA').sum()
    sell_on_da = (results['sell_market'] == 'DA').sum()
    buy_on_ida = (results['buy_market'] == 'IDA').sum()
    buy_on_da = (results['buy_market'] == 'DA').sum()
    
    summary_data.append({'Metric': 'Discharge - Day-Ahead (%)', 'Value': f"{sell_on_da/total_periods*100:.1f}%"})
    summary_data.append({'Metric': 'Discharge - Intraday (%)', 'Value': f"{sell_on_ida/total_periods*100:.1f}%"})
    summary_data.append({'Metric': 'Charge - Day-Ahead (%)', 'Value': f"{buy_on_da/total_periods*100:.1f}%"})
    summary_data.append({'Metric': 'Charge - Intraday (%)', 'Value': f"{buy_on_ida/total_periods*100:.1f}%"})
    
    ida_premium = results['ida_premium'].sum()
    da_equivalent = results['arbitrage_revenue_da_equivalent'].sum()
    discharge_rev_da = results['discharge_revenue_da'].sum()
    discharge_rev_ida = results['discharge_revenue_ida'].sum()
    charge_cost_da = results['charge_cost_da'].sum()
    charge_cost_ida = results['charge_cost_ida'].sum()
    
    summary_data.append({'Metric': '', 'Value': ''})
    summary_data.append({'Metric': 'Discharge Revenue - Day-Ahead (€)', 'Value': discharge_rev_da})
    summary_data.append({'Metric': 'Discharge Revenue - Intraday (€)', 'Value': discharge_rev_ida})
    summary_data.append({'Metric': 'Charge Cost - Day-Ahead (€)', 'Value': charge_cost_da})
    summary_data.append({'Metric': 'Charge Cost - Intraday (€)', 'Value': charge_cost_ida})
    summary_data.append({'Metric': 'IDA Premium (€)', 'Value': ida_premium})
    summary_data.append({'Metric': 'DA Equivalent Revenue (€)', 'Value': da_equivalent})
    summary_data.append({'Metric': '', 'Value': ''})
    
    # Operational Metrics
    summary_data.append({'Metric': 'OPERATIONAL METRICS', 'Value': ''})
    
    # FIX: Calculate effective capacity based on SOH for accurate FEC
    # For single-year simulation, operational_year = 0
    operational_year = 0
    _, current_soh, current_capacity = optimizer.get_yearly_parameters(operational_year)
    
    # Use effective capacity for FEC calculation
    total_cycles = results['discharge_mw'].sum() * 0.25 / current_capacity
    summary_data.append({'Metric': 'Effective Capacity (MWh)', 'Value': current_capacity})
    summary_data.append({'Metric': 'State of Health (SOH)', 'Value': f"{current_soh:.2%}"})
    summary_data.append({'Metric': 'Total Cycles (FEC)', 'Value': total_cycles})
    summary_data.append({'Metric': 'Average Cycles per Day', 'Value': total_cycles/365})
    summary_data.append({'Metric': 'Average SOC (%)', 'Value': results['soc'].mean() * 100})
    summary_data.append({'Metric': 'SOC Standard Deviation (%)', 'Value': results['soc'].std() * 100})
    summary_data.append({'Metric': '', 'Value': ''})
    
    # Price Threshold Statistics
    summary_data.append({'Metric': 'PRICE THRESHOLD STATISTICS', 'Value': ''})
    
    # FCR threshold stats
    fcr_high_threshold_mean = optimizer.thresholds.fcr_threshold_high_eur_mw_4h.mean()
    fcr_medium_threshold_mean = optimizer.thresholds.fcr_threshold_medium_eur_mw_4h.mean()
    summary_data.append({'Metric': 'FCR HIGH Threshold (€/MW/4h) - Mean', 'Value': fcr_high_threshold_mean})
    summary_data.append({'Metric': 'FCR MEDIUM Threshold (€/MW/4h) - Mean', 'Value': fcr_medium_threshold_mean})
    
    # aFRR threshold stats
    afrr_pos_high_threshold_mean = optimizer.thresholds.afrr_pos_threshold_high_eur_mw_4h.mean()
    afrr_pos_medium_threshold_mean = optimizer.thresholds.afrr_pos_threshold_medium_eur_mw_4h.mean()
    summary_data.append({'Metric': 'aFRR Pos HIGH Threshold (€/MW/4h) - Mean', 'Value': afrr_pos_high_threshold_mean})
    summary_data.append({'Metric': 'aFRR Pos MEDIUM Threshold (€/MW/4h) - Mean', 'Value': afrr_pos_medium_threshold_mean})
    
    afrr_neg_high_threshold_mean = optimizer.thresholds.afrr_neg_threshold_high_eur_mw_4h.mean()
    afrr_neg_medium_threshold_mean = optimizer.thresholds.afrr_neg_threshold_medium_eur_mw_4h.mean()
    summary_data.append({'Metric': 'aFRR Neg HIGH Threshold (€/MW/4h) - Mean', 'Value': afrr_neg_high_threshold_mean})
    summary_data.append({'Metric': 'aFRR Neg MEDIUM Threshold (€/MW/4h) - Mean', 'Value': afrr_neg_medium_threshold_mean})
    summary_data.append({'Metric': '', 'Value': ''})
    
    # Market Price Statistics (for allocated blocks)
    summary_data.append({'Metric': 'MARKET PRICE STATISTICS (Allocated Blocks)', 'Value': ''})
    
    # Get prices for allocated blocks only
    results['block'] = results['period'] // 16
    blocks_df = results.groupby(['day', 'block']).agg({
        'fcr_allocated_mw': 'first',
        'fcr_price': 'first',
        'afrr_pos_allocated_mw': 'first',
        'afrr_cap_pos': 'first',        'afrr_neg_allocated_mw': 'first',
        'afrr_cap_neg': 'first'
    }).reset_index()
    
    # FCR prices when allocated HIGH
    fcr_high_blocks = blocks_df[blocks_df['fcr_allocated_mw'] == 0.40 * config.hardware.nominal_power_mw]
    if len(fcr_high_blocks) > 0:
        fcr_high_price_mean = fcr_high_blocks['fcr_price'].mean() * 4  # Convert to €/MW/4h
        summary_data.append({'Metric': 'FCR Price - HIGH Allocation (€/MW/4h)', 'Value': fcr_high_price_mean})
    
    # FCR prices when allocated MEDIUM
    fcr_medium_blocks = blocks_df[blocks_df['fcr_allocated_mw'] == 0.20 * config.hardware.nominal_power_mw]
    if len(fcr_medium_blocks) > 0:
        fcr_medium_price_mean = fcr_medium_blocks['fcr_price'].mean() * 4
        summary_data.append({'Metric': 'FCR Price - MEDIUM Allocation (€/MW/4h)', 'Value': fcr_medium_price_mean})
    
    # aFRR Pos prices when allocated HIGH
    afrr_pos_high_blocks = blocks_df[blocks_df['afrr_pos_allocated_mw'] == 0.25 * config.hardware.nominal_power_mw]
    if len(afrr_pos_high_blocks) > 0:
        afrr_pos_high_price_mean = afrr_pos_high_blocks['afrr_cap_pos'].mean() * 4
        summary_data.append({'Metric': 'aFRR Pos Price - HIGH Allocation (€/MW/4h)', 'Value': afrr_pos_high_price_mean})
    
    # aFRR Pos prices when allocated MEDIUM
    afrr_pos_medium_blocks = blocks_df[blocks_df['afrr_pos_allocated_mw'] == 0.125 * config.hardware.nominal_power_mw]
    if len(afrr_pos_medium_blocks) > 0:
        afrr_pos_medium_price_mean = afrr_pos_medium_blocks['afrr_cap_pos'].mean() * 4
        summary_data.append({'Metric': 'aFRR Pos Price - MEDIUM Allocation (€/MW/4h)', 'Value': afrr_pos_medium_price_mean})
    
    # aFRR Neg prices when allocated HIGH
    afrr_neg_high_blocks = blocks_df[blocks_df['afrr_neg_allocated_mw'] == 0.25 * config.hardware.nominal_power_mw]
    if len(afrr_neg_high_blocks) > 0:
        afrr_neg_high_price_mean = afrr_neg_high_blocks['afrr_cap_neg'].mean() * 4
        summary_data.append({'Metric': 'aFRR Neg Price - HIGH Allocation (€/MW/4h)', 'Value': afrr_neg_high_price_mean})
    
    # aFRR Neg prices when allocated MEDIUM
    afrr_neg_medium_blocks = blocks_df[blocks_df['afrr_neg_allocated_mw'] == 0.125 * config.hardware.nominal_power_mw]
    if len(afrr_neg_medium_blocks) > 0:
        afrr_neg_medium_price_mean = afrr_neg_medium_blocks['afrr_cap_neg'].mean() * 4
        summary_data.append({'Metric': 'aFRR Neg Price - MEDIUM Allocation (€/MW/4h)', 'Value': afrr_neg_medium_price_mean})
    
    summary_df = pd.DataFrame(summary_data)
    
    # Organize detailed results columns for Excel export
    # Group related columns together for easier analysis
    detailed_columns = [
        # Time identifiers
        'timestamp', 'day', 'period', 'block',
        
        # Market prices
        'da_price', 'ida_bid', 'ida_ask', 'ida_mid',
        'fcr_price', 'afrr_cap_pos', 'afrr_cap_neg',
        
        # Reserve allocations
        'fcr_allocated_mw', 'afrr_pos_allocated_mw', 'afrr_neg_allocated_mw',
        'available_power_mw',
        
        # Arbitrage operations and market selection
        'charge_mw', 'buy_market', 'buy_price_used', 'buy_advantage',
        'discharge_mw', 'sell_market', 'sell_price_used', 'sell_advantage',
        
        # State of charge
        'soc', 'soc_arbitrage_only', 'soc_activation_delta',
        
        # Reserve activations
        'fcr_activation_mw', 'afrr_pos_activation_mwh', 'afrr_neg_activation_mwh',
        
        # Revenues (period-level)
        'fcr_revenue', 'afrr_capacity_revenue', 'afrr_energy_revenue',
        'energy_arbitrage_revenue', 'arbitrage_revenue_da_equivalent', 'ida_premium',
        'discharge_revenue_da', 'discharge_revenue_ida',
        'charge_cost_da', 'charge_cost_ida',
        'total_revenue'
    ]
    
    # Reorder columns (keep all columns, but prioritize the ones in detailed_columns)
    existing_columns = [col for col in detailed_columns if col in results.columns]
    remaining_columns = [col for col in results.columns if col not in detailed_columns]
    results_organized = results[existing_columns + remaining_columns]
    
    # Save results to Excel with multiple sheets
    output_file = 'BESS_Dispatch_Results_2026_v4.xlsx'
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        # Annual Summary sheet
        summary_df.to_excel(writer, sheet_name='Annual Summary', index=False)
        
        # Detailed results sheet (organized columns)
        results_organized.to_excel(writer, sheet_name='Detailed 15-Min Data', index=False)
    
    # Print performance summary (Section 6 metrics)
    optimizer.print_performance_summary(results)
    
    print(f"\n{'='*70}")
    print(f"Results saved to: {output_file}")
    print(f"  - Annual Summary sheet: {len(summary_df)} rows")
    print(f"  - Detailed 15-Min Data sheet: {len(results):,} periods")
    print(f"{'='*70}")
