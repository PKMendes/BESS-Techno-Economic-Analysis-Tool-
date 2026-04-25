
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# Import required modules
from bess_dispatch_optimizer import BESSDispatchOptimizer, CompleteBESSConfig
from load_market_data import load_all_market_years


@dataclass
class CashFlowYear:
    """Cash flow data for a single year"""
    year: int
    warranty_year: int
    market_year: int
    
    # Operational metrics
    gross_revenue: float
    degradation_cost: float
    net_revenue: float
    opex: float
    
    # Financial calculations
    ebitda: float  # Net Revenue - OPEX
    depreciation: float
    ebit: float  # EBITDA - Depreciation
    tax: float
    net_income: float
    cash_flow: float  # Net Income + Depreciation
    cumulative_cash_flow: float
    discounted_cash_flow: float
    cumulative_discounted_cash_flow: float
    
    # Battery health
    rte: float
    soh: float
    effective_capacity_mwh: float
    cycles: float


class BESSFinancialAnalyzer:
    
    def __init__(
        self,
        battery_config_file: str,
        market_data_file: str,
        discount_rate: float = 0.08,
        tax_rate: float = 0.30,
        depreciation_years: int = 10,
        start_market_year: int = 2026
    ):

        self.battery_config_file = battery_config_file
        self.market_data_file = market_data_file
        self.discount_rate = discount_rate
        self.tax_rate = tax_rate
        self.depreciation_years = depreciation_years
        self.start_market_year = start_market_year
        
        # Load battery configuration
        print(f"\n{'='*80}")
        print("LOADING BATTERY CONFIGURATION")
        print(f"{'='*80}")
        self.config = CompleteBESSConfig.from_excel(battery_config_file, verbose=True)
        
        # Get warranty years
        self.warranty_years = self._get_warranty_years()
        
        # Validate OPEX vs warranty years
        self._validate_opex_warranty_match()
        
        # Extract financial parameters
        self.capex = self.config.financial.capex_total_eur
        self.annual_depreciation = self.capex / depreciation_years if depreciation_years > 0 else 0
        
        # Load all market data at once (optimized)
        self.market_data_dict = self._load_all_market_data()
        
        # Storage for results
        self.yearly_results: Dict[int, pd.DataFrame] = {}
        self.cash_flows: List[CashFlowYear] = []
        
    def _get_warranty_years(self) -> List[int]:
        """Extract warranty years from degradation curve"""
        print(f"\n{'='*80}")
        print("DETERMINING WARRANTY PERIOD")
        print(f"{'='*80}")
        
        df = pd.read_excel(self.battery_config_file, sheet_name='Degradation Curve', header=0)
        
        warranty_years = []
        for idx, row in df.iterrows():
            year_label = str(row.iloc[0]).strip()
            if year_label.startswith('Year'):
                try:
                    year_num = int(year_label.split()[1])
                    rte = float(row.iloc[2])  # RTE column
                    soh = float(row.iloc[3])  # SOH column
                    
                    if pd.notna(rte) and pd.notna(soh):
                        warranty_years.append(year_num)
                except (ValueError, IndexError):
                    continue
        
        if not warranty_years:
            raise ValueError("No valid warranty years found in degradation curve!")
        
        print(f"✓ Warranty period: Year {min(warranty_years)} to Year {max(warranty_years)}")
        print(f"✓ Total warranty years: {len(warranty_years)}")
        
        return warranty_years
    
    def _validate_opex_warranty_match(self):
        """Validate that OPEX years match warranty years"""
        print(f"\n{'='*80}")
        print("VALIDATING OPEX vs WARRANTY YEARS")
        print(f"{'='*80}")
        
        if not self.config.financial.opex_curve:
            raise ValueError("❌ ERROR: No OPEX curve found in battery configuration!")
        
        opex_years = [year for year, _ in self.config.financial.opex_curve]
        warranty_years_set = set(self.warranty_years)
        opex_years_set = set(opex_years)
        
        print(f"✓ Warranty years: {sorted(warranty_years_set)}")
        print(f"✓ OPEX years: {sorted(opex_years_set)}")
        
        # Check if OPEX covers all warranty years
        missing_opex_years = warranty_years_set - opex_years_set
        extra_opex_years = opex_years_set - warranty_years_set
        
        if missing_opex_years:
            print(f"\n❌ ERROR: OPEX data missing for warranty years: {sorted(missing_opex_years)}")
            raise ValueError(
                f"OPEX data incomplete! Missing years: {sorted(missing_opex_years)}\n"
                f"Please ensure OPEX sheet has entries for all warranty years."
            )
        
        if extra_opex_years:
            print(f"⚠️  Warning: OPEX data exists for non-warranty years: {sorted(extra_opex_years)}")
            print("   These will be ignored in the analysis.")
        
        print(f"\n✓ OPEX validation passed: All {len(warranty_years_set)} warranty years have OPEX data")
    
    def _load_all_market_data(self) -> Dict[int, object]:
        """Load all required market data years at once"""
        print(f"\n{'='*80}")
        print("LOADING MARKET DATA")
        print(f"{'='*80}")
        
        max_warranty_year = max(self.warranty_years)
        num_years = max_warranty_year + 1
        
        market_data_dict = load_all_market_years(
            forecast_file=self.market_data_file,
            start_year=self.start_market_year,
            num_years=num_years,
            verbose=True
        )
        
        # Check which warranty years have market data
        missing_market_years = []
        for warranty_year in self.warranty_years:
            market_year = self.start_market_year + warranty_year
            if market_year not in market_data_dict:
                missing_market_years.append(market_year)
        
        if missing_market_years:
            print(f"\n⚠️  WARNING: Market data missing for years: {missing_market_years}")
            print(f"   Will only simulate years with available market data")
        
        return market_data_dict
    
    def _calculate_irr_manual(self, cash_flows: List[float], guess: float = 0.1) -> float:

        def npv_at_rate(rate, cfs):
            """Calculate NPV at a given rate"""
            return sum(cf / ((1 + rate) ** i) for i, cf in enumerate(cfs))
        
        def npv_derivative(rate, cfs):
            """Calculate derivative of NPV with respect to rate"""
            return sum(-i * cf / ((1 + rate) ** (i + 1)) for i, cf in enumerate(cfs))
        
        # Newton-Raphson iteration
        rate = guess
        for iteration in range(100):  # Max 100 iterations
            npv = npv_at_rate(rate, cash_flows)
            
            # Check convergence
            if abs(npv) < 0.01:  # Converged
                return rate
            
            # Newton-Raphson step
            derivative = npv_derivative(rate, cash_flows)
            if abs(derivative) < 1e-10:  # Avoid division by zero
                break
            
            rate = rate - npv / derivative
            
            # Keep rate in reasonable bounds
            if rate < -0.99 or rate > 10.0:
                break
        
        # If didn't converge, return None
        return None
    
    def run_multiyear_simulation(self):
        """Run dispatch optimization for all warranty years"""
        print(f"\n{'='*80}")
        print("RUNNING MULTI-YEAR DISPATCH OPTIMIZATION")
        print(f"{'='*80}")
        
        for warranty_year in self.warranty_years:
            market_year = self.start_market_year + warranty_year
            
            # Check if market data exists
            if market_year not in self.market_data_dict:
                print(f"\n⚠️  Skipping Year {warranty_year}: No market data for {market_year}")
                continue
            
            print(f"\n{'='*80}")
            print(f"SIMULATING WARRANTY YEAR {warranty_year} (Market Year {market_year})")
            print(f"{'='*80}")
            
            # Get year-specific parameters
            rte = self.config.degradation.get_rte_at_year(warranty_year)
            soh = self.config.degradation.get_soh_at_year(warranty_year)
            effective_capacity = self.config.hardware.nominal_capacity_mwh * soh
            
            print(f"RTE: {rte:.4f} ({rte*100:.2f}%)")
            print(f"SOH: {soh:.4f} ({soh*100:.2f}%)")
            print(f"Effective Capacity: {effective_capacity:.2f} MWh")
            
            # Get market data
            market_data = self.market_data_dict[market_year]
            
            # Run optimizer
            optimizer = BESSDispatchOptimizer(
                config=self.config,
                market_data=market_data,
                warranty_year=warranty_year
            )
            results = optimizer.optimize_full_year()
            
            # Store results
            self.yearly_results[warranty_year] = results
            
            # Print summary
            gross_revenue = results['gross_revenue'].sum()
            degradation = results['degradation_total'].sum()
            net_revenue = results['net_profit'].sum()
            
            print(f"\n✓ Year {warranty_year} Optimization Complete")
            print(f"  Gross Revenue: €{gross_revenue:,.0f}")
            print(f"  Degradation: €{degradation:,.0f}")
            print(f"  Net Revenue: €{net_revenue:,.0f}")
    
    def calculate_cash_flows(self):
        """Calculate year-by-year cash flows with financial modeling"""
        print(f"\n{'='*80}")
        print("CALCULATING CASH FLOWS")
        print(f"{'='*80}")
        
        cumulative_cash_flow = 0
        cumulative_discounted_cash_flow = -self.capex  # Initial investment
        
        for warranty_year in sorted(self.yearly_results.keys()):
            results = self.yearly_results[warranty_year]
            market_year = self.start_market_year + warranty_year
            
            # Operational metrics
            gross_revenue = results['gross_revenue'].sum()
            degradation_cost = results['degradation_total'].sum()
            net_revenue = results['net_profit'].sum()
            
            # Get OPEX for this year
            opex = self.config.financial.get_opex_at_year(warranty_year)
            
            # Financial calculations
            ebitda = net_revenue - opex
            
            # Apply depreciation (only for years within depreciation period)
            if warranty_year < self.depreciation_years:
                depreciation = self.annual_depreciation
            else:
                depreciation = 0
            
            ebit = ebitda - depreciation
            
            # Calculate tax (only on positive EBIT)
            tax = max(0, ebit * self.tax_rate)
            
            # Net income
            net_income = ebit - tax
            
            # Cash flow = Net Income + Depreciation (add back non-cash charge)
            cash_flow = net_income + depreciation
            
            # Cumulative cash flows
            cumulative_cash_flow += cash_flow
            
            # Discounted cash flow
            discount_factor = 1 / ((1 + self.discount_rate) ** (warranty_year + 1))
            discounted_cash_flow = cash_flow * discount_factor
            cumulative_discounted_cash_flow += discounted_cash_flow
            
            # Battery health
            rte = self.config.degradation.get_rte_at_year(warranty_year)
            soh = self.config.degradation.get_soh_at_year(warranty_year)
            effective_capacity = self.config.hardware.nominal_capacity_mwh * soh
            
            # Calculate cycles - FIX: Use effective (degraded) capacity for accurate FEC
            cycles = results['discharge_mw'].sum() * 0.25 / effective_capacity
            
            # Create cash flow record
            cf = CashFlowYear(
                year=warranty_year + 1,  # Year 1, 2, 3... for display
                warranty_year=warranty_year,
                market_year=market_year,
                gross_revenue=gross_revenue,
                degradation_cost=degradation_cost,
                net_revenue=net_revenue,
                opex=opex,
                ebitda=ebitda,
                depreciation=depreciation,
                ebit=ebit,
                tax=tax,
                net_income=net_income,
                cash_flow=cash_flow,
                cumulative_cash_flow=cumulative_cash_flow,
                discounted_cash_flow=discounted_cash_flow,
                cumulative_discounted_cash_flow=cumulative_discounted_cash_flow,
                rte=rte,
                soh=soh,
                effective_capacity_mwh=effective_capacity,
                cycles=cycles
            )
            
            self.cash_flows.append(cf)
            
            print(f"Year {warranty_year}: Cash Flow = €{cash_flow:,.0f}, "
                  f"Cumulative DCF = €{cumulative_discounted_cash_flow:,.0f}")
    
    def calculate_financial_metrics(self) -> Dict:
        """Calculate NPV, IRR, and payback periods"""
        print(f"\n{'='*80}")
        print("CALCULATING FINANCIAL METRICS")
        print(f"{'='*80}")
        
        # NPV calculation
        npv = -self.capex  # Initial investment
        for cf in self.cash_flows:
            discount_factor = 1 / ((1 + self.discount_rate) ** cf.year)
            npv += cf.cash_flow * discount_factor
        
        # IRR calculation
        cash_flow_series = [-self.capex] + [cf.cash_flow for cf in self.cash_flows]
        
        print(f"\nCash flow series for IRR calculation:")
        print(f"  Year 0 (Investment): €{-self.capex:,.2f}")
        for i, cf in enumerate(self.cash_flows):
            print(f"  Year {cf.year}: €{cf.cash_flow:,.2f}")
        
        irr = None
        try:
            # Try numpy_financial first (if installed)
            import numpy_financial as npf
            irr = npf.irr(cash_flow_series)
            print(f"\n✓ IRR calculated using numpy_financial: {irr*100:.2f}%")
        except ImportError:
            print(f"\n⚠️  numpy_financial not installed, using manual calculation...")
            # Fallback to manual IRR calculation
            try:
                irr = self._calculate_irr_manual(cash_flow_series)
                if irr is not None:
                    print(f"✓ IRR calculated using Newton-Raphson: {irr*100:.2f}%")
                else:
                    print(f"⚠️  IRR calculation did not converge")
            except Exception as e:
                print(f"❌ IRR calculation error: {e}")
                irr = None
        except Exception as e:
            print(f"❌ IRR calculation error: {e}")
            irr = None
        
        # Simple payback period
        cumulative = -self.capex
        simple_payback = None
        for cf in self.cash_flows:
            cumulative += cf.cash_flow
            if cumulative >= 0 and simple_payback is None:
                # Linear interpolation for fractional year
                prev_cumulative = cumulative - cf.cash_flow
                simple_payback = cf.year - 1 + abs(prev_cumulative) / cf.cash_flow
                break
        
        # Discounted payback period
        cumulative_dcf = -self.capex
        discounted_payback = None
        for cf in self.cash_flows:
            cumulative_dcf += cf.discounted_cash_flow
            if cumulative_dcf >= 0 and discounted_payback is None:
                prev_cumulative_dcf = cumulative_dcf - cf.discounted_cash_flow
                discounted_payback = cf.year - 1 + abs(prev_cumulative_dcf) / cf.discounted_cash_flow
                break
        
        # Lifetime totals
        total_gross_revenue = sum(cf.gross_revenue for cf in self.cash_flows)
        total_degradation = sum(cf.degradation_cost for cf in self.cash_flows)
        total_net_revenue = sum(cf.net_revenue for cf in self.cash_flows)
        total_opex = sum(cf.opex for cf in self.cash_flows)
        total_tax = sum(cf.tax for cf in self.cash_flows)
        total_cash_flow = sum(cf.cash_flow for cf in self.cash_flows)
        
        # Project status
        project_status = "PROFITABLE" if npv > 0 else "NOT PROFITABLE"
        
        metrics = {
            'npv': npv,
            'irr': irr * 100 if irr is not None else None,
            'simple_payback': simple_payback,
            'discounted_payback': discounted_payback,
            'total_gross_revenue': total_gross_revenue,
            'total_degradation': total_degradation,
            'total_net_revenue': total_net_revenue,
            'total_opex': total_opex,
            'total_tax': total_tax,
            'total_cash_flow': total_cash_flow,
            'project_status': project_status
        }
        
        print(f"\n{'='*60}")
        print(f"NPV: €{npv:,.2f}")
        if irr is not None:
            print(f"IRR: {irr*100:.2f}%")
        else:
            print(f"IRR: N/A (calculation did not converge)")
        print(f"Simple Payback: {simple_payback:.2f} years" if simple_payback else "Simple Payback: > Project Life")
        print(f"Discounted Payback: {discounted_payback:.2f} years" if discounted_payback else "Discounted Payback: > Project Life")
        print(f"Status: {project_status}")
        print(f"{'='*60}")
        
        return metrics
    
    def export_to_excel(self, output_file: str = 'BESS_Financial_Feasibility.xlsx'):
        """Export all results to a comprehensive Excel file"""
        print(f"\n{'='*80}")
        print("EXPORTING RESULTS TO EXCEL")
        print(f"{'='*80}")
        
        # Calculate financial metrics
        metrics = self.calculate_financial_metrics()
        
        # Ensure output directory exists
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Delete existing file if it exists (to ensure clean overwrite)
        if output_path.exists():
            try:
                output_path.unlink()
                print(f"  ✓ Deleted existing file: {output_path}")
            except Exception as e:
                print(f"  ⚠️  Warning: Could not delete existing file: {e}")
                print(f"  ⚠️  Attempting to overwrite anyway...")
        
        # Create Excel writer with explicit engine
        try:
            with pd.ExcelWriter(output_path, engine='openpyxl', mode='w') as writer:
                # Sheet 1: Executive Summary
                print("  Creating Executive Summary...")
                self._create_executive_summary(writer, metrics)
                
                # Sheet 2: Cash Flow Projection
                print("  Creating Cash Flow Projection...")
                self._create_cash_flow_projection(writer)
                
                # Sheet 3: Multi-Year Operational Summary
                print("  Creating Multi-Year Operational Summary...")
                self._create_multiyear_summary(writer)
                
                # Sheets 4+: Year-by-year summaries and details
                for warranty_year in sorted(self.yearly_results.keys()):
                    market_year = self.start_market_year + warranty_year
                    results = self.yearly_results[warranty_year]
                    
                    print(f"  Creating Year {warranty_year} sheets...")
                    
                    # Summary sheet
                    summary_df = self._create_year_summary(warranty_year, market_year, results)
                    sheet_name = f'Year_{warranty_year}_Summary'
                    summary_df.to_excel(writer, sheet_name=sheet_name, index=False)
                    
                    # Details sheet
                    details_df = self._organize_year_details(results)
                    sheet_name = f'Year_{warranty_year}_Details'
                    details_df.to_excel(writer, sheet_name=sheet_name, index=False)
            
            print(f"\n✓ Results exported to: {output_path}")
            print(f"  Total sheets: {3 + len(self.yearly_results) * 2}")
            print(f"  - Executive Summary")
            print(f"  - Cash Flow Projection")
            print(f"  - Multi-Year Operational Summary")
            for warranty_year in sorted(self.yearly_results.keys()):
                print(f"  - Year_{warranty_year}_Summary")
                print(f"  - Year_{warranty_year}_Details")
            
            # Verify file was created successfully
            if output_path.exists():
                file_size = output_path.stat().st_size / (1024 * 1024)  # Size in MB
                print(f"\n✓ File created successfully")
                print(f"  File size: {file_size:.2f} MB")
            else:
                print(f"\n❌ Warning: File not found after creation!")
                
        except PermissionError as e:
            print(f"\n❌ Error: Permission denied when writing to {output_path}")
            print(f"   Make sure the file is not open in Excel or another program.")
            print(f"   Error details: {e}")
            raise
        except Exception as e:
            print(f"\n❌ Error exporting to Excel: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def _create_executive_summary(self, writer, metrics):
        """Create executive summary sheet"""
        summary_data = []
        
        # Project Overview
        summary_data.append({'Metric': 'PROJECT OVERVIEW', 'Value': ''})
        summary_data.append({'Metric': 'Project Name', 'Value': 'BESS Energy Storage Project'})
        summary_data.append({'Metric': 'Analysis Date', 'Value': datetime.now().strftime('%Y-%m-%d')})
        summary_data.append({'Metric': 'Project Lifetime (years)', 'Value': len(self.cash_flows)})
        summary_data.append({'Metric': '', 'Value': ''})
        
        # Capital Investment
        summary_data.append({'Metric': 'CAPITAL INVESTMENT', 'Value': ''})
        summary_data.append({'Metric': 'Total CAPEX (€)', 'Value': self.capex})
        summary_data.append({'Metric': 'Depreciation Period (years)', 'Value': self.depreciation_years})
        summary_data.append({'Metric': '', 'Value': ''})
        
        # Financial Returns
        summary_data.append({'Metric': 'FINANCIAL RETURNS', 'Value': ''})
        summary_data.append({'Metric': 'Net Present Value (NPV) (€)', 'Value': metrics['npv']})
        
        # Handle IRR - may be None
        if metrics['irr'] is not None:
            summary_data.append({'Metric': 'Internal Rate of Return (IRR) (%)', 'Value': metrics['irr']})
        else:
            summary_data.append({'Metric': 'Internal Rate of Return (IRR) (%)', 'Value': 'N/A'})
        
        # Handle payback periods - may be None
        if metrics['simple_payback'] is not None:
            summary_data.append({'Metric': 'Simple Payback Period (years)', 'Value': metrics['simple_payback']})
        else:
            summary_data.append({'Metric': 'Simple Payback Period (years)', 'Value': 'N/A'})
            
        if metrics['discounted_payback'] is not None:
            summary_data.append({'Metric': 'Discounted Payback Period (years)', 'Value': metrics['discounted_payback']})
        else:
            summary_data.append({'Metric': 'Discounted Payback Period (years)', 'Value': 'N/A'})
            
        summary_data.append({'Metric': '', 'Value': ''})
        
        # Profitability Assessment
        summary_data.append({'Metric': 'PROFITABILITY ASSESSMENT', 'Value': ''})
        summary_data.append({'Metric': 'Project Status', 'Value': metrics['project_status']})
        summary_data.append({'Metric': 'Discount Rate (WACC) (%)', 'Value': self.discount_rate * 100})
        summary_data.append({'Metric': 'Tax Rate (%)', 'Value': self.tax_rate * 100})
        summary_data.append({'Metric': '', 'Value': ''})
        
        # Revenue Summary (Lifetime)
        summary_data.append({'Metric': 'REVENUE SUMMARY (Lifetime)', 'Value': ''})
        summary_data.append({'Metric': 'Total Gross Revenue (€)', 'Value': metrics['total_gross_revenue']})
        summary_data.append({'Metric': 'Total Degradation Cost (€)', 'Value': metrics['total_degradation']})
        summary_data.append({'Metric': 'Total Net Revenue (€)', 'Value': metrics['total_net_revenue']})
        summary_data.append({'Metric': 'Total OPEX (€)', 'Value': metrics['total_opex']})
        summary_data.append({'Metric': 'Total Tax Paid (€)', 'Value': metrics['total_tax']})
        summary_data.append({'Metric': 'Total Cash Flow (€)', 'Value': metrics['total_cash_flow']})
        summary_data.append({'Metric': '', 'Value': ''})
        
        # First Year Operations
        if self.cash_flows:
            first_year = self.cash_flows[0]
            summary_data.append({'Metric': 'FIRST YEAR OPERATIONS', 'Value': ''})
            summary_data.append({'Metric': 'First Year Net Revenue (€)', 'Value': first_year.net_revenue})
            summary_data.append({'Metric': 'First Year OPEX (€)', 'Value': first_year.opex})
            summary_data.append({'Metric': 'First Year Cash Flow (€)', 'Value': first_year.cash_flow})
            summary_data.append({'Metric': 'First Year Cycles', 'Value': first_year.cycles})
        
        df = pd.DataFrame(summary_data)
        df.to_excel(writer, sheet_name='Executive Summary', index=False)
    
    def _create_cash_flow_projection(self, writer):
        """Create cash flow projection sheet"""
        cash_flow_data = []
        
        for cf in self.cash_flows:
            cash_flow_data.append({
                'Year': cf.year,
                'Warranty Year': cf.warranty_year,
                'Market Year': cf.market_year,
                'Gross Revenue (€)': cf.gross_revenue,
                'Degradation Cost (€)': cf.degradation_cost,
                'Net Revenue (€)': cf.net_revenue,
                'OPEX (€)': cf.opex,
                'EBITDA (€)': cf.ebitda,
                'Depreciation (€)': cf.depreciation,
                'EBIT (€)': cf.ebit,
                'Tax (€)': cf.tax,
                'Net Income (€)': cf.net_income,
                'Cash Flow (€)': cf.cash_flow,
                'Cumulative Cash Flow (€)': cf.cumulative_cash_flow,
                'Discounted Cash Flow (€)': cf.discounted_cash_flow,
                'Cumulative Discounted CF (€)': cf.cumulative_discounted_cash_flow,
                'RTE': cf.rte,
                'SOH': cf.soh,
                'Effective Capacity (MWh)': cf.effective_capacity_mwh,
                'Cycles (FEC)': cf.cycles
            })
        
        df = pd.DataFrame(cash_flow_data)
        df.to_excel(writer, sheet_name='Cash Flow Projection', index=False)
    
    def _create_multiyear_summary(self, writer):
        """Create multi-year operational summary"""
        summary_data = []
        
        for warranty_year in sorted(self.yearly_results.keys()):
            results = self.yearly_results[warranty_year]
            market_year = self.start_market_year + warranty_year
            
            # Find corresponding cash flow
            cf = next((c for c in self.cash_flows if c.warranty_year == warranty_year), None)
            
            # Calculate effective capacity for FEC calculation
            year_effective_capacity = self.config.hardware.nominal_capacity_mwh * self.config.degradation.get_soh_at_year(warranty_year)
            
            summary_data.append({
                'Warranty Year': warranty_year,
                'Market Year': market_year,
                'RTE': self.config.degradation.get_rte_at_year(warranty_year),
                'SOH': self.config.degradation.get_soh_at_year(warranty_year),
                'Effective Capacity (MWh)': year_effective_capacity,
                'FCR Revenue (€)': results['fcr_revenue'].sum(),
                'aFRR Revenue (€)': results['afrr_capacity_revenue'].sum() + results['afrr_energy_revenue'].sum(),
                'Arbitrage Revenue (€)': results['energy_arbitrage_revenue'].sum(),
                'Gross Revenue (€)': results['gross_revenue'].sum(),
                'Degradation Cost (€)': results['degradation_total'].sum(),
                'Net Revenue (€)': results['net_profit'].sum(),
                'OPEX (€)': cf.opex if cf else 0,
                'Cash Flow (€)': cf.cash_flow if cf else 0,
                'Cycles (FEC)': results['discharge_mw'].sum() * 0.25 / year_effective_capacity  # FIX: Use degraded capacity
            })
        
        df = pd.DataFrame(summary_data)
        df.to_excel(writer, sheet_name='Multi-Year Operational Summary', index=False)
    
    def _create_year_summary(self, warranty_year: int, market_year: int, results: pd.DataFrame) -> pd.DataFrame:
        """Create summary for a specific year"""
        summary_data = []
        
        # Header
        summary_data.append({'Metric': f'YEAR {warranty_year} SUMMARY (Market Data: {market_year})', 'Value': ''})
        summary_data.append({'Metric': '', 'Value': ''})
        
        # Year Parameters
        rte = self.config.degradation.get_rte_at_year(warranty_year)
        soh = self.config.degradation.get_soh_at_year(warranty_year)
        effective_capacity = self.config.hardware.nominal_capacity_mwh * soh
        
        summary_data.append({'Metric': 'YEAR PARAMETERS', 'Value': ''})
        summary_data.append({'Metric': 'Warranty Year', 'Value': warranty_year})
        summary_data.append({'Metric': 'Market Data Year', 'Value': market_year})
        summary_data.append({'Metric': 'Round-Trip Efficiency (RTE)', 'Value': f"{rte:.4f}"})
        summary_data.append({'Metric': 'State of Health (SOH)', 'Value': f"{soh:.4f}"})
        summary_data.append({'Metric': 'Effective Capacity (MWh)', 'Value': f"{effective_capacity:.2f}"})
        summary_data.append({'Metric': 'Nominal Power (MW)', 'Value': f"{self.config.hardware.nominal_power_mw:.2f}"})
        summary_data.append({'Metric': '', 'Value': ''})
        
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
        summary_data.append({'Metric': 'DEGRADATION COSTS', 'Value': ''})
        summary_data.append({'Metric': 'Arbitrage Degradation (€)', 'Value': results['degradation_arbitrage'].sum()})
        summary_data.append({'Metric': 'FCR Degradation (€)', 'Value': results['degradation_fcr'].sum()})
        summary_data.append({'Metric': 'aFRR Positive Degradation (€)', 'Value': results['degradation_afrr_pos'].sum()})
        summary_data.append({'Metric': 'aFRR Negative Degradation (€)', 'Value': results['degradation_afrr_neg'].sum()})
        summary_data.append({'Metric': 'TOTAL DEGRADATION COST (€)', 'Value': results['degradation_total'].sum()})
        summary_data.append({'Metric': '', 'Value': ''})
        
        # Net Profit
        summary_data.append({'Metric': 'NET PROFIT (after degradation) (€)', 'Value': results['net_profit'].sum()})
        summary_data.append({'Metric': 'Net Profit per MW (€/MW/year)', 'Value': results['net_profit'].sum() / self.config.hardware.nominal_power_mw})
        summary_data.append({'Metric': 'Degradation as % of Gross Revenue', 'Value': f"{results['degradation_total'].sum() / results['gross_revenue'].sum() * 100:.1f}%"})
        
        return pd.DataFrame(summary_data)
    
    def _organize_year_details(self, results: pd.DataFrame) -> pd.DataFrame:
        """Organize detailed 15-min results for a year"""
        detailed_columns = [
            'timestamp', 'day', 'period',
            'da_price', 'ida_bid', 'ida_ask', 'ida_mid',
            'fcr_price', 'afrr_cap_pos', 'afrr_cap_neg',
            'fcr_allocated_mw', 'afrr_pos_allocated_mw', 'afrr_neg_allocated_mw',
            'charge_mw', 'discharge_mw',
            'soc',
            'fcr_revenue', 'afrr_capacity_revenue', 'afrr_energy_revenue',
            'energy_arbitrage_revenue', 'gross_revenue', 'degradation_total', 'net_profit'
        ]
        
        existing_columns = [col for col in detailed_columns if col in results.columns]
        remaining_columns = [col for col in results.columns if col not in detailed_columns]
        
        return results[existing_columns + remaining_columns]


def main():
    """Main execution function"""
    print("\n" + "="*80)
    print("BESS FINANCIAL FEASIBILITY ANALYSIS")
    print("="*80)
    print("\nIntegrated Multi-Year Simulation + Financial Modeling")
    print("Generates comprehensive Excel report with NPV, IRR, and cash flow analysis")
    print("="*80)
    
    # Configuration
    battery_config_file = 'input data/Battery_Manufacturer_Data_to run.xlsx'
    market_data_file = 'input data/Germany_priceforecast_25years.xlsx'
    output_file = 'output/BESS_Financial_Feasibility.xlsx'
    
    # Financial parameters
    discount_rate = 0.08  # 8% WACC
    tax_rate = 0.30  # 30% corporate tax
    depreciation_years = 10
    
    # Create analyzer
    analyzer = BESSFinancialAnalyzer(
        battery_config_file=battery_config_file,
        market_data_file=market_data_file,
        discount_rate=discount_rate,
        tax_rate=tax_rate,
        depreciation_years=depreciation_years
    )
    
    # Run multi-year simulation
    analyzer.run_multiyear_simulation()
    
    # Calculate cash flows
    analyzer.calculate_cash_flows()
    
    # Export comprehensive results
    analyzer.export_to_excel(output_file)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"Output file: {output_file}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
