#!/usr/bin/env python3
# calc_re.py
import argparse
import pandas as pd
import matplotlib.pyplot as plt

# Import the graph creation and Re calculation functions
from label_components import create_graph_classic, calculate_Re

def main():
    parser = argparse.ArgumentParser(description="Calculate overall and time-varying Re from EpiHiper output.")
    parser.add_argument("--epihiper", required=True, help="Path to raw EpiHiper output.csv.gz")
    parser.add_argument("--out", required=True, help="Output path for the Re_over_time.csv file")
    
    # Arguments for filtering
    parser.add_argument("--variant", type=str, default=None, help="Specific variant number to filter for (e.g., '2' for E2, I2)")
    parser.add_argument("--start_tick", type=int, default=None, help="Only consider infections occurring on or after this tick")
    parser.add_argument("--stop_tick", type=int, default=None, help="Only consider infections occurring on or before this tick")
    
    args = parser.parse_args()

    print(f"Loading EpiHiper data from {args.epihiper}...")
    epihiper_df = pd.read_csv(args.epihiper)

    # 1. FILTER BY TIME (if requested)
    if args.start_tick is not None:
        epihiper_df = epihiper_df[epihiper_df['tick'] >= args.start_tick]
    if args.stop_tick is not None:
        epihiper_df = epihiper_df[epihiper_df['tick'] <= args.stop_tick]

    # 2. FILTER BY VARIANT (if requested)
    if args.variant is not None:
        print(f"Filtering for Variant {args.variant}...")
        # Keep only rows where the exit_state contains the variant number (e.g., '2')
        # This assumes your states are strictly named like 'E2_a', 'I2_s', etc.
        epihiper_df = epihiper_df[epihiper_df['exit_state'].str.contains(f"{args.variant}_")]

    print("Creating infection graph...")
    # This function expects the full epihiper output to trace contact_pid -> pid
    infection_graph = create_graph_classic(epihiper_df)

    print("Filtering for initial Exposure events...")
    # We grab the 'E' states to establish the correct time of infection for Re(t)
    infection_df = epihiper_df[epihiper_df['exit_state'].str.startswith('E')].copy()
    
    # Create alias_pid to match the graph nodes
    infection_df['alias_pid'] = infection_df['pid'].astype(str) + '.' + infection_df['tick'].astype(str)

    print("Calculating Re...")
    overall_Re, Re_time_series = calculate_Re(
        infection_graph=infection_graph, 
        infection_df=infection_df, 
        time_col='tick'
    )

    print("\n====================================")
    print(f"OVERALL EPIDEMIC R_e: {overall_Re:.3f}")
    print("====================================\n")

    if Re_time_series is not None:
        # Save to CSV, naming the index 'tick' and the column 'R_e'
        Re_time_series.to_csv(args.out, header=['R_e'], index_label='tick')
        print(f"Time-varying R_e successfully saved to: {args.out}")
        
        # --- NEW PLOTTING LOGIC ---
        try:
            print("Generating companion plot...")
            # Automatically determine plot filename
            plot_file = str(args.out).replace('.csv', '.png')
            if not plot_file.endswith('.png'):
                plot_file += '.png'
                
            plt.figure(figsize=(12, 6))
            
            # Plot the raw daily values (lightly)
            plt.plot(Re_time_series.index, Re_time_series.values, 
                     marker='o', linestyle='-', color='dodgerblue', alpha=0.4, markersize=4, 
                     label='Daily Cohort $R_e$')
            
            # Plot a 7-tick rolling average to show the trend
            if len(Re_time_series) > 7:
                rolling_re = Re_time_series.rolling(window=7, min_periods=1).mean()
                plt.plot(rolling_re.index, rolling_re.values, 
                         color='firebrick', linewidth=2.5, 
                         label='7-Tick Rolling Average')
                
            # Add the Epidemic Threshold line
            plt.axhline(y=1.0, color='black', linestyle='--', linewidth=1.5, label='Epidemic Threshold ($R_e=1$)')
            
            # Formatting
            plt.title('Time-Varying Cohort Reproduction Number ($R_c$) from Simulation', fontsize=14)
            plt.xlabel('Simulation Tick (Time of Exposure)', fontsize=12)
            plt.ylabel('Effective Reproduction Number ($R_e$)', fontsize=12)
            plt.grid(True, linestyle=':', alpha=0.7)
            plt.legend(fontsize=11)
            plt.tight_layout()
            
            # Save the plot
            plt.savefig(plot_file, dpi=300)
            plt.close()
            print(f"Companion plot saved to: {plot_file}")
            
        except Exception as e:
            print(f"Warning: Failed to generate or save plot. Error: {e}")
    else:
        print("Warning: Could not calculate time-varying R_e.")

if __name__ == "__main__":
    main()
