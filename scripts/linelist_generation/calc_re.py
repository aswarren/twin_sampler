#!/usr/bin/env python3
# calc_re.py
import argparse
import pandas as pd

# Import the graph creation and Re calculation functions
from label_components import create_graph_classic, calculate_Re

def main():
    parser = argparse.ArgumentParser(description="Calculate overall and time-varying Re from EpiHiper output.")
    parser.add_argument("--epihiper", required=True, help="Path to raw EpiHiper output.csv.gz")
    parser.add_argument("--out", required=True, help="Output path for the Re_over_time.csv file")
    
    # NEW ARGUMENTS FOR FILTERING
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
    else:
        print("Warning: Could not calculate time-varying R_e.")

if __name__ == "__main__":
    main()

