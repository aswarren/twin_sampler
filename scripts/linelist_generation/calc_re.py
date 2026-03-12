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
    args = parser.parse_args()

    print(f"Loading EpiHiper data from {args.epihiper}...")
    epihiper_df = pd.read_csv(args.epihiper)

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
