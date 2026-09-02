#!/usr/bin/env python3
import csv
import argparse
import lzma
import gzip
import sys

def open_file(filename, mode='rt'):
    """Transparently open regular, gzip, or xz files."""
    if filename.endswith('.xz'):
        return lzma.open(filename, mode)
    elif filename.endswith('.gz'):
        return gzip.open(filename, mode)
    else:
        return open(filename, mode)

def main():
    parser = argparse.ArgumentParser(description="Generate pairwise strain TSV from linelist metadata.")
    parser.add_argument("-i", "--input", required=True, help="Input CSV file (can be .csv, .csv.gz, or .csv.xz)")
    parser.add_argument("-o", "--output", required=True, help="Output TSV file")
    args = parser.parse_args()

    alias_to_strain = {}
    unique_edges = set()

    print(f"Reading {args.input} to build lookups...")
    
    # Pass 1: Build the alias_pid -> strain lookup and collect unique transmission edges
    try:
        with open_file(args.input) as f:
            reader = csv.DictReader(f)
            
            for row_count, row in enumerate(reader, 1):
                alias_pid = row['alias_pid']
                alias_contact = row['alias_contact']
                strain = row['strain']
                
                # Build the 1:1 lookup dictionary
                alias_to_strain[alias_pid] = strain
                
                # Only record an edge if there is an actual infector 
                # (ignores seed infections where alias_contact is -1)
                if alias_contact != '-1':
                    unique_edges.add((alias_pid, alias_contact))
                    
                if row_count % 500000 == 0:
                    print(f"  Processed {row_count} rows...")
                    
    except KeyError as e:
        print(f"Error: Missing expected column in input file: {e}")
        sys.exit(1)

    print(f"Found {len(alias_to_strain)} unique infections and {len(unique_edges)} transmission pairs.")
    print(f"Writing pairwise IDs to {args.output}...")

    # Pass 2: Resolve the pairs and write to TSV
    written_count = 0
    missing_contact_count = 0
    
    with open(args.output, 'w', newline='') as out_f:
        writer = csv.writer(out_f, delimiter='\t')
        writer.writerow(['ID1', 'ID2']) # Header
        
        for alias_pid, alias_contact in unique_edges:
            id1_strain = alias_to_strain.get(alias_pid)
            id2_strain = alias_to_strain.get(alias_contact)
            
            # Ensure both strains were successfully looked up
            if id1_strain and id2_strain:
                writer.writerow([id1_strain, id2_strain])
                written_count += 1
            else:
                missing_contact_count += 1

    print("Done!")
    print(f"Successfully wrote {written_count} pairs.")
    if missing_contact_count > 0:
        print(f"Note: {missing_contact_count} pairs were skipped because the contact's alias_pid was not found in the file.")

if __name__ == "__main__":
    main()
