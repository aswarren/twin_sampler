import pandas as pd
import numpy as np
import argparse
import lzma
from Bio import AlignIO
from Bio import SeqIO # Use SeqIO for FASTA parsing/writing of individual records



#line_keys=["virus","region","country","division","divisionExposure","date","strain"]

def sample_metadata_uniform(metadata_df, num_samples, time_range_days=7):
    """
    Samples metadata uniformly at random within a specified time range.
    
    Parameters:
    metadata_df (pd.DataFrame): The metadata DataFrame.
    num_samples (int): The number of samples to draw.
    time_range_days (int): The time range in days for sampling. Default is 7 days.
    
    Returns:
    pd.DataFrame: The sampled metadata DataFrame.
    """
    # Convert date column to datetime
    metadata_df['date'] = pd.to_datetime(metadata_df['date'])
    
    # Get the minimum and maximum dates
    min_date = metadata_df['date'].min()
    max_date = metadata_df['date'].max()
    
    # Generate random start dates within the range
    start_dates = pd.to_datetime(np.random.choice(pd.date_range(min_date, max_date - pd.Timedelta(days=time_range_days)), num_samples))
    
    sampled_metadata = pd.DataFrame()
    
    for start_date in start_dates:
        end_date = start_date + pd.Timedelta(days=time_range_days)
        sample = metadata_df[(metadata_df['date'] >= start_date) & (metadata_df['date'] <= end_date)]
        sampled_metadata = pd.concat([sampled_metadata, sample.sample(n=1)])
    
    return sampled_metadata

def sample_metadata_true_uniform_records(metadata_df, num_samples_target):
    """
    Samples num_samples_target records uniformly at random from the entire metadata_df.
    Ensures sampling without replacement if num_samples_target <= len(metadata_df).
    """
    if num_samples_target <= 0:
        return pd.DataFrame()
    if num_samples_target > len(metadata_df):
        print(f"Warning: Requested {num_samples_target} samples, but only {len(metadata_df)} records available. Returning all records.")
        return metadata_df.copy() # Or .sample(n=len(metadata_df), replace=False)
    
    return metadata_df.sample(n=num_samples_target, replace=False)

def sample_metadata(metadata_df, strategy, num_samples, time_range_days=7, time_bin_days_stratified=7):
    """
    Samples metadata using the specified strategy.
    
    Parameters:
    metadata_df (pd.DataFrame): The metadata DataFrame.
    strategy (str): The sampling strategy to use.
    num_samples (int): The number of samples to draw.
    time_range_days (int): The time range in days for sampling. Default is 7 days.
    
    Returns:
    pd.DataFrame: The sampled metadata DataFrame.
    """
    if strategy == 'uniform_window': # Original interpretation
        return sample_metadata_uniform_v2(metadata_df, num_samples, time_range_days)
    elif strategy == 'uniform_record': # Simple random sample of records
        return sample_metadata_true_uniform_records(metadata_df, num_samples)
    elif strategy == 'stratified_temporal': # Samples spread over time bins
        return sample_metadata_stratified_temporal(metadata_df, num_samples, time_bin_days_stratified)
    else:
        raise ValueError(f"Unknown sampling strategy: {strategy}")
    

def filter_fasta_by_metadata(metadata_df, fasta_path, output_path):
    """
    Filters a FASTA file to include only sequences with IDs present in the metadata DataFrame.
    
    Parameters:
    metadata_df (pd.DataFrame): The metadata DataFrame containing the IDs to filter by.
    fasta_path (str): The path to the input FASTA file (can be .xz compressed).
    output_path (str): The path to the output filtered FASTA file (will be plain text).
    """
    ids_to_keep = set(metadata_df['strain'])
    sequences_to_write = []

    # Handle compressed input FASTA
    if fasta_path.endswith(".xz"):
        with lzma.open(fasta_path, "rt") as fasta_handle: # "rt" for read text
            for record in SeqIO.parse(fasta_handle, "fasta"):
                if record.id in ids_to_keep:
                    sequences_to_write.append(record)
    elif fasta_path.endswith(".gz"): # Add .gz support just in case
        with gzip.open(fasta_path, "rt") as fasta_handle:
            for record in SeqIO.parse(fasta_handle, "fasta"):
                if record.id in ids_to_keep:
                    sequences_to_write.append(record)
    else: # Plain FASTA
        with open(fasta_path, "rt") as fasta_handle:
            for record in SeqIO.parse(fasta_handle, "fasta"):
                if record.id in ids_to_keep:
                    sequences_to_write.append(record)
    
    # Write the collected sequences to the output file
    # Output will be plain FASTA. If you want it compressed, open output_handle accordingly.
    with open(output_path, 'w') as output_handle:
        if sequences_to_write: # Check if there's anything to write
            SeqIO.write(sequences_to_write, output_handle, "fasta")
        else:
            print(f"Warning: No sequences found in {fasta_path} matching IDs in the sampled metadata.")

    print(f"Filtered FASTA saved to {output_path}, containing {len(sequences_to_write)} sequences.")

# Example usage:
# metadata_df = read_metadata('path_to_metadata.csv')
# sampled_metadata = sample_metadata(metadata_df, 'uniform', 10, 7)
def read_metadata(file_path):
    """
    Reads metadata from a TSV file into a pandas DataFrame.
    
    Parameters:
    file_path (str): The path to the metadata TSV file.
    
    Returns:
    pd.DataFrame: The metadata DataFrame.
    """
    return pd.read_csv(file_path, sep='\t')

def main():
    parser = argparse.ArgumentParser(description='Sample infections from a (fasta, tsv) pair.')
    parser.add_argument('--metadata_path', required=True, type=str, help='The path to the metadata TSV file.')
    parser.add_argument('--num_samples', type=int, default=500, help='The number of samples to draw per time window.')
    parser.add_argument('--time_range_days', type=int, default=7, help='The time range in days for sampling.')
    parser.add_argument('--strategy', type=str, default='uniform_record', help='The sampling strategy to use.')
    parser.add_argument('--fasta_path', required=True, type=str, help='The path to the input FASTA file.')
    parser.add_argument('--output_path', required= True, type=str, help='The path to the output directory.')
    
    args = parser.parse_args()
    # if no parameters are passed, print help and exit
    if len(vars(args)) == 0:
        parser.print_help()
        parser.exit()
    
    metadata_df = read_metadata(args.metadata_path)
    #sampled_metadata = sample_metadata(metadata_df, args.strategy, args.num_samples, args.time_range_days)
    sampled_metadata = sample_metadata(metadata_df, args.strategy, args.num_samples, 
                                    args.time_range_days, 
                                    args.time_bin_days if hasattr(args, 'time_bin_days') else 7)
   

    
    metadata_df = read_metadata(args.metadata_path)
    sampled_metadata = sample_metadata(metadata_df, args.strategy, args.num_samples, args.time_range_days)
    
    # Generate output file names
    metadata_base_name = args.metadata_path.split('/')[-1].replace('.tsv', '')
    fasta_base_name = args.fasta_path.split('/')[-1].replace('.fasta', '')
    fasta_out_name = f"{fasta_base_name}_samples_{args.num_samples}_days_{args.time_range_days}"
    meta_out_name = f"{metadata_base_name}_samples_{args.num_samples}_days_{args.time_range_days}"

    
    output_metadata_path = f"{args.output_path}/{meta_out_name}.tsv"
    output_fasta_path = f"{args.output_path}/{fasta_out_name}.fasta"
    
    # Write sampled metadata to file
    sampled_metadata.to_csv(output_metadata_path, sep='\t', index=False)
    
    # Filter FASTA file by sampled metadata
    filter_fasta_by_metadata(sampled_metadata, args.fasta_path, output_fasta_path)

if __name__ == '__main__':
    main()
