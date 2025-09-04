import pandas as pd
import os
from pathlib import Path

def combine_parquet_files(directory_path, output_file="combined_dataset.parquet"):
    """
    Combine all parquet files in a directory into a single dataset
    """
    parquet_files = []
    directory = Path(directory_path)
    
    # Find all parquet files
    for file_path in directory.glob("*.parquet"):
        print(f"Found: {file_path.name}")
        parquet_files.append(file_path)
    
    if not parquet_files:
        print("No parquet files found in the directory")
        return
    
    print(f"\nCombining {len(parquet_files)} files...")
    
    # Read and combine all files
    dataframes = []
    for file_path in parquet_files:
        df = pd.read_parquet(file_path)
        print(f"{file_path.name}: {len(df)} rows")
        dataframes.append(df)
    
    # Concatenate all dataframes
    combined_df = pd.concat(dataframes, ignore_index=True)
    
    print(f"\nCombined dataset: {len(combined_df)} total rows")
    print(f"Columns: {list(combined_df.columns)}")
    
    # Save combined dataset
    output_path = directory / output_file
    combined_df.to_parquet(output_path, index=False)
    print(f"\nSaved to: {output_path}")
    
    return combined_df

# Usage examples:
if __name__ == "__main__":
    # Replace with your actual path
    directory_path = "data/reverb/test/used-IT"  # MODIFY TO CHANGE EXECUTION
    
    # Combine files
    combined_data = combine_parquet_files(directory_path)
    
    # Optional: Quick data exploration
    if combined_data is not None:
        print(f"\nDataset shape: {combined_data.shape}")
        print(f"Memory usage: {combined_data.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
        
        # Show sample
        print("\nFirst few rows:")
        print(combined_data.head())
        
        # Show product types distribution
        if 'product_type' in combined_data.columns:
            print("\nProduct types distribution:")
            print(combined_data['product_type'].value_counts())