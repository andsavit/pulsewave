import pandas as pd
import numpy as np
from datetime import datetime, timezone
import os
import glob
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

def enforce_data_types(df):
    """
    Enforce proper data types based on the schema from ingestions_test.py
    """
    logger.info("Enforcing data types...")
    
    # Numeric fields
    numeric_int_fields = [
        'id', 'year', 'inventory', 'price', 'buyer_price'
    ]
    
    # Boolean fields
    boolean_fields = [
        'preferred_seller', 'offers_enabled', 'has_inventory', 
        'auction', 'price_taxIncluded', 'buyer_price_taxIncluded',
        'snap_is_current'
    ]
    
    # String fields
    string_fields = [
        'make', 'model', 'finish', 'sku', 'product_type', 'category',
        'title', 'shop_slug', 'condition', 'state', 'permalink',
        'price_currency', 'buyer_price_currency'
    ]
    
    # DateTime fields
    datetime_fields = [
        'created_at', 'published_at', 'snap_valid_from', 'snap_valid_to'
    ]
    
    # Apply numeric conversions with better error handling
    for field in numeric_int_fields:
        if field in df.columns:
            try:
                logger.debug(f"Converting {field} to numeric...")
                # First convert to numeric, then handle NaN values
                df[field] = pd.to_numeric(df[field], errors='coerce')
                # Only convert to Int64 if we have numeric data
                if not df[field].isna().all():
                    df[field] = df[field].astype('Int64')
                else:
                    logger.warning(f"Field {field} contains no valid numeric data")
            except Exception as e:
                logger.warning(f"Could not convert {field} to numeric: {e}")
                # Keep as object type if conversion fails
                continue
    
    # Apply boolean conversions
    for field in boolean_fields:
        if field in df.columns:
            try:
                logger.debug(f"Converting {field} to boolean...")
                # Handle various boolean representations
                df[field] = df[field].astype(str).str.lower()
                df[field] = df[field].map({
                    'true': True, 'false': False, '1': True, '0': False,
                    'yes': True, 'no': False, 'nan': None, 'none': None,
                    '': None, 'null': None
                })
                df[field] = df[field].astype('boolean')
            except Exception as e:
                logger.warning(f"Could not convert {field} to boolean: {e}")
                continue
    
    # Apply string conversions
    for field in string_fields:
        if field in df.columns:
            try:
                logger.debug(f"Converting {field} to string...")
                df[field] = df[field].astype(str).replace('nan', '').replace('None', '')
                df[field] = df[field].astype('string')
            except Exception as e:
                logger.warning(f"Could not convert {field} to string: {e}")
                continue
    
    # Apply datetime conversions
    for field in datetime_fields:
        if field in df.columns:
            try:
                logger.debug(f"Converting {field} to datetime...")
                df[field] = pd.to_datetime(df[field], errors='coerce', utc=True)
            except Exception as e:
                logger.warning(f"Could not convert {field} to datetime: {e}")
                continue
    
    logger.info("Data types enforced successfully")
    return df

def consolidate_reverb_data():
    """
    Consolidate all reverb category parquet files into a single well-formatted file
    """
    # Define paths
    source_dir = "data/reverb/test/used-IT"
    output_dir = "data/reverb/processed"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate timestamp for filename
    timestamp = datetime.now().strftime("%Y%m-%d%H%M")
    output_filename = f"st_complete_{timestamp}.parquet"
    output_path = os.path.join(output_dir, output_filename)
    
    logger.info(f"Starting consolidation of Reverb data")
    logger.info(f"Source directory: {source_dir}")
    logger.info(f"Output file: {output_path}")
    
    # Find all parquet files except combined_dataset.parquet
    pattern = os.path.join(source_dir, "*.parquet")
    all_files = glob.glob(pattern)
    
    # Filter out combined_dataset.parquet and any other files to exclude
    exclude_files = ['combined_dataset.parquet']
    data_files = [f for f in all_files if not any(excl in os.path.basename(f) for excl in exclude_files)]
    
    if not data_files:
        logger.error(f"No parquet files found in {source_dir}")
        return False
    
    logger.info(f"Found {len(data_files)} category files to process:")
    for file in data_files:
        logger.info(f"  - {os.path.basename(file)}")
    
    # Read and combine all files
    all_dataframes = []
    total_rows = 0
    
    for file_path in data_files:
        try:
            logger.info(f"Reading {os.path.basename(file_path)}...")
            df = pd.read_parquet(file_path)
            
            if df.empty:
                logger.warning(f"File {os.path.basename(file_path)} is empty, skipping")
                continue
                
            rows_count = len(df)
            total_rows += rows_count
            logger.info(f"  - Loaded {rows_count:,} rows")
            
            all_dataframes.append(df)
            
        except Exception as e:
            logger.error(f"Error reading {file_path}: {e}")
            continue
    
    if not all_dataframes:
        logger.error("No valid data files could be read")
        return False
    
    # Combine all dataframes
    logger.info("Combining all dataframes...")
    combined_df = pd.concat(all_dataframes, ignore_index=True)
    logger.info(f"Combined dataset shape: {combined_df.shape}")
    
    # Enforce data types
    combined_df = enforce_data_types(combined_df)
    
    # Basic data quality summary
    logger.info("Data quality summary:")
    logger.info(f"  - Total rows: {len(combined_df):,}")
    logger.info(f"  - Total columns: {len(combined_df.columns)}")
    logger.info(f"  - Unique listings (by id): {combined_df['id'].nunique():,}")
    logger.info(f"  - Date range: {combined_df['created_at'].min()} to {combined_df['created_at'].max()}")
    logger.info(f"  - Product types: {combined_df['product_type'].nunique()}")
    logger.info(f"  - Categories: {combined_df['category'].nunique()}")
    
    # Memory usage before save
    memory_usage_mb = combined_df.memory_usage(deep=True).sum() / 1024 / 1024
    logger.info(f"  - Memory usage: {memory_usage_mb:.1f} MB")
    
    # Save to parquet
    logger.info(f"Saving consolidated dataset to {output_path}...")
    try:
        combined_df.to_parquet(
            output_path, 
            index=False,
            compression='snappy'  # Good balance of speed and compression
        )
        
        # Verify file was created and get size
        file_size_mb = os.path.getsize(output_path) / 1024 / 1024
        logger.info(f"Successfully saved! File size: {file_size_mb:.1f} MB")
        
        return True
        
    except Exception as e:
        logger.error(f"Error saving file: {e}")
        return False

def main():
    """
    Main execution function
    """
    logger.info("=" * 60)
    logger.info("REVERB DATA CONSOLIDATION SCRIPT")
    logger.info("=" * 60)
    
    success = consolidate_reverb_data()
    
    if success:
        logger.info("=" * 60)
        logger.info("CONSOLIDATION COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)
    else:
        logger.error("=" * 60)
        logger.error("CONSOLIDATION FAILED")
        logger.error("=" * 60)

if __name__ == "__main__":
    main()