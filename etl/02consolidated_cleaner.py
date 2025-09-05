import pandas as pd
import numpy as np
from datetime import datetime, timezone
import os
import glob
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

def load_latest_consolidated_file():
    """Load the most recent consolidated file"""
    processed_dir = "data/reverb/processed" #INPUT DIRECTORY
    pattern = os.path.join(processed_dir, "st_complete_*.parquet")
    files = glob.glob(pattern)
    
    if not files:
        raise FileNotFoundError(f"No consolidated files found in {processed_dir}")
    
    latest_file = max(files, key=os.path.getctime)
    logger.info(f"Loading: {os.path.basename(latest_file)}")
    
    df = pd.read_parquet(latest_file)
    logger.info(f"Original dataset: {df.shape[0]:,} rows, {df.shape[1]} columns")
    
    return df, latest_file

def standardize_nulls(df):
    """Standardize various null representations to proper nulls"""
    logger.info("Standardizing null values...")
    
    # Fields that use "unknown" as null
    unknown_fields = ['make', 'model']
    
    for field in unknown_fields:
        if field in df.columns:
            # Replace "unknown" with empty string
            df[field] = df[field].replace('unknown', '')
            df[field] = df[field].replace('Unknown', '')
            df[field] = df[field].replace('UNKNOWN', '')
    
    # Global null standardization
    null_values = ['None', 'none', 'NONE', 'null', 'NULL', 'Null', 'nan', 'NaN', 'NAN']
    
    for col in df.columns:
        if df[col].dtype == 'object' or df[col].dtype == 'string':
            for null_val in null_values:
                df[col] = df[col].replace(null_val, '')
    
    # Convert empty strings to pandas NA, then back to empty string for CSV
    df = df.replace('', pd.NA).fillna('')
    
    logger.info("Null standardization completed")
    return df

def filter_bad_listings(df):
    """Remove listings with problematic data"""
    logger.info("Filtering out bad listings...")
    
    initial_count = len(df)
    
    # Remove listings with title "Delete" (case insensitive)
    if 'title' in df.columns:
        before_delete_filter = len(df)
        df = df[~df['title'].str.lower().str.contains('delete', na=False)]
        deleted_listings = before_delete_filter - len(df)
        logger.info(f"Removed {deleted_listings:,} listings with 'Delete' in title")
    
    # Could add more filters here if needed
    # Examples:
    # df = df[df['price'] > 0]  # Remove zero price listings
    # df = df[df['title'].str.len() > 5]  # Remove very short titles
    
    total_removed = initial_count - len(df)
    logger.info(f"Total bad listings filtered: {total_removed:,}")
    
    return df

def fix_condition_field(df):
    """Fix the condition field issue"""
    logger.info("Fixing condition field...")
    
    if 'condition' in df.columns:
        # Since we know they're all "used" from params, set this explicitly
        df['condition'] = 'used'
        logger.info("Set all condition values to 'used' based on scraping parameters")
    
    return df

def remove_useless_fields(df):
    """Remove fields that don't add analytical value"""
    logger.info("Removing useless fields...")
    
    fields_to_remove = [
        'auction',  # Always false
        'sku',      # Usually empty/not useful for analysis
        'permalink' # URLs not needed in Tableau
    ]
    
    removed_fields = []
    for field in fields_to_remove:
        if field in df.columns:
            df = df.drop(columns=[field])
            removed_fields.append(field)
    
    if removed_fields:
        logger.info(f"Removed fields: {', '.join(removed_fields)}")
    
    return df

def collapse_minimal_changes(df, price_tolerance=0.05):
    """Collapse records that only have minimal price changes"""
    logger.info("Collapsing minimal price changes...")
    
    if 'id' not in df.columns:
        logger.warning("Cannot collapse changes without 'id' column")
        return df
    
    initial_rows = len(df)
    cleaned_records = []
    
    # Group by listing ID and process each group
    for listing_id, group in df.groupby('id'):
        if len(group) == 1:
            # Single version, keep as-is
            cleaned_records.append(group.iloc[0])
            continue
        
        # Sort by snap_valid_from to process chronologically
        group = group.sort_values('snap_valid_from').copy()
        
        # Start with first record
        current_record = group.iloc[0].copy()
        
        for i in range(1, len(group)):
            next_record = group.iloc[i]
            
            # Check if only minimal price changes
            is_minimal_change = True
            
            # Check price difference
            curr_price = float(current_record.get('price', 0) or 0)
            next_price = float(next_record.get('price', 0) or 0)
            if abs(curr_price - next_price) > price_tolerance:
                is_minimal_change = False
            
            # Check buyer_price difference
            curr_buyer_price = float(current_record.get('buyer_price', 0) or 0)
            next_buyer_price = float(next_record.get('buyer_price', 0) or 0)
            if abs(curr_buyer_price - next_buyer_price) > price_tolerance:
                is_minimal_change = False
            
            # Check non-price fields for changes
            important_fields = ['state', 'inventory', 'has_inventory', 'offers_enabled']
            for field in important_fields:
                if field in current_record and field in next_record:
                    if str(current_record.get(field, '')) != str(next_record.get(field, '')):
                        is_minimal_change = False
                        break
            
            if is_minimal_change:
                # Merge with current record (extend valid period)
                current_record['snap_valid_to'] = next_record.get('snap_valid_to')
                current_record['snap_is_current'] = next_record.get('snap_is_current')
                # Keep latest prices (they're very close anyway)
                current_record['price'] = next_record.get('price')
                current_record['buyer_price'] = next_record.get('buyer_price')
            else:
                # Meaningful change, save current and start new
                cleaned_records.append(current_record)
                current_record = next_record.copy()
        
        # Add the final record
        cleaned_records.append(current_record)
    
    # Create new dataframe
    df_cleaned = pd.DataFrame(cleaned_records)
    
    collapsed_rows = initial_rows - len(df_cleaned)
    logger.info(f"Collapsed {collapsed_rows:,} rows with minimal changes")
    logger.info(f"Dataset reduced from {initial_rows:,} to {len(df_cleaned):,} rows")
    
    return df_cleaned

def calculate_analytical_fields(df):
    """Add calculated fields useful for Tableau analysis"""
    logger.info("Calculating analytical fields...")
    
    # 1. Permanence time for closed listings
    if 'snap_valid_from' in df.columns and 'snap_valid_to' in df.columns:
        df['snap_valid_from'] = pd.to_datetime(df['snap_valid_from'])
        df['snap_valid_to'] = pd.to_datetime(df['snap_valid_to'])
        
        # Calculate permanence time in hours
        df['permanence_hours'] = ''
        closed_mask = df['snap_valid_to'].notna()
        
        if closed_mask.any():
            time_diff = df.loc[closed_mask, 'snap_valid_to'] - df.loc[closed_mask, 'snap_valid_from']
            df.loc[closed_mask, 'permanence_hours'] = (time_diff.dt.total_seconds() / 3600).round(2)
    
    # 2. Listing status
    df['listing_status'] = ''
    if 'snap_is_current' in df.columns:
        df.loc[df['snap_is_current'] == True, 'listing_status'] = 'Active'
        df.loc[df['snap_is_current'] == False, 'listing_status'] = 'Closed'
    
    # 3. Price range categories
    if 'price' in df.columns:
        df['price_range'] = ''
        price_numeric = pd.to_numeric(df['price'], errors='coerce')
        
        df.loc[price_numeric <= 100, 'price_range'] = '0-100'
        df.loc[(price_numeric > 100) & (price_numeric <= 500), 'price_range'] = '101-500'
        df.loc[(price_numeric > 500) & (price_numeric <= 1000), 'price_range'] = '501-1000'
        df.loc[(price_numeric > 1000) & (price_numeric <= 2000), 'price_range'] = '1001-2000'
        df.loc[price_numeric > 2000, 'price_range'] = '2000+'
    
    # 4. Version count per listing (how many times it changed)
    if 'id' in df.columns:
        version_counts = df.groupby('id').size().reset_index(name='total_versions')
        df = df.merge(version_counts, on='id', how='left')
        
        df['change_frequency'] = ''
        df.loc[df['total_versions'] == 1, 'change_frequency'] = 'Never Changed'
        df.loc[df['total_versions'] == 2, 'change_frequency'] = 'Changed Once'
        df.loc[df['total_versions'] >= 3, 'change_frequency'] = 'Changed Multiple Times'
    
    # 5. Time since creation (for active listings)
    current_time = datetime.now(timezone.utc)
    if 'snap_valid_from' in df.columns:
        df['days_since_creation'] = ''
        active_mask = df['snap_is_current'] == True
        
        if active_mask.any():
            time_diff = current_time - df.loc[active_mask, 'snap_valid_from']
            df.loc[active_mask, 'days_since_creation'] = (time_diff.dt.total_seconds() / (24 * 3600)).round(1)
    
    logger.info("Analytical fields calculated")
    return df

def optimize_for_tableau(df):
    """Optimize data types and formats for Tableau"""
    logger.info("Optimizing for Tableau...")
    
    # Convert datetime columns to string format that Tableau likes
    datetime_cols = ['snap_valid_from', 'snap_valid_to', 'created_at', 'published_at']
    for col in datetime_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col]).dt.strftime('%Y-%m-%d %H:%M:%S')
    
    # Ensure numeric fields are properly formatted
    numeric_cols = ['price', 'buyer_price', 'year', 'inventory']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna('')
    
    # Ensure boolean fields are clear
    boolean_cols = ['preferred_seller', 'offers_enabled', 'has_inventory', 'snap_is_current']
    for col in boolean_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.lower()
            df[col] = df[col].replace({'true': 'Yes', 'false': 'No', 'nan': ''})
    
    logger.info("Tableau optimization completed")
    return df

def generate_cleaning_summary(original_shape, final_shape):
    """Generate a summary of the cleaning process"""
    logger.info("=" * 60)
    logger.info("CLEANING SUMMARY")
    logger.info("=" * 60)
    
    rows_removed = original_shape[0] - final_shape[0]
    reduction_pct = (rows_removed / original_shape[0]) * 100
    
    print(f"\n📊 DATASET TRANSFORMATION:")
    print(f"   Original: {original_shape[0]:,} rows × {original_shape[1]} columns")
    print(f"   Final:    {final_shape[0]:,} rows × {final_shape[1]} columns")
    print(f"   Removed:  {rows_removed:,} rows ({reduction_pct:.1f}% reduction)")
    
    print(f"\n✅ CLEANINGS APPLIED:")
    print(f"   • Standardized null values (unknown → empty)")
    print(f"   • Filtered out 'Delete' listings")
    print(f"   • Fixed condition field (all → 'used')")
    print(f"   • Removed useless fields (auction, sku, permalink)")
    print(f"   • Collapsed minimal price changes (±5¢)")
    print(f"   • Added analytical calculated fields")
    print(f"   • Optimized for Tableau format")

def main():
    """Main execution function"""
    logger.info("=" * 60)
    logger.info("TABLEAU DATA CLEANING PIPELINE")
    logger.info("=" * 60)
    
    try:
        # Load data
        df, source_file = load_latest_consolidated_file()
        original_shape = df.shape
        
        # Apply cleanings
        df = standardize_nulls(df)
        df = filter_bad_listings(df)
        df = fix_condition_field(df)
        df = remove_useless_fields(df)
        df = collapse_minimal_changes(df)
        df = calculate_analytical_fields(df)
        df = optimize_for_tableau(df)
        
        # Generate output filename
        timestamp = datetime.now().strftime("%Y%m-%d%H%M")
        output_filename = f"reverb_tableau_{timestamp}.csv"
        output_path = f"data/reverb/cleaned/{output_filename}"
        
        # Save to CSV
        logger.info(f"Saving cleaned data to {output_path}...")
        df.to_csv(output_path, index=False)
        
        # Generate summary
        generate_cleaning_summary(original_shape, df.shape)
        
        file_size_mb = os.path.getsize(output_path) / 1024 / 1024
        logger.info(f"\n💾 File saved: {output_filename} ({file_size_mb:.1f} MB)")
        
        logger.info("=" * 60)
        logger.info("CLEANING COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Error during cleaning: {e}")
        raise

if __name__ == "__main__":
    main()