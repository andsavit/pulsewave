import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timezone
import os
import glob
import logging
from collections import defaultdict

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

def load_latest_consolidated_file():
    """Load the most recent consolidated file"""
    processed_dir = "data/reverb/processed"
    pattern = os.path.join(processed_dir, "st_complete_*.parquet")
    files = glob.glob(pattern)
    
    if not files:
        raise FileNotFoundError(f"No consolidated files found in {processed_dir}")
    
    latest_file = max(files, key=os.path.getctime)
    logger.info(f"Loading: {os.path.basename(latest_file)}")
    
    df = pd.read_parquet(latest_file)
    logger.info(f"Loaded dataset: {df.shape[0]:,} rows, {df.shape[1]} columns")
    
    return df, latest_file

def analyze_snapshot_timeline(df):
    """Analyze the timeline of when records were created and updated"""
    logger.info("=== SNAPSHOT TIMELINE ANALYSIS ===")
    
    print("\n📅 SNAPSHOT TIMING:")
    print("=" * 50)
    
    # Analyze snap_valid_from timestamps
    if 'snap_valid_from' in df.columns:
        valid_from_counts = df['snap_valid_from'].value_counts().sort_index()
        print(f"\nSNAP_VALID_FROM distribution:")
        for timestamp, count in valid_from_counts.items():
            print(f"  {timestamp}: {count:,} records")
    
    # Analyze snap_valid_to timestamps
    if 'snap_valid_to' in df.columns:
        closed_records = df[df['snap_valid_to'].notna()]
        if not closed_records.empty:
            valid_to_counts = closed_records['snap_valid_to'].value_counts().sort_index()
            print(f"\nSNAP_VALID_TO distribution (closed records):")
            for timestamp, count in valid_to_counts.items():
                print(f"  {timestamp}: {count:,} records closed")
        else:
            print("\nNo closed records found (all snap_valid_to are null)")
    
    # Analyze current vs historical
    current_count = df[df['snap_is_current'] == True].shape[0]
    historical_count = df[df['snap_is_current'] == False].shape[0]
    
    print(f"\n📊 CURRENT VS HISTORICAL:")
    print(f"  Current records: {current_count:,} ({current_count/len(df)*100:.1f}%)")
    print(f"  Historical records: {historical_count:,} ({historical_count/len(df)*100:.1f}%)")
    
    return valid_from_counts if 'snap_valid_from' in df.columns else None

def analyze_listing_lifecycles(df):
    """Analyze how long listings stay active and what changes"""
    logger.info("=== LISTING LIFECYCLE ANALYSIS ===")
    
    print("\n🔄 LISTING CHANGE PATTERNS:")
    print("=" * 50)
    
    # Group by listing ID to see change history
    if 'id' in df.columns:
        listing_versions = df.groupby('id').agg({
            'snap_valid_from': ['count', 'min', 'max'],
            'snap_is_current': 'sum'
        }).round(2)
        
        listing_versions.columns = ['total_versions', 'first_seen', 'last_updated', 'current_versions']
        
        # Analyze version counts
        version_dist = listing_versions['total_versions'].value_counts().sort_index()
        print(f"\nLISTING VERSION DISTRIBUTION:")
        for versions, count in version_dist.items():
            print(f"  {versions} version(s): {count:,} listings")
        
        # Find listings with multiple versions
        multi_version = listing_versions[listing_versions['total_versions'] > 1]
        if not multi_version.empty:
            print(f"\n🔍 LISTINGS WITH CHANGES:")
            print(f"  Total listings that changed: {len(multi_version):,}")
            print(f"  Average versions per changed listing: {multi_version['total_versions'].mean():.1f}")
            print(f"  Max versions for one listing: {multi_version['total_versions'].max()}")
            
            # Show some examples
            print(f"\n📋 SAMPLE CHANGED LISTINGS:")
            sample_changed = multi_version.head(10)
            for listing_id, row in sample_changed.iterrows():
                print(f"  ID {listing_id}: {int(row['total_versions'])} versions, first seen {row['first_seen']}")
    
    return listing_versions if 'id' in df.columns else None

def analyze_field_changes(df):
    """Analyze which specific fields are changing most often"""
    logger.info("=== FIELD CHANGE ANALYSIS ===")
    
    print("\n🔄 WHAT'S CHANGING?")
    print("=" * 50)
    
    if 'id' not in df.columns:
        print("Cannot analyze field changes without 'id' column")
        return None
    
    # Get listings that have multiple versions
    multi_version_ids = df.groupby('id').size()
    multi_version_ids = multi_version_ids[multi_version_ids > 1].index
    
    if len(multi_version_ids) == 0:
        print("No listings found with multiple versions")
        return None
    
    print(f"Analyzing {len(multi_version_ids):,} listings with changes...")
    
    # Sample some listings to analyze changes
    sample_ids = multi_version_ids[:100]  # Analyze first 100 changed listings
    change_counts = defaultdict(int)
    field_changes = defaultdict(list)
    
    # Define fields to track (excluding SCD fields)
    tracking_fields = ['price', 'buyer_price', 'state', 'condition', 'inventory', 
                      'offers_enabled', 'has_inventory', 'title']
    
    for listing_id in sample_ids:
        listing_history = df[df['id'] == listing_id].sort_values('snap_valid_from')
        
        if len(listing_history) < 2:
            continue
            
        # Compare consecutive versions
        for i in range(1, len(listing_history)):
            current = listing_history.iloc[i]
            previous = listing_history.iloc[i-1]
            
            for field in tracking_fields:
                if field in df.columns:
                    curr_val = current[field]
                    prev_val = previous[field]
                    
                    # Handle different data types and NaN
                    if pd.isna(curr_val) and pd.isna(prev_val):
                        continue
                    elif pd.isna(curr_val) or pd.isna(prev_val):
                        change_counts[field] += 1
                        field_changes[field].append(f"{prev_val} → {curr_val}")
                    elif str(curr_val) != str(prev_val):
                        change_counts[field] += 1
                        field_changes[field].append(f"{prev_val} → {curr_val}")
    
    # Display results
    if change_counts:
        print(f"\nFIELD CHANGE FREQUENCY (in sample of {len(sample_ids)} changed listings):")
        sorted_changes = sorted(change_counts.items(), key=lambda x: x[1], reverse=True)
        
        for field, count in sorted_changes:
            pct = (count / len(sample_ids)) * 100
            print(f"  {field}: {count} changes ({pct:.1f}% of sampled listings)")
            
            # Show some example changes
            if field_changes[field]:
                examples = field_changes[field][:3]  # Show first 3 examples
                for example in examples:
                    print(f"    Example: {example}")
                if len(field_changes[field]) > 3:
                    print(f"    ... and {len(field_changes[field]) - 3} more")
                print()
    else:
        print("No field changes detected in sample")
    
    return change_counts, field_changes

def analyze_timing_patterns(df):
    """Analyze when changes happen (time patterns)"""
    logger.info("=== TIMING PATTERN ANALYSIS ===")
    
    print("\n⏰ WHEN DO CHANGES HAPPEN?")
    print("=" * 50)
    
    if 'snap_valid_from' not in df.columns:
        print("Cannot analyze timing without snap_valid_from column")
        return
    
    # Convert to datetime if not already
    df['snap_valid_from'] = pd.to_datetime(df['snap_valid_from'])
    
    # Analyze by hour of day
    df['hour'] = df['snap_valid_from'].dt.hour
    hourly_dist = df['hour'].value_counts().sort_index()
    
    print(f"\nUPDATE FREQUENCY BY HOUR (UTC):")
    for hour, count in hourly_dist.items():
        print(f"  {hour:02d}:00: {count:,} updates")
    
    # Analyze by day
    df['date'] = df['snap_valid_from'].dt.date
    daily_dist = df['date'].value_counts().sort_index()
    
    print(f"\nUPDATE FREQUENCY BY DATE:")
    for date, count in daily_dist.items():
        print(f"  {date}: {count:,} updates")
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Hourly pattern
    hourly_dist.plot(kind='bar', ax=ax1, color='lightblue')
    ax1.set_title('Updates by Hour of Day (UTC)')
    ax1.set_xlabel('Hour')
    ax1.set_ylabel('Number of Updates')
    ax1.tick_params(axis='x', rotation=0)
    
    # Daily pattern
    daily_dist.plot(kind='bar', ax=ax2, color='lightcoral')
    ax2.set_title('Updates by Date')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Number of Updates')
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()

def detect_suspicious_patterns(df):
    """Detect potentially suspicious update patterns"""
    logger.info("=== SUSPICIOUS PATTERN DETECTION ===")
    
    print("\n🚨 POTENTIAL ISSUES:")
    print("=" * 50)
    
    issues_found = []
    
    # Check 1: Mass updates at exact same timestamp
    if 'snap_valid_from' in df.columns:
        timestamp_counts = df['snap_valid_from'].value_counts()
        large_batches = timestamp_counts[timestamp_counts > 1000]
        
        if not large_batches.empty:
            issues_found.append("Large batch updates detected")
            print(f"⚠️  LARGE BATCH UPDATES:")
            for timestamp, count in large_batches.items():
                print(f"   {timestamp}: {count:,} records updated simultaneously")
    
    # Check 2: Too many versions per listing
    if 'id' in df.columns:
        version_counts = df.groupby('id').size()
        excessive_versions = version_counts[version_counts > 5]
        
        if not excessive_versions.empty:
            issues_found.append("Listings with excessive version history")
            print(f"⚠️  LISTINGS WITH MANY VERSIONS:")
            print(f"   {len(excessive_versions)} listings have >5 versions")
            print(f"   Max versions: {excessive_versions.max()}")
            print(f"   Example listing IDs: {list(excessive_versions.head().index)}")
    
    # Check 3: All records closed at same time
    if 'snap_valid_to' in df.columns:
        closed_records = df[df['snap_valid_to'].notna()]
        if not closed_records.empty:
            close_times = closed_records['snap_valid_to'].value_counts()
            if len(close_times) == 1:
                issues_found.append("All records closed at exactly same time")
                print(f"⚠️  MASS CLOSURE:")
                print(f"   {closed_records.shape[0]:,} records all closed at: {close_times.index[0]}")
    
    # Check 4: Unrealistic change rate
    total_listings = df['id'].nunique() if 'id' in df.columns else 0
    changed_listings = len(df.groupby('id').size()[df.groupby('id').size() > 1]) if 'id' in df.columns else 0
    
    if total_listings > 0:
        change_rate = (changed_listings / total_listings) * 100
        if change_rate > 30:  # More than 30% changed
            issues_found.append("High change rate detected")
            print(f"⚠️  HIGH CHANGE RATE:")
            print(f"   {change_rate:.1f}% of listings changed in short timeframe")
    
    if not issues_found:
        print("✅ No obvious suspicious patterns detected")
    else:
        print(f"\n📋 SUMMARY: {len(issues_found)} potential issues found")
        for i, issue in enumerate(issues_found, 1):
            print(f"   {i}. {issue}")

def main():
    """Main execution function"""
    logger.info("=" * 60)
    logger.info("REVERB CHANGE ANALYSIS")
    logger.info("=" * 60)
    
    try:
        # Load data
        df, filepath = load_latest_consolidated_file()
        
        # Run analyses
        analyze_snapshot_timeline(df)
        analyze_listing_lifecycles(df)
        analyze_field_changes(df)
        analyze_timing_patterns(df)
        detect_suspicious_patterns(df)
        
        logger.info("=" * 60)
        logger.info("CHANGE ANALYSIS COMPLETED")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Error during analysis: {e}")
        raise

if __name__ == "__main__":
    main()