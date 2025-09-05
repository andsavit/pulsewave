import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import glob
import logging
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Setup plotting style
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

def load_latest_consolidated_file():
    """
    Load the most recent consolidated file from data/reverb/processed/
    """
    processed_dir = "data/reverb/processed"
    pattern = os.path.join(processed_dir, "st_complete_*.parquet")
    files = glob.glob(pattern)
    
    if not files:
        raise FileNotFoundError(f"No consolidated files found in {processed_dir}")
    
    # Get the most recent file
    latest_file = max(files, key=os.path.getctime)
    logger.info(f"Loading: {os.path.basename(latest_file)}")
    
    df = pd.read_parquet(latest_file)
    logger.info(f"Loaded dataset: {df.shape[0]:,} rows, {df.shape[1]} columns")
    
    return df, latest_file

def analyze_column_types(df):
    """
    Analyze and display column data types
    """
    logger.info("=== COLUMN TYPES ANALYSIS ===")
    
    type_info = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        null_count = df[col].isnull().sum()
        null_pct = (null_count / len(df)) * 100
        unique_count = df[col].nunique()
        
        type_info.append({
            'Column': col,
            'Data_Type': dtype,
            'Null_Count': null_count,
            'Null_Pct': f"{null_pct:.1f}%",
            'Unique_Values': unique_count,
            'Sample_Value': str(df[col].dropna().iloc[0]) if not df[col].dropna().empty else 'N/A'
        })
    
    type_df = pd.DataFrame(type_info)
    
    print("\nCOLUMN TYPES SUMMARY:")
    print("=" * 80)
    print(type_df.to_string(index=False))
    
    # Group by data type
    print(f"\nDATA TYPE DISTRIBUTION:")
    print("-" * 40)
    type_counts = type_df['Data_Type'].value_counts()
    for dtype, count in type_counts.items():
        print(f"{dtype}: {count} columns")
    
    return type_df

def analyze_categorical_values(df, max_unique=50):
    """
    Count distinct values for categorical columns
    """
    logger.info("=== CATEGORICAL VALUES ANALYSIS ===")
    
    # Identify categorical columns (string type or low cardinality)
    categorical_cols = []
    for col in df.columns:
        if (df[col].dtype == 'string' or df[col].dtype == 'object' or 
            (df[col].dtype in ['category'] or df[col].nunique() <= max_unique)):
            categorical_cols.append(col)
    
    print(f"\nCATEGORICAL COLUMNS ({len(categorical_cols)} found):")
    print("=" * 60)
    
    for col in categorical_cols:
        unique_count = df[col].nunique()
        print(f"\n{col.upper()}: {unique_count} unique values")
        print("-" * 40)
        
        if unique_count <= 20:
            # Show all values for small sets
            value_counts = df[col].value_counts(dropna=False)
            for value, count in value_counts.items():
                pct = (count / len(df)) * 100
                print(f"  {value}: {count:,} ({pct:.1f}%)")
        else:
            # Show top 10 for larger sets
            value_counts = df[col].value_counts(dropna=False).head(10)
            print("  Top 10 values:")
            for value, count in value_counts.items():
                pct = (count / len(df)) * 100
                print(f"    {value}: {count:,} ({pct:.1f}%)")
            
            if unique_count > 10:
                print(f"    ... and {unique_count - 10} more values")
    
    return categorical_cols

def create_numeric_boxplots(df):
    """
    Create boxplots for numeric columns
    """
    logger.info("=== NUMERIC VALUES ANALYSIS ===")
    
    # Identify numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Remove ID-like columns from visualization
    id_like_cols = [col for col in numeric_cols if 'id' in col.lower()]
    numeric_cols = [col for col in numeric_cols if col not in id_like_cols]
    
    if not numeric_cols:
        print("No numeric columns found for visualization")
        return
    
    print(f"\nNUMERIC COLUMNS SUMMARY:")
    print("=" * 50)
    
    # Basic statistics
    numeric_stats = df[numeric_cols].describe()
    print(numeric_stats)
    
    # Create boxplots
    n_cols = len(numeric_cols)
    if n_cols == 0:
        return
    
    # Calculate subplot layout
    n_rows = (n_cols + 2) // 3  # 3 plots per row
    n_subplot_cols = min(3, n_cols)
    
    fig, axes = plt.subplots(n_rows, n_subplot_cols, figsize=(15, 5 * n_rows))
    fig.suptitle('Numeric Columns Distribution (Boxplots)', fontsize=16, y=1.02)
    
    # Handle single subplot case
    if n_cols == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes if hasattr(axes, '__len__') else [axes]
    else:
        axes = axes.flatten()
    
    for i, col in enumerate(numeric_cols):
        if i < len(axes):
            # Remove outliers for better visualization
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Filter data for visualization
            filtered_data = df[col][(df[col] >= lower_bound) & (df[col] <= upper_bound)]
            
            sns.boxplot(y=filtered_data, ax=axes[i])
            axes[i].set_title(f'{col}\n(outliers removed for clarity)')
            axes[i].set_ylabel('Value')
    
    # Hide empty subplots
    for i in range(n_cols, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.show()
    
    return numeric_cols

def analyze_date_distributions(df):
    """
    Create visualizations for date columns
    """
    logger.info("=== DATE DISTRIBUTION ANALYSIS ===")
    
    # Identify datetime columns
    date_cols = df.select_dtypes(include=['datetime64[ns, UTC]', 'datetime64']).columns.tolist()
    
    if not date_cols:
        print("No datetime columns found")
        return
    
    print(f"\nDATE COLUMNS ({len(date_cols)} found):")
    print("=" * 40)
    
    for col in date_cols:
        print(f"\n{col.upper()}:")
        valid_dates = df[col].dropna()
        if len(valid_dates) > 0:
            print(f"  Range: {valid_dates.min()} to {valid_dates.max()}")
            print(f"  Valid entries: {len(valid_dates):,} ({(len(valid_dates)/len(df)*100):.1f}%)")
        else:
            print(f"  No valid dates found")
    
    # Create date distribution plots
    n_cols = len(date_cols)
    fig, axes = plt.subplots(n_cols, 1, figsize=(16, 5 * n_cols))
    fig.suptitle('Date Columns Distribution', fontsize=16)
    
    if n_cols == 1:
        axes = [axes]
    
    for i, col in enumerate(date_cols):
        valid_dates = df[col].dropna()
        
        if len(valid_dates) > 0:
            # Create histogram by month
            dates_df = pd.DataFrame({'date': valid_dates})
            dates_df['year_month'] = dates_df['date'].dt.to_period('M')
            monthly_counts = dates_df['year_month'].value_counts().sort_index()
            
            # Plot
            monthly_counts.plot(kind='bar', ax=axes[i], color='skyblue')
            axes[i].set_title(f'{col} - Monthly Distribution', fontsize=14)
            axes[i].set_xlabel('Year-Month', fontsize=12)
            axes[i].set_ylabel('Count', fontsize=12)
            
            # Improve x-axis readability
            n_ticks = len(monthly_counts)
            if n_ticks > 12:
                # Show every 3rd tick for many data points
                step = max(1, n_ticks // 8)  # Show roughly 8 labels max
                tick_positions = range(0, n_ticks, step)
                tick_labels = [str(monthly_counts.index[pos]) for pos in tick_positions]
                axes[i].set_xticks(tick_positions)
                axes[i].set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=10)
            else:
                # Show all ticks for smaller datasets
                axes[i].tick_params(axis='x', rotation=45, labelsize=10)
            
            # Add value labels on bars for better readability
            for j, (period, count) in enumerate(monthly_counts.items()):
                if j % max(1, len(monthly_counts) // 10) == 0:  # Label every 10th bar or so
                    axes[i].text(j, count + max(monthly_counts) * 0.01, 
                               f'{count:,}', ha='center', va='bottom', fontsize=8)
        else:
            axes[i].text(0.5, 0.5, f'No valid dates in {col}', 
                        ha='center', va='center', transform=axes[i].transAxes, fontsize=12)
            axes[i].set_title(f'{col} - No Data', fontsize=14)
    
    plt.tight_layout()
    plt.show()
    
    return date_cols

def generate_summary_report(df, type_df, categorical_cols, numeric_cols, date_cols):
    """
    Generate a comprehensive summary report
    """
    logger.info("=== GENERATING SUMMARY REPORT ===")
    
    print("\n" + "=" * 80)
    print("DATASET SUMMARY REPORT")
    print("=" * 80)
    
    print(f"\n📊 BASIC STATISTICS:")
    print(f"   • Total rows: {df.shape[0]:,}")
    print(f"   • Total columns: {df.shape[1]}")
    print(f"   • Memory usage: {df.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB")
    
    print(f"\n📋 COLUMN TYPES:")
    print(f"   • Categorical columns: {len(categorical_cols)}")
    print(f"   • Numeric columns: {len(numeric_cols)}")
    print(f"   • Date columns: {len(date_cols)}")
    print(f"   • Other columns: {df.shape[1] - len(categorical_cols) - len(numeric_cols) - len(date_cols)}")
    
    print(f"\n🔍 DATA QUALITY:")
    total_nulls = type_df['Null_Count'].sum()
    total_cells = df.shape[0] * df.shape[1]
    null_pct = (total_nulls / total_cells) * 100
    print(f"   • Total null values: {total_nulls:,} ({null_pct:.1f}% of all cells)")
    
    # Columns with high null rates
    high_null_cols = type_df[type_df['Null_Count'] > df.shape[0] * 0.5]
    if not high_null_cols.empty:
        print(f"   • Columns with >50% nulls: {len(high_null_cols)}")
        for _, row in high_null_cols.iterrows():
            print(f"     - {row['Column']}: {row['Null_Pct']}")
    
    print(f"\n📈 KEY INSIGHTS:")
    
    # Product type distribution
    if 'product_type' in df.columns:
        top_product = df['product_type'].mode().iloc[0] if not df['product_type'].mode().empty else 'N/A'
        print(f"   • Most common product type: {top_product}")
    
    # Price insights
    if 'price' in df.columns:
        price_col = df['price']
        if price_col.dtype in [np.number] and price_col.notna().sum() > 0:
            print(f"   • Price range: {price_col.min():.0f} - {price_col.max():.0f}")
            print(f"   • Median price: {price_col.median():.0f}")
    
    # Date range
    if date_cols:
        for col in date_cols:
            valid_dates = df[col].dropna()
            if len(valid_dates) > 0:
                days_span = (valid_dates.max() - valid_dates.min()).days
                print(f"   • {col} spans {days_span} days")
                break

def main():
    """
    Main execution function
    """
    logger.info("=" * 60)
    logger.info("REVERB DATA PROFILING ANALYSIS")
    logger.info("=" * 60)
    
    try:
        # Load data
        df, filepath = load_latest_consolidated_file()
        
        # Run analyses
        type_df = analyze_column_types(df)
        categorical_cols = analyze_categorical_values(df)
        numeric_cols = create_numeric_boxplots(df)
        date_cols = analyze_date_distributions(df)
        
        # Generate summary
        generate_summary_report(df, type_df, categorical_cols, numeric_cols, date_cols)
        
        logger.info("=" * 60)
        logger.info("PROFILING COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Error during profiling: {e}")
        raise

if __name__ == "__main__":
    main()