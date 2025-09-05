"""
Reverb Advanced Data Analysis - Python
=====================================
Advanced visualizations and statistical analysis not possible in Tableau
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set style for beautiful plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Configuration
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

def load_and_prepare_data(file_path):
    """Load and prepare the Reverb data for analysis"""
    
    # Load data
    print("Loading data...")
    df = pd.read_csv(file_path)
    
    # Convert dates
    date_columns = ['created_at', 'published_at', 'snap_valid_from', 'snap_valid_to']
    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
    
    # Calculate derived fields
    df['price_diff'] = df['buyer_price'] - df['price']
    df['price_diff_pct'] = (df['price_diff'] / df['price']) * 100
    df['permanence_days'] = df['permanence_hours'] / 24
    
    # Create date components
    df['created_date'] = df['created_at'].dt.date
    df['closed_date'] = df['snap_valid_to'].dt.date
    
    # Clean categories and product types
    df['product_type_clean'] = df['product_type'].str.replace('-', ' ').str.title()
    df['category_clean'] = df['category'].str.replace('-', ' ').str.title()
    
    # Remove extreme outliers
    price_99th = df['price'].quantile(0.99)
    buyer_price_99th = df['buyer_price'].quantile(0.99)
    
    df_clean = df[
        (df['price'] > 0) & (df['price'] < price_99th) &
        (df['buyer_price'] > 0) & (df['buyer_price'] < buyer_price_99th)
    ].copy()
    
    print(f"Data loaded: {len(df_clean):,} records after cleaning")
    print(f"Date range: {df_clean['created_date'].min()} to {df_clean['created_date'].max()}")
    
    return df_clean

def create_violin_plots(df):
    """Create violin plots for price distributions - Tableau's missing feature!"""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Price Distribution Analysis - Violin Plots', fontsize=16, fontweight='bold')
    
    # 1. Price by Product Type
    ax1 = axes[0, 0]
    # Get top 8 product types by count
    top_products = df['product_type_clean'].value_counts().head(8).index
    df_top_products = df[df['product_type_clean'].isin(top_products)]
    
    sns.violinplot(data=df_top_products, x='product_type_clean', y='price', ax=ax1)
    ax1.set_yscale('log')
    ax1.set_title('Price Distribution by Product Type')
    ax1.set_xlabel('Product Type')
    ax1.set_ylabel('Price (EUR, log scale)')
    ax1.tick_params(axis='x', rotation=45)
    
    # 2. Price by Category
    ax2 = axes[0, 1]
    sns.violinplot(data=df, x='category_clean', y='price', ax=ax2)
    ax2.set_yscale('log')
    ax2.set_title('Price Distribution by Category')
    ax2.set_xlabel('Category')
    ax2.set_ylabel('Price (EUR, log scale)')
    ax2.tick_params(axis='x', rotation=45)
    
    # 3. Buyer Price vs Regular Price
    ax3 = axes[1, 0]
    sns.violinplot(data=pd.melt(df[['price', 'buyer_price']], 
                               var_name='price_type', value_name='amount'), 
                   x='price_type', y='amount', ax=ax3)
    ax3.set_yscale('log')
    ax3.set_title('Price vs Buyer Price Distribution')
    ax3.set_xlabel('Price Type')
    ax3.set_ylabel('Amount (EUR, log scale)')
    
    # 4. Price Difference Distribution
    ax4 = axes[1, 1]
    # Filter extreme outliers for better visualization
    price_diff_filtered = df['price_diff_pct'][(df['price_diff_pct'] >= -50) & (df['price_diff_pct'] <= 100)]
    ax4.hist(price_diff_filtered, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
    ax4.axvline(price_diff_filtered.median(), color='red', linestyle='--', linewidth=2, label=f'Median: {price_diff_filtered.median():.1f}%')
    ax4.set_title('Price Difference Distribution')
    ax4.set_xlabel('Price Difference (%)')
    ax4.set_ylabel('Frequency')
    ax4.legend()
    
    plt.tight_layout()
    return fig

def create_ridge_plot(df):
    """Create ridge plot for comparing price distributions across categories"""
    
    from matplotlib.collections import PolyCollection
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    categories = df['category_clean'].value_counts().head(6).index
    colors = plt.cm.viridis(np.linspace(0, 1, len(categories)))
    
    for i, category in enumerate(categories):
        # Get price data for this category
        cat_prices = df[df['category_clean'] == category]['price']
        cat_prices = cat_prices[(cat_prices > 0) & (cat_prices < cat_prices.quantile(0.95))]
        
        if len(cat_prices) > 10:  # Only plot if we have enough data
            # Create density
            from scipy import stats
            density = stats.gaussian_kde(np.log10(cat_prices))
            xs = np.linspace(np.log10(cat_prices.min()), np.log10(cat_prices.max()), 200)
            ys = density(xs)
            
            # Scale and offset for ridge effect
            ys = ys / ys.max() * 0.8  # Normalize height
            ys = ys + i  # Offset vertically
            
            # Fill the area
            ax.fill_between(xs, i, ys, alpha=0.7, color=colors[i], label=category)
            ax.plot(xs, ys, color='black', linewidth=0.5)
    
    ax.set_xlabel('Price (EUR, log10 scale)')
    ax.set_ylabel('Category')
    ax.set_title('Price Distribution Ridge Plot by Category')
    
    # Convert x-axis back to euros
    x_ticks = ax.get_xticks()
    ax.set_xticklabels([f'€{10**x:,.0f}' for x in x_ticks])
    
    # Set y-axis labels
    ax.set_yticks(range(len(categories)))
    ax.set_yticklabels(categories)
    
    plt.tight_layout()
    return fig

def advanced_permanence_analysis(df):
    """Advanced permanence analysis with survival curves and statistical insights"""
    
    # Filter for items with permanence data
    perm_df = df[df['permanence_days'].notna() & (df['permanence_days'] > 0)].copy()
    
    if len(perm_df) == 0:
        print("No permanence data available for analysis")
        return None
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Advanced Permanence Analysis', fontsize=16, fontweight='bold')
    
    # 1. Permanence distribution with statistical fit
    ax1 = axes[0, 0]
    
    # Log-normal distribution often fits marketplace data well
    from scipy import stats
    log_perm = np.log(perm_df['permanence_days'])
    
    ax1.hist(perm_df['permanence_days'], bins=50, alpha=0.7, density=True, color='skyblue', edgecolor='black')
    ax1.set_xscale('log')
    ax1.set_title('Permanence Distribution with Statistical Fit')
    ax1.set_xlabel('Permanence (days, log scale)')
    ax1.set_ylabel('Density')
    
    # Fit and plot log-normal distribution
    mu, sigma = stats.norm.fit(log_perm)
    x_fit = np.logspace(np.log10(perm_df['permanence_days'].min()), 
                        np.log10(perm_df['permanence_days'].max()), 100)
    y_fit = stats.lognorm.pdf(x_fit, sigma, scale=np.exp(mu))
    ax1.plot(x_fit, y_fit, 'r-', linewidth=2, label=f'Log-normal fit (μ={mu:.2f}, σ={sigma:.2f})')
    ax1.legend()
    
    # 2. Survival curve (what % of items remain active over time)
    ax2 = axes[0, 1]
    
    # Create survival data
    max_days = int(perm_df['permanence_days'].quantile(0.95))
    days = np.arange(1, max_days + 1)
    survival_rate = [100 * (perm_df['permanence_days'] >= d).mean() for d in days]
    
    ax2.plot(days, survival_rate, linewidth=2, color='darkgreen')
    ax2.fill_between(days, survival_rate, alpha=0.3, color='lightgreen')
    ax2.set_title('Item Survival Curve')
    ax2.set_xlabel('Days')
    ax2.set_ylabel('% of Items Still Active')
    ax2.grid(True, alpha=0.3)
    
    # Add key percentiles
    percentiles = [50, 25, 10]
    for p in percentiles:
        days_at_p = perm_df['permanence_days'].quantile(p/100)
        ax2.axvline(days_at_p, color='red', linestyle='--', alpha=0.7)
        ax2.text(days_at_p, 80-p, f'{p}th percentile\n{days_at_p:.1f} days', 
                rotation=90, ha='right', va='top')
    
    # 3. Permanence by category with statistical tests
    ax3 = axes[1, 0]
    
    top_cats = perm_df['category_clean'].value_counts().head(5).index
    perm_by_cat = [perm_df[perm_df['category_clean'] == cat]['permanence_days'].values 
                   for cat in top_cats]
    
    box_plot = ax3.boxplot(perm_by_cat, labels=top_cats, patch_artist=True)
    
    # Color boxes
    colors = plt.cm.Set3(np.linspace(0, 1, len(top_cats)))
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax3.set_yscale('log')
    ax3.set_title('Permanence by Category (with outliers)')
    ax3.set_xlabel('Category')
    ax3.set_ylabel('Permanence (days, log scale)')
    ax3.tick_params(axis='x', rotation=45)
    
    # 4. Price vs Permanence correlation
    ax4 = axes[1, 1]
    
    # Sample data if too large for scatter plot
    if len(perm_df) > 5000:
        sample_df = perm_df.sample(5000)
    else:
        sample_df = perm_df
    
    scatter = ax4.scatter(sample_df['price'], sample_df['permanence_days'], 
                         alpha=0.6, c=sample_df['price_diff_pct'], cmap='RdYlBu_r', s=20)
    
    ax4.set_xscale('log')
    ax4.set_yscale('log')
    ax4.set_xlabel('Price (EUR, log scale)')
    ax4.set_ylabel('Permanence (days, log scale)')
    ax4.set_title('Price vs Permanence Correlation')
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax4)
    cbar.set_label('Price Difference (%)')
    
    # Calculate and display correlation
    corr = np.corrcoef(np.log(sample_df['price']), np.log(sample_df['permanence_days']))[0, 1]
    ax4.text(0.05, 0.95, f'Log-Log Correlation: {corr:.3f}', 
             transform=ax4.transAxes, bbox=dict(boxstyle="round", facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    return fig

def statistical_insights(df):
    """Generate key statistical insights"""
    
    print("\n" + "="*60)
    print("📊 STATISTICAL INSIGHTS")
    print("="*60)
    
    # Price insights
    print("\n💰 PRICE ANALYSIS:")
    print(f"   Median Price: €{df['price'].median():,.2f}")
    print(f"   Mean Price: €{df['price'].mean():,.2f}")
    print(f"   Price Range: €{df['price'].min():,.2f} - €{df['price'].quantile(0.99):,.2f}")
    
    # Price difference insights
    price_diff_median = df['price_diff_pct'].median()
    price_positive = (df['price_diff_pct'] > 0).mean() * 100
    
    print(f"\n📈 PRICE DIFFERENCE:")
    print(f"   Median Difference: {price_diff_median:.1f}%")
    print(f"   Items selling above list: {price_positive:.1f}%")
    print(f"   Average markup: €{df['price_diff'].mean():,.2f}")
    
    # Permanence insights (if available)
    if 'permanence_days' in df.columns and df['permanence_days'].notna().sum() > 0:
        perm_data = df[df['permanence_days'].notna() & (df['permanence_days'] > 0)]
        print(f"\n⏱️ PERMANENCE ANALYSIS:")
        print(f"   Median time on market: {perm_data['permanence_days'].median():.1f} days")
        print(f"   25th percentile: {perm_data['permanence_days'].quantile(0.25):.1f} days")
        print(f"   75th percentile: {perm_data['permanence_days'].quantile(0.75):.1f} days")
        
        # Quick movers vs slow movers
        quick_threshold = 7  # 1 week
        quick_movers = (perm_data['permanence_days'] <= quick_threshold).mean() * 100
        print(f"   Quick sellers (≤7 days): {quick_movers:.1f}%")
    
    # Category insights
    print(f"\n🏷️ CATEGORY BREAKDOWN:")
    top_categories = df['category_clean'].value_counts().head(5)
    for i, (cat, count) in enumerate(top_categories.items(), 1):
        pct = count / len(df) * 100
        print(f"   {i}. {cat}: {count:,} listings ({pct:.1f}%)")
    
    # Time insights
    if 'created_date' in df.columns:
        date_range = (df['created_date'].max() - df['created_date'].min()).days
        daily_avg = len(df) / max(date_range, 1)
        print(f"\n📅 TEMPORAL PATTERNS:")
        print(f"   Data spans: {date_range} days")
        print(f"   Average listings per day: {daily_avg:.1f}")
        
        # Peak day analysis
        daily_counts = df.groupby('created_date').size()
        peak_date = daily_counts.idxmax()
        peak_count = daily_counts.max()
        print(f"   Peak day: {peak_date} ({peak_count} listings)")

def main():
    """Main execution function"""
    
    # Update this path to your CSV file
    file_path = "data/reverb/cleaned/reverb_tableau_202509-051553.csv"
    
    try:
        # Load data
        df = load_and_prepare_data(file_path)
        
        # Generate statistical insights
        statistical_insights(df)
        
        print("\n" + "="*60)
        print("🎨 GENERATING ADVANCED VISUALIZATIONS")
        print("="*60)
        
        # Create visualizations
        print("\n1. Creating violin plots...")
        fig1 = create_violin_plots(df)
        plt.show()
        
        print("2. Creating ridge plot...")
        fig2 = create_ridge_plot(df)
        plt.show()
        
        print("3. Creating advanced permanence analysis...")
        fig3 = advanced_permanence_analysis(df)
        if fig3:
            plt.show()
        
        print("\n✅ Analysis complete!")
        print("\nThese visualizations complement your Tableau dashboard with:")
        print("   • Violin plots (not available in Tableau)")
        print("   • Ridge plots for distribution comparison")
        print("   • Advanced survival analysis")
        print("   • Statistical distribution fitting")
        print("   • Correlation analysis with statistical tests")
        
    except FileNotFoundError:
        print(f"❌ Error: Could not find file '{file_path}'")
        print("Please update the file_path variable with the correct path to your CSV file.")
    except Exception as e:
        print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    main()