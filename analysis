import pandas as pd
import logging
from datetime import datetime
import os

# Set up logging to track progress and errors
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='fundraising_analysis.log'
)
logger = logging.getLogger(__name__)

def load_fundraising_data(file_path):
    """
    Load the fundraising data from a CSV file.
    Returns a DataFrame or None if there's an issue.
    """
    logger.info(f"Attempting to load data from {file_path}")
    try:
        data = pd.read_csv(file_path)
        logger.info(f"Successfully loaded {len(data)} records")
        return data
    except FileNotFoundError:
        logger.error(f"File {file_path} not found")
        print(f"Oops! Couldn't find {file_path}. Please check the file path.")
        return None
    except pd.errors.EmptyDataError:
        logger.error("CSV file is empty")
        print("The CSV file is empty. Please provide valid data.")
        return None
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        print(f"Something went wrong while loading the data: {str(e)}")
        return None

def clean_fundraising_data(data):
    """
    Clean the data by handling missing values and ensuring correct data types.
    Returns the cleaned DataFrame.
    """
    logger.info("Starting data cleaning")
    original_rows = len(data)
    
    # Define critical columns
    critical_columns = ['id', 'category', 'donation_volume_usd', 'count_donations', 'count_team_members']
    
    # Check for missing critical columns
    missing_cols = [col for col in critical_columns if col not in data.columns]
    if missing_cols:
        logger.error(f"Missing critical columns: {missing_cols}")
        print(f"Error: Missing columns {missing_cols}. Please check the CSV file.")
        return None
    
    # Remove rows with missing critical data
    data = data.dropna(subset=critical_columns)
    
    # Convert numeric columns
    numeric_cols = ['donation_volume_usd', 'count_donations', 'count_team_members']
    for col in numeric_cols:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    
    # Convert publish_at to datetime
    data['publish_at'] = pd.to_datetime(data['publish_at'], errors='coerce')
    
    # Drop rows with invalid numeric data
    data = data.dropna(subset=numeric_cols)
    
    # Log cleaning results
    cleaned_rows = len(data)
    logger.info(f"Cleaned data: {original_rows - cleaned_rows} rows removed, {cleaned_rows} rows remain")
    
    if cleaned_rows == 0:
        logger.warning("No valid data after cleaning")
        print("Warning: No valid data left after cleaning. Please check the input data.")
        return None
    
    return data

def add_derived_columns(data):
    """
    Add derived columns to enrich the dataset for analysis.
    Returns the DataFrame with new columns.
    """
    logger.info("Adding derived columns")
    
    # Success indicator: Fundraiser is successful if it has donations
    data['is_successful'] = (data['donation_volume_usd'] > 0).astype(int)
    
    # Team indicator: Fundraiser has a team if count_team_members > 0
    data['has_team'] = (data['count_team_members'] > 0).astype(int)
    
    # Average donation size (handle zero donations)
    data['avg_donation_size'] = data['donation_volume_usd'] / data['count_donations'].replace(0, pd.NA)
    data['avg_donation_size'] = data['avg_donation_size'].fillna(0).round(2)
    
    # Days since published (use current date: 2025-04-21)
    current_date = pd.to_datetime('2025-04-21')
    data['days_since_published'] = (current_date - data['publish_at']).dt.days
    data['days_since_published'] = data['days_since_published'].fillna(0).astype(int)
    
    # Donation rate: Donations per day (handle zero days)
    data['donation_rate'] = data['count_donations'] / data['days_since_published'].replace(0, pd.NA)
    data['donation_rate'] = data['donation_rate'].fillna(0).round(2)
    
    # Donation velocity: Donation volume per day
    data['donation_velocity'] = data['donation_volume_usd'] / data['days_since_published'].replace(0, pd.NA)
    data['donation_velocity'] = data['donation_velocity'].fillna(0).round(2)
    
    # Team efficiency: Donations per team member (handle zero team members)
    data['donations_per_team_member'] = data['count_donations'] / data['count_team_members'].replace(0, pd.NA)
    data['donations_per_team_member'] = data['donations_per_team_member'].fillna(0).round(2)
    
    # Success momentum: Donation volume per day for successful fundraisers
    data['success_momentum'] = data.apply(
        lambda row: row['donation_volume_usd'] / row['days_since_published'] if row['is_successful'] and row['days_since_published'] > 0 else 0,
        axis=1
    ).round(2)
    
    logger.info("Derived columns added successfully")
    return data

def aggregate_by_category(data):
    """
    Aggregate data by category and calculate key metrics.
    Returns the aggregated DataFrame.
    """
    logger.info("Aggregating data by category")
    
    # Aggregate metrics
    summary = data.groupby('category').agg({
        'donation_volume_usd': ['sum', 'median', 'mean'],
        'count_donations': ['sum', 'median', 'mean'],
        'count_team_members': ['sum', 'mean'],
        'id': 'count',
        'is_successful': 'sum',
        'has_team': 'sum',
        'avg_donation_size': 'median',
        'days_since_published': 'median',
        'donation_rate': 'median',
        'donation_velocity': 'median',
        'donations_per_team_member': 'median',
        'success_momentum': 'median'
    }).reset_index()
    
    # Flatten column names
    summary.columns = [
        'category',
        'total_donation_volume', 'median_donation_volume', 'avg_donation_volume',
        'total_donations', 'median_donations', 'avg_donations',
        'total_team_members', 'avg_team_members',
        'count_fundraisers',
        'successful_fundraisers',
        'fundraisers_with_team',
        'median_avg_donation_size',
        'median_days_active',
        'median_donation_rate',
        'median_donation_velocity',
        'median_donations_per_team_member',
        'median_success_momentum'
    ]
    
    # Additional derived metrics
    # Success rate
    summary['success_rate'] = (summary['successful_fundraisers'] / summary['count_fundraisers'] * 100).round(2)
    
    # Team usage percentage
    summary['team_usage_pct'] = (summary['fundraisers_with_team'] / summary['count_fundraisers'] * 100).round(2)
    
    # Team members per fundraiser
    summary['team_members_per_fundraiser'] = (summary['total_team_members'] / summary['count_fundraisers']).round(4)
    
    # Team members per successful fundraiser
    summary['team_members_per_successful'] = (summary['total_team_members'] / summary['successful_fundraisers'].replace(0, pd.NA)).round(4)
    summary['team_members_per_successful'] = summary['team_members_per_successful'].fillna(0)
    
    # Average donation volume per fundraiser
    summary['avg_donation_volume_per_fundraiser'] = (summary['total_donation_volume'] / summary['count_fundraisers']).round(2)
    
    # Donation concentration: Total donations per fundraiser
    summary['donations_per_fundraiser'] = (summary['total_donations'] / summary['count_fundraisers']).round(2)
    
    # Grand Total row
    grand_total = pd.DataFrame({
        'category': ['Grand Total'],
        'total_donation_volume': [summary['total_donation_volume'].sum()],
        'median_donation_volume': [summary['median_donation_volume'].median()],
        'avg_donation_volume': [summary['avg_donation_volume'].mean()],
        'total_donations': [summary['total_donations'].sum()],
        'median_donations': [summary['median_donations'].median()],
        'avg_donations': [summary['avg_donations'].mean()],
        'total_team_members': [summary['total_team_members'].sum()],
        'avg_team_members': [summary['avg_team_members'].mean()],
        'count_fundraisers': [summary['count_fundraisers'].sum()],
        'successful_fundraisers': [summary['successful_fundraisers'].sum()],
        'fundraisers_with_team': [summary['fundraisers_with_team'].sum()],
        'median_avg_donation_size': [summary['median_avg_donation_size'].median()],
        'median_days_active': [summary['median_days_active'].median()],
        'median_donation_rate': [summary['median_donation_rate'].median()],
        'median_donation_velocity': [summary['median_donation_velocity'].median()],
        'median_donations_per_team_member': [summary['median_donations_per_team_member'].median()],
        'median_success_momentum': [summary['median_success_momentum'].median()],
        'success_rate': [(summary['successful_fundraisers'].sum() / summary['count_fundraisers'].sum() * 100).round(2)],
        'team_usage_pct': [(summary['fundraisers_with_team'].sum() / summary['count_fundraisers'].sum() * 100).round(2)],
        'team_members_per_fundraiser': [(summary['total_team_members'].sum() / summary['count_fundraisers'].sum()).round(4)],
        'team_members_per_successful': [(summary['total_team_members'].sum() / summary['successful_fundraisers'].sum()).round(4)],
        'avg_donation_volume_per_fundraiser': [(summary['total_donation_volume'].sum() / summary['count_fundraisers'].sum()).round(2)],
        'donations_per_fundraiser': [(summary['total_donations'].sum() / summary['count_fundraisers'].sum()).round(2)]
    })
    
    # Append Grand Total
    summary = pd.concat([summary, grand_total], ignore_index=True)
    
    logger.info("Data aggregation completed")
    return summary

def analyze_and_print_findings(data, summary):
    """
    Perform detailed analysis and print findings, focusing on discoverability.
    """
    logger.info("Starting analysis and findings generation")
    print("\n=== Fundraising Data Analysis Findings ===")
    
    # 1. Overall Performance
    total_fundraisers = summary[summary['category'] == 'Grand Total']['count_fundraisers'].iloc[0]
    total_donations = summary[summary['category'] == 'Grand Total']['total_donation_volume'].iloc[0]
    overall_success_rate = summary[summary['category'] == 'Grand Total']['success_rate'].iloc[0]
    print("\n1. Overall Performance:")
    print(f"   - Total fundraisers: {total_fundraisers:,}")
    print(f"   - Total donation volume: ${total_donations:,.2f}")
    print(f"   - Overall success rate: {overall_success_rate:.2f}%")
    print("   - Insight: High donation volumes indicate strong donor engagement, likely driven by discoverability.")
    
    # 2. Top Performing Categories
    top_categories = summary[summary['category'] != 'Grand Total'].nlargest(3, 'total_donation_volume')
    print("\n2. Top Performing Categories:")
    for _, row in top_categories.iterrows():
        print(f"   - {row['category']}: ${row['total_donation_volume']:,.2f}, {row['success_rate']:.2f}% success rate")
    print("   - Insight: Categories with high volumes and success rates likely benefit from emotional narratives and platform visibility.")
    
    # 3. Team Impact on Discoverability
    team_analysis = summary[summary['category'] != 'Grand Total'][['category', 'team_usage_pct', 'success_rate', 'median_donations_per_team_member']]
    high_team_usage = team_analysis.nlargest(3, 'team_usage_pct')
    print("\n3. Team Impact on Discoverability:")
    for _, row in high_team_usage.iterrows():
        print(f"   - {row['category']}: {row['team_usage_pct']:.2f}% team usage, {row['success_rate']:.2f}% success rate, {row['median_donations_per_team_member']:.2f} donations/team member")
    print("   - Insight: Teams boost discoverability by leveraging networks, increasing donations and success.")
    
    # 4. Discoverability via Donation Patterns
    donation_patterns = summary[summary['category'] != 'Grand Total'][['category', 'median_donation_volume', 'median_donations', 'median_avg_donation_size']]
    high_donation = donation_patterns.nlargest(3, 'median_donation_volume')
    print("\n4. Discoverability via Donation Patterns:")
    for _, row in high_donation.iterrows():
        print(f"   - {row['category']}: ${row['median_donation_volume']:,.0f} median donation, {row['median_donations']} median donations, ${row['median_avg_donation_size']:,.2f} avg donation size")
    print("   - Insight: High median donations and frequent contributions suggest visibility through social sharing or algorithmic promotion.")
    
    # 5. Low Performers and Discoverability Challenges
    low_performers = summary[summary['category'] != 'Grand Total'].nsmallest(3, 'success_rate')
    print("\n5. Low Performing Categories:")
    for _, row in low_performers.iterrows():
        print(f"   - {row['category']}: {row['success_rate']:.2f}% success rate, {row['team_usage_pct']:.2f}% team usage")
    print("   - Insight: Low success and team usage indicate discoverability issues; focus on team recruitment and compelling stories.")
    
    # 6. Time-Based Discoverability
    time_analysis = summary[summary['category'] != 'Grand Total'][['category', 'median_donation_rate', 'median_days_active', 'median_success_momentum']]
    high_donation_rate = time_analysis.nlargest(3, 'median_donation_rate')
    print("\n6. Time-Based Discoverability:")
    for _, row in high_donation_rate.iterrows():
        print(f"   - {row['category']}: {row['median_donation_rate']:.2f} donations/day, {row['median_days_active']} days active, ${row['median_success_momentum']:.2f}/day momentum")
    print("   - Insight: High donation rates and momentum suggest early traction enhances visibility.")
    
    # 7. Correlation Analysis
    correlations = data[['donation_volume_usd', 'count_donations', 'count_team_members', 'days_since_published', 'donation_rate']].corr()
    print("\n7. Correlation Analysis:")
    print(f"   - Donation volume vs. Team members: {correlations.loc['donation_volume_usd', 'count_team_members']:.2f}")
    print(f"   - Donation volume vs. Donation count: {correlations.loc['donation_volume_usd', 'count_donations']:.2f}")
    print(f"   - Donation rate vs. Team members: {correlations.loc['donation_rate', 'count_team_members']:.2f}")
    print("   - Insight: Strong correlations between team size and donations highlight teams' role in discoverability.")
    
    # 8. Outlier Detection
    outliers = data[data['donation_volume_usd'] > data['donation_volume_usd'].quantile(0.95)]
    outlier_categories = outliers['category'].value_counts().head(3)
    print("\n8. Outlier Analysis (Top 5% Donation Volumes):")
    for category, count in outlier_categories.items():
        print(f"   - {category}: {count} high-value fundraisers")
    print("   - Insight: Outliers in top categories may drive visibility through viral sharing or media attention.")
    
    logger.info("Analysis and findings generation completed")

def save_results(summary, output_file='fundraising_analysis.csv'):
    """
    Save the aggregated results to a CSV file.
    """
    logger.info(f"Saving results to {output_file}")
    try:
        summary.to_csv(output_file, index=False)
        print(f"\nResults saved to {output_file}")
    except Exception as e:
        logger.error(f"Error saving results: {str(e)}")
        print(f"Oops! Couldn't save the results: {str(e)}")

def main():
    """
    Main function to orchestrate the fundraising data analysis.
    """
    print("Starting Fundraising Data Analysis...")
    logger.info("Program started")
    
    # File paths
    input_file = 'fundraising_data.csv'
    output_file = 'fundraising_analysis.csv'
    
    # Load data
    data = load_fundraising_data(input_file)
    if data is None:
        logger.warning("Exiting due to data loading failure")
        return
    
    # Clean data
    data = clean_fundraising_data(data)
    if data is None:
        logger.warning("Exiting due to data cleaning failure")
        return
    
    # Add derived columns
    data = add_derived_columns(data)
    
    # Aggregate data by category
    summary = aggregate_by_category(data)
    
    # Perform analysis and print findings
    analyze_and_print_findings(data, summary)
    
    # Save results
    save_results(summary, output_file)
    
    print("\nAnalysis complete! Check the log file (fundraising_analysis.log) for details.")
    logger.info("Program completed successfully")

if __name__ == '__main__':
    main()
