import pandas as pd

def convert_cumulative_to_interval(csv_file):
    # Read the CSV file
    df = pd.read_csv(csv_file, parse_dates=['utc_timestamp'])
    
    # Calculate the difference between consecutive readings
    # This gives us the actual consumption for each 15-min interval
    df['interval_consumption'] = df['DE_KN_residential1_grid_import'].diff()
    
    # Drop the first row since it will have NaN (no previous value to subtract)
    df = df.dropna()
    
    # Round to 3 decimal places for cleaner output
    df['interval_consumption'] = df['interval_consumption'].round(3)
    
    # Create new DataFrame with timestamp and interval consumption
    result_df = pd.DataFrame({
        'utc_timestamp': df['utc_timestamp'],
        'energy_consumption': df['interval_consumption']
    })
    
    # Save to new CSV file
    output_file = 'data/interval_consumption.csv'
    result_df.to_csv(output_file, index=False)
    print(f"Converted data saved to {output_file}")
    
    return result_df

# Usage example
if __name__ == "__main__":
    # Replace 'your_input_file.csv' with your actual file name
    df = convert_cumulative_to_interval('data/household_15min.csv')
    print("\nFirst few rows of converted data:")
    print(df.head())