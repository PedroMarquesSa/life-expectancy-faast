"""
Clean and process EU life expectancy data for Portugal.
Saves the cleaned data to a CSV file.
"""
from pathlib import Path
import argparse
import pandas as pd


def clean_data(country: str, input_file: str = None, output_file: str = None) -> pd.DataFrame:
    """
    Clean and process EU life expectancy data.
    
    Args:
        country: Country code to filter by. Defaults to PT (Portugal)
        input_file: Path to input TSV file. Defaults to data/eu_life_expectancy_raw.tsv
        output_file: Path to output CSV file. Defaults to data/pt_life_expectancy.csv
    
    Returns:
        Cleaned DataFrame filtered for the specified country
    """
    # Set default paths relative to the script location
    script_dir = Path(__file__).parent
    if input_file is None:
        input_file = script_dir / "data" / "eu_life_expectancy_raw.tsv"
    if output_file is None:
        output_file = script_dir / "data" / f"{country.lower()}_life_expectancy.csv"

    # Load the data
    df = pd.read_csv(input_file, sep='\t')

    # The first column contains multiple fields separated by commas
    # Split it into separate columns
    first_col = df.columns[0]
    split_cols = df[first_col].str.split(',', expand=True)
    split_cols.columns = ['unit', 'sex', 'age', 'region']

    # Drop the original column and add the split columns
    df = df.drop(columns=[first_col])
    df = pd.concat([split_cols, df], axis=1)

    # Unpivot to long format
    id_vars = ['unit', 'sex', 'age', 'region']
    df = df.melt(id_vars=id_vars, var_name='year', value_name='value')

    # Clean year column - remove whitespace and convert to int
    df['year'] = df['year'].str.strip()
    df['year'] = pd.to_numeric(df['year'], errors='coerce')
    df = df.dropna(subset=['year'])
    df['year'] = df['year'].astype(int)

    # Clean value column - remove non-numeric characters and convert to float
    # Replace ':' and other non-numeric values with NaN
    df['value'] = df['value'].astype(str).str.strip()
    df['value'] = df['value'].replace(':', pd.NA)
    df['value'] = df['value'].str.replace(r'[^\d.]', '', regex=True)
    df['value'] = pd.to_numeric(df['value'], errors='coerce')

    # Remove NaN values
    df = df.dropna(subset=['value'])

    # Filter for specified country
    df = df[df['region'] == country]

    # Save to CSV without index
    df.to_csv(output_file, index=False)

    return df


def main():    # pragma: no cover
    """Main entry point for command-line execution."""
    parser = argparse.ArgumentParser(
        description="Clean and process EU life expectancy data"
    )
    parser.add_argument(
        "--country",
        type=str,
        default="PT",
        help="Country code to filter by (default: PT)"
    )
    args = parser.parse_args()
    clean_data(country=args.country)


if __name__ == '__main__':    # pragma: no cover
    main()
