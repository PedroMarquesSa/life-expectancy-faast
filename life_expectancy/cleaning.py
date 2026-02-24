"""
Clean and process EU life expectancy data for Portugal.
Saves the cleaned data to a CSV file.
"""

from pathlib import Path
import argparse
from enum import Enum
import pandas as pd
from typing import Optional


class Region(Enum):
    """Enum containing all possible region codes in the life expectancy dataset."""
    AL = "AL"
    AM = "AM"
    AT = "AT"
    AZ = "AZ"
    BE = "BE"
    BG = "BG"
    BY = "BY"
    CH = "CH"
    CY = "CY"
    CZ = "CZ"
    DE = "DE"
    DE_TOT = "DE_TOT"
    DK = "DK"
    EA18 = "EA18"
    EA19 = "EA19"
    EE = "EE"
    EEA30_2007 = "EEA30_2007"
    EEA31 = "EEA31"
    EFTA = "EFTA"
    EL = "EL"
    ES = "ES"
    EU27_2007 = "EU27_2007"
    EU27_2020 = "EU27_2020"
    EU28 = "EU28"
    FI = "FI"
    FR = "FR"
    FX = "FX"
    GE = "GE"
    HR = "HR"
    HU = "HU"
    IE = "IE"
    IS = "IS"
    IT = "IT"
    LI = "LI"
    LT = "LT"
    LU = "LU"
    LV = "LV"
    MD = "MD"
    ME = "ME"
    MK = "MK"
    MT = "MT"
    NL = "NL"
    NO = "NO"
    PL = "PL"
    PT = "PT"
    RO = "RO"
    RS = "RS"
    RU = "RU"
    SE = "SE"
    SI = "SI"
    SK = "SK"
    SM = "SM"
    TR = "TR"
    UA = "UA"
    UK = "UK"
    XK = "XK"

    @classmethod
    def countries(cls) -> list["Region"]:
        """
        Return a list of Region members that represent actual countries.

        Excludes aggregate regions like EU27, EA18, EFTA, etc.

        Returns:
            List of Region enum members representing individual countries
        """
        # Define aggregate/non-country region codes to exclude
        aggregates = {
            "DE_TOT",  # Germany total (aggregate)
            "EA18",    # Euro area (18 countries)
            "EA19",    # Euro area (19 countries)
            "EEA30_2007",  # European Economic Area
            "EEA31",   # European Economic Area
            "EFTA",    # European Free Trade Association
            "EU27_2007",  # European Union (27 countries, 2007 definition)
            "EU27_2020",  # European Union (27 countries, 2020 definition)
            "EU28",    # European Union (28 countries)
            "FX",      # France (metropolitan)
        }

        return [region for region in cls if region.name not in aggregates]


def load_data(input_file: Optional[str] = None) -> pd.DataFrame:
    """
    Load EU life expectancy data.
    Args:
        input_file: Path to input TSV file. Defaults to data/eu_life_expectancy_raw.tsv
    Returns:
        DataFrame loaded from the specified TSV file
    """
    # Set default paths relative to the script location
    script_dir = Path(__file__).parent
    if input_file is None:
        input_file = script_dir / "data" / "eu_life_expectancy_raw.tsv"

    # Load the data
    df = pd.read_csv(input_file, sep="\t")
    return df


def clean_data(df: pd.DataFrame, country: Optional[Region]) -> pd.DataFrame:
    """
    Clean and process EU life expectancy data for a specified country.
    Args:
        df: DataFrame containing the raw life expectancy data
        country: Optional Region enum to filter by
    Returns:
        Cleaned DataFrame filtered for the specified country
    """
    # The first column contains multiple fields separated by commas
    # Split it into separate columns
    first_col = df.columns[0]
    split_cols = df[first_col].str.split(",", expand=True)
    split_cols.columns = ["unit", "sex", "age", "region"]

    # Drop the original column and add the split columns
    df = df.drop(columns=[first_col])
    df = pd.concat([split_cols, df], axis=1)

    # Unpivot to long format
    id_vars = ["unit", "sex", "age", "region"]
    df = df.melt(id_vars=id_vars, var_name="year", value_name="value")

    # Clean year column - remove whitespace and convert to int
    df["year"] = df["year"].str.strip()
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df.dropna(subset=["year"])
    df["year"] = df["year"].astype(int)

    # Clean value column - remove non-numeric characters and convert to float
    # Replace ':' and other non-numeric values with NaN
    df["value"] = df["value"].astype(str).str.strip()
    df["value"] = df["value"].replace(":", pd.NA)
    df["value"] = df["value"].str.replace(r"[^\d.]", "", regex=True)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    # Remove NaN values
    df = df.dropna(subset=["value"])

    # Filter for specified country if provided
    if country is not None:
        df = df[df["region"] == country.value]
        if df.empty:
            raise ValueError(f"No data found for country code: {country.value}")

    # Reset index to have a clean sequential index
    df = df.reset_index(drop=True)

    return df


def save_data(df: pd.DataFrame, country: Region, output_file: Optional[str]) -> None:
    """
    Save cleaned DataFrame to a CSV file.
    Args:
        df: Cleaned DataFrame to save
        country: Region enum for the country
        output_file: Path to output CSV file
    """
    # Set default paths relative to the script location
    script_dir = Path(__file__).parent
    if output_file is None:
        output_file = script_dir / "data" / f"{country.value.lower()}_life_expectancy_raw.tsv"

    # Save to CSV without index
    df.to_csv(output_file, index=False)


def main():  # pragma: no cover
    """Main entry point for command-line execution."""
    parser = argparse.ArgumentParser(
        description="Clean and process EU life expectancy data"
    )
    parser.add_argument(
        "--country",
        type=str,
        default="PT",
        help="Country code to filter by (default: PT)",
    )
    args = parser.parse_args()

    # Convert string to Region enum
    country = Region[args.country]

    data = load_data()
    cleaned_data = clean_data(data, country=country)
    save_data(cleaned_data, country=country)
    return cleaned_data


if __name__ == "__main__":  # pragma: no cover
    main()
