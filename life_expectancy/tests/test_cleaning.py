"""Tests for the cleaning module"""

from unittest.mock import patch
import pandas as pd
import pytest

from life_expectancy.cleaning import (
    load_data,
    clean_data,
    save_data,
    main,
    Region,
    TSVDataLoader,
    JSONDataLoader
)
from . import FIXTURES_DIR


# Unit Tests

def test_load_data_default():
    """Test that load_data loads the default file correctly"""
    # This test will use the actual default file
    df = load_data()

    # Verify it's a DataFrame
    assert isinstance(df, pd.DataFrame)
    # Verify it has data
    assert len(df) > 0
    # Verify it has the expected columns after loading through TSV loader
    expected_columns = ["unit", "sex", "age", "region", "year", "value"]
    assert list(df.columns) == expected_columns


def test_load_data_custom_file():
    """Test that load_data can load a custom file"""
    fixture_path = FIXTURES_DIR / "eu_life_expectancy_raw.tsv"
    df = load_data(input_file=str(fixture_path))

    # Verify it's a DataFrame
    assert isinstance(df, pd.DataFrame)
    # Verify it has the expected columns after loading through TSV loader
    expected_columns = ["unit", "sex", "age", "region", "year", "value"]
    assert list(df.columns) == expected_columns
    # Verify it has data
    assert len(df) > 0


def test_clean_data_no_country_filter():
    """Test clean_data without country filtering"""
    # Load data through the TSV loader
    fixture_path = FIXTURES_DIR / "eu_life_expectancy_raw.tsv"
    raw_data = load_data(input_file=str(fixture_path))
    cleaned = clean_data(raw_data, country=None)

    # Verify the structure
    assert isinstance(cleaned, pd.DataFrame)
    expected_columns = ["unit", "sex", "age", "region", "year", "value"]
    assert list(cleaned.columns) == expected_columns

    # Verify data types
    assert cleaned["year"].dtype == "int64"
    assert cleaned["value"].dtype == "float64"

    # Verify no NaN values in year or value
    assert cleaned["year"].notna().all()
    assert cleaned["value"].notna().all()

    # Verify regions are present
    assert len(cleaned["region"].unique()) > 0


def test_clean_data_with_country_filter():
    """Test clean_data with country filtering"""
    # Load data through the TSV loader
    fixture_path = FIXTURES_DIR / "eu_life_expectancy_raw.tsv"
    raw_data = load_data(input_file=str(fixture_path))
    cleaned = clean_data(raw_data, country=Region.PT)

    # Verify only PT data is present
    assert (cleaned["region"] == "PT").all()

    # Verify we have PT data
    assert len(cleaned) > 0


def test_clean_data_invalid_country():
    """Test clean_data raises error for non-existent region in data"""
    # Load data through the TSV loader
    fixture_path = FIXTURES_DIR / "eu_life_expectancy_raw.tsv"
    raw_data = load_data(input_file=str(fixture_path))
    # Use a valid Region enum that doesn't exist in the fixture data
    with pytest.raises(ValueError, match="No data found for country code"):
        clean_data(raw_data, country=Region.UK)


def test_clean_data_transforms_correctly():
    """Test that clean_data performs expected transformations"""
    # Load data through the TSV loader
    fixture_path = FIXTURES_DIR / "eu_life_expectancy_raw.tsv"
    raw_data = load_data(input_file=str(fixture_path))
    cleaned = clean_data(raw_data, country=Region.PT)

    # Verify index is reset (starts at 0)
    assert cleaned.index[0] == 0
    assert cleaned.index[-1] == len(cleaned) - 1

    # Verify all units, sex, age, region are strings (object or StringDtype)
    assert pd.api.types.is_string_dtype(cleaned["unit"])
    assert pd.api.types.is_string_dtype(cleaned["sex"])
    assert pd.api.types.is_string_dtype(cleaned["age"])
    assert pd.api.types.is_string_dtype(cleaned["region"])


@patch('pandas.DataFrame.to_csv')
def test_save_data_default_path(mock_to_csv):
    """Test save_data with default output path"""
    # Create a sample DataFrame
    df = pd.DataFrame({
        "unit": ["YR", "YR"],
        "sex": ["F", "M"],
        "age": ["Y1", "Y1"],
        "region": ["PT", "PT"],
        "year": [2020, 2020],
        "value": [80.0, 75.0]
    })

    # Call save_data with default path
    save_data(df, country=Region.PT)

    # Verify to_csv was called
    assert mock_to_csv.called
    # Verify it was called with index=False
    mock_to_csv.assert_called_once()
    call_kwargs = mock_to_csv.call_args[1]
    assert call_kwargs.get("index") is False


@patch('pandas.DataFrame.to_csv')
def test_save_data_custom_path(mock_to_csv):
    """Test save_data with custom output path"""
    # Create a sample DataFrame
    df = pd.DataFrame({
        "unit": ["YR"],
        "sex": ["F"],
        "age": ["Y1"],
        "region": ["PT"],
        "year": [2020],
        "value": [80.0]
    })

    custom_path = "/tmp/test_output.csv"
    save_data(df, country=Region.PT, output_file=custom_path)

    # Verify to_csv was called with the custom path
    assert mock_to_csv.called
    call_args = mock_to_csv.call_args[0]
    assert call_args[0] == custom_path


@patch('life_expectancy.cleaning.save_data')
@patch('life_expectancy.cleaning.clean_data')
@patch('life_expectancy.cleaning.load_data')
def test_main_function(mock_load, mock_clean, mock_save):
    """Test the main function orchestrates all steps correctly"""
    # Setup mocks
    mock_df_raw = pd.DataFrame({"col": [1, 2, 3]})
    mock_df_clean = pd.DataFrame({"col": [1, 2]})

    mock_load.return_value = mock_df_raw
    mock_clean.return_value = mock_df_clean

    # Mock sys.argv to simulate command line arguments
    with patch('sys.argv', ['cleaning.py', '--country', 'PT']):
        result = main()

    # Verify all functions were called
    mock_load.assert_called_once()
    mock_clean.assert_called_once_with(mock_df_raw, country=Region.PT)
    mock_save.assert_called_once_with(mock_df_clean, country=Region.PT)

    # Verify the cleaned data is returned
    assert result is mock_df_clean


@patch('life_expectancy.cleaning.save_data')
@patch('life_expectancy.cleaning.clean_data')
@patch('life_expectancy.cleaning.load_data')
def test_main_function_default_country(mock_load, mock_clean, mock_save):
    """Test the main function with default country argument"""
    # Setup mocks
    mock_df_raw = pd.DataFrame({"col": [1, 2, 3]})
    mock_df_clean = pd.DataFrame({"col": [1, 2]})

    mock_load.return_value = mock_df_raw
    mock_clean.return_value = mock_df_clean

    # Mock sys.argv with no country argument (should default to PT)
    with patch('sys.argv', ['cleaning.py']):
        main()

    # Verify clean_data was called with default Region.PT
    mock_clean.assert_called_once_with(mock_df_raw, country=Region.PT)
    mock_save.assert_called_once_with(mock_df_clean, country=Region.PT)


# Region Enum Tests

def test_region_countries():
    """Test that Region.countries() returns only actual countries"""
    countries = Region.countries()

    # Verify it returns a list
    assert isinstance(countries, list)

    # Verify all items are Region enum members
    assert all(isinstance(region, Region) for region in countries)

    # Verify we have countries (should be less than total regions)
    assert len(countries) > 0
    assert len(countries) < len(Region)

    # Verify actual countries are included
    assert Region.PT in countries
    assert Region.ES in countries
    assert Region.FR in countries
    assert Region.IT in countries
    assert Region.DE in countries

    # Verify aggregates are excluded
    assert Region.EU27_2007 not in countries
    assert Region.EU27_2020 not in countries
    assert Region.EU28 not in countries
    assert Region.EA18 not in countries
    assert Region.EA19 not in countries
    assert Region.EFTA not in countries
    assert Region.EEA30_2007 not in countries
    assert Region.EEA31 not in countries
    assert Region.DE_TOT not in countries
    assert Region.FX not in countries

    # Verify the count is correct (56 total - 10 aggregates = 46 countries)
    assert len(countries) == 46


# Integration Test

def test_clean_data_integration(eu_life_expectancy_expected):
    """Integration test: Run the `clean_data` function and compare to expected output"""
    # Load the raw fixture data through the TSV loader
    fixture_path = FIXTURES_DIR / "eu_life_expectancy_raw.tsv"
    raw_data = load_data(input_file=str(fixture_path))
    cleaned_data = clean_data(raw_data, country=None)

    # Compare with expected output
    pd.testing.assert_frame_equal(
        cleaned_data, eu_life_expectancy_expected
    )

# Strategy Pattern Tests - Data Loader Tests

def test_tsv_data_loader():
    """Test TSVDataLoader can load TSV format correctly"""
    loader = TSVDataLoader()
    fixture_path = FIXTURES_DIR / "eu_life_expectancy_raw.tsv"
    df = loader.load(str(fixture_path))

    # Verify it's a DataFrame
    assert isinstance(df, pd.DataFrame)

    # Verify it has the expected columns
    expected_columns = ["unit", "sex", "age", "region", "year", "value"]
    assert list(df.columns) == expected_columns

    # Verify it has data
    assert len(df) > 0

    # Verify data types (before cleaning, year and value are still strings)
    assert pd.api.types.is_string_dtype(df["unit"])
    assert pd.api.types.is_string_dtype(df["region"])


def test_json_data_loader():
    """Test JSONDataLoader can load JSON format correctly"""
    loader = JSONDataLoader()
    fixture_path = FIXTURES_DIR / "eu_life_expectancy_raw.json"
    df = loader.load(str(fixture_path))

    # Verify it's a DataFrame
    assert isinstance(df, pd.DataFrame)

    # Verify it has the expected columns
    expected_columns = ["unit", "sex", "age", "region", "year", "value"]
    assert list(df.columns) == expected_columns

    # Verify it has data
    assert len(df) > 0

    # Verify the column mapping worked (country -> region, life_expectancy -> value)
    assert "country" not in df.columns
    assert "life_expectancy" not in df.columns
    assert "region" in df.columns
    assert "value" in df.columns


def test_load_data_with_tsv_loader():
    """Test load_data with explicit TSV loader"""
    fixture_path = FIXTURES_DIR / "eu_life_expectancy_raw.tsv"
    loader = TSVDataLoader()
    df = load_data(input_file=str(fixture_path), loader=loader)

    # Verify it's a DataFrame
    assert isinstance(df, pd.DataFrame)

    # Verify it has the expected columns
    expected_columns = ["unit", "sex", "age", "region", "year", "value"]
    assert list(df.columns) == expected_columns


def test_load_data_with_json_loader():
    """Test load_data with explicit JSON loader"""
    fixture_path = FIXTURES_DIR / "eu_life_expectancy_raw.json"
    loader = JSONDataLoader()
    df = load_data(input_file=str(fixture_path), loader=loader)

    # Verify it's a DataFrame
    assert isinstance(df, pd.DataFrame)

    # Verify it has the expected columns
    expected_columns = ["unit", "sex", "age", "region", "year", "value"]
    assert list(df.columns) == expected_columns


def test_clean_data_from_json():
    """Test clean_data works with JSON-loaded data"""
    fixture_path = FIXTURES_DIR / "eu_life_expectancy_raw.json"
    loader = JSONDataLoader()
    df = load_data(input_file=str(fixture_path), loader=loader)
    cleaned = clean_data(df, country=Region.PT)

    # Verify only PT data is present
    assert (cleaned["region"] == "PT").all()

    # Verify we have PT data
    assert len(cleaned) > 0

    # Verify data types
    assert cleaned["year"].dtype == "int64"
    assert cleaned["value"].dtype == "float64"


def test_json_pipeline_end_to_end():
    """End-to-end integration test using JSON data format"""
    # Load JSON data
    fixture_path = FIXTURES_DIR / "eu_life_expectancy_raw.json"
    loader = JSONDataLoader()
    raw_data = load_data(input_file=str(fixture_path), loader=loader)

    # Clean the data for Portugal
    cleaned_data = clean_data(raw_data, country=Region.PT)

    # Load expected output
    expected_path = FIXTURES_DIR / "pt_life_expectancy_expected_json.csv"
    expected_data = pd.read_csv(expected_path)

    # Compare with expected output
    pd.testing.assert_frame_equal(
        cleaned_data.reset_index(drop=True),
        expected_data.reset_index(drop=True)
    )


def test_json_pipeline_no_country_filter():
    """Test JSON pipeline without country filtering"""
    fixture_path = FIXTURES_DIR / "eu_life_expectancy_raw.json"
    loader = JSONDataLoader()
    raw_data = load_data(input_file=str(fixture_path), loader=loader)

    # Clean without country filter
    cleaned_data = clean_data(raw_data, country=None)

    # Verify we have data from multiple countries
    assert len(cleaned_data["region"].unique()) > 1

    # Verify AT, BE, BG, PT are all present
    regions = set(cleaned_data["region"].unique())
    assert "AT" in regions
    assert "BE" in regions
    assert "BG" in regions
    assert "PT" in regions

    # Verify data structure
    expected_columns = ["unit", "sex", "age", "region", "year", "value"]
    assert list(cleaned_data.columns) == expected_columns