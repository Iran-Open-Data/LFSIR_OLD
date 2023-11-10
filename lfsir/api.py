"""HBSIR - Iran Household Budget Survey Data API.

This module provides an API for loading, processing, and analyzing
Iran's Household Budget Survey (HBSIR) data.

The key functions provided are:

- load_table: Load HBSIR data for a given table name and year range.

- add_classification: Add commodity/occupation classification codes.

- add_attribute: Add household attributes like urban/rural status. 

- add_weight: Add sampling weights.

- add_cpi: Join CPI index data.

- select: Filter table by urban/rural, province etc. 

- adjust_by_cpi: Adjust monetary values for inflation using CPI.

- adjust_by_equivalence_scale: Adjust by household composition.

- create_table_with_schema: Create table from schema.

- setup: Download, extract and process survey data.

- setup_config: Copy default config files.

- setup_metadata: Copy default metadata.

See the API docstrings for more details.

"""
# pylint: disable=too-many-arguments
# pylint: disable=unused-argument
# pylint: disable=too-many-locals

from typing import Iterable, Literal, overload
import shutil

import pandas as pd

from .core import archive_handler, data_cleaner, data_engine, decoder, metadata_reader
from . import utils
from .core.metadata_reader import (
    metadata,
    defaults,
    original_tables,
    _Attribute,
    _OriginalTable,
    _StandardTable,
    # _Province,
    _Table,
    _Years,
)


def _extract_parameters(local_variables: dict) -> dict:
    return {key: value for key, value in local_variables.items() if value is not None}


@overload
def load_table(
    table_name: _OriginalTable = "data",
    years: _Years = "last",
    form: Literal["processed", "cleaned", "raw"] | None = None,
    *,
    on_missing: Literal["error", "download", "create"] | None = None,
    redownload: bool | None = None,
    save_downloaded: bool | None = None,
    recreate: bool | None = None,
    save_created: bool | None = None,
) -> pd.DataFrame:
    ...


@overload
def load_table(
    table_name: _StandardTable,
    years: _Years = "last",
    form: Literal["processed"] | None = None,
    *,
    on_missing: Literal["error", "download", "create"] | None = None,
    redownload: bool | None = None,
    save_downloaded: bool | None = None,
    recreate: bool | None = None,
    save_created: bool | None = None,
) -> pd.DataFrame:
    ...


def load_table(
    table_name: _Table = "data",
    years: _Years = "last",
    form: Literal["processed", "cleaned", "raw"] | None = None,
    *,
    on_missing: Literal["error", "download", "create"] | None = None,
    redownload: bool | None = None,
    save_downloaded: bool | None = None,
    recreate: bool | None = None,
    save_created: bool | None = None,
) -> pd.DataFrame:
    """Load a table for the given table name and year(s).

    This function loads original and standard tables.
    Original tables are survey tables and available in three types:
    original, cleaned and processed.

    - The 'raw' dataset contains the raw data, identical to the
    survey data, without any modifications.
    - The 'cleaned' dataset contains the raw data with added column
    labels, data types, and removal of irrelevant values, but no
    changes to actual data values.
    - The 'processed' dataset applies operations like adding columns,
    calculating durations, and standardizing tables across years.

    Standard tables are defined in this package to facilitate
    working with the data and are only available in processed form.

    For more information about available tables check the
    [tables wiki page](https://github.com/Iran-Open-Data/HBSIR/wiki/Tables).

    Parameters
    ----------
    table_name : str
        Name of the table to load.
    years : _Years, default "last"
        Year or list of years to load data for.
    dataset : str, default "processed"
        Which dataset to load from - 'processed', 'cleaned', or 'original'.
    on_missing : str, default "download"
        Action if data is missing - 'error', 'download', or 'create'
    recreate : bool, default False
        Whether to recreate the data instead of loading it
    redownload : bool, default False
        Whether to re-download the data instead of loading it
    save_downloaded : bool, default True
        Whether to save downloaded data
    save_created : bool, default True
        Whether to save newly created data

    Returns
    -------
    DataFrame
        Loaded data for the specified table and years.

    Examples
    --------
    >>> import hbsir
    >>> df = hbsir.load_table('food')
    # Loads processed 'food' table from original survey tables for
        latest available year
    >>> df = hbsir.load_table('Expenditures', '1399-1401')
    # Loads standard 'Expenditures' table for years 1399 - 1401

    Raises
    ------
    FileNotFoundError
        If data is missing and on_missing='error'.

    """
    metadata.reload_file("schema")
    parameters = _extract_parameters(locals())
    settings = data_engine.LoadTableSettings(**parameters)
    if settings.form == "raw":
        if table_name not in original_tables:
            raise ValueError
        years = utils.parse_years(years)
        table_parts = []
        for year in years:
            table_parts.append(data_cleaner.load_raw_data(year=year))
        table = pd.concat(table_parts)
    elif settings.form == "cleaned":
        if table_name not in original_tables:
            raise ValueError
        years = utils.parse_years(years)
        table_parts = []
        for year in years:
            table_parts.append(
                data_engine.TableHandler([table_name], year, settings)[table_name]
            )
        table = pd.concat(table_parts)
    else:
        table = data_engine.create_table(
            table_name=table_name,
            years=years,
            settings=settings,
        )
    return table


def add_classification(
    table: pd.DataFrame,
    name: str = "original",
    classification_type: Literal["commodity", "industry", "occupation"] | None = None,
    *,
    aspects: Iterable[str] | None = None,
    levels: Iterable[int] | int | None = None,
    column_names: Iterable[str] | str | None = None,
    drop_value: bool | None = None,
    missing_value_replacements: dict[str, str] | None = None,
    code_col: str | None = None,
    year_col: str | None = None,
) -> pd.DataFrame:
    """Add classification columns to table.

    Classifies codes in the table using specified classification system.

    Supported systems:

    - 'commodity': Classifies commodity codes
    - 'industry': Classifies industry codes
    - 'occupation': Classifies occupation codes

    Parameters
    ----------
    table : DataFrame
        Table containing code column to classify.
    name : str, default 'original'
        Name of classification to apply.
    classification_type : str, optional
        Type of classification system.
    aspects : list, optional
        Aspects of classification to add as columns.
    levels : list of int, optional
        Number of digits for each classification level.
    column_names: list of str, optional
        Names of output columns.
    drop_value : bool, optional
        Whether to drop unclassified values.
    missing_value_replacements : dict, optional
        Replacements for missing values in columns.
    code_col: str, optional
        Name of the code column.
    year_col: str, optional
        Name of the year column.

    Returns
    -------
    DataFrame
        Table with added classification columns.

    """
    parameters = _extract_parameters(locals())
    if "classification_type" not in parameters:
        if "code_column_name" in parameters:
            if table[parameters["code_column_name"]].le(10_000).mean() < 0.9:
                class_type = "occupation"
            else:
                class_type = "commodity"
        elif metadata_reader.defaults.columns.commodity_code in table.columns:
            class_type = "commodity"
        elif metadata_reader.defaults.columns.job_code in table.columns:
            class_type = "occupation"
        else:
            raise ValueError("Missing Code Column")
        parameters["classification_type"] = class_type
    settings = decoder.DecoderSettings(**parameters)
    table = decoder.Decoder(table=table, settings=settings).add_classification()
    return table


def add_attribute(
    table: pd.DataFrame,
    name: _Attribute,
    *,
    aspects: Iterable[str] | str | None = None,
    column_names: Iterable[str] | str | None = None,
    id_col: str | None = None,
    year_col: str | None = None,
) -> pd.DataFrame:
    """Add household attributes to table based on ID.

    Takes a DataFrame with a household ID column and adds columns
    for the specified attribute such as urban/rural status or province.

    Supported attributes:

    - 'Urban_Rural': Urban or rural classification
    - 'Province': Province name
    - 'County': County name

    Parameters
    ----------
    table : DataFrame
        DataFrame containing ID column.
    name : str
        Name of attribute to add.
    aspects: list of str, optional
        Aspects of attribute to add as columns.
    column_names: list of str, optional
        Output column names.
    id_col: str, optional
        Name of ID column.
    year_col: str, optional
        Name of year column.

    Returns
    -------
    DataFrame
        Input DataFrame with added attribute columns.

    """
    parameters = _extract_parameters(locals())
    settings = decoder.IDDecoderSettings(**parameters)
    table = decoder.IDDecoder(table=table, settings=settings).add_attribute()
    return table


def setup(
    years: _Years = "last",
    method: Literal["create", "download"] = "create",
    table_names: _OriginalTable | Iterable[_OriginalTable] | Literal["all"] = "all",
    replace: bool = False,
) -> None:
    """Download, extract, and process HBSIR data.

    Handles downloading and extracting the HBSIR data files,
    saving them locally, and processing tables.

    This sets up the raw data for further analysis. It:

    - Downloads and extracts archive files using archive_handler.
    - Cleans tables and saves as Parquet using data_cleaner.

    Parameters
    ----------
    years : str/list/int, default "last"
        Year(s) to download and process.
    method : str, default "create"
        "create" to process raw tables or
        "download" to download processed tables.
    table_names : str/list, default "all"
        Tables to process. "all" for all tables.
    replace : bool, default False
        Whether to overwrite existing files.

    Examples
    --------
    setup(1399)
        Process data for 1399.

    setup([1390,1400], table_names=['food'])
        Process food table for 1390 and 1400.

    setup('1380-1390', method='download')
        Download processed tables for 1380-1390.

    """
    if method == "create":
        archive_handler.setup(years, replace)
        data_cleaner.save_cleaned_tables(table_names, years)
    else:
        utils.download_processed_data(table_names, years)


def setup_config(replace=False) -> None:
    """Copy default config file to data directory.

    Copies the default config file 'settings-sample.yaml' from
    the package directory to 'settings.yaml' in the root data
    directory.

    Overwrites any existing file if replace=True.

    Parameters
    ----------
    replace : bool, default False
        Whether to overwrite existing config file.

    """
    src = defaults.package_dir.joinpath("config", "settings-sample.yaml")
    dst = defaults.root_dir.joinpath("settings.yaml")
    if (not dst.exists()) or replace:
        shutil.copy(src, dst)


def setup_metadata(replace=False) -> None:
    """Copy default metadata files to data directory.

    Copies metadata files like schema.csv and info.csv from the
    package metadata folder to the root data/metadata folder.

    Overwrites any existing files if replace=True.

    Parameters
    ----------
    replace : bool, default False
        Whether to overwrite existing metadata files.

    """
    src_folder = defaults.package_dir.joinpath("metadata")
    dst_folder = defaults.root_dir.joinpath("metadata")
    if not dst_folder.exists():
        dst_folder.mkdir()
    for src in src_folder.iterdir():
        dst = dst_folder.joinpath(src.name)
        if (not dst.exists()) or replace:
            shutil.copy(src, dst)
