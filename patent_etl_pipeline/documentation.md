# Patent ETL Pipeline Documentation

## Project Overview

The `patent_etl_pipeline` project is designed to extract, transform, and load (ETL) biological activity data from various Excel files provided in patent applications into a structured SQL database. The goal is to standardize the data from different sources and make it easily accessible for analysis.

## File Structure

The project directory is organized as follows:
```
patent_etl_pipeline/
├── database/
│   └── patent_data.db
├── data_processor.py
├── parsers.py
├── run_etl.py
└── etl.log
```

## Database Schema

The SQLite database (`patent_data.db`) has the following tables:

- **compounds**: Stores unique chemical compounds.
  - `compound_id` (INTEGER PRIMARY KEY AUTOINCREMENT): Unique identifier for each compound.
  - `smiles` (TEXT UNIQUE NOT NULL): The SMILES string representing the compound's structure.

- **targets**: Stores unique biological targets.
  - `target_id` (INTEGER PRIMARY KEY AUTOINCREMENT): Unique identifier for each target.
  - `target_name` (TEXT UNIQUE NOT NULL): The name of the biological target (e.g., KU-19-19, EGFR).

- **activities**: Stores the activity data linking compounds to targets.
  - `activity_id` (INTEGER PRIMARY KEY AUTOINCREMENT): Unique identifier for each activity measurement.
  - `compound_id` (INTEGER): Foreign key referencing the `compounds` table.
  - `target_id` (INTEGER): Foreign key referencing the `targets` table.
  - `ic50` (REAL): The IC50 value in μM.
  - `pic50` (REAL): The calculated pIC50 value.
  - `activity_category` (TEXT): The categorized activity level (e.g., 'Highly Active', 'Moderately Active').

## Scripts

- **`data_processor.py`**: Contains the core logic for processing individual Excel sheets, extracting and transforming IC50 data and calculating pIC50 and activity categories. This function is intended to be used by the parser scripts.

- **`parsers.py`**: Contains specific parser functions for each of the four patent Excel files. These functions utilize `data_processor.py` to extract and process relevant data from their respective file formats and return dictionaries of processed DataFrames.

- **`run_etl.py`**: The main script that orchestrates the ETL pipeline. It connects to the database, iterates through the defined Excel files, calls the appropriate parser function for each file, and loads the processed data into the database tables (`compounds`, `targets`, and `activities`). It includes error handling and logging.

## Usage

1.  Place the four patent Excel files in the `/content/` directory (or update the `excel_files_dir` variable in `run_etl.py`).
2.  Ensure you have the necessary libraries installed:
    ```bash
    pip install pandas openpyxl numpy
    ```
3.  Run the ETL pipeline script:
    ```bash
    python patent_etl_pipeline/run_etl.py
    ```
The script will process each file, load the data into the SQLite database (`patent_data.db` in the `database` directory), and log its progress and any errors to `etl.log`.

## Error Handling and Logging

The `run_etl.py` script includes logging to track the ETL process. Information, warnings, and errors are logged to the console and to the `etl.log` file within the `patent_etl_pipeline` directory. This helps in monitoring the pipeline's execution and debugging issues.

Specific error handling is implemented for file not found errors, database errors, and errors during row processing.

## Extensibility

To add support for new Excel files:

1.  Create a new parser function in `parsers.py` similar to `parse_file_1` to handle the structure of the new file.
2.  Add the new filename to the `excel_files_to_process` list in `run_etl.py`.
3.  Add a mapping from the new filename to the new parser function in the `parser_mapping` dictionary in `run_etl.py`.

If the new file contains different types of data or requires a different processing logic, you might need to update `data_processor.py` or create new processing functions.
