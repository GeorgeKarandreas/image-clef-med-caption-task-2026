"""File with shared helper functions"""

import csv
import random
import re
from pathlib import Path
import numpy as np
import pandas as pd
import torch


def set_seed(seed, deterministic=False):
    """Set random seeds and configure fast or deterministic cuDNN behavior."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = not deterministic


def is_csv_valid(annotations_csv_path, verbose=False):
    """
    Validate CSV columns for missing values, empty strings, spacing issues, 
    and (for CUIs) format and duplicate labels.

    Args:
        file_path (str): Path to the CSV file.
        verbose (bool): If it should print specifics issues
    Returns:
        bool: True if no issues. False if it has issues
    """
    df = pd.read_csv(annotations_csv_path)
    no_issues = True

    for col in df.columns:
        if verbose:
            print(f"\n===== Column: {col} =====")

        series = df[col]
        series_str = series.astype(str)

        nan_mask = series.isna()
        empty_mask = series_str.str.strip() == ""
        leading_trailing_mask = series_str != series_str.str.strip()

        if verbose:
            print("NaN count:", nan_mask.sum())
            print("Empty/Whitespace count:", empty_mask.sum())
            print("Leading/Trailing spaces:", leading_trailing_mask.sum())

        # Defaults
        invalid_cui_mask = pd.Series(False, index=df.index)
        double_semicolon_mask = pd.Series(False, index=df.index)
        trailing_semicolon_mask = pd.Series(False, index=df.index)
        empty_label_mask = pd.Series(False, index=df.index)
        separator_space_mask = pd.Series(False, index=df.index)
        dup_mask = pd.Series(False, index=df.index)
        split_mask = pd.Series(False, index=df.index)

        if col == "CUIs":
            separator_space_mask = series_str.str.contains(r"\s;|;\s", regex=True)

            pattern = re.compile(r"^C\d+(;C\d+)*$")
            invalid_cui_mask = ~series.fillna("").str.match(pattern)

            raw = series.fillna("").astype(str)

            double_semicolon_mask = raw.str.contains(r";;")
            trailing_semicolon_mask = raw.str.contains(r";\s*$")
            empty_label_mask = raw.apply(
                lambda x: any(l.strip() == "" for l in x.split(";"))
            )
            dup_mask = raw.apply(
                lambda x: len(x.split(";")) != len(set(x.split(";"))) if x else False
            )

            if verbose:
                print("Spaces around ';':", separator_space_mask.sum())
                print("Invalid CUI format:", invalid_cui_mask.sum())
                print("Rows with ';;':", double_semicolon_mask.sum())
                print("Rows with trailing ';':", trailing_semicolon_mask.sum())
                print("Rows with empty labels:", empty_label_mask.sum())
                print("Duplicate labels in row:", dup_mask.sum())

        if col == "ID":
            split_mask = ~series_str.str.contains(r"_(?:train|valid|test)_", regex=True)
            if verbose:
                print("Invalid split naming in ID:", split_mask.sum())

        issues_mask = (
            nan_mask |
            empty_mask |
            leading_trailing_mask |
            separator_space_mask |
            split_mask |
            invalid_cui_mask |
            double_semicolon_mask |
            trailing_semicolon_mask |
            empty_label_mask |
            dup_mask
        )

        if issues_mask.any():
            no_issues = False
            if verbose:
                print("\nExample problematic rows:")
                print(df.loc[issues_mask, col].head(5).to_string(index=False))
        elif verbose:
            print("No issues found.")

    return no_issues


def append_run_tracking_csv(csv_path, row):
    """Append a run-tracking row to a CSV, expanding the schema if new columns appear."""
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    new_fieldnames = list(row.keys())

    if not csv_path.exists():
        with csv_path.open("w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=new_fieldnames, lineterminator="\n")
            writer.writeheader()
            writer.writerow(row)
        return

    with csv_path.open("r", newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        existing_fieldnames = reader.fieldnames or []
        existing_rows = list(reader)

    merged_fieldnames = list(existing_fieldnames)
    for fieldname in new_fieldnames:
        if fieldname not in merged_fieldnames:
            merged_fieldnames.append(fieldname)

    if merged_fieldnames != existing_fieldnames:
        with csv_path.open("w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=merged_fieldnames, lineterminator="\n")
            writer.writeheader()
            for existing_row in existing_rows:
                writer.writerow(existing_row)
            writer.writerow(row)
        return

    with csv_path.open("a", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=merged_fieldnames, lineterminator="\n")
        writer.writerow(row)
