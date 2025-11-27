"""
DataValidator class for validating dataset integrity.
"""

import pandas as pd
from typing import List
from dataclasses import dataclass
from typing import Optional


@dataclass
class ValidationIssue:
    """Represents a single validation issue."""
    row: Optional[int]
    column: str
    issue_type: str  # 'missing', 'invalid_type', 'out_of_range'
    message: str


@dataclass
class ValidationResult:
    """Result of dataset validation."""
    is_valid: bool
    issues: List[ValidationIssue]


class DataValidator:
    """Validates dataset integrity and reports issues."""
    
    def validate_columns(self, df: pd.DataFrame, required_columns: List[str]) -> ValidationResult:
        """
        Validate that all required columns are present.
        
        Args:
            df: DataFrame to validate
            required_columns: List of required column names
            
        Returns:
            ValidationResult with pass/fail status and issues
        """
        issues = []
        
        # Check for missing columns
        missing_columns = set(required_columns) - set(df.columns)
        
        for col in missing_columns:
            issues.append(ValidationIssue(
                row=None,
                column=col,
                issue_type='missing',
                message=f"Required column '{col}' is missing from dataset"
            ))
        
        is_valid = len(issues) == 0
        return ValidationResult(is_valid=is_valid, issues=issues)
    
    def validate_numeric_values(self, df: pd.DataFrame, columns: List[str]) -> ValidationResult:
        """
        Validate that specified columns contain numeric values.
        
        Args:
            df: DataFrame to validate
            columns: List of columns to check
            
        Returns:
            ValidationResult with pass/fail status and issues
        """
        issues = []
        
        for col in columns:
            if col not in df.columns:
                # Skip columns that don't exist (will be caught by validate_columns)
                continue
            
            # Check if column is numeric type
            if not pd.api.types.is_numeric_dtype(df[col]):
                # Try to identify specific non-numeric values
                non_numeric_mask = pd.to_numeric(df[col], errors='coerce').isna() & df[col].notna()
                non_numeric_rows = df.index[non_numeric_mask].tolist()
                
                if len(non_numeric_rows) > 0:
                    # Report first few non-numeric values
                    sample_rows = non_numeric_rows[:5]
                    for row_idx in sample_rows:
                        # Get the position in the dataframe
                        row_position = df.index.get_loc(row_idx)
                        issues.append(ValidationIssue(
                            row=row_position,
                            column=col,
                            issue_type='invalid_type',
                            message=f"Non-numeric value '{df.loc[row_idx, col]}' found in column '{col}'"
                        ))
                    
                    if len(non_numeric_rows) > 5:
                        issues.append(ValidationIssue(
                            row=None,
                            column=col,
                            issue_type='invalid_type',
                            message=f"Column '{col}' has {len(non_numeric_rows)} total non-numeric values"
                        ))
        
        is_valid = len(issues) == 0
        return ValidationResult(is_valid=is_valid, issues=issues)
    
    def detect_missing_data(self, df: pd.DataFrame) -> ValidationResult:
        """
        Detect missing data in the DataFrame.
        
        Args:
            df: DataFrame to check
            
        Returns:
            ValidationResult with missing data locations
        """
        issues = []
        
        # Check each column for missing values
        for col in df.columns:
            missing_mask = df[col].isna()
            missing_indices = df.index[missing_mask].tolist()
            
            if len(missing_indices) > 0:
                # Report first few missing values with specific row identifiers
                sample_indices = missing_indices[:5]
                for idx in sample_indices:
                    # Get the position in the dataframe
                    row_position = df.index.get_loc(idx)
                    issues.append(ValidationIssue(
                        row=row_position,
                        column=col,
                        issue_type='missing',
                        message=f"Missing value in column '{col}' at index {idx}"
                    ))
                
                # If there are many missing values, add a summary
                if len(missing_indices) > 5:
                    issues.append(ValidationIssue(
                        row=None,
                        column=col,
                        issue_type='missing',
                        message=f"Column '{col}' has {len(missing_indices)} total missing values"
                    ))
        
        is_valid = len(issues) == 0
        return ValidationResult(is_valid=is_valid, issues=issues)
