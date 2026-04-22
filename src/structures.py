from typing import Optional, Union, List, Literal

import numpy as np
from pydantic import BaseModel, Field
import pandas as pd
import json
from dataclasses import dataclass, field
from typing import Dict, Set
import os


class ReferencingColumnPair(BaseModel):
    identifier_column: str = Field(
        ...,
        description='The column which uniquely identifies some entity represented by a row.'
    )
    reference_column: str = Field(
        ...,
        description='The `reference_column` specifies another entity that is related to the current entity through a specific relationship (e.g., peer, manager, parent, etc.).'
    )
    details: str = Field(
        ...,
        description='The details of the referencing relationship formed by these two columns.'
    )


class AllSelfReferenceColumns(BaseModel):
    column_pairs: Optional[List[ReferencingColumnPair]] = Field(
        default=None,
        description='A list of identified column groups that form valid referencing relationships. '
                    'If there are no such column groups, then just set it to None.'
    )
    explanation: Optional[str] = Field(
        None,
        description='The reasoning process of determining which column pairs form valid referencing relationships.'
    )


class RuleDetail(BaseModel):
    rule: str = Field(
        ...,
        description='A natural language expression of the candidate rule.'
    )

    explanation: str = Field(
        ...,
        description='A clear explanation of the logic and rationale behind the rule'
    )

    columns: Set[str] = Field(
        ...,
        description='A set of unique column names mentioned in this rule, '
                    'indicating which columns are referenced in this rule. '
                    'Note that this field should be specified regardless of possible specifications of columns '
                    'in the rule field.'
    )


class FDRuleDetail(RuleDetail):
    left_side: Optional[Union[str, List[str]]] = Field(
        ...,
        description=(
            "Only specify this field for (conditional) functional dependency checking rules. "
            "A functional dependency can be expressed as A → B (rows with the same A should have same B), "
            "where A (the determinant) and B (the dependent) are each a single column or a list of columns. "
            "Specify the determinant (A) in this `left_side` field."
        )
    )

    right_side: Optional[Union[str, List[str]]] = Field(
        ...,
        description=(
            "Only specify this field for (conditional) functional dependency checking rules. "
            "A functional dependency can be expressed as A → B, (rows with the same A should have same B),"
            "where A (the determinant) and B (the dependent) are each a single column or a list of columns. "
            "Specify the dependent (B) in this `right_side` field."
        )
    )


class TemporalRuleDetail(RuleDetail):
    date_column_1: str = Field(
        ...,
        description=(
            "The first date column in the temporal rule expression. "
            "Only specify this field for Temporal Order Validation Rules with the format: "
            "date_column_1 operator date_column_2, e.g., project_start_time < project_end_time."
        )
    )

    operator: Literal['<', '<=', '==', '>', '>='] = Field(
        ...,
        description=(
            "The comparison operator for the temporal rule. "
            "Must be one of: `<`, `<=`, `==`. "
            "Used in the format: date_column_1 operator date_column_2."
        )
    )

    date_column_2: str = Field(
        ...,
        description=(
            "The second date column in the temporal rule expression. "
            "Only specify this field for Temporal Order Validation Rules with the format: "
            "date_column_1 operator date_column_2, e.g., project_start_time < project_end_time."
        )
    )


class ReferenceBasedTemporalRuleDetail(RuleDetail):
    date_column1: str = Field(
        ...,
        description=(
            "The first date column in the rule with the format: "
            "<date_column1> of the referencing row should be <OPERATOR> <date_column2> of the referenced row "
            "with <identifier_column> equals to the referencing row's <reference_column>. "
            "This can be either the same as or different from `date_column2`."
        )
    )

    date_column2: str = Field(
        ...,
        description=(
            "The second date column in the rule with the format: "
            "<date_column1> of the referencing row should be <OPERATOR> <date_column2> of the referenced row "
            "with <identifier_column> equals to the referencing row's <reference_column>. "
            "This can be either the same as or different from `date_column1`."
        )
    )

    operator: Literal['<', '<=', '==', '>', '>='] = Field(
        ...,
        description=(
            "The comparison operator used in the rule. Must be one of: '<', '<=', '==', '>=', '>'."
        )
    )

    identifier_column: str = Field(
        ...,
        description=(
            "The identifier_column in the rule with the format:"
            "<date_column1> of the referencing row should be <OPERATOR> <date_column2> of the referenced row "
            "with <identifier_column> equals to the referencing row's <reference_column>. "
        )
    )

    reference_column: str = Field(
        ...,
        description=(
            "The reference_column in the rule with the format:"
            "<date_column1> of the referencing row should be <OPERATOR> <date_column2> of the referenced row "
            "with <identifier_column> equals to the referencing row's <reference_column>. "
        )
    )


class ReferenceBasedEqualityRuleDetail(RuleDetail):
    column1: str = Field(
        ...,
        description=(
            "The first column in the rule with the format: "
            "<column1> of the referencing row should be equal to <column2> of the referenced row "
            "with <identifier_column> equals to the referencing row's <reference_column>. "
            "This can be either the same as or different from `column2`."
        )
    )

    column2: str = Field(
        ...,
        description=(
            "The second date column in the rule with the format: "
            "<column1> of the referencing row should be equal to <column2> of the referenced row "
            "with <identifier_column> equals to the referencing row's <reference_column>. "
            "This can be either the same as or different from `column1`."
        )
    )

    identifier_column: str = Field(
        ...,
        description=(
            "The identifier_column in the rule with the format:"
            "<column1> of the referencing row should be equal to <column2> of the referenced row "
            "with <identifier_column> equals to the referencing row's <reference_column>. "
        )
    )

    reference_column: str = Field(
        ...,
        description=(
            "The reference_column in the rule with the format:"
            "<column1> of the referencing row should be equal to <column2> of the referenced row "
            "with <identifier_column> equals to the referencing row's <reference_column>. "
        )
    )


class UniqueKeyRuleDetail(RuleDetail):
    unique_key: Union[str, List[str]] = Field(
        ...,
        description=(
            "The unique key, which can be either a single column or a list of columns, "
            "that uniquely identifies each row. In other words, no two rows should have identical values for this key."
        )
    )


class MonotonicRuleDetail(RuleDetail):
    column_1: str = Field(
        ...,
        description=(
            "The base column that defines the order of comparison in the monotonic rule. "
            "For two rows, if row1's column_1 < row2's column_1, a trend is expected in column_2."
        )
    )

    operator_1: Literal['<', '>'] = Field(
        ...,
        description=(
            "The strict comparison operator for column_1. Must be '<' or '>'. "
            "Used in the format: row1's column_1 operator_1 row2's column_1."
        )
    )

    column_2: str = Field(
        ...,
        description=(
            "The dependent column that is expected to follow the monotonic trend defined by column_1."
        )
    )

    operator_2: Literal['<', '<=', '>', '>='] = Field(
        ...,
        description=(
            "The expected comparison operator for column_2. Must be one of: '<', '<=', '>', '>='."
        )
    )

@dataclass
class CandidateRule:
    rule_id: Union[int, str]
    rule_type: str
    rule: RuleDetail
    code: Optional[str] = field(default=None)
    # The execution result (execution_result) is represented as a dictionary with the following keys:
    #   - 'rule_id'
    #   - 'support'
    #   - 'confidence'
    #   - 'satisfactions'
    #   - 'violations'
    #   - 'total_violations'
    #   - 'total_satisfactions'
    # If the rule execution fails, the execution result will be None.
    execution_result: Optional[dict] = field(default=None)

    # indicate whether a rule is semantically meaningful or semantically fragile
    semantic_validity: Optional[bool] = field(default=None)
    condition: any = field(default=None)

    def __str__(self):
        return (
            f"CandidateRule(rule_id={self.rule_id}, "
            f"rule details: {self.rule}, execution_result=omitted)"
        )



@dataclass
class DataUnderstanding:
    """
    Represents the results of the data understanding process for a dataset.

    :param detailed_description: A textual summary of the dataset, including its structure and key insights.
    :type detailed_description: str

    :param categorical_columns: A dictionary describing categorical columns, where:
                                - The key is the column name.
                                - The value is a set of all possible values in that column.
    :type categorical_columns: Dict[str, Set]

    :param special_values: A dictionary that identifies special values for each column, where:
                           - The key is the column name.
                           - The value is a set of special values detected in that column.
    :type special_values: Dict[str, Set]

    :param referencing_column_pairs: A list of column pairs forming valid referencing relationships.
    :type referencing_column_pairs: Optional[List[ReferencingColumnPair]]

    :param samples: A dataframe that contains sample rows from the original dataset by an informative sampling process.
    :type samples: pd.DataFrame
    """
    detailed_description: str
    column_meanings: Dict[str, str]
    categorical_columns: Dict[str, Set]
    syntactic_columns: List[str]
    special_values: Dict[str, Set]
    column_groups: List[List[str]]
    referencing_column_pairs: Optional[List[ReferencingColumnPair]]
    samples: pd.DataFrame = field(default_factory=pd.DataFrame)

    class NumpyEncoder(json.JSONEncoder):
        """Custom JSON encoder to handle numpy types and Pydantic BaseModel instances."""

        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, BaseModel):
                return obj.model_dump()
            else:
                return super().default(obj)

    def save_to_cache(self, folder_path: str):
        """
        Saves the current instance of DataUnderstanding to a folder in a human-readable format (JSON and CSV).

        :param folder_path: The path to the folder where the files will be saved.
        :type folder_path: str
        """
        # Ensure the directory exists
        if not os.path.exists(folder_path):
            os.makedirs(folder_path, exist_ok=True)

        # Prepare JSON data (handle serialization issues for sets and other non-serializable objects)
        json_data = {
            "detailed_description": self.detailed_description,
            "column_meanings": self.column_meanings,
            "syntactic_columns": self.syntactic_columns,
            "categorical_columns": {k: list(v) for k, v in self.categorical_columns.items()},
            "special_values": {k: list(v) if v else None for k, v in self.special_values.items()},
            "column_groups": self.column_groups,
            "referencing_column_pairs": self.referencing_column_pairs
        }

        # Save JSON with custom encoder
        json_file_path = os.path.join(folder_path, "data_understanding.json")
        try:
            with open(json_file_path, "w") as json_file:
                json.dump(json_data, json_file, indent=4, cls=self.NumpyEncoder)
            print(f"JSON file saved to: {json_file_path}")
        except TypeError as e:
            print(f"Error saving JSON file: {e}")
            raise

        # Save the DataFrame as CSV
        csv_file_path = os.path.join(folder_path, "samples.csv")
        try:
            self.samples.to_csv(csv_file_path, index=False)
            print(f"CSV file saved to: {csv_file_path}")
        except Exception as e:
            print(f"Error saving CSV file: {e}")
            raise

        print(f"DataUnderstanding instance successfully saved to {folder_path}")

    @staticmethod
    def load_from_cache(folder_path: str):
        """
        Loads a DataUnderstanding instance from a folder containing JSON and CSV files.

        :param folder_path: The path to the folder containing the cache files.
        :type folder_path: str

        :return: A DataUnderstanding instance loaded from the cache files.
        :rtype: DataUnderstanding
        """
        # Define file paths
        json_file_path = os.path.join(folder_path, "data_understanding.json")
        csv_file_path = os.path.join(folder_path, "samples.csv")

        # Validate folder existence
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"Cache folder not found: {folder_path}")

        # Load JSON data with error handling
        try:
            with open(json_file_path, "r") as json_file:
                json_data = json.load(json_file)
        except FileNotFoundError:
            raise FileNotFoundError(f"JSON file not found: {json_file_path}")
        except json.JSONDecodeError:
            raise ValueError(f"Error decoding JSON file: {json_file_path}")

        column_meanings = json_data.get("column_meanings")

        # Convert lists back to sets (handling empty or None values safely)
        categorical_columns = {k: set(v) if isinstance(v, list) else set() for k, v in
                               json_data.get("categorical_columns", {}).items()}
        special_values = {k: set(v) if isinstance(v, list) else set() for k, v in
                          json_data.get("special_values", {}).items()}
        column_groups = json_data.get("column_groups")
        syntactic_columns = json_data.get("syntactic_columns")

        if json_data.get("referencing_column_pairs") is not None:
            referencing_column_pairs = AllSelfReferenceColumns.model_validate(json_data.get("referencing_column_pairs"))
        else:
            referencing_column_pairs = None

        # Load the DataFrame from CSV with error handling
        try:
            samples = pd.read_csv(csv_file_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"CSV file not found: {csv_file_path}")
        except pd.errors.EmptyDataError:
            samples = pd.DataFrame()  # Handle empty CSV case gracefully

        # Return the loaded instance
        return DataUnderstanding(
            detailed_description=json_data.get("detailed_description", ""),
            column_meanings=column_meanings,
            syntactic_columns=syntactic_columns,
            categorical_columns=categorical_columns,
            special_values=special_values,
            column_groups=column_groups,
            samples=samples,
            referencing_column_pairs=referencing_column_pairs
        )
