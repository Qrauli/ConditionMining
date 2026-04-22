import concurrent.futures
import itertools

import pandas as pd
import csv
import numpy as np

from typing import List, Union, Set, Optional, Dict

import os
import json
import ast

import src.condition_suggestion as condition_suggestion_mod
import src.structures as structures_mod
from src.structures import CandidateRule

def _deserialize_condition(cond_dict):
    """
    Reconstructs condition objects from the dictionary format, handling recursion 
    for ConjunctiveCondition and set conversion for MergedColumnValueCondition.
    """
    if not cond_dict or not isinstance(cond_dict, dict):
        return None

    c_type = cond_dict.get("type")
    kwargs = cond_dict.get("kwargs", {})

    if not c_type:
        return None

    # Case 1: ConjunctiveCondition (Recursive)
    if c_type == "ConjunctiveCondition":
        raw_conditions = kwargs.get("conditions", [])
        parsed_conditions = []
        for rc in raw_conditions:
            pc = _deserialize_condition(rc)
            if pc:
                parsed_conditions.append(pc)
        
        return condition_suggestion_mod.ConjunctiveCondition(conditions=frozenset(parsed_conditions))

    # Case 2: MergedColumnValueCondition (List -> Frozenset)
    if c_type == "MergedColumnValueCondition":
        val_set = kwargs.get("value_set", [])
        kwargs["value_set"] = frozenset(val_set)
        return condition_suggestion_mod.MergedColumnValueCondition(**kwargs)

    # Case 3: Other Atomic Conditions (PairWise, ColumnValue, Numerical)
    cls = getattr(condition_suggestion_mod, c_type, None)
    if cls:
        try:
            return cls(**kwargs)
        except TypeError as e:
            return None
    
    return None

def _reconstruct_from_json(obj):
    """Recursively reconstructs an object loaded from JSON, converting stringified tuple keys back to tuples."""
    if isinstance(obj, dict):
        new_dict = {}
        for k, v in obj.items():
            new_key = k
            # Try to convert string keys that look like tuples back to tuples
            if isinstance(k, str) and k.startswith('(') and k.endswith(')'):
                try:
                    # Safely evaluate the string representation of the tuple
                    evaluated_key = ast.literal_eval(k)
                    if isinstance(evaluated_key, tuple):
                        new_key = evaluated_key
                except (ValueError, SyntaxError):
                    # Not a valid tuple string, keep it as a string
                    pass
            new_dict[new_key] = _reconstruct_from_json(v)
        return new_dict
    if isinstance(obj, list):
        return [_reconstruct_from_json(item) for item in obj]
    return obj


def cache_load_all_rules(dataset_name: str, rule_type: str, cache_dir: Optional[str] = None) -> Optional[Dict[int, CandidateRule]]:
    safe_name = str(rule_type).replace(' ', '_')
    if cache_dir:
        base = cache_dir
    else:
        base = f'./tmp_folder_for_{dataset_name}__cache'
    
    fp = os.path.join(base, safe_name)
    cache_fp = os.path.join(fp, 'candidate_rules_cache.json')
    if not os.path.exists(cache_fp):
        # Fallback to base folder if specific rule folder is not used
        cache_fp = os.path.join(base, 'candidate_rules_cache.json')
        if not os.path.exists(cache_fp):
            return None

    try:
        with open(cache_fp, mode='r', encoding='utf-8') as f:
            payload = json.load(f)
    except Exception:
        return None

    loaded = dict()
    for entry in payload:
        rid = entry.get('rule_id')
        rtype = entry.get('rule_type')
        model_name = entry.get('rule_model')
        rule_dict = entry.get('rule')
        
        # Reconstruct the condition object
        condition_dict = entry.get('condition')
        condition_obj = _deserialize_condition(condition_dict)

        # find model class
        model_cls = getattr(structures_mod, model_name, None) if model_name else None
        if model_cls is None:
            model_cls = structures_mod.RuleDetail
        try:
            # pydantic models expect sets for some fields; here lists will be accepted
            rule_obj = model_cls.model_validate(rule_dict) if rule_dict is not None else None
        except Exception:
            # fallback: attempt to create RuleDetail
            try:
                rule_obj = structures_mod.RuleDetail.model_validate(rule_dict) if rule_dict is not None else None
            except Exception:
                rule_obj = None

        execution_result = entry.get('execution_result')
        if execution_result:
            execution_result = _reconstruct_from_json(execution_result)

        cr = CandidateRule(
            rule_id=rid,
            rule_type=rtype,
            rule=rule_obj,
            code=entry.get('code'),
            execution_result=execution_result,
            semantic_validity=entry.get('semantic_validity'),
            condition=condition_obj  # Populate the condition field
        )
        loaded[rid] = cr
    return loaded


def load_csv_with_encodings(file_path, delimiter=",", dtype=None, encodings=None):
    """
    Tries to load a CSV file using a list of encodings, returns the DataFrame if successful.

    :param file_path: Path to the CSV file
    :param delimiter: CSV delimiter (default: ',')
    :param dtype: Optional dtype dictionary for pandas
    :param encodings: List of encodings to try
    :return: pandas.DataFrame
    """
    if encodings is None:
        encodings = [
            'utf-8-sig', 'utf-8', 'utf-16', 'utf-16-le', 'utf-16-be',
            'utf-32', 'iso-8859-1', 'cp1252', 'cp1250', 'macroman', 'latin1'
        ]

    last_exception = None
    for enc in encodings:
        try:
            df = pd.read_csv(file_path, delimiter=delimiter, dtype=dtype, encoding=enc)
            return df
        except (UnicodeDecodeError, pd.errors.ParserError) as e:
            last_exception = e

    raise ValueError(f"Failed to read CSV with all known encodings. Last error: {last_exception}")


def read_csv(file_path: str):
    """
    Reads a CSV file into a pandas DataFrame, automatically detecting the delimiter.

    :param file_path: Path to the CSV file to be read.
    :type file_path: str

    :return: A pandas DataFrame containing the parsed data.
    :rtype: pandas.DataFrame
    """
    try:
        delimiter = ','
        common_delimiters = [',', ';']
        with open(file_path, 'r', encoding='utf-8', errors='replace') as file:
            # Read a sample from the first 100 lines
            sample = ''.join(itertools.islice(file, 100))

            if not sample.strip():
                raise ValueError("File is empty or unreadable.")

            # Try CSV Sniffer
            sniffer_succeed = False
            sniffer = csv.Sniffer()
            try:
                detected_delimiter = sniffer.sniff(sample).delimiter
                if detected_delimiter in common_delimiters:
                    sniffer_succeed = True
                    delimiter = detected_delimiter
            except csv.Error:
                pass
            if not sniffer_succeed:
                # **Manual Delimiter Detection (Backup Method)**
                delimiter_counts = {delim: sample.count(delim) for delim in common_delimiters}
                best_delimiter = max(delimiter_counts, key=delimiter_counts.get)

                print(f"Detected delimiter manually: {repr(best_delimiter)}")
                delimiter = best_delimiter

        df = load_csv_with_encodings(file_path, delimiter=delimiter)

        str_cols = df.select_dtypes(include=['object', 'string']).columns
        df[str_cols] = df[str_cols].applymap(lambda x: np.nan if isinstance(x, str) and x.strip() == '' else x)

        return df

    except Exception as e:
        raise ValueError(f"Failed to read CSV file. Error: {e}")


def create_model_for_process(process_name, predefined_temperature=None):
    import os
    from langchain_openai import ChatOpenAI
    
    model_name = os.getenv("OPENAI_MODEL", "gpt-4o")
    temperature = predefined_temperature if predefined_temperature is not None else 0.0
    
    return ChatOpenAI(model_name=model_name, temperature=temperature)


from tenacity import retry, stop_after_attempt, wait_exponential


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=10, max=60)
)
def invoke_with_retries(structured_llm, prompt_content, logger=None, timeout_seconds=60):
    """
    Invokes structured_llm.invoke(prompt_content) with a timeout and retries upon exceptions.
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(structured_llm.invoke, prompt_content)
        try:
            raw_result = future.result(timeout=timeout_seconds)
            return raw_result

        except Exception as e:
            if logger:
                logger.error(f"Exception occurred during structured_llm invocation.")
                logger.exception(e)
            raise  # Retry on timeout


def df_to_list_of_dicts(df: pd.DataFrame) -> list:
    """
    Converts a Pandas DataFrame into a list of dictionaries.
    Each dictionary represents a row in the DataFrame.

    :param df: The Pandas DataFrame to convert.
    :return: A list of dictionaries, each containing row data.
    """
    # 'records' orientation returns a list of row-wise dictionaries
    return df.to_dict(orient='records')



def get_entities_based_on_identifiers(df: pd.DataFrame,
                                      identifiers_of_entities: Set[Union[int, set]],
                                      index_column_name=None) -> Union[pd.DataFrame, List[pd.DataFrame]]:
    """
    Retrieve entities from a DataFrame based on their identifiers. Entities can be single rows or pairs/groups of rows.

    The identifier of an entity is defined as:
    - For a single-row entity: The identifier is the value of the column specified by `index_column_name` for that row.
    - For a pair or group of rows: The identifier is a set containing the values of `index_column_name`
      for all rows included in the entity.

    :param df: The DataFrame from which entities are to be retrieved.
    :param identifiers_of_entities: A set of identifiers for the entities to retrieve. Each identifier can be:
        - A single value (integer) for single-row entities.
        - A set of values for pairs or groups of rows.
        **All identifiers in the list must belong to the same category (single-row or multi-row entities).**
    :param index_column_name: The name of the column in `df` that holds the row identifiers (default is None).
    :return:
        - A single DataFrame containing all rows if the identifiers correspond to single-row entities.
        - A list of DataFrames, where each DataFrame corresponds to one entity, if the identifiers correspond to pairs or groups of rows.
    """
    # Validate that the index_column_name exists in the DataFrame
    if index_column_name is not None and (index_column_name not in df.columns):
        raise ValueError(f"The specified index column '{index_column_name}' does not exist in the DataFrame.")

    if all(isinstance(identifier, (str, int)) for identifier in identifiers_of_entities):
        # Case 1: Single-row entities
        # Retrieve all rows where the 'ruleforge_id' column matches any of the identifiers
        if index_column_name is not None:
            result = df[df[index_column_name].isin(identifiers_of_entities)]
        else:
            result = df[df.index.isin(identifiers_of_entities)]
        return result

    elif all(isinstance(identifier, set) for identifier in identifiers_of_entities):
        # Case 2: Multi-row entities (pairs/groups)
        # Retrieve each group of rows based on the identifiers
        result = []
        for identifier in identifiers_of_entities:
            if index_column_name:
                entity = df[df[index_column_name].isin(identifier)]
            else:
                entity = df[df.index.isin(identifier)]
            if not entity.empty and len(entity) == len(identifier):  # Ensure all identifiers in the group match rows
                result.append(entity)
            else:
                raise ValueError(f"One or more identifiers in the group {identifier} do not match rows in the DataFrame.")
        return result

    else:
        raise ValueError("All identifiers in `identifiers_of_entities` must be of the same type: "
                         "either all single-row identifiers or all multi-row identifiers.")

