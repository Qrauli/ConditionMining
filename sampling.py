from collections import defaultdict
import json
from typing import Optional, List
import logging

import numpy as np
import pandas as pd
import random

from ruleforge.utils.log import setup_logging

sample_logger = setup_logging(log_directory='logs/sampling',
                              logger_name='sampling', level=logging.INFO)


def sample(dataset: pd.DataFrame, k: int,
           informative: bool = True,
           categorical_columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Samples `k` rows from the given dataset. By default, it performs an informative sampling.
    It performs random sampling when the `informative` parameter is set to False.

    :param dataset: The input dataset to sample from. Must be a Pandas DataFrame.
    :type dataset: pd.DataFrame
    :param k: The number of rows to sample. Must be less than or equal to the number of rows in the dataset.
    :type k: int
    :param informative: If True, applies an informative sampling method (not yet implemented).
                        If False, performs random sampling.
    :type informative: bool
    :param categorical_columns: List of columns considered categorical.
    :type categorical_columns:  Optional[List[str]]
    :return: A DataFrame containing the sampled rows.
    :rtype: pd.DataFrame
    :raises ValueError: If `k` is greater than the number of rows in the dataset.

    :raises NotImplementedError: If `informative` is set to True, as the informative sampling process is not implemented.
    """
    #if k > len(dataset):
     #   raise ValueError(f"The number of rows to sample, k={k}, cannot exceed the dataset size of {len(dataset)}.")

    if informative:
        if categorical_columns is None:
            raise ValueError('For informative sampling, the categorical_columns cannot be None. '
                             'If there are no categorical columns, just specify an empty list.')
        return _informative_sample(dataset, k, categorical_columns)

    sampled_data = dataset.sample(n=min(k, len(dataset)))
    return sampled_data


def _informative_sample(df, n, categorical_columns):
    """
    Generates an informative sampled DataFrame of size n.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    n (int): The desired number of rows in the sampled DataFrame.
    categorical_columns (list): List of columns considered categorical.

    Returns:
    pd.DataFrame: An informative sampled DataFrame with n rows.

    Sampling Method:
    - **Categorical Columns:**
      - For each unique value in the specified categorical columns, the function samples at least one row where the column has that value.
      - This ensures all possible values of categorical columns are represented in the sampled DataFrame.
    - **Non-Categorical String Columns:**
      - Randomly samples rows that have non-null string values in these columns.
      - This includes various string values to represent the diversity in the data.
    - **Numerical Columns:**
      - Numerical columns are binned into quartiles using pandas' `qcut` function.
      - Samples are drawn from each bin to cover different ranges of numerical values.
      - This approach ensures that the sampled data spans the entire range of numerical values.
    - **Optional Columns (Columns with Missing Values):**
      - For each optional column, the function includes both rows where the value is missing (null) and where it is present (not null).
      - This ensures that the sampled DataFrame captures the presence of missing data.

    The function collects indices of rows that satisfy the above conditions and combines them.
    If the number of collected indices exceeds n, it randomly selects n indices.
    If it's less than n, it randomly samples additional rows to reach n rows.
    """
    if df.empty:
        return df

    # Identify columns
    all_columns = df.columns.tolist()
    non_categorical_columns = [col for col in all_columns if col not in categorical_columns]

    # Identify numerical and string columns among non-categorical columns
    numerical_columns = df[non_categorical_columns].select_dtypes(include=np.number).columns.tolist()
    string_columns = df[non_categorical_columns].select_dtypes(include='object').columns.tolist()

    # Identify optional columns (columns with missing values)
    optional_columns = df.columns[df.isnull().any()].tolist()

    # Set to collect indices
    indices_set = set()

    # Collect indices covering all unique values of categorical columns
    for col in categorical_columns:
        unique_values = df[col].dropna().unique()
        for val in unique_values:
            matching_rows = df[df[col] == val]
            if not matching_rows.empty:
                idx = matching_rows.sample(1).index[0]
                indices_set.add(idx)

    # Include both missing and non-missing values for optional columns
    for col in optional_columns:
        not_null_rows = df[df[col].notnull()]
        null_rows = df[df[col].isnull()]
        if not not_null_rows.empty:
            idx = not_null_rows.sample(1).index[0]
            indices_set.add(idx)
        if not null_rows.empty:
            idx = null_rows.sample(1).index[0]
            indices_set.add(idx)

    # Sample from different ranges for numerical columns
    for col in numerical_columns:
        df[col + '_bin'] = pd.qcut(df[col], q=4, duplicates='drop')
        bins = df[col + '_bin'].unique()
        for b in bins:
            bin_rows = df[df[col + '_bin'] == b]
            if not bin_rows.empty:
                idx = bin_rows.sample(1).index[0]
                indices_set.add(idx)
        df.drop(columns=[col + '_bin'], inplace=True)

    # Randomly sample from string columns
    for col in string_columns:
        valid_rows = df[df[col].notnull()]
        if not valid_rows.empty:
            idx = valid_rows.sample(1).index[0]
            indices_set.add(idx)

    # Convert indices set to list
    indices_list = list(indices_set)

    # Adjust the number of indices to match n
    if len(indices_list) > n:
        indices_list = np.random.choice(indices_list, n, replace=False)
    elif len(indices_list) < n:
        additional_needed = n - len(indices_list)
        remaining_indices = df.index.difference(indices_list)
        if len(remaining_indices) > 0:
            additional_indices = np.random.choice(
                remaining_indices, min(additional_needed, len(remaining_indices)), replace=False
            )
            indices_list = np.concatenate([indices_list, additional_indices])
        else:
            extra_indices = np.random.choice(indices_list, additional_needed, replace=True)
            indices_list = np.concatenate([indices_list, extra_indices])

    # Create the sampled DataFrame
    sample_df = df.loc[indices_list].drop_duplicates()

    return sample_df

def hierarchical_informative_sample(dictionary, k, df, categorical_columns, cn,
                                    force_full_output_if_small: bool = False):
    """
    Perform hierarchical informative sampling on the given dictionary of multi-row rule representations.

    The dictionary has the format:
      {
          group_key:
              a list of dictionaries
              or
              a single dictionary
      }

    Each sub-dictionary (repre_dict) represents a satisfaction or violation. In each repre_dict,
    the keys (strings or tuples) represent roles, and the values are either a list of row indexes
    or a single row index. In our sampling procedure, each repre_dict (together with its group_key)
    is considered one unit.

    If `force_full_output_if_small` is True and the total number of units is less than or equal to `k`,
    sampling is skipped and all units are returned after replacing indices with full row records.

    Otherwise, the hierarchical sampling proceeds in rounds to satisfy the following requirements:
      1. Categorical Columns: Ensure at least `cn` units for each unique value.
      2. Numerical Columns: Ensure at least one unit from each quartile bin.
      3. Optional Columns: Ensure at least one unit with a missing value and one with a non-missing value.

    After satisfying these requirements, the sample size is adjusted to `k` by adding or removing units randomly.
    Finally, the output dictionary is rebuilt with the same grouping structure as the input,
    but the row indexes in each repre_dict are replaced by the corresponding full row records (as dictionaries).

    Parameters:
      dictionary (dict): The input dictionary.
      k (int): The maximum (or desired) number of units (repre_dicts) to sample.
      df (pd.DataFrame): The dataframe containing the data.
      categorical_columns (list): List of columns considered categorical.
      cn (int): Minimum number of units to sample for each distinct categorical value.
      force_full_output_if_small (bool): If True and the number of input units <= k,
                                         skips sampling and returns all units. Defaults to False.

    Returns:
      dict: A sample with the same structure as the input dictionary, but with row indexes in
            each repre_dict replaced by full row records (dicts).
    """
    # --- Step 0: Optimization Pre-processing ---
    # Collect all row indices referenced in the input dictionary first
    # to avoid converting the entire dataframe if the dictionary only uses a subset.
    referenced_indices = set()
    
    # Flatten structure for iteration
    flat_units = []
    for group_key, reps in dictionary.items():
        if isinstance(reps, dict):
            reps = [reps]
        for rep in reps:
            unit_rows = set()
            for role, value in rep.items():
                if isinstance(value, list):
                    unit_rows.update(value)
                else:
                    unit_rows.add(value)
            referenced_indices.update(unit_rows)
            flat_units.append({
                'group_key': group_key,
                'rep': rep,
                'rows': list(unit_rows)
            })

    # Convert relevant DF rows to a dictionary for O(1) access. 
    # df.loc[].to_dict('index') is faster than repeated df.loc accesses.
    # We include columns needed for sampling logic + final output.
    
    # Identify columns
    df_num_cols = df.select_dtypes(include=np.number).columns.tolist()
    optional_columns = df.columns[df.isnull().any()].tolist()
    
    # Create the row cache
    # Note: If referenced_indices is huge, this might consume memory, but it's much faster.
    # If referenced_indices covers most of df, just converting df is fine.
    try:
        relevant_df = df.loc[list(referenced_indices)]
    except KeyError:
        # Fallback if some indices in dictionary don't exist in df
        valid_indices = [idx for idx in referenced_indices if idx in df.index]
        relevant_df = df.loc[valid_indices]
    
    # Determine which columns we actually need for logic
    cols_needed = set(categorical_columns) | set(df_num_cols) | set(optional_columns)
    
    # We need all data for the output, so we cache the whole row
    row_cache = relevant_df.to_dict('index')

    # --- Step 1: Build Candidate Units & Inverted Indexes ---
    candidate_units = []
    
    # Inverted indexes to find units matching criteria quickly: 
    # Structure: index[col][val] = [list_of_candidate_indices]
    cat_index = defaultdict(lambda: defaultdict(list))
    opt_index = defaultdict(lambda: defaultdict(list))
    num_values_map = defaultdict(list) # store (unit_idx, value) for qcut later

    for i, unit_struct in enumerate(flat_units):
        # Check validity
        if not unit_struct['rows']: 
            continue
            
        # Aggregate unit statistics using python loops (faster than creating mini-dfs)
        cat_values = {}
        num_values = {}
        opt_info = {}
        
        # Get data for rows in this unit
        rows_data = [row_cache[rid] for rid in unit_struct['rows'] if rid in row_cache]
        if not rows_data:
            continue

        # 1. Categorical
        for col in categorical_columns:
            vals = set()
            for r in rows_data:
                val = r.get(col)
                if pd.notna(val):
                    vals.add(val)
            cat_values[col] = vals
            # Populate inverted index
            for v in vals:
                cat_index[col][v].append(i)

        # 2. Numerical (Mean)
        for col in df_num_cols:
            vals = [r.get(col) for r in rows_data if pd.notna(r.get(col))]
            if vals:
                mean_val = sum(vals) / len(vals)
                num_values[col] = mean_val
                num_values_map[col].append(mean_val)
            else:
                num_values[col] = None
                num_values_map[col].append(None) # Ensure alignment for qcut? No, just skip.

        # 3. Optional
        for col in optional_columns:
            has_missing = any(pd.isna(r.get(col)) for r in rows_data)
            has_non_missing = any(pd.notna(r.get(col)) for r in rows_data)
            opt_info[col] = {'missing': has_missing, 'non_missing': has_non_missing}
            
            if has_missing:
                opt_index[col]['missing'].append(i)
            if has_non_missing:
                opt_index[col]['non_missing'].append(i)

        unit_struct['categorical'] = cat_values
        unit_struct['numerical'] = num_values
        unit_struct['optional'] = opt_info
        unit_struct['original_index'] = i
        candidate_units.append(unit_struct)

    total_units = len(candidate_units)
    if total_units == 0:
        return {}

    # --- Conditional Exit ---
    if force_full_output_if_small and total_units <= k:
        sample_logger.info(f"Input has {total_units} units <= k={k}. Returning all.")
        final_selection_indices = range(total_units)
    else:
        # --- Step 2: Pre-calculate Bin Assignments for Numerical ---
        # Doing qcut once globally is faster
        num_bin_index = defaultdict(lambda: defaultdict(list))
        
        for col in df_num_cols:
            # Extract values for valid units
            valid_items = [(u['original_index'], u['numerical'][col]) 
                           for u in candidate_units if u['numerical'][col] is not None]
            
            if not valid_items:
                continue
                
            indices, values = zip(*valid_items)
            try:
                # Calculate bins
                bins = pd.qcut(values, q=4, duplicates='drop', labels=False)
                # Map back
                for idx, bin_label in zip(indices, bins):
                    num_bin_index[col][bin_label].append(idx)
            except ValueError:
                # Happens if not enough unique values
                pass

        # --- Step 3: Hierarchical Sampling Logic ---
        selected_indices_set = set()
        
        # Helper to check current coverage
        # We maintain counts to avoid recalculating over selected_indices_set
        # current_cat_counts[col][val] = count
        current_cat_counts = defaultdict(lambda: defaultdict(int))
        
        def add_unit(u_idx):
            if u_idx in selected_indices_set:
                return
            selected_indices_set.add(u_idx)
            # Update counters
            u = candidate_units[u_idx] # This indexing relies on list order matching iteration 0..N
            for col, vals in u['categorical'].items():
                for v in vals:
                    current_cat_counts[col][v] += 1

        # -- Round 1: Categorical --
        for col in categorical_columns:
            # Get all unique values available in candidates for this column
            available_vals = cat_index[col].keys()
            
            for val in available_vals:
                current_count = current_cat_counts[col][val]
                if current_count < cn:
                    needed = cn - current_count
                    # Find candidates that have this value and aren't selected yet
                    possible_indices = cat_index[col][val]
                    available = [idx for idx in possible_indices if idx not in selected_indices_set]
                    
                    if len(available) < needed:
                         sample_logger.warning(f"Not enough units for {col}={val}. Needed {needed}, got {len(available)}")
                    
                    # Random selection
                    if len(available) > needed:
                        picked = random.sample(available, needed)
                    else:
                        picked = available
                    
                    for idx in picked:
                        add_unit(idx)

        # -- Round 2: Numerical --
        for col in df_num_cols:
            bins = num_bin_index[col].keys()
            for b in bins:
                # Check if we already have a unit in this bin
                possible_indices = num_bin_index[col][b]
                # Check intersection size
                # Optimization: iterate possible and break if 1 found
                found = False
                for idx in possible_indices:
                    if idx in selected_indices_set:
                        found = True
                        break
                
                if not found:
                    # Pick one
                    available = [idx for idx in possible_indices if idx not in selected_indices_set]
                    if available:
                        add_unit(random.choice(available))

        # -- Round 3: Optional --
        for col in optional_columns:
            for status in ['missing', 'non_missing']:
                possible_indices = opt_index[col][status]
                
                found = False
                for idx in possible_indices:
                    if idx in selected_indices_set:
                        found = True
                        break
                
                if not found:
                    available = [idx for idx in possible_indices if idx not in selected_indices_set]
                    if available:
                        add_unit(random.choice(available))

        # -- Final Step: Adjust to k --
        current_count = len(selected_indices_set)
        if current_count < k:
            needed = k - current_count
            all_indices = set(range(len(candidate_units)))
            remaining = list(all_indices - selected_indices_set)
            
            if len(remaining) >= needed:
                picked = random.sample(remaining, needed)
                for idx in picked:
                    add_unit(idx)
            else:
                # Add all remaining
                for idx in remaining:
                    add_unit(idx)
                # Random sampling with replacement to fill gap
                needed_after_rem = k - len(selected_indices_set)
                # We can't add duplicates to a set, so we handle the final output construction differently
                # If we need duplicates, we handle it in the final list construction
                # For now, let's just select unique indices to the set
                pass 
                
        elif current_count > k:
            # Prune randomly
            selected_indices_set = set(random.sample(list(selected_indices_set), k))

        final_selection_indices = list(selected_indices_set)
        
        # NOTE: We intentionally do NOT sample with replacement.
        # If fewer units exist than k, we just use all available unique units.
        # Sampling with replacement would cause memory blowup and is counterintuitive
        # for a "sampling" operation (the output could be larger than the input).

    # --- Step 4: Rebuild Output (Fast) ---
    output = {}
    
    # Using dictionary lookup row_cache instead of df.loc
    for unit_idx in final_selection_indices:
        unit = candidate_units[unit_idx]
        group_key = unit['group_key']
        rep = unit['rep']
        
        new_rep = {}
        for role, value in rep.items():
            # value is either an index or list of indices
            if isinstance(value, list):
                # Fetch row dicts
                rows = [row_cache[rid] for rid in value if rid in row_cache]
                if rows:
                    new_rep[role] = rows
            else:
                if value in row_cache:
                    new_rep[role] = row_cache[value]
                    
        if new_rep:
            if group_key in output:
                if isinstance(output[group_key], list):
                    output[group_key].append(new_rep)
                else:
                    # Convert existing single dict to list
                    output[group_key] = [output[group_key], new_rep]
            else:
                output[group_key] = new_rep

    return output


def hierarchical_informative_sample_rows(
    dictionary: dict,
    k: int,
    df: pd.DataFrame,
    categorical_columns: List[str],
    cn: int = 1,
    min_rows_per_role: int = 1,
    force_full_output_if_small: bool = False
) -> dict:
    """
    Perform budget-aware hierarchical informative sampling on multi-row rule representations.
    
    This function samples rows from a hierarchical structure while:
    1. Respecting a total row budget k
    2. Ensuring coverage of categorical values and numerical ranges
    3. Maintaining structural integrity (at least min_rows_per_role per role)
    
    The sampling strategy is budget-aware from the start:
    - First allocates a per-unit budget based on k / num_units
    - Uses informative sampling within each unit to maximize diversity
    - Prioritizes rows that cover under-represented categorical values
    
    Parameters:
      dictionary (dict): The input dictionary of multi-row representations.
      k (int): Maximum total number of rows to sample.
      df (pd.DataFrame): The dataframe containing the data.
      categorical_columns (list): List of columns considered categorical.
      cn (int): Target coverage per categorical value (soft constraint, within budget).
      min_rows_per_role (int): Minimum rows to keep per role per unit (default 1).
      force_full_output_if_small (bool): If True and total rows <= k, returns all rows.
    
    Returns:
      dict: A sample with the same structure as the input dictionary, but with row indexes
            replaced by full row records (dicts). Total rows will not exceed k.
    """
    if not dictionary:
        return {}
    
    # --- Step 0: Analyze structure and calculate budgets ---
    # Build unit info: each (group_key, rep_idx) is a unit
    units = []  # List of {'group_key', 'rep_idx', 'roles': {role: [df_indices]}}
    total_rows = 0
    
    for group_key, reps in dictionary.items():
        rep_list = reps if isinstance(reps, list) else [reps]
        for rep_idx, rep in enumerate(rep_list):
            unit = {
                'group_key': group_key,
                'rep_idx': rep_idx,
                'roles': {}
            }
            for role, value in rep.items():
                row_indices = value if isinstance(value, list) else [value]
                unit['roles'][role] = row_indices
                total_rows += len(row_indices)
            units.append(unit)
    
    if not units:
        return {}
    
    num_units = len(units)
    sample_logger.info(f"Budget-aware sampling: {total_rows} total rows across {num_units} units, budget k={k}")
    
    # --- Early exit if small ---
    if force_full_output_if_small and total_rows <= k:
        sample_logger.info(f"Total rows {total_rows} <= k={k}. Returning all rows.")
        # Build row cache and return everything
        all_df_indices = set()
        for unit in units:
            for role, indices in unit['roles'].items():
                all_df_indices.update(indices)
        
        try:
            relevant_df = df.loc[list(all_df_indices)]
        except KeyError:
            valid_indices = [idx for idx in all_df_indices if idx in df.index]
            relevant_df = df.loc[valid_indices]
        row_cache = relevant_df.to_dict('index')
        
        # Rebuild output with full data
        output = {}
        for unit in units:
            group_key = unit['group_key']
            new_rep = {}
            for role, indices in unit['roles'].items():
                rows = [row_cache[idx] for idx in indices if idx in row_cache]
                if len(rows) == 1:
                    new_rep[role] = rows[0]
                elif rows:
                    new_rep[role] = rows
            
            if new_rep:
                if group_key in output:
                    if isinstance(output[group_key], list):
                        output[group_key].append(new_rep)
                    else:
                        output[group_key] = [output[group_key], new_rep]
                else:
                    output[group_key] = new_rep
        return output
    
    # --- Step 1: Calculate per-unit budget ---
    # Calculate actual row counts and minimum required per unit
    rows_per_unit = []
    min_rows_per_unit = []
    for unit in units:
        unit_rows = sum(len(indices) for indices in unit['roles'].values())
        rows_per_unit.append(unit_rows)
        
        num_roles = len(unit['roles'])
        min_required = num_roles * min_rows_per_role
        min_rows_per_unit.append(min_required)
    
    # Allocate budget proportionally to each unit's original size
    # Each unit gets: k * (unit_rows / total_rows), but at least min_rows_per_unit
    budget_per_unit = []
    for unit_idx, unit in enumerate(units):
        unit_rows = rows_per_unit[unit_idx]
        min_required = min_rows_per_unit[unit_idx]
        
        if total_rows > 0:
            # Proportional allocation based on original size
            proportional_budget = int(k * unit_rows / total_rows)
            # At least min_required, but not more than available rows
            allocated = max(min_required, proportional_budget)
            allocated = min(allocated, unit_rows)  # Can't sample more than exists
        else:
            allocated = min_required
        
        budget_per_unit.append(allocated)
    
    # If total allocated exceeds k, scale down proportionally
    total_allocated = sum(budget_per_unit)
    if total_allocated > k:
        scale_factor = k / total_allocated
        budget_per_unit = [max(1, int(b * scale_factor)) for b in budget_per_unit]
        sample_logger.info(f"Scaled down budgets by {scale_factor:.2f} to fit k={k}")
    
    # --- Step 2: Create row cache ---
    all_df_indices = set()
    for unit in units:
        for role, indices in unit['roles'].items():
            all_df_indices.update(indices)
    
    try:
        relevant_df = df.loc[list(all_df_indices)]
    except KeyError:
        valid_indices = [idx for idx in all_df_indices if idx in df.index]
        relevant_df = df.loc[valid_indices]
    
    row_cache = relevant_df.to_dict('index')
    
    # Identify column types for informative sampling
    df_num_cols = df.select_dtypes(include=np.number).columns.tolist()
    
    # --- Step 3: Track global categorical coverage ---
    # We'll prioritize sampling rows that increase coverage
    global_cat_counts = defaultdict(lambda: defaultdict(int))
    
    # --- Step 4: Sample from each unit within budget ---
    selected_rows = []  # List of (group_key, rep_idx, role, df_idx)
    
    for unit_idx, unit in enumerate(units):
        unit_budget = budget_per_unit[unit_idx]
        group_key = unit['group_key']
        rep_idx = unit['rep_idx']
        roles = unit['roles']
        num_roles = len(roles)
        
        if num_roles == 0:
            continue
        
        # Allocate budget across roles proportionally
        role_budgets = {}
        total_role_rows = sum(len(indices) for indices in roles.values())
        
        for role, indices in roles.items():
            if total_role_rows > 0:
                # Proportional allocation based on role size, but at least 1
                role_budget = max(1, int(unit_budget * len(indices) / total_role_rows))
            else:
                role_budget = 1
            role_budgets[role] = min(role_budget, len(indices))  # Can't exceed available
        
        # Ensure we don't exceed unit budget
        while sum(role_budgets.values()) > unit_budget:
            # Reduce largest budget by 1
            max_role = max(role_budgets, key=lambda r: role_budgets[r])
            if role_budgets[max_role] > 1:
                role_budgets[max_role] -= 1
            else:
                break
        
        # Sample from each role using informative selection
        for role, indices in roles.items():
            role_budget = role_budgets[role]
            if role_budget <= 0:
                continue
            
            if len(indices) <= role_budget:
                # Take all
                for df_idx in indices:
                    selected_rows.append((group_key, rep_idx, role, df_idx))
                    # Update coverage
                    if df_idx in row_cache:
                        for col in categorical_columns:
                            val = row_cache[df_idx].get(col)
                            if pd.notna(val):
                                global_cat_counts[col][val] += 1
            else:
                # Need to subsample - prioritize coverage
                # Score each row by how much it improves coverage
                scored_indices = []
                for df_idx in indices:
                    if df_idx not in row_cache:
                        continue
                    row_data = row_cache[df_idx]
                    
                    # Coverage score: sum of 1/(count+1) for each categorical value
                    # Lower count = higher priority
                    coverage_score = 0
                    for col in categorical_columns:
                        val = row_data.get(col)
                        if pd.notna(val):
                            current_count = global_cat_counts[col][val]
                            coverage_score += 1.0 / (current_count + 1)
                    
                    scored_indices.append((df_idx, coverage_score))
                
                # Sort by coverage score (descending) and take top role_budget
                scored_indices.sort(key=lambda x: x[1], reverse=True)
                
                for df_idx, _ in scored_indices[:role_budget]:
                    selected_rows.append((group_key, rep_idx, role, df_idx))
                    # Update coverage
                    if df_idx in row_cache:
                        for col in categorical_columns:
                            val = row_cache[df_idx].get(col)
                            if pd.notna(val):
                                global_cat_counts[col][val] += 1
    
    # --- Step 5: Final budget enforcement ---
    if len(selected_rows) > k:
        sample_logger.info(f"Selected {len(selected_rows)} rows, trimming to k={k} using coverage-aware selection")
        # Score rows by their coverage contribution and keep most valuable
        # But we need to ensure we keep at least 1 row per unit
        
        # Group by unit
        unit_rows = defaultdict(list)
        for row_info in selected_rows:
            group_key, rep_idx, role, df_idx = row_info
            unit_rows[(group_key, rep_idx)].append(row_info)
        
        # Keep at least 1 row per unit, then fill remaining with best coverage
        final_selected = []
        guaranteed_per_unit = []
        
        for unit_key, rows in unit_rows.items():
            # Keep first row as guaranteed
            guaranteed_per_unit.append(rows[0])
        
        # Remaining budget after guaranteeing 1 per unit
        remaining_budget = k - len(guaranteed_per_unit)
        
        if remaining_budget > 0:
            # Pool all other rows
            other_rows = []
            for unit_key, rows in unit_rows.items():
                other_rows.extend(rows[1:])  # Skip first (already guaranteed)
            
            if len(other_rows) <= remaining_budget:
                final_selected = guaranteed_per_unit + other_rows
            else:
                # Score by coverage
                scored_other = []
                for row_info in other_rows:
                    group_key, rep_idx, role, df_idx = row_info
                    if df_idx not in row_cache:
                        continue
                    row_data = row_cache[df_idx]
                    coverage_score = 0
                    for col in categorical_columns:
                        val = row_data.get(col)
                        if pd.notna(val):
                            current_count = global_cat_counts[col][val]
                            coverage_score += 1.0 / (current_count + 1)
                    scored_other.append((row_info, coverage_score))
                
                scored_other.sort(key=lambda x: x[1], reverse=True)
                final_selected = guaranteed_per_unit + [r[0] for r in scored_other[:remaining_budget]]
        else:
            # Can only keep guaranteed rows, may need to trim
            final_selected = guaranteed_per_unit[:k]
        
        selected_rows = final_selected
    
    sample_logger.info(f"Budget-aware sampling complete: {len(selected_rows)} rows selected")
    
    # --- Step 6: Rebuild output structure ---
    # Ensure we have row_cache (might not exist if we went through early budget enforcement path)
    if 'row_cache' not in dir() or row_cache is None:
        all_df_indices = set(row_info[3] for row_info in selected_rows)  # df_idx is at index 3
        try:
            relevant_df = df.loc[list(all_df_indices)]
        except KeyError:
            valid_indices = [idx for idx in all_df_indices if idx in df.index]
            relevant_df = df.loc[valid_indices]
        row_cache = relevant_df.to_dict('index')
    
    # Group selected rows by (group_key, rep_idx, role)
    grouped_selection = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    
    for row_info in selected_rows:
        gk, ri, role, df_idx = row_info  # Tuple unpacking
        
        if df_idx in row_cache:
            grouped_selection[gk][ri][role].append(row_cache[df_idx])
    
    # Build output in original structure
    output = {}
    
    for group_key, rep_dict in grouped_selection.items():
        if len(rep_dict) == 1 and 0 in rep_dict:
            # Single rep_dict case
            roles_data = rep_dict[0]
            new_rep = {}
            for role, rows in roles_data.items():
                if len(rows) == 1:
                    new_rep[role] = rows[0]
                else:
                    new_rep[role] = rows
            output[group_key] = new_rep
        else:
            # Multiple rep_dicts
            rep_list = []
            for rep_idx in sorted(rep_dict.keys()):
                roles_data = rep_dict[rep_idx]
                new_rep = {}
                for role, rows in roles_data.items():
                    if len(rows) == 1:
                        new_rep[role] = rows[0]
                    else:
                        new_rep[role] = rows
                if new_rep:
                    rep_list.append(new_rep)
            
            if len(rep_list) == 1:
                output[group_key] = rep_list[0]
            elif rep_list:
                output[group_key] = rep_list
    
    return output



if __name__ == '__main__':
    file_path = '/Users/jinsonng/Documents/companyworkspace/Chat2Data-git/RuleForge/core/MonotonicityCheck/Tax.csv'
    dataset = pd.read_csv(file_path)
    categorical_columns = [
        'Gender',
        'MaritalStatus',
        'HasChild',
        'Sate'
    ]
    with open(
        '/Users/jinsonng/Documents/companyworkspace/Chat2Data-git/RuleForge/core/RuleForge/input_dict.json',
        mode='r'
    ) as f:
        vio = json.load(f)

    vio_samples = hierarchical_informative_sample(
        dictionary=vio,
        k=5000,
        df=dataset,
        categorical_columns=categorical_columns,
        cn=50,
    )

    print(vio_samples)
