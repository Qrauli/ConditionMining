from collections import defaultdict
from typing import Dict, Optional, List

import numpy as np
import pandas as pd


def sample(dataset: pd.DataFrame, k: int,
           informative: bool = True,
           categorical_columns: Optional[List[str]] = None,
           min_per_group: int = 1) -> pd.DataFrame:
    """Sample *k* rows from a DataFrame.

    By default the function uses **informative sampling**, which
    guarantees that every categorical value, numerical quartile, and
    missing-value pattern is represented at least once.  Set
    ``informative=False`` to fall back to uniform random sampling.

    Parameters
    ----------
    dataset : pd.DataFrame
        The input DataFrame to sample from.
    k : int
        Desired number of rows in the returned sample.  If *k* is
        greater than or equal to the number of rows in *dataset*, the
        full DataFrame is returned unchanged.
    informative : bool, default ``True``
        When ``True``, delegates to :func:`_informative_sample` which
        applies a stratified, coverage-aware strategy.  When ``False``,
        uses ``pd.DataFrame.sample`` for simple random sampling.
    categorical_columns : list of str or None, default ``None``
        Column names that should be treated as categorical during
        informative sampling.  Ignored when ``informative=False``.
        If ``None``, defaults to an empty list internally.
    min_per_group : int, default ``1``
        Minimum number of rows to retain for each unique value in
        categorical columns and for each numerical bin during
        informative sampling.  Ignored when ``informative=False``.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing exactly *k* rows (or fewer if the
        dataset itself is smaller than *k*).

    Notes
    -----
    * An empty DataFrame or ``k <= 0`` immediately returns an empty
      DataFrame with the same columns.
    * The returned DataFrame preserves the original index labels.

    Examples
    --------
    >>> df = pd.DataFrame({'cat': ['a','b','a','b'], 'val': [1,2,3,4]})
    >>> sample(df, k=2, informative=True, categorical_columns=['cat'])
    """
    if dataset.empty or k <= 0:
        return dataset.iloc[:0]
        
    if k >= len(dataset):
        return dataset

    if informative:
        if categorical_columns is None:
            # Default to empty list if None, rather than raising error, for better UX
            categorical_columns = []
        return _informative_sample(dataset, k, categorical_columns, min_per_group)

    sampled_data = dataset.sample(n=k)
    return sampled_data


def _informative_sample(df: pd.DataFrame, n: int, categorical_columns: List[str], min_per_group: int = 1) -> pd.DataFrame:
    """Build an informative sample of *n* rows from *df*.

    The function collects "must-have" row positions using four coverage
    strategies, then adjusts the final set to exactly *n* rows.

    Parameters
    ----------
    df : pd.DataFrame
        Source DataFrame (must contain at least one row).
    n : int
        Target sample size.
    categorical_columns : list of str
        Columns whose every unique value must appear in the sample.
    min_per_group : int, default ``1``
        Minimum rows to collect per unique value / bin.

    Returns
    -------
    pd.DataFrame
        A DataFrame with *n* rows selected from *df* (via ``iloc``).

    Notes
    -----
    The coverage strategies are applied in order:

    1. **Categorical columns** – for each unique value in every
       categorical column, at least ``min_per_group`` rows are selected.
    2. **Numerical columns** – each numerical column is binned into
       quartiles (``pd.qcut``), and at least ``min_per_group`` rows are
       drawn from each bin.
    3. **Optional columns** (any column with at least one ``NaN``) –
       both null and non-null rows are represented.
    4. **Non-categorical string columns** – at least ``min_per_group``
       rows with non-null string values are included.

    If the collected rows exceed *n*, a random subsample is taken.
    If fewer than *n* rows were collected, additional rows are drawn
    uniformly at random from the remaining pool.
    """
    # 1. Identify Column Types
    all_columns = df.columns.tolist()
    # Ensure categorical columns actually exist in df
    valid_cat_cols = [c for c in categorical_columns if c in df.columns]
    
    non_categorical_columns = [col for col in all_columns if col not in valid_cat_cols]
    
    # Identify numerical and string columns among non-categorical columns
    numerical_columns = df[non_categorical_columns].select_dtypes(include=np.number).columns.tolist()
    string_columns = df[non_categorical_columns].select_dtypes(include='object').columns.tolist()
    optional_columns = df.columns[df.isnull().any()].tolist()

    # We use a set of INTEGER POSITIONS (row numbers)
    indices_set = set()

    # Helper function to sample from indices list/array
    def sample_indices(indices_arr):
        count = len(indices_arr)
        if count <= min_per_group:
            return indices_arr
        return np.random.choice(indices_arr, min_per_group, replace=False)

    # 2. Categorical Columns Strategy
    # df.groupby().indices returns dict {value: array([int_positions])}
    # This aligns perfectly with iloc/position-based logic.
    for col in valid_cat_cols:
        groups = df.groupby(col, sort=False).indices
        for idx_array in groups.values():
            indices_set.update(sample_indices(idx_array))

    # 3. Numerical Columns Strategy (Binning)
    for col in numerical_columns:
        try:
            # Create bins. qcut is preferred for distribution, cut as fallback for low variance
            bins = pd.qcut(df[col], q=4, duplicates='drop')
            
            # Group by these bins
            groups = df.groupby(bins, observed=True, sort=False).indices
            for idx_array in groups.values():
                indices_set.update(sample_indices(idx_array))
        except ValueError:
            # Fallback if qcut fails
            pass

    # 4. Optional Columns (Missing vs Non-Missing)
    # Convert boolean masks to integer positions using np.flatnonzero
    for col in optional_columns:
        # Get underlying numpy array for speed
        col_array = df[col].to_numpy()
        
        # Positions where null
        null_indices = np.flatnonzero(pd.isna(col_array))
        # Positions where not null
        not_null_indices = np.flatnonzero(~pd.isna(col_array))
        
        if len(null_indices) > 0:
            indices_set.update(sample_indices(null_indices))
        if len(not_null_indices) > 0:
            indices_set.update(sample_indices(not_null_indices))

    # 5. String Columns (Non-Categorical)
    for col in string_columns:
        col_array = df[col].to_numpy()
        valid_indices = np.flatnonzero(~pd.isna(col_array))
        if len(valid_indices) > 0:
            indices_set.update(sample_indices(valid_indices))

    # 6. Final Adjustment to size N
    indices_list = np.array(list(indices_set), dtype=int)
    current_count = len(indices_list)

    if current_count > n:
        # Subsample if we have too many
        selected_indices = np.random.choice(indices_list, n, replace=False)
    elif current_count < n:
        # Fill remainder
        needed = n - current_count
        
        # Calculate remaining positions efficiently
        # Total rows is len(df)
        all_positions = np.arange(len(df))
        
        # setdiff1d returns sorted unique values in ar1 that are not in ar2
        remaining_indices = np.setdiff1d(all_positions, indices_list, assume_unique=True)
        
        if len(remaining_indices) >= needed:
            additional_indices = np.random.choice(remaining_indices, needed, replace=False)
            selected_indices = np.concatenate([indices_list, additional_indices])
        else:
            # Take all remaining and then sample with replacement (unlikely if k <= len(df))
            selected_indices = np.concatenate([indices_list, remaining_indices])
            # If we are somehow still short (e.g. k > len(df) passed despite check), handled implicitly by iloc bounds
    else:
        selected_indices = indices_list

    # Use iloc because we collected integer positions, not index labels
    return df.iloc[selected_indices]

def hierarchical_informative_sample_rows(
    dictionary: Dict,
    k: int,
    df: pd.DataFrame,
    categorical_columns: List[str],
    numerical_columns: Optional[List[str]] = None,
    n_bins: int = 10,
    cn: int = 1,
    min_rows_per_role: int = 1,
    force_full_output_if_small: bool = False
) -> Dict:
    """Budget-aware hierarchical sampling at the *row* level.

    Unlike :func:`hierarchical_informative_sample`, which treats each
    representation dict as an atomic unit, this function allocates a
    per-unit row budget and selects individual rows within each unit
    based on an *informativeness score*.  The score rewards rows whose
    categorical values or numerical-bin membership are rare in the
    candidate pool, ensuring diverse coverage.

    Parameters
    ----------
    dictionary : dict
        Input dictionary with the same structure accepted by
        :func:`hierarchical_informative_sample`::

            {
                group_key: [
                    {role: row_index_or_list, ...},
                    ...
                ]
            }

    k : int
        **Global row budget** – the maximum total number of individual
        rows to keep across all units and roles.
    df : pd.DataFrame
        Source DataFrame whose rows are referenced by the indices in
        *dictionary*.
    categorical_columns : list of str
        Columns treated as categorical for scoring purposes.
    numerical_columns : list of str or None, default ``None``
        Columns treated as numerical for scoring.  If ``None``, all
        numeric columns in *df* that are not listed in
        *categorical_columns* are used automatically.
    n_bins : int, default ``10``
        Number of equal-width bins (``pd.cut``) used when scoring
        numerical columns.  More bins yield finer-grained diversity.
    cn : int, default ``1``
        *Currently unused in the row-level variant* but kept for API
        symmetry with :func:`hierarchical_informative_sample`.
    min_rows_per_role : int, default ``1``
        Each role inside a unit is guaranteed at least this many rows
        (the highest-scoring ones), provided enough rows exist.
    force_full_output_if_small : bool, default ``False``
        When ``True`` and the total number of candidate rows across all
        units is ≤ *k*, skip scoring / pruning and return all data.

    Returns
    -------
    dict
        A dictionary with the same grouping structure as *dictionary*.
        Row indices are replaced by full row records (``dict`` of
        column → value).  Roles with a single row are stored as a plain
        dict; roles with multiple rows are stored as a list of dicts.

    Notes
    -----
    **Scoring**

    Each candidate row receives an *informativeness score* equal to the
    sum of inverse-frequency weights across categorical and binned
    numerical columns.  Rare values contribute more, biasing selection
    toward underrepresented regions of the feature space.

    **Budget allocation**

    1. A global keep-ratio ``k / total_candidate_rows`` is computed.
    2. Each unit receives a budget of
       ``max(num_roles × min_rows_per_role, unit_rows × keep_ratio)``.
    3. Within a unit, every role first gets its ``min_rows_per_role``
       highest-scoring rows; the remaining budget is filled by the
       next-highest-scoring rows across all roles.
    4. A final global hard-cut trims the combined selection to *k*.

    Examples
    --------
    >>> inp = {'g1': [{'buyer': [0, 1, 2], 'seller': [3, 4]}]}
    >>> out = hierarchical_informative_sample_rows(
    ...     inp, k=3, df=df,
    ...     categorical_columns=['region'],
    ...     min_rows_per_role=1)
    """
    if not dictionary:
        return {}

    # --- Step 1: Flatten Structure & Collect Indices ---
    candidates = []
    unit_stats = defaultdict(lambda: {'total': 0, 'roles': defaultdict(int)})
    
    # Collect all indices first to create a valid subset for scoring
    for group_key, reps in dictionary.items():
        rep_list = reps if isinstance(reps, list) else [reps]
        for rep_idx, rep in enumerate(rep_list):
            # Normalize bare row indices (int/float) into dict format
            if isinstance(rep, (int, float, np.integer)):
                rep = {'row': rep}
            unit_id = (group_key, rep_idx)
            for role, value in rep.items():
                row_indices = value if isinstance(value, list) else [value]
                
                unit_stats[unit_id]['total'] += len(row_indices)
                unit_stats[unit_id]['roles'][role] = len(row_indices)
                
                for idx in row_indices:
                    candidates.append((group_key, rep_idx, role, idx))

    total_rows = len(candidates)
    
    # --- Step 2: Early Exit ---
    if force_full_output_if_small and total_rows <= k:
        return _fetch_and_reconstruct(candidates, df)

    # --- Step 3: Vectorized "Informativeness" Scoring ---
    unique_indices = list(set(c[3] for c in candidates))
    
    # Identify numerical columns if not provided
    if numerical_columns is None:
        all_num = df.select_dtypes(include=np.number).columns.tolist()
        numerical_columns = [c for c in all_num if c not in categorical_columns]

    scoring_cols = list(set(categorical_columns + (numerical_columns or [])))
    scores = {} 
    
    if scoring_cols and unique_indices:
        # Load ONLY scoring columns for candidate rows
        try:
            subset = df.loc[unique_indices, scoring_cols]
        except KeyError:
            valid_indices = [idx for idx in unique_indices if idx in df.index]
            subset = df.loc[valid_indices, scoring_cols]

        # Initialize scores as float
        row_scores = pd.Series(0.0, index=subset.index)
        
        # 3a. Score Categorical Columns
        for col in categorical_columns:
            if col in subset.columns:
                vc = subset[col].value_counts(normalize=False)
                weights = 1.0 / vc
                
                # FIX: Explicitly cast to float to strip Categorical dtype constraints before fillna
                score_add = subset[col].map(weights).astype(float)
                row_scores += score_add.fillna(0.0)
        
        # 3b. Score Numerical Columns (Binning)
        if numerical_columns:
            for col in numerical_columns:
                if col in subset.columns:
                    col_data = subset[col]
                    if col_data.isna().all():
                        continue
                        
                    try:
                        # Bin the data
                        binned = pd.cut(col_data, bins=n_bins, duplicates='drop')
                        
                        # Calculate rarity of the BIN
                        vc = binned.value_counts(normalize=False)
                        weights = 1.0 / vc
                        
                        # FIX: Cast to float immediately. 
                        # Without this, 'score_add' remains a Categorical Series in some pandas versions,
                        # causing 'fillna(0)' to fail because 0 is not a valid Category/Interval.
                        score_add = binned.map(weights).astype(float)
                        
                        row_scores += score_add.fillna(0.0)
                    except ValueError:
                        continue

        scores = row_scores.to_dict()
    
    # --- Step 4: Budget Allocation & Selection ---
    if total_rows == 0:
        return {}
        
    global_keep_ratio = k / total_rows
    selected_candidates = []
    
    # Group candidates by unit
    candidates_by_unit = defaultdict(list)
    for c in candidates:
        candidates_by_unit[(c[0], c[1])].append(c)

    for unit_id, unit_cands in candidates_by_unit.items():
        stats = unit_stats[unit_id]
        unit_total = stats['total']
        
        # Unit Budget Calculation
        min_required = len(stats['roles']) * min_rows_per_role
        target_unit_budget = int(unit_total * global_keep_ratio)
        unit_budget = max(min_required, target_unit_budget)
        unit_budget = min(unit_budget, unit_total)
        
        # Distribute to Roles
        role_map = defaultdict(list)
        for c in unit_cands:
            role_map[c[2]].append(c)
            
        unit_selection = []
        remaining_roles = []
        
        for role, items in role_map.items():
            # Sort by Score (Descending) + Index Hash tie-breaker
            items.sort(key=lambda x: scores.get(x[3], 0) + (hash(x[3]) % 1e-5), reverse=True)
            
            # Guarantee Minimums
            take_count = min(len(items), min_rows_per_role)
            unit_selection.extend(items[:take_count])
            
            if len(items) > take_count:
                remaining_roles.extend(items[take_count:])
        
        # Fill remaining unit budget
        remaining_budget = unit_budget - len(unit_selection)
        if remaining_budget > 0 and remaining_roles:
            remaining_roles.sort(key=lambda x: scores.get(x[3], 0) + (hash(x[3]) % 1e-5), reverse=True)
            unit_selection.extend(remaining_roles[:remaining_budget])
            
        selected_candidates.extend(unit_selection)

    # --- Step 5: Global Hard Cut ---
    if len(selected_candidates) > k:
        selected_candidates.sort(key=lambda x: scores.get(x[3], 0), reverse=True)
        selected_candidates = selected_candidates[:k]

    # --- Step 6: Fetch Data & Reconstruct ---
    return _fetch_and_reconstruct(selected_candidates, df)

def _fetch_and_reconstruct(candidates: List[tuple], df: pd.DataFrame) -> Dict:
    """Resolve row indices to full records and rebuild the grouped dict.

    This is a shared helper used by
    :func:`hierarchical_informative_sample_rows` (and its early-exit
    path) to convert a flat list of ``(group_key, rep_index, role,
    df_index)`` tuples back into the nested dictionary structure
    expected by callers.

    Parameters
    ----------
    candidates : list of tuple
        Each element is a 4-tuple
        ``(group_key, rep_index, role, df_index)`` identifying one
        selected row.
    df : pd.DataFrame
        Source DataFrame used to look up the actual row data.

    Returns
    -------
    dict
        Nested dictionary mirroring the original input structure::

            {
                group_key: {
                    role: row_record_dict | [row_record_dict, ...],
                    ...
                }
            }

        When a group key has a single representation (``rep_index == 0``
        and no others), the value is a plain dict; otherwise it is a
        list of dicts.
    """
    if not candidates:
        return {}
    
    final_indices = list(set(c[3] for c in candidates))
    
    try:
        relevant_df = df.loc[final_indices]
    except KeyError:
        valid_indices = [idx for idx in final_indices if idx in df.index]
        relevant_df = df.loc[valid_indices]
        
    row_cache = relevant_df.to_dict('index')
    
    grouped = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    
    for gk, ri, role, df_idx in candidates:
        if df_idx in row_cache:
            grouped[gk][ri][role].append(row_cache[df_idx])
            
    output = {}
    for group_key, rep_dict in grouped.items():
        sorted_reps = []
        for rep_idx in sorted(rep_dict.keys()):
            role_data = rep_dict[rep_idx]
            new_rep = {}
            for role, rows in role_data.items():
                if len(rows) == 1:
                    new_rep[role] = rows[0]
                else:
                    new_rep[role] = rows
            sorted_reps.append(new_rep)
            
        if len(sorted_reps) == 1 and 0 in rep_dict:
            output[group_key] = sorted_reps[0]
        else:
            output[group_key] = sorted_reps
            
    return output
