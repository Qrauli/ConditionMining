import collections
from typing import List, Dict, Optional, Any, Union
from collections import defaultdict
from itertools import combinations

from dataclasses import dataclass, field
import pandas as pd
import numpy as np

import copy


@dataclass(frozen=True)
class PairWiseCondition:
    """
    Represents a pairwise equality/inequality condition between two rows.
    
    Used in multi-row rules to express conditions like "two rows have the same
    value in column X" (for partitioning) or "two rows have different values
    in column X" (for exclusion).
    
    Attributes:
        column_name: The column to compare between two rows.
        operator: Either '=' (same value) or '!=' (different values).
    
    Example:
        >>> cond = PairWiseCondition('department_id', '=')
        >>> str(cond)
        'two rows have the same value in the "department_id" column'
    """
    column_name: str
    operator: str

    def __str__(self):
        if self.operator == '=':
            return (
                f'two rows have the same value in the "{self.column_name}" column'
            )
        else:
            assert self.operator == '!='
            return (
                f'two rows have different values in the "{self.column_name}" column'
            )


@dataclass(frozen=True)
class ColumnValueCondition:
    """
    Represents an atomic condition checking a column against a specific value.
    
    The most basic condition type, used to filter rows where a column equals
    or does not equal a particular value.
    
    Attributes:
        column_name: The column to check.
        operator: Either '=' (equals) or '!=' (not equals).
        value: The value to compare against. Can be any type including None for NULL checks.
    
    Example:
        >>> cond = ColumnValueCondition('status', '=', 'Active')
        >>> str(cond)
        'the "status" column has the value of: Active'
    """
    column_name: str
    operator: str
    value: Any

    def __str__(self):
        if self.operator == '=':
            return (
                f'the "{self.column_name}" column has the value of: {self.value}'
            )
        else:
            assert self.operator == '!='
            return (
                f'the "{self.column_name}" column does not have the value of: {self.value}'
            )

@dataclass(frozen=True)
class NumericalCondition:
    """
    Represents a threshold condition on a numerical column.
    
    Used to partition data based on numerical ranges, typically found via
    information gain optimization on the combined violating/satisfying samples.
    
    Attributes:
        column_name: The numerical column to check.
        operator: Either '<' (less than) or '>=' (greater than or equal to).
        value: The threshold value (typically a split point between unique values).
    
    Example:
        >>> cond = NumericalCondition('amount', '>=', 1000.0)
        >>> str(cond)
        'the "amount" column is greater than or equal to: 1000.0'
    """
    column_name: str
    operator: str  # Will be '<' or '>='
    value: float

    def __str__(self):
        op_str = "is less than" if self.operator == '<' else "is greater than or equal to"
        return f'the "{self.column_name}" column {op_str}: {self.value}'

@dataclass(frozen=True)
class MergedColumnValueCondition:
    """
    Represents a set-based condition (IN or NOT IN a set of values).
    
    Created by merging multiple atomic ColumnValueConditions on the same column.
    More compact than expressing multiple OR/AND conditions separately.
    
    Attributes:
        column_name: The column to check.
        value_set: Frozenset of values to check membership against.
        operator: Either '=' (IN set, logical OR) or '!=' (NOT IN set, logical AND).
    
    Example:
        >>> cond = MergedColumnValueCondition('status', frozenset(['Active', 'Pending']), '=')
        >>> str(cond)
        'the "status" column value is in the set: ['Active', 'Pending']'
    
    Note:
        The system automatically avoids redundant merged conditions by preferring
        the smaller representation (e.g., IN {A, B} vs NOT IN {C} for a 3-value domain).
    """
    column_name: str
    value_set: frozenset
    operator: str

    def __str__(self):
        # Use 'is in' for '=' and 'is not in' for '!='
        op_str = "is in the set" if self.operator == '=' else "is not in the set"
        # Convert frozenset to a sorted list for consistent, readable output
        value_str = sorted(list(self.value_set)) 
        return f'the "{self.column_name}" column value {op_str}: {value_str}'

# Type alias for conditions that can be used within conjunctions
AtomicCondition = Union[ColumnValueCondition, NumericalCondition, MergedColumnValueCondition]

@dataclass(frozen=True)
class ConjunctiveCondition:
    """
    Represents a conjunction (AND) of multiple atomic conditions.
    
    Created when combining multiple conditions that individually have moderate
    scores but together achieve higher precision. The conditions must be on
    different columns to be valid.
    
    Attributes:
        conditions: Frozenset of atomic conditions combined with AND logic.
            Using frozenset ensures order-independence and hashability.
    
    Example:
        >>> cond1 = ColumnValueCondition('status', '=', 'Active')
        >>> cond2 = NumericalCondition('amount', '>=', 100.0)
        >>> conj = ConjunctiveCondition(frozenset([cond1, cond2]))
        >>> str(conj)
        'the "amount" column is greater than or equal to: 100.0 AND the "status" column has the value of: Active'
    """
    # Use a frozenset to ensure the order doesn't matter and it's hashable
    conditions: frozenset[AtomicCondition] = field(default_factory=frozenset)

    def __str__(self):
        return " AND ".join(sorted([str(c) for c in self.conditions]))


import sys

class CompactMaskCache:
    """
    Memory-efficient cache for boolean masks using bit packing.
    
    Stores boolean masks as packed bits (8x memory savings) with LRU eviction
    to prevent out-of-memory crashes during condition evaluation. This is critical
    when evaluating thousands of conditions across large datasets.
    
    Attributes:
        cache: Dictionary mapping condition objects to packed numpy arrays.
        max_items: Maximum number of items before LRU eviction occurs.
        insertion_order: OrderedDict tracking access order for LRU eviction.
    
    Example:
        >>> cache = CompactMaskCache(max_items=1000)
        >>> cache.set(condition, boolean_array)
        >>> retrieved = cache.get(condition, original_length=len(boolean_array))
    """
    
    def __init__(self, max_items: int = 10000):
        """
        Initialize the cache with a maximum capacity.
        
        Args:
            max_items: Maximum number of masks to store before evicting oldest entries.
        """
        self.cache = {}
        self.max_items = max_items
        # Keep track of insertion order for simple LRU (Least Recently Used)
        self.insertion_order = collections.OrderedDict()
        # Store the original length for unpacking (all masks in one cache should have same length)
        self.stored_length: Optional[int] = None

    def get(self, key, original_length: int) -> Optional[np.ndarray]:
        """
        Retrieve and unpack a cached boolean mask.
        
        Args:
            key: The condition object used as cache key.
            original_length: The original length of the boolean array before packing.
                Needed because np.packbits pads to byte boundaries.
        
        Returns:
            The unpacked boolean numpy array, or None if not found in cache.
        """
        if key in self.cache:
            # Move to end (recently used)
            self.insertion_order.move_to_end(key)
            packed = self.cache[key]
            # Unpack and trim to original length
            unpacked = np.unpackbits(packed)[:original_length]
            return unpacked.astype(bool)
        return None

    def get_packed(self, key) -> Optional[np.ndarray]:
        """
        Retrieve the raw packed mask without unpacking.
        
        Useful for efficient bitwise operations on multiple masks.
        
        Args:
            key: The condition object used as cache key.
        
        Returns:
            The packed numpy array (uint8), or None if not found.
        """
        if key in self.cache:
            self.insertion_order.move_to_end(key)
            return self.cache[key]
        return None

    def set(self, key, mask_array: np.ndarray):
        """
        Pack and store a boolean mask in the cache.
        
        If the cache is at capacity, evicts the least recently used entry.
        
        Args:
            key: The condition object to use as cache key.
            mask_array: Boolean numpy array to pack and store.
        """
        # Evict if full
        if len(self.cache) >= self.max_items:
            # Remove first item (oldest)
            oldest_key, _ = self.insertion_order.popitem(last=False)
            del self.cache[oldest_key]
        
        # Store the original length on first insert
        if self.stored_length is None:
            self.stored_length = len(mask_array)
            
        packed = np.packbits(mask_array)
        self.cache[key] = packed
        self.insertion_order[key] = True

    def set_packed(self, key, packed_array: np.ndarray):
        """
        Store an already-packed mask in the cache.
        
        Useful when the result of bitwise operations is already packed.
        
        Args:
            key: The condition object to use as cache key.
            packed_array: Already packed numpy array (uint8).
        """
        # Evict if full
        if len(self.cache) >= self.max_items:
            oldest_key, _ = self.insertion_order.popitem(last=False)
            del self.cache[oldest_key]
            
        self.cache[key] = packed_array
        self.insertion_order[key] = True

    def __contains__(self, key):
        """Check if a condition is cached."""
        return key in self.cache
    
    def clear(self):
        """Remove all entries from the cache."""
        self.cache.clear()
        self.insertion_order.clear()
        self.stored_length = None

import numpy as np


def _calculate_purity_score(confidence: float, penalty: float, 
                            sat_matches: int, min_support: int = 10) -> float:
    """
    Calculates a condition's purity score based on Normalized Positive Likelihood Ratio.
    
    The purity score measures how well a condition discriminates between satisfying
    and violating samples. A high score means the condition captures many satisfying
    cases while avoiding violating ones.
    
    Formula: Score = Confidence / (Confidence + Penalty)
    
    Score Interpretation:
        - 0.0: Purely violating (Confidence=0, condition only matches violations)
        - 0.5: Neutral/random (Confidence=Penalty, no discrimination)
        - 1.0: Purely satisfying (Penalty=0, condition only matches satisfying samples)
    
    Args:
        confidence: Fraction of satisfying samples that match the condition.
        penalty: Fraction of violating samples that match the condition.
        sat_matches: Absolute count of satisfying samples matching the condition.
        min_support: Minimum required sat_matches for a non-zero score.
            Conditions with low support are unreliable and get score 0.
    
    Returns:
        Purity score in range [0, 1], or 0.0 if support is insufficient.
    """
    if sat_matches < min_support:
        return 0.0
    
    # Edge case: If both are 0 (should be caught by support, but for safety)
    if confidence + penalty == 0:
        return 0.0
        
    return confidence / (confidence + penalty)

def find_optimal_numerical_split(
    rows: Union[List[dict], pd.DataFrame],
    numerical_column: str,
    min_support: int = 10,
    max_splits: int = 20
) -> List[dict]:
    """
    Finds optimal split points for a numerical column using Information Gain.
    
    Uses a decision tree-style algorithm to find threshold values that best
    separate satisfying from violating samples. The algorithm:
    1. Sorts data by column value
    2. Identifies candidate split points (midpoints between unique values)
    3. Calculates Information Gain for each split using vectorized operations
    4. Returns top N splits with their statistics
    
    Args:
        rows: DataFrame or list of dicts containing the data.
            Must include '__is_violating' boolean column.
        numerical_column: Name of the numerical column to find splits for.
        min_support: Minimum samples required for score calculation.
        max_splits: Maximum number of split points to return.
    
    Returns:
        List of split dictionaries, each containing:
        - 'split_value': The threshold value
        - 'info_gain': Information gain achieved by this split
        - 'stats': Dict with 'lt' and 'gte' sub-dicts containing
            confidence, penalty, and purity score for each side
        - 'column_name': The column name (for reference)
    
    Note:
        Uses O(N log N) sorting plus O(K) gain calculations where K is
        the number of unique values. Fully vectorized for performance.
    """
    # 1. Prepare Data
    if isinstance(rows, list):
        df = pd.DataFrame(rows)
    else:
        df = rows

    if numerical_column not in df.columns or '__is_violating' not in df.columns:
        return []

    # Drop nulls for calculation
    data = df[[numerical_column, '__is_violating']].dropna()
    if len(data) < 2:
        return []

    # 2. Sort by feature value (O(N log N))
    data = data.sort_values(by=numerical_column)
    
    values = data[numerical_column].values
    labels = data['__is_violating'].values.astype(int)
    n_samples = len(labels)

    # 3. Identify candidate split points (where values change)
    unique_values, unique_indices = np.unique(values, return_index=True)
    if len(unique_values) < 2:
        return []
    
    # Indices in 'labels' where the value changes (skip the first one which is 0)
    split_indices = unique_indices[1:]
    
    # Candidate split values are midpoints
    split_values = (unique_values[:-1] + unique_values[1:]) / 2.0

    # 4. Vectorized Information Gain Calculation
    
    # Calculate cumulative sums of the labels (1 for violating, 0 for satisfying)
    label_cumsum = np.cumsum(labels)
    
    # Total counts
    total_violating = label_cumsum[-1]
    total_satisfying = n_samples - total_violating
    
    if total_violating == 0 or total_satisfying == 0:
        return [] # Pure node, no gain possible

    # Parent Entropy
    parent_p = total_violating / n_samples
    parent_entropy = -parent_p * np.log2(parent_p) - (1-parent_p) * np.log2(1-parent_p) if 0 < parent_p < 1 else 0

    # --- Arrays for Split Calculation ---
    n_left = split_indices
    # V_left: number of violating samples to the left
    v_left = label_cumsum[split_indices - 1] 
    
    # Probabilities of violating class in left/right children
    # We explicitly calculate n_right and v_right during iteration only for top candidates? 
    # No, vectorization is faster.
    n_right = n_samples - n_left
    v_right = total_violating - v_left
    
    p_v_left = v_left / n_left
    p_v_right = v_right / n_right
    
    # Calculate entropy for children (vectorized)
    entropy_left = np.zeros_like(p_v_left)
    mask_l = (p_v_left > 0) & (p_v_left < 1)
    entropy_left[mask_l] = -p_v_left[mask_l]*np.log2(p_v_left[mask_l]) - (1-p_v_left[mask_l])*np.log2(1-p_v_left[mask_l])
    
    entropy_right = np.zeros_like(p_v_right)
    mask_r = (p_v_right > 0) & (p_v_right < 1)
    entropy_right[mask_r] = -p_v_right[mask_r]*np.log2(p_v_right[mask_r]) - (1-p_v_right[mask_r])*np.log2(1-p_v_right[mask_r])

    # Weighted average child entropy
    weighted_child_entropy = (n_left / n_samples) * entropy_left + (n_right / n_samples) * entropy_right
    
    # Information Gain
    info_gain = parent_entropy - weighted_child_entropy
    
    # --- Find Top N Splits ---
    # Filter out non-positive gain
    positive_mask = info_gain > 0
    if not np.any(positive_mask):
        return []
    
    valid_indices = np.where(positive_mask)[0]
    valid_gains = info_gain[valid_indices]
    
    # Sort descending by gain
    sorted_idx_local = np.argsort(valid_gains)[::-1]
    
    # Take top N
    top_indices_local = sorted_idx_local[:max_splits]
    top_indices_global = valid_indices[top_indices_local]

    results = []

    for best_idx in top_indices_global:
        best_split_val = split_values[best_idx]
        best_gain = info_gain[best_idx]
        
        # 5. Calculate Stats for this split
        # Re-retrieve counts for the specific index
        
        # Stats for " < split_value "
        sat_in_left = n_left[best_idx] - v_left[best_idx]
        vio_in_left = v_left[best_idx]
        
        confidence_lt = sat_in_left / total_satisfying if total_satisfying > 0 else 0
        penalty_lt = vio_in_left / total_violating if total_violating > 0 else 0
        
        score_lt = _calculate_purity_score(confidence_lt, penalty_lt, sat_in_left, min_support)
        
        # Stats for " >= split_value " (Complement)
        sat_in_right = (total_satisfying - sat_in_left)
        confidence_gte = 1.0 - confidence_lt
        penalty_gte = 1.0 - penalty_lt
        
        score_gte = _calculate_purity_score(confidence_gte, penalty_gte, sat_in_right, min_support)

        best_split_stats = {
            'lt': {'confidence': confidence_lt, 'penalty': penalty_lt, 'score': score_lt},
            'gte': {'confidence': confidence_gte, 'penalty': penalty_gte, 'score': score_gte}
        }

        results.append({
            "split_value": best_split_val,
            "info_gain": best_gain,
            "stats": best_split_stats,
            "column_name": numerical_column
        })

    return results
    

def _filter_redundant_conditions(results: List[dict], keep_top_k: int = 1) -> List[dict]:
    """
    Removes structurally redundant conditions, keeping only the best per category.
    
    Groups conditions by their 'structural signature' and retains only the top K
    best-scoring conditions from each group. This prevents the output from being
    dominated by many slight variations of the same condition type.
    
    Grouping Logic:
        1. Atomic/Merged/Numerical: Group by (Column Name, Operator, Type)
           - Example: All (State, !=, ColumnValueCondition) conditions compete
           - Effect: Keeps only the best value exclusion per column
        2. Conjunctive: Group by (Set of Column Names)
           - Example: All conditions using {State, Status} columns compete
        3. Pairwise: Group by (Column Name, Operator)
    
    Args:
        results: List of condition result dictionaries with 'condition' and 'score'.
        keep_top_k: Number of best conditions to keep per structural group.
    
    Returns:
        Filtered list with reduced redundancy.
    """
    grouped_results = defaultdict(list)
    
    for res in results:
        cond = res['condition']
        
        # --- Generate Signature ---
        signature = None
        
        if isinstance(cond, (ColumnValueCondition, MergedColumnValueCondition, NumericalCondition)):
            # "Similar" means same column and same operator
            # e.g. All "State !=" conditions go into one bucket
            signature = (cond.column_name, cond.operator, type(cond).__name__)
            
        elif isinstance(cond, ConjunctiveCondition):
            # "Similar" means they involve the exact same set of columns
            # e.g. All rules involving {State, Amount} go into one bucket
            cols = set()
            for sub in cond.conditions:
                cols.add(sub.column_name)
            # Use tuple(sorted) to make the set hashable
            signature = (tuple(sorted(list(cols))), 'AND', 'Conjunctive')
            
        else:
            # Fallback for unknown types: treat as unique
            signature = str(cond)

        grouped_results[signature].append(res)

    # --- Filter ---
    final_results = []
    
    for sig, group in grouped_results.items():
        # Sort by Score (desc)
        sorted_group = sorted(
            group, 
            key=lambda x: (x['score'] + 0.3 * (x['confidence'] - x['penalty'])), 
            reverse=True
        )
        # Keep Top K (default 1)
        final_results.extend(sorted_group[:keep_top_k])
        
    return final_results

def suggest_conditions_for_single_row_rule(
        violating_samples: List[dict],
        satisfying_samples: List[dict],
        categorical_columns: dict,
        numerical_columns: List[str],
        special_values: dict,
        column_groups: List[List[str]],
        min_merge_count: int = 2,
        max_merge_count: int = 4,
        min_score_threshold: float = 0.5, 
        min_support_rows: int = 5,       
        top_n_atomic_for_conjunction: int = 25
) -> List[dict]:
    """
    Suggests refinement conditions for single-row data quality rules.
    
    Analyzes violating and satisfying row samples to find conditions that:
    - Match a high proportion of satisfying samples (high confidence)
    - Match a low proportion of violating samples (low penalty)
    
    The algorithm generates and evaluates:
    1. Atomic conditions: column = value, column != value
    2. Numerical conditions: column < threshold, column >= threshold
    3. Merged conditions: column IN {set}, column NOT IN {set}
    4. Conjunctive conditions: cond1 AND cond2 (across different columns)
    
    Memory Optimization:
        Calculates metrics before storing masks and discards masks for
        low-scoring conditions immediately to prevent OOM on large datasets.
    
    Args:
        violating_samples: List of row dictionaries where the rule is violated.
        satisfying_samples: List of row dictionaries where the rule is satisfied.
        categorical_columns: Dict mapping column names to sets of observed values.
        numerical_columns: List of numerical column names for threshold conditions.
        special_values: Dict mapping column names to additional special values
            (e.g., NULL representations) to consider.
        column_groups: List of column name lists that are semantically related.
            Used to guide conjunctive condition generation.
        min_merge_count: Minimum values to combine in a merged condition.
        max_merge_count: Maximum values to combine in a merged condition.
        min_score_threshold: Minimum purity score for output (default 0.5).
        min_support_rows: Minimum satisfying matches for valid score.
        top_n_atomic_for_conjunction: Number of top atomic conditions to use
            as building blocks for conjunctive conditions.
    
    Returns:
        List of condition dictionaries sorted by score, each containing:
        - 'condition': The Condition object
        - 'confidence': Fraction of satisfying samples matching
        - 'penalty': Fraction of violating samples matching
        - 'score': Purity score
        - 'support': Count of satisfying matches
    """
    if not violating_samples or not satisfying_samples:
        return []

    # --- Step 0: Data Preparation ---
    df_vio = pd.DataFrame(violating_samples)
    df_sat = pd.DataFrame(satisfying_samples)
    
    df_vio['__is_violating'] = True
    df_sat['__is_violating'] = False
    
    df_all = pd.concat([df_vio, df_sat], ignore_index=True)
    
    total_satisfying = len(df_sat)
    total_violating = len(df_vio)

    for col in categorical_columns.keys():
        if col not in df_all.columns:
            df_all[col] = None
            
    # Storage for masks to speed up complex rule evaluation - use CompactMaskCache for memory efficiency
    condition_masks = CompactMaskCache(max_items=10000)
    n_total_rows = len(df_all)  # Store for unpacking

    # Helper for metrics
    is_violating_mask = df_all['__is_violating'].values # numpy array
    
    def calculate_metrics(cond_obj, boolean_mask):
        mask_arr = boolean_mask.values if isinstance(boolean_mask, pd.Series) else boolean_mask
        
        # Vectorized count
        sat_matches = np.sum(mask_arr & (~is_violating_mask))
        vio_matches = np.sum(mask_arr & is_violating_mask)
        
        conf = sat_matches / total_satisfying if total_satisfying > 0 else 0
        pena = vio_matches / total_violating if total_violating > 0 else 0
        
        score = _calculate_purity_score(conf, pena, sat_matches, min_support_rows)
        
        return {
            'condition': cond_obj,
            'confidence': conf,
            'penalty': pena,
            'score': score,
            'support': sat_matches 
        }
    pruning_threshold = 0.4

    # --- Step 1 & 2: Generate ATOMIC Condition Masks ---
    # We iterate and calculate immediately, only storing if useful.
    
    all_values_by_col = copy.deepcopy(categorical_columns)
    for col, values in special_values.items():
        if values is None:
            continue
        if col in all_values_by_col:
            all_values_by_col[col].update(values)
        else:
            all_values_by_col[col] = set(values)

    # We keep lists of candidates for merging/conjunction
    all_results = []
    positive_candidates_for_merge = []
    negative_candidates_for_merge = []
    
    # 1. Categorical
    for col, values in all_values_by_col.items():
        if col not in df_all.columns:
            continue
        col_series = df_all[col]
        for value in values:
            cond = ColumnValueCondition(col, '=', value)
            # Handle NaN/None
            mask = col_series.isna() if value is None else col_series == value
            
            stats_eq = calculate_metrics(cond, mask)
            
            # MEMORY FIX: Only store mask if it has potential (Score > 0.55 OR meets threshold)
            # We use 0.5 as a loose filter for things that might be merged later.
            is_useful = stats_eq['score'] > pruning_threshold
                        
            if is_useful:
                # Convert to numpy array before storing in CompactMaskCache
                mask_arr = mask.values if isinstance(mask, pd.Series) else mask
                condition_masks.set(cond, mask_arr)
            if stats_eq['score'] > 0.53:
                all_results.append(stats_eq)
            
            if isinstance(cond, ColumnValueCondition) and stats_eq['score'] > min_score_threshold:
                positive_candidates_for_merge.append(stats_eq)

            # Check Negative (!=)
            domain_vals = all_values_by_col.get(col, set())
            if len(domain_vals) > 2:
                mask_neq = ~mask
                cond_neq = ColumnValueCondition(col, '!=', value)
                stats_neq = calculate_metrics(cond_neq, mask_neq)
                
                if stats_neq['score'] > pruning_threshold:
                    mask_neq_arr = mask_neq.values if isinstance(mask_neq, pd.Series) else mask_neq
                    condition_masks.set(cond_neq, mask_neq_arr)
                
                if stats_neq['score'] > 0.53:
                    all_results.append(stats_neq)
                
                if stats_neq['score'] > min_score_threshold:
                    negative_candidates_for_merge.append(stats_neq)

    # 2. Numerical
    for col in numerical_columns:
        split_list = find_optimal_numerical_split(
            df_all, col, min_support=min_support_rows
        )
        
        for split_info in split_list:
            split_val = split_info['split_value']
            col_series = df_all[col]
            
            # Less Than
            cond_lt = NumericalCondition(col, '<', split_val)
            mask_lt = col_series < split_val
            stats_lt = calculate_metrics(cond_lt, mask_lt)
            
            if stats_lt['score'] > pruning_threshold:
                mask_lt_arr = mask_lt.values if isinstance(mask_lt, pd.Series) else mask_lt
                condition_masks.set(cond_lt, mask_lt_arr)
                if stats_lt['score'] > min_score_threshold:
                    all_results.append(stats_lt)
            
            # Greater Than or Equal
            cond_gte = NumericalCondition(col, '>=', split_val)
            mask_gte = col_series >= split_val
            stats_gte = calculate_metrics(cond_gte, mask_gte)
            
            if stats_gte['score'] > pruning_threshold:
                mask_gte_arr = mask_gte.values if isinstance(mask_gte, pd.Series) else mask_gte
                condition_masks.set(cond_gte, mask_gte_arr)
                if stats_gte['score'] > min_score_threshold:
                    all_results.append(stats_gte)

    #for res in all_results:
     #   print(f"Atomic Condition: {res['condition']} | Score: {res['score']:.3f} | Conf: {res['confidence']:.3f} | Pen: {res['penalty']:.3f}")
    # --- Step 4: Merged Conditions (Vectorized) ---    
    MAX_CANDIDATES_PER_COL = 10

    col_to_pos_conds = defaultdict(list)
    col_to_neg_conds = defaultdict(list)
    
    # Combine lists to sort by quality before capping
    all_merge_candidates = positive_candidates_for_merge + negative_candidates_for_merge
    
    # Sort: Prioritize high score, tie-break with informedness (Confidence - Penalty)
    sorted_merge_candidates = sorted(
        all_merge_candidates, 
        key=lambda x: x['score'] + 0.3 * (x['confidence'] - x['penalty']), 
        reverse=True
    )

    # Populate groups, respecting the limit per column/operator
    for res in sorted_merge_candidates:
        cond = res['condition']
        col = cond.column_name
        
        if cond.operator == '=':
            if len(col_to_pos_conds[col]) < MAX_CANDIDATES_PER_COL:
                col_to_pos_conds[col].append(cond)
        elif cond.operator == '!=':
            if len(col_to_neg_conds[col]) < MAX_CANDIDATES_PER_COL:
                col_to_neg_conds[col].append(cond)

    # --- Build a score lookup for atomic conditions (for prioritizing combinations) ---
    atomic_scores = {}
    for res in positive_candidates_for_merge + negative_candidates_for_merge:
        atomic_scores[res['condition']] = res['score']
    
    # --- Collect all potential merged condition candidates first ---
    merged_candidates_to_eval = []  # List of (combo, col, operator_type, estimated_priority)
    
    def collect_merged_candidates(col_groups, operator_type):
        for col, conds in col_groups.items():
            if len(conds) < min_merge_count:
                continue
            
            domain_values = all_values_by_col.get(col, set())
            norm_domain = set(_normalize_value(v) for v in domain_values) if domain_values else set()
            domain_size = len(norm_domain)

            for k in range(min_merge_count, min(len(conds), max_merge_count) + 1):
                for combo in combinations(conds, k):
                    
                    # --- Deduplication & Optimization Logic ---
                    if norm_domain:
                        current_set_size = k
                        complement_size = domain_size - current_set_size

                        if operator_type == '=':
                            if complement_size <= 0: continue 
                            if current_set_size > complement_size: continue
                            if complement_size == 1: continue 

                        elif operator_type == '!=':
                            if complement_size <= 0: continue
                            if current_set_size >= complement_size: continue
                            if complement_size == 1: continue
                    
                    # Ensure masks exist
                    valid_combo = all(c in condition_masks for c in combo)
                    if not valid_combo: continue

                    # Calculate priority: mean score of component conditions
                    combo_scores = [atomic_scores.get(c, 0) for c in combo]
                    priority = sum(combo_scores) / len(combo_scores) if combo_scores else 0
                    
                    merged_candidates_to_eval.append((combo, col, operator_type, priority))
    
    collect_merged_candidates(col_to_pos_conds, '=')
    collect_merged_candidates(col_to_neg_conds, '!=')

        
    # --- Now evaluate the limited set ---
    for combo, col, operator_type, _ in merged_candidates_to_eval:
        # Retrieve packed masks from cache for bitwise operations
        packed_masks = [condition_masks.get_packed(c) for c in combo]
        
        # Skip if any mask is missing
        if any(m is None for m in packed_masks):
            continue
        
        # Use np.bitwise_or/and on packed data for efficiency
        if operator_type == '=':
            # OR operation for IN set
            combined_packed = packed_masks[0].copy()
            for pm in packed_masks[1:]:
                np.bitwise_or(combined_packed, pm, out=combined_packed)
        else:
            # AND operation for NOT IN set
            combined_packed = packed_masks[0].copy()
            for pm in packed_masks[1:]:
                np.bitwise_and(combined_packed, pm, out=combined_packed)
        
        # Unpack for metrics calculation
        combined_mask = np.unpackbits(combined_packed)[:n_total_rows].astype(bool)
        
        value_set = frozenset(c.value for c in combo)
        merged_cond = MergedColumnValueCondition(col, value_set, operator_type)
        
        stats = calculate_metrics(merged_cond, combined_mask)
        
        # Store if good - use set_packed to avoid re-packing
        if stats['score'] > min_score_threshold:
            condition_masks.set_packed(merged_cond, combined_packed)
            all_results.append(stats)

    # --- Step 5: Conjunctive Conditions (Vectorized) ---
    
    # 1. Prepare Candidates (Same sorting logic as original)
    # Instead of filtering by min_score_threshold, we look at all cached candidates (pruning_threshold)
    # We reconstruct the 'stats' for everything in the cache or use a list we tracked
    potential_parents = []
    
    # We need to look at conditions that might have failed the strict output check
    # but passed the pruning check. Iterate over the cache's internal dict.
    for cond in condition_masks.cache.keys():
        # Retrieve and unpack the mask for metrics calculation
        mask = condition_masks.get(cond, n_total_rows)
        if mask is None:
            continue
        # Quick re-calc or retrieval (optimization: could store these stats earlier)
        s = calculate_metrics(cond, mask) 
        if s['score'] > pruning_threshold:
            potential_parents.append(s)
    by_score = sorted(potential_parents, key=lambda x: (x['score'], x['confidence']), reverse=True)
    by_informedness = sorted(potential_parents, key=lambda x: (x['confidence'] - x['penalty']), reverse=True)
    by_coverage = sorted(potential_parents, key=lambda x: x['confidence'], reverse=True)
    
    #for p in potential_parents:
     #   print(f"Candidate: {p['condition']} | Score: {p['score']:.3f} | Conf: {p['confidence']:.3f} | Pen: {p['penalty']:.3f}")

    # Combine lists (Top N from each strategy)
    global_candidates_list = (
        by_score[:top_n_atomic_for_conjunction] + 
        by_informedness[:top_n_atomic_for_conjunction] + 
        by_coverage[:top_n_atomic_for_conjunction]
    )
    global_top_candidates = []
    seen_globals = set()
    for res in global_candidates_list:
        if res['condition'] not in seen_globals:
            global_top_candidates.append(res)
            seen_globals.add(res['condition'])

    conditions_by_col = defaultdict(list)
    results_by_col_map = defaultdict(list)
    for res in potential_parents:
        results_by_col_map[res['condition'].column_name].append(res)

    top_k_per_group_col = 25
    for col, items in results_by_col_map.items():
        top_score = sorted(items, key=lambda x: (x['score'], x['confidence']), reverse=True)[:top_k_per_group_col]
        top_conf = sorted(items, key=lambda x: (x['confidence'] - x['penalty']), reverse=True)[:top_k_per_group_col]
        combined = {res['condition'] for res in top_score + top_conf}
        conditions_by_col[col] = list(combined)
        
    global_top_conds = [res['condition'] for res in global_top_candidates]
    global_top_set = set(global_top_conds)
    
    combinations_to_test = set()

    def get_valid_combos(candidate_pool, k):
        for combo in combinations(candidate_pool, k):
            cols = {c.column_name for c in combo}
            if len(cols) == k:
                    yield frozenset(combo)

    # Strategy 1 & 2 (Same logic)
    if len(global_top_conds) > 1:
        for k in [2, 3]:
            for combo_set in get_valid_combos(global_top_conds, k):
                combinations_to_test.add(combo_set)

    if column_groups:
        for group in column_groups:
            local_pool = []
            for col_name in group:
                local_pool.extend(conditions_by_col[col_name])
            if not local_pool: continue
            augmented_pool = list(set(local_pool) | global_top_set)
            for k in [2, 3]:
                if len(augmented_pool) < k: continue
                for combo_set in get_valid_combos(augmented_pool, k):
                    if combo_set in combinations_to_test: continue
                    combinations_to_test.add(combo_set)

    # --- Execute Evaluation using np.bitwise_and on packed masks ---
    if combinations_to_test:
        conjunctive_results = []
        
        # Helper to retrieve packed mask or reconstruct from atomic components
        def get_packed_mask_on_demand(c):
            """Returns packed mask for a condition, reconstructing from atomics if needed."""
            # 1. Check if directly in cache
            packed = condition_masks.get_packed(c)
            if packed is not None:
                return packed
            
            # 2. Merged conditions are reconstructed from atomic components
            if isinstance(c, MergedColumnValueCondition):
                atomic_op = '=' if c.operator == '=' else '!='
                
                packed_sub_masks = []
                for val in c.value_set:
                    atom_key = ColumnValueCondition(c.column_name, atomic_op, val)
                    packed_atom = condition_masks.get_packed(atom_key)
                    if packed_atom is not None:
                        packed_sub_masks.append(packed_atom)
                
                if not packed_sub_masks: 
                    return None
                
                # Use bitwise operations on packed data
                if c.operator == '=':
                    # OR operation for IN set
                    result = packed_sub_masks[0].copy()
                    for pm in packed_sub_masks[1:]:
                        np.bitwise_or(result, pm, out=result)
                else:
                    # AND operation for NOT IN set
                    result = packed_sub_masks[0].copy()
                    for pm in packed_sub_masks[1:]:
                        np.bitwise_and(result, pm, out=result)
                return result
                    
            return None

        for combo_set in combinations_to_test:
            cond_list = list(combo_set)
            
            # Get packed masks for all conditions in the conjunction
            packed_masks = [get_packed_mask_on_demand(c) for c in cond_list]
            
            # Skip if any mask is missing
            if any(m is None for m in packed_masks):
                continue

            try:
                # Use np.bitwise_and on packed data for efficiency
                conj_packed = packed_masks[0].copy()
                for pm in packed_masks[1:]:
                    np.bitwise_and(conj_packed, pm, out=conj_packed)
                
                # Unpack for metrics calculation
                conj_mask = np.unpackbits(conj_packed)[:n_total_rows].astype(bool)
                conj_cond = ConjunctiveCondition(combo_set)
                stats = calculate_metrics(conj_cond, conj_mask)
                
                if stats['score'] > min_score_threshold:
                    conjunctive_results.append(stats)
            except ValueError:
                continue

        all_results.extend(conjunctive_results)

    # --- Final Deduplication and Sort ---
    final_map = {res['condition']: res for res in all_results}
    unique_results = list(final_map.values())

    # Apply aggressive redundancy filtering
    filtered_results = _filter_redundant_conditions(unique_results, keep_top_k=2)
    #filtered_results = unique_results

    filtered = sorted(filtered_results, key=lambda x: x['score'] + 0.4 * (x['confidence'] - x['penalty']), reverse=True)

    return filtered


def _normalize_value(value: Any) -> Any:
    """
    Normalizes common string representations of null values to Python None.
    
    Ensures consistent handling of NULL-like values that may appear differently
    in source data (e.g., 'None', 'null', 'NaN', 'NA', '<NA>').
    
    Args:
        value: Any value that might represent null.
    
    Returns:
        None if the value is a null-like string, otherwise the original value.
    """
    if isinstance(value, str):
        val_lower = value.lower()
        if val_lower in ['none', 'null', 'nan', 'na', '<na>']:
            return None
    return value


def _flatten_multi_row_samples_to_df(samples: dict) -> pd.DataFrame:
    """
    Flattens nested multi-row sample structures into a flat DataFrame.
    
    Multi-row rules use a complex nested structure to represent groups of rows
    that together form a violation or satisfaction. This function flattens that
    structure for vectorized processing.
    
    Input Structure:
        {
            'group_key_1': [  # Group identifier
                {  # First representative sample
                    'role_1': [{row1}, {row2}],  # Rows with role_1
                    'role_2': [{row3}]           # Rows with role_2
                }
            ],
            'group_key_2': [...]
        }
    
    Output: DataFrame with columns:
        - __sample_id: Unique identifier for each sample within groups
        - __role: The semantic role of the row (e.g., 'target', 'reference')
        - __row_id: Index within the role's row list
        - [all original row columns]
    
    Args:
        samples: Nested dictionary of multi-row samples.
    
    Returns:
        Flattened DataFrame suitable for vectorized condition evaluation.
    """
    flat_rows = []
    sample_id_counter = 0
    
    for group_key, repre_content in samples.items():
        repre_list = repre_content if isinstance(repre_content, list) else [repre_content]
        for repre_dict in repre_list:
            current_sample_id = sample_id_counter
            sample_id_counter += 1
            for role, rows_content in repre_dict.items():
                rows_list = rows_content if isinstance(rows_content, list) else ([rows_content] if rows_content else [])
                for i, row_data in enumerate(rows_list):
                    record = row_data.copy()
                    record['__sample_id'] = current_sample_id
                    record['__role'] = role
                    record['__row_id'] = i 
                    flat_rows.append(record)

    if not flat_rows: return pd.DataFrame()
    df = pd.DataFrame(flat_rows)
    # Downcast for memory efficiency
    for col in df.select_dtypes(include=['int64', 'float64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='integer' if 'int' in str(df[col].dtype) else 'float')
    return df

def _create_pairs_dataframe(df: pd.DataFrame, strategy: str, max_pairs_limit: int = 5_000_000) -> pd.DataFrame:
    """
    Creates a DataFrame of row pairs within samples for multi-row rule evaluation.
    
    For multi-row rules, we often need to evaluate conditions on pairs of rows
    (e.g., checking if two rows have the same department). This function generates
    those pairs according to different strategies.
    
    Pairing Strategies:
        - 'between_groups': Pairs rows with different __role values
            (e.g., compare 'target' rows with 'reference' rows)
        - 'within_groups': Pairs rows with the same __role, different __row_id
            (e.g., find duplicates within the same group)
        - 'all_combinations': All pairs with row_id_1 < row_id_2
            (used for confidence calculation)
    
    Args:
        df: Flattened DataFrame from _flatten_multi_row_samples_to_df.
        strategy: Pairing strategy ('between_groups', 'within_groups', 'all_combinations').
        max_pairs_limit: Maximum pairs before downsampling. Prevents OOM on large datasets.
    
    Returns:
        DataFrame with columns {col}_1 and {col}_2 for each original column,
        representing the two rows in each pair.
    
    Note:
        If estimated pairs exceed max_pairs_limit, groups are downsampled
        to keep computation tractable.
    """
    if df.empty: return pd.DataFrame()

    # Estimate and Downsample
    group_sizes = df.groupby('__sample_id').size()
    estimated_pairs = (group_sizes ** 2).sum()

    if estimated_pairs > max_pairs_limit:
        avg_rows = max(2, int((max_pairs_limit / len(group_sizes))**0.5))
        #Sample rows per group, if group has less than avg_rows, take all
        df = df.groupby('__sample_id').apply(lambda x: x.sample(n=min(len(x), avg_rows))).reset_index(drop=True)

    try:
        df_pairs = pd.merge(df, df, on='__sample_id', suffixes=('_1', '_2'))
    except MemoryError:
        return pd.DataFrame()

    if strategy == 'between_groups':
        df_pairs = df_pairs[df_pairs['__role_1'] != df_pairs['__role_2']]
    elif strategy == 'within_groups':
        df_pairs = df_pairs[(df_pairs['__role_1'] == df_pairs['__role_2']) & (df_pairs['__row_id_1'] < df_pairs['__row_id_2'])]
    elif strategy == 'all_combinations':
        df_pairs = df_pairs[df_pairs['__row_id_1'] < df_pairs['__row_id_2']]
    
    return df_pairs

def _prepare_confidence_data(df_sat: pd.DataFrame, df_vio: pd.DataFrame, satisfying_in_violations: bool) -> pd.DataFrame:
    """
    Prepares the DataFrame for confidence calculation in multi-row rules.
    
    Confidence is calculated as the fraction of satisfying pairs/samples that
    match a condition. For some rule types, the 'satisfying' set includes
    parts of violating samples (e.g., individual rows that are valid even if
    the group relationship is invalid).
    
    Args:
        df_sat: Flattened DataFrame of satisfying samples.
        df_vio: Flattened DataFrame of violating samples.
        satisfying_in_violations: If True, adds rows from violating samples
            to the confidence dataset with modified sample IDs to prevent
            them from forming violating pairs with each other.
    
    Returns:
        DataFrame containing all rows to use for confidence calculation.
    """
    df_conf = df_sat.copy()
    if satisfying_in_violations and not df_vio.empty:
        df_vio_sub = df_vio.copy()
        # Split violating roles into separate groups so they don't form violating pairs
        df_vio_sub['__sample_id'] = df_vio_sub['__sample_id'].astype(str) + "_" + df_vio_sub['__role'].astype(str)
        df_conf = pd.concat([df_conf, df_vio_sub], ignore_index=True)
    return df_conf

def _evaluate_conditions_with_cache(
    candidate_conditions: List[Union[PairWiseCondition, AtomicCondition, ConjunctiveCondition]],
    # Data Containers
    df_conf_rows: pd.DataFrame,
    df_conf_pairs: pd.DataFrame,
    target_vio: pd.DataFrame, 
    vio_mode: str,
    # Counts
    total_conf_rows: int,
    total_conf_pairs: int,
    total_vio_checks: int,
    # Threshold
    min_score: float,
    min_support: int,
    # Caches (Mutable Dictionaries)
    cache_conf_rows: Union[Dict, CompactMaskCache],
    cache_vio: Union[Dict, CompactMaskCache],
    pruning_threshold: float = 0.37
) -> List[dict]:
    """
    Evaluates conditions using vectorized operations with mask caching.
    
    Core evaluation engine that computes confidence and penalty for each
    condition using boolean masks. Masks are cached (optionally compressed)
    to enable efficient evaluation of conjunctive conditions that reuse
    atomic masks.
    
    Memory Management:
        Low-scoring conditions (below pruning_threshold) have their masks
        removed from the cache immediately to prevent memory buildup.
    
    Args:
        candidate_conditions: List of condition objects to evaluate.
        df_conf_rows: Flattened rows DataFrame for confidence calculation.
        df_conf_pairs: Pairs DataFrame for pairwise confidence (may be empty).
        target_vio: DataFrame for penalty calculation (rows or pairs).
        vio_mode: Either 'group' (penalty by sample) or 'pairs' (penalty by pair).
        total_conf_rows: Total rows for confidence normalization.
        total_conf_pairs: Total pairs for pairwise confidence normalization.
        total_vio_checks: Total checks for penalty normalization.
        min_score: Minimum score threshold for output inclusion.
        min_support: Minimum satisfying matches for valid score.
        cache_conf_rows: Cache for confidence row masks (dict or CompactMaskCache).
        cache_vio: Cache for penalty masks (dict or CompactMaskCache).
        pruning_threshold: Score below which masks are removed from cache.
    
    Returns:
        List of condition dictionaries with score >= min_score.
    """
    results = []

    # --- Mask Helper with Caching ---
    def get_mask(cond, context_type):
        """
        Retrieves or computes the boolean mask.
        context_type: 'conf_rows', 'conf_pairs', 'vio'
        """
        # 1. Determine Target DF and Cache
        if context_type == 'conf_rows':
            df_target = df_conf_rows
            cache = cache_conf_rows
            is_pair_data = False
        elif context_type == 'conf_pairs':
            df_target = df_conf_pairs
            cache = None # Don't cache pairwise confidence masks (usually unique/large)
            is_pair_data = True
        elif context_type == 'vio':
            df_target = target_vio
            cache = cache_vio
            is_pair_data = (vio_mode == 'pairs')
        else:
            raise ValueError("Invalid context")

        # 2. Check Cache (Atomic only)
        if cache is not None:
            if isinstance(cache, CompactMaskCache):
                cached_mask = cache.get(cond, len(df_target))
                if cached_mask is not None: return cached_mask
            elif cond in cache:
                return cache[cond]

        # 3. Compute Mask
        mask = None
        
        # A. Compound Conditions (Recursive reuse of cache)
        if isinstance(cond, ConjunctiveCondition):
            # Optimization: Use bitwise_and on packed masks when using CompactMaskCache
            if cache is not None and isinstance(cache, CompactMaskCache):
                packed_sub_masks = []
                for sub in cond.conditions:
                    # Try to get packed mask directly
                    packed = cache.get_packed(sub)
                    if packed is None:
                        # Compute and store it first
                        sub_mask = get_mask(sub, context_type)
                        packed = cache.get_packed(sub)
                    if packed is not None:
                        packed_sub_masks.append(packed)
                
                if packed_sub_masks:
                    # Use np.bitwise_and on packed data
                    result_packed = packed_sub_masks[0].copy()
                    for pm in packed_sub_masks[1:]:
                        np.bitwise_and(result_packed, pm, out=result_packed)
                    # Unpack for return
                    mask = np.unpackbits(result_packed)[:len(df_target)].astype(bool)
                else:
                    mask = np.ones(len(df_target), dtype=bool)
            else:
                # Fallback to regular logical_and for non-CompactMaskCache
                sub_masks = [get_mask(sub, context_type) for sub in cond.conditions]
                if sub_masks:
                    mask = np.logical_and.reduce(sub_masks)
                else:
                    mask = np.ones(len(df_target), dtype=bool)

                
        elif isinstance(cond, MergedColumnValueCondition):
            col = cond.column_name
            if is_pair_data:
                # --- FIX START ---
                # Must check BOTH sides of the pair for set membership
                c1, c2 = f"{col}_1", f"{col}_2"
                if c1 in df_target.columns and c2 in df_target.columns:
                    s1 = df_target[c1]
                    s2 = df_target[c2]
                    
                    if cond.operator == '=':
                        # Both rows must be IN the set
                        mask = s1.isin(cond.value_set) & s2.isin(cond.value_set)
                    else: # !=
                        # Both rows must be NOT IN the set
                        mask = (~s1.isin(cond.value_set)) & (~s2.isin(cond.value_set))
                else:
                    mask = np.zeros(len(df_target), dtype=bool)
                # --- FIX END ---
            else:
                # Standard single row logic
                s = df_target.get(col)
                # Fallback if referencing single cols in a pair DF (rare edge case in confidence calc)
                if s is None: s = df_target.get(f"{col}_1")

                if s is not None:
                    mask = s.isin(cond.value_set) if cond.operator == '=' else ~s.isin(cond.value_set)
                else:
                    mask = np.zeros(len(df_target), dtype=bool)

        # B. Atomic Computation (if not found in recursion)
        if mask is None:
            if isinstance(cond, PairWiseCondition):
                if not is_pair_data: 
                    mask = np.zeros(len(df_target), dtype=bool)
                else:
                    c1, c2 = f"{cond.column_name}_1", f"{cond.column_name}_2"
                    if c1 in df_target.columns and c2 in df_target.columns:
                        v1, v2 = df_target[c1], df_target[c2]
                        not_na = v1.notna() & v2.notna()
                        mask = ((v1 == v2) if cond.operator == '=' else (v1 != v2)) & not_na
                    else:
                        mask = np.zeros(len(df_target), dtype=bool)

            elif isinstance(cond, (AtomicCondition, ConjunctiveCondition)): # Atomic base cases
                
                def single_col_mask(dframe, c_obj, suffix=""):
                    col = f"{c_obj.column_name}{suffix}"
                    if col not in dframe.columns: return np.zeros(len(dframe), dtype=bool)
                    s = dframe[col]
                    
                    if isinstance(c_obj, ColumnValueCondition):
                        if c_obj.value is None: return s.isna()
                        return (s == c_obj.value) if c_obj.operator == '=' else (s != c_obj.value)
                    elif isinstance(c_obj, NumericalCondition):
                        return (s < c_obj.value) if c_obj.operator == '<' else (s >= c_obj.value)
                    return np.zeros(len(dframe), dtype=bool)

                if is_pair_data:
                    mask = single_col_mask(df_target, cond, "_1") & single_col_mask(df_target, cond, "_2")
                else:
                    mask = single_col_mask(df_target, cond, "")

        # 4. Store in Cache (Atomic/Merged only)
        if cache is not None and not isinstance(cond, ConjunctiveCondition):
            if isinstance(cache, CompactMaskCache):
                cache.set(cond, mask)
            else:
                cache[cond] = mask
            
        return mask

    # --- Main Loop ---
    for cond in candidate_conditions:
        
        # 1. Confidence
        sat_matches = 0
        if isinstance(cond, PairWiseCondition):
            if total_conf_pairs == 0: conf = 0.0
            else: 
                match_count = get_mask(cond, 'conf_pairs').sum()
                sat_matches = match_count 
                conf = match_count / total_conf_pairs
        else:
            if total_conf_rows == 0: conf = 0.0
            else: 
                match_count = get_mask(cond, 'conf_rows').sum()
                sat_matches = match_count
                conf = match_count / total_conf_rows

        # 2. Penalty
        if total_vio_checks == 0:
            pen = 0.0
        else:
            mask_vio = get_mask(cond, 'vio')
            if vio_mode == 'group':
                temp_df = pd.DataFrame({'id': target_vio['__sample_id'], 'match': mask_vio})
                pen = temp_df.groupby('id')['match'].all().sum() / total_vio_checks
            else:
                pen = mask_vio.sum() / total_vio_checks

        score = _calculate_purity_score(conf, pen, sat_matches, min_support)
                
        # 1. Output Check (Strict)
        if score >= min_score:
            results.append({'condition': cond, 'confidence': conf, 'penalty': pen, 'score': score})
        
        # 2. Memory Pruning Check (Loose)
        # CHANGED: We now use pruning_threshold to decide whether to delete from cache
        if score < pruning_threshold:
            # If the condition is garbage (very low score), remove from cache to save RAM.
            # However, if it's between 0.15 and 0.53, KEEP IT. It might be a good parent.
            if not isinstance(cond, ConjunctiveCondition) and not isinstance(cache_conf_rows, CompactMaskCache):
                if cond in cache_conf_rows: del cache_conf_rows[cond]
                if cond in cache_vio: del cache_vio[cond]
            
    return results

def suggest_conditions_for_multi_row_rule(
        violating_samples: dict,
        satisfying_samples: dict,
        categorical_columns: dict,
        numerical_columns: List[str],
        special_values: Optional[dict],
        column_groups: Optional[List[List[str]]] = None,
        only_pairwise_columns: Optional[List[str]] = None,
        violation_pattern: str = 'between_groups',
        satisfying_in_violations: bool = False,
        min_score_threshold: float = 0.53, 
        min_support_rows: int = 10,
        min_merge_count: int = 2,
        max_merge_count: int = 4,
        top_n_atomic_for_conjunction: int = 25
) -> List[Dict]:
    """
    Suggests refinement conditions for multi-row data quality rules.
    
    Multi-row rules involve relationships between multiple rows (e.g., functional
    dependencies, uniqueness constraints). This function finds conditions that
    refine when such relationships should hold.
    
    The algorithm handles three types of conditions:
    1. Single-row conditions: Applied to both/all rows in a pair/group
    2. Pairwise conditions: Compare values between two rows (same/different)
    3. Conjunctive conditions: Combinations of the above
    
    Violation Patterns:
        - 'between_groups': Compare rows with different roles
            (e.g., the source vs target in a functional dependency)
        - 'within_groups': Compare rows within the same role
            (e.g., finding duplicates)
        - 'whole_group': Check if ALL rows in a violation sample match
    
    Args:
        violating_samples: Nested dict of violating sample groups.
        satisfying_samples: Nested dict of satisfying sample groups.
        categorical_columns: Dict mapping column names to observed values.
        numerical_columns: List of numerical column names.
        special_values: Dict of special values per column.
        column_groups: Semantically related column groups for conjunction hints.
        only_pairwise_columns: Columns to use only for pairwise conditions
            (typically identifier columns detected by semantic analysis).
        violation_pattern: How violations form between rows.
        satisfying_in_violations: Include parts of violating samples in confidence.
        min_score_threshold: Minimum purity score for output.
        min_support_rows: Minimum satisfying matches for valid score.
        min_merge_count: Minimum values for merged conditions.
        max_merge_count: Maximum values for merged conditions.
        top_n_atomic_for_conjunction: Number of top atomics for conjunctions.
    
    Returns:
        Sorted list of condition dictionaries by score.
    """
    
    # --- Step 0: Flatten and Prepare Data ONCE ---
    df_vio_raw = _flatten_multi_row_samples_to_df(violating_samples)
    df_sat_raw = _flatten_multi_row_samples_to_df(satisfying_samples)
    #print(f"  Violating samples flattened to {len(df_vio_raw)} rows across {df_vio_raw['__sample_id'].nunique() if not df_vio_raw.empty else 0} samples.")
    #print(f"  Satisfying samples flattened to {len(df_sat_raw)} rows across {df_sat_raw['__sample_id'].nunique() if not df_sat_raw.empty else 0} samples.")
    # Ensure columns
    all_cols = set(categorical_columns.keys()).union(numerical_columns)
    for df in [df_vio_raw, df_sat_raw]:
        if not df.empty:
            missing = all_cols - set(df.columns)
            for c in missing: df[c] = None

    # A. Prepare Confidence Data
    df_conf_rows = _prepare_confidence_data(df_sat_raw, df_vio_raw, satisfying_in_violations)
    total_conf_rows = len(df_conf_rows)
    
    # Only generate conf pairs if we actually check PairWise conditions (check later)
    df_conf_pairs = pd.DataFrame() 
    total_conf_pairs = 0
    
    # B. Prepare Penalty Data
    if violation_pattern == 'whole_group':
        target_vio = df_vio_raw
        vio_mode = 'group'
        total_vio_checks = df_vio_raw['__sample_id'].nunique() if not df_vio_raw.empty else 0
    else:
        target_vio = _create_pairs_dataframe(df_vio_raw, violation_pattern)
        vio_mode = 'pairs'
        total_vio_checks = len(target_vio)
        
    # --- Caching Structures ---
    # We maintain these across the steps to reuse atomic calculations in merged/conjunctive steps
    cache_conf_rows = CompactMaskCache() # {condition: np.array(bool)}
    cache_vio = CompactMaskCache()       # {condition: np.array(bool)}

    # --- Step 1: Generate Candidates ---
    conditions_to_eval = set()
    
    all_cols_with_values = copy.deepcopy(categorical_columns)
    if not all_cols_with_values:
        all_cols_with_values = dict()
    if special_values:
        for col, values in special_values.items():
            if values is None:
                continue
            #if col not in numerical_columns:
            if col not in all_cols_with_values:
                all_cols_with_values[col] = set()
            all_cols_with_values[col].update(values)
            
    for col in all_cols_with_values:
        domain_size = len(all_cols_with_values[col])

        conditions_to_eval.add(PairWiseCondition(col, '='))
        conditions_to_eval.add(PairWiseCondition(col, '!='))
        for value in all_cols_with_values[col]:
            if only_pairwise_columns and col in only_pairwise_columns:
                continue
            conditions_to_eval.add(ColumnValueCondition(col, '=', value))
            if domain_size > 2:
                conditions_to_eval.add(ColumnValueCondition(col, '!=', value))

    if numerical_columns:
        df_combined = pd.concat([df_vio_raw.assign(__is_violating=True), df_sat_raw.assign(__is_violating=False)], ignore_index=True)
        for col in numerical_columns:
            split_list = find_optimal_numerical_split(
                df_combined, col, min_support=min_support_rows
            )
            for split_info in split_list:
                conditions_to_eval.add(NumericalCondition(col, '<', split_info['split_value']))
                conditions_to_eval.add(NumericalCondition(col, '>=', split_info['split_value']))

    # Lazy load conf pairs if needed
    if any(isinstance(c, PairWiseCondition) for c in conditions_to_eval):
        df_conf_pairs = _create_pairs_dataframe(df_conf_rows, 'all_combinations')
        total_conf_pairs = len(df_conf_pairs)

    # --- Step 2: Evaluate Atomic (Fills Cache) ---
    candidate_score_threshold = 0.53
    pruning_threshold = 0.37

    
    atomic_results = _evaluate_conditions_with_cache(
        list(conditions_to_eval),
        df_conf_rows, df_conf_pairs, target_vio, vio_mode,
        total_conf_rows, total_conf_pairs, total_vio_checks,
        min_score=candidate_score_threshold,
        min_support=min_support_rows, 
        cache_conf_rows=cache_conf_rows, 
        cache_vio=cache_vio,
        pruning_threshold=pruning_threshold
    )
    
    atomic_results_broad = _evaluate_conditions_with_cache(
        list(conditions_to_eval),
        df_conf_rows, df_conf_pairs, target_vio, vio_mode,
        total_conf_rows, total_conf_pairs, total_vio_checks,
        min_score=pruning_threshold, # Get everything useful back
        min_support=min_support_rows, 
        cache_conf_rows=cache_conf_rows, 
        cache_vio=cache_vio,
        pruning_threshold=pruning_threshold
    )
    
    all_results = [r for r in atomic_results_broad if r['score'] > min_score_threshold]
    
# --- Step 3: Merged Conditions (Reuses Cache) ---
    # 1. Categorize and Limit Candidates to prevent explosion
    # limit N in nCk to ensure reasonable processing time
    MAX_CANDIDATES_PER_COL = 12
    
    col_to_positive_conds = defaultdict(list)
    col_to_negative_conds = defaultdict(list)
    
    # Sort results by score to ensure we only merge the best candidates
    sorted_atomics = sorted(atomic_results, key=lambda x: x['score'] + 0.3 * (x['confidence'] - x['penalty']), reverse=True)
    
    #for res in sorted_atomics:
    #    print(f"  Condition: {res['condition']}, Score: {res['score']:.4f}, Conf: {res['confidence']:.4f}, Penalty: {res['penalty']:.4f}")
    
    for res in sorted_atomics:
        cond = res['condition']
        if isinstance(cond, ColumnValueCondition):
            # Enforce limit per column
            if cond.operator == '=':
                if len(col_to_positive_conds[cond.column_name]) < MAX_CANDIDATES_PER_COL:
                    col_to_positive_conds[cond.column_name].append(cond)
            elif cond.operator == '!=':
                if len(col_to_negative_conds[cond.column_name]) < MAX_CANDIDATES_PER_COL:
                    col_to_negative_conds[cond.column_name].append(cond)
    
    merged_candidates = []
    
    # Helper to generate combinations
    def generate_merge_candidates(grouped_conds, operator_type):
        for col, conds in grouped_conds.items():
            if len(conds) < min_merge_count:
                continue
            
            domain_values = all_cols_with_values.get(col, set())
            norm_domain = set(_normalize_value(v) for v in domain_values) if domain_values else set()
            domain_size = len(norm_domain)

            limit = min(len(conds), max_merge_count)
            
            for k in range(min_merge_count, limit + 1):
                for combo in combinations(conds, k):
                    val_set = frozenset(_normalize_value(c.value) for c in combo)
                    
                    # --- Validation Logic ---
                    if norm_domain:
                        
                        current_set_size = len(val_set)
                        complement_size = domain_size - current_set_size
                        
                        # Calculate the size of the "Allowed" set vs "Excluded" set
                        if operator_type == '=':
                            # Condition: IN {A, B...}
                            # Validation: Ensure we aren't selecting the whole world
                            if complement_size <= 0: continue 

                            # Preference: Prefer Positive if Size <= Complement (Equal -> Prefer Positive)
                            if current_set_size > complement_size: continue
                            
                            # Optimization: If complement is 1, use atomic '!=' instead
                            if complement_size == 1: continue
                        
                        elif operator_type == '!=':
                            # Condition: NOT IN {A, B...}
                            # Validation: Ensure we aren't excluding everything
                            if complement_size <= 0: continue 

                            # Preference: Prefer Negative ONLY if Size < Complement (Strictly Smaller)
                            # If sizes are equal, we prefer Positive.
                            if current_set_size >= complement_size: continue
                            
                            # Optimization: If complement (Allowed) is 1, use atomic '=' instead
                            if complement_size == 1: continue

                    merged_candidates.append(MergedColumnValueCondition(col, val_set, operator_type))

    # 2. Generate Positive Merges (IN {A, B}) from Positive Atomics (= A, = B)
    # Logic: "Rows where Col is A OR Col is B"
    generate_merge_candidates(col_to_positive_conds, '=')

    # 3. Generate Negative Merges (NOT IN {A, B}) from Negative Atomics (!= A, != B)
    # Logic: "Rows where Col != A AND Col != B" -> "Rows where Col NOT IN {A, B}"
    generate_merge_candidates(col_to_negative_conds, '!=')
    merged_results = []
    if merged_candidates:
        merged_results = _evaluate_conditions_with_cache(
            merged_candidates,
            df_conf_rows, df_conf_pairs, target_vio, vio_mode,
            total_conf_rows, total_conf_pairs, total_vio_checks,
            candidate_score_threshold,
            min_support_rows, 
            cache_conf_rows, cache_vio,
            pruning_threshold=0.47
        )
        #for res in merged_results:
        #   print(f"  Merged Condition: {res['condition']}, Score: {res['score']:.4f}, Conf: {res['confidence']:.4f}, Penalty: {res['penalty']:.4f}")
        all_results.extend(merged_results)

    # --- Step 4: Conjunctive Conditions (Reuses Cache) ---
    # 1. Prepare the candidate lists
    valid_candidates_pool = [res for res in atomic_results_broad 
                             if not isinstance(res['condition'], PairWiseCondition)]
    valid_candidates_pool += [res for res in merged_results]
    
    # Sort A: High Purity (Precision)
    by_purity = sorted(valid_candidates_pool, key=lambda x: (x['score'], x['confidence']), reverse=True)
    
    # Sort B: High Informedness (Confidence - Penalty)
    # This finds broad "Base" rules (like != Remote)
    by_informedness = sorted(valid_candidates_pool, key=lambda x: (x['confidence'] - x['penalty']), reverse=True)
    
    # Sort C: High Coverage (Confidence) 
    # This grabs the "negative influence" rules that are actually great bases.
    by_coverage = sorted(valid_candidates_pool, key=lambda x: x['confidence'], reverse=True)

    # --- A. GLOBAL CANDIDATES ---
    # Top N from Purity + Top N from Informedness + Top N from Coverage
    global_candidates_list = (
        by_purity[:top_n_atomic_for_conjunction] + 
        by_informedness[:top_n_atomic_for_conjunction] +
        by_coverage[:top_n_atomic_for_conjunction]
    )
    
    #for res in global_candidates_list:
    #    print(f"  Global Candidate: {res['condition']}, Score: {res['score']:.4f}, Conf: {res['confidence']:.4f}, Penalty: {res['penalty']:.4f}")
    
    # Deduplicate (keep the dicts)
    global_top_candidates = []
    seen_globals = set()
    for res in global_candidates_list:
        if res['condition'] not in seen_globals:
            global_top_candidates.append(res)
            seen_globals.add(res['condition'])
            
    # DEFINITION RESTORED: This set is required for Strategy 2 (Augmented Group Search)
    global_top_set = seen_globals 

    # --- B. LOCAL GROUP CANDIDATES ---
    conditions_by_col = defaultdict(list)
    
    # Helper map to organize results by column name
    results_by_col_map = defaultdict(list)
    for res in valid_candidates_pool:
        results_by_col_map[res['condition'].column_name].append(res)

    # PARAMETER DEFINED: How many specific conditions to look at per column
    top_k_per_group_col = 25

    for col, items in results_by_col_map.items():
        # 1. Select Top K Purest (e.g., Location=HQ)
        top_pure = sorted(items, key=lambda x: (x['score'], x['confidence']), reverse=True)[:top_k_per_group_col]
        
        # 2. Select Top 2 Broadest/Best Bases (e.g., Location!=Remote)
        # We slice explicitly here so they aren't lost at the bottom of a sorted list
        top_informed = sorted(items, key=lambda x: (x['confidence'] - x['penalty']), reverse=True)[:top_k_per_group_col]
        
        # Union them to ensure both types are present for this column
        combined_conditions = {res['condition'] for res in top_pure + top_informed}
        conditions_by_col[col] = list(combined_conditions)
    
    # --- C. GENERATE COMBINATIONS ---
    combinations_to_test = set()

    def get_valid_combos(candidate_pool, k):
        """Generator that yields valid condition sets (no duplicate columns)."""
        for combo in combinations(candidate_pool, k):
            cols = {c.column_name for c in combo}
            if len(cols) == k:
                yield frozenset(combo)

    # STRATEGY 1: Pure Global Search
    if len(global_top_candidates) > 1:
        pool = [res['condition'] for res in global_top_candidates]
        for k in [2, 3]:
            for combo_set in get_valid_combos(pool, k):
                combinations_to_test.add(combo_set)

    # STRATEGY 2: Augmented Group Search
    if column_groups:
        for group in column_groups:
            local_pool = []
            for col_name in group:
                # This now contains the Union of Pure + Informed for this column
                local_pool.extend(conditions_by_col[col_name])
            
            if not local_pool:
                continue
            
            # Combine Local specific rules with Global best rules
            augmented_pool = list(set(local_pool) | global_top_set)
            
            for k in [2, 3]:
                if len(augmented_pool) < k: continue
                
                for combo_set in get_valid_combos(augmented_pool, k):
                    if combo_set in combinations_to_test:
                        continue 
                    combinations_to_test.add(combo_set)


    # --- Execute Evaluation ---
    conjunctive_candidates = [ConjunctiveCondition(conds) for conds in combinations_to_test]
    
    if conjunctive_candidates:
        conj_results = _evaluate_conditions_with_cache(
            conjunctive_candidates,
            df_conf_rows, df_conf_pairs, target_vio, vio_mode,
            total_conf_rows, total_conf_pairs, total_vio_checks,
            candidate_score_threshold,
            min_support_rows, 
            cache_conf_rows, cache_vio
        )
        all_results.extend(conj_results)

    # --- Final Filter ---
    final_map = {res['condition']: res for res in all_results if res['score'] > min_score_threshold}
    unique_results = list(final_map.values())
    
    # Apply aggressive redundancy filtering
    filtered_results = _filter_redundant_conditions(unique_results, keep_top_k=2)
    #filtered_results = unique_results
    
    filtered = sorted(filtered_results, key=lambda x: x['score'] + 0.4 * (x['confidence'] - x['penalty']), reverse=True)

    return filtered


# Test function for debugging specific conditions
def evaluate_specific_condition(
    condition: Any,
    violating_samples: Union[List[dict], dict],
    satisfying_samples: Union[List[dict], dict],
    rule_type: str,
    violation_pattern: str = 'between_groups',
    min_support_rows: int = 1
) -> Dict:
    """
    Evaluates a specific condition against provided samples for debugging.
    
    Use this function to understand why a particular condition (e.g., a known
    ground truth) was or wasn't suggested by the mining algorithm. Returns
    detailed metrics for analysis.
    
    Args:
        condition: Any condition object to evaluate.
        violating_samples: Samples where the rule is violated.
        satisfying_samples: Samples where the rule is satisfied.
        rule_type: Either 'single_row_rule' or multi-row type.
        violation_pattern: For multi-row rules, how violations form.
        min_support_rows: Minimum support threshold (default 1 for debugging).
    
    Returns:
        Dictionary containing:
        - 'condition': The input condition
        - 'confidence': Fraction of satisfying samples matching
        - 'penalty': Fraction of violating samples matching
        - 'score': Purity score
        - 'sat_matches': (single-row) Count of satisfying matches
        - 'vio_matches': (single-row) Count of violating matches
        - 'error': (on failure) Error message
    
    Example:
        >>> cond = ColumnValueCondition('status', '=', 'Active')
        >>> result = evaluate_specific_condition(
        ...     cond, violations, satisfying, 'single_row_rule'
        ... )
        >>> print(f"Score: {result['score']:.3f}")
    """
    
    # --- SINGLE ROW LOGIC ---
    if rule_type == 'single_row_rule':
        df_vio = pd.DataFrame(violating_samples)
        df_sat = pd.DataFrame(satisfying_samples)
        df_vio['__is_violating'] = True
        df_sat['__is_violating'] = False
        df_all = pd.concat([df_vio, df_sat], ignore_index=True)
        
        total_satisfying = len(df_sat)
        total_violating = len(df_vio)

        # Helper to recursively calculate mask for single row
        def _get_single_row_mask(df, cond):
            if isinstance(cond, ConjunctiveCondition):
                sub_masks = [_get_single_row_mask(df, c) for c in cond.conditions]
                return np.logical_and.reduce(sub_masks) if sub_masks else np.ones(len(df), dtype=bool)
            
            col = cond.column_name
            if col not in df.columns: return np.zeros(len(df), dtype=bool)
            s = df[col]
            
            if isinstance(cond, ColumnValueCondition):
                # Handle None/NaN
                if cond.value is None: return s.isna()
                return (s == cond.value) if cond.operator == '=' else (s != cond.value)
            
            elif isinstance(cond, NumericalCondition):
                return (s < cond.value) if cond.operator == '<' else (s >= cond.value)
            
            elif isinstance(cond, MergedColumnValueCondition):
                # Using isin
                return s.isin(cond.value_set) if cond.operator == '=' else ~s.isin(cond.value_set)
            
            return np.zeros(len(df), dtype=bool)

        mask = _get_single_row_mask(df_all, condition)
        
        is_violating = df_all['__is_violating'].values
        sat_matches = np.sum(mask & (~is_violating))
        vio_matches = np.sum(mask & is_violating)
        
        conf = sat_matches / total_satisfying if total_satisfying > 0 else 0
        pen = vio_matches / total_violating if total_violating > 0 else 0
        score = _calculate_purity_score(conf, pen, sat_matches, min_support_rows)
        
        return {
            'condition': condition,
            'confidence': conf,
            'penalty': pen,
            'score': score,
            'sat_matches': sat_matches,
            'vio_matches': vio_matches
        }

    # --- MULTI ROW LOGIC ---
    else:
        # Replicate data prep from suggest_conditions_for_multi_row_rule
        df_vio_raw = _flatten_multi_row_samples_to_df(violating_samples)
        df_sat_raw = _flatten_multi_row_samples_to_df(satisfying_samples)
        
        # Add missing columns found in condition but not in df
        def _extract_cols(c):
            if isinstance(c, ConjunctiveCondition):
                s = set()
                for sub in c.conditions: s.update(_extract_cols(sub))
                return s
            return {c.column_name}
        
        req_cols = _extract_cols(condition)
        for df in [df_vio_raw, df_sat_raw]:
            if not df.empty:
                for rc in req_cols: 
                    if rc not in df.columns: df[rc] = None

        # Prepare Confidence Data
        # Assume satisfying_in_violations=False for generic eval, or pass as arg if needed
        df_conf_rows = _prepare_confidence_data(df_sat_raw, df_vio_raw, False)
        total_conf_rows = len(df_conf_rows)
        
        # Prepare Pairwise Confidence Data (Only if needed)
        has_pairwise = False
        def _check_pairwise(c):
            if isinstance(c, PairWiseCondition): return True
            if isinstance(c, ConjunctiveCondition): return any(_check_pairwise(sub) for sub in c.conditions)
            return False
        
        df_conf_pairs = pd.DataFrame()
        total_conf_pairs = 0
        if _check_pairwise(condition):
            df_conf_pairs = _create_pairs_dataframe(df_conf_rows, 'all_combinations')
            total_conf_pairs = len(df_conf_pairs)

        # Prepare Penalty Data
        if violation_pattern == 'whole_group':
            target_vio = df_vio_raw
            vio_mode = 'group'
            total_vio_checks = df_vio_raw['__sample_id'].nunique() if not df_vio_raw.empty else 0
        else:
            target_vio = _create_pairs_dataframe(df_vio_raw, violation_pattern)
            vio_mode = 'pairs'
            total_vio_checks = len(target_vio)

        # Evaluate using the existing vectorized function
        # Set min_score to -1 to ensure it isn't pruned
        results = _evaluate_conditions_with_cache(
            [condition],
            df_conf_rows, df_conf_pairs, target_vio, vio_mode,
            total_conf_rows, total_conf_pairs, total_vio_checks,
            min_score=-1.0, 
            min_support=0, # No support filtering
            cache_conf_rows={}, 
            cache_vio={}
        )
        
        if results:
            return results[0]
        else:
            return {'condition': condition, 'confidence': 0, 'penalty': 0, 'score': 0, 'error': 'Evaluation failed'}