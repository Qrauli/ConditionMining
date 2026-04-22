import asyncio
import time
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import pandas as pd
from src.condition_suggestion import ConjunctiveCondition, _flatten_multi_row_samples_to_df, suggest_conditions_for_multi_row_rule, suggest_conditions_for_single_row_rule
from src.structures import RuleDetail
from src.condition_semantic import analyze_column_utility, analyze_semantic_relevance


@dataclass
class DiscoveryTimingInfo:
    """Timing breakdown for condition discovery phases.
    
    Attributes:
        llm_time: Time spent on LLM queries (semantic analysis + column utility).
        mining_time: Time spent on statistical condition mining.
        total_time: Total time for the entire discovery process.
    """
    llm_time: float = 0.0
    mining_time: float = 0.0
    total_time: float = 0.0

class ConditionDiscoveryEngine:
    """
    Orchestrates the discovery of refinement conditions for data quality rules.
    
    This engine combines statistical analysis with semantic understanding (via LLM)
    to discover conditions that can refine broad rules into more specific, accurate ones.
    It filters columns by semantic relevance and distinguishes between categorical
    columns (useful for value filtering) and identifier columns (useful only for
    pairwise comparisons in multi-row rules).
    
    The discovery pipeline consists of:
    1. Semantic relevance analysis - Identifies which columns are logically related to the rule
    2. Column utility analysis - Classifies columns as 'categorical' or 'identifier'
    3. Statistical condition mining - Finds conditions with high confidence and low penalty
    
    Attributes:
        column_meanings (Dict[str, str]): Maps column names to their semantic descriptions.
        column_groups (List): Groups of related columns for conjunction generation.
        unique_values_per_column (Dict[str, tuple]): Maps column names to (count, [values]).
        use_llm (bool): If False, skips LLM analysis and uses all columns (for ablation studies).
    """
    
    def __init__(self, column_meanings: Dict[str, str], column_groups: List, 
                 unique_values_per_column: Dict[str, tuple], use_llm: bool = True):
        """
        Initialize the ConditionDiscoveryEngine.
        
        Args:
            column_meanings: Dictionary mapping column names to human-readable descriptions
                explaining the semantic meaning of each column.
            column_groups: List of column name lists representing semantically related columns.
                Used as hints for generating conjunctive conditions.
            unique_values_per_column: Dictionary mapping column names to tuples of
                (unique_count, list_of_unique_values). Used for semantic analysis
                and domain size estimation.
            use_llm: Whether to use LLM for semantic filtering. Set to False for ablation
                studies to test the system without semantic pruning.
        """
        self.column_meanings = column_meanings
        self.column_groups = column_groups
        self.unique_values_per_column = unique_values_per_column
        self._column_utility_cache = None
        self.use_llm = use_llm
        

    async def _get_column_utility(self, categorical_cols_to_check: List[str]) -> Dict[str, str]:
        """
        Fetches or computes the column utility classification via LLM.
        
        This method determines whether each categorical column should be treated as:
        - 'categorical': Useful for filtering on specific values (e.g., Status = 'Active')
        - 'identifier': High cardinality column where specific values are meaningless,
          but useful for pairwise comparisons (e.g., t1.ID = t2.ID)
        
        Results are cached to avoid repeated LLM calls.
        
        Args:
            categorical_cols_to_check: List of categorical column names to classify.
        
        Returns:
            Dict mapping column names to their utility type ('categorical' or 'identifier').
            Returns empty dict if LLM call fails.
        """
        if self._column_utility_cache:
            return self._column_utility_cache
        
        # Filter info to only relevant categorical columns to save tokens
        relevant_info = {k: self.column_meanings.get(k, "No description") for k in categorical_cols_to_check}
        relevant_vals = {k: self.unique_values_per_column.get(k, (0, [])) for k in categorical_cols_to_check}

        utility_result = await analyze_column_utility(relevant_info, relevant_vals)
        
        if utility_result:
            # Convert list to dict for fast lookup
            self._column_utility_cache = {
                item.column_name: item.column_type for item in utility_result.columns
            }
        else:
            self._column_utility_cache = {} # Fallback
            
        return self._column_utility_cache
    
    def assess_refinement_potential(self, 
                                  vio_samples: list | dict, 
                                  sat_samples: list | dict, 
                                  rule_type: str,
                                  categorical_columns: List[str],
                                  numerical_columns: List[str],
                                  min_confidence_gain: float = 0.20,
                                  min_group_size: int = 10) -> Tuple[bool, str]:
        """
        Pre-screens whether a rule has refinement potential before running expensive discovery.
        
        This method performs a fast statistical check to determine if any categorical
        value or numerical range creates a subset of data where the rule's confidence
        is significantly higher than the global average. This allows skipping costly
        LLM calls and statistical mining for rules that appear to fail randomly.
        
        Args:
            vio_samples: Samples where the rule is violated. For single-row rules,
                this is a list of dicts. For multi-row rules, this is a nested dict
                structure with group keys and role-based row collections.
            sat_samples: Samples where the rule is satisfied, same structure as vio_samples.
            rule_type: Either 'single_row_rule' or any multi-row rule type identifier.
            categorical_columns: List of categorical column names to check for correlations.
            numerical_columns: List of numerical column names to check for range splits.
            min_confidence_gain: Minimum improvement in confidence required to consider
                refinement worthwhile. Default 0.20 (20% improvement).
            min_group_size: Minimum number of samples required in a subset to avoid
                statistical noise from small groups. Default 10.
        
        Returns:
            Tuple of (should_refine, reason):
            - should_refine: True if a subset with significantly higher confidence was found.
            - reason: Human-readable explanation of the decision.
        
        Example:
            >>> should_refine, reason = engine.assess_refinement_potential(
            ...     vio_samples=violations, sat_samples=satisfying,
            ...     rule_type='single_row_rule',
            ...     categorical_columns=['status', 'department'],
            ...     numerical_columns=['amount']
            ... )
            >>> if should_refine:
            ...     conditions = await engine.discover_conditions_for_rule(...)
        """
        # 1. Prepare Data
        if rule_type == 'single_row_rule':
            df_vio = pd.DataFrame(vio_samples)
            df_sat = pd.DataFrame(sat_samples)
        else:
            # Reuse the flattener from condition_suggestion
            df_vio = _flatten_multi_row_samples_to_df(vio_samples)
            df_sat = _flatten_multi_row_samples_to_df(sat_samples)

        if df_vio.empty and df_sat.empty:
            return False, "No data samples"
        
        # Label and Merge
        df_vio['__rule_valid'] = 0
        df_sat['__rule_valid'] = 1
        
        # Align columns
        all_cols = set(categorical_columns + numerical_columns)
        for c in all_cols:
            if c not in df_vio.columns: df_vio[c] = None
            if c not in df_sat.columns: df_sat[c] = None
            
        df_all = pd.concat([df_vio, df_sat], ignore_index=True)
        
        global_confidence = df_all['__rule_valid'].mean()
        
        # If rule is already > 95% perfect, probably doesn't need basic refinement
        if global_confidence > 0.95:
            return False, f"Rule is already high confidence ({global_confidence:.2f})"
            
        best_gain = 0.0
        best_feature = None
        best_val_desc = None

        # 2. Check Categorical Correlations
        for col in categorical_columns:
            if col not in df_all.columns: continue
            
            # Group by value and calc mean validity (confidence) & count
            stats = df_all.groupby(col)['__rule_valid'].agg(['mean', 'count'])
            
            # Filter for small groups to avoid statistical noise
            valid_groups = stats[stats['count'] >= min_group_size]
            
            if valid_groups.empty: continue
            
            max_group_conf = valid_groups['mean'].max()
            gain = max_group_conf - global_confidence
            
            if gain > best_gain:
                best_gain = gain
                best_feature = col
                best_val = valid_groups['mean'].idxmax()
                best_val_desc = f"{col} == '{best_val}'"

        # 3. Check Numerical Correlations (Binning)
        for col in numerical_columns:
            if col not in df_all.columns: continue
            
            # Drop NaNs for binning
            series = df_all[col].dropna()
            if len(series) < min_group_size * 2: continue
            
            try:
                # Create 4 bins (Quartiles). 
                # 'duplicates="drop"' handles cases where many rows have value 0
                df_all[f'{col}_bin'] = pd.qcut(df_all[col], q=4, duplicates='drop')
                
                stats = df_all.groupby(f'{col}_bin', observed=True)['__rule_valid'].agg(['mean', 'count'])
                valid_groups = stats[stats['count'] >= min_group_size]
                
                if valid_groups.empty: continue
                
                max_group_conf = valid_groups['mean'].max()
                gain = max_group_conf - global_confidence
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = col
                    best_val_desc = f"{col} range {valid_groups['mean'].idxmax()}"
            except Exception as e:
                # qcut can fail on weird distributions or single values
                continue

        # 4. Decision
        if best_gain >= min_confidence_gain:
            reason = (f"Found subset '{best_val_desc}' with confidence "
                      f"{global_confidence + best_gain:.2f} (Global: {global_confidence:.2f})")
            return True, reason
        
        return False, f"Failures appear random (Max Gain: {best_gain:.2f})"
    """
    for rule_result in failed_rules:
    # 1. Check if the rule is structurally refinable (has potential)
    # This runs purely on CPU (statistical check), no LLM cost yet
    should_refine, reason = engine.assess_refinement_potential(
        vio_samples=rule_result.violation_samples,
        sat_samples=rule_result.satisfying_samples,
        rule_type=rule_result.type,
        categorical_columns=list(dataset.categorical_columns),
        numerical_columns=list(dataset.numerical_columns)
    )
    
    if should_refine:
        print(f"Selecting Rule {rule_result.id} for refinement. Reason: {reason}")
        # 2. NOW run the expensive discovery process
        new_conditions = await engine.discover_conditions_for_rule(...)
    else:
        print(f"Skipping Rule {rule_result.id}. {reason}")
    """
    async def discover_conditions_for_rule(self, rule_detail: RuleDetail, 
                                           detailed_rule_type: str,
                                           vio_samples, sat_samples, 
                                           rule_type: str, categorical_columns: dict, 
                                           numerical_columns: List,
                                           special_values: dict) -> Tuple[List[dict], DiscoveryTimingInfo]:
        """
        Main entry point for discovering refinement conditions for a data quality rule.
        
        This method orchestrates the full condition discovery pipeline:
        1. Semantic Analysis (LLM): Identifies which columns are logically relevant
        2. Utility Classification (LLM): Separates identifiers from categorical columns
        3. Statistical Mining: Finds conditions with high satisfying match rate and low violating match rate
        
        Args:
            rule_detail: The RuleDetail object containing the rule text and metadata.
            detailed_rule_type: Specific rule type classification (e.g., 'functional dependency
                checking rules', 'unique key constraint rules'). Used to determine the
                violation pattern for multi-row rules.
            vio_samples: Samples where the rule is violated.
            sat_samples: Samples where the rule is satisfied.
            rule_type: General type - either 'single_row_rule' or a multi-row type.
            categorical_columns: Dict mapping column names to sets of observed values.
            numerical_columns: List of numerical column names for threshold conditions.
            special_values: Dict mapping column names to special values (e.g., NULL, N/A)
                that should also be considered as potential condition values.
        
        Returns:
            Tuple of (conditions, timing_info):
            - conditions: List of condition dictionaries, each containing:
              - 'condition': The Condition object (Atomic, Merged, Conjunctive, etc.)
              - 'confidence': Fraction of satisfying samples matching the condition
              - 'penalty': Fraction of violating samples matching the condition
              - 'score': Purity score combining confidence and penalty
            - timing_info: DiscoveryTimingInfo with breakdown of LLM and mining times.
        
        Note:
            If use_llm=False, all semantic filtering is skipped and all columns are
            used for condition mining (useful for ablation studies).
        """
        total_start = time.time()
        timing = DiscoveryTimingInfo()
        
        # Phase 1: Semantic Analysis (skip if LLM is disabled for ablation study)
        llm_start = time.time()
        if self.use_llm:
            # ; Explanation: {rule_detail.explanation}
            semantic_task  = analyze_semantic_relevance(f"Rule: {rule_detail.rule}; Explanation: {rule_detail.explanation}", self.column_meanings, self.unique_values_per_column)
            #print(f'Semantic analysis result for rule "{rule_detail.rule}": {semantic_analysis_result}')
            
            utility_task = self._get_column_utility(list(categorical_columns.keys()))
            
            # Run concurrently
            semantic_result, utility_map = await asyncio.gather(semantic_task, utility_task)
            
            #print(semantic_result)
            #print(utility_map)
            relevant_cols_set = set()
            if semantic_result:
                for item in semantic_result.relevant_columns:
                    if item.relevance_category != 'unrelated':
                        relevant_cols_set.add(item.column_name)
            else:
                # Fallback if LLM fails: assume all are relevant
                relevant_cols_set = set(categorical_columns.keys()) | set(numerical_columns)
        else:
            # Ablation mode: Skip LLM, treat all columns as relevant
            relevant_cols_set = set(categorical_columns.keys()) | set(numerical_columns)
            utility_map = {}  # Empty map means all categorical cols treated as 'categorical' type
        
        timing.llm_time = time.time() - llm_start
        
        # B. Apply Utility Constraints (ID vs Category)
        final_constrained_categorical = {}
        final_only_pairwise_cols = []
        
        # Process Categorical Columns
        for col, values in categorical_columns.items():
            if col not in relevant_cols_set:
                continue
                
            col_type = utility_map.get(col, 'categorical') # Default to categorical if unknown
            
            if col_type == 'identifier':
                # If it's an Identifier:
                # 1. It is NEVER used for single value checks (removed from categorical dict)
                # 2. It is added to pairwise checks ONLY if rule is multi-row
                if rule_type != 'single_row_rule':
                    final_only_pairwise_cols.append(col)
            else:
                # If it's a Category:
                # Keep it for value checking
                final_constrained_categorical[col] = values
        
        final_constrained_numerical = [c for c in numerical_columns if c in relevant_cols_set]
        
        # Process Special Values
        final_constrained_special = {k: v for k, v in special_values.items() if k in relevant_cols_set}
        
        #print(f'Constrained categorical columns: {final_constrained_categorical}')
        #print(f'Constrained numerical columns: {final_constrained_numerical}')
        
        mining_start = time.time()
        if rule_type == 'single_row_rule':
            statistical_conditions = suggest_conditions_for_single_row_rule(
                violating_samples=vio_samples,
                satisfying_samples=sat_samples,
                column_groups=self.column_groups,
                categorical_columns=final_constrained_categorical, # Use the constrained dict
                numerical_columns=final_constrained_numerical,
                special_values=final_constrained_special,
            )

        else:
            detailed_rule_type = detailed_rule_type.lower()
            violation_pattern = "between_groups"
            if detailed_rule_type == "functional dependency checking rules":
                violation_pattern = "between_groups"
            elif detailed_rule_type == "cross-attribute checking rules":
                violation_pattern = "between_groups"
            elif detailed_rule_type == "monotonic amount rules":
                violation_pattern = "between_groups"
            elif detailed_rule_type == "sum amount comparison rules":
                violation_pattern = "between_groups"
            elif detailed_rule_type == "sum amount to threshold comparison rules":
                violation_pattern = "whole_group"
            elif detailed_rule_type == "unique key constraint rules":
                violation_pattern = "within_groups"
            elif detailed_rule_type == "temporal order validation rules":
                violation_pattern = "between_groups"
            elif detailed_rule_type == "monotonic relationship rules":
                violation_pattern = "between_groups"
            
             
            statistical_conditions = suggest_conditions_for_multi_row_rule(
                violating_samples=vio_samples,
                satisfying_samples=sat_samples,
                column_groups=self.column_groups,
                violation_pattern=violation_pattern,
                categorical_columns=final_constrained_categorical,
                numerical_columns=final_constrained_numerical,
                special_values=final_constrained_special,
                only_pairwise_columns=final_only_pairwise_cols
            )

        timing.mining_time = time.time() - mining_start
        timing.total_time = time.time() - total_start
        
        # Phase 3: Hybrid Scoring
        ranked_atomic_conditions = statistical_conditions
        return ranked_atomic_conditions, timing
        """ranked_atomic_conditions = self.calculate_hybrid_scores(
            statistical_conditions, 
            semantic_map, 
            self.weights
        )"""
        """
        top_candidates = statistical_conditions[:20]

        # --- PHASE 4: Semantic Validity Check (The "Sense" Check) ---
        print(f"Semantically verifying top {len(top_candidates)} conditions...")
        
        # 1. Identify columns used in the RULE itself
        # rule_detail.columns is likely a list/set of strings.
        rule_columns = set(rule_detail.columns) if rule_detail.columns else set()
        
        verification_tasks = []
        
        for cond_detail in top_candidates:
            condition_obj = cond_detail['condition']
            
            # 2. Identify columns used in this specific CONDITION
            cond_columns = self._extract_columns_from_condition(condition_obj)
            
            # 3. Combine them to get full context
            all_involved_columns = rule_columns.union(cond_columns)
            
            # 4. Build the Context Dictionary (Name -> Meaning)
            # Fallback to "No description" if meaningful string is missing
            context_dict = {}
            for col in all_involved_columns:
                desc = self.column_meanings.get(col, "No description provided.")
                context_dict[col] = desc
            
            # 5. Schedule Verification
            cond_str = str(condition_obj)
            verification_tasks.append(
                verify_condition_semantically(
                    original_rule=rule_detail.rule,
                    suggested_condition=cond_str,
                    columns_context=context_dict
                )
            )
            
        # Run validations in parallel
        verdicts = await asyncio.gather(*verification_tasks)
        
        final_verified_conditions = []
        for cond_detail, verdict in zip(top_candidates, verdicts):
            if verdict.is_valid:
                # Optionally append the reasoning to the result for transparency
                cond_detail['semantic_verdict'] = verdict.reason
                final_verified_conditions.append(cond_detail)
            else:
                print(f"REJECTED ({verdict.verdict_type}): {cond_detail['condition']} | Reason: {verdict.reason}")

        # Return the surviving conditions (e.g., top 10)
        return final_verified_conditions[:10]
    """
    def _extract_columns_from_condition(self, condition_obj) -> set:
        """
        Recursively extracts all column names referenced in a condition object.
        
        This utility method traverses condition objects of any type (atomic, merged,
        conjunctive) and collects the set of all column names they reference.
        Used for building context dictionaries for semantic verification.
        
        Args:
            condition_obj: A condition object of any type (ColumnValueCondition,
                NumericalCondition, MergedColumnValueCondition, ConjunctiveCondition,
                PairWiseCondition).
        
        Returns:
            Set of column name strings referenced by the condition.
        """
        cols = set()
        if isinstance(condition_obj, ConjunctiveCondition):
            for sub_cond in condition_obj.conditions:
                cols.update(self._extract_columns_from_condition(sub_cond))
        elif hasattr(condition_obj, 'column_name'):
            # Atomic, Pairwise, Numerical, Merged all have column_name
            cols.add(condition_obj.column_name)
        return cols
    
"""    def calculate_hybrid_scores(self, statistical_conditions: List[Dict],
                                semantic_map: Dict[str, str], # e.g., {'department_id': 'highly_relevant', ...}
                                weights: Dict[str, float] = {'semantic': 0.2, 'stat': 0.8}
                                ) -> List[Dict]:
        
        Scores and ranks conditions based on a hybrid of statistical and semantic metrics.
        
        
        # 1. Define semantic scores
        semantic_score_map = {
            "highly_relevant": 1.0,
            "moderately_relevant": 0.6,
            "unrelated": 0.1
        }

        scored_conditions = []
        for cond_detail in statistical_conditions:
            condition = cond_detail['condition']
            column_name = condition.column_name

            # 2. Get semantic score
            relevance_category = semantic_map.get(column_name, "unrelated")
            s_semantic = semantic_score_map[relevance_category]

            # 3. Get statistical score (and normalize)
            s_stat_raw = cond_detail.get('score', 0.0) # confidence - penalty
            s_stat_normalized = (s_stat_raw + 1) / 2.0

            # 4. Calculate combined score
            s_combined = (weights['semantic'] * s_semantic) + (weights['stat'] * s_stat_normalized)
            
            # Append all scores for full transparency
            new_cond_detail = cond_detail.copy()
            new_cond_detail['semantic_score'] = s_semantic
            new_cond_detail['stat_score_normalized'] = s_stat_normalized
            new_cond_detail['hybrid_score'] = s_combined
            scored_conditions.append(new_cond_detail)
            
        # 5. Sort by the final hybrid score
        scored_conditions.sort(key=lambda x: x['hybrid_score'], reverse=True)
        
        return scored_conditions
"""
# --- Example Usage ---
# engine = ConditionDiscoveryEngine(...)
# top_conditions = await engine.discover_conditions_for_rule(...)
# for cond in top_conditions:
#    print(f"Condition: {cond['condition']} | Score: {cond['hybrid_score']:.2f} | Semantics: {cond['semantic_score']:.2f}")