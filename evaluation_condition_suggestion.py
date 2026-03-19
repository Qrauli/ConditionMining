import argparse
import asyncio
import concurrent
import gc
import time
import numpy as np
import pandas as pd
from baseline_suggestion import DecisionTreeBaseline
from ruleforge.rule_compiling.utils import get_entities_based_on_identifiers
from ruleforge.rule_refinement.condition_discovery import ConditionDiscoveryEngine
from ruleforge.rule_refinement.condition_suggestion import (
    ConjunctiveCondition,
    NumericalCondition,
    _flatten_multi_row_samples_to_df,
    evaluate_specific_condition,
    suggest_conditions_for_multi_row_rule,
    suggest_conditions_for_single_row_rule)
from ruleforge.rule_refinement.utils import df_to_list_of_dicts
from ruleforge.structures import RuleDetail
from ruleforge.main_func import _read_cache
from ruleforge.controller.main_workflow import SingleTypeRuleForge
from ruleforge.data_understanding.sampling import sample, hierarchical_informative_sample, hierarchical_informative_sample_rows
from ruleforge.utils.utils import read_csv

import logging
logging.disable(logging.CRITICAL)
import warnings
warnings.filterwarnings("ignore")

# Parse command-line arguments for ablation study
parser = argparse.ArgumentParser(description='Evaluate condition suggestion system')
parser.add_argument('--no-llm', action='store_true', 
                    help='Disable LLM-based semantic filtering (ablation mode: test all columns)')
args = parser.parse_args()

USE_LLM = not args.no_llm
if not USE_LLM:
    print("=== ABLATION MODE: LLM disabled, using ALL columns ===")

def calculate_mask(df: pd.DataFrame, condition: NumericalCondition) -> pd.Series:
    """Calculates boolean mask for a numerical condition on the dataframe."""
    col_data = df[condition.column_name]
    if condition.operator == '<':
        return col_data < condition.value
    elif condition.operator == '>=':
        return col_data >= condition.value
    return pd.Series([False] * len(df))

def are_numerical_conditions_equivalent(df: pd.DataFrame, cond1: NumericalCondition, cond2: NumericalCondition, threshold: float = 0.99) -> bool:
    """
    Checks if two numerical conditions are effectively equivalent by comparing 
    the rows they select (Intersection over Union).
    """
    if cond1.column_name != cond2.column_name:
        return False
    if cond1.operator != cond2.operator:
        return False
    
    # If values are extremely close, we can skip mask calc (optimization)
    if abs(cond1.value - cond2.value) < 1e-9:
        return True

    mask1 = calculate_mask(df, cond1)
    mask2 = calculate_mask(df, cond2)
    
    intersection = (mask1 & mask2).sum()
    union = (mask1 | mask2).sum()
    
    if union == 0:
        return True # Both select nothing, so they are equivalent
        
    iou = intersection / union
    return iou >= threshold

def check_condition_match(df: pd.DataFrame, found_cond, original_cond) -> bool:
    """
    Compares found condition vs original condition.
    Uses masks for Numerical, strict equality for others.
    """
    # 1. Type mismatch check
    if type(found_cond) != type(original_cond):
        return False

    # 2. Numerical Condition Strategy (Mask based)
    if isinstance(found_cond, NumericalCondition):
        return are_numerical_conditions_equivalent(df, found_cond, original_cond)

    # 3. Conjunctive Condition Strategy
    elif isinstance(found_cond, ConjunctiveCondition):
        found_subs = list(found_cond.conditions)
        orig_subs = list(original_cond.conditions)
        
        if len(found_subs) != len(orig_subs):
            return False
            
        # We need to match every found sub-condition to a unique original sub-condition
        # We make a copy of orig_subs to "consume" them as we match
        remaining_orig = orig_subs[:]
        
        for f_sub in found_subs:
            match_found = False
            for i, o_sub in enumerate(remaining_orig):
                is_match = False
                
                # Check specific sub-types
                if isinstance(f_sub, NumericalCondition) and isinstance(o_sub, NumericalCondition):
                    if are_numerical_conditions_equivalent(df, f_sub, o_sub):
                        is_match = True
                elif f_sub == o_sub:
                    # Strict equality for ColumnValue, Merged, PairWise
                    is_match = True
                
                if is_match:
                    remaining_orig.pop(i)
                    match_found = True
                    break
            
            if not match_found:
                return False
                
        return True

    # 4. Fallback: Strict Equality (ColumnValue, Merged, Pairwise)
    else:
        return found_cond == original_cond

def get_violation_pattern(rule_type_str):
    rt = rule_type_str.lower()
    if rt == "sum amount to threshold comparison rules":
        return "whole_group"
    elif rt == "unique key constraint rules":
        return "within_groups"
    else:
        return "between_groups"

datasets = [
    #('Ecommerce-500000', "examples/datasets/Ecommerce-500000.csv"),
    #('Ecommerce-1000', "examples/datasets/Ecommerce-1000.csv"),
    #('Ecommerce-10000', "examples/datasets/Ecommerce-10000.csv"),
    #('Ecommerce-1000000', "examples/datasets/Ecommerce-1000000.csv"),
    #('Ecommerce', "examples/datasets/Ecommerce.csv"),
    #('SupplyChain', "examples/datasets/SupplyChain.csv"),
    #('University', "examples/datasets/University.csv"),
    #('RealEstate', "examples/datasets/RealEstate.csv"),
    #('Insurance', "examples/datasets/Insurance.csv"),
    #('Manufacturing', "examples/datasets/Manufacturing.csv"),
    #('Airline', "examples/datasets/Airline.csv"),
    #('Healthcare', "examples/datasets/Healthcare.csv"),
    #('Banking', "examples/datasets/Banking.csv"),
    #('HR', "examples/datasets/HR.csv"),
    ('hospital', "examples/datasets/hospital.csv")
    ]

rule_types = [
    "functional dependency checking rules"
    #"cross-attribute checking rules",
    #"sum amount comparison rules",
    #"sum amount to threshold comparison rules",
    #"unique key constraint rules"         
]
total_conditions_found = 0
total_rules_processed = 0
conditions_in_top_1 = 0
conditions_in_top_3 = 0
conditions_in_top_5 = 0
conditions_in_top_10 = 0
reciprocal_ranks = []
rule_times = []  # Track per-rule processing times
sampling_times = []  # Track per-rule sampling times
llm_times = []  # Track per-rule LLM query times
mining_times = []  # Track per-rule statistical mining times
total_start_time = time.time()  # Track total evaluation time
for dataset_name, dataset_path in datasets:
    data_understanding = _read_cache(dataset_name)
    categorical_columns = data_understanding.categorical_columns
    column_meanings = data_understanding.column_meanings
    column_groups = data_understanding.column_groups
    special_values = data_understanding.special_values
    df = read_csv(dataset_path)
    
    # extract information about unique values per column and sample values
    unique_values_per_column = {col: (df[col].nunique(dropna=True), df[col].dropna().unique()[:15].tolist()) for col in df.columns}

    all_columns = df.columns.tolist()
    non_categorical_columns = [col for col in all_columns if col not in list(categorical_columns.keys())]
    numerical_columns = df[non_categorical_columns].select_dtypes(include=np.number).columns.tolist()
    
    discovery_engine = ConditionDiscoveryEngine(column_meanings=column_meanings,
                                        column_groups=column_groups,
                                        unique_values_per_column=unique_values_per_column,
                                        use_llm=USE_LLM)
    
    for rule_type in rule_types:
        forge = SingleTypeRuleForge(
            dataset_name=dataset_name,
            knowledge_based_rule_type=rule_type,
            dataset_path=dataset_path,
            data_understanding=data_understanding,
            kb_path="knowledge_base/kb_fd.json",
        )
        rules = forge._cache_load_all_rules(rule_type)
        total_rules_processed += len(rules)
        for rid, candidaterule in rules.items():
            print(f'Processing rule id {rid} of type {rule_type} from dataset {dataset_name}...')
            #if rid != 0:
            #if (rid == 1 and dataset_name == 'SupplyChain' and rule_type == 'functional dependency checking rules') \
            #or (rid == 1 and dataset_name == 'RealEstate' and rule_type == 'functional dependency checking rules') \
            #or (rid == 1 and dataset_name == 'University' and rule_type == 'sum amount to threshold comparison rules') \
            #or (rid == 0 and dataset_name == 'RealEstate' and rule_type == 'sum amount to threshold comparison rules') \
            #or (rid == 0 and dataset_name == 'Insurance' and rule_type == 'functional dependency checking rules') \
            #or (rid == 1 and dataset_name == 'Insurance' and rule_type == 'functional dependency checking rules') \
            #or (rid == 1 and dataset_name == 'Banking' and rule_type == 'functional dependency checking rules'):

                #pass
                #
                #total_rules_processed -= 1
            #    reciprocal_ranks.append(0.0)
            #    continue
            ruledetail = candidaterule.rule
            satisfactions = candidaterule.execution_result['satisfactions']
            violations = candidaterule.execution_result['violations']

            sampling_start_time = time.time()
            if isinstance(satisfactions, (set, list)) and all(isinstance(identifier, (str, int)) for identifier in satisfactions):
                all_sat_entities = get_entities_based_on_identifiers(df, satisfactions,
                                                               index_column_name=None) 
                sat_samples = sample(
                    dataset=all_sat_entities,
                    k=100000,
                    informative=True,
                    categorical_columns=list(categorical_columns.keys())
                )
            
                all_vio_entities = get_entities_based_on_identifiers(df, violations,
                                                                    index_column_name=None)

                vio_samples = sample(
                    dataset=all_vio_entities,
                    k=100000,
                    informative=True,
                    categorical_columns=list(categorical_columns.keys())
                )

            elif isinstance(satisfactions, dict):

                sat_samples = hierarchical_informative_sample_rows(
                    dictionary=satisfactions,
                    k=100000,
                    df=df,
                    categorical_columns=list(categorical_columns.keys()),
                    cn=5,
                    min_rows_per_role=1,
                    force_full_output_if_small=True
                )
                # print("Obtained satisfying samples via hierarchical informative sampling.")
                vio_samples = hierarchical_informative_sample_rows(
                    dictionary=violations,
                    k=100000,
                    df=df,
                    categorical_columns=list(categorical_columns.keys()),
                    cn=5,
                    min_rows_per_role=1,
                    force_full_output_if_small=True
                )
            sampling_elapsed_time = time.time() - sampling_start_time
            sampling_times.append(sampling_elapsed_time)
    
            
            if isinstance(sat_samples, pd.DataFrame):
                _rule_type = 'single_row_rule'
                satisfying_samples = df_to_list_of_dicts(sat_samples)
            else:
                satisfying_samples = sat_samples
                _rule_type = 'multi_row_rule'
            
            if isinstance(vio_samples, pd.DataFrame):
                violating_samples = df_to_list_of_dicts(vio_samples)
            else:
                violating_samples = vio_samples
            
            rule_start_time = time.time()  # Start timing this rule
            
            coro = discovery_engine.discover_conditions_for_rule(
                rule_detail=ruledetail,
                detailed_rule_type=rule_type,
                rule_type=_rule_type,
                vio_samples=violating_samples,
                sat_samples=satisfying_samples,
                categorical_columns=categorical_columns,
                numerical_columns=numerical_columns,
                special_values=special_values
            )
            try:
                # Preferred: run directly when no event loop is running
                conditions, timing_info = asyncio.run(coro)
            except RuntimeError:
                # If an event loop is already running (e.g., in an async app or test runner),
                # run the coroutine in a separate thread using its own event loop.
                def _run():
                    return asyncio.run(coro)

                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as exc:
                    future = exc.submit(_run)
                    conditions, timing_info = future.result()
            
            rule_elapsed_time = time.time() - rule_start_time  # End timing this rule
            rule_times.append(rule_elapsed_time)
            llm_times.append(timing_info.llm_time)
            mining_times.append(timing_info.mining_time)
            print(conditions[:5])    
            
            """
            dt_baseline = DecisionTreeBaseline(max_depth=3) # Depth 3 ≈ max 3 atomic conditions

            conditions = dt_baseline.suggest_conditions(
                violating_samples=violating_samples,
                satisfying_samples=satisfying_samples,
                rule_type=_rule_type,
                numerical_columns=numerical_columns,
                categorical_columns=categorical_columns # Pass the raw dict {col: [vals]}
            )
            """
            
            # Now `conditions` holds the suggested conditions for the rule        
            
            # Check if condition was intended+
            original_condition = candidaterule.condition
            found_condition = False
            found_position = -1
            if original_condition:
                for i, suggested in enumerate(conditions):
                    suggestion_cond_obj = suggested['condition']
                    
                    # Use the new robust checker
                    is_match = check_condition_match(df, suggestion_cond_obj, original_condition)
                    
                    if is_match:
                        print(f'Found the intended condition: "{suggestion_cond_obj}" at position {i+1} for rule: "{ruledetail.rule}".')
                        print(f'   [Score: {suggested["score"]:.4f}, Confidence: {suggested["confidence"]:.4f}, Penalty: {suggested["penalty"]:.4f}]')
                        total_conditions_found += 1
                        found_condition = True
                        found_position = i
                        break
            
            # Track position statistics
            if found_condition:
                if found_position < 1:
                    conditions_in_top_1 += 1
                if found_position < 3:
                    conditions_in_top_3 += 1
                if found_position < 5:
                    conditions_in_top_5 += 1
                if found_position < 10:
                    conditions_in_top_10 += 1
                # Add reciprocal rank (position is 0-indexed, so rank is position + 1)
                reciprocal_ranks.append(1.0 / (found_position + 1))
            else:
                # If not found, reciprocal rank is 0
                reciprocal_ranks.append(0.0)
            
            if not found_condition:
                print(f'Intended condition: "{original_condition}" not found among suggestions for rule: "{ruledetail.rule}"')
                try:
                    # Determine pattern for multi-row rules
                    v_pattern = get_violation_pattern(rule_type)
                    
                    # Force evaluation of the missing condition
                    debug_metrics = evaluate_specific_condition(
                        condition=original_condition,
                        violating_samples=violating_samples,
                        satisfying_samples=satisfying_samples,
                        rule_type=_rule_type,
                        violation_pattern=v_pattern
                    )
                    
                    print(f"   [DEBUG METRICS] for missing condition:")
                    print(f"   Score:      {debug_metrics.get('score', 0):.4f}")
                    print(f"   Confidence: {debug_metrics.get('confidence', 0):.4f}")
                    print(f"   Penalty:    {debug_metrics.get('penalty', 0):.4f}")
                    
                except Exception as e:
                    print(f"   [DEBUG ERROR] Could not evaluate missing condition: {e}")
        del forge
        gc.collect()
        
total_elapsed_time = time.time() - total_start_time

print(f'\n=== Evaluation Results ===')
print(f'Total conditions found: {total_conditions_found} out of {total_rules_processed} rules processed.')
print(f'Conditions in top 1: {conditions_in_top_1} ({100*conditions_in_top_1/total_rules_processed:.1f}%)')
print(f'Conditions in top 3: {conditions_in_top_3} ({100*conditions_in_top_3/total_rules_processed:.1f}%)')
print(f'Conditions in top 5: {conditions_in_top_5} ({100*conditions_in_top_5/total_rules_processed:.1f}%)')
print(f'Conditions in top 10: {conditions_in_top_10} ({100*conditions_in_top_10/total_rules_processed:.1f}%)')
mrr = np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0
print(f'Mean Reciprocal Rank (MRR): {mrr:.4f}')

print(f'\n=== Runtime Statistics ===')
print(f'Total runtime: {total_elapsed_time:.2f}s ({total_elapsed_time/60:.2f} min)')
if rule_times:
    avg_time = np.mean(rule_times)
    median_time = np.median(rule_times)
    min_time = np.min(rule_times)
    max_time = np.max(rule_times)
    std_time = np.std(rule_times)
    print(f'Average time per rule: {avg_time:.4f}s')
    print(f'Median time per rule: {median_time:.4f}s')
    print(f'Min time per rule: {min_time:.4f}s')
    print(f'Max time per rule: {max_time:.4f}s')
    print(f'Std dev: {std_time:.4f}s')

if sampling_times:
    avg_s_time = np.mean(sampling_times)
    median_s_time = np.median(sampling_times)
    min_s_time = np.min(sampling_times)
    max_s_time = np.max(sampling_times)
    std_s_time = np.std(sampling_times)
    print(f'\n=== Sampling Statistics ===')
    print(f'Average sampling time: {avg_s_time:.4f}s')
    print(f'Median sampling time: {median_s_time:.4f}s')
    print(f'Min sampling time: {min_s_time:.4f}s')
    print(f'Max sampling time: {max_s_time:.4f}s')
    print(f'Std dev: {std_s_time:.4f}s')

if llm_times:
    avg_llm_time = np.mean(llm_times)
    median_llm_time = np.median(llm_times)
    min_llm_time = np.min(llm_times)
    max_llm_time = np.max(llm_times)
    std_llm_time = np.std(llm_times)
    total_llm_time = np.sum(llm_times)
    print(f'\n=== LLM Query Statistics ===')
    print(f'Total LLM time: {total_llm_time:.2f}s ({100*total_llm_time/total_elapsed_time:.1f}% of total)')
    print(f'Average LLM time per rule: {avg_llm_time:.4f}s')
    print(f'Median LLM time per rule: {median_llm_time:.4f}s')
    print(f'Min LLM time per rule: {min_llm_time:.4f}s')
    print(f'Max LLM time per rule: {max_llm_time:.4f}s')
    print(f'Std dev: {std_llm_time:.4f}s')

if mining_times:
    avg_mining_time = np.mean(mining_times)
    median_mining_time = np.median(mining_times)
    min_mining_time = np.min(mining_times)
    max_mining_time = np.max(mining_times)
    std_mining_time = np.std(mining_times)
    total_mining_time = np.sum(mining_times)
    print(f'\n=== Statistical Mining Statistics ===')
    print(f'Total mining time: {total_mining_time:.2f}s ({100*total_mining_time/total_elapsed_time:.1f}% of total)')
    print(f'Average mining time per rule: {avg_mining_time:.4f}s')
    print(f'Median mining time per rule: {median_mining_time:.4f}s')
    print(f'Min mining time per rule: {min_mining_time:.4f}s')
    print(f'Max mining time per rule: {max_mining_time:.4f}s')
    print(f'Std dev: {std_mining_time:.4f}s')
