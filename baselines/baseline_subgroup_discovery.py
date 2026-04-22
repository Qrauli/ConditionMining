import pandas as pd
import warnings
from typing import List, Dict

try:
    import pysubgroup as ps
except ImportError:
    ps = None
    warnings.warn("pysubgroup is not installed. Please install it using `pip install pysubgroup`.")

from src.condition_suggestion import (
    ConjunctiveCondition,
    NumericalCondition,
    ColumnValueCondition,
    _flatten_multi_row_samples_to_df,
    _create_pairs_dataframe,
    _calculate_purity_score
)

class SubgroupDiscoveryBaseline:
    """
    Subgroup Discovery Baseline using `pysubgroup`.
    Uses Beam Search to find subsets enriched in Satisfying samples.
    """
    def __init__(self, max_depth: int = 3, result_set_size: int = 20, beam_width: int = 20, max_cardinality: int = 50):
        self.max_depth = max_depth
        self.result_set_size = result_set_size
        self.beam_width = beam_width
        self.max_cardinality = max_cardinality

    def suggest_conditions(self, 
                          violating_samples: List[dict], 
                          satisfying_samples: List[dict],
                          rule_type: str = 'single_row_rule',
                          numerical_columns: List[str] = None,
                          categorical_columns: Dict[str, list] = None) -> List[Dict]:
        
        if ps is None:
            raise ImportError("pysubgroup is required. Run `pip install pysubgroup`.")

        # 1. Prepare Data (Explicitly cast target to Boolean for pysubgroup)
        if rule_type == 'single_row_rule':
            df = self._prepare_single_row_data(violating_samples, satisfying_samples)
        else:
            df = self._prepare_multi_row_data(violating_samples, satisfying_samples)
            
        if df is None or df.empty or len(df['__label'].unique()) < 2:
            print("[SD Baseline] Dataframe is empty or has only 1 class. Aborting.")
            return []

        # Target columns mapping
        valid_cols = []
        if numerical_columns: valid_cols.extend(numerical_columns)
        if categorical_columns: valid_cols.extend(list(categorical_columns.keys()))
        
        if rule_type != 'single_row_rule':
            flat_valid_cols = []
            for c in valid_cols:
                flat_valid_cols.extend([f"{c}_1", f"{c}_2"])
            valid_cols = flat_valid_cols
            
        # 2. Filter Search Space (Drop high cardinality columns to prevent SD explosion)
        keep_cols = ['__label']
        dropped_cols = []
        for c in df.columns:
            if c in valid_cols:
                if pd.api.types.is_numeric_dtype(df[c]):
                    keep_cols.append(c)
                elif df[c].nunique() <= self.max_cardinality:
                    keep_cols.append(c)
                else:
                    dropped_cols.append(c)

        df = df[keep_cols]
        
        if len(keep_cols) == 1:
            print(f"[SD Baseline] All features were dropped due to cardinality > {self.max_cardinality}. Search space is empty!")
            return []

        # 3. Setup Subgroup Discovery Task
        # Target: __label == True (which represents Satisfying/0 from original logic)
        target = ps.BinaryTarget('__label', True)
        
        # Limit numeric bins to 5 to prevent continuous variables from exploding the search space
        searchspace = ps.create_selectors(df, ignore=['__label'], nbins=5)
        
        if not searchspace:
            print("[SD Baseline] pysubgroup failed to create any selectors from the dataframe.")
            return []

        task = ps.SubgroupDiscoveryTask(
            df,
            target,
            searchspace,
            result_set_size=self.result_set_size,
            depth=self.max_depth,
            qf=ps.WRAccQF()
        )
        
        # Run Beam Search
        result = ps.BeamSearch(beam_width=self.beam_width).execute(task)
        
        if not result.results:
            print("[SD Baseline] Beam Search executed but returned 0 results.")
            return []

        # 4. Extract, Map, and Score Rules
        total_sat = len(df[df['__label'] == True])
        total_vio = len(df[df['__label'] == False])
        
        ranked_rules = self._extract_rules_and_score(result, total_sat, total_vio, df)
        
        if not ranked_rules:
            print("[SD Baseline] Rules were found, but they mapped to empty conditions (e.g., Universal rule).")
            
        return ranked_rules

    def _prepare_single_row_data(self, vio_samples, sat_samples):
        df_vio = pd.DataFrame(vio_samples)
        df_sat = pd.DataFrame(sat_samples)
        
        # True = Satisfying (Target), False = Violating
        if not df_vio.empty: df_vio['__label'] = False 
        if not df_sat.empty: df_sat['__label'] = True 
        
        return pd.concat([df_vio, df_sat], ignore_index=True)

    def _prepare_multi_row_data(self, vio_samples, satisfying_samples):
        df_vio_raw = _flatten_multi_row_samples_to_df(vio_samples)
        df_sat_raw = _flatten_multi_row_samples_to_df(satisfying_samples)
        
        df_vio_pairs = _create_pairs_dataframe(df_vio_raw, 'between_groups')
        df_sat_pairs = _create_pairs_dataframe(df_sat_raw, 'between_groups') 
        
        if df_sat_pairs.empty and not df_sat_raw.empty:
            df_sat_pairs = _create_pairs_dataframe(df_sat_raw, 'all_combinations')

        if not df_vio_pairs.empty: df_vio_pairs['__label'] = False
        if not df_sat_pairs.empty: df_sat_pairs['__label'] = True
        
        if df_vio_pairs.empty and df_sat_pairs.empty:
            return None
            
        return pd.concat([df_vio_pairs, df_sat_pairs], ignore_index=True)

    def _parse_selector(self, root_selector):
        """
        Iterative parser (Stack-based) to prevent RecursionError.
        Translates pysubgroup selectors to Custom Condition objects robustly.
        """
        conds = []
        stack = [root_selector]
        visited = set()
        
        while stack:
            if len(conds) > 20: 
                break
                
            selector = stack.pop()
            
            sel_id = id(selector)
            if sel_id in visited:
                continue
            visited.add(sel_id)
            
            cname = type(selector).__name__
            
            if cname == 'Conjunction':
                subs = getattr(selector, 'selectors', getattr(selector, '_selectors', []))
                stack.extend(subs)
                continue
                
            if cname == 'EqualitySelector':
                col = getattr(selector, 'attribute_name', None)
                val = getattr(selector, 'attribute_value', None)
                if col is not None:
                    if col.endswith('_1') or col.endswith('_2'):
                        col = col[:-2]
                        
                    if pd.isna(val):
                        conds.append(ColumnValueCondition(col, '=', None))
                    else:
                        conds.append(ColumnValueCondition(col, '=', val))
                continue
                
            if cname == 'IntervalSelector':
                col = getattr(selector, 'attribute_name', None)
                lb = getattr(selector, 'lower_bound', float('-inf'))
                ub = getattr(selector, 'upper_bound', float('inf'))
                
                if col is not None:
                    if col.endswith('_1') or col.endswith('_2'):
                        col = col[:-2]
                        
                    if lb > float('-inf'):
                        conds.append(NumericalCondition(col, '>=', float(lb)))
                    if ub < float('inf'):
                        conds.append(NumericalCondition(col, '<', float(ub)))
                continue
                
            if cname == 'TrueSelector' or str(selector) == 'True':
                continue
                
            print(f"[SD Baseline Debug] Unrecognized selector type: {cname} -> {selector}")
                
        return list(set(conds))

    def _extract_rules_and_score(self, sd_result, total_sat, total_vio, df):
        rules = []
        is_sat = df['__label'] == True
        is_vio = df['__label'] == False
        
        # Safely handle different pysubgroup versions
        for res_tuple in sd_result.results:
            sg = res_tuple[1]
            description = sg.subgroup_description if hasattr(sg, 'subgroup_description') else sg
            
            try:
                # Use covers() to get boolean mask natively
                if hasattr(sg, 'covers'):
                    mask = sg.covers(df)
                else:
                    mask = description.covers(df)
            except Exception as e:
                # Sometimes complex rules fail to evaluate natively on weird DFs
                print(f"[SD Baseline] Failed to apply mask for a rule: {e}")
                continue
            
            sat_matches = (mask & is_sat).sum()
            vio_matches = (mask & is_vio).sum()
            
            # Map into Condition objects
            conds = self._parse_selector(description)
            
            # If conds is empty, this means it's the "Universal Rule" (all rows), skip it
            if not conds: 
                continue
                
            if len(conds) == 1:
                final_cond = conds[0]
            else:
                final_cond = ConjunctiveCondition(frozenset(conds))
                
            confidence = sat_matches / total_sat if total_sat > 0 else 0
            penalty = vio_matches / total_vio if total_vio > 0 else 0
            
            score = _calculate_purity_score(confidence, penalty, sat_matches, min_support=5)
            
            rules.append({
                'condition': final_cond,
                'confidence': confidence,
                'penalty': penalty,
                'score': score
            })
            
        return sorted(rules, key=lambda x: x['score'], reverse=True)