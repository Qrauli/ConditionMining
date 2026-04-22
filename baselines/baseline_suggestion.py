import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, _tree
from typing import List, Dict
from src.condition_suggestion import (
    ConjunctiveCondition,
    NumericalCondition,
    ColumnValueCondition,
    _flatten_multi_row_samples_to_df,
    _create_pairs_dataframe,
    _calculate_purity_score
)

class DecisionTreeBaseline:
    def __init__(self, max_depth: int = 3, min_samples_leaf: int = 5):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.dt_clf = None

    def suggest_conditions(self, 
                          violating_samples: List[dict], 
                          satisfying_samples: List[dict],
                          rule_type: str = 'single_row_rule',
                          numerical_columns: List[str] = None,
                          categorical_columns: Dict[str, list] = None) -> List[Dict]:
        
        # 1. Prepare Data
        if rule_type == 'single_row_rule':
            df, X, y, feature_map = self._prepare_single_row_data(
                violating_samples, satisfying_samples, numerical_columns, categorical_columns
            )
        else:
            df, X, y, feature_map = self._prepare_multi_row_data(
                violating_samples, satisfying_samples, numerical_columns, categorical_columns
            )
            
        if X is None or len(X) == 0:
            return []

        # --- FIX 1: Check for single-class data ---
        # If y only contains 0s or only contains 1s, a tree cannot find a splitting condition.
        if len(np.unique(y)) < 2:
            return []

        # 2. Train Decision Tree
        self.dt_clf = DecisionTreeClassifier(
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            criterion='entropy',
            random_state=42
        )
        self.dt_clf.fit(X, y)

        # 3. Extract Rules
        rules = self._extract_rules_from_tree(X, y, feature_map)
        
        # 4. Filter and Rank
        ranked_rules = self._rank_rules(rules, len(satisfying_samples), len(violating_samples))
        
        return ranked_rules

    def _extract_rules_from_tree(self, X, y, feature_map):
        tree_ = self.dt_clf.tree_
        
        # --- FIX 2: Dynamic Class Index Mapping ---
        # self.dt_clf.classes_ contains the unique labels sorted. 
        # It could be [0, 1], or just [0], or just [1] (though we handled single-class above).
        classes = self.dt_clf.classes_
        class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
        
        # Get array index for Satisfying (0) and Violating (1)
        idx_sat = class_to_idx.get(0) 
        idx_vio = class_to_idx.get(1) 

        rules = []
        
        def recurse(node, current_conditions):
            # If leaf node
            if tree_.feature[node] == _tree.TREE_UNDEFINED:
                # counts is an array of weighted class counts in this node
                counts = tree_.value[node][0]
                
                # Safely retrieve counts. If a class doesn't exist in the tree, count is 0.
                sat_count = counts[idx_sat] if idx_sat is not None else 0
                vio_count = counts[idx_vio] if idx_vio is not None else 0
                
                # We are interested in leaves that are predominantly Satisfying
                if sat_count > vio_count and sat_count > 0:
                    # Ignore root node with no conditions (trivial rule)
                    if current_conditions:
                        rules.append({
                            'conditions': current_conditions,
                            'sat_matches': sat_count,
                            'vio_matches': vio_count
                        })
                return

            # Branching logic (Same as before)
            feat_idx = tree_.feature[node]
            threshold = tree_.threshold[node]
            feat_info = feature_map.get(feat_idx)
            
            if not feat_info: return

            # Left Child (Feature <= Threshold)
            left_cond = self._create_condition(feat_info, threshold, go_left=True)
            if left_cond:
                recurse(tree_.children_left[node], current_conditions + [left_cond])
            
            # Right Child (Feature > Threshold)
            right_cond = self._create_condition(feat_info, threshold, go_left=False)
            if right_cond:
                recurse(tree_.children_right[node], current_conditions + [right_cond])

        recurse(0, [])
        return rules

    # ... Include _prepare_single_row_data, _prepare_multi_row_data, 
    # _encode_features, _create_condition, _rank_rules from previous code ...
    
    # (Here are the helper methods again for completeness if you are pasting the whole file)
    def _prepare_single_row_data(self, vio_samples, sat_samples, num_cols, cat_cols):
        df_vio = pd.DataFrame(vio_samples)
        df_sat = pd.DataFrame(sat_samples)
        
        df_vio['__label'] = 1 
        df_sat['__label'] = 0 
        
        df = pd.concat([df_vio, df_sat], ignore_index=True).fillna(np.nan)
        return self._encode_features(df, num_cols, cat_cols)

    def _prepare_multi_row_data(self, vio_samples, satisfying_samples, num_cols, cat_cols):
        df_vio_raw = _flatten_multi_row_samples_to_df(vio_samples)
        df_sat_raw = _flatten_multi_row_samples_to_df(satisfying_samples)
        
        df_vio_pairs = _create_pairs_dataframe(df_vio_raw, 'between_groups')
        df_sat_pairs = _create_pairs_dataframe(df_sat_raw, 'between_groups') 
        
        if df_sat_pairs.empty and not df_sat_raw.empty:
            df_sat_pairs = _create_pairs_dataframe(df_sat_raw, 'all_combinations')

        df_vio_pairs['__label'] = 1
        df_sat_pairs['__label'] = 0
        
        df = pd.concat([df_vio_pairs, df_sat_pairs], ignore_index=True)
        
        flat_num_cols = []
        flat_cat_cols = {}
        
        if num_cols:
            for c in num_cols:
                flat_num_cols.extend([f"{c}_1", f"{c}_2"])
        if cat_cols:
            for c, vals in cat_cols.items():
                flat_cat_cols[f"{c}_1"] = vals
                flat_cat_cols[f"{c}_2"] = vals

        return self._encode_features(df, flat_num_cols, flat_cat_cols)

    def _encode_features(self, df, num_cols, cat_cols):
        X_parts = []
        feature_map = {} 
        
        if num_cols:
            valid_num = [c for c in num_cols if c in df.columns]
            if valid_num:
                X_num = df[valid_num].fillna(0).values
                start_idx = 0
                for i, col in enumerate(valid_num):
                    feature_map[start_idx + i] = {'col': col, 'type': 'num'}
                X_parts.append(X_num)

        if cat_cols:
            valid_cat = [c for c in cat_cols.keys() if c in df.columns]
            if valid_cat:
                X_cat_df = pd.get_dummies(df[valid_cat], dummy_na=True, prefix_sep='###')
                start_idx = sum(x.shape[1] for x in X_parts)
                for i, col_name in enumerate(X_cat_df.columns):
                    if '###' in col_name:
                        orig_col, val = col_name.split('###', 1)
                        try:
                            if float(val).is_integer(): val = int(float(val))
                        except: pass
                    else:
                        orig_col, val = col_name, 1
                    feature_map[start_idx + i] = {'col': orig_col, 'type': 'cat', 'val': val}
                X_parts.append(X_cat_df.values.astype(float))

        if not X_parts:
            return df, None, None, {}

        X = np.concatenate(X_parts, axis=1)
        y = df['__label'].values
        return df, X, y, feature_map

    def _create_condition(self, feat_info, threshold, go_left):
        col_name = feat_info['col']
        if feat_info['type'] == 'num':
            op = '<=' if go_left else '>'
            if op == '<=':
                return NumericalCondition(col_name, '<', threshold + 1e-9) 
            else:
                return NumericalCondition(col_name, '>=', threshold + 1e-9)
        elif feat_info['type'] == 'cat':
            target_val = feat_info['val']
            if go_left:
                return ColumnValueCondition(col_name, '!=', target_val)
            else:
                return ColumnValueCondition(col_name, '=', target_val)
        return None

    def _rank_rules(self, raw_rules, total_sat, total_vio):
        formatted_results = []
        for r in raw_rules:
            conds = r['conditions']
            conds = [c for c in conds if c is not None]
            if not conds: continue
            
            if len(conds) == 1:
                final_cond = conds[0]
            else:
                final_cond = ConjunctiveCondition(frozenset(conds))
                
            sat_local = r['sat_matches']
            vio_local = r['vio_matches']
            
            confidence = sat_local / total_sat if total_sat > 0 else 0
            penalty = vio_local / total_vio if total_vio > 0 else 0
            score = _calculate_purity_score(confidence, penalty, sat_local, self.min_samples_leaf)
            
            formatted_results.append({
                'condition': final_cond,
                'confidence': confidence,
                'penalty': penalty,
                'score': score
            })
            
        return sorted(formatted_results, key=lambda x: x['score'], reverse=True)