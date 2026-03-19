import streamlit as st
import pandas as pd
import asyncio
import os
import tempfile
import logging
from typing import Set

# --- RuleForge Imports ---
from ruleforge.controller.main_workflow import DataUnderstander
from ruleforge.structures import RuleDetail, CandidateRule
from ruleforge.rule_refinement.condition_discovery import ConditionDiscoveryEngine
from ruleforge.rule_compiling import convert_rules_to_python_programs, run_source_codes_in_parallel
from ruleforge.rule_refinement.utils import df_to_list_of_dicts
from ruleforge.rule_compiling.utils import get_entities_based_on_identifiers
from ruleforge.data_understanding.sampling import sample, hierarchical_informative_sample, hierarchical_informative_sample_rows

# Configure Logging to show in console
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Helper Functions ---

def save_uploaded_file(uploaded_file):
    """Saves uploaded streamlit file to a temp file and returns the path."""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            return tmp_file.name
    except Exception as e:
        st.error(f"Error saving file: {e}")
        return None

def get_violation_pattern(rule_type_str):
    """Logic extracted from evaluation script to determine grouping."""
    rt = rule_type_str.lower()
    if rt == "sum amount to threshold comparison rules":
        return "whole_group"
    elif rt == "unique key constraint rules":
        return "within_groups"
    else:
        return "between_groups"

# --- Main Application ---

st.set_page_config(page_title="RuleForge Condition Discovery", layout="wide")
st.title("🛠️ RuleForge: Condition Discovery Engine")

# 1. Sidebar: Data & Configuration
with st.sidebar:
    st.header("Data Input")
    uploaded_file = st.file_uploader("Upload Dataset (CSV)", type=['csv'])
    
    dataset_name = st.text_input("Dataset Name", value="MyDataset")
    
    # Initialize Session State for Data Understanding
    if 'data_understanding' not in st.session_state:
        st.session_state['data_understanding'] = None
    if 'dataset_path' not in st.session_state:
        st.session_state['dataset_path'] = None

    if uploaded_file:
        # Save file to temp path because RuleForge expects a file path
        if st.session_state['dataset_path'] is None:
            path = save_uploaded_file(uploaded_file)
            st.session_state['dataset_path'] = path
        
        st.info("File uploaded successfully.")
        
        # Trigger Data Understanding
        if st.button("Run Data Understanding"):
            with st.spinner("Analyzing dataset columns and values (this calls LLMs)..."):
                try:
                    du_engine = DataUnderstander(
                        dataset_name=dataset_name,
                        dataset_path=st.session_state['dataset_path']
                    )
                    # We run the understanding process
                    du_result = du_engine.understand(use_existing_desc=False)
                    st.session_state['data_understanding'] = du_result
                    st.success("Data Understanding Complete!")
                except Exception as e:
                    st.error(f"Data Understanding Failed: {e}")
                    logger.exception(e)

# 2. Main Area: Rule Definition & Processing
if st.session_state['data_understanding'] is None:
    st.warning("Please upload a dataset and run 'Data Understanding' in the sidebar to proceed.")
else:
    du = st.session_state['data_understanding']
    
    # Show summary of understanding
    with st.expander("View Data Understanding Summary"):
        st.write("**Categorical Columns detected:**", list(du.categorical_columns.keys()))
        st.write("**Column Meanings:**", du.column_meanings)

    st.header("Define Rule without Condition")
    
    col1, col2 = st.columns(2)
    
    with col1:
        rule_text = st.text_area("Rule Text", 
                                 placeholder="e.g., The delivery_date must be later than order_date",
                                 help="Enter the base rule logic here.")
        explanation = st.text_area("Explanation", 
                                   placeholder="e.g., An order cannot be delivered before it is placed.",
                                   height=100)

    with col2:
        # User must match standard rule types for the logic to work best
        rule_type_options = [
            "functional dependency checking rules",
            "cross-attribute checking rules",
            "sum amount comparison rules",
            "sum amount to threshold comparison rules",
            "unique key constraint rules",
            "temporal order validation rules",
            "monotonic relationship rules",
            "single_row_rule" # Generic fallback
        ]
        
        selected_rule_type = st.selectbox("Rule Type", rule_type_options)
        
        # Column selection is important for RuleDetail
        all_cols = list(du.column_meanings.keys())
        selected_columns = st.multiselect("Relevant Columns", all_cols, 
                                          help="Select columns involved in this rule.")

    if st.button("🚀 Discover Conditions"):
        if not rule_text or not selected_columns:
            st.error("Please provide Rule Text and select Relevant Columns.")
        else:
            # --- PHASE 1: Construct Rule Object ---
            st.markdown("### 1. Generating & Executing Rule Code")
            status_container = st.empty()
            
            with st.spinner("Converting rule to executable Python code..."):
                try:
                    # Construct RuleDetail
                    rule_obj = RuleDetail(
                        rule=rule_text,
                        explanation=explanation,
                        columns=set(selected_columns)
                    )
                    
                    # Wrap in CandidateRule (ID 0 for this temp session)
                    c_rule = CandidateRule(
                        rule_id=0,
                        rule_type=selected_rule_type,
                        rule=rule_obj
                    )
                    
                    # Use RuleForge compiler
                    # Note: We pass a list containing our single rule
                    compile_res = convert_rules_to_python_programs(
                        candidate_rules=[c_rule],
                        dataset=pd.read_csv(st.session_state['dataset_path']), # Pass DF
                        return_generators=False
                    )
                    
                    if not compile_res or 'rules_and_codes' not in compile_res:
                        st.error("Failed to generate code for this rule.")
                        st.stop()
                        
                    rule_and_code = compile_res['rules_and_codes'][0]
                    generated_code = rule_and_code['code']
                    
                    with st.expander("View Generated Python Code"):
                        st.code(generated_code, language='python')

                except Exception as e:
                    st.error(f"Error during code generation: {e}")
                    logger.exception(e)
                    st.stop()

            # --- PHASE 2: Execution ---
            with st.spinner("Executing rule on dataset..."):
                try:
                    # Execute
                    exec_results = run_source_codes_in_parallel(
                        rule_ids=[0],
                        source_codes=[generated_code],
                        function_names=['execute_rule0'],
                        dataset_path=st.session_state['dataset_path'],
                        metadata_file_path=None,
                        num_workers=1
                    )
                    
                    # Handle result extraction
                    result_data = exec_results[0]
                    if 'error' in result_data:
                        st.error(f"Execution Error: {result_data['error']}")
                        st.stop()
                        
                    satisfactions = result_data['satisfactions']
                    violations = result_data['violations']
                    
                    st.success(f"Execution complete. Found {len(violations)} violations and {len(satisfactions)} satisfying rows/groups.")
                    
                except Exception as e:
                    st.error(f"Error during execution: {e}")
                    logger.exception(e)
                    st.stop()

            # --- PHASE 3: Sampling ---
            st.markdown("### 2. Sampling & Condition Discovery")
            with st.spinner("Sampling and mining conditions..."):
                try:
                    df = pd.read_csv(st.session_state['dataset_path'])
                    categorical_columns = du.categorical_columns
                    
                    # Logic adapted from evaluation_condition_suggestion.py
                    sat_samples = None
                    vio_samples = None
                    
                    # Case A: List of identifiers (Single Row Rule Logic usually)
                    if isinstance(satisfactions, (set, list)) and len(satisfactions) > 0:
                        all_sat_entities = get_entities_based_on_identifiers(df, list(satisfactions), index_column_name=None)
                        sat_samples = sample(all_sat_entities, k=1000, informative=True, categorical_columns=list(categorical_columns.keys()))
                        
                        all_vio_entities = get_entities_based_on_identifiers(df, list(violations), index_column_name=None)
                        vio_samples = sample(all_vio_entities, k=1000, informative=True, categorical_columns=list(categorical_columns.keys()))

                    # Case B: Dict (Multi-row Logic usually)
                    elif isinstance(satisfactions, dict):
                        sat_samples = hierarchical_informative_sample_rows(
                            dictionary=satisfactions, k=100000, df=df, 
                            categorical_columns=list(categorical_columns.keys()), 
                            cn=5,
                            min_rows_per_role=1,
                            force_full_output_if_small=True
                        )
                        vio_samples = hierarchical_informative_sample_rows(
                            dictionary=violations, k=100000, df=df, 
                            categorical_columns=list(categorical_columns.keys()), 
                            cn=5,
                            min_rows_per_role=1,
                            force_full_output_if_small=True
                        )
                    
                    # Convert to list of dicts for engine
                    _rule_type_tag = 'single_row_rule'
                    if isinstance(sat_samples, pd.DataFrame):
                        satisfying_samples = df_to_list_of_dicts(sat_samples)
                        violating_samples = df_to_list_of_dicts(vio_samples)
                    else:
                        # If hierarchical sampling returned complex structures
                        satisfying_samples = sat_samples
                        violating_samples = vio_samples
                        _rule_type_tag = 'multi_row_rule'

                    # Prepare Data for Discovery Engine
                    unique_values_per_column = {col: (df[col].nunique(dropna=True), df[col].dropna().unique()[:15].tolist()) for col in df.columns}
                    all_columns = df.columns.tolist()
                    non_categorical_columns = [col for col in all_columns if col not in list(categorical_columns.keys())]
                    numerical_columns = df[non_categorical_columns].select_dtypes(include=['number']).columns.tolist()

                    discovery_engine = ConditionDiscoveryEngine(
                        column_meanings=du.column_meanings,
                        column_groups=du.column_groups,
                        unique_values_per_column=unique_values_per_column
                    )
                    
                    # Run Async Discovery
                    async def run_discovery():
                        return await discovery_engine.discover_conditions_for_rule(
                            rule_detail=rule_obj,
                            detailed_rule_type=selected_rule_type,
                            rule_type=_rule_type_tag,
                            vio_samples=violating_samples,
                            sat_samples=satisfying_samples,
                            categorical_columns=categorical_columns,
                            numerical_columns=numerical_columns,
                            special_values=du.special_values
                        )

                    conditions = asyncio.run(run_discovery())
                    
                except Exception as e:
                    st.error(f"Error during condition discovery: {e}")
                    logger.exception(e)
                    st.stop()

            # --- PHASE 4: Display Results ---
            st.markdown("### 3. Suggested Conditions")
            
            if not conditions:
                st.warning("No conditions were suggested.")
            else:
                results_data = []
                for idx, item in enumerate(conditions[0]):
                    # item structure depends on condition_discovery.py return
                    # usually contains 'condition', 'score', 'confidence', etc.
                    cond_obj = item.get('condition', str(item))
                    score = item.get('score', 0)
                    confidence = item.get('confidence', 0)
                    penalty = item.get('penalty', 0)
                    
                    results_data.append({
                        "Rank": idx + 1,
                        "Condition": str(cond_obj),
                        "Score": f"{score:.4f}",
                        "Confidence": f"{confidence:.4f}",
                        "Penalty": f"{penalty:.4f}"
                    })
                
                res_df = pd.DataFrame(results_data[:100])
                st.dataframe(res_df, use_container_width=True)