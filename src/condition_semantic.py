"""Semantic analysis module for enriching data quality rule refinement with LLM intelligence.

This module provides LLM-based analysis functions that bridge statistical condition mining
with semantic understanding. It enables:

1. **Semantic Relevance Analysis** - Identifies which columns are logically related to a rule
2. **Column Utility Classification** - Distinguishes categorical columns from identifiers
3. **Condition Verification** - Validates statistically-mined conditions for semantic sense

All functions use structured LLM outputs (via Pydantic models) for reliable parsing.
Functions are async for efficient parallel execution.
"""
from typing import List, Optional

from pydantic import BaseModel, Field
from typing import List, Dict

from src.utils import create_model_for_process, invoke_with_retries


class ColumnRelevance(BaseModel):
    """
    Represents the semantic relevance assessment of a single column.
    
    Output by the LLM when analyzing which columns could meaningfully
    refine a data quality rule.
    
    Attributes:
        column_name: Name of the column being evaluated.
        relevance_category: One of 'highly_relevant', 'moderately_relevant', 'unrelated'.
        justification: LLM's explanation for the assigned category.
    """
    column_name: str = Field(..., description="The name of the column being evaluated.")
    relevance_category: str = Field(
        ...,
        description="The category of relevance for this column.",
        enum=["highly_relevant", "moderately_relevant", "unrelated"]
    )
    justification: str = Field(..., description="A brief justification for the assigned category.")

class RuleRelevanceAnalysis(BaseModel):
    """
    Container for semantic relevance analysis of multiple columns.
    
    Returned by analyze_semantic_relevance() containing the LLM's assessment
    of how each candidate column relates to a specific data quality rule.
    
    Attributes:
        relevant_columns: List of ColumnRelevance objects, one per analyzed column.
    """
    relevant_columns: List[ColumnRelevance] = Field(
        ..., description="A list of columns with their semantic relevance to the rule."
    )
    
async def analyze_semantic_relevance(
    original_rule: str,
    candidate_columns_info: Dict[str, str], # e.g., {'col_name': 'description'}
    unique_values_per_column: Dict[str, tuple]
    ) -> Optional[RuleRelevanceAnalysis]:
    """
    Uses an LLM to analyze the semantic relevance of candidate columns for refining a data quality rule.
    This version explicitly instructs the model on different types of conditions (partitioning, filtering, etc.).
    """

    prompt_template = """
    You are an expert data analyst. Your task is to evaluate how semantically relevant a list of candidate columns are for refining a given data quality rule.
    A "refining" column is one that can form a logical condition to define a subgroup of data where the rule is more likely to apply consistently.

    **Data Quality Rule:**
    "{original_rule}"

    **Candidate Columns and their Descriptions:**
    {columns_info_formatted}

    ---
    **Types of Refining Conditions**

    A column can be used to refine a rule in several ways. Evaluate each column's potential to form any of the following types of conditions:

    1.  **Partitioning (Grouping):** The rule applies only to rows that have or do not have the *same value* in this column. Only possible for multi-row rules where the rule compares multiple rows.
        *Example*: "If two employees have the same 'department_id', then their salaries should be comparable."

    2.  **Filtering (Inclusion):** The rule applies only to rows where this column has a *specific value*.
        *Example*: "If the 'transaction_type' is 'SALE', then the 'tax_amount' must be calculated."

    3.  **Exclusion:** The rule applies only to rows where this column does *not* have a specific value.
        *Example*: "If the 'account_status' is NOT 'suspended', then the user must have a valid email."

    4.  **Thresholding:** (For numerical/ordinal columns) The rule applies only to rows where the column's value is *above or below a certain threshold*.
        *Example*: "If 'patient_age' is over 65, then a different 'treatment_protocol' applies."

    ---
    **Analysis Task**

    For each candidate column, determine its relevance for refining the rule by forming one or more of the condition types above. Categorize it as:
    - `highly_relevant`: The column has a strong, direct logical connection. Conditioning on it is very likely to create a meaningful and accurate version of the rule.
    - `moderately_relevant`: The column has a plausible connection. It might define useful subgroups, but the link isn't as direct. If there is even the slightest logical connection, consider at least moderately relevant.
    - `unrelated`: The column has no logical connection to the rule's context.

    Provide a justification for each choice.
    
    - If a list of values is provided for a column, use them to judge if they define meaningful business subgroups.
    - Many columns already appear in the rule; in that case, consider if additional conditioning on them (e.g., specific values) could still help. This is heavily context-dependent could be either highly relevant or unrelated.
    Especially if the column is already checked to be a fixed value in the rule, further conditioning on it is highly likely unrelated. E.g. if the rule states "If ... then status = 'active'", conditioning on 'status' again is likely unrelated and might make the rule trivially true.
    Furthermore for something like a functional dependency rule conditioning on either the determinant or dependent column is often unrelated unless there is a strong contextual reason. Especially restricting the dependent column to a specific value makes the rule trivially true and should NEVER be considered highly relevant.
    - Be cautious with assigning `unrelated`. Use it when there is clearly no logical way the column could help refine the rule. Think of real-world scenarios where the column's values might still define meaningful subgroups, even if they might seem unrelated there could be some sort of hidden/weird connection. "Unrelated" will discard the column from consideration and therefore the right condition might be missed if you are too strict.
    - Sometimes a column may not seem directly related to the columns or concepts in the rule, but could still define meaningful subgroups based on domain knowledge. Consider this when evaluating relevance.
    
    **CRITICAL WARNING: Triviality & Redundancy**
    You must identify columns that are *functionally identical* or *direct prerequisites* to the rule itself.
    - If the rule is "Salary must be > 0", then the column "Salary" is TRIVIAL. Creating a condition "Salary > 0" is tautological. Mark this as `unrelated`.
    - If the rule is "Start Date < End Date", conditioning on "Start Date" is usually redundant unless strictly partitioning.
    ---
    
    **Examples of Correct Analysis**

    ✅ **Example 1 (Partitioning)**
    Rule: "Products with the same 'product_code' should have the same 'unit_price'."
    Column: `supplier_id` (The ID of the supplier)
    Output:
      - column_name: 'supplier_id'
        relevance_category: 'highly_relevant'
        justification: "Different suppliers may have different pricing contracts for the same product. This column is ideal for **partitioning** the data, so the rule is checked per-supplier."

    ✅ **Example 2 (Filtering)**
    Rule: "All transactions must have a non-zero 'tax_amount'."
    Column: `transaction_type` (Values like 'SALE', 'REFUND', 'VOID')
    Output:
      - column_name: 'transaction_type'
        relevance_category: 'highly_relevant'
        justification: "Tax likely only applies to sales. This column is perfect for **filtering** the rule to apply only when transaction_type = 'SALE'."

    ✅ **Example 3 (Exclusion)**
    Rule: "All active projects must have an assigned 'project_manager'."
    Column: `project_status` (Values like 'planning', 'active', 'completed', 'on_hold')
    Output:
      - column_name: 'project_status'
        relevance_category: 'highly_relevant'
        justification: "The rule is about 'active' projects. We can use this column for both **filtering** (status = 'active') and **exclusion** (status != 'completed')."

    ✅ **Example 4 (Unrelated)**
    Rule: "AreaCode uniquely determines City and State."
    Column: `HasChild` (Whether a person has children)
    Output:
      - column_name: 'HasChild'
        relevance_category: 'unrelated'
        justification: "A person's family status has no logical connection to geographical data like Area Codes. This column cannot form a meaningful condition for this rule."
    ---

    Now, perform the analysis for the given rule and columns.
    """
    
    

    formatted_lines = []
    MAX_values_TO_SHOW = 10 # Threshold: if fewer than this, show all. If more, show count.

    for name, desc in candidate_columns_info.items():
        base_info = f"- **{name}**: {desc}"
        
        # Check if we have categorical data for this column
        if name in unique_values_per_column and unique_values_per_column[name]:
            values = unique_values_per_column[name][1]
            count = unique_values_per_column[name][0]
            
            if count <= MAX_values_TO_SHOW:
                # Convert values to string representation
                val_str = ", ".join(str(v) for v in values)
                extra_info = f"\n    *   *Values ({count}):* [{val_str}]"
            else:
                # Show count and the first few as examples
                examples = ", ".join(str(v) for v in list(values)[:5])
                extra_info = f"\n    *   *Values Info:* {count} unique values (examples: {examples}, ...)"
            
            formatted_lines.append(base_info + extra_info)
        else:
            # No categorical info or empty list
            formatted_lines.append(base_info)

    columns_info_str = "\n".join(formatted_lines)
    
    model = create_model_for_process('validity_check')
    prompt_content = prompt_template.format(
        original_rule=original_rule,
        columns_info_formatted=columns_info_str
    )
    
    # Optional: Log or save the prompt for debugging
    # with open('semantic_relevance_prompt.txt', 'w') as f:
    #     f.write(prompt_content)
        
    structured_llm = model.with_structured_output(RuleRelevanceAnalysis)
    
    try:
        result = invoke_with_retries(structured_llm, prompt_content, timeout_seconds=60)
        return result
    except Exception as e:
        return None
    
    
class ColumnUtility(BaseModel):
    """
    Represents the utility classification of a single column.
    
    Distinguishes between categorical columns (useful for value filtering)
    and identifier columns (high cardinality, only useful for pairwise comparisons).
    
    Attributes:
        column_name: Name of the column.
        column_type: Either 'categorical' or 'identifier'.
        reason: LLM's explanation for the classification.
    """
    column_name: str = Field(..., description="The name of the column.")
    column_type: str = Field(
        ...,
        description="The functional type of the column.",
        enum=["categorical", "identifier"]
    )
    reason: str = Field(..., description="Why this column is classified as such.")

class ColumnUtilityAnalysis(BaseModel):
    """
    Container for column utility classification results.
    
    Returned by analyze_column_utility() containing the LLM's classification
    of each column as either categorical or identifier type.
    
    Attributes:
        columns: List of ColumnUtility objects, one per analyzed column.
    """
    columns: List[ColumnUtility] = Field(..., description="Classification of columns.")
    
async def analyze_column_utility(
    categorical_columns_info: Dict[str, str], # name -> description
    unique_values_per_column: Dict[str, tuple] # name -> (count, [values])
) -> Optional[ColumnUtilityAnalysis]:
    """
    Uses an LLM to classify columns as categorical or identifier types.
    
    This analysis determines how each column should be used in condition mining:
    - **categorical**: Limited meaningful values defining business subgroups.
        Useful for filtering rules like "If Status = 'Active'..."
    - **identifier**: High cardinality with arbitrary specific values.
        Useless for value filtering, but useful for pairwise comparisons.
    
    Args:
        categorical_columns_info: Dict mapping column names to semantic descriptions.
        unique_values_per_column: Dict mapping column names to (unique_count, [sample_values]).
            The count and sample values help the LLM assess cardinality.
    
    Returns:
        ColumnUtilityAnalysis containing classifications for all columns,
        or None if the LLM call fails.
    
    Example:
        >>> result = await analyze_column_utility(
        ...     {'department': 'The department name', 'employee_id': 'Unique employee ID'},
        ...     {'department': (5, ['Sales', 'HR', 'IT']), 'employee_id': (1000, ['E001', 'E002'])}
        ... )
        >>> # result.columns[0].column_type == 'categorical'
        >>> # result.columns[1].column_type == 'identifier'
    """
    
    prompt_template = """
    You are a Data Architect. Your task is to classify a list of columns based on their statistical and semantic nature to determine how they should be used in data quality rules.

    **Task:**
    For each column, determine if it is:
    1. **`categorical`**: A column with a limited set of meaningful values that define business subgroups. 
       - *Usage:* Useful for rules like "If Status = 'Active'..."
       - *Examples:* Country, Currency, Status, Department, Product_Category.
    
    2. **`identifier`**: A column that distinguishes specific entities or has high cardinality where specific values are arbitrary.
       - *Usage:* Pointless to filter on specific values (e.g., "If ID = 502..."), but useful for comparing two rows (e.g., "If t1.ID = t2.ID").
       - *Examples:* UUIDs, Transaction IDs, Phone Numbers, Emails, Serial Numbers, or names that act like IDs (e.g., "Subdivision_25", "User_992").
       - *Note:* Even if a column is text (string), if it has hundreds of unique values like "Area_1", "Area_2", treat it as an `identifier`.

    **Input Data:**
    {columns_info_formatted}

    **Output:**
    Classify every column provided.
    """

    formatted_lines = []
    MAX_VALUES_TO_SHOW = 15

    for name, desc in categorical_columns_info.items():
        base_info = f"- **{name}**: {desc}"
        
        if name in unique_values_per_column and unique_values_per_column[name]:
            count = unique_values_per_column[name][0]
            values = unique_values_per_column[name][1]
            
            # Heuristic hint for the LLM in the prompt string
            cardinality_hint = "High Cardinality" if count > 50 else "Low Cardinality"
            
            if count <= MAX_VALUES_TO_SHOW:
                val_str = ", ".join(str(v) for v in values)
                extra_info = f" | Unique Count: {count} ({val_str})"
            else:
                examples = ", ".join(str(v) for v in list(values)[:5])
                extra_info = f" | Unique Count: {count} ({cardinality_hint}, examples: {examples}, ...)"
            
            formatted_lines.append(base_info + extra_info)
        else:
            formatted_lines.append(base_info + " | No value stats available.")

    columns_info_str = "\n".join(formatted_lines)

    model = create_model_for_process('validity_check') # Reuse existing process/model config
    prompt_content = prompt_template.format(columns_info_formatted=columns_info_str)
    
    structured_llm = model.with_structured_output(ColumnUtilityAnalysis)
    
    try:
        result = invoke_with_retries(structured_llm, prompt_content, timeout_seconds=45)
        return result
    except Exception as e:
        return None

class ConditionVerdict(BaseModel):
    """
    Represents the semantic verification result for a suggested condition.
    
    Output by the LLM when performing a "sanity check" on statistically-mined
    conditions to filter out tautologies and spurious correlations.
    
    Attributes:
        is_valid: True if the condition is potentially useful as a rule refinement.
            False only for tautologies or clearly nonsensical conditions.
        reason: Brief explanation of the verdict.
        verdict_type: Classification of the verdict:
            - 'logical': Valid, makes logical sense
            - 'valid': Valid, plausible business rule
            - 'trivial_tautology': Rejected, makes rule automatically pass
            - 'nonsensical_spurious': Rejected, coincidental/absurd correlation
    """
    is_valid: bool = Field(..., description="True if the condition is potentially useful. False ONLY if it is a tautology or completely nonsensical.")
    reason: str = Field(..., description="Brief justification.")
    verdict_type: str = Field(..., enum=["logical", "valid", "trivial_tautology", "nonsensical_spurious"])

async def verify_condition_semantically(
    original_rule: str,
    suggested_condition: str,
    columns_context: Dict[str, str]
) -> ConditionVerdict:
    """
    Performs a semantic "sanity check" on a statistically-mined condition.
    
    Statistical mining can discover conditions with strong correlations that are
    actually tautologies or spurious. This function uses an LLM to detect:
    
    1. **Tautologies**: Conditions that make the rule trivially true
        - Example: Rule="Salary > 0", Condition="Salary > 0" (redundant)
    2. **Spurious Correlations**: Conditions with no logical connection
        - Example: "If ZipCode=90210, then Gender=Male" (coincidental)
    
    The function is intentionally permissive - it only rejects clearly invalid
    conditions to avoid discarding potentially useful refinements.
    
    Args:
        original_rule: The data quality rule being refined.
        suggested_condition: String representation of the condition to verify.
        columns_context: Dict mapping column names (used in rule and condition)
            to their semantic descriptions, providing context for the LLM.
    
    Returns:
        ConditionVerdict with is_valid=True/False and explanation.
        Defaults to is_valid=True on LLM error to avoid losing good conditions.
    
    Example:
        >>> verdict = await verify_condition_semantically(
        ...     original_rule="All transactions must have tax_amount > 0",
        ...     suggested_condition='transaction_type = "SALE"',
        ...     columns_context={'tax_amount': 'Tax amount', 'transaction_type': 'Type of transaction'}
        ... )
        >>> verdict.is_valid  # True - plausible business logic
        >>> verdict.verdict_type  # 'valid'
    """
    
    context_str = "\n".join([f"- {col}: {desc}" for col, desc in columns_context.items()])

    prompt = f"""
    You are a Data Quality Assistant. Your task is to perform a "Sanity Check" on a suggested condition found by a statistical mining algorithm.
    
    The algorithm found a strong correlation. Your job is ONLY to reject the condition if it is a **Fatal Error**. 
    If the condition is even remotely plausible as a business rule, mark it as **VALID**.

    ---
    **The Data Quality Rule:** "{original_rule}"
    **The Suggested Condition:** "{suggested_condition}"
    
    **Column Meanings:**
    {context_str}
    ---

    **Evaluation Logic (Be Permissive):**

    1. **CHECK FOR TAUTOLOGY / TRIVIALITY (Reject):**
       - Does the condition make the rule pass automatically? 
       - Does the condition simply repeat the rule's logic?
       - *Example:* Rule="Salary must be > 0", Condition="Salary > 0". (REJECT)
       - *Example:* Rule="End Date > Start Date", Condition="End Date is not null". (REJECT - Trivial)

    2. **CHECK FOR NONSENSE / SPURIOUS CORRELATION (Reject):**
       - Is the connection clearly coincidental or absurd?
       - *Example:* "If `ZipCode` is 90210, then `Gender` is Male". (REJECT - Geography doesn't dictate gender).
       - *Example:* "If `RowID` is even...". (REJECT - IDs are arbitrary).

    3. **DEFAULT TO VALID (Accept):**
       - If it's not a Tautology and not obviously Nonsense, it is VALID.
       - Even if you don't fully understand the business context, assume there might be a specific policy for this subset.
       - *Example:* "If `Department` is 'Sales', then `Commission` must be > 0". (VALID - Plausible business logic).
       - *Example:* "If `LegacySystem` is 'True', then `Email` can be null". (VALID - Plausible technical constraint).

    **Verdict:**
    """

    model = create_model_for_process('condition_verification')
    # Use a low temperature (0.0) to make it strict about following instructions
    structured_llm = model.with_structured_output(ConditionVerdict)
    
    try:
        result = invoke_with_retries(structured_llm, prompt, timeout_seconds=30)
        return result
    except Exception as e:
        # Default to True (Valid) on error to avoid dropping potentially good rules
        return ConditionVerdict(is_valid=True, reason="LLM check failed, defaulting to valid", verdict_type="valid")