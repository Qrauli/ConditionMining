# Condition Mining for Data Quality Rules

This repository contains the official implementation of the condition mining system for data quality rules, designed as an artifact for reproducing the paper's experiments.

## 📌 Overview

Data quality rules often suffer from being too broad, leading to high false-positive violation rates. This project introduces a condition suggestion engine that takes a rule, violating samples, and satisfying samples, and discovers statistically significant and semantically meaningful conditions to refine the rule. The system employs both statistical condition mining and LLM-based semantic filtering.

## 📂 Repository Structure

```text
ConditionMining/
├── baselines/                  # Competitor/baseline implementations
│   ├── baseline_subgroup_discovery.py
│   └── baseline_suggestion.py
├── data/                       # Datasets used in the evaluation
├── data_understandings/        # Cached LLM data understandings for datasets
├── rules/                      # Cached rules and evaluation results
├── src/                        # Core system logic
│   ├── condition_discovery.py  # Orchestrates the discovery pipeline
│   ├── condition_semantic.py   # LLM integration for semantic relevance & utility
│   ├── condition_suggestion.py # Statistical condition mining core logic
│   ├── sampling.py             # Informative sampling strategies
│   ├── structures.py           # Core data structures (e.g., RuleDetail, Conditions)
│   └── utils.py                # Utilities for LLM calls and data handling
├── evaluate.py                 # Main entry point for evaluation
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## ⚙️ Setup and Installation

This project requires **Python 3.10+**. It is recommended to use a virtual environment.

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd ConditionMining
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure API Keys:**
   The semantic filtering uses OpenAI's API. Ensure you have the `OPENAI_API_KEY` environment variable set:
   ```bash
   # On Linux/macOS
   export OPENAI_API_KEY="your-api-key-here"

   # On Windows PowerShell
   $env:OPENAI_API_KEY="your-api-key-here"
   ```

## 🚀 Reproducing Experiments

The primary script for running experiments is `evaluate.py`. This script iterates over datasets and cached rules, evaluates the discovery engine against them, and produces accuracy and performance metrics.

### Run the Full Evaluation

To run the full evaluation using the default configurations:

```bash
python evaluate.py
```

### Run Ablation Study (No LLM Semantic Filtering)

To evaluate the system without the LLM-based semantic pruning phase (which tests all columns statistically):

```bash
python evaluate.py --no-llm
```

### Custom Directory Paths

If your datasets, rules, or data understanding caches are located elsewhere, you can specify them using command-line arguments:

```bash
python evaluate.py --data-dir path/to/datasets --rules-dir path/to/rules --data-understanding-dir path/to/data_understandings
```

## 📊 Expected Output

The evaluation script provides real-time progress and outputs summary statistics at the end, including:
- **Total conditions found** (where the engine correctly suggested the intended refinement condition).
- **Hits at K** (Percentage of conditions found in the Top 1, 3, 5, and 10 suggestions).
- **Mean Reciprocal Rank (MRR)**.
- **Runtime Statistics** (Time spent on statistical mining vs. LLM querying).

## 📄 License

This code is provided as a research artifact. Please refer to the accompanying publication for citation information.
