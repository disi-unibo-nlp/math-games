# MathGames Evaluation Scripts

This repo contains two scripts for evaluating model performance on the MathGames dataset:
- **evaluate.py** - LLM-as-a-judge evaluation of model answers
- **category_evaluate.py** - Detailed accuracy analysis across categories and metadata

## Setup

Create a `.env` file in your project root with your API keys:
```
HF_TOKEN=your_huggingface_token
OPENAI_KEY=your_openai_api_key
GEMINI_API_KEY=your_gemini_api_key
```

For vision mode evaluation, ensure you have a `jpg_images/` directory with images named `image_{id}.jpg`

---

## Script 1: evaluate.py

### Overview

Compares model-generated answers against gold standard answers from the MathGames dataset. Uses GPT-4 or Gemini models as judges to evaluate whether student answers are equivalent to the correct answers.

### Usage

```bash
python3 src/evaluation/evaluate.py \
  --jsonl_file_path out/completions/model_name/tir/vision/outputs.jsonl \
  --judge_model gpt-4o-2024-08-06 \
  --ids_to_modify 42 87
```

### Arguments

- `--jsonl_file_path` (required): Path to JSONL file containing model predictions with fields: `id`, `final_answer`, `gold_answer`
- `--dataset_name`: HuggingFace dataset name (default: `disi-unibo-nlp/MathGames`)
- `--judge_model`: LLM judge model (default: `gpt-4o-2024-08-06`)
  - Alternatives: `gpt-4o-mini-2024-07-18`, `gemini-2.0-flash`
- `--ids_to_modify`: List of IDs to force-mark as incorrect for debug (e.g., `--ids_to_modify 1 5 10`)

### How It Works

1. **Automatic Evaluation**: Numeric answers matching gold answers are marked correct automatically
2. **LLM Judging**: Non-numeric or non-matching answers are evaluated by the judge model
3. **Vision Support**: If path contains `/vision/`, images are included in judge prompts
4. **Output**: Creates two files:
   - `*_eval_{judge_model}.csv`: Final evaluation results
   - `eval_{judge_model}.jsonl`: Detailed judge responses

### Input Format

JSONL file should contain:
```json
{"id": 1, "final_answer": "42", "gold_answer": "42", ...}
{"id": 2, "final_answer": "The answer is 7", "gold_answer": "7", ...}
```

### Output Format

CSV with columns: `id`, `gold_answer`, `final_answer`, `model_response` (yes/no/MISSING_ANSWER)

---

## Script 2: category_evaluate.py

### Overview

Computes comprehensive accuracy metrics from evaluation results, breaking down performance by category, difficulty level, subject, year, competition phase, and their combinations.

### Usage

```bash
python3 src/evaluation/category_evaluate.py \
  --file_path out/completions/model_name/tir/vision/outputs_eval_gpt-4o-2024-08-06.csv
```

### Arguments

- `--file_path` (required): Path to CSV file containing evaluation results (output from evaluate.py)
- `--dataset_name`: HuggingFace dataset name (default: `disi-unibo-nlp/MathGames`)

### Output Files

Generates two files:

1. **`category_accuracy.txt`** (in same directory as input CSV):
   - Global accuracy
   - Year-by-year accuracy (1996-2024)
   - Difficulty-level accuracy (easy/medium/hard)
   - Category accuracy (CE, C1, C2, L1, L2, GP, HC)
   - Subject accuracy (Arithmetic, Logic, Geometry, Combinatorics, Algebra, Pattern Recognition)
   - Category × Year breakdown
   - Category × Competition Phase breakdown
   - Category × Subject breakdown

2. **`out/completions/category_accuracy.jsonl`** (appended):
   - JSON line with model name and category-wise scores for easy aggregation

---

## Complete Workflow Example

```bash
# Step 1: Evaluate model outputs using LLM judge
python3 src/evaluation/evaluate.py \
  --jsonl_file_path out/completions/gpt-4o/tir/textual/outputs.jsonl \
  --judge_model gpt-4o-2024-08-06

# Step 2: Calculate category-wise accuracy metrics
python3 src/evaluation/category_evaluate.py \
  --file_path out/completions/gpt-4o/tir/textual/outputs_eval_gpt-4o-2024-08-06.csv
```

This will produce:
- `outputs_eval_gpt-4o-2024-08-06.csv` - Individual problem evaluations
- `eval_gpt-4o-2024-08-06.jsonl` - Detailed judge responses
- `category_accuracy.txt` - Comprehensive accuracy breakdown
- Entry in `out/completions/category_accuracy.jsonl` - Aggregated scores