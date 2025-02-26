# Experiments 

## Setup

**Environment Variables:**
  * Create a `.env` file in the root directory of the project.
  * Add the following API keys, replacing `<your_key>` and `<your_token>` with your actual credentials:

  ```
  HF_TOKEN=<your_token>
  OPENAI_KEY=<your_key>
  GEMINI_API_KEY=<your_key>
  DEEPSEEK_API_KEY=<your_key>
  ```

## vLLM Inference

### Text-only Problems
We support two inference modes: Chain-of-Thought (CoT) and Tool-Integrated-Reasoning (TIR).

**CoT:**

```bash
python3 -m src.bench_vllm \
    --model_name "Qwen/Qwen2.5-Math-7B-Instruct" \
    --dataset_name "path to HF repo" \
    --out_dir "./out" \
    --max_samples -1 \ # if > 0 you set a maximun prompt to consider for the execution, useful for debug
    --batch_size 8 \ # this is considered only for COT
    --cache_dir None \
    --n_out_sequences 8 \ # set to 8 for maj@8, while to 1 for pass@1
    --temperature 0.7 \ # set = 0 for pass@1, while set to > 0 for maj@8
    --top_p 0.8 \
    --mode cot \
    --text_only True \ # set True to use text-only problems
    --n_gpus 1 \
```

**TIR:**

```bash
python3 -m src.bench_vllm \
    --model_name "Qwen/Qwen2.5-Math-7B-Instruct" \
    --dataset_name "path to HF repo" \
    --out_dir "./out" \
    --max_samples -1 \ # if > 0 you set a maximun prompt to consider for the execution, useful for debug
    --cache_dir None \
    --n_out_sequences 1 \ 
    --n_sampling 8 \ # used only for tir: set to 8 for maj@8, while to 1 for pass@1
    --temperature 0.7 \
    --top_p 0.8 \
    --mode tir \
    --text_only True \ # set True to use text-only problems
    --n_gpus 1 \
    --n_rounds 3 # number of rounds to run ONLY for TIR
```

### Multimodal Problems

Only CoT is supported for multimodality.

```bash
python3 -m src.bench_vllm_multimodal \
    --model_name "Qwen/Qwen2-VL-7B-Instruct" \
    --dataset_name "path to HF repo" \
    --out_dir "./out" \
    --max_samples -1 \
    --batch_size 4 \
    --cache_dir None \
    --n_out_sequences 8 \ # set to 8 for maj@8, while to 1 for pass@1
    --temperature 0.8 \ # set = 0 for pass@1, while set to > 0 for maj@8
    --top_p 1.0 \
    --mode cot \
    --text_only False \ # set True to use multimodal problems (text + images)
    --n_gpus 1 
```

## OpenAI Batch Inference
Currently supporting only CoT greedy decoding.

```bash
python3 -m src.bench_openai_batch \
    --model_name "gpt-4o-2024-08-06" \
    --dataset_name "path to HF repo" \
    --max_samples -1 \
    --start_idx 0 \
    --top_p 1.0 \
    --n_sampling 1 \
    --n_out_sequences 1 \ 
    --temperature 0.0 \ 
    --mode cot \
    --text_only False \ # set True to use text-only problems
    --img_only True # set True to use multimodal problems (text + images)
```
To extract batch results:

```bash
python3 -m src.batch_results \
    --batch_id "API batch id" \
    --out_dir "out/batch_api/2025-01-02_15-08-45" \
```

## DeepSeek Inference
Currently supporting only CoT greedy decoding. Multimodality is NOT supported for deepseek's model.

```bash
python3 -m src.bench_openai \
    --model_name "deepseek-reasoner" \
    --dataset_name "path to HF repo" \
    --out_dir "./out" \
    --max_samples -1 \
    --start_idx 10 \
    --top_p 1.0 \
    --n_out_sequences 1 \
    --temperature 0.0 \
    --mode cot \
    --text_only True # set True to use text-only problems 
```

## Gemini Inference
Currently supporting only CoT greedy decoding.

```bash
python3 -m src.bench_gemini \
    --model_name "gemini-2.0-flash-thinking-exp" \
    --dataset_name "path to HF repo" \
    --max_samples -1 \
    --start_idx 0 \
    --top_p 1.0 \
    --top_k 40 \
    --n_sampling 1 \
    --temperature 0.0 \
    --mode cot \
    --text_only False \ # set True to use text-only problems (text + images)
    --img_only True # set True to use multimodal problems (text + images)
```


