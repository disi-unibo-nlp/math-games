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
    --n_out_sequences 8 \ # for maj@8
    --temperature 0.7 \
    --top_p 0.8 \
    --mode cot \
    --text_only True \ # ignoring images
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
    --n_sampling 8 \ # used only for tir for maj@8
    --temperature 0.7 \
    --top_p 0.8 \
    --mode tir \
    --text_only True \ # ignoring images
    --n_gpus 1 \
    --n_rounds 3 # number of rounds to run ONLY for TIR
```

### Run the code
For convenience, a shell script is provided to execute the vLLM benchmark:
```bash
./src/bench_vllm.sh
```

### Example of TIR conversation
```
<|im_start|>system
Please integrate natural language reasoning with programs to solve the problem above, and put your final answer within \boxed{}.<|im_end|>
<|im_start|>user
Taxes, what a passion! Carla paved her entire rectangular room using square tiles (of the same size). Counting the used tiles, she realizes that on the edge of the room there are as many as inside. How many tiles did Carla use altogether (knowing that they are less than 50)?<|im_end|>
<|im_start|>assistant
To solve this problem, let's break it down step-by-step:

1. Let the dimensions of the room be \(m \times n\) tiles.
2. The total number of tiles used is \(m \times n\).
3. The number of tiles on the edge of the room is \(2m + 2n - 4\).
4. The number of tiles inside the room is \((m-2) \times (n-2)\).
5. According to the problem, the number of tiles on the edge is equal to the number of tiles inside, so we can set up the equation:
   \[
   2m + 2n - 4 = (m-2)(n-2)
   \]
6. Simplifying the equation, we get:
   \[
   2m + 2n - 4 = mn - 2m - 2n + 4
   \]
   \[
   4m + 4n - 8 = mn
   \]
   \[
   mn - 4m - 4n + 8 = 0
   \]
7. Adding 16 to both sides, we get:
   \[
   mn - 4m - 4n + 16 = 8
   \]
   \[
   (m-4)(n-4) = 8
   \]
8. We need to find pairs \((m-4, n-4)\) that multiply to 8. The factor pairs of 8 are \((1,8)\), \((2,4)\), \((-1,-8)\), and \((-2,-4)\). Since \(m\) and \(n\) are positive integers, the valid pairs are \((1,8)\) and \((2,4)\).
9. Converting back to \((m,n)\), the pairs are \((5,12)\) and \((6,8)\).
10. We need to find the total number of tiles for each pair and check which one is less than 50.
   - For \((m,n) = (5,12)\), the total number of tiles is \(5 \times 12 = 60\).
   - For \((m,n) = (6,8)\), the total number of tiles is \(6 \times 8 = 48\).

Since 48 is less than 50, the answer is 48. Let's confirm this with Python code.
```python
# Define the function to check the number of tiles
def find_tiles():
    # Possible factor pairs of 8
    factor_pairs = [(1, 8), (2, 4)]
    # List to store the total number of tiles
    total_tiles = []
    for (a, b) in factor_pairs:
        m, n = a + 4, b + 4
        total_tiles.append(m * n)
    return total_tiles

# Find the total number of tiles
total_tiles = find_tiles()
# Filter the tiles that are less than 50
valid_tiles = [tiles for tiles in total_tiles if tiles < 50]
print(valid_tiles[0])
'''
<|im_end|>
<|im_start|>user
'''output
(<function find_tiles at 0x7dd7f4089360>, [60, 48], [48])
'''<|im_end|>
<|im_start|>assistant
The total number of tiles Carla used altogether, knowing that they are less than 50, is \(\boxed{48}\).<|im_end|>
```
