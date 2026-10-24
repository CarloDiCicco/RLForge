# RLForge

**Status: work in progress.** RLForge is a personal lab for studying
reinforcement learning (RL) on LLMs. Training and evaluation run on a
single NVIDIA DGX Spark. Results so far come from one training run and
should be read as early findings, not final claims.

## What it does

A model writes code. A sandbox runs that code against hidden tests. The test
result becomes the reward for RL training:

generate → execute in sandbox → grade → train → evaluate

## First result: GRPO on LeetCode, thinking mode

**Setup**
- **Model:** Qwen3.5-4B in thinking mode, with LoRA (rank 32) on all linear layers.
- **Algorithm:** GRPO (TRL) with DAPO-style token averaging and no KL penalty.
- **Data:** 300 LeetCode training tasks from LiveCodeBench. 44 validation tasks
  were used for checks during training, and 100 test tasks were scored once,
  after training.
- **Training:** 38 steps of 8 tasks × 8 answers, so each training task was seen
  about once.
- **Reward:** 1 if the code passes all tests, otherwise 0. Only the text after
  the model's reasoning is graded. An answer cut off at the 32k-token limit
  scores 0.
- **Compute:** about 4 days on one DGX Spark, with vLLM generating answers on
  the same GPU.

**Test results** (100 tasks × 2 answers each):

| Sampling setting | Before RL | After RL (final) |
|---|---|---|
| Training setting (temp 1.0) | 62.0% | **73.0%** |
| Qwen's suggested setting (temp 0.6, presence penalty 1.5) | 74.5% | 77.5% |

At the training setting the gain is clearly larger than run-to-run noise. At
Qwen's suggested setting the +3 points are within noise.

| Training setting | Before RL | After RL |
|---|---|---|
| Answers cut off at 32k | 35.5% | 0% |
| Average answer length | 22.2k tokens | 11.6k tokens |
| Pass rate within a 16k budget | 37% | 68% |

**What we learned**
- **Most of the gain comes from finishing in time.** Before RL, many answers ran
  past the token limit without producing any code. After RL, the model finishes
  almost every answer, in about half the tokens.
- **Thinking less has a cost on tasks the model could already solve.** On
  tasks where the untrained model always finished in time, accuracy did not
  improve: it was flat at Qwen's suggested setting and fell from 97% to 90% at
  the training setting. One likely reason is that the model now reasons less
  even where longer reasoning helped (part of the drop may also be chance).
  Overall, the gain comes from ending endless reasoning, not from better code.
- **The gain depends on the sampling setting.** With Qwen's suggested
  sampling, the untrained model rarely runs out of tokens, so the gain is small.

**Limitations:** the data is small (300 training and 100 test tasks), there was
one training run with one seed, and one main benchmark.

## Next

- Control runs with no training, such as a forced stop after a fixed thinking
  budget, to measure how much of the gain is just learning to stop.
- More and harder training data, to test whether RL improves the code itself
  once finishing in time is no longer the bottleneck.
- Larger and newer test sets, including problems released after the model's
  training cutoff.
- A comparison with other methods on the same data, such as other online RL
  algorithms or rejection-sampling fine-tuning.

## Repo layout

| Path | Contents |
|---|---|
| `src/rlforge/` | dataset adapters, prompts, sandbox, grading, training code |
| `scripts/` | data preparation, training (`train_grpo.py`), evaluation (`thinking_length_study_vllm.py`) |
| `configs/` | experiment configs; the run above is `grpo-thinking-leetcode.yaml` |
| `data/splits/` | the exact train / validation / test task IDs |

## Running it

Tested only on one DGX Spark (GB10, CUDA 13, peak ~90 GB of GPU memory). A
fresh setup may need some manual installs. The main entry points:

```bash
.venv/bin/python scripts/prepare_livecodebench.py        # build the dataset
.venv/bin/python scripts/train_grpo.py configs/grpo-thinking-leetcode.yaml

# evaluate the trained adapter on the test split
.venv/bin/python scripts/thinking_length_study_vllm.py \
  --split data/splits/lcb_leetcode_v2_test.json --allow-test \
  --rounds 2 --temperature 1.0 --presence-penalty 0 \
  --adapter runs/train/<run>/final_adapter
```

## License

Apache 2.0
