
## Workflow (recommended)

> Note: the legacy "LLM pipeline" script is not required for the main experiments.
> Use the scripts below in order.

### Environment prerequisites
- MATLAB installed and callable as `matlab` from terminal
- MATPOWER available for MATLAB (some scripts also use `MATPOWER_DIR`)
- Python env with `openai`, `pandas`, etc.
- `OPENAI_API_KEY` set in your shell

### Run order
1. `python python/build_sft_from_gt.py`
2. `python python/sft_pack.py`
3. `python python/train_sft_gpt.py`
4. `python python/build_dpo_pairs.py`
5. `python python/train_dpo_openai.py`
6. `python python/run_inference_all.py`
7. `python python/analyze_eval_vselect.py`
8. `python python/nn_supervised_train_eval_from_sft_with_plots.py`
9. `python python/plot_train.py`
10. `python python/analyze_eval_vselect_with_nn.py`

### Key inputs/outputs (high level)
- (1) builds `io/sft_raw.jsonl` (+ cached summaries / GT)
- (2) writes `io/sft_openai_chat_train.jsonl` (+ train/test indices)
- (3) writes `config/ft_model_sft.txt`
- (4) writes `io/dpo_openai_prefs.jsonl`
- (5) writes `config/ft_model_dpo.txt`
- (6)-(7) write evaluation CSVs under `io/`
- (8)-(10) train/evaluate NN baseline + plots, and compare vs LLM results
