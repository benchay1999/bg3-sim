# BG3-SIM

What to do (in order)

## Generate approval paths
```python
python batch_approval_paths.py -w 10
```
The above will generate `approval-paths/` folder with conversation trajectories containing approvals.(TODO: remove my personal paths in the code)

## Generate context for the sessions
```python
python batch_generate_contexts.py --output-root output --dest-root qa-context-rag --model openai/gpt-5-mini
```

## Build the QA dataset
```python
python scripts/build_approval_dataset.py
```
### Subset
```python
python scripts/sample_approval_dataset_subset.py
```
The above will generate a QA set in `result-dataset/`
(TODO: remove my personal paths in the code)

## Run LLM approval inference
```python
mkdir test
python3 scripts/run_llm_approval_inference.py --input result-dataset/astarion_approval_dataset_subset.json --output test/gpt-4o-mini_astarion_llm_approvals.jsonl --character Astarion --model gpt-4o-mini --sleep 0.1 --metrics_dir test
```

## Human QA
```python
# Balanced sub-sample with 20 per class (total up to 80)
  python scripts/cli_poll.py \
      --input /home/wschay/bg3-sim/test/1002_gpt-4o-mini_astarion_llm_approvals.jsonl \
      --output /home/wschay/bg3-sim/poll/astarion_cli_answers.jsonl \
      --base-dir /home/wschay/bg3-sim \
      --per_class 20 --seed 123

# Resume a previous session (just re-run with same --output path)
python scripts/cli_poll.py --input ... --output /home/wschay/bg3-sim/poll/astarion_cli_answers.jsonl

# Compute stats only (without running the poll)
python scripts/cli_poll.py --input ... --output /home/wschay/bg3-sim/poll/astarion_cli_answers.jsonl --stats-only
```