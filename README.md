# BG3-SIM

What to do (in order)

## Generate approval paths
```python
python src/dataset_construction/batch_approval_paths.py -w 10
```
The above will generate `approval-paths/` folder with conversation trajectories containing approvals.(TODO: remove my personal paths in the code)

## Generate context for the sessions
```python
python src/dataset_construction/batch_generate_contexts.py --output-root output --dest-root qa-contexts-rag --model openai/gpt-5-mini
```

## Build the QA dataset
```python
python src/persona_evaluation/build_approval_dataset.py
```
### Subset
```python
python src/persona_evaluation/sample_approval_dataset_subset.py
```
The above will generate a QA set in `result-dataset/`
(TODO: remove my personal paths in the code)

## Run LLM approval inference
```python
mkdir test
python src/persona_evaluation/run_llm_approval_inference.py --input result-dataset/astarion_approval_dataset_subset.json --output test/gpt-4o-mini_astarion_llm_approvals.jsonl --character Astarion --model gpt-4o-mini --sleep 0.1 --metrics_dir test
```

## Human QA
```python
# Balanced sub-sample with 20 per class (total up to 80)
  python src/persona_evaluation/cli_poll.py \
      --input test/1002_gpt-4o-mini_astarion_llm_approvals.jsonl \
      --output poll/astarion_cli_answers.jsonl \
      --base-dir . \
      --per_class 20 --seed 123

# Resume a previous session (just re-run with same --output path)
python src/persona_evaluation/cli_poll.py --input ... --output poll/astarion_cli_answers.jsonl

# Compute stats only (without running the poll)
python src/persona_evaluation/cli_poll.py --input ... --output poll/astarion_cli_answers.jsonl --stats-only
```