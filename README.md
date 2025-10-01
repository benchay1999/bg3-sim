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
The above will generate a QA set in `result-dataset/`0
(TODO: remove my personal paths in the code)

## Run LLM approval inference
```python
mkdir test
python3 scripts/run_llm_approval_inference.py --input result-dataset/astarion_approval_dataset_subset.json --output test/gpt-4o-mini_wyll_llm_approvals.jsonl --character Wyll --model gpt-4o-mini --sleep 0.1 --metrics_dir test
```