# BG3-SIM

What to do (in order)

## Generate approval paths
```python
python batch_approval_paths.py -w 10
```

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

## Run LLM approval inference
```python
python3 /home/wschay/bg3sim/scripts/run_llm_approval_inference.py \
  --input /home/wschay/bg3sim/result-dataset/astarion_approval_dataset_subset.json \
  --output /home/wschay/bg3sim/test/gpt-4o-mini_astarion_llm_approvals.jsonl \
  --template /home/wschay/bg3sim/bg3_characters_llm_input_prompt_example.txt \
  --model gpt-4o-mini \
  --sleep 0.2 \
  --metrics_dir /home/wschay/bg3sim/test
```