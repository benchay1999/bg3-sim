python3 src/persona_evaluationrun_llm_approval_inference.py \
  --input result-dataset/astarion_approval_dataset_subset.json \
  --output test/gpt-5_astarion_llm_approvals.jsonl \
  --model gpt-5 \
  --sleep 0.1 \
  --metrics_dir test \
  --character Astarion

python3 src/persona_evaluationrun_llm_approval_inference.py \
  --input result-dataset/astarion_approval_dataset_subset.json \
  --output test/gpt-5-mini_astarion_llm_approvals.jsonl \
  --model gpt-5-mini \
  --sleep 0.1 \
  --metrics_dir test \
  --character Astarion

python3 src/persona_evaluationrun_llm_approval_inference.py \
  --input result-dataset/astarion_approval_dataset_subset.json \
  --output test/gpt-4o-mini_astarion_llm_approvals.jsonl \
  --model gpt-4o-mini \
  --sleep 0.1 \
  --metrics_dir test \
  --character Astarion


python3 src/persona_evaluationrun_llm_approval_inference.py \
  --input result-dataset/astarion_approval_dataset_subset.json \
  --output test/gemini-2.0-flash_astarion_llm_approvals.jsonl \
  --model gemini/gemini-2.0-flash \
  --sleep 0.1 \
  --metrics_dir test \
  --character Astarion


python3 src/persona_evaluationrun_llm_approval_inference.py \
  --input result-dataset/astarion_approval_dataset_subset.json \
  --output test/qwen3-32b_astarion_llm_approvals.jsonl \
  --model hosted_vllm/Qwen/Qwen3-32B \
  --api_base http://localhost:8010/v1 \
  --api_key dummy \
  --sleep 0.02 \
  --metrics_dir test \
  --character Astarion