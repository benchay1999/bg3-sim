python3 scripts/run_llm_approval_inference.py \
  --input result-dataset/astarion_approval_dataset_subset.json \
  --output test/gpt-4o-mini_astarion_llm_approvals.jsonl \
  --template bg3_characters_llm_input_prompt_example.txt \
  --model gpt-4o-mini \
  --sleep 0.1 \
  --metrics_dir test

python3 scripts/run_llm_approval_inference.py \
  --input result-dataset/astarion_approval_dataset_subset.json \
  --output test/qwen3-32b_astarion_llm_approvals.jsonl \
  --template bg3_characters_llm_input_prompt_example.txt \
  --model hosted_vllm/Qwen/Qwen3-32B \
  --api_base http://localhost:8010/v1 \
  --api_key dummy \
  --sleep 0.1 \
  --metrics_dir test


python3 /home/wschay/bg3sim/scripts/run_llm_approval_inference.py \
  --input /home/wschay/bg3sim/result-dataset/astarion_approval_dataset_subset.json \
  --output /home/wschay/bg3sim/test/gemini-2.5-flash_astarion_llm_approvals.jsonl \
  --template /home/wschay/bg3sim/bg3_characters_llm_input_prompt_example.txt \
  --model gemini/gemini-2.5-flash \
  --sleep 0.2 \
  --metrics_dir /home/wschay/bg3sim/test

python3 /home/wschay/bg3sim/scripts/run_llm_approval_inference.py \
  --input /home/wschay/bg3sim/result-dataset/astarion_approval_dataset_subset.json \
  --output /home/wschay/bg3sim/test/gpt-5-mini_astarion_llm_approvals.jsonl \
  --template /home/wschay/bg3sim/bg3_characters_llm_input_prompt_example.txt \
  --model gpt-5-mini \
  --sleep 0.1 \
  --metrics_dir /home/wschay/bg3sim/test


python3 /home/wschay/bg3sim/scripts/run_llm_approval_inference.py \
  --input /home/wschay/bg3sim/result-dataset/astarion_approval_dataset_subset.json \
  --output /home/wschay/bg3sim/test/gpt-5_astarion_llm_approvals.jsonl \
  --template /home/wschay/bg3sim/bg3_characters_llm_input_prompt_example.txt \
  --model gpt-5 \
  --sleep 0.1 \
  --metrics_dir /home/wschay/bg3sim/test