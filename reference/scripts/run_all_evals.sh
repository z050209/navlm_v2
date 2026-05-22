#!/bin/bash
# Wait for both v4a and v4b training to finish, then run all 6 evals.
cd /pub/evaluation_group/ning/test/navlm

echo "[$(date)] waiting for v4a and v4b training to finish..."
while [ ! -f results/lora_zurich_v4a/adapter_model.safetensors ] \
   || [ ! -f results/lora_zurich_v4b/adapter_model.safetensors ]; do
    sleep 120
done
echo "[$(date)] both trainings finished."

# At this point GPUs 1 and 2 are free. Run 4 sequential evals on GPU 1.
echo "[$(date)] running 4 base/C1 evals on GPU 1..."
CUDA_VISIBLE_DEVICES=1 python scripts/eval_lora.py \
    --eval data/cities/zurich/synth_v3_eval.jsonl \
    --tag base_v3 --skip-hallucination > results/eval_v3_base_v3.log 2>&1
CUDA_VISIBLE_DEVICES=1 python scripts/eval_lora.py \
    --eval data/cities/zurich/synth_v4a_eval.jsonl \
    --tag base_v4a --skip-hallucination > results/eval_v3_base_v4a.log 2>&1
CUDA_VISIBLE_DEVICES=1 python scripts/eval_lora.py \
    --eval data/cities/zurich/synth_v4b_eval.jsonl \
    --tag base_v4b --skip-hallucination > results/eval_v3_base_v4b.log 2>&1
CUDA_VISIBLE_DEVICES=1 python scripts/eval_lora.py \
    --eval data/cities/zurich/synth_v3_eval.jsonl \
    --lora results/lora_zurich_v3 --tag lora_v3 \
    --skip-hallucination > results/eval_v3_lora_v3.log 2>&1
echo "[$(date)] 4 base/C1 evals done. Running C2 + C3 in parallel..."

# C2 + C3 evals on GPU 1 + 2 in parallel
CUDA_VISIBLE_DEVICES=1 python scripts/eval_lora.py \
    --eval data/cities/zurich/synth_v4a_eval.jsonl \
    --lora results/lora_zurich_v4a --tag lora_v4a \
    --skip-hallucination > results/eval_v3_lora_v4a.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 python scripts/eval_lora.py \
    --eval data/cities/zurich/synth_v4b_eval.jsonl \
    --lora results/lora_zurich_v4b --tag lora_v4b \
    --skip-hallucination > results/eval_v3_lora_v4b.log 2>&1 &
wait
echo "[$(date)] all 6 evals complete."

# Final comparison
python scripts/eval_lora.py --compare \
    results/eval_v3_base_v3.json results/eval_v3_lora_v3.json
python scripts/eval_lora.py --compare \
    results/eval_v3_base_v4a.json results/eval_v3_lora_v4a.json
python scripts/eval_lora.py --compare \
    results/eval_v3_base_v4b.json results/eval_v3_lora_v4b.json
