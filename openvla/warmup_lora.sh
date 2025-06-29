cuda="0"
task_name="warmup"
vla_path="/mnt/workspace/luodx/RL4VLA/openvla/openvla-7b"

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=$cuda \
torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/merge_lora.py \
--vla_path $vla_path \
  --run_path "checkpoints/${task_name}/steps_2000" \
  --lora_name "lora_002000"







