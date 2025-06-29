cuda="0,1,2,3"
task_name="warmup"
vla_path="/mnt/workspace/luodx/RL4VLA/openvla/openvla-7b"

# 启用详细错误信息
export TORCHELASTIC_ERROR_FILE=/tmp/error.log
export TORCH_SHOW_CPP_STACKTRACES=1

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=$cuda \
torchrun --standalone --nnodes 1 --nproc-per-node 4 vla-scripts/finetune.py \
  --vla_path $vla_path \
  --data_root_dir "../datasets" \
  --dataset_name ${task_name} \
  --run_root_dir checkpoints/${task_name} \
  --lora_rank 32 \
  --batch_size 8 \
  --max_steps 2000 \
  --eval_steps 50 \
  --save_steps "0,500,1000,1500,2000" \
  --grad_accumulation_steps 1 \
  --learning_rate 5e-4 \
  --image_aug True \
  --unnorm_key="bridge_orig" \
  --wandb_project "RLVLA_sft"