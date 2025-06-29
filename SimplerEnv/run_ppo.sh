cuda=0 # env and model on the same GPU (for 80G GPU)
vla_path="/mnt/workspace/luodx/RL4VLA/openvla/openvla-7b"
vla_load_path="/mnt/workspace/luodx/RL4VLA/openvla/checkpoints/warmup/steps_2000/lora_002000"

# 设置ManiSkill数据目录
export MS_ASSET_DIR="/mnt/workspace/luodx/RL4VLA/.maniskill"

CUDA_VISIBLE_DEVICES=$cuda XLA_PYTHON_CLIENT_PREALLOCATE=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python simpler_env/train_ms3_ppo.py \
  --name="PPO-pc25m_v3-warmup" \
  --env_id="PutOnPlateInScene25Main-v3" \
  --vla_path=$vla_path --vla_unnorm_key="bridge_orig" \
  --vla_load_path=$vla_load_path \
  --seed=0