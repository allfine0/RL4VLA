#!/bin/bash

# PPO-SFT数据对齐脚本使用示例

echo "🚀 PPO-SFT数据对齐流程"
echo "========================"

# 设置路径
PPO_DATA_DIR="/mnt/workspace/luodx/RL4VLA/SimplerEnv/wandb/run-20250701_151557-4j0b73ri/glob/vis_*_train"
ALIGNED_DATA_DIR="/mnt/workspace/luodx/RL4VLA/ppo_sft_aligned_data"
TFDS_OUTPUT_DIR="/mnt/workspace/luodx/RL4VLA/ppo_sft_tfds"

echo "📁 路径设置:"
echo "  PPO数据目录: $PPO_DATA_DIR"
echo "  对齐数据目录: $ALIGNED_DATA_DIR"
echo "  TFDS输出目录: $TFDS_OUTPUT_DIR"
echo ""

# 步骤1: 查找最新的PPO数据目录
echo "🔍 步骤1: 查找最新的PPO数据..."
LATEST_PPO_DIR=$(ls -td /mnt/workspace/luodx/RL4VLA/SimplerEnv/wandb/run-*/glob/vis_*_train 2>/dev/null | head -1)

if [ -z "$LATEST_PPO_DIR" ]; then
    echo "❌ 未找到PPO数据目录"
    echo "请先运行PPO训练: bash run_ppo.sh"
    exit 1
fi

echo "✅ 找到PPO数据目录: $LATEST_PPO_DIR"
echo ""

# 步骤2: 数据对齐
echo "🔄 步骤2: 将PPO数据对齐为SFT格式..."
python align_ppo_sft_data.py \
    --ppo_data_dir "$LATEST_PPO_DIR" \
    --output_dir "$ALIGNED_DATA_DIR" \
    --min_success_steps 6 \
    --pos_thresh 0.01 \
    --rot_thresh 0.06

if [ $? -ne 0 ]; then
    echo "❌ 数据对齐失败"
    exit 1
fi

echo "✅ 数据对齐完成"
echo ""

# 步骤3: 构建TensorFlow Dataset
echo "📊 步骤3: 构建TensorFlow Dataset..."
python build_ppo_sft_dataset.py \
    --data_dir "$ALIGNED_DATA_DIR" \
    --output_dir "$TFDS_OUTPUT_DIR" \
    --test

if [ $? -ne 0 ]; then
    echo "❌ TensorFlow Dataset构建失败"
    exit 1
fi

echo "✅ TensorFlow Dataset构建完成"
echo ""

# 步骤4: 显示结果
echo "📈 结果统计:"
echo "========================"
echo "对齐数据目录: $ALIGNED_DATA_DIR"
echo "  - 文件数量: $(ls -1 $ALIGNED_DATA_DIR/*.npz 2>/dev/null | wc -l)"
echo "  - 数据集信息: $ALIGNED_DATA_DIR/dataset_info.json"

echo ""
echo "TensorFlow Dataset目录: $TFDS_OUTPUT_DIR"
echo "  - 训练集大小: $(find $TFDS_OUTPUT_DIR -name "train*" -type f | wc -l)"
echo "  - 验证集大小: $(find $TFDS_OUTPUT_DIR -name "val*" -type f | wc -l)"

echo ""
echo "🎉 数据对齐流程完成！"
echo ""
echo "💡 使用提示:"
echo "1. 对齐后的数据可用于进一步的SFT训练"
echo "2. 可以与其他SFT数据合并使用"
echo "3. 数据格式与原始SFT数据集完全兼容"
echo ""
echo "📝 下一步:"
echo "   - 使用对齐后的数据进行模型微调"
echo "   - 或者将数据合并到现有的SFT数据集中" 