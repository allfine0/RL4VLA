# PPO-SFT 数据对齐工具

这个工具包用于将PPO训练中收集的成功episode数据转换为与SFT（监督微调）数据相同的格式，便于后续的训练和微调。

## 📋 功能概述

### 🔄 数据对齐流程
1. **PPO Rollout数据收集** → PPO训练过程中收集的成功episode
2. **数据格式转换** → 将PPO数据转换为SFT格式
3. **TensorFlow Dataset构建** → 构建标准化的训练数据集
4. **数据验证** → 确保数据质量和格式正确性

### 🎯 主要特性
- ✅ **自动成功检测**: 基于连续成功步数过滤episode
- ✅ **动作过滤**: 移除太小的动作，保留有效动作
- ✅ **格式标准化**: 与原始SFT数据格式完全兼容
- ✅ **数据统计**: 详细的处理统计信息
- ✅ **错误处理**:  robust的错误处理和日志记录

## 📁 文件结构

```
RL4VLA/
├── align_ppo_sft_data.py          # 主要的数据对齐脚本
├── build_ppo_sft_dataset.py       # TensorFlow Dataset构建脚本
├── run_data_alignment.sh          # 一键运行脚本
└── README_PPO_SFT_Alignment.md    # 本说明文档
```

## 🚀 快速开始

### 方法1: 一键运行（推荐）
```bash
bash run_data_alignment.sh
```

### 方法2: 分步运行

#### 步骤1: 数据对齐
```bash
python align_ppo_sft_data.py \
    --ppo_data_dir "/path/to/ppo/rollout/data" \
    --output_dir "/path/to/aligned/data" \
    --min_success_steps 6 \
    --pos_thresh 0.01 \
    --rot_thresh 0.06
```

#### 步骤2: 构建TensorFlow Dataset
```bash
python build_ppo_sft_dataset.py \
    --data_dir "/path/to/aligned/data" \
    --output_dir "/path/to/tfds" \
    --test
```

## 📊 参数说明

### align_ppo_sft_data.py 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--ppo_data_dir` | str | 必需 | PPO rollout数据目录 |
| `--output_dir` | str | 必需 | 对齐后数据输出目录 |
| `--min_success_steps` | int | 6 | 最小连续成功步数 |
| `--pos_thresh` | float | 0.01 | 位置动作阈值 |
| `--rot_thresh` | float | 0.06 | 旋转动作阈值 |

### build_ppo_sft_dataset.py 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--data_dir` | str | 必需 | 对齐后的数据目录 |
| `--output_dir` | str | 必需 | TensorFlow Dataset输出目录 |
| `--test` | flag | False | 是否测试构建的数据集 |

## 📈 数据格式

### 输入格式 (PPO Rollout)
```python
{
    "image": np.array,      # [T+1, H, W, 3] 观察图像
    "action": np.array,     # [T, 7] 动作序列
    "info": List[Dict],     # [T] 环境信息列表
    "instruction": str      # 语言指令
}
```

### 输出格式 (SFT)
```python
{
    "steps": [
        {
            "observation": {
                "image": np.array,  # [H, W, 3] 单帧图像
            },
            "action": np.array,     # [7] 单步动作
            "language_instruction": str
        },
        # ... 更多步骤
    ],
    "episode_metadata": {
        "file_path": str,
        "source": "ppo_rollout",
        "original_length": int,
        "filtered_length": int
    }
}
```

## 🔍 数据处理细节

### 成功条件检测
- **连续成功步数**: 默认要求至少6步连续成功
- **成功标志**: 基于环境info中的`success`字段
- **时间窗口**: 在整个episode中检查连续成功

### 动作过滤
- **位置阈值**: 过滤掉位置变化小于0.01的动作
- **旋转阈值**: 过滤掉旋转变化小于0.06的动作
- **夹爪保留**: 保留所有夹爪状态切换事件

### 数据质量保证
- **长度一致性**: 确保图像和动作序列长度匹配
- **字段完整性**: 检查必要字段是否存在
- **格式验证**: 验证数据类型和形状

## 📊 统计信息

脚本会生成详细的处理统计信息：

```
Processing Statistics:
Total files: 1000
Successful episodes: 850
Failed episodes: 100
Filtered episodes: 200
Error episodes: 50
Success rate: 85.00%
```

## 🛠️ 故障排除

### 常见问题

1. **找不到PPO数据目录**
   ```bash
   # 检查PPO训练是否完成
   ls -la /mnt/workspace/luodx/RL4VLA/SimplerEnv/wandb/run-*/
   ```

2. **数据格式不匹配**
   ```bash
   # 检查数据文件格式
   python -c "import numpy as np; data=np.load('your_file.npz'); print(data.keys())"
   ```

3. **内存不足**
   ```bash
   # 减少批处理大小或使用更小的数据子集
   python align_ppo_sft_data.py --ppo_data_dir /path/to/subset
   ```

### 调试模式
```bash
# 启用详细日志
python align_ppo_sft_data.py --ppo_data_dir /path/to/data --output_dir /path/to/output --debug
```

## 🔗 集成到训练流程

### 与现有SFT数据合并
```python
# 在训练脚本中加载合并的数据
import tensorflow_datasets as tfds

# 加载原始SFT数据
sft_dataset = tfds.load('original_sft_dataset', split='train')

# 加载PPO对齐数据
ppo_dataset = tfds.load('ppo_sft_aligned', split='train')

# 合并数据集
combined_dataset = sft_dataset.concatenate(ppo_dataset)
```

### 用于模型微调
```python
# 使用对齐后的数据进行微调
from transformers import AutoProcessor, AutoModelForVision2Seq

processor = AutoProcessor.from_pretrained("openvla/openvla-7b")
model = AutoModelForVision2Seq.from_pretrained("openvla/openvla-7b")

# 使用对齐后的数据训练
train_dataset = tfds.load('ppo_sft_aligned', split='train')
# ... 训练代码
```

## 📝 注意事项

1. **数据质量**: 确保PPO训练产生了高质量的成功episode
2. **存储空间**: 对齐后的数据可能占用较大存储空间
3. **计算资源**: 大规模数据处理需要足够的CPU和内存
4. **版本兼容**: 确保TensorFlow和TensorFlow Datasets版本兼容

## 🤝 贡献

欢迎提交Issue和Pull Request来改进这个工具！

## 📄 许可证

本项目遵循与主项目相同的许可证。 