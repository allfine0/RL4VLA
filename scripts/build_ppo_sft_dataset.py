#!/usr/bin/env python3
"""
构建PPO-SFT对齐数据集的TensorFlow Dataset

将对齐后的PPO数据构建为与原始SFT数据集相同的TensorFlow Dataset格式，
便于在训练中使用。

使用方法:
    python build_ppo_sft_dataset.py --data_dir /path/to/aligned/data --output_dir /path/to/tfds
"""

import os
import json
import argparse
import numpy as np
from pathlib import Path
from typing import Iterator, Tuple, Any
import glob
from tqdm import tqdm
import tensorflow_datasets as tfds


class PPOSFTDataset(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for PPO-SFT aligned dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
        '1.0.0': 'Initial release.',
    }

    def __init__(self, data_dir: Path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_dir = data_dir

    def _info(self) -> tfds.core.DatasetInfo:
        """Dataset metadata (homepage, citation,...)."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                'steps': tfds.features.Dataset({
                    'observation': tfds.features.FeaturesDict({
                        'image': tfds.features.Image(
                            shape=(480, 640, 3), dtype=np.uint8, encoding_format='jpeg',
                            doc='Observation image.'
                        ),
                    }),
                    'language_instruction': tfds.features.Text(
                        doc='Language Instruction.'
                    ),
                    'action': tfds.features.Tensor(shape=(7,), dtype=np.float32, ),
                }),
                'episode_metadata': tfds.features.FeaturesDict({
                    'file_path': tfds.features.Text(
                        doc='Path to the original data file.'
                    ),
                    'source': tfds.features.Text(
                        doc='Source of the data (ppo_rollout).'
                    ),
                    'original_length': tfds.features.Tensor(shape=(), dtype=np.int32),
                    'filtered_length': tfds.features.Tensor(shape=(), dtype=np.int32),
                }),
            }))

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Define data splits."""
        # 获取所有数据文件
        data_files = []
        for ext in ['*.npz']:
            data_files.extend(glob.glob(str(self.data_dir / ext)))
        
        # 按文件名排序
        data_files = sorted(data_files)
        
        # 划分训练集和验证集 (90% 训练, 10% 验证)
        num_files = len(data_files)
        train_split = int(0.9 * num_files)
        
        train_files = data_files[:train_split]
        val_files = data_files[train_split:]
        
        print(f"Total files: {num_files}")
        print(f"Train files: {len(train_files)}")
        print(f"Val files: {len(val_files)}")
        
        return {
            'train': self._generate_examples(train_files),
            'val': self._generate_examples(val_files),
        }

    def _generate_examples(self, data_files: list) -> Iterator[Tuple[str, Any]]:
        """Generator of examples for each split."""

        def _parse_example(episode_path):
            # 加载.npz文件
            data = np.load(episode_path, allow_pickle=True)["arr_0"].item()
            
            # 直接返回数据，因为已经是SFT格式
            return data

        for file_path in tqdm(data_files, desc="Building dataset"):
            episode_path = Path(file_path)
            episode_id = episode_path.stem
            
            try:
                sample = _parse_example(episode_path)
                yield episode_id, sample
            except Exception as e:
                print(f"Error processing {episode_path}: {e}")
                continue


def build_dataset(data_dir: Path, output_dir: Path):
    """
    构建TensorFlow Dataset
    
    Args:
        data_dir: 对齐后的数据目录
        output_dir: 输出目录
    """
    print(f"Building dataset from: {data_dir}")
    print(f"Output directory: {output_dir}")
    
    # 创建数据集
    dataset = PPOSFTDataset(data_dir)
    
    # 构建数据集
    dataset.download_and_prepare(
        download_dir=output_dir,
        download_config=tfds.download.DownloadConfig(
            manual_dir=str(data_dir),
            extract_dir=str(output_dir / "extracted"),
        )
    )
    
    print(f"Dataset built successfully!")
    print(f"Dataset location: {output_dir}")
    
    # 打印数据集信息
    print("\nDataset Info:")
    print(f"Train size: {dataset.info.splits['train'].num_examples}")
    print(f"Val size: {dataset.info.splits['val'].num_examples}")
    print(f"Total size: {dataset.info.splits['train'].num_examples + dataset.info.splits['val'].num_examples}")


def test_dataset(data_dir: Path):
    """
    测试数据集
    
    Args:
        data_dir: 数据集目录
    """
    print("Testing dataset...")
    
    # 加载数据集
    dataset = tfds.load(
        str(data_dir),
        split='train',
        with_info=True
    )
    
    # 获取一个样本
    for example in dataset.take(1):
        print("\nSample episode:")
        print(f"Number of steps: {len(example['steps'])}")
        print(f"Instruction: {example['steps'][0]['language_instruction']}")
        print(f"Action shape: {example['steps'][0]['action'].shape}")
        print(f"Image shape: {example['steps'][0]['observation']['image'].shape}")
        print(f"Source: {example['episode_metadata']['source']}")
        print(f"Original length: {example['episode_metadata']['original_length']}")
        print(f"Filtered length: {example['episode_metadata']['filtered_length']}")
        break


def main():
    parser = argparse.ArgumentParser(description='Build PPO-SFT aligned TensorFlow Dataset')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Directory containing aligned PPO-SFT data')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for TensorFlow Dataset')
    parser.add_argument('--test', action='store_true',
                       help='Test the built dataset')
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    
    if not data_dir.exists():
        print(f"Error: Data directory {data_dir} does not exist")
        return
    
    # 构建数据集
    build_dataset(data_dir, output_dir)
    
    # 测试数据集
    if args.test:
        test_dataset(output_dir)


if __name__ == "__main__":
    main() 