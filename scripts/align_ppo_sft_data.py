#!/usr/bin/env python3
"""
PPO Rollout数据与SFT数据对齐脚本

将PPO训练中收集的成功episode数据转换为与SFT数据相同的格式，
便于后续的训练和微调。

使用方法:
    python align_ppo_sft_data.py --ppo_data_dir /path/to/ppo/data --output_dir /path/to/output
"""

import os
import json
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple
import glob
from tqdm import tqdm
import cv2
from PIL import Image
import shutil


def load_ppo_episode_data(episode_path: Path) -> Dict[str, Any]:
    """
    加载PPO episode数据
    
    Args:
        episode_path: episode数据文件路径
        
    Returns:
        包含episode数据的字典
    """
    try:
        # 尝试加载.npy文件
        if episode_path.suffix == '.npy':
            data = np.load(episode_path, allow_pickle=True).item()
        # 尝试加载.npz文件
        elif episode_path.suffix == '.npz':
            data = np.load(episode_path, allow_pickle=True)["arr_0"].item()
        else:
            raise ValueError(f"Unsupported file format: {episode_path.suffix}")
        
        return data
    except Exception as e:
        print(f"Error loading {episode_path}: {e}")
        return None


def filter_small_actions(actions: np.ndarray, pos_thresh: float = 0.01, rot_thresh: float = 0.06, check_gripper: bool = True) -> np.ndarray:
    """
    过滤掉太小的动作，与SFT数据处理保持一致
    
    Args:
        actions: 动作数组 [T, 7]
        pos_thresh: 位置阈值
        rot_thresh: 旋转阈值
        check_gripper: 是否检查夹爪动作
        
    Returns:
        过滤后的动作掩码
    """
    actions = np.asarray(actions)
    N = actions.shape[0]
    valid_mask = np.zeros(N, dtype=bool)

    for i in range(N):
        act = actions[i]
        delta_xyz = act[:3]
        delta_euler = act[3:6]
        gripper = act[6]

        pos_movement = np.linalg.norm(delta_xyz)
        rot_movement = np.linalg.norm(delta_euler)

        if pos_thresh is None and rot_thresh is None:
            is_valid = True
        elif pos_thresh is None:
            is_valid = (rot_movement > rot_thresh)
        elif rot_thresh is None:
            is_valid = (pos_movement > pos_thresh)
        else:
            is_valid = (pos_movement > pos_thresh) or (rot_movement > rot_thresh)

        # 保留夹爪切换事件
        if check_gripper and i > 0 and actions[i - 1][6] != gripper:
            is_valid = True

        valid_mask[i] = is_valid

    return valid_mask


def check_success_condition(info_list: List[Dict], min_success_steps: int = 6) -> bool:
    """
    检查episode是否满足成功条件
    
    Args:
        info_list: 环境信息列表
        min_success_steps: 最小连续成功步数
        
    Returns:
        是否满足成功条件
    """
    success_count = 0
    for info in info_list:
        if info.get("success", False):
            success_count += 1
        else:
            success_count = 0
        
        if success_count >= min_success_steps:
            return True
    
    return False


def convert_ppo_to_sft_format(ppo_data: Dict[str, Any], episode_id: str) -> Dict[str, Any]:
    """
    将PPO数据转换为SFT格式
    
    Args:
        ppo_data: PPO episode数据
        episode_id: episode ID
        
    Returns:
        SFT格式的数据
    """
    # 提取数据
    images = ppo_data["image"]  # [T+1, H, W, 3]
    actions = ppo_data["action"]  # [T, 7]
    info_list = ppo_data["info"]  # [T]
    instruction = ppo_data["instruction"]  # str
    
    # 确保数据长度一致
    assert len(images) == len(actions) + 1, f"Image length {len(images)} != action length {len(actions)} + 1"
    assert len(actions) == len(info_list), f"Action length {len(actions)} != info length {len(info_list)}"
    
    # 过滤小动作
    action_mask = filter_small_actions(actions)
    filtered_actions = actions[action_mask]
    filtered_images = images[:-1][action_mask]  # 去掉最后一帧，因为action比image少一个
    filtered_info = [info_list[i] for i in range(len(info_list)) if action_mask[i]]
    
    # 检查成功条件
    if not check_success_condition(filtered_info):
        return None
    
    # 构建SFT格式的steps
    steps = []
    for i in range(len(filtered_actions)):
        step = {
            'observation': {
                'image': filtered_images[i],  # [H, W, 3]
            },
            'action': filtered_actions[i],    # [7]
            'language_instruction': instruction,
        }
        steps.append(step)
    
    # 构建SFT格式的episode
    sft_episode = {
        'steps': steps,
        'episode_metadata': {
            'file_path': str(episode_id),
            'source': 'ppo_rollout',
            'original_length': len(actions),
            'filtered_length': len(filtered_actions),
        }
    }
    
    return sft_episode


def save_sft_episode(sft_episode: Dict[str, Any], output_path: Path, episode_id: str):
    """
    保存SFT格式的episode
    
    Args:
        sft_episode: SFT格式的episode数据
        output_path: 输出路径
        episode_id: episode ID
    """
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 保存为.npz格式
    npz_path = output_path / f"{episode_id}.npz"
    np.savez_compressed(npz_path, sft_episode)
    
    # 同时保存为.json格式便于查看
    json_path = output_path / f"{episode_id}.json"
    # 转换numpy数组为列表以便JSON序列化
    json_data = {
        'steps': [],
        'episode_metadata': sft_episode['episode_metadata']
    }
    
    for step in sft_episode['steps']:
        json_step = {
            'observation': {
                'image_shape': step['observation']['image'].shape,
                'image_dtype': str(step['observation']['image'].dtype),
            },
            'action': step['action'].tolist(),
            'language_instruction': step['language_instruction'],
        }
        json_data['steps'].append(json_step)
    
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2)


def process_ppo_data_directory(ppo_data_dir: Path, output_dir: Path, min_success_steps: int = 6) -> Dict[str, Any]:
    """
    处理PPO数据目录
    
    Args:
        ppo_data_dir: PPO数据目录
        output_dir: 输出目录
        min_success_steps: 最小连续成功步数
        
    Returns:
        处理统计信息
    """
    stats = {
        'total_files': 0,
        'successful_episodes': 0,
        'failed_episodes': 0,
        'filtered_episodes': 0,
        'error_episodes': 0,
    }
    
    # 查找所有数据文件
    data_files = []
    for ext in ['*.npy', '*.npz']:
        data_files.extend(glob.glob(str(ppo_data_dir / ext)))
    
    print(f"Found {len(data_files)} data files in {ppo_data_dir}")
    
    # 视频目录（假设为 ppo_data_dir/videos）
    video_dir = ppo_data_dir / "videos"
    output_video_dir = output_dir / "videos"
    output_video_dir.mkdir(parents=True, exist_ok=True)
    
    # 记录成功episode的原始路径和新路径
    success_records = []
    
    # 处理每个文件
    for file_path in tqdm(data_files, desc="Processing PPO episodes"):
        stats['total_files'] += 1
        episode_path = Path(file_path)
        
        try:
            # 加载PPO数据
            ppo_data = load_ppo_episode_data(episode_path)
            if ppo_data is None:
                stats['error_episodes'] += 1
                continue
            
            # 检查是否包含必要字段
            required_fields = ['image', 'action', 'info', 'instruction']
            if not all(field in ppo_data for field in required_fields):
                print(f"Missing required fields in {episode_path}")
                stats['error_episodes'] += 1
                continue
            
            # 转换为SFT格式
            episode_id = episode_path.stem
            sft_episode = convert_ppo_to_sft_format(ppo_data, episode_id)
            
            if sft_episode is None:
                stats['failed_episodes'] += 1
                continue
            
            # 保存SFT格式数据
            save_sft_episode(sft_episode, output_dir, episode_id)
            stats['successful_episodes'] += 1
            
            # 检查是否被过滤
            original_length = sft_episode['episode_metadata']['original_length']
            filtered_length = sft_episode['episode_metadata']['filtered_length']
            if filtered_length < original_length:
                stats['filtered_episodes'] += 1
            
            # 自动复制视频
            video_name = episode_id.replace("data", "video") + ".mp4"
            src_video = video_dir / video_name
            dst_video = output_video_dir / video_name
            video_copied = False
            if src_video.exists():
                shutil.copy(src_video, dst_video)
                video_copied = True
            
            # 记录路径
            success_records.append({
                "episode_id": episode_id,
                "original_data": str(episode_path.resolve()),
                "original_video": str(src_video.resolve()),
                "aligned_npz": str((output_dir / f"{episode_id}.npz").resolve()),
                "aligned_video": str(dst_video.resolve()) if video_copied else "NOT_FOUND"
            })
            
        except Exception as e:
            print(f"Error processing {episode_path}: {e}")
            stats['error_episodes'] += 1
    
    # 保存成功episode的路径映射
    record_path = output_dir / "success_episodes.txt"
    with open(record_path, "w") as f:
        f.write("episode_id\toriginal_data\toriginal_video\taligned_npz\taligned_video\n")
        for rec in success_records:
            f.write(f"{rec['episode_id']}\t{rec['original_data']}\t{rec['original_video']}\t{rec['aligned_npz']}\t{rec['aligned_video']}\n")
    
    return stats


def create_dataset_info(output_dir: Path, stats: Dict[str, Any]):
    """
    创建数据集信息文件
    
    Args:
        output_dir: 输出目录
        stats: 处理统计信息
    """
    dataset_info = {
        'dataset_name': 'ppo_rollout_aligned',
        'description': 'PPO rollout success episodes aligned to SFT format',
        'creation_date': str(Path().cwd()),
        'statistics': stats,
        'format': {
            'steps': {
                'observation': {
                    'image': 'numpy array (H, W, 3) uint8'
                },
                'action': 'numpy array (7,) float32',
                'language_instruction': 'string'
            },
            'episode_metadata': {
                'file_path': 'string',
                'source': 'string',
                'original_length': 'int',
                'filtered_length': 'int'
            }
        }
    }
    
    info_path = output_dir / 'dataset_info.json'
    with open(info_path, 'w') as f:
        json.dump(dataset_info, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description='Align PPO rollout data to SFT format')
    parser.add_argument('--ppo_data_dir', type=str, required=True,
                       help='Directory containing PPO rollout data')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for aligned SFT data')
    parser.add_argument('--min_success_steps', type=int, default=6,
                       help='Minimum consecutive successful steps (default: 6)')
    parser.add_argument('--pos_thresh', type=float, default=0.01,
                       help='Position threshold for action filtering (default: 0.01)')
    parser.add_argument('--rot_thresh', type=float, default=0.06,
                       help='Rotation threshold for action filtering (default: 0.06)')
    
    args = parser.parse_args()
    
    ppo_data_dir = Path(args.ppo_data_dir)
    output_dir = Path(args.output_dir)
    
    if not ppo_data_dir.exists():
        print(f"Error: PPO data directory {ppo_data_dir} does not exist")
        return
    
    print(f"Processing PPO data from: {ppo_data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Min success steps: {args.min_success_steps}")
    print(f"Position threshold: {args.pos_thresh}")
    print(f"Rotation threshold: {args.rot_thresh}")
    print("-" * 50)
    
    # 处理数据
    stats = process_ppo_data_directory(ppo_data_dir, output_dir, args.min_success_steps)
    
    # 创建数据集信息
    create_dataset_info(output_dir, stats)
    
    # 打印统计信息
    print("\n" + "=" * 50)
    print("Processing Statistics:")
    print(f"Total files: {stats['total_files']}")
    print(f"Successful episodes: {stats['successful_episodes']}")
    print(f"Failed episodes: {stats['failed_episodes']}")
    print(f"Filtered episodes: {stats['filtered_episodes']}")
    print(f"Error episodes: {stats['error_episodes']}")
    print(f"Success rate: {stats['successful_episodes']/stats['total_files']*100:.2f}%")
    print("=" * 50)
    
    print(f"\nAligned SFT data saved to: {output_dir}")
    print("You can now use this data for further training!")


if __name__ == "__main__":
    main() 