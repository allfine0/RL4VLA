#!/usr/bin/env python3
"""
收集wandb目录下所有offline-run的数据文件到同一个文件夹

将所有offline-run-*目录下的数据文件（npz、mp4、yaml等）复制到指定目录，
并重命名以避免冲突。

使用方法:
    python collect_offline_runs.py --wandb_dir /path/to/wandb --output_dir /path/to/output
"""

import os
import shutil
import argparse
from pathlib import Path
from typing import List, Optional
import glob
from tqdm import tqdm


def find_offline_runs(wandb_dir: Path) -> List[Path]:
    """
    查找所有offline-run目录
    
    Args:
        wandb_dir: wandb目录路径
        
    Returns:
        offline-run目录列表
    """
    offline_runs = []
    for item in wandb_dir.iterdir():
        if item.is_dir() and item.name.startswith("offline-run-"):
            offline_runs.append(item)
    
    # 按时间排序
    offline_runs.sort(key=lambda x: x.name)
    return offline_runs


def collect_files_from_run(run_dir: Path, file_extensions: Optional[List[str]] = None) -> List[Path]:
    """
    从单个run目录收集指定扩展名的文件
    
    Args:
        run_dir: run目录路径
        file_extensions: 文件扩展名列表，如['.npz', '.mp4', '.yaml']
        
    Returns:
        文件路径列表
    """
    if file_extensions is None:
        file_extensions = ['.npz', '.npy', '.mp4', '.avi', '.yaml', '.json']
    
    files = []
    for ext in file_extensions:
        pattern = f"**/*{ext}"
        files.extend(run_dir.glob(pattern))
    
    return files


def copy_with_unique_name(src_file: Path, dst_dir: Path, run_name: str) -> Path:
    """
    复制文件到目标目录，使用唯一的文件名
    
    Args:
        src_file: 源文件路径
        dst_dir: 目标目录
        run_name: run名称，用于文件名前缀
        
    Returns:
        目标文件路径
    """
    # 构建新文件名：run名称_原文件名
    new_name = f"{run_name}_{src_file.name}"
    dst_file = dst_dir / new_name
    
    # 如果文件已存在，添加数字后缀
    counter = 1
    while dst_file.exists():
        name_parts = src_file.stem, counter, src_file.suffix
        new_name = f"{run_name}_{name_parts[0]}_{name_parts[1]}{name_parts[2]}"
        dst_file = dst_dir / new_name
        counter += 1
    
    # 复制文件
    shutil.copy2(src_file, dst_file)
    return dst_file


def main():
    parser = argparse.ArgumentParser(description='Collect files from all offline-run directories')
    parser.add_argument('--wandb_dir', type=str, 
                       default='/mnt/workspace/luodx/RL4VLA/SimplerEnv/wandb',
                       help='wandb directory path')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for collected files')
    parser.add_argument('--file_types', type=str, nargs='+',
                       default=['.npz', '.npy', '.mp4', '.avi', '.yaml', '.json'],
                       help='File extensions to collect (default: .npz .npy .mp4 .avi .yaml .json)')
    parser.add_argument('--create_subdirs', action='store_true',
                       help='Create subdirectories by file type')
    
    args = parser.parse_args()
    
    wandb_dir = Path(args.wandb_dir)
    output_dir = Path(args.output_dir)
    
    if not wandb_dir.exists():
        print(f"Error: wandb directory {wandb_dir} does not exist")
        return
    
    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 查找所有offline-run目录
    offline_runs = find_offline_runs(wandb_dir)
    print(f"Found {len(offline_runs)} offline-run directories")
    
    if not offline_runs:
        print("No offline-run directories found")
        return
    
    # 统计信息
    stats = {
        'total_runs': len(offline_runs),
        'total_files': 0,
        'files_by_type': {},
        'copied_files': 0,
        'failed_files': 0,
    }
    
    # 如果需要按文件类型创建子目录
    if args.create_subdirs:
        for ext in args.file_types:
            subdir = output_dir / ext.lstrip('.')
            subdir.mkdir(exist_ok=True)
    
    # 收集文件映射记录
    file_mapping = []
    
    # 处理每个offline-run目录
    for run_dir in tqdm(offline_runs, desc="Processing offline-runs"):
        run_name = run_dir.name
        print(f"\nProcessing: {run_name}")
        
        # 收集文件
        files = collect_files_from_run(run_dir, args.file_types)
        print(f"  Found {len(files)} files")
        
        stats['total_files'] += len(files)
        
        # 复制文件
        for file_path in tqdm(files, desc=f"  Copying from {run_name}", leave=False):
            try:
                # 确定目标目录
                if args.create_subdirs:
                    target_dir = output_dir / file_path.suffix.lstrip('.')
                else:
                    target_dir = output_dir
                
                # 复制文件
                dst_file = copy_with_unique_name(file_path, target_dir, run_name)
                
                # 记录映射
                file_mapping.append({
                    'source': str(file_path.resolve()),
                    'destination': str(dst_file.resolve()),
                    'run_name': run_name,
                    'file_type': file_path.suffix
                })
                
                # 统计
                ext = file_path.suffix
                if ext not in stats['files_by_type']:
                    stats['files_by_type'][ext] = 0
                stats['files_by_type'][ext] += 1
                stats['copied_files'] += 1
                
            except Exception as e:
                print(f"    Error copying {file_path}: {e}")
                stats['failed_files'] += 1
    
    # 保存文件映射记录
    import json
    mapping_file = output_dir / 'file_mapping.json'
    with open(mapping_file, 'w') as f:
        json.dump(file_mapping, f, indent=2)
    
    # 保存统计信息
    stats_file = output_dir / 'collection_stats.json'
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    # 打印统计信息
    print("\n" + "=" * 50)
    print("Collection Statistics:")
    print(f"Total offline-runs processed: {stats['total_runs']}")
    print(f"Total files found: {stats['total_files']}")
    print(f"Files successfully copied: {stats['copied_files']}")
    print(f"Files failed to copy: {stats['failed_files']}")
    print("\nFiles by type:")
    for ext, count in stats['files_by_type'].items():
        print(f"  {ext}: {count}")
    print("=" * 50)
    
    print(f"\nFiles collected to: {output_dir}")
    print(f"File mapping saved to: {mapping_file}")
    print(f"Statistics saved to: {stats_file}")


if __name__ == "__main__":
    main() 