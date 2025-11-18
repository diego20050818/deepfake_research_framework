#!/usr/bin/env python3
"""
video2image_refactored.py

功能：
从目录中读取视频文件（可递归），每隔 n 帧提取一帧保存为图像。
每个视频单独一个输出子文件夹。
所有提取的帧路径与标签信息统一存入一个 CSV（自动去重、可追加）。

用法：
    python video2image_refactored.py --input /path/to/videos --output /path/to/frames --step 30
可选参数与原版一致。
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

import cv2
import pandas as pd
from loguru import logger
from tqdm.contrib.concurrent import process_map
from tqdm.auto import tqdm # 保持引入，尽管主要使用 process_map


# ======= 配置和工具函数 =======

def read_label_file(label_path: Path, video_field: str = None, label_field: str = None) -> pd.DataFrame:
    """读取 CSV 或 JSON 格式的标签文件，并标准化列名"""
    if not label_path.exists():
        raise FileNotFoundError(f"标签文件不存在: {label_path}")
    
    suf = label_path.suffix.lower()
    if suf == ".csv":
        df = pd.read_csv(label_path)
    elif suf in (".json", ".ndjson"):
        with open(label_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        df = pd.DataFrame(raw if isinstance(raw, list) else dict(raw))
    else:
        raise ValueError("仅支持 .csv 或 .json 标签文件")

    # 统一列名和字段检测
    colmap = {c.lower(): c for c in df.columns}
    
    def find_field(target_field: Optional[str], default_alts: List[str]) -> str:
        if target_field and target_field in df.columns:
            return target_field
        if target_field and target_field not in df.columns:
             raise ValueError(f"指定的字段 '{target_field}' 在标签文件中不存在")
             
        for alt in default_alts:
            if alt in colmap:
                return colmap[alt]
        return "" # 返回空字符串表示未找到

    actual_video_field = find_field(video_field, ["video", "filename", "file", "video_name", "name", "file path"])
    actual_label_field = find_field(label_field, ["label", "class", "cls", "category"])
            
    if not actual_video_field or not actual_label_field:
        missing = []
        if not actual_video_field: missing.append("视频路径字段")
        if not actual_label_field: missing.append("标签字段")
        raise ValueError(f"标签文件必须包含: {', '.join(missing)}")
        
    df = df.rename(columns={actual_video_field: "video", actual_label_field: "label"})
    df["video"] = df["video"].astype(str).str.strip()
    
    # 标准化标签值
    # 使用 map 批量替换更高效
    label_norm_map = {
        "TRUE": "1", "True": "1", "true": "1", 1: "1", "1": "1",
        "FALSE": "0", "False": "0", "false": "0", 0: "0", "0": "0"
    }
    df["label"] = df["label"].astype(str).str.strip().replace(label_norm_map)

    return df[["video", "label"]]


def find_videos(input_dir: Path, exts: List[str], recursive: bool) -> List[Path]:
    """查找指定目录下所有匹配扩展名的视频"""
    video_list = []
    if recursive:
        for ext in exts:
            video_list.extend(input_dir.rglob(f"*.{ext}"))
    else:
        for ext in exts:
            video_list.extend(input_dir.glob(f"*.{ext}"))
    return video_list


def safe_imwrite(path: Path, image, params: List[int]) -> bool:
    """cv2.imwrite 的安全封装"""
    try:
        # 确保目录存在
        path.parent.mkdir(parents=True, exist_ok=True)
        ok = cv2.imwrite(str(path), image, params)
        if not ok:
            logger.warning(f"写入失败：{path}")
        return ok
    except Exception as e:
        logger.error(f"保存图片出错: {e}")
        return False


# ======= 核心逻辑 =======

# 定义结果记录的类型
# (image_path: str, label_value: str, video_stem: str, frame_index: str)
ExtractionRecord = Tuple[str, str, str, str]

def extract_from_video(
    video_path: Path,
    out_dir: Path,
    step: int,
    jpg_quality: int,
    label_for_video: Optional[str] = None,
    max_frames: Optional[int] = None,
) -> Tuple[int, List[ExtractionRecord]]:
    """从单个视频中提取帧"""
    if step <= 0:
        return 0, [] # step 为非正数时直接返回

    out_sub_dir = out_dir / video_path.stem
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error(f"无法打开视频: {video_path}")
        return 0, []

    params = [int(cv2.IMWRITE_JPEG_QUALITY), int(jpg_quality)]
    frame_idx, saved = 0, 0
    records: List[ExtractionRecord] = []

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % step == 0:
                if max_frames and saved >= max_frames:
                    break
                
                # 图像文件名: {video_stem}_frame{frame_idx:06d}.jpg
                fname = f"{video_path.stem}_frame{frame_idx:06d}.jpg"
                img_path = out_sub_dir / fname
                
                if safe_imwrite(img_path, frame, params):
                    saved += 1
                    label_val = label_for_video or ""
                    records.append((
                        str(img_path),
                        label_val,
                        video_path.stem,
                        f"{frame_idx:06d}",
                    ))
            
            frame_idx += 1
    finally:
        cap.release()

    return saved, records


def process_single_video_wrapper(args: Tuple[Path, Path, int, int, Optional[str], Optional[int]]):
    """
    处理单个视频的包装函数，用于多进程处理。
    **重要：移除所有 print/tqdm 打印，只返回结果，让 process_map 统一管理进度条。**
    """
    video_path, output_dir, step, jpg_quality, label_for_video, max_frames = args
    
    # 不再需要 out_sub, match_by，因为 extract_from_video 内部会处理 out_sub
    saved, recs = extract_from_video(
        video_path,
        output_dir,
        step,
        jpg_quality,
        label_for_video,
        max_frames,
    )
    
    # 移除原代码中的 print 语句，以确保进度条的稳定性
    # print(f"{video_path.name}: 提取 {saved} 张帧") # <--- 移除!
    
    # 我们可以用 logger.debug 记录，但多进程默认可能无法实时显示，
    # 保持沉默是最好的多进程/进度条实践。
    
    return saved, recs

# ======= 主程序 =======

def setup_arg_parser() -> argparse.Namespace:
    """设置和解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="按帧提取视频并生成CSV，支持多进程加速。",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--input", "-i", type=Path, help="输入视频目录",
                        default='/home/liangshuqiao/hong/deepfake/rine/data/FaceForensics++_C23/original')
    parser.add_argument("--output", "-o", type=Path, help="输出图像目录",
                        default='dataset/FF++')
    parser.add_argument("--step", "-s", type=int, help="每隔多少帧保存一张",
                        default=20)
    parser.add_argument("--recursive", "-r", action="store_true", help="递归搜索子目录",
                        default=True)
    parser.add_argument("--ext", default="mp4", help="视频扩展名，逗号分隔（默认mp4）")
    parser.add_argument("--jpg_quality", type=int, default=95, help="JPG质量（0-100）")
    parser.add_argument("--load_file", "-f", action="store_true", help="是否加载标签文件",
                        default=True)
    parser.add_argument("--label_file", type=Path, help="标签文件路径（CSV或JSON）",
                        default='/home/liangshuqiao/hong/deepfake/rine/data/FaceForensics++_C23/csv/original.csv')
    parser.add_argument("--video_field", type=str, help="视频路径字段名（默认自动检测）",
                        default='File Path')
    parser.add_argument("--label_field", type=str, help="标签字段名（默认自动检测）",
                        default='Label')
    parser.add_argument("--max_frames", type=int, default=25, help="每个视频最多提取多少帧")
    parser.add_argument("--match_by", choices=["stem", "name"], default="name", 
                        help="标签匹配方式：\n  - stem: 匹配文件名（不含扩展名）\n  - name: 匹配文件名（含扩展名）")
    parser.add_argument("--workers", type=int, default=4, help="并行处理视频的进程数（默认4）")
    
    args = parser.parse_args()
    return args


def main():
    args = setup_arg_parser()

    
    if not args.input.exists() or not args.input.is_dir():
        logger.error(f"输入目录不存在: {args.input}")
        sys.exit(1)

    args.output.mkdir(parents=True, exist_ok=True)
    exts = [e.strip().lower() for e in args.ext.split(",") if e.strip()]
    if not exts:
        logger.error("未指定有效扩展名")
        sys.exit(1)

    # ===== 读取标签文件 =====
    label_map: Dict[str, str] = {}
    if args.load_file:
        if not args.label_file:
            logger.error("已启用 --load_file，但未提供 --label_file")
            sys.exit(1)
        try:
            df_labels = read_label_file(args.label_file, args.video_field, args.label_field)
            
            # 使用更健壮的方式处理匹配键
            get_key = lambda p: Path(p).stem if args.match_by == "stem" else Path(p).name
            df_labels["video_key"] = df_labels["video"].apply(get_key)
            
            label_map = dict(zip(df_labels["video_key"], df_labels["label"]))
            logger.info(f"标签加载完成，共 {len(label_map)} 条")
        except Exception as e:
            logger.error(f"读取或解析标签文件失败: {e}")
            sys.exit(1)

    # ===== 搜索视频 =====
    videos = find_videos(args.input, exts, args.recursive)
    if not videos:
        logger.warning(f"未在 {args.input} 中找到任何视频文件 ({', '.join(exts)})。")
        sys.exit(0)
    logger.info(f"共找到 {len(videos)} 个视频文件。")

    # ===== 并行处理视频 =====
    # 准备处理参数
    # process_single_video_wrapper 的参数：(Path, Path, int, int, Optional[str], Optional[int])
    process_args = [
        (
            vid,
            args.output,
            args.step,
            args.jpg_quality,
            label_map.get(vid.stem if args.match_by == "stem" else vid.name),
            args.max_frames,
        )
        for vid in videos
    ]

    # 使用 tqdm.contrib.concurrent.process_map 实现带进度条的多进程处理
    logger.info(f"使用 {args.workers} 个进程并行处理 {len(videos)} 个视频...")
    
    # process_map 自动显示进度条，并确保了多进程环境下的稳定性
    results = process_map(
        process_single_video_wrapper, 
        process_args, 
        max_workers=args.workers,
        chunksize=1, # 小 chunksize 有助于更平滑的进度更新
        desc="处理视频",
        unit="video"
    )

    # 收集结果
    all_records: List[ExtractionRecord] = []
    total_saved = 0
    for saved, recs in results:
        total_saved += saved
        all_records.extend(recs)

    # ===== 写出或追加 CSV =====
    csv_path = args.output / "labels.csv"
    if all_records:
        df = pd.DataFrame(all_records, columns=["path", "label", "video", "frame"])
        # 自动去重
        before_dedup = len(df)
        df.drop_duplicates(subset=["path"], inplace=True)
        after_dedup = len(df)
        
        write_header = not csv_path.exists()
        df.to_csv(csv_path, mode="a", header=write_header, index=False)
        
        if before_dedup != after_dedup:
             logger.info(f"去重 {before_dedup - after_dedup} 条。")
        logger.success(f"写入 {len(df)} 条记录到 CSV 文件: {csv_path}")
    else:
        logger.warning("没有生成任何帧。")

    logger.info(f"任务完成，共保存 {total_saved} 张帧。")


if __name__ == "__main__":
    # 使用 try-except 捕获 ctrl+c 或其他异常，确保程序健壮退出
    try:
        main()
    except KeyboardInterrupt:
        logger.error("程序被用户中断 (Ctrl+C)。")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"程序发生未知错误: {e}")
        sys.exit(1)