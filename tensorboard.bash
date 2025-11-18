#!/bin/bash

# 设置日志目录路径
LOG_DIR="./runs"

# 检查日志目录是否存在
if [ ! -d "$LOG_DIR" ]; then
  echo "错误: 日志目录 $LOG_DIR 不存在。"
  exit 1
fi

# 启动 TensorBoard
echo "正在启动 TensorBoard，日志目录: $LOG_DIR"
tensorboard --logdir="$LOG_DIR"