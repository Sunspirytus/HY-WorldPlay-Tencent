#!/bin/bash

# 批量生成脚本
# 用法: ./batch_generate.sh [输入目录] [输出目录]

# 设置默认值
INPUT_DIR=${1:-./millitary_assets/}  # 输入目录，包含txt和png文件
OUTPUT_DIR=${2:-./millitary_output/}  # 输出目录
SCRIPT_DIR=$(dirname "$0")

# 导出环境变量
export T2V_REWRITE_BASE_URL="<your_vllm_server_base_url>"
export T2V_REWRITE_MODEL_NAME="<your_model_name>"
export I2V_REWRITE_BASE_URL="<your_vllm_server_base_url>"
export I2V_REWRITE_MODEL_NAME="<your_model_name>"

# 模型配置
SEED=3479
ASPECT_RATIO=16:9
RESOLUTION=480p
MODEL_PATH=./models/tencent---HunyuanVideo-1.5
AR_ACTION_MODEL_PATH=./models/tencent---HY-WorldPlay/ar_model/diffusion_pytorch_model.safetensors
BI_ACTION_MODEL_PATH=
AR_DISTILL_ACTION_MODEL_PATH=./models/tencent---HY-WorldPlay/ar_distilled_action_model/diffusion_pytorch_model.safetensors
POSE='w-23'
NUM_FRAMES=93
WIDTH=832
HEIGHT=480
N_INFERENCE_GPU=8
REWRITE=false
ENABLE_SR=false

# 确保输出目录存在
mkdir -p "$OUTPUT_DIR"

# 创建日志文件
LOG_FILE="$OUTPUT_DIR/batch_generate_$(date +%Y%m%d_%H%M%S).log"
echo "开始批量处理 - $(date)" | tee -a "$LOG_FILE"
echo "输入目录: $INPUT_DIR" | tee -a "$LOG_FILE"
echo "输出目录: $OUTPUT_DIR" | tee -a "$LOG_FILE"

# 查找所有txt文件
txt_files=("$INPUT_DIR"/*.txt)
total_files=${#txt_files[@]}

if [ $total_files -eq 0 ]; then
    echo "错误: 在 $INPUT_DIR 中没有找到txt文件" | tee -a "$LOG_FILE"
    exit 1
fi

echo "找到 $total_files 个txt文件" | tee -a "$LOG_FILE"

# 处理计数器
processed=0
failed=0
success=0

# 批量处理函数
process_file() {
    local txt_file="$1"
    local base_name=$(basename "$txt_file" .txt)
    local image_path="$INPUT_DIR/$base_name.png"
    
    # 检查对应的图片文件是否存在
    if [ ! -f "$image_path" ]; then
        echo "警告: 找不到图片文件 $image_path，跳过 $txt_file" | tee -a "$LOG_FILE"
        return 1
    fi
    
    # 读取prompt
    local PROMPT=$(cat "$txt_file")
    if [ -z "$PROMPT" ]; then
        echo "警告: $txt_file 文件为空，跳过" | tee -a "$LOG_FILE"
        return 1
    fi
    
    echo "处理文件: $base_name" | tee -a "$LOG_FILE"
    echo "图片路径: $image_path" | tee -a "$LOG_FILE"
    echo "prompt长度: ${#PROMPT} 字符" | tee -a "$LOG_FILE"
    
    # 为每个文件创建独立的输出目录
    local file_output_dir="$OUTPUT_DIR/$base_name"
    mkdir -p "$file_output_dir"
    
    # 执行生成命令
    echo "开始生成视频..." | tee -a "$LOG_FILE"
    
    torchrun --nproc_per_node=$N_INFERENCE_GPU "$SCRIPT_DIR/hyvideo/generate.py" \
      --prompt "$PROMPT" \
      --image_path "$image_path" \
      --resolution $RESOLUTION \
      --aspect_ratio $ASPECT_RATIO \
      --video_length $NUM_FRAMES \
      --seed $SEED \
      --rewrite $REWRITE \
      --sr $ENABLE_SR \
      --save_pre_sr_video \
      --pose "$POSE" \
      --output_path "$file_output_dir" \
      --model_path "$MODEL_PATH" \
      --action_ckpt "$AR_DISTILL_ACTION_MODEL_PATH" \
      --few_step true \
      --num_inference_steps 4 \
      --model_type 'ar' \
      --with-ui \
      --use_vae_parallel false \
      --use_sageattn false \
      --use_fp8_gemm false 2>&1 | tee -a "$file_output_dir/generate.log"
    
    local exit_code=${PIPESTATUS[0]}
    
    if [ $exit_code -eq 0 ]; then
        echo "成功生成: $base_name" | tee -a "$LOG_FILE"
        # 检查是否生成了视频文件
        if ls "$file_output_dir"/*.mp4 1> /dev/null 2>&1; then
            echo "视频文件已保存到: $file_output_dir/" | tee -a "$LOG_FILE"
            return 0
        else
            echo "警告: 未找到生成的视频文件" | tee -a "$LOG_FILE"
            return 1
        fi
    else
        echo "生成失败: $base_name (退出码: $exit_code)" | tee -a "$LOG_FILE"
        return 1
    fi
}

# 可选：并行处理控制（如果GPU内存允许）
# 设置最大并行任务数
MAX_PARALLEL=1  # 根据你的GPU内存调整

# 创建函数用于并行处理
run_parallel() {
    local current_jobs=0
    local pid_list=()
    
    for txt_file in "${txt_files[@]}"; do
        # 等待直到有空闲的并行槽位
        while [ $current_jobs -ge $MAX_PARALLEL ]; do
            # 检查哪些进程已经结束
            local new_pid_list=()
            local new_current_jobs=0
            
            for pid in "${pid_list[@]}"; do
                if kill -0 $pid 2>/dev/null; then
                    new_pid_list+=($pid)
                    ((new_current_jobs++))
                fi
            done
            
            pid_list=("${new_pid_list[@]}")
            current_jobs=$new_current_jobs
            
            if [ $current_jobs -ge $MAX_PARALLEL ]; then
                sleep 5
            fi
        done
        
        # 启动新任务
        process_file "$txt_file" &
        local pid=$!
        pid_list+=($pid)
        ((current_jobs++))
        ((processed++))
        
        echo "进度: $processed/$total_files" | tee -a "$LOG_FILE"
        sleep 1  # 避免同时启动多个任务导致资源争抢
    done
    
    # 等待所有任务完成
    echo "等待所有任务完成..." | tee -a "$LOG_FILE"
    for pid in "${pid_list[@]}"; do
        wait $pid
        if [ $? -eq 0 ]; then
            ((success++))
        else
            ((failed++))
        fi
    done
}

# 可选：串行处理（更稳定，默认）
run_serial() {
    for txt_file in "${txt_files[@]}"; do
        ((processed++))
        echo "========== 处理文件 $processed/$total_files ==========" | tee -a "$LOG_FILE"
        
        if process_file "$txt_file"; then
            ((success++))
        else
            ((failed++))
        fi
        
        echo "进度: $processed/$total_files" | tee -a "$LOG_FILE"
        echo "" | tee -a "$LOG_FILE"
        
        # 可选：添加延迟以避免过度使用GPU
        sleep 2
    done
}

# 执行批量处理
if [ $MAX_PARALLEL -gt 1 ]; then
    echo "使用并行处理，最大并行数: $MAX_PARALLEL" | tee -a "$LOG_FILE"
    run_parallel
else
    echo "使用串行处理" | tee -a "$LOG_FILE"
    run_serial
fi

# 输出总结
echo "" | tee -a "$LOG_FILE"
echo "========== 批量处理完成 ==========" | tee -a "$LOG_FILE"
echo "总文件数: $total_files" | tee -a "$LOG_FILE"
echo "成功: $success" | tee -a "$LOG_FILE"
echo "失败: $failed" | tee -a "$LOG_FILE"
echo "处理时间: $(date)" | tee -a "$LOG_FILE"

# 如果有失败的，列出失败的文件
if [ $failed -gt 0 ]; then
    echo "失败的文件:" | tee -a "$LOG_FILE"
    grep "失败\|警告" "$LOG_FILE" | grep -v "警告: 找不到图片文件" | tee -a "$LOG_FILE"
fi

# 保存配置文件
echo "保存配置信息..." | tee -a "$LOG_FILE"
cat > "$OUTPUT_DIR/batch_config.txt" << EOF
批量处理配置信息
================
处理时间: $(date)
输入目录: $INPUT_DIR
输出目录: $OUTPUT_DIR
总文件数: $total_files
成功数: $success
失败数: $failed

模型配置:
- 模型路径: $MODEL_PATH
- AR蒸馏模型: $AR_DISTILL_ACTION_MODEL_PATH
- 分辨率: $RESOLUTION
- 宽高比: $ASPECT_RATIO
- 帧数: $NUM_FRAMES
- 种子: $SEED
- 并行GPU数: $N_INFERENCE_GPU
EOF

echo "详细日志已保存到: $LOG_FILE" | tee -a "$LOG_FILE"
echo "配置信息已保存到: $OUTPUT_DIR/batch_config.txt" | tee -a "$LOG_FILE"