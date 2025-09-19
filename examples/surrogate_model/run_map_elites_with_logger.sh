#!/bin/bash

# MAP-Elites训练器 + 损失记录器启动脚本
# 使用方法: ./run_map_elites_with_logger.sh [模式] [其他参数]

set -e  # 遇到错误时退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 打印带颜色的消息
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查Python环境
check_python_env() {
    print_info "检查Python环境..."
    
    if ! command -v python3 &> /dev/null; then
        print_error "Python3未找到，请确保已安装Python3"
        exit 1
    fi
    
    # 检查虚拟环境
    if [[ "$VIRTUAL_ENV" != "" ]]; then
        print_success "检测到虚拟环境: $VIRTUAL_ENV"
    else
        print_warning "未检测到虚拟环境，建议使用虚拟环境"
    fi
    
    # 检查必要的Python包
    print_info "检查Python包依赖..."
    python3 -c "import torch, numpy, matplotlib" 2>/dev/null || {
        print_error "缺少必要的Python包，请安装: torch, numpy, matplotlib"
        exit 1
    }
    
    print_success "Python环境检查通过"
}

# 显示使用帮助
show_help() {
    echo "MAP-Elites训练器 + 损失记录器启动脚本"
    echo ""
    echo "使用方法:"
    echo "  $0 [模式] [选项]"
    echo ""
    echo "训练模式:"
    echo "  basic       - 基础MAP-Elites训练"
    echo "  advanced    - 高级MAP-Elites训练（交互式配置）"
    echo "  multiprocess- 多进程MAP-Elites训练"
    echo "  shared-ppo  - 共享PPO MAP-Elites训练"
    echo "  custom      - 自定义参数训练"
    echo ""
    echo "选项:"
    echo "  --experiment-name NAME    实验名称"
    echo "  --disable-loss-logging    禁用损失记录"
    echo "  --loss-update-interval N  损失图表更新间隔（秒，默认15）"
    echo "  --help                    显示此帮助信息"
    echo ""
    echo "示例:"
    echo "  $0 basic --experiment-name my_experiment"
    echo "  $0 custom --training-steps-per-individual 5000 --num-generations 30"
    echo "  $0 multiprocess --disable-loss-logging"
}

# 清理函数
cleanup() {
    print_info "正在清理资源..."
    # 杀死可能残留的Python进程
    pkill -f "network_loss_logger.py" 2>/dev/null || true
    pkill -f "map_elites_with_loss_logger.py" 2>/dev/null || true
    print_success "清理完成"
}

# 设置信号处理
trap cleanup EXIT
trap 'print_warning "接收到中断信号，正在清理..."; cleanup; exit 130' INT TERM

# 主函数
main() {
    print_info "🚀 MAP-Elites训练器 + 损失记录器启动脚本"
    print_info "================================================"
    
    # 检查参数
    if [[ $# -eq 0 ]] || [[ "$1" == "--help" ]] || [[ "$1" == "-h" ]]; then
        show_help
        exit 0
    fi
    
    # 检查环境
    check_python_env
    
    # 获取脚本目录
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    PYTHON_SCRIPT="$SCRIPT_DIR/map_elites_with_loss_logger.py"
    
    # 检查Python脚本是否存在
    if [[ ! -f "$PYTHON_SCRIPT" ]]; then
        print_error "找不到Python脚本: $PYTHON_SCRIPT"
        exit 1
    fi
    
    # 创建日志目录
    LOG_DIR="$SCRIPT_DIR/network_loss_logs"
    mkdir -p "$LOG_DIR"
    print_info "日志目录: $LOG_DIR"
    
    # 解析模式
    MODE="$1"
    shift  # 移除第一个参数
    
    print_info "训练模式: $MODE"
    print_info "额外参数: $*"
    
    # 构建Python命令
    PYTHON_CMD="python3 '$PYTHON_SCRIPT' --mode '$MODE'"
    
    # 添加额外参数
    if [[ $# -gt 0 ]]; then
        PYTHON_CMD="$PYTHON_CMD $*"
    fi
    
    print_info "执行命令: $PYTHON_CMD"
    print_info "开始训练..."
    
    # 执行Python脚本
    eval $PYTHON_CMD
    
    EXIT_CODE=$?
    
    if [[ $EXIT_CODE -eq 0 ]]; then
        print_success "训练完成！"
    else
        print_error "训练失败，退出码: $EXIT_CODE"
        exit $EXIT_CODE
    fi
}

# 运行主函数
main "$@"

