#!/usr/bin/env python3
"""
IndexTTS-2 模型预下载脚本

功能：
  1. 自动检测运行环境（系统、虚拟环境、依赖包）
  2. 自动检测网络环境（直连 HF / 需要镜像）
  3. 一次性下载所有依赖模型到 checkpoints/hf_cache/
  4. 支持断点续传和自动重试

用法：
  python download_all.py
  python download_all.py --model-dir /path/to/models
  python download_all.py --force          # 强制重新下载
  python download_all.py --source hf-mirror  # 强制指定下载源
"""

import argparse
import os
import shutil
import sys
import platform
import subprocess
import importlib.util
from pathlib import Path

# ============================================================================
#  全局变量 - 环境信息
# ============================================================================

# --- 网络环境 ---
G_IN_CHINA = True         # 是否在中国大陆网络环境（默认 true）
G_AUTO_DOWNLOAD = False   # 是否自动确认所有下载操作（无需用户输入）

# --- 系统信息 ---
G_OS_NAME = None          # 操作系统名称 (Linux, Windows, Darwin)
G_OS_VERSION = None       # 操作系统版本
G_OS_ARCH = None          # 系统架构 (x86_64, arm64, ...)
G_PYTHON_VERSION = None   # Python 版本字符串
G_PYTHON_EXECUTABLE = None  # 当前 Python 解释器路径

# --- 虚拟环境信息 ---
G_IN_VIRTUALENV = None    # 是否在虚拟环境中
G_VENV_PATH = None        # 虚拟环境根路径
G_VENV_NAME = None        # 虚拟环境名称

# --- 依赖包信息 ---
G_PACKAGES = {}           # 包名 -> {installed: bool, version: str, source: str}
G_MISSING_PACKAGES = []   # 缺失的包列表
G_USE_MODELSCOPE = None   # 是否使用 modelscope 下载
G_USE_HF_HUB = None       # 是否使用 huggingface_hub 下载

# --- 下载配置 ---
G_MODEL_DIR = None        # 模型存放目录
G_CACHE_DIR = None        # 实际缓存目录 (model_dir/hf_cache)
G_EXAMPLES_DIR = None     # 示例音频存放目录


def ask_continue(prompt="是否继续? (y/N): "):
    """
    询问用户是否继续。

    如果 G_AUTO_DOWNLOAD 为 True，直接返回 True。
    否则等待用户输入。
    """
    if G_AUTO_DOWNLOAD:
        return True

    try:
        response = input(f"  {prompt}").strip().lower()
        return response in ("y", "yes")
    except (EOFError, KeyboardInterrupt):
        print("\n  ⊘ 用户取消")
        return False

# --- 硬件信息 ---
G_RAM_TOTAL = None        # 总内存 (bytes)
G_RAM_AVAILABLE = None    # 可用内存 (bytes)
G_DISK_TOTAL = None       # 磁盘总大小 (bytes)
G_DISK_USED = None        # 磁盘已用 (bytes)
G_DISK_FREE = None        # 磁盘可用 (bytes)
G_GPU_NAME = None         # GPU 名称
G_GPU_VRAM = None         # GPU 显存 (bytes)
G_GPU_COUNT = 0           # GPU 数量
G_GPU_ARCH = None         # GPU 架构代号 (Blackwell, Ada, Ampere, Turing, Pascal)
G_CUDA_VERSION = None     # CUDA 版本
G_CUDNN_VERSION = None    # cuDNN 版本
G_NVIDIA_DRIVER = None    # NVIDIA 驱动版本
G_RECOMMENDED_CUDA = None  # 推荐的 CUDA 版本 (如 "cu128")
G_RECOMMENDED_TORCH_INDEX = None  # 推荐的 PyTorch 安装源 URL
G_DISK_INSUFFICIENT = False  # 磁盘空间是否不足

# 总磁盘需求估算（含依赖包 + 全部模型 + 示例音频）：
#   .venv 依赖包 (torch + CUDA + 全部):  ~8.2 GB
#   主模型 (checkpoints/):               ~5.5 GB
#   辅助模型 (hf_cache/):                ~2.5 GB
#   示例音频:                            ~11 MB
#   合计:                                ~16.2 GB
TOTAL_DISK_REQUIREMENT_ESTIMATE = 16.2 * 1024 * 1024 * 1024  # ~16.2 GB

# Python 版本要求
MIN_PYTHON_VERSION = (3, 10)   # 项目最低要求 (pyproject.toml: requires-python >= 3.10)
MAX_PYTHON_VERSION = (3, 13)   # torch 预编译 wheel 最高支持到 3.13

# --- 核心依赖包清单（下载脚本运行所需）---
REQUIRED_PACKAGES = [
    "torch",
    "torchaudio",
    "modelscope",
    "huggingface_hub",
    "transformers",
    "safetensors",
    "gradio",
]

# ============================================================================
#  模型清单
# ============================================================================

# 主模型仓库（IndexTTS-2 核心）
MAIN_MODEL_REPOS = {
    "hf": "IndexTeam/IndexTTS-2",
    "ms": "IndexTeam/IndexTTS-2",
}

# 主模型必需文件清单（webui.py 中 required_files 对应）
MAIN_MODEL_REQUIRED_FILES = [
    "config.yaml",
    "bpe.model",
    "gpt.pth",
    "s2mel.pth",
    "wav2vec2bert_stats.pt",
    "feat1.pt",
    "feat2.pt",
]

# 主模型目录（qwen 情感识别模型）
MAIN_MODEL_REQUIRED_DIRS = [
    "qwen0.6bemo4-merge",
]

# 辅助模型清单（存放在 hf_cache/）
AUX_MODELS = [
    {
        "name": "w2v-bert-2.0",
        "type": "repo",
        "hf_repo": "facebook/w2v-bert-2.0",
        "ms_model": "AI-ModelScope/w2v-bert-2.0",
        "local_dir": "w2v-bert-2.0",
        "min_files": 5,
        "description": "音频特征提取模型",
    },
    {
        "name": "MaskGCT semantic codec",
        "type": "file",
        "hf_repo": "amphion/MaskGCT",
        "hf_file": "semantic_codec/model.safetensors",
        "local_path": "semantic_codec_model.safetensors",
        "min_size": 100 * 1024 * 1024,
        "description": "语义编解码器",
    },
    {
        "name": "CAMPPlus speaker embedding",
        "type": "file",
        "hf_repo": "funasr/campplus",
        "hf_file": "campplus_cn_common.bin",
        "ms_model": "iic/speech_campplus_sv_zh-cn_16k-common",
        "ms_file": "campplus_cn_common.bin",
        "local_path": "campplus_cn_common.bin",
        "min_size": 10 * 1024 * 1024,
        "description": "说话人声纹嵌入模型",
    },
    {
        "name": "BigVGAN vocoder",
        "type": "files",
        "hf_repo": "nvidia/bigvgan_v2_22khz_80band_256x",
        "local_dir": "bigvgan",
        "files": [
            {"hf": "config.json",          "local": "config.json",          "min_size": 100},
            {"hf": "bigvgan_generator.pt", "local": "bigvgan_generator.pt", "min_size": 100 * 1024 * 1024},
        ],
        "description": "声码器",
    },
]

# 示例音频文件（WebUI 演示用）
EXAMPLE_AUDIO_FILES = [
    "voice_01.wav",
    "voice_02.wav",
    "voice_03.wav",
    "voice_04.wav",
    "voice_05.wav",
    "voice_06.wav",
    "voice_07.wav",
    "voice_08.wav",
    "voice_09.wav",
    "voice_11.wav",
    "voice_12.wav",
    "emo_sad.wav",
    "emo_hate.wav",
]


# ============================================================================
#  Python 版本检查
# ============================================================================

def check_python_compatibility():
    """
    检查 Python 版本是否满足项目要求。

    要求: Python >= 3.10 且 <= 3.13（torch 预编译 wheel 支持范围）

    返回:
        (ok: bool, message: str, level: str)
        level: "error" (不兼容，直接退出), "ok" (完全兼容)
    """
    version = (sys.version_info.major, sys.version_info.minor)

    if version < MIN_PYTHON_VERSION:
        return (False,
            f"Python 版本过低！需要 >= {MIN_PYTHON_VERSION[0]}.{MIN_PYTHON_VERSION[1]}，"
            f"当前为 {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "error")

    if version > MAX_PYTHON_VERSION:
        return (False,
            f"Python 版本过高！torch 预编译 wheel 最高支持到 {MAX_PYTHON_VERSION[0]}.{MAX_PYTHON_VERSION[1]}，"
            f"当前为 {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "error")

    return (True, "Python 版本符合要求", "ok")


# ============================================================================
#  环境检测函数
# ============================================================================

def detect_system_info():
    """检测系统和 Python 版本信息"""
    global G_OS_NAME, G_OS_VERSION, G_OS_ARCH
    global G_PYTHON_VERSION, G_PYTHON_EXECUTABLE

    G_OS_NAME = platform.system()
    G_OS_VERSION = platform.version()
    G_OS_ARCH = platform.machine()
    G_PYTHON_VERSION = platform.python_version()
    G_PYTHON_EXECUTABLE = sys.executable


def detect_virtualenv():
    """检测是否在虚拟环境中，如果是则获取虚拟环境路径"""
    global G_IN_VIRTUALENV, G_VENV_PATH, G_VENV_NAME

    # 判断是否在虚拟环境中
    G_IN_VIRTUALENV = (
        hasattr(sys, "real_prefix") or          # virtualenv
        (hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix)  # venv
    )

    if G_IN_VIRTUALENV:
        # 虚拟环境根目录是 prefix 的父目录（Linux/Mac）或 prefix 本身（Windows）
        venv_base = Path(sys.prefix)
        # 常见虚拟环境结构：venv_base/bin/python 或 venv_base/Scripts/python
        G_VENV_PATH = str(venv_base.parent) if venv_base.name in ("bin", "Scripts") else str(venv_base)
        G_VENV_NAME = Path(G_VENV_PATH).name
    else:
        G_VENV_PATH = None
        G_VENV_NAME = None


def check_package(pkg_name):
    """
    检查单个包是否已安装。

    返回字典：{installed: bool, version: str, source: str}
    source 为 "system" 或 "venv"
    """
    result = {
        "installed": False,
        "version": None,
        "source": "venv" if G_IN_VIRTUALENV else "system",
    }

    # 尝试通过 importlib 查找
    spec = importlib.util.find_spec(pkg_name)
    if spec is None:
        return result

    result["installed"] = True

    # 尝试获取版本号
    try:
        module = importlib.import_module(pkg_name)
        result["version"] = getattr(
            module, "__version__",
            getattr(module, "__version_info__", "unknown")
        )
    except Exception:
        result["version"] = "installed (version unknown)"

    return result


def detect_packages():
    """检测所有核心依赖包的安装状态"""
    global G_PACKAGES, G_MISSING_PACKAGES, G_USE_MODELSCOPE, G_USE_HF_HUB

    G_PACKAGES = {}
    G_MISSING_PACKAGES = []

    for pkg in REQUIRED_PACKAGES:
        info = check_package(pkg)
        G_PACKAGES[pkg] = info
        if not info["installed"]:
            G_MISSING_PACKAGES.append(pkg)

    # 根据检测结果设置下载策略标志
    G_USE_MODELSCOPE = G_PACKAGES.get("modelscope", {}).get("installed", False)
    G_USE_HF_HUB = G_PACKAGES.get("huggingface_hub", {}).get("installed", False)


def detect_system_python_packages():
    """
    检测系统 Python（非虚拟环境）中安装了哪些包。
    返回包名 -> version 的字典。
    """
    system_python = None

    # 尝试找到系统 Python
    if G_IN_VIRTUALENV:
        # 尝试通过 pyenv/uv 等工具找到系统 Python
        candidates = [
            "python3",
            "/usr/bin/python3",
        ]
        for cand in candidates:
            try:
                result = subprocess.run(
                    [cand, "--version"],
                    capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0:
                    system_python = cand
                    break
            except Exception:
                continue

    if system_python is None:
        return {}

    system_packages = {}
    for pkg in REQUIRED_PACKAGES:
        try:
            result = subprocess.run(
                [system_python, "-c", f"import {pkg}; print({pkg}.__version__)"],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0 and result.stdout.strip():
                system_packages[pkg] = result.stdout.strip()
        except Exception:
            pass

    return system_packages


# ============================================================================
#  硬件检测函数
# ============================================================================

def detect_hardware(model_dir: str):
    """
    检测硬件信息：内存、磁盘、GPU、CUDA/cuDNN、NVIDIA 驱动。

    参数：
        model_dir: 模型存放目录（用于检测磁盘空间）
    """
    global G_RAM_TOTAL, G_RAM_AVAILABLE
    global G_DISK_TOTAL, G_DISK_USED, G_DISK_FREE
    global G_GPU_NAME, G_GPU_VRAM, G_GPU_COUNT
    global G_CUDA_VERSION, G_CUDNN_VERSION, G_NVIDIA_DRIVER
    global G_DISK_INSUFFICIENT

    # --- (a) 内存检测 ---
    _detect_memory()

    # --- (b) 磁盘检测 ---
    _detect_disk(model_dir)

    # --- (c) GPU / CUDA 检测 ---
    _detect_gpu()


def _detect_memory():
    """检测系统内存信息"""
    global G_RAM_TOTAL, G_RAM_AVAILABLE

    # Linux: 读取 /proc/meminfo
    if os.path.exists("/proc/meminfo"):
        try:
            meminfo = {}
            with open("/proc/meminfo", "r") as f:
                for line in f:
                    parts = line.split()
                    key = parts[0].rstrip(":")
                    value = int(parts[1])  # kB
                    meminfo[key] = value * 1024  # 转为 bytes
            G_RAM_TOTAL = meminfo.get("MemTotal")
            G_RAM_AVAILABLE = meminfo.get("MemAvailable", meminfo.get("MemFree"))
            return
        except Exception:
            pass

    # macOS / 通用回退: os.sysconf
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        page_count = os.sysconf("SC_PHYS_PAGES")
        G_RAM_TOTAL = page_size * page_count
        G_RAM_AVAILABLE = G_RAM_TOTAL  # 无法精确获取可用内存
        return
    except Exception:
        pass

    # Windows 回退: subprocess 调用 systeminfo
    try:
        result = subprocess.run(
            ["systeminfo"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            import re
            total_match = re.search(r"Total Physical Memory:\s+([\d,]+)\s+MB", result.stdout)
            avail_match = re.search(r"Available Physical Memory:\s+([\d,]+)\s+MB", result.stdout)
            if total_match:
                G_RAM_TOTAL = int(total_match.group(1).replace(",", "")) * 1024 * 1024
            if avail_match:
                G_RAM_AVAILABLE = int(avail_match.group(1).replace(",", "")) * 1024 * 1024
            return
    except Exception:
        pass

    # 最后回退: psutil（如果安装了）
    try:
        import psutil
        vm = psutil.virtual_memory()
        G_RAM_TOTAL = vm.total
        G_RAM_AVAILABLE = vm.available
    except Exception:
        pass


def _detect_disk(model_dir: str):
    """检测磁盘空间信息"""
    global G_DISK_TOTAL, G_DISK_USED, G_DISK_FREE
    global G_DISK_INSUFFICIENT

    try:
        usage = shutil.disk_usage(model_dir)
        G_DISK_TOTAL = usage.total
        G_DISK_USED = usage.used
        G_DISK_FREE = usage.free
    except Exception:
        try:
            usage = shutil.disk_usage("/")
            G_DISK_TOTAL = usage.total
            G_DISK_USED = usage.used
            G_DISK_FREE = usage.free
        except Exception:
            pass

    # 检查磁盘空间是否充足（需要 3 倍于模型总大小）
    required_space = TOTAL_DISK_REQUIREMENT_ESTIMATE * 3
    if G_DISK_FREE is not None and G_DISK_FREE < required_space:
        G_DISK_INSUFFICIENT = True


def _detect_gpu():
    """
    检测 GPU 信息，并根据 GPU 型号 + 驱动版本推荐 torch/CUDA 版本。

    检测策略（优先级从高到低）：
      1. nvidia-smi → 最准确，含驱动版本 + GPU 型号
      2. torch.cuda → 如果 torch 已安装
      3. 都不可用 → 标记为无 GPU
    """
    global G_GPU_NAME, G_GPU_VRAM, G_GPU_COUNT, G_GPU_ARCH
    global G_CUDA_VERSION, G_CUDNN_VERSION, G_NVIDIA_DRIVER
    global G_RECOMMENDED_CUDA, G_RECOMMENDED_TORCH_INDEX

    nvidia_smi_ok = False

    # --- 策略 1: nvidia-smi（最准确）---
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0 and result.stdout.strip():
            gpu_lines = [l for l in result.stdout.strip().split("\n") if l.strip()]
            G_GPU_COUNT = len(gpu_lines)
            # 取第一张 GPU 的信息
            lines = gpu_lines[0].split(", ")
            if len(lines) >= 3:
                G_GPU_NAME = lines[0].strip()
                G_NVIDIA_DRIVER = lines[1].strip()
                # 解析显存: "32607 MiB" → bytes
                mem_str = lines[2].strip()
                try:
                    mem_mb = float(mem_str.split()[0])
                    G_GPU_VRAM = int(mem_mb * 1024 * 1024)
                except ValueError:
                    pass
                nvidia_smi_ok = True
    except Exception:
        pass

    # --- 策略 2: torch.cuda（无论 nvidia-smi 是否可用，都尝试获取已装版本）---
    try:
        import torch
        if torch.cuda.is_available():
            # 如果 nvidia-smi 已提供 GPU 信息，不再覆盖
            if not nvidia_smi_ok:
                G_GPU_COUNT = torch.cuda.device_count()
                if G_GPU_COUNT > 0:
                    G_GPU_NAME = torch.cuda.get_device_name(0)
                    props = torch.cuda.get_device_properties(0)
                    G_GPU_VRAM = props.total_memory
            # 始终尝试获取已安装的 CUDA/cuDNN 版本
            if not G_CUDA_VERSION:
                G_CUDA_VERSION = torch.version.cuda
            if not G_CUDNN_VERSION:
                try:
                    cudnn_ver = torch.backends.cudnn.version()
                    G_CUDNN_VERSION = str(cudnn_ver) if cudnn_ver else None
                except Exception:
                    pass
        else:
            # CUDA 不可用，但仍可能有版本信息（安装了 CUDA 版但无 GPU）
            if not G_CUDA_VERSION:
                G_CUDA_VERSION = torch.version.cuda
            if not G_CUDNN_VERSION:
                try:
                    cudnn_ver = torch.backends.cudnn.version()
                    G_CUDNN_VERSION = str(cudnn_ver) if cudnn_ver else None
                except Exception:
                    pass
    except Exception:
        pass

    # --- 识别 GPU 架构 ---
    if G_GPU_NAME:
        G_GPU_ARCH = _identify_gpu_architecture(G_GPU_NAME)

    # --- 根据 GPU 架构 + 驱动版本推荐 CUDA/torch 版本 ---
    if G_GPU_ARCH and G_NVIDIA_DRIVER:
        G_RECOMMENDED_CUDA, G_RECOMMENDED_TORCH_INDEX = _recommend_torch_cuda(
            G_GPU_ARCH, G_NVIDIA_DRIVER
        )
    elif G_GPU_ARCH:
        # 没有驱动版本信息，使用保守推荐
        G_RECOMMENDED_CUDA, G_RECOMMENDED_TORCH_INDEX = _recommend_torch_cuda(
            G_GPU_ARCH, "0.0"
        )


def _identify_gpu_architecture(gpu_name: str) -> str:
    """
    根据 GPU 名称识别架构代号。

    返回: Blackwell, Ada, Ampere, Turing, Pascal, Volta, 或 Unknown
    """
    name_lower = gpu_name.lower()

    # RTX 50xx 系列 (Blackwell)
    if "rtx 50" in name_lower or "rtx50" in name_lower:
        return "Blackwell"

    # RTX 40xx 系列 (Ada Lovelace)
    if "rtx 40" in name_lower or "rtx40" in name_lower:
        return "Ada"

    # RTX 30xx 系列 (Ampere)
    if "rtx 30" in name_lower or "rtx30" in name_lower:
        return "Ampere"

    # RTX 20xx 系列 (Turing)
    if "rtx 20" in name_lower or "rtx20" in name_lower:
        return "Turing"

    # GTX 10xx / 16xx 系列 (Pascal / Turing)
    if "gtx 16" in name_lower or "gtx16" in name_lower:
        return "Turing"
    if "gtx 10" in name_lower or "gtx10" in name_lower:
        return "Pascal"

    # GTX 9xx 系列 (Maxwell)
    if "gtx 9" in name_lower or "gtx9" in name_lower:
        return "Maxwell"

    # GTX 10xx 系列 (Pascal)
    if "gtx" in name_lower or "gtx" in name_lower:
        return "Pascal"

    # Quadro / Tesla / A100 / H100 等专业卡
    if "a100" in name_lower:
        return "Ampere"
    if "h100" in name_lower:
        return "Hopper"
    if "v100" in name_lower:
        return "Volta"
    if "a6000" in name_lower or "a40" in name_lower or "a10" in name_lower:
        return "Ampere"
    if "l40" in name_lower or "l4" in name_lower:
        return "Ada"

    return "Unknown"


# NVIDIA 驱动版本 → 最高支持的 CUDA 版本
# 数据来源: https://docs.nvidia.com/deploy/cuda-compatibility/
DRIVER_CUDA_COMPAT = [
    (590, "13.0"),
    (560, "12.7"),
    (550, "12.6"),
    (545, "12.5"),
    (535, "12.4"),
    (530, "12.3"),
    (525, "12.2"),
    (520, "12.1"),
    (510, "12.0"),
    (470, "11.8"),
    (450, "11.7"),
    (440, "11.6"),
    (418, "11.0"),
    (396, "10.2"),
    (384, "10.0"),
    (367, "9.0"),
    (361, "8.0"),
]

# GPU 架构 → 最低需要的 CUDA 版本
ARCH_MIN_CUDA = {
    "Blackwell": "12.0",   # RTX 50xx
    "Ada": "11.4",        # RTX 40xx
    "Ampere": "11.0",     # RTX 30xx, A100
    "Turing": "10.0",     # RTX 20xx, GTX 16xx
    "Volta": "9.0",       # V100
    "Pascal": "8.0",      # GTX 10xx
    "Maxwell": "7.0",     # GTX 9xx
}

# 可用的 PyTorch CUDA 构建版本（从最新到最旧）
# 格式: (cuda_tag, torch_version, official_url, aliyun_url)
AVAILABLE_TORCH_CUDA = [
    ("cu128", "2.8", "https://download.pytorch.org/whl/cu128", "https://mirrors.aliyun.com/pytorch-wheels/cu128"),
    ("cu126", "2.7", "https://download.pytorch.org/whl/cu126", "https://mirrors.aliyun.com/pytorch-wheels/cu126"),
    ("cu124", "2.6", "https://download.pytorch.org/whl/cu124", "https://mirrors.aliyun.com/pytorch-wheels/cu124"),
    ("cu121", "2.4", "https://download.pytorch.org/whl/cu121", "https://mirrors.aliyun.com/pytorch-wheels/cu121"),
    ("cu118", "2.4", "https://download.pytorch.org/whl/cu118", "https://mirrors.aliyun.com/pytorch-wheels/cu118"),
]


def _parse_driver_major(driver_version: str) -> int:
    """解析驱动版本号，返回主版本号。如 "550.54.15" → 550"""
    try:
        return int(driver_version.split(".")[0])
    except (ValueError, IndexError):
        return 0


def _parse_cuda_version(cuda_ver: str) -> float:
    """解析 CUDA 版本字符串为浮点数。如 "12.4" → 12.4"""
    try:
        parts = cuda_ver.split(".")
        return int(parts[0]) * 10 + int(parts[1]) if len(parts) >= 2 else int(parts[0]) * 10
    except (ValueError, IndexError):
        return 0


def _recommend_torch_cuda(gpu_arch: str, driver_version: str):
    """
    根据 GPU 架构和驱动版本，推荐最佳的 CUDA 和 PyTorch 版本。

    返回: (cuda_tag, torch_index_url)
        cuda_tag: 如 "cu128", "cu124"
        torch_index_url: PyTorch 安装源 URL（国内使用阿里云镜像）
    """
    driver_major = _parse_driver_major(driver_version)

    # 确定驱动支持的最高 CUDA 版本
    max_cuda = "11.0"  # 默认最低
    for drv_ver, cuda_ver in DRIVER_CUDA_COMPAT:
        if driver_major >= drv_ver:
            max_cuda = cuda_ver
            break

    # 确定 GPU 架构需要的最低 CUDA 版本
    min_cuda = ARCH_MIN_CUDA.get(gpu_arch, "11.0")

    # 从可用版本中选择：最高但不超过驱动支持的版本
    for cuda_tag, torch_ver, official_url, aliyun_url in AVAILABLE_TORCH_CUDA:
        # 解析 cuda_tag: "cu128" → "12.8", "cu118" → "11.8"
        cuda_num = cuda_tag.replace("cu", "")
        cuda_major = int(cuda_num[:-1])
        cuda_minor = int(cuda_num[-1])
        cuda_ver_str = f"{cuda_major}.{cuda_minor}"

        # 检查驱动是否支持这个 CUDA 版本
        if _parse_cuda_version(cuda_ver_str) <= _parse_cuda_version(max_cuda):
            # 检查 GPU 架构是否支持这个 CUDA 版本
            if _parse_cuda_version(cuda_ver_str) >= _parse_cuda_version(min_cuda):
                # 国内使用阿里云镜像，海外使用官方源
                if G_IN_CHINA:
                    return cuda_tag, aliyun_url
                else:
                    return cuda_tag, official_url

    # 如果找不到完美匹配，使用 cu118（最广泛兼容）
    if G_IN_CHINA:
        return "cu118", AVAILABLE_TORCH_CUDA[-1][3]
    else:
        return "cu118", AVAILABLE_TORCH_CUDA[-1][2]


# ============================================================================
#  打印环境信息
# ============================================================================

def print_separator(char="=", length=60):
    """打印分隔线"""
    print(char * length)


def print_env_info():
    """打印完整的环境检测信息"""
    print_separator()
    print("  IndexTTS-2 环境检测")
    print_separator()
    print()

    # --- Python 版本兼容性检查 ---
    py_ok, py_msg, py_level = check_python_compatibility()
    if py_level == "error":
        print(f"[✗ Python 版本不兼容]")
        print(f"  {py_msg}")
        print(f"  项目要求: Python {MIN_PYTHON_VERSION[0]}.{MIN_PYTHON_VERSION[1]} ~ {MAX_PYTHON_VERSION[0]}.{MAX_PYTHON_VERSION[1]}")
        print()

    # --- 系统信息 ---
    print("[网络环境]")
    print(f"  中国大陆:    {'是' if G_IN_CHINA else '否'}")
    print()

    print("[系统信息]")
    print(f"  操作系统:    {G_OS_NAME} {G_OS_VERSION}")
    print(f"  架构:        {G_OS_ARCH}")
    print(f"  Python:      {G_PYTHON_VERSION}")
    print(f"  解释器路径:  {G_PYTHON_EXECUTABLE}")
    print()

    # --- 硬件信息 ---
    print("[硬件信息]")

    # 内存
    if G_RAM_TOTAL:
        ram_total_gb = G_RAM_TOTAL / 1024**3
        ram_avail_gb = G_RAM_AVAILABLE / 1024**3 if G_RAM_AVAILABLE else 0
        print(f"  内存:          {ram_total_gb:.1f} GB (可用 {ram_avail_gb:.1f} GB)")
    else:
        print(f"  内存:          未检测到")

    # 磁盘
    if G_DISK_TOTAL:
        disk_total_gb = G_DISK_TOTAL / 1024**3
        disk_used_gb = G_DISK_USED / 1024**3
        disk_free_gb = G_DISK_FREE / 1024**3
        print(f"  磁盘:          {disk_total_gb:.1f} GB (已用 {disk_used_gb:.1f} GB, 可用 {disk_free_gb:.1f} GB)")
    else:
        print(f"  磁盘:          未检测到")

    # GPU
    if G_GPU_NAME:
        vram_gb = G_GPU_VRAM / 1024**3 if G_GPU_VRAM else 0
        gpu_info = f"{G_GPU_NAME} ({vram_gb:.1f} GB)"
        if G_GPU_COUNT > 1:
            gpu_info += f"  × {G_GPU_COUNT}"
        if G_GPU_ARCH:
            gpu_info += f" [{G_GPU_ARCH}]"
        print(f"  GPU:           {gpu_info}")
    else:
        print(f"  GPU:           未检测到 NVIDIA GPU")

    # CUDA
    if G_CUDA_VERSION:
        print(f"  已装 CUDA:     {G_CUDA_VERSION}")
    else:
        print(f"  已装 CUDA:     未检测到 (torch 未安装或无 CUDA)")

    # cuDNN
    print(f"  已装 cuDNN:    {G_CUDNN_VERSION if G_CUDNN_VERSION else '未检测到'}")

    # NVIDIA 驱动
    if G_NVIDIA_DRIVER:
        print(f"  NVIDIA 驱动:   {G_NVIDIA_DRIVER}")
    else:
        print(f"  NVIDIA 驱动:   未检测到 (nvidia-smi 不可用)")

    # 推荐的 torch/CUDA 版本
    if G_RECOMMENDED_CUDA:
        print(f"  推荐 CUDA:     {G_RECOMMENDED_CUDA}")
        print(f"  推荐 torch 源: {G_RECOMMENDED_TORCH_INDEX}")
    elif G_GPU_NAME:
        print(f"  推荐 CUDA:     无法确定 (缺少驱动版本信息)")
    else:
        print(f"  推荐:          CPU 版 torch (无 NVIDIA GPU)")

    print()

    # --- 磁盘空间检查 ---
    print("[磁盘空间检查]")
    total_disk_gb = TOTAL_DISK_REQUIREMENT_ESTIMATE / 1024**3
    required_gb = total_disk_gb * 3
    print(f"  模型下载:      约 8.2 GB")
    print(f"  依赖包:        约 8.0 GB (torch + CUDA + 全部)")
    print(f"  总磁盘需求:    约 {total_disk_gb:.1f} GB")
    print(f"  建议预留:      约 {required_gb:.1f} GB (3x，含临时文件)")

    if G_DISK_FREE is not None:
        disk_free_gb = G_DISK_FREE / 1024**3
        print(f"  可用空间:      {disk_free_gb:.1f} GB")
        if G_DISK_INSUFFICIENT:
            print(f"  状态:          ✗ 空间不足！需要 {required_gb:.1f} GB，仅有 {disk_free_gb:.1f} GB")
        else:
            print(f"  状态:          ✓ 空间充足")
    else:
        print(f"  可用空间:      无法检测")
        print(f"  状态:          ⚠ 请手动确认磁盘空间")

    print()

    # --- 虚拟环境信息 ---
    print("[虚拟环境]")
    if G_IN_VIRTUALENV:
        print(f"  已激活:      是")
        print(f"  环境名称:    {G_VENV_NAME}")
        print(f"  环境路径:    {G_VENV_PATH}")
    else:
        print(f"  已激活:      否 (使用系统 Python)")
    print()

    # --- 依赖包信息 ---
    print("[依赖包状态]")
    print(f"  {'包名':<20} {'已安装':<8} {'版本':<20} {'来源':<8}")
    print(f"  {'-'*18}  {'-'*6}  {'-'*18}  {'-'*6}")

    for pkg_name, info in G_PACKAGES.items():
        status = "✓" if info["installed"] else "✗"
        version = str(info["version"]) if info["version"] else "-"
        source = info["source"]
        print(f"  {pkg_name:<20} {status:<8} {version:<20} {source:<8}")

    print()

    # --- 系统 Python 包（如果在虚拟环境中）---
    if G_IN_VIRTUALENV:
        system_pkgs = detect_system_python_packages()
        if system_pkgs:
            print("[系统 Python 已安装的包]")
            for pkg, ver in system_pkgs.items():
                print(f"  {pkg}: {ver}")
            print()

    # --- 下载策略 ---
    print("[下载策略]")
    print(f"  可用下载源: ", end="")
    sources = []
    if G_USE_MODELSCOPE:
        sources.append("ModelScope ✓")
    else:
        sources.append("ModelScope ✗ (未安装)")
    if G_USE_HF_HUB:
        sources.append("HuggingFace Hub ✓")
    else:
        sources.append("HuggingFace Hub ✗ (未安装)")
    sources.append("直接 HTTP ✓")
    print(", ".join(sources))
    print()

    # --- 警告信息 ---
    if G_MISSING_PACKAGES:
        print("[警告] 以下包未安装，将在后续步骤中自动安装:")
        for pkg in G_MISSING_PACKAGES:
            print(f"  - {pkg}")
        print()
        print("  如果不需要自动安装，请使用 --no-install 参数跳过")
        print()

    print_separator()
    print()


# ============================================================================
#  安装缺失的包和下载工具
# ============================================================================

def get_pip_command():
    """
    获取当前 Python 环境对应的 pip 命令。

    策略：
      1. 优先使用 uv pip（更快）
      2. 回退到 python -m pip

    返回:
        (command: list, is_uv: bool)
        command: pip 命令列表
        is_uv: 是否使用 uv
    """
    # 检查 uv 是否可用
    uv_path = shutil.which("uv")
    if uv_path:
        # uv pip install --python <path> <packages>
        return ["uv", "pip", "install", "--python", G_PYTHON_EXECUTABLE], True

    # 回退到 python -m pip
    return [G_PYTHON_EXECUTABLE, "-m", "pip", "install"], False


# PyTorch 相关包（需要从专门的 CUDA 源安装）
PYTORCH_CUDA_PACKAGES = {"torch", "torchaudio", "torchvision"}


def get_pip_install_args(package_names, upgrade=False, use_torch_index=False, is_uv=False):
    """
    构建 pip install 命令参数。
    根据 G_IN_CHINA 决定是否使用阿里云镜像源。
    如果 use_torch_index=True，添加 PyTorch CUDA 源。
    如果 is_uv=True，使用 uv 的参数格式。
    """
    args = []
    if upgrade:
        args.append("--upgrade")

    if G_IN_CHINA:
        if is_uv:
            # uv 使用 --index-url 和 --index-extra-url
            args.extend([
                "--index-url",
                "https://mirrors.aliyun.com/pypi/simple/",
            ])
        else:
            args.extend([
                "--index-url",
                "https://mirrors.aliyun.com/pypi/simple/",
                "--trusted-host",
                "mirrors.aliyun.com",
            ])

    # 如果需要 PyTorch CUDA 源
    if use_torch_index and G_RECOMMENDED_TORCH_INDEX:
        if is_uv:
            args.extend([
                "--extra-index-url",
                G_RECOMMENDED_TORCH_INDEX,
                "--index-strategy",
                "unsafe-best-match",
            ])
        else:
            args.extend([
                "--extra-index-url",
                G_RECOMMENDED_TORCH_INDEX,
            ])

    args.extend(package_names)
    return args


def install_packages_with_progress(package_names, upgrade=False, use_torch_index=False):
    """
    安装包并显示实时进度。

    策略：
      - 优先使用 uv pip（更快）
      - 回退到 python -m pip
      - 实时输出 pip 的每一行日志（用户可以看到正在做什么）
      - 每行前面添加时间戳和进度指示
      - 如果超过 60 秒没有新输出，显示警告提示

    参数：
        package_names: 要安装的包名列表
        upgrade: 是否升级已安装的包
        use_torch_index: 是否添加 PyTorch CUDA 源

    返回：
        True 如果全部安装成功，False 如果有失败
    """
    import time as _time

    pip_cmd, is_uv = get_pip_command()
    install_args = get_pip_install_args(package_names, upgrade=upgrade, use_torch_index=use_torch_index, is_uv=is_uv)
    full_cmd = pip_cmd + install_args

    if is_uv:
        print(f"  使用 uv pip 安装 (更快)")
    else:
        print(f"  使用 python -m pip 安装")

    print(f"  执行: {' '.join(full_cmd)}")
    print(f"  开始安装 {len(package_names)} 个包...")
    print(f"  {'─' * 52}")

    # 启动 subprocess 捕获输出
    process = subprocess.Popen(
        full_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    start_time = _time.time()
    last_output_time = start_time
    line_count = 0
    warning_shown = False

    try:
        for line in process.stdout:
            line_count += 1
            line = line.rstrip()
            if not line:
                continue

            # 计算耗时
            elapsed = _time.time() - start_time
            minutes = int(elapsed // 60)
            seconds = int(elapsed % 60)
            time_str = f"{minutes}m{seconds:02d}s"

            # 计算距离上次输出的时间
            idle_time = _time.time() - last_output_time
            last_output_time = _time.time()

            # 如果超过 60 秒没有新输出，显示警告
            if idle_time > 60 and not warning_shown:
                print(f"  ⚠ 注意: 已等待 {idle_time:.0f} 秒无新输出，可能正在下载大文件...")
                warning_shown = True
                # 重置警告标志，下次再等 60 秒
                last_output_time = _time.time()

            # 输出格式: [时间] 行号: 内容
            # 截断过长的行
            if len(line) > 120:
                line = line[:117] + "..."
            print(f"  [{time_str}] #{line_count}: {line}")

        process.wait()

        # 最终结果
        total_time = _time.time() - start_time
        print(f"  {'─' * 52}")
        print(f"  总耗时: {total_time:.1f} 秒")

        if process.returncode == 0:
            print(f"  ✓ 成功安装 {len(package_names)} 个包")
            return True
        else:
            print(f"  ✗ 安装失败 (退出码: {process.returncode})")
            return False

    except KeyboardInterrupt:
        process.terminate()
        print("\n  ⊘ 安装被用户中断")
        return False
    except Exception as e:
        print(f"\n  ✗ 安装异常: {e}")
        return False


def install_missing_packages():
    """
    安装所有缺失的依赖包。
    如果存在虚拟环境，则在虚拟环境中安装。
    根据 G_IN_CHINA 决定是否使用阿里云镜像源。
    PyTorch 相关包使用推荐的 CUDA 源安装。
    """
    if not G_MISSING_PACKAGES:
        print("[依赖包安装]")
        print("  所有依赖包已安装，无需安装")
        print()
        return True

    print("[依赖包安装]")
    print(f"  发现 {len(G_MISSING_PACKAGES)} 个缺失的包:")
    for pkg in G_MISSING_PACKAGES:
        print(f"    - {pkg}")
    print()

    # 确认安装环境
    if G_IN_VIRTUALENV:
        print(f"  安装目标: 虚拟环境 [{G_VENV_NAME}]")
        print(f"  安装路径: {G_VENV_PATH}")
    else:
        print(f"  安装目标: 系统 Python")
        print(f"  安装路径: {G_PYTHON_EXECUTABLE}")
    print()

    # 确认镜像源
    if G_IN_CHINA:
        print(f"  镜像源:    阿里云 (mirrors.aliyun.com)")
    else:
        print(f"  镜像源:    PyPI 官方 (pypi.org)")

    # 分离 PyTorch CUDA 包和普通包
    torch_pkgs = [p for p in G_MISSING_PACKAGES if p in PYTORCH_CUDA_PACKAGES]
    other_pkgs = [p for p in G_MISSING_PACKAGES if p not in PYTORCH_CUDA_PACKAGES]

    if torch_pkgs and G_RECOMMENDED_CUDA:
        print(f"  PyTorch CUDA 源: {G_RECOMMENDED_TORCH_INDEX} ({G_RECOMMENDED_CUDA})")
    print()

    # 询问用户确认（可以跳过）
    if not ask_continue("是否继续安装? (y/N): "):
        print("  ⊘ 用户取消安装")
        print()
        return False

    # 执行安装：先装 PyTorch CUDA 包，再装普通包
    success = True
    failed_pkgs = []

    if torch_pkgs:
        print(f"\n  [1/{(1 if other_pkgs else 0) + 1}] 安装 PyTorch CUDA 包: {' '.join(torch_pkgs)}")
        if not install_packages_with_progress(torch_pkgs, use_torch_index=True):
            success = False
            failed_pkgs.extend(torch_pkgs)

    if other_pkgs:
        step = 1 if not torch_pkgs else 2
        print(f"\n  [{step}/2] 安装其他包: {' '.join(other_pkgs)}")
        if not install_packages_with_progress(other_pkgs, use_torch_index=False):
            success = False
            failed_pkgs.extend(other_pkgs)

    # 无论成功失败，都重新检测包状态（部分包可能已安装）
    detect_packages()
    print()
    print("  安装完成，重新检测包状态:")
    for pkg_name, info in G_PACKAGES.items():
        status = "✓" if info["installed"] else "✗"
        version = str(info["version"]) if info["version"] else "-"
        print(f"    {pkg_name:<20} {status:<8} {version:<20}")
    print()

    if not success:
        print(f"  ⚠ 以下包安装失败: {' '.join(failed_pkgs)}")
        print()

    return success


# ============================================================================
#  下载主模型
# ============================================================================

def fmt_size(nbytes: int) -> str:
    """格式化字节数为可读字符串"""
    for unit in ("B", "KB", "MB", "GB"):
        if nbytes < 1024:
            return f"{nbytes:.1f} {unit}"
        nbytes /= 1024
    return f"{nbytes:.1f} TB"


def check_main_model_exists(model_dir: str) -> bool:
    """
    检查主模型是否已完整下载。
    返回 True 如果所有必需文件和目录都存在。
    """
    # 检查必需文件
    for filename in MAIN_MODEL_REQUIRED_FILES:
        filepath = os.path.join(model_dir, filename)
        if not os.path.isfile(filepath):
            return False

    # 检查必需目录
    for dirname in MAIN_MODEL_REQUIRED_DIRS:
        dirpath = os.path.join(model_dir, dirname)
        if not os.path.isdir(dirpath):
            return False
        # 检查 qwen 目录中是否有 model.safetensors
        if dirname == "qwen0.6bemo4-merge":
            safetensors_path = os.path.join(dirpath, "model.safetensors")
            if not os.path.isfile(safetensors_path):
                return False

    return True


def get_main_model_size(model_dir: str) -> int:
    """计算主模型目录的总大小"""
    total = 0
    for dirpath, dirnames, filenames in os.walk(model_dir):
        # 跳过 hf_cache 子目录
        if "hf_cache" in dirnames:
            dirnames.remove("hf_cache")
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            try:
                total += os.path.getsize(filepath)
            except OSError:
                pass
    return total


def download_main_model(model_dir: str, force: bool = False) -> bool:
    """
    下载 IndexTTS-2 主模型。

    策略：
      - 如果在国内，设置 HF_ENDPOINT=hf-mirror.com
      - 使用 huggingface_hub.snapshot_download 下载
      - 实时显示下载进度

    参数：
        model_dir: 模型存放目录
        force: 强制重新下载

    返回：
        True 如果下载成功
    """
    import time as _time

    # 确保 huggingface_hub 已安装（下载主模型必需）
    if not G_USE_HF_HUB:
        print("[依赖包检查]")
        print("  huggingface_hub 未安装，正在安装...")
        if install_missing_packages():
            print("  ✓ huggingface_hub 安装完成")
            print()
        else:
            print("  ✗ huggingface_hub 安装失败，无法下载主模型")
            print()
            return False

    model_dir = os.path.abspath(model_dir)
    os.makedirs(model_dir, exist_ok=True)

    # 检查是否已下载
    if not force and check_main_model_exists(model_dir):
        size = get_main_model_size(model_dir)
        print("[主模型下载]")
        print(f"  主模型已存在: {model_dir}")
        print(f"  总大小: {fmt_size(size)}")
        print(f"  必需文件检查:")
        for filename in MAIN_MODEL_REQUIRED_FILES:
            filepath = os.path.join(model_dir, filename)
            exists = os.path.isfile(filepath)
            size = os.path.getsize(filepath) if exists else 0
            status = "✓" if exists else "✗"
            print(f"    {status} {filename:<30} {fmt_size(size)}")
        for dirname in MAIN_MODEL_REQUIRED_DIRS:
            dirpath = os.path.join(model_dir, dirname)
            exists = os.path.isdir(dirpath)
            status = "✓" if exists else "✗"
            print(f"    {status} {dirname}/")
        print()
        return True

    print("[主模型下载]")
    print(f"  目标目录: {model_dir}")
    print()

    # 确定下载源
    if G_IN_CHINA:
        endpoint = "https://hf-mirror.com"
        repo_id = MAIN_MODEL_REPOS["hf"]
        print(f"  下载源:    hf-mirror.com (国内镜像)")
    else:
        endpoint = "https://huggingface.co"
        repo_id = MAIN_MODEL_REPOS["hf"]
        print(f"  下载源:    huggingface.co (官方)")
    print(f"  仓库:       {repo_id}")
    print()

    # 询问用户确认
    if not ask_continue("是否继续下载? (y/N): "):
        print("  ⊘ 用户取消下载")
        print()
        return False

    # 设置环境变量（必须在调用 huggingface_hub 之前）
    old_endpoint = os.environ.get("HF_ENDPOINT")
    os.environ["HF_ENDPOINT"] = endpoint

    try:
        from huggingface_hub import snapshot_download, get_hf_file_metadata
        print(f"  开始下载...")
        print(f"  {'─' * 52}")

        start_time = _time.time()

        # 使用 huggingface_hub 下载，带进度回调
        def progress_callback(status):
            """下载进度回调"""
            elapsed = _time.time() - start_time
            minutes = int(elapsed // 60)
            seconds = int(elapsed % 60)
            time_str = f"{minutes}m{seconds:02d}s"

            if hasattr(status, 'downloaded') and hasattr(status, 'total'):
                downloaded = status.downloaded
                total = status.total
                if total > 0:
                    pct = downloaded / total * 100
                    speed = downloaded / elapsed if elapsed > 0 else 0
                    speed_str = fmt_size(int(speed)) + "/s"
                    bar_len = 20
                    filled = int(pct / 100 * bar_len)
                    bar = "█" * filled + "░" * (bar_len - filled)
                    print(f"\r  [{time_str}] [{bar}] {pct:5.1f}% "
                          f"({fmt_size(downloaded)}/{fmt_size(total)}) "
                          f"{speed_str}", end="", flush=True)
                else:
                    print(f"\r  [{time_str}] 已下载 {fmt_size(downloaded)}",
                          end="", flush=True)
            elif hasattr(status, 'filename'):
                filename = os.path.basename(status.filename)
                print(f"\r  [{time_str}] 下载: {filename}", end="", flush=True)

        # 执行下载
        try:
            snapshot_download(
                repo_id=repo_id,
                local_dir=model_dir,
                local_dir_use_symlinks=False,
                progress_callback=progress_callback,
                endpoint=endpoint,  # 显式指定下载源
            )
            print()  # 换行
            print(f"  {'─' * 52}")

            total_time = _time.time() - start_time
            size = get_main_model_size(model_dir)
            print(f"  总耗时: {total_time:.1f} 秒")
            print(f"  总大小: {fmt_size(size)}")
            print(f"  ✓ 主模型下载完成")
            print()

            # 验证下载结果
            print("  验证下载结果:")
            all_ok = True
            for filename in MAIN_MODEL_REQUIRED_FILES:
                filepath = os.path.join(model_dir, filename)
                exists = os.path.isfile(filepath)
                size = os.path.getsize(filepath) if exists else 0
                status = "✓" if exists else "✗"
                print(f"    {status} {filename:<30} {fmt_size(size)}")
                if not exists:
                    all_ok = False
            for dirname in MAIN_MODEL_REQUIRED_DIRS:
                dirpath = os.path.join(model_dir, dirname)
                exists = os.path.isdir(dirpath)
                status = "✓" if exists else "✗"
                print(f"    {status} {dirname}/")
                if not exists:
                    all_ok = False

            print()
            return all_ok

        except Exception as e:
            print()
            print(f"  ✗ 下载失败: {e}")
            print()

            # 尝试回退到 ModelScope
            if G_IN_CHINA:
                print("  尝试回退到 ModelScope...")
                print()
                return download_main_model_from_modelscope(model_dir)
            else:
                return False

    finally:
        # 恢复环境变量
        if old_endpoint is not None:
            os.environ["HF_ENDPOINT"] = old_endpoint
        elif "HF_ENDPOINT" in os.environ:
            del os.environ["HF_ENDPOINT"]


def download_main_model_from_modelscope(model_dir: str) -> bool:
    """
    从 ModelScope 下载主模型（作为 hf-mirror 失败后的回退）。
    """
    import time as _time

    model_dir = os.path.abspath(model_dir)
    os.makedirs(model_dir, exist_ok=True)

    if not G_USE_MODELSCOPE:
        print("  ModelScope 未安装，无法回退")
        return False

    print("[主模型下载 - ModelScope 回退]")
    print(f"  目标目录: {model_dir}")
    print(f"  下载源:    ModelScope")
    print(f"  仓库:       {MAIN_MODEL_REPOS['ms']}")
    print()

    try:
        from modelscope.hub.snapshot_download import snapshot_download as ms_snapshot
        import tempfile

        start_time = _time.time()

        print(f"  开始下载...")
        print(f"  {'─' * 52}")

        # ModelScope 下载到临时目录，然后移动到目标目录
        with tempfile.TemporaryDirectory() as tmpdir:
            ms_snapshot(
                model_id=MAIN_MODEL_REPOS["ms"],
                cache_dir=tmpdir,
            )

            # 找到下载的文件
            downloaded = None
            for root, dirs, files in os.walk(tmpdir):
                if files and root != tmpdir:
                    downloaded = root
                    break

            if downloaded is None:
                print("  ✗ 未在临时目录中找到下载的文件")
                return False

            # 移动到目标目录
            import shutil
            for item in os.listdir(downloaded):
                src = os.path.join(downloaded, item)
                dst = os.path.join(model_dir, item)
                if not os.path.exists(dst):
                    if os.path.isdir(src):
                        shutil.copytree(src, dst)
                    else:
                        shutil.copy2(src, dst)

        total_time = _time.time() - start_time
        size = get_main_model_size(model_dir)
        print(f"  {'─' * 52}")
        print(f"  总耗时: {total_time:.1f} 秒")
        print(f"  总大小: {fmt_size(size)}")
        print(f"  ✓ 主模型下载完成 (ModelScope)")
        print()

        # 验证
        print("  验证下载结果:")
        all_ok = True
        for filename in MAIN_MODEL_REQUIRED_FILES:
            filepath = os.path.join(model_dir, filename)
            exists = os.path.isfile(filepath)
            size = os.path.getsize(filepath) if exists else 0
            status = "✓" if exists else "✗"
            print(f"    {status} {filename:<30} {fmt_size(size)}")
            if not exists:
                all_ok = False
        for dirname in MAIN_MODEL_REQUIRED_DIRS:
            dirpath = os.path.join(model_dir, dirname)
            exists = os.path.isdir(dirpath)
            status = "✓" if exists else "✗"
            print(f"    {status} {dirname}/")
            if not exists:
                all_ok = False

        print()
        return all_ok

    except Exception as e:
        print(f"  ✗ ModelScope 下载失败: {e}")
        print()
        return False


# ============================================================================
#  下载辅助模型
# ============================================================================

# HF → ModelScope 仓库映射（只有这些仓库在 ModelScope 上有镜像）
HF_TO_MODELSCOPE_MAP = {
    "facebook/w2v-bert-2.0": "AI-ModelScope/w2v-bert-2.0",
    "funasr/campplus": "iic/speech_campplus_sv_zh-cn_16k-common",
}


def _get_download_endpoint():
    """根据 G_IN_CHINA 返回下载 endpoint"""
    if G_IN_CHINA:
        return "https://hf-mirror.com"
    return "https://huggingface.co"


def _get_source_name():
    """返回当前下载源名称"""
    if G_IN_CHINA:
        return "hf-mirror.com"
    return "huggingface.co"


def download_single_file_from_hf(repo_id: str, file_path: str, local_path: str,
                                  ms_model_id: str = None, ms_file_path: str = None) -> bool:
    """
    从 HuggingFace 下载单个文件，支持多源回退。

    策略（国内）:
      - 有 ModelScope 映射: ModelScope → hf-mirror 直接 HTTP
      - 无 ModelScope 映射: hf-mirror 直接 HTTP
    策略（海外）:
      - huggingface.co 直接 HTTP → hf-mirror → ModelScope

    参数：
        repo_id: HF 仓库 ID，如 "facebook/w2v-bert-2.0"
        file_path: 仓库内文件路径，如 "semantic_codec/model.safetensors"
        local_path: 本地保存路径
        ms_model_id: ModelScope 仓库 ID（如有映射）
        ms_file_path: ModelScope 上的文件路径（默认与 file_path 相同）

    返回：
        True 如果下载成功
    """
    import time as _time
    import ssl
    from urllib.request import urlopen, Request
    from urllib.error import HTTPError, URLError

    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    # 构建下载源列表
    hf_url = f"https://huggingface.co/{repo_id}/resolve/main/{file_path}"
    hf_mirror_url = f"https://hf-mirror.com/{repo_id}/resolve/main/{file_path}"

    if G_IN_CHINA:
        if ms_model_id:
            sources = [
                ("ModelScope", ms_model_id, ms_file_path or file_path),
                ("hf-mirror HTTP", hf_mirror_url, None),
            ]
        else:
            sources = [
                ("hf-mirror HTTP", hf_mirror_url, None),
            ]
    else:
        sources = [
            ("huggingface HTTP", hf_url, None),
            ("hf-mirror HTTP", hf_mirror_url, None),
        ]
        if ms_model_id:
            sources.append(("ModelScope", ms_model_id, ms_file_path or file_path))

    last_error = None
    for source_name, source_value, _ in sources:
        try:
            if source_name == "ModelScope":
                return _download_from_modelscope_file(source_value, ms_file_path or file_path, local_path)
            else:
                return _download_file_http(source_value, local_path)
        except Exception as e:
            last_error = e
            print(f"     {source_name} 失败: {e}，尝试下一个源...")

    print(f"     ✗ 所有源均失败: {last_error}")
    return False


def _download_from_modelscope_file(model_id: str, file_path: str, local_path: str) -> bool:
    """从 ModelScope 下载单个文件"""
    try:
        from modelscope.hub.file_download import model_file_download
        print(f"     从 ModelScope 下载: {model_id}/{file_path}")
        tmp = model_file_download(model_id=model_id, file_path=file_path)
        shutil.copy2(tmp, local_path)
        return True
    except Exception as e:
        raise RuntimeError(f"ModelScope 下载失败: {e}")


def _download_file_http(url: str, local_path: str, timeout: int = 300,
                         max_retries: int = 3, chunk_size: int = 65536) -> bool:
    """
    通过 HTTP 下载单个文件，支持断点续传、自动重试和进度显示。
    """
    import time as _time
    import ssl
    from urllib.request import urlopen, Request
    from urllib.error import HTTPError

    _ssl_ctx = ssl.create_default_context()
    _ssl_ctx.check_hostname = False
    _ssl_ctx.verify_mode = ssl.CERT_NONE

    tmp_path = str(local_path) + ".tmp"
    last_error = None

    for attempt in range(1, max_retries + 1):
        # 检查是否有未完成的下载（断点续传）
        resume_pos = 0
        if os.path.exists(tmp_path):
            resume_pos = os.path.getsize(tmp_path)

        if attempt > 1:
            wait = min(2 ** (attempt - 1) * 2, 30)
            print(f"     重试 {attempt}/{max_retries} (等待 {wait}s)...")
            _time.sleep(wait)

        headers = {"User-Agent": "IndexTTS-Downloader/1.0"}
        if resume_pos > 0:
            headers["Range"] = f"bytes={resume_pos}-"

        try:
            req = Request(url, headers=headers)
            with urlopen(req, timeout=timeout, context=_ssl_ctx) as resp:
                status = resp.status

                if status not in (200, 206):
                    raise RuntimeError(f"HTTP {status}")

                # 获取总大小
                content_range = resp.headers.get("Content-Range", "")
                if "/" in content_range:
                    total = int(content_range.split("/")[1])
                else:
                    total = int(resp.headers.get("Content-Length", 0))

                downloaded = resume_pos
                start_time = _time.time()

                # 以追加模式写入（支持断点续传）
                mode = "ab" if resume_pos > 0 and status == 206 else "wb"
                with open(tmp_path, mode) as f:
                    while True:
                        chunk = resp.read(chunk_size)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)

                        # 显示进度
                        elapsed = max(_time.time() - start_time, 0.1)
                        speed = (downloaded - resume_pos) / elapsed / 1024 / 1024
                        if total > 0:
                            pct = downloaded / total * 100
                            print(f"\r     {pct:5.1f}% ({fmt_size(downloaded)}/{fmt_size(total)}) "
                                  f"{speed:.1f} MB/s", end="", flush=True)
                        else:
                            print(f"\r     {fmt_size(downloaded)} {speed:.1f} MB/s",
                                  end="", flush=True)

                print()  # 换行
                os.replace(tmp_path, local_path)
                last_error = None
                break  # 成功

        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError,
                ConnectionError, OSError) as e:
            last_error = e
            print(f"\n     网络错误: {e}")
            continue
        except HTTPError as e:
            if e.code == 503:
                last_error = e
                continue
            raise
        except Exception as e:
            last_error = e
            if attempt < max_retries:
                continue
            raise

    if last_error is not None:
        raise RuntimeError(f"失败 (已重试 {max_retries} 次): {last_error}")

    # 清理残留的临时文件
    if os.path.exists(tmp_path):
        os.remove(tmp_path)

    return True


def download_w2v_bert(cache_dir: str, force: bool = False) -> bool:
    """
    下载 w2v-bert-2.0 完整仓库（~2GB）。

    策略:
      - 有 ModelScope 映射，优先 ModelScope
      - 回退到 hf-mirror / huggingface
    """
    import time as _time
    local_dir = os.path.join(cache_dir, "w2v-bert-2.0")

    # 检查是否已下载
    if not force and os.path.isdir(local_dir):
        file_count = sum(1 for _ in Path(local_dir).rglob("*") if _.is_file())
        if file_count >= 5:
            size = _get_dir_size(local_dir)
            print(f"  ✓ w2v-bert-2.0 已存在 ({file_count} 个文件, {fmt_size(size)})，跳过")
            return True

    print(f"  ⬇ 下载 w2v-bert-2.0 (~2 GB) ...")
    os.makedirs(local_dir, exist_ok=True)

    # 策略 1: ModelScope（有映射）
    if G_USE_MODELSCOPE:
        try:
            print(f"     尝试 ModelScope: AI-ModelScope/w2v-bert-2.0")
            return _download_repo_from_modelscope("AI-ModelScope/w2v-bert-2.0", local_dir)
        except Exception as e:
            print(f"     ModelScope 失败: {e}")

    # 策略 2: huggingface_hub
    try:
        from huggingface_hub import snapshot_download as hf_snapshot
        endpoint = _get_download_endpoint()
        print(f"     尝试 {_get_source_name()}: facebook/w2v-bert-2.0")

        start_time = _time.time()

        def progress_callback(status):
            elapsed = _time.time() - start_time
            minutes = int(elapsed // 60)
            seconds = int(elapsed % 60)
            time_str = f"{minutes}m{seconds:02d}s"
            if hasattr(status, 'downloaded') and hasattr(status, 'total'):
                downloaded = status.downloaded
                total = status.total
                if total > 0:
                    pct = downloaded / total * 100
                    speed = downloaded / elapsed if elapsed > 0 else 0
                    speed_str = fmt_size(int(speed)) + "/s"
                    bar_len = 20
                    filled = int(pct / 100 * bar_len)
                    bar = "█" * filled + "░" * (bar_len - filled)
                    print(f"\r     [{time_str}] [{bar}] {pct:5.1f}% "
                          f"({fmt_size(downloaded)}/{fmt_size(total)}) "
                          f"{speed_str}", end="", flush=True)

        hf_snapshot(
            repo_id="facebook/w2v-bert-2.0",
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            progress_callback=progress_callback,
            endpoint=endpoint,
        )
        print()
        file_count = sum(1 for _ in Path(local_dir).rglob("*") if _.is_file())
        size = _get_dir_size(local_dir)
        print(f"     ✓ 下载完成: {file_count} 个文件, {fmt_size(size)}")
        return True

    except Exception as e:
        print(f"     ✗ huggingface_hub 下载失败: {e}")
        return False


def download_maskgct(cache_dir: str, force: bool = False) -> bool:
    """
    下载 MaskGCT semantic codec 单文件（~169MB）。

    无 ModelScope 映射，直接 HTTP 下载。
    """
    local_path = os.path.join(cache_dir, "semantic_codec_model.safetensors")

    if not force and os.path.isfile(local_path):
        size = os.path.getsize(local_path)
        if size >= 100 * 1024 * 1024:
            print(f"  ✓ MaskGCT semantic codec 已存在 ({fmt_size(size)})，跳过")
            return True

    print(f"  ⬇ 下载 MaskGCT semantic codec (~169 MB) ...")
    repo_id = "amphion/MaskGCT"
    file_path = "semantic_codec/model.safetensors"

    success = download_single_file_from_hf(repo_id, file_path, local_path)
    if success:
        size = os.path.getsize(local_path)
        print(f"  ✓ MaskGCT semantic codec 下载完成 ({fmt_size(size)})")
    else:
        print(f"  ✗ MaskGCT semantic codec 下载失败")
    return success


def download_campplus(cache_dir: str, force: bool = False) -> bool:
    """
    下载 CAMPPlus speaker embedding 单文件（~200MB）。

    有 ModelScope 映射。
    """
    local_path = os.path.join(cache_dir, "campplus_cn_common.bin")

    if not force and os.path.isfile(local_path):
        size = os.path.getsize(local_path)
        if size >= 10 * 1024 * 1024:
            print(f"  ✓ CAMPPlus speaker embedding 已存在 ({fmt_size(size)})，跳过")
            return True

    print(f"  ⬇ 下载 CAMPPlus speaker embedding (~200 MB) ...")
    repo_id = "funasr/campplus"
    file_path = "campplus_cn_common.bin"
    ms_model_id = "iic/speech_campplus_sv_zh-cn_16k-common"

    success = download_single_file_from_hf(repo_id, file_path, local_path,
                                            ms_model_id=ms_model_id)
    if success:
        size = os.path.getsize(local_path)
        print(f"  ✓ CAMPPlus speaker embedding 下载完成 ({fmt_size(size)})")
    else:
        print(f"  ✗ CAMPPlus speaker embedding 下载失败")
    return success


def download_bigvgan(cache_dir: str, force: bool = False) -> bool:
    """
    下载 BigVGAN vocoder 多文件（~150MB）。

    无 ModelScope 映射，直接 HTTP 下载。
    """
    bigvgan_dir = os.path.join(cache_dir, "bigvgan")
    os.makedirs(bigvgan_dir, exist_ok=True)

    files = [
        ("config.json", "config.json", 100),
        ("bigvgan_generator.pt", "bigvgan_generator.pt", 100 * 1024 * 1024),
    ]

    # 检查是否已下载
    if not force:
        all_exist = True
        for hf_name, local_name, min_size in files:
            fp = os.path.join(bigvgan_dir, local_name)
            if not os.path.isfile(fp) or os.path.getsize(fp) < min_size:
                all_exist = False
                break
        if all_exist:
            size = _get_dir_size(bigvgan_dir)
            print(f"  ✓ BigVGAN vocoder 已存在 ({fmt_size(size)})，跳过")
            return True

    print(f"  ⬇ 下载 BigVGAN vocoder (~150 MB) ...")
    repo_id = "nvidia/bigvgan_v2_22khz_80band_256x"
    all_ok = True

    for hf_name, local_name, min_size in files:
        local_path = os.path.join(bigvgan_dir, local_name)
        if not force and os.path.isfile(local_path) and os.path.getsize(local_path) >= min_size:
            print(f"     ✓ {local_name} 已存在，跳过")
            continue

        print(f"     ⬇ {local_name} ...")
        success = download_single_file_from_hf(repo_id, hf_name, local_path)
        if success:
            size = os.path.getsize(local_path)
            print(f"     ✓ {local_name} ({fmt_size(size)})")
        else:
            print(f"     ✗ {local_name} 下载失败")
            all_ok = False

    if all_ok:
        size = _get_dir_size(bigvgan_dir)
        print(f"  ✓ BigVGAN vocoder 下载完成 ({fmt_size(size)})")
    else:
        print(f"  ✗ BigVGAN vocoder 部分下载失败")
    return all_ok


def _download_repo_from_modelscope(model_id: str, local_dir: str) -> bool:
    """从 ModelScope 下载完整仓库"""
    import tempfile
    try:
        from modelscope.hub.snapshot_download import snapshot_download as ms_snapshot

        with tempfile.TemporaryDirectory() as tmpdir:
            ms_snapshot(model_id=model_id, cache_dir=tmpdir)

            # 找到下载的文件
            downloaded = None
            for root, dirs, files in os.walk(tmpdir):
                if files and root != tmpdir:
                    downloaded = root
                    break

            if downloaded is None:
                print(f"     错误: 未在临时目录中找到下载的文件")
                return False

            # 移动到目标目录
            os.makedirs(local_dir, exist_ok=True)
            for item in os.listdir(downloaded):
                src = os.path.join(downloaded, item)
                dst = os.path.join(local_dir, item)
                if not os.path.exists(dst):
                    if os.path.isdir(src):
                        shutil.copytree(src, dst)
                    else:
                        shutil.copy2(src, dst)

        file_count = sum(1 for _ in Path(local_dir).rglob("*") if _.is_file())
        size = _get_dir_size(local_dir)
        print(f"     ✓ ModelScope 下载完成: {file_count} 个文件, {fmt_size(size)}")
        return True
    except Exception as e:
        raise RuntimeError(f"ModelScope 下载失败: {e}")


def _get_dir_size(dir_path: str) -> int:
    """计算目录总大小"""
    total = 0
    for dirpath, dirnames, filenames in os.walk(dir_path):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            try:
                total += os.path.getsize(filepath)
            except OSError:
                pass
    return total


def download_aux_models(cache_dir: str, force: bool = False) -> bool:
    """
    下载所有辅助模型到 hf_cache/ 目录。

    包含 4 个模型：
      1. w2v-bert-2.0 (~2 GB) - 音频特征提取
      2. MaskGCT semantic codec (~169 MB) - 语义编解码器
      3. CAMPPlus speaker embedding (~200 MB) - 说话人声纹
      4. BigVGAN vocoder (~150 MB) - 声码器

    参数：
        cache_dir: hf_cache 目录路径
        force: 强制重新下载

    返回：
        True 如果全部下载成功
    """
    # 确保 modelscope 已安装（辅助模型下载必需）
    if not G_USE_MODELSCOPE:
        print("[依赖包检查]")
        print("  modelscope 未安装，正在安装...")
        if install_missing_packages():
            print("  ✓ modelscope 安装完成")
            print()
        else:
            print("  ✗ modelscope 安装失败，部分模型可能无法下载")
            print()

    print()
    print("[辅助模型下载]")
    print(f"  目标目录: {cache_dir}")
    print(f"  下载源:    {_get_source_name()}")
    if G_USE_MODELSCOPE:
        print(f"  备用源:    ModelScope ✓")
    print()

    os.makedirs(cache_dir, exist_ok=True)

    # 询问用户确认
    if not ask_continue("是否继续下载? (y/N): "):
        print("  ⊘ 用户取消下载")
        print()
        return False

    results = []

    # 1. w2v-bert-2.0 (~2 GB)
    print(f"  [1/4] w2v-bert-2.0 (音频特征提取模型)")
    results.append(("w2v-bert-2.0", download_w2v_bert(cache_dir, force)))

    # 2. MaskGCT semantic codec (~169 MB)
    print(f"\n  [2/4] MaskGCT semantic codec (语义编解码器)")
    results.append(("MaskGCT", download_maskgct(cache_dir, force)))

    # 3. CAMPPlus speaker embedding (~200 MB)
    print(f"\n  [3/4] CAMPPlus speaker embedding (说话人声纹嵌入)")
    results.append(("CAMPPlus", download_campplus(cache_dir, force)))

    # 4. BigVGAN vocoder (~150 MB)
    print(f"\n  [4/4] BigVGAN vocoder (声码器)")
    results.append(("BigVGAN", download_bigvgan(cache_dir, force)))

    # 汇总
    print()
    print("  辅助模型下载汇总:")
    all_ok = True
    for name, ok in results:
        status = "✓" if ok else "✗"
        print(f"    {status} {name}")
        if not ok:
            all_ok = False

    # 计算总大小
    size = _get_dir_size(cache_dir)
    print(f"  hf_cache/ 总大小: {fmt_size(size)}")
    print()

    return all_ok


# ============================================================================
#  下载示例音频
# ============================================================================

# 示例音频远程下载源
EXAMPLE_AUDIO_MS_URL = "https://modelscope.cn/studio/IndexTeam/IndexTTS-2-Demo/resolve/master/examples"
EXAMPLE_AUDIO_HF_URL = "https://huggingface.co/spaces/IndexTeam/IndexTTS-2-Demo/resolve/main/examples"


def download_example_audio(examples_dir: str, force: bool = False) -> bool:
    """
    下载 WebUI 演示用示例音频文件（~11 MB）。

    包含 13 个文件：
      - voice_01~09.wav (9 个演示音色)
      - voice_11.wav, voice_12.wav (2 个演示音色)
      - emo_sad.wav, emo_hate.wav (2 个情感样本)

    策略（国内）:
      - ModelScope Studio → HuggingFace Spaces
    策略（海外）:
      - HuggingFace Spaces → ModelScope Studio

    参数：
        examples_dir: 示例音频存放目录
        force: 强制重新下载

    返回：
        True 如果全部下载成功
    """
    print()
    print("[示例音频下载]")
    print(f"  目标目录: {examples_dir}")
    print(f"  文件数量: {len(EXAMPLE_AUDIO_FILES)} 个")
    print()

    os.makedirs(examples_dir, exist_ok=True)

    # 检查是否已下载
    missing = []
    existing_count = 0
    existing_size = 0
    for filename in EXAMPLE_AUDIO_FILES:
        filepath = os.path.join(examples_dir, filename)
        if os.path.isfile(filepath) and not force:
            existing_count += 1
            existing_size += os.path.getsize(filepath)
        else:
            missing.append(filename)

    if not missing:
        print(f"  ✓ 全部 {existing_count} 个示例音频已存在 ({fmt_size(existing_size)})，跳过")
        print()
        return True

    print(f"  已存在: {existing_count}/{len(EXAMPLE_AUDIO_FILES)}")
    print(f"  需下载: {len(missing)}/{len(EXAMPLE_AUDIO_FILES)}")
    print()

    # 询问用户确认
    if not ask_continue("是否继续下载? (y/N): "):
        print("  ⊘ 用户取消下载")
        print()
        return False

    # 构建下载源列表（根据网络环境）
    if G_IN_CHINA:
        urls_to_try = [EXAMPLE_AUDIO_MS_URL, EXAMPLE_AUDIO_HF_URL]
    else:
        urls_to_try = [EXAMPLE_AUDIO_HF_URL, EXAMPLE_AUDIO_MS_URL]

    results = []
    total_downloaded = 0

    for i, filename in enumerate(missing, 1):
        local_path = os.path.join(examples_dir, filename)
        downloaded = False

        for base_url in urls_to_try:
            url = f"{base_url}/{filename}"
            source_name = "ModelScope" if "modelscope" in base_url else "HuggingFace"

            try:
                print(f"     [{i}/{len(missing)}] ⬇ {filename} (from {source_name}) ...")
                _download_file_http(url, local_path, timeout=120, max_retries=2)
                size = os.path.getsize(local_path)
                total_downloaded += size
                print(f"     ✓ {filename} ({fmt_size(size)})")
                downloaded = True
                break
            except Exception as e:
                print(f"     {source_name} 失败: {e}，尝试下一个源...")
                # 清理不完整的临时文件
                tmp_path = str(local_path) + ".tmp"
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
                continue

        if not downloaded:
            print(f"     ✗ {filename} 所有源均失败")
            results.append((filename, False))
        else:
            results.append((filename, True))

    # 汇总
    print()
    print("  示例音频下载汇总:")
    all_ok = True
    for filename, ok in results:
        status = "✓" if ok else "✗"
        filepath = os.path.join(examples_dir, filename)
        size = os.path.getsize(filepath) if os.path.isfile(filepath) else 0
        print(f"    {status} {filename:<20} {fmt_size(size)}")
        if not ok:
            all_ok = False

    # 计算总大小
    total_size = _get_dir_size(examples_dir)
    print(f"  examples/ 总大小: {fmt_size(total_size)}")
    print()

    return all_ok


# ============================================================================
#  初始化入口
# ============================================================================

def init_env(model_dir: str = None):
    """执行完整的环境检测"""
    detect_system_info()
    detect_virtualenv()
    detect_packages()
    if model_dir:
        detect_hardware(model_dir)
    print_env_info()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="IndexTTS-2 模型预下载脚本",
        epilog="示例:\n"
               "  python download_all.py              # 默认中国大陆网络环境\n"
               "  python download_all.py false         # 非中国大陆网络环境\n"
               "  python download_all.py --no-install  # 不自动安装缺失的包\n"
               "  python download_all.py --skip-model  # 不下载模型\n"
               "  python download_all.py --skip-examples  # 不下载示例音频\n"
               "  python download_all.py --model-dir /path/to/models\n",
    )
    parser.add_argument(
        "in_china",
        nargs="?",
        default=None,
        help="是否在中国大陆网络环境 (true/false, 默认 true)",
    )
    parser.add_argument(
        "--no-install",
        action="store_true",
        help="不自动安装缺失的依赖包",
    )
    parser.add_argument(
        "--skip-model",
        action="store_true",
        help="不下载模型（只检测环境和安装包）",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="./checkpoints",
        help="模型存放目录 (默认: ./checkpoints)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="强制重新下载已存在的模型",
    )
    parser.add_argument(
        "--skip-examples",
        action="store_true",
        help="不下载示例音频文件",
    )
    parser.add_argument(
        "--examples-dir",
        type=str,
        default="./examples",
        help="示例音频存放目录 (默认: ./examples)",
    )
    parser.add_argument(
        "--auto-download",
        action="store_true",
        help="自动确认所有下载操作，无需用户输入",
    )
    args = parser.parse_args()

    # 根据参数设置 G_IN_CHINA
    if args.in_china is not None:
        val = args.in_china.lower()
        if val in ("false", "0", "no"):
            G_IN_CHINA = False
        elif val in ("true", "1", "yes"):
            G_IN_CHINA = True

    # 设置自动下载模式
    global G_AUTO_DOWNLOAD
    G_AUTO_DOWNLOAD = args.auto_download

    # 设置模型目录
    global G_MODEL_DIR, G_CACHE_DIR
    G_MODEL_DIR = os.path.abspath(args.model_dir)
    G_CACHE_DIR = os.path.join(G_MODEL_DIR, "hf_cache")
    G_EXAMPLES_DIR = os.path.abspath(args.examples_dir)

    # Python 版本兼容性检查（不兼容则退出）
    py_ok, py_msg, py_level = check_python_compatibility()
    if py_level == "error":
        version = (sys.version_info.major, sys.version_info.minor)
        print("=" * 60)
        print("  ✗ Python 版本不兼容")
        print("=" * 60)
        print()
        print(f"  项目要求: Python {MIN_PYTHON_VERSION[0]}.{MIN_PYTHON_VERSION[1]} ~ {MAX_PYTHON_VERSION[0]}.{MAX_PYTHON_VERSION[1]}")
        print(f"  当前版本: Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
        print()
        if version < MIN_PYTHON_VERSION:
            print(f"  Python 版本过低，建议升级到 3.10 ~ 3.12")
        else:
            print(f"  Python 版本过高，建议降级到 3.10 ~ 3.12")
        print(f"  使用 uv/conda 创建新环境:")
        print(f"    uv venv --python 3.12")
        print(f"    conda create -n indextts python=3.12")
        print()
        print("=" * 60)
        sys.exit(1)

    # 环境检测
    init_env(G_MODEL_DIR)

    # 安装缺失的包（除非用户指定 --no-install）
    if not args.no_install:
        install_missing_packages()

    # 下载主模型（除非用户指定 --skip-model）
    if not args.skip_model:
        download_main_model(G_MODEL_DIR, force=args.force)

    # 下载辅助模型（除非用户指定 --skip-model）
    if not args.skip_model:
        download_aux_models(G_CACHE_DIR, force=args.force)

    # 下载示例音频（除非用户指定 --skip-examples）
    if not args.skip_examples:
        download_example_audio(G_EXAMPLES_DIR, force=args.force)


if __name__ == "__main__":
    main()
