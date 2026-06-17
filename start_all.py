#!/usr/bin/env python3
"""
IndexTTS-2 一键部署脚本

功能：
  1. 自动检测 Python 版本兼容性
  2. 自动创建/激活虚拟环境 (.venv)
  3. 调用 download_all.py 完成全部部署

用法：
  python start_all.py              # 默认中国大陆网络环境
  python start_all.py false        # 非中国大陆网络环境
  python start_all.py --skip-model # 不下载模型（只创建环境和安装包）
  python start_all.py --help       # 查看帮助

作者: IndexTTS Team
日期: 2026-06-17
"""

import os
import sys
import subprocess
import shutil
import argparse
from pathlib import Path


# ==================== 配置 ====================

# Python 版本要求
MIN_PYTHON_VERSION = (3, 10)   # 项目最低要求
MAX_PYTHON_VERSION = (3, 13)   # torch 预编译 wheel 最高支持到 3.13
RECOMMENDED_PYTHON = "3.12"    # 推荐版本

# 虚拟环境名称
VENV_NAME = ".venv"

# 项目根目录（脚本所在目录）
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


# ==================== 工具函数 ====================

def print_banner():
    """打印启动横幅"""
    print("=" * 60)
    print("  IndexTTS-2 一键部署脚本")
    print("=" * 60)
    print()


def check_python_version():
    """
    检查系统 Python 版本是否满足要求。

    返回:
        True 如果版本兼容，False 如果不兼容
    """
    version = (sys.version_info.major, sys.version_info.minor)
    version_str = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

    if version < MIN_PYTHON_VERSION:
        print(f"✗ Python 版本过低！")
        print(f"  当前版本: {version_str}")
        print(f"  需要版本: >= {MIN_PYTHON_VERSION[0]}.{MIN_PYTHON_VERSION[1]}")
        print()
        print(f"  建议升级到 Python {RECOMMENDED_PYTHON}:")
        print(f"    uv venv --python {RECOMMENDED_PYTHON}")
        print(f"    conda create -n indextts python={RECOMMENDED_PYTHON}")
        print()
        return False

    if version > MAX_PYTHON_VERSION:
        print(f"✗ Python 版本过高！")
        print(f"  当前版本: {version_str}")
        print(f"  需要版本: <= {MAX_PYTHON_VERSION[0]}.{MAX_PYTHON_VERSION[1]}")
        print()
        print(f"  建议降级到 Python {RECOMMENDED_PYTHON}:")
        print(f"    uv venv --python {RECOMMENDED_PYTHON}")
        print(f"    conda create -n indextts python={RECOMMENDED_PYTHON}")
        print()
        return False

    return True


def is_in_virtualenv():
    """检查当前是否在虚拟环境中（用于警告用户）"""
    return (
        hasattr(sys, "real_prefix") or
        (hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix)
    )


def is_in_project_venv():
    """
    检查当前 Python 是否位于项目根目录的 .venv 中。

    使用 sys.prefix 判断，因为 venv 中的 python 可能是系统 Python 的软链接，
    os.path.realpath 会解析到系统路径导致误判。

    只认可项目自己的虚拟环境，其他项目的虚拟环境一律视为无效。
    """
    venv_path = os.path.realpath(os.path.join(PROJECT_ROOT, VENV_NAME))

    # sys.prefix 在虚拟环境中指向虚拟环境目录
    current_prefix = os.path.realpath(sys.prefix)

    return current_prefix == venv_path


def get_venv_python_path():
    """获取虚拟环境中 Python 解释器的路径"""
    venv_path = os.path.join(PROJECT_ROOT, VENV_NAME)

    if os.name == "nt":  # Windows
        return os.path.join(venv_path, "Scripts", "python.exe")
    else:  # Linux/Mac
        return os.path.join(venv_path, "bin", "python")


def create_virtualenv():
    """
    创建虚拟环境。

    策略：
      1. 优先使用 uv（更快）
      2. 回退到 python -m venv

    返回:
        True 如果创建成功，False 如果失败
    """
    venv_path = os.path.join(PROJECT_ROOT, VENV_NAME)

    # 如果已经存在，不需要创建
    if os.path.isdir(venv_path):
        return True

    print(f"  正在创建虚拟环境: {venv_path}")
    print()

    # 策略 1: 尝试使用 uv
    uv_path = shutil.which("uv")
    if uv_path:
        print(f"  检测到 uv，使用 uv 创建虚拟环境...")
        try:
            result = subprocess.run(
                ["uv", "venv", "--python", RECOMMENDED_PYTHON],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                timeout=180
            )
            if result.returncode == 0:
                print(f"  ✓ 虚拟环境创建完成 (uv)")
                return True
            else:
                print(f"  uv 创建失败: {result.stderr.strip()}")
                print(f"  回退到 python -m venv...")
        except subprocess.TimeoutExpired:
            print(f"  uv 创建超时")
            print(f"  回退到 python -m venv...")
        except Exception as e:
            print(f"  uv 创建异常: {e}")
            print(f"  回退到 python -m venv...")
    else:
        print(f"  未检测到 uv，使用 python -m venv 创建虚拟环境...")

    # 策略 2: 使用标准 venv 模块
    try:
        import venv
        print(f"  使用 venv 模块创建虚拟环境...")
        builder = venv.EnvBuilder(with_pip=True, symlinks=False)
        builder.create(venv_path)
        print(f"  ✓ 虚拟环境创建完成 (venv)")
        return True
    except Exception as e:
        print(f"  ✗ 虚拟环境创建失败: {e}")
        print()
        print(f"  请手动创建虚拟环境:")
        print(f"    cd {PROJECT_ROOT}")
        print(f"    python -m venv {VENV_NAME}")
        print(f"    source {VENV_NAME}/bin/activate  # Linux/Mac")
        print(f"    {VENV_NAME}\\Scripts\\activate   # Windows")
        print()
        return False


def activate_and_restart(venv_python: str):
    """
    使用虚拟环境中的 Python 重新执行自己。

    保留所有命令行参数。
    """
    print(f"  切换到虚拟环境 Python: {venv_python}")
    print()

    # 构建新的命令行（sys.argv[0] 是脚本路径，必须保留）
    new_argv = [venv_python] + sys.argv

    print(f"  重新执行: {' '.join(new_argv)}")
    print()
    print("=" * 60)
    print()

    # 使用 os.execv 替换当前进程
    os.execv(venv_python, new_argv)


def run_download_script(args):
    """
    调用 download_all.py 完成部署。

    参数:
        args: 命令行参数列表（传递给 download_all.py）
    """
    download_script = os.path.join(PROJECT_ROOT, "download_all.py")

    if not os.path.isfile(download_script):
        print(f"✗ 找不到 download_all.py: {download_script}")
        print(f"  请确保在 IndexTTS-2 项目根目录下运行此脚本")
        sys.exit(1)

    # 构建命令行
    cmd = [sys.executable, download_script] + args

    print(f"  执行: {' '.join(cmd)}")
    print()

    # 执行 download_all.py
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)

    # 返回退出码
    sys.exit(result.returncode)


# ==================== 主函数 ====================

def print_help():
    """打印帮助信息"""
    print("用法: python start_all.py [选项]")
    print()
    print("选项:")
    print("  --skip-model      不下载模型（只创建环境和安装包）")
    print("  --skip-examples   不下载示例音频")
    print("  --no-install      不自动安装缺失的包")
    print("  --force           强制重新下载已存在的模型")
    print("  --model-dir DIR   模型存放目录 (默认: ./checkpoints)")
    print("  --examples-dir DIR 示例音频目录 (默认: ./examples)")
    print("  false             非中国大陆网络环境")
    print("  -h, --help        显示此帮助信息")
    print()
    print("示例:")
    print("  python start_all.py              # 一键部署（中国大陆）")
    print("  python start_all.py false        # 一键部署（海外）")
    print("  python start_all.py --skip-model # 只创建环境和安装包")
    print()


def main():
    """主函数"""
    # 检查帮助参数
    if "--help" in sys.argv or "-h" in sys.argv:
        print_banner()
        print_help()
        sys.exit(0)

    print_banner()

    # 步骤 1: 检查 Python 版本
    print("[步骤 1/4] 检查 Python 版本")
    if not check_python_version():
        sys.exit(1)
    print(f"  ✓ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro} 符合要求")
    print()

    # 步骤 2: 检查/创建虚拟环境
    print("[步骤 2/4] 检查虚拟环境")
    venv_python = get_venv_python_path()

    if is_in_project_venv():
        print(f"  ✓ 已在项目虚拟环境中")
        print(f"    Python: {sys.executable}")
        print()
    else:
        if is_in_virtualenv():
            print(f"  ⚠ 当前在其他项目的虚拟环境中，将切换到本项目虚拟环境")
            print(f"    当前 Python: {sys.executable}")
        else:
            print(f"  当前使用系统 Python")

        # 确保虚拟环境存在
        if not create_virtualenv():
            sys.exit(1)

        if not os.path.isfile(venv_python):
            print(f"  ✗ 虚拟环境中的 Python 不存在: {venv_python}")
            sys.exit(1)

        # 重新执行自己
        activate_and_restart(venv_python)
        # 如果执行到这里，说明 os.execv 失败（不应该发生）
        print(f"  ✗ 切换到虚拟环境失败")
        sys.exit(1)

    # 步骤 3: 检查 download_all.py
    print("[步骤 3/4] 检查下载脚本")
    download_script = os.path.join(PROJECT_ROOT, "download_all.py")
    if not os.path.isfile(download_script):
        print(f"  ✗ 找不到 download_all.py")
        sys.exit(1)
    print(f"  ✓ download_all.py 存在")
    print()

    # 步骤 4: 调用 download_all.py
    print("[步骤 4/4] 开始部署")
    print()

    # 传递所有命令行参数给 download_all.py
    # sys.argv[1:] 包含用户传入的所有参数
    run_download_script(sys.argv[1:])


if __name__ == "__main__":
    main()
