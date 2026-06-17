# download_all.py 开发记录

## 目标
编写一个独立的模型预下载脚本 `download_all.py`，要求：
- 自动检测运行环境（系统、虚拟环境、依赖包）
- 自动检测网络环境（中国大陆 / 海外）
- 自动安装缺失的依赖包
- 一次性下载所有依赖模型
- 支持断点续传和自动重试
- 不依赖业务代码，换了环境也能直接用

## 需要下载的全部模型清单

### 分类一：主模型（IndexTTS-2 核心，约 5.7 GB）

| 文件 | 大小 | 用途 |
|------|------|------|
| `gpt.pth` | 3.3 GB | GPT 声学模型（核心） |
| `s2mel.pth` | 1.2 GB | DiT 声码器中间层 |
| `qwen0.6bemo4-merge/` | 1.2 GB | Qwen 情感识别模型（目录） |
| `feat1.pt` | 56 KB | 说话人特征矩阵 |
| `feat2.pt` | 367 KB | 情感特征矩阵 |
| `bpe.model` | 465 KB | 分词器模型 |
| `config.yaml` | 2.9 KB | 模型配置 |
| `wav2vec2bert_stats.pt` | 9.2 KB | 音频统计信息 |
| ~~`pinyin.vocab`~~ | ~~11 KB~~ | ~~拼音词表~~ | ~~IndexTTS-1 遗留，已移除~~ |

**下载源**：
- HuggingFace: `IndexTeam/IndexTTS-2`
- ModelScope: `IndexTeam/IndexTTS-2`

### 分类二：辅助模型（存放在 hf_cache/，约 2.5 GB）

| 模型 | 类型 | HF 仓库 | ModelScope 映射 | 大小 |
|------|------|---------|-----------------|------|
| w2v-bert-2.0 | 完整仓库 | facebook/w2v-bert-2.0 | AI-ModelScope/w2v-bert-2.0 | ~2GB |
| MaskGCT semantic codec | 单文件 | amphion/MaskGCT/semantic_codec/model.safetensors | 无 | ~169MB |
| CAMPPlus speaker embedding | 单文件 | funasr/campplus/campplus_cn_common.bin | iic/speech_campplus_sv_zh-cn_16k-common | ~200MB |
| BigVGAN vocoder | 多文件 | nvidia/bigvgan_v2_22khz_80band_256x | 无 | ~150MB |

### 分类三：示例音频（WebUI 演示用，约 11 MB）

| 文件 | 大小 | 用途 |
|------|------|------|
| voice_01~12.wav | ~10 MB | 演示音色 |
| emo_sad.wav, emo_hate.wav | ~1 MB | 情感样本 |

**下载源**：
- ModelScope: `https://modelscope.cn/studio/IndexTeam/IndexTTS-2-Demo/resolve/master/examples/`
- HuggingFace: `https://huggingface.co/spaces/IndexTeam/IndexTTS-2-Demo/resolve/main/examples/`

### 总计

| 分类 | 大小 | 必需 |
|------|------|------|
| 主模型 | 5.7 GB | ✓ |
| 辅助模型 | 2.5 GB | ✓ |
| 示例音频 | 11 MB | 可选 |
| **合计** | **约 8.2 GB** | |

## 已知问题
- huggingface.co 在国内返回 502 Bad Gateway
- hf-mirror.com 元数据可访问，但文件下载走 Xet 存储 (cas-bridge.xethub.hf.co)，在国内被墙
- modelscope 可以正常下载有映射的仓库
- 直接 HTTP 下载 (urllib) 从 hf-mirror.com 可行（不走 Xet）

## 核心依赖包清单

| 包名 | 用途 |
|------|------|
| torch | PyTorch 深度学习框架 |
| torchaudio | 音频处理 |
| modelscope | ModelScope 模型下载 |
| huggingface_hub | HuggingFace 模型下载 |
| transformers | 模型加载和推理 |
| safetensors | 安全张量格式 |
| gradio | WebUI 框架 |

## 开发对话记录

### 2026-06-17 - 环境检测模块

**用户要求**：
- 检测系统环境
- 检测是否存在虚拟环境
- 如果存在虚拟环境，检测系统环境和虚拟环境是否包含所需包
- 设置全局变量记录环境信息
- 打印全部环境信息

**实现**：
- 文件：`download_all.py`
- 全局变量：`G_OS_NAME`, `G_OS_VERSION`, `G_OS_ARCH`, `G_PYTHON_VERSION`, `G_PYTHON_EXECUTABLE`
- 虚拟环境：`G_IN_VIRTUALENV`, `G_VENV_PATH`, `G_VENV_NAME`
- 包信息：`G_PACKAGES`, `G_MISSING_PACKAGES`, `G_USE_MODELSCOPE`, `G_USE_HF_HUB`
- 函数：`detect_system_info()`, `detect_virtualenv()`, `check_package()`, `detect_packages()`, `detect_system_python_packages()`, `print_env_info()`, `init_env()`

**测试结果**：能正确检测虚拟环境、包安装状态、系统 Python 包情况

### 2026-06-17 - 增加 G_IN_CHINA 和命令行参数

**用户要求**：
- 增加 `G_IN_CHINA` 全局变量，默认值为 `True`
- 增加第一个位置参数，传入 `false` 时设置 `G_IN_CHINA = False`

**实现**：
- 新增 `G_IN_CHINA = True` 全局变量（网络环境区域）
- 使用 `argparse` 添加可选位置参数 `in_china`
- 支持 `true/false`、`1/0`、`yes/no` 多种写法
- 输出中新增 `[网络环境]` 区块显示中国大陆标识

**测试结果**：
- `python download_all.py` → 中国大陆: 是
- `python download_all.py false` → 中国大陆: 否

### 2026-06-17 - 安装缺失的包和下载工具

**用户要求**：
- 如果存在虚拟环境，在虚拟环境中安装
- 判断 G_IN_CHINA，如果为 true 使用阿里云作为 pip 源
- 安装过程输出进度条，以便用户可以看到是否卡住

**实现**：
- `get_pip_command()`: 获取当前 Python 环境对应的 pip 命令（使用 `python -m pip`）
- `get_pip_install_args()`: 构建 pip install 参数，根据 G_IN_CHINA 决定是否使用阿里云镜像
- `install_packages_with_progress()`: 安装包并显示实时进度
  - 实时输出 pip 的每一行日志
  - 每行前面添加时间戳和行号
  - 如果超过 60 秒没有新输出，显示警告提示
- `install_missing_packages()`: 安装所有缺失的依赖包
  - 显示缺失包列表
  - 确认安装环境（虚拟环境/系统 Python）
  - 确认镜像源
  - 询问用户确认
  - 安装完成后重新检测包状态
- 新增 `--no-install` 参数跳过自动安装

**测试结果**：
- 检测到缺失 torchaudio
- 确认安装到虚拟环境 [.browser-use-env]
- 使用阿里云镜像源
- 实时显示安装进度（带时间戳）
- 6.9 秒安装完成，重新检测显示已安装

### 2026-06-17 - 更新模型清单

**用户要求**：
- 增加主模型文件（gpt.pth、s2mel.pth、qwen 情感模型等）
- 确认 huggingface_hub 在检测清单中

**实现**：
- 新增 `MAIN_MODEL_REPOS` - 主模型仓库地址（HF/MS）
- 新增 `MAIN_MODEL_REQUIRED_FILES` - 主模型必需文件清单
- 新增 `MAIN_MODEL_REQUIRED_DIRS` - 主模型必需目录清单
- 新增 `AUX_MODELS` - 辅助模型清单（4 个模型）
- 新增 `EXAMPLE_AUDIO_FILES` - 示例音频文件清单（13 个文件）
- `huggingface_hub` 已在 `REQUIRED_PACKAGES` 中（第 60 行）
- 新增 `gradio` 到依赖包清单（WebUI 必需）

**模型总计**：约 8.2 GB（主模型 5.5 GB + 辅助模型 2.5 GB + 示例音频 11 MB）
**含依赖包总磁盘需求**：约 16.2 GB（.venv 依赖包 ~8.0 GB + 模型 ~8.2 GB）

### 2026-06-17 - 硬件检测 + 磁盘空间检查

**用户要求**：
- 检测操作系统（已有）、内存、硬盘、显卡、显卡驱动、CUDA/cuDNN 工具包
- 磁盘可用空间 < 总模型大小 × 3 时，提示磁盘空间不足（下载过程会产生临时文件）

**实现**：
- 新增全局变量：`G_RAM_TOTAL`, `G_RAM_AVAILABLE`, `G_DISK_TOTAL`, `G_DISK_USED`, `G_DISK_FREE`
- 新增全局变量：`G_GPU_NAME`, `G_GPU_VRAM`, `G_GPU_COUNT`, `G_CUDA_VERSION`, `G_CUDNN_VERSION`
- 新增全局变量：`G_NVIDIA_DRIVER`, `G_DISK_INSUFFICIENT`, `TOTAL_MODEL_SIZE_ESTIMATE`
- 新增 `detect_hardware(model_dir)` 函数，内部调用三个子函数：
  - `_detect_memory()`: Linux 读 `/proc/meminfo`，macOS 用 `os.sysconf`，Windows 用 `systeminfo`，回退 `psutil`
  - `_detect_disk(model_dir)`: 使用 `shutil.disk_usage()`，检查可用空间是否 < 3 × 8.2GB
  - `_detect_gpu()`: 通过 `torch.cuda` 检测 GPU/CUDA/cuDNN，通过 `nvidia-smi` 获取驱动版本
- `init_env()` 增加 `model_dir` 参数，调用 `detect_hardware(model_dir)`
- `print_env_info()` 新增 `[硬件信息]` 区块和 `[磁盘空间检查]` 区块

**测试结果**：
- 内存: 93.9 GB (可用 87.4 GB) ✓
- 磁盘: 1907.7 GB (已用 420.2 GB, 可用 1487.5 GB) ✓
- GPU: NVIDIA GeForce RTX 5090 D (31.8 GB) ✓
- CUDA: 12.8 ✓
- cuDNN: 91002 ✓
- NVIDIA 驱动: 610.47 ✓
- 磁盘空间检查: ✓ 空间充足 ✓

### 2026-06-17 - GPU 型号识别 + torch/CUDA 版本自动推荐

**用户要求**：
- 根据显卡型号自动调整 torch、CUDA 和其他包的版本
- 让脚本可以给所有 N 卡用户使用
- 如果没有 nvidia-smi 不需要让用户装一个（随驱动安装，不可单独安装）
- 如果 Python 版本太高或太低要给出提示

**实现**：

1. **GPU 架构识别** `_identify_gpu_architecture(gpu_name)`:
   - Blackwell (RTX 50xx), Ada (RTX 40xx), Ampere (RTX 30xx, A100)
   - Turing (RTX 20xx, GTX 16xx), Pascal (GTX 10xx), Maxwell (GTX 9xx)
   - Volta (V100), Hopper (H100)

2. **驱动-CUDA 兼容性表** `DRIVER_CUDA_COMPAT`:
   - 590.xx → CUDA 13.x, 560.xx → 12.7, 550.xx → 12.6, ...
   - 数据来源: NVIDIA 官方 CUDA compatibility 文档

3. **GPU 架构最低 CUDA 要求** `ARCH_MIN_CUDA`:
   - Blackwell ≥ 12.0, Ada ≥ 11.4, Ampere ≥ 11.0, Turing ≥ 10.0

4. **版本推荐** `_recommend_torch_cuda(gpu_arch, driver_version)`:
   - 从 `AVAILABLE_TORCH_CUDA` 列表中选择最高且驱动支持的版本
   - 回退: cu118 (最广泛兼容)

5. **nvidia-smi 优先检测策略**:
   - 优先 nvidia-smi（最准确，含驱动版本 + GPU 型号 + 显存）
   - 回退 torch.cuda（如果 torch 已安装）
   - 无论 nvidia-smi 是否成功，都尝试从 torch 获取已装 CUDA/cuDNN 版本

6. **包安装分离**:
   - PyTorch 相关包 (torch, torchaudio, torchvision) 使用 `--extra-index-url` 从 CUDA 源安装
   - 其他包从阿里云/PyPI 安装

7. **Python 版本兼容性检查** `check_python_compatibility()`:
   - 范围: 3.10 ~ 3.13 (pyproject.toml 要求 + torch 预编译 wheel 支持)
   - < 3.10 → 直接退出 (sys.exit(1))，提示升级到 3.10~3.12，给出 uv/conda 命令
   - > 3.13 → 直接退出 (sys.exit(1))，提示降级到 3.10~3.12，给出 uv/conda 命令
   - 3.10~3.13 → 正常运行，无提示

**全局变量新增**:
- `G_GPU_ARCH`, `G_RECOMMENDED_CUDA`, `G_RECOMMENDED_TORCH_INDEX`
- `MIN_PYTHON_VERSION`, `MAX_PYTHON_VERSION`, `RECOMMENDED_PYTHON`

**输出新增**:
- GPU 架构代号显示: `[Blackwell]`
- 已装 CUDA/cuDNN 版本
- 推荐 CUDA 版本和 torch 安装源
- Python 版本兼容性警告/错误

### 2026-06-17 - 辅助模型下载模块

**用户要求**：
- 下载 4 个辅助模型（w2v-bert-2.0, MaskGCT, CAMPPlus, BigVGAN）
- 规则和主模型一样：多源回退、进度条、断点续传、自动重试

**实现**：

1. **HF → ModelScope 映射表** `HF_TO_MODELSCOPE_MAP`:
   - `facebook/w2v-bert-2.0` → `AI-ModelScope/w2v-bert-2.0`
   - `funasr/campplus` → `iic/speech_campplus_sv_zh-cn_16k-common`
   - `nvidia/bigvgan_v2_22khz_80band_256x` → 无映射
   - `amphion/MaskGCT` → 无映射

2. **多源回退下载** `download_single_file_from_hf()`:
   - 国内: ModelScope (如有映射) → hf-mirror HTTP
   - 海外: huggingface HTTP → hf-mirror HTTP → ModelScope (如有映射)

3. **HTTP 下载工具** `_download_file_http()`:
   - 断点续传: Range headers + .tmp 临时文件 + 追加模式写入
   - 自动重试: 指数退避 (2s, 4s, 8s, max 30s)
   - 实时进度: 百分比 + 已下载/总大小 + 下载速度 (MB/s)
   - SSL 容错: 禁用证书验证

4. **各模型下载函数**:
   - `download_w2v_bert()`: ~2GB 完整仓库，优先 ModelScope，回退 huggingface_hub
   - `download_maskgct()`: ~169MB 单文件，直接 hf-mirror HTTP
   - `download_campplus()`: ~200MB 单文件，优先 ModelScope
   - `download_bigvgan()`: ~150MB 多文件 (config.json + bigvgan_generator.pt)，直接 hf-mirror HTTP

5. **总控函数** `download_aux_models()`:
   - 按顺序下载，显示 [1/4]~[4/4] 进度
   - 下载前询问用户确认，下载后汇总结果和总大小

6. **主函数集成**:
   - `main()` 中 `download_main_model()` 之后调用 `download_aux_models()`
   - `--skip-model` 同时跳过主模型和辅助模型

**国内下载源分配**:
| 模型 | 主要下载源 | 备用下载源 |
|------|-----------|-----------|
| w2v-bert-2.0 | ModelScope | hf-mirror snapshot |
| MaskGCT | hf-mirror HTTP | — |
| CAMPPlus | ModelScope | hf-mirror HTTP |
| BigVGAN | hf-mirror HTTP | — |

**测试结果**：
- BigVGAN 下载测试: config.json (1.4 KB) ✓, bigvgan_generator.pt (428 MB) 下载正常
- 断点续传 .tmp 文件机制正常
- 进度条 + 速度显示正常
- URL 确认使用 hf-mirror.com 国内镜像

### 2026-06-17 - 示例音频下载模块

**用户要求**：
- 下载 13 个示例音频文件（voice_01~09.wav, voice_11.wav, voice_12.wav, emo_sad.wav, emo_hate.wav）
- 规则和主模型一样

**实现**：

1. **远程下载源**:
   - ModelScope: `https://modelscope.cn/studio/IndexTeam/IndexTTS-2-Demo/resolve/master/examples/`
   - HuggingFace: `https://huggingface.co/spaces/IndexTeam/IndexTTS-2-Demo/resolve/main/examples/`

2. **多源回退策略**:
   - 国内: ModelScope → HuggingFace
   - 海外: HuggingFace → ModelScope

3. **下载函数** `download_example_audio()`:
   - 复用 `_download_file_http()` 工具函数（断点续传、自动重试、进度显示）
   - 下载前检查已存在文件，跳过无需下载的文件
   - 显示 [1/13], [2/13], ... 进度
   - 下载前询问用户确认
   - 下载后汇总结果 (✓/✗) 和总大小

4. **命令行参数**:
   - `--skip-examples`: 跳过示例音频下载
   - `--examples-dir`: 指定示例音频存放目录 (默认: ./examples)

5. **主函数集成**:
   - `main()` 中 `download_aux_models()` 之后调用 `download_example_audio()`
   - `--force` 强制重新下载

**测试结果**：
- 13 个文件全部从 ModelScope 下载成功
- 进度条 + 速度显示正常
- 总大小: 9.8 MB
- 已存在文件正确跳过

### 2026-06-17 - 一键部署脚本 start_all.py

**用户要求**：
- 保留 download_all.py 不变
- 创建 start_all.py 作为一键部署入口
- git clone 后直接运行 `python start_all.py` 完成全部部署

**实现**：

1. **自动虚拟环境管理**:
   - 检测当前是否在虚拟环境中
   - 不在 → 检查项目根目录 `.venv` 是否存在
   - 不存在 → 优先 `uv venv`，回退 `python -m venv`
   - 使用 `os.execv()` 重新执行自己，切换到虚拟环境 Python

2. **Python 版本检查**:
   - 范围: 3.10 ~ 3.13
   - 不兼容 → 提示升级/降级，给出 uv/conda 命令

3. **调用 download_all.py**:
   - 传递所有命令行参数
   - 返回退出码

4. **命令行参数**:
   - `--skip-model`: 跳过模型下载
   - `--skip-examples`: 跳过示例音频
   - `--no-install`: 跳过依赖包安装
   - `--force`: 强制重新下载
   - `--model-dir DIR`: 模型目录
   - `--examples-dir DIR`: 示例音频目录
   - `false`: 非中国大陆网络环境
   - `-h, --help`: 显示帮助

**使用流程**:
```bash
git clone <repo>
cd index-tts
python start_all.py  # 完成全部部署
```

**测试结果**：
- 帮助信息显示正确
- Python 版本检测正常
- 虚拟环境检测正常
- download_all.py 调用正常
