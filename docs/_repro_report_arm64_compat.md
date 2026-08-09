# DrugCLIP (Drug-The-Whole-Genome) — arm64 / Blackwell 迁移兼容性调研报告

- **调研日期**: 2026-08（实测于 blue-whale / sperm-whale 两只 aarch64 鲸鱼）
- **目标机器**: NVIDIA GB10 (DGX Spark 级), aarch64, compute capability **12.1 (sm_121)**, driver **580.173.02**（对应 CUDA 13.0），host 已装 CUDA 13.0 toolkit（`/usr/local/cuda-13.0`），Docker 29.2.1 linux/arm64 + nvidia-container-toolkit 1.19.1
- **现状**: Dockerfile 基于 `nvidia/cuda:12.1.1-cudnn8-devel-ubuntu20.04`（x86_64 构建），conda 装 pytorch==2.1.2 / pytorch-cuda==12.1 / faiss-gpu==1.8.0，python 3.9
- **结论先行**: 该镜像在 arm64 上**无法构建也无法运行**，需整体重写基础镜像与 torch 安装路径；**torch 2.7.x 在 linux_aarch64 上没有 CUDA 轮子**（最低 CUDA 版是 2.8.0+cu129），**官方对 DGX Spark 的推荐是 `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130`**（即 torch 2.9.x），本报告主推 **torch 2.9.1+cu130 + Python 3.11 纯 pip 方案**。

---

## 1. 依赖清单（静态分析汇总）

### 1.1 Dockerfile 内安装项
| 来源 | 内容 |
|---|---|
| apt | build-essential, cmake, g++-7, git, curl, wget, libjpeg/libpng-dev, librdmacm1, libibverbs1, ibverbs-providers, zsh/vim/zip/unzip/rsync/htop 等工具、字体包 |
| conda | python 3.9、pytorch==2.1.2、torchvision==0.16.2、torchaudio==2.1.2、pytorch-cuda==12.1、faiss-gpu==1.8.0 |
| pip（Dockerfile 内） | iopath, lmdb, ml_collections, numpy, scipy, tensorboardX, tqdm, tokenizers, typing-extensions>=4.15.0 |
| git | Uni-Core（dptech-corp/Uni-Core，`pip install --no-build-isolation .`） |
| pip（requirements.txt） | lightning==2.2.0, torchmetrics, hydra-core==1.3.2, hydra-colorlog, hydra-optuna-sweeper, wandb, einops, pandas==1.5.3, matplotlib, scikit-learn==1.2.2, scipy==1.10.1, nanopq, transformers==4.36.0, biotite==0.37.0, biopython==1.81, rdkit==2023.9.5, biopandas, h5py, python-dotenv, rich, pytest, sh, ipython, jupyterlab, jsonlines, ray, pre-commit, black, isort, flake8, nbstripout |

### 1.2 environment.yaml（conda 备选路径，已标注 x86 专用）
`pytorch=2.1.2 / torchvision=0.16.2 / torchaudio=2.1.2 / pytorch-cuda=11.8 / faiss-gpu=1.8.0 / python=3.9` — **该 conda 路径在 aarch64 上整体不可行**（见 §2.2）。

### 1.3 代码真实使用面（精确 grep 验证）
- **有引用（必须保留）**: torch（大量）、unicore（checkpoint_utils / distributed_utils / options / utils / tasks / metrics / losses）、rdkit（27 处）、pandas（7）、sklearn（10）、scipy（1）、biopython `Bio.PDB`（17）、h5py（1）、lmdb（8）、numpy。
- **零引用（可删除，共 13 项）**: **faiss**（全仓无 `import faiss`）、lightning、torchmetrics、wandb（unicore 传递依赖会带上）、hydra-*（3 项）、einops、matplotlib、nanopq、transformers、biotite、biopandas、ray、optuna。dev/工具类（ipython/jupyterlab/pytest/sh/pre-commit/black/isort/flake8/nbstripout/jsonlines/dotenv/rich）按需保留。

---

## 2. arm64 可用性实测矩阵（PyPI JSON API + pip dry-run，跑在 blue-whale 上）

### 2.1 关键项：torch 系列（linux_aarch64 CUDA 轮子）

| 版本 | PyPI CPU (default index) | download.pytorch.org CUDA 索引 |
|---|---|---|
| torch 2.6.0 | ✅ cp39/310/311/312 aarch64 | cu126/cu128 均**无** aarch64 |
| torch 2.7.1 | ✅ cp39/310/311/312 aarch64 | cu126/cu128 均**无** aarch64（实测 `pip download --platform linux_aarch64` → "from versions: none"）|
| torch 2.8.0 | ✅ cp39/310/311/312 aarch64 | **cu129 起有**：`torch-2.8.0+cu129-cp{310,311,312,313}-manylinux_2_28_aarch64.whl` ✅ |
| torch 2.9.1 | ✅ cp310/311/312（无 cp39）| cu129 ✅、**cu130 ✅**（`torch-2.9.1+cu130-cp311-cp311-manylinux_2_28_aarch64.whl` 实测存在）|
| cu126/cu128 索引 | — | aarch64 文件**最早只到 2.10.0**（2.7/2.8/2.9 全部缺失）|

> **结论：torch 2.7.x + CUDA 在 arm64 上不存在任何轮子。** 满足 "2.7+" 的**最低** CUDA 版本是 **torch 2.8.0+cu129**；官方推荐 **torch 2.9.1+cu130**（NVIDIA 论坛/DGX Spark 文档原话：`pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130`）。GB10 是 sm_121，2.9.x+cu130 已含 Blackwell 架构支持。

配套版本（同样有 aarch64 轮子，已逐一验证文件名）：
- torch 2.9.1+cu130 → torchvision **0.24.1**、torchaudio **2.9.1**（cu130 索引）
- torch 2.8.0+cu129 → torchvision **0.23.0**、torchaudio **2.8.0**（cu129 索引）

cu130 全依赖解析实测（`pip install --dry-run` on aarch64，全部成功，均来自 pypi.nvidia.com 的 aarch64 轮子）：
`nvidia-cublas 13.0.0.19`、`nvidia-cufft 12.0.0.15`、`nvidia-curand/cusolver/cusparse`、`nvidia-cusparselt-cu13 0.8.0`、`nvidia-nccl-cu13 2.27.7`、`nvidia-nvshmem-cu13`、`nvidia-nvtx`、`nvidia-nvjitlink`、`nvidia-cufile`、`triton 3.5.1`（aarch64 ✅）。**torch 轮子自带 CUDA 库，容器内无需 CUDA toolkit，只需 host driver ≥ 580（本机 580.173.02 ✅）**。

### 2.2 conda 通道（linux-aarch64）实测
- `pytorch` 官方通道（api.anaconda.org/package/pytorch/pytorch）：**0 个 linux-aarch64 文件**（仅 linux-64/osx-64/win-64/osx-arm64）
- `pytorch/pytorch-cuda`：无 linux-aarch64
- **结论：conda 装 CUDA torch 在 aarch64 上不可行，必须走 pip 轮子。** environment.yaml 的 pytorch 段只能用于 x86，或整体废弃。
- 附带发现：conda-forge 的 biotite 有 linux-aarch64 构建（py311/312/313），但本项目零引用，无需理会。

### 2.3 其余依赖 aarch64 可用性（版本号 = 实测到的具体轮子）

| 包 | 当前版本 | arm64 轮子 | 需升级/处理 | 备注 |
|---|---|---|---|---|
| **rdkit** | 2023.9.5 | ✅ cp39/310/311/312 | **无需升级** | manylinux_2_17 aarch64 |
| **pandas** | 1.5.3 | ✅ cp39/310/311（**无 cp312**）| 无需升级；**Python 必须 ≤3.11** | 若 py312 则需升 pandas≥2.x |
| **scikit-learn** | 1.2.2 | ✅ cp39/310/311 | 同左 | |
| **scipy** | 1.10.1 | ✅ cp39/310/311 | 同左；依赖 numpy<1.27 | |
| **biopython** | 1.81 | ❌ 无 | **升 ≥1.84**（1.84/1.85 ✅ cp39-312）| 1.83 也无；代码只用 Bio.PDB 稳定 API，风险低 |
| **biotite** | 0.37.0 | ❌ 全系无（0.38–0.41 全测）| **删除**（零引用）| 保留则需 sdist+Cython 构建或 conda-forge |
| **transformers** | 4.36.0 | 纯 py 通用轮子 | **删除**（零引用）| 若保留需 tokenizers<0.19 |
| **lightning** | 2.2.0 | 纯 py ✅，pip 约束 torch<4.0 无冲突 | **删除**（零引用）| 官方 CI 矩阵只测到 torch 2.2/2.3，torch 2.9 属非官方支持 |
| **torchmetrics** | unpinned | 纯 py ✅ | 随 lightning 删除 | |
| **hydra-core** 1.3.2 / colorlog / optuna-sweeper | 固定 | 纯 py ✅ | 可删（零引用）| |
| **h5py** | unpinned | **3.12.1+ ✅**（3.11.0 无）| 建议显式 `h5py>=3.12.1` | 当前 unpinned 会解析到 3.16.0 ✅ |
| **lmdb** | unpinned | ✅ cp39-312 | 无需处理 | |
| **iopath** | 0.1.10 | 仅 sdist，**纯 Python 无 Cython** | pip 源码安装 OK；或删除（零引用）| setup.py 已核实 |
| **ml_collections / tensorboardX / typing-extensions** | unpinned | 纯 py ✅ | 保留（unicore 依赖）| |
| **tokenizers** | unpinned | **0.20.3 ✅ cp311**；0.21.1/0.22.0/0.23.1 **仅 cp310** | **需 pin `tokenizers==0.20.3`（py311）** | 否则 py311 下 pip 会走 sdist 触发 Rust 编译 |
| **numpy** | unpinned | 1.26.4 ✅ cp39-312 | **必须 pin `numpy==1.26.4`** | pandas 1.5.3/scipy 1.10.1 与 numpy 2.x 不兼容；torch cu130 也兼容 1.26 |
| **ray** | unpinned | ✅ cp39-312 | 删除（零引用）| |
| **einops / nanopq / biopandas / matplotlib / wandb** | unpinned | 纯 py ✅ | 删除（零引用）| matplotlib 3.9.4+ 有 cp39-312 aarch64，如需留则 pin |
| **unicore**（git）| master | 纯 Python 安装 | 见 §3.4 | setup.py：`DISABLE_CUDA_EXTENSION=True` 默认，torch 要求仅 `>=2.0.0` |

### 2.4 Python 版本选择
- **推荐 Python 3.11**：全部关键包都有 cp311 aarch64 轮子（torch 2.9.1+cu130 ✅、rdkit ✅、pandas 1.5.3 ✅、sklearn/scipy ✅、biopython 1.84+ ✅）。
- py39：torch 2.9 已无 cp39 轮子（最多 2.8.0），且生态在 2026 已淘汰，不推荐。
- py312：torch ✅ 但 pandas 1.5.3 / sklearn 1.2.2 / scipy 1.10.1 **无 cp312 aarch64 轮子**，会被迫升级这三个科学计算包，引入代码回归风险。

---

## 3. Dockerfile 修改方案（逐段）

> 推荐改造思路：**放弃 conda + x86 CUDA 基础镜像，改为 arm64 官方 Python 镜像 + 纯 pip（cu130 索引）**。pip 轮子自带 CUDA 库，容器内不需要 CUDA toolkit/cudnn，host 驱动 580.173.02 直接满足。

### 3.1 基础镜像 + 全局参数
```dockerfile
# 原: FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu20.04
FROM python:3.11-slim-bookworm        # 官方 multi-arch，arm64 直接可用（glibc 2.36 ≥ manylinux_2_28 要求）
ARG http_proxy
ARG https_proxy
ENV LANG=C.UTF-8 \
    DEBIAN_FRONTEND=noninteractive \
    TORCH_CUDA_ARCH_LIST="12.1"        # GB10 sm_121；pip 轮子预编译不需要，仅编译扩展时生效
```
- 备选：`arm64v8/ubuntu:24.04` + apt 装 python3.11（deadsnakes）——多一步，无必要。
- 备选（不推荐作主路径）：以 `nvcr.io/nvidia/pytorch:25.10-py3`（官方 DGX Spark aarch64 容器）为基础镜像，自带 torch 2.10 系；缺点：镜像巨大、与现有 Dockerfile 结构差异大、NGC 拉取需 API key。
- `nvidia/cuda:12.1.1-*` 虽有 arm64 tag，但 CUDA 12.1 对 sm_121/驱动 580 无意义且没有对应 torch 轮子，**不要沿用**。

### 3.2 apt 段（arm64 关键修正）
```dockerfile
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git curl wget ca-certificates \
    libjpeg-dev libpng-dev tzdata libgl1 libglib2.0-0 fontconfig \
    && rm -rf /var/lib/apt/lists/*
```
- ⚠️ **arm64 Ubuntu 的 apt 源是 `ports.ubuntu.com`，不是 `archive.ubuntu.com`**。原 Dockerfile 的 TUNA 源替换 sed 只处理了 archive/security 域名——arm64 上必须改为处理 `ports.ubuntu.com`（TUNA 提供 `ubuntu-ports` 镜像）。建议：基础镜像保留官方源或显式写成 `mirrors.tuna.tsinghua.edu.cn/ubuntu-ports`。
- `g++-7`、`librdmacm1/libibverbs1/ibverbs-providers`（InfiniBand，双机 NCCL 用）：Ubuntu 24.04/bookworm 下 g++-7 已不在仓库，直接删；ibverbs 包名在 bookworm 为 `libibverbs-dev`/`ibverbs-providers`，可选保留。
- 工具段（zsh/vim/zip/…）与字体段可原样保留（纯 apt/wget，与架构无关）。

### 3.3 PyTorch 安装段（替换整个 conda 块 + faiss）
```dockerfile
# 原: conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda==12.1 faiss-gpu=1.8.0 -c pytorch -c nvidia -c conda-forge
# 新: 纯 pip，官方 DGX Spark 推荐路径（dry-run 已在 aarch64 实测全部依赖可解析）
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu130 \
    torch==2.9.1+cu130 \
    torchvision==0.24.1 \
    torchaudio==2.9.1
```
- 若希望"最小改动"（保持 torch 2.8 线）：`--index-url https://download.pytorch.org/whl/cu129` + `torch==2.8.0+cu129 torchvision==0.23.0 torchaudio==2.8.0`（同样有 aarch64 轮子，已实测）。
- 原 `faiss-gpu=1.8.0` **整行删除**（PyPI 上无此包、全仓无 import）。
- 原 conda 通道配置（~/.condarc TUNA）与 Miniconda 安装段**整体删除**。
- ⚠️ 网络提示：`--index-url` 指向 download.pytorch.org 后，nvidia-* 依赖会自动从 `pypi.nvidia.com` 解析（dry-run 日志可见，均为 aarch64 轮子）；若网络受限需保留 TUNA pip 源，可改用 `--extra-index-url https://download.pytorch.org/whl/cu130` 并将 torch 系 pin 到 `==2.9.1+cu130` 等本地版本串。

### 3.4 Uni-Core 段（基本保留，加版本 pin）
```dockerfile
RUN pip install --no-cache-dir numpy==1.26.4 \
    lmdb ml_collections tensorboardX tqdm tokenizers==0.20.3 \
    "typing-extensions>=4.15.0"
# 原注释中的 typing-extensions 坑依然成立；tokenizers 必须 pin 0.20.3（py311 aarch64 无 0.21+ 轮子）
RUN git clone https://github.com/dptech-corp/Uni-Core.git /tmp/Uni-core \
    && cd /tmp/Uni-core \
    && git checkout <固定commit> \          # 建议 pin，避免 master 漂移（现状不可复现）
    && pip install --no-build-isolation . \
    && rm -rf /tmp/Uni-core
```
- 保留 `--no-build-isolation`（setup.py 顶部 `import torch`，需要已装的 torch）。
- `DISABLE_CUDA_EXTENSION=True` 默认生效 → 纯 Python 安装，**aarch64 上无需 nvcc/无需编译 CUDA kernel**；setup.py 已核实 torch 约束仅 `>=2.0.0`，2.9.1 满足。
- iopath 可删（零引用；且 0.1.10 仅 sdist，纯 Python 装起来也快，删不删皆可）。

### 3.5 requirements.txt 修改（精简 + 升级）
```text
# 删除整段: lightning==2.2.0 / torchmetrics（零引用）
# 删除整段: hydra-core / hydra-colorlog / hydra-optuna-sweeper（零引用）
# 删除: wandb einops matplotlib nanopq transformers==4.36.0 biotite==0.37.0 biopandas ray
# 升级: biopython==1.81 -> biopython==1.85
# 保留: pandas==1.5.3 scikit-learn==1.2.2 scipy==1.10.1 rdkit==2023.9.5
# 显式: h5py>=3.12.1, numpy==1.26.4（放这里而非 Dockerfile 也可，但需先于 unicore 装）
# 工具类: python-dotenv rich pytest ipython jupyterlab jsonlines + linters 按需保留（均纯 py）
```
- requirements.txt 顶部关于 "Do NOT reinstall torch with pip" 的注释要改：现在 torch 本身就是 pip 装的。

### 3.6 environment.yaml
- 标注或删除：pytorch 通道在 aarch64 无构建，`pytorch-cuda=11.8`、`faiss-gpu` 在 arm64 均不可用。若要保留一份"无 Docker 的 arm64 环境"脚本，应改为：conda 建 py311 空环境 + 本报告 §3.3/3.4 的 pip 命令。

---

## 4. torch 升级对代码的影响评估（2.1.2 → 2.9.1+cu130）

| 关注点 | 结论 | 证据 |
|---|---|---|
| unicore `torch.load` 默认 weights_only=True（torch 2.6+ 的著名破坏点）| **无影响**：Uni-Core master 两处 `torch.load`（checkpoint_utils.py:251、distributed/utils.py:494）均已显式 `weights_only=False`；drugclip 的加载走 `checkpoint_utils.load_checkpoint_to_cpu` | 已读 Uni-Core 源码 |
| unicore API 面 | **全部存在**：`call_main`、`load_checkpoint_to_cpu`、`move_to_cuda`、`get_activation_fn`、`numpy_seed`、`data()`、`get_validation_parser`、`add_model_args`、`parse_args_and_arch` 均在 master 中 | grep 核实 |
| fp16（drugclip 的 `--fp16` 路径）| **无影响**：unicore 用 fairseq 风格自研 DynamicLossScaler（unicore/optim/fp16_optimizer.py），不碰 `torch.cuda.amp`/`GradScaler`；脚本参数 `--fp16-init-scale/--fp16-scale-window` 均保留。推理侧 `model.half()` + `model.cuda()` 老式用法在 2.9 仍完全兼容 | 源码 + 脚本核实 |
| 分布式 API（双机 NCCL 场景）| **无影响**：unicore 只用稳定 API：`init_process_group / get_rank / get_world_size / all_reduce / all_gather / all_to_all_single / broadcast`；`--ddp-backend=c10d` 仍有效。cu130 轮子自带 NCCL 2.27.7（aarch64 ✅）| grep 核实 |
| `torch.distributed.launch` | drugclip.sh:37 用 `python -m torch.distributed.launch`（2.9 仍可用但已 deprecated）；pretrain.sh 已用 `torchrun` → **建议统一改 torchrun** | 脚本核实 |
| 模型层 API | transformer_encoder_with_pair / unimol / drugclip 只用 `F.linear / F.dropout / masked_fill_ / relu / matmul / bmm / einsum` 等稳定 API；无 `torch.compile`/triton 显式依赖 | 源码核实 |
| 预存在问题（非迁移引入）| `models/drugclip.py:74` `nn.Parameter(torch.ones([1], device="cuda") * np.log(14))` 硬编码 cuda —— 任何 CPU 推理路径都会崩；与 torch 版本无关，但迁移时可顺手改为 `device=self.device` 或延迟创建 | 源码核实 |
| torchvision/torchaudio | 代码零引用，保留仅作生态对齐；删掉也不影响任何脚本 | grep 核实 |

---

## 5. 风险点清单（按严重度排序）

1. **torch 2.7.x + CUDA 在 arm64 上不存在**：用户的 "torch 2.7+" 诉求在 arm64 上最低只能到 **2.8.0+cu129**，官方推荐 **2.9.1+cu130**；若强行用 PyPI 的 CPU aarch64 轮子（2.7.1 有），GPU 完全不可用。
2. **镜像体积暴增**：cu130 aarch64 轮子 torch 本体 ~512MB，nvidia-* 依赖合计 ~2GB+；初始构建下载量大（3 个 dry-run 实测均需拉满整轮子）。
3. **tokenizers 版本陷阱**：0.21+/0.22/0.23 在 py311+aarch64 无轮子（只有 cp310），不 pin 就会触发 Rust 源码编译（甚至失败）；**必须 pin 0.20.3**。
4. **numpy 2.x 陷阱**：torch 依赖里 numpy 无上限，若让 pip 自由解析会装到 numpy 2.x，**pandas 1.5.3 / scipy 1.10.1 直接崩溃**；必须 pin numpy==1.26.4。
5. **Python 版本与科学栈耦合**：pandas 1.5.3/sklearn 1.2.2/scipy 1.10.1 无 cp312 aarch64 轮子 → Python 锁 **3.11**；3.9 会被 torch 2.9 抛弃。
6. **Uni-Core 未 pin**：Dockerfile `git clone master` 不可复现；master 目前是兼容的（weights_only=False 等），但建议 checkout 固定 commit。
7. **双机 NCCL 未验证**：compose 用 host 网络 + 10.100.0.0 双机直连，torch cu130 自带 NCCL aarch64 可用，但 GB10 上 NCCL 拓扑/两机 RDMA 行为需跑通后实测；RDMA 依赖的 ibverbs 相关 apt 包在 arm64 镜像上要按 bookworm 包名重配。
8. **apt 源架构坑**：arm64 Ubuntu 的源是 `ports.ubuntu.com`，原 Dockerfile 的 TUNA sed 对 arm64 失效，需改为 ubuntu-ports 镜像。
9. **biotite 若未来要用**：全系无 aarch64 pip 轮子，需走 conda-forge linux-aarch64（有 py311 构建）或源码 + Cython 构建；当前零引用已删。
10. **行为差异点**：`--fp16` 在 2.9 上仍走 unicore 自研 scaler，但 cu130 轮子的 BF16/FP16 kernel 行为与 2.1.2 不同，预训练/微调数字可能出现微小漂移（属预期，非错误）。

---

## 6. 验证方法与证据（可复现命令）

以下命令均在 `blue-whale.local`（aarch64/GB10）实测，网络可达 pypi / download.pytorch.org / pypi.nvidia.com / api.anaconda.org：

```bash
# 1) 逐包 aarch64 轮子存在性（PyPI JSON API，全量矩阵）
python3 /tmp/dtwg_check_aarch64.py     # 38 项，见 §2.3 结果

# 2) CUDA 索引 aarch64 轮子直接验证（文件名级）
curl -s https://download.pytorch.org/whl/cu130/torch/ | grep -o "torch-2.9.1[^\"]*aarch64[^\"]*\.whl"
curl -s https://download.pytorch.org/whl/cu129/torch/ | grep -o "torch-2.8.0[^\"]*aarch64[^\"]*\.whl"
# 结果: cu126/cu128 无 aarch64；cu129 自 2.8.0 起；cu130 自 2.9.0 起

# 3) 全依赖解析 dry-run（证明 nvidia-* 传递依赖 aarch64 可解）
pip install --dry-run --break-system-packages --only-binary=:all: \
  "torch==2.9.1+cu130" "torchvision==0.24.1" "torchaudio==2.9.1" \
  --index-url https://download.pytorch.org/whl/cu130
# 日志确认: nvidia-cublas/cufft/curand/cusolver/cusparse/cusparselt/nccl/nvshmem/nvtx/nvjitlink/cufile + triton 3.5.1 全部 aarch64

# 4) conda 通道（证明不可行）
curl -s https://api.anaconda.org/package/pytorch/pytorch | python3 -c "..."   # linux-aarch64 文件数 = 0
curl -s https://api.anaconda.org/package/pytorch/pytorch-cuda | python3 -c "..." # 无 linux-aarch64

# 5) 本机静态分析
grep -rn "import faiss" --include="*.py" .          # 0 结果 → faiss 可删
grep -rnw --include="*.py" -E "lightning|transformers|biotite|ray|wandb" src/ # 0 结果
```
