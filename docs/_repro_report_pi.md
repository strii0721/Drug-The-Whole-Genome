# DrugCLIP (Drug-The-Whole-Genome) 复现准备 · 静态分析报告

> 生成时间：2026-08-09
> 方式：纯静态分析（读脚本/源码 + 查 conda 通道 repodata + 查 HuggingFace 数据集文件清单），未下载大文件、未装环境。
> 目标机器：Ubuntu 24.04 (WSL2) + RTX 4090 Laptop GPU ×1 + 驱动 610.43 / CUDA 13.3

---

## 0. 本机环境实测快照

| 项目 | 实测结果 |
|---|---|
| OS | Ubuntu 24.04.4 LTS，内核 6.18.33.1-microsoft-standard-WSL2 |
| GPU | NVIDIA GeForce RTX 4090 Laptop GPU，**仅 1 张**（GPU 0） |
| 驱动 | NVIDIA-SMI 610.43.02，CUDA UMD 13.3 |
| 系统 CUDA | nvcc 13.3（系统级 toolkit 已装） |
| 内存 / 磁盘 | 31 GB RAM；`/` 剩余 926 GB |
| conda | **未安装**（`which conda/mamba/micromamba` 均无） |
| 系统 python | 3.12.3（与项目要求的 3.9 无关，conda 环境自建） |
| `./data/` | 目前仅有 `targets.zip`（258 MB，刚下载）；缺 DUD-E.zip / LIT-PCBA.zip / model_weights.zip / encoded_mol_embs.zip |

---

## 1. `build_conda_env.sh` + `environment.yaml` 无 Docker 搭建流程审查

脚本原文（只有两行）：

```bash
conda env create -f environment.yaml
git clone https://github.com/dptech-corp/Uni-Core.git /tmp/Uni-core && pip install --no-build-isolation /tmp/Uni-core
```

### 1.1 致命问题（按执行顺序必然失败）

**F1. `environment.yaml` 引用了不存在的 `docker/requirements.txt` → `conda env create` 在 pip 阶段必败。**
`environment.yaml` 的 pip 段写的是 `- -r docker/requirements.txt`，但该文件在 commit `3fe8c92`（2026-07-23）已从 `docker/requirements.txt` **移动为仓库根目录的 `requirements.txt`**，仓库里已没有 `docker/` 目录（已用 `git ls-tree HEAD` 与磁盘双重确认）。`conda env create` 会在装完所有 conda 包（含 pytorch ≈ 2.7 GB 的 cuda 包下载）之后，在 pip 阶段报 `ERROR: Could not open requirements file: ... 'docker/requirements.txt'` 失败。
→ **应改为 `- -r requirements.txt`**。

**F2. 脚本第二行没有 `conda activate drugclip` → Uni-Core 装错环境/找不到 torch。**
`conda env create` 只创建环境（env 名 `drugclip`），不会激活它。第二行 `pip install` 会落在当前 shell 的 base 或系统 pip 上。本机连 conda 都没有，`pip` 是系统 pip3（python 3.12）；即使装了 conda 不激活，base 里也没有 torch。Uni-Core 的 `setup.py` **第一行就是 `import torch`**（已抓取 Uni-Core main 分支源码确认），没有 torch 时 `pip install --no-build-isolation` 立即 `ModuleNotFoundError`。
→ 必须加 `source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate drugclip` 再装 Uni-Core。

**F3. 本机未安装 conda → 脚本第 0 步就无法执行。** 见第 3 节安装建议。

### 1.2 镜像源问题（国内网络下是硬门槛）

`environment.yaml` 声明了 `pytorch / nvidia / conda-forge / defaults` 四个通道。没有任何 `.condarc` 镜像配置时，全部走 anaconda.org —— 其中 **nvidia 通道在国内基本不可用/极慢**，pytorch 通道也常超时。
Dockerfile 里其实已经写好了镜像方案（TUNA 的 `channel_alias`/`custom_channels` + 南科大 nvidia 镜像 + 清华 pip 源），本地复刻时应照抄到 `~/.condarc` 和 pip 配置，否则环境创建大概率在下载阶段卡死。见第 3 节给出的 `.condarc` 建议。

### 1.3 `pytorch-cuda=11.8` vs `12.1` 的选择

- **两条安装路径不一致**：`environment.yaml` 用 `pytorch-cuda=11.8`，Dockerfile 用 `pytorch-cuda==12.1`。
- 已直接查验 pytorch 通道 linux-64 repodata：`pytorch 2.1.2` 的 py3.9 构建同时存在 `cuda11.8_cudnn8.7.0` 与 `cuda12.1_cudnn8.9.2` 两个版本，`torchvision 0.16.2` 也有 `py39_cu118` / `py39_cu121`，`pytorch-cuda` 11.8 与 12.1 包都在。**两个选择在 RTX 4090 (sm_89) 上都可用**（torch 2.1.x 预编译包内置 sm_89 支持）；驱动 610.43 向后兼容 11.x/12.x runtime，无问题。
- **结论：若保留 `faiss-gpu=1.8.0`，建议与 Dockerfile 一致用 `pytorch-cuda=12.1`**，理由见下一条。

### 1.4 `faiss-gpu=1.8.0` 与 pytorch 2.1.2 的匹配（重点）

- 已查验 pytorch 通道 repodata：**`faiss-gpu 1.8.0` 的 py3.9 构建只有两个 CUDA 变体：`cuda11.4.4` 和 `cuda12.1.1`，没有 `cuda11.8` 构建**（`cuda11.8` 只存在于另一个包 `faiss-gpu-raft`，不适用）。`libfaiss 1.8.0` 的 cuda11.4.4 版依赖 `cuda-cudart >=11.4,<12`。
- 因此 `pytorch-cuda=11.8 + faiss-gpu=1.8.0` 时，conda 求解器只能选 **cuda11.4.4 版 faiss**，与 torch 的 cuda11.8 不一致（能解、能跑，但 faiss 实际是 11.4 运行时）；若把 `pytorch-cuda` 改为 12.1，则 faiss 选 cuda12.1.1 与 torch 完全一致，更自洽。
- `faiss-gpu 1.8.0` 的依赖里**并不依赖 pytorch**（只有 libfaiss + numpy<2 + python + libgcc 等），所以它与 pytorch 2.1.2 不存在版本级冲突，只需要 CUDA 变体对齐。
- **但更重要的发现：整个仓库没有任何 `import faiss`**（对全部 `*.py` 做了 grep，含 unimol/、utils/）。benchmark 路径（test.sh/benchmark.sh → `unimol/test.py` → `drugclip.py` 的 test_dude/test_pcba）根本不碰 faiss；`nanopq` 同样无任何引用。`faiss-gpu` 是从 Uni-Mol 环境照搬的多余依赖。
- **建议**：直接删掉 environment.yaml 里的 `faiss-gpu=1.8.0`，消除求解器冲突面（libfaiss 的 `mkl 2023.*` / `cuda-cudart` 约束与四通道混合求解偶尔会冲突）；需要时再补。若坚持保留，用 `pytorch-cuda=12.1`。

### 1.5 python 3.9 兼容性

- pytorch 2.1.2 官方支持 3.8–3.11，py3.9 完全 OK；conda-forge 在 2026 年仍维护 3.9（3.9.21）。
- requirements.txt 里的锁定版本在 py3.9 下均有 cp39 wheel：pandas 1.5.3、scikit-learn 1.2.2、scipy 1.10.1、rdkit 2023.9.5、biotite 0.37.0、biopython 1.81、transformers 4.36.0、lightning 2.2.0、hydra 1.3.2 等，全部可 pip 安装，无需系统编译器。
- 已知坑已在 yaml 里规避：Dockerfile 注释记录了"conda pytorch 会把 typing-extensions 降到 4.14.1，导致 Uni-Core 依赖的 wandb→pydantic 2.x 解析失败"；`environment.yaml` 的 pip 段已显式写 `typing-extensions>=4.15.0`，OK。
- 注意：requirements.txt 顶部注释要求"非 Docker 时取消注释 torch 三行"——但在 conda env 方案里 torch 已由 conda 提供，**不要**取消注释（会 pip 覆盖成别的 CUDA 构建）。
- 一个与 3.9 无关但会踩的坑：2025 年后的 Miniconda 安装器/新 conda 会带 `conda-tos` 插件（Anaconda ToS 弹窗），Dockerfile 的做法是**锁死 py39 的旧安装器** `Miniconda3-py39_24.5.0-0`，本地安装建议照做（见第 3 节）。

### 1.6 Uni-Core `pip install --no-build-isolation` 细节

- `setup.py` 顶层 `import torch`（源码已确认）→ 所以 `--no-build-isolation` 是**必须的**（否则隔离环境里没有 torch，构建直接挂）；前提是**已激活的 env 里先有 torch**（对应 F2）。
- Uni-Core 默认 `DISABLE_CUDA_EXTENSION = True`，是**纯 Python 包**，不需要 nvcc、不需要编译 CUDA 扩展；本项目代码也没有引用 `unicore_fused_*`（已 grep unimol/、utils/ 确认）。所以系统里装的 nvcc 13.3 与 conda torch 的 CUDA 11.8/12.1 不匹配也无所谓。
- 若有人手贱加 `--enable-cuda-ext`：Uni-Core 的 `-gencode` 只编 sm_70/80/90，**不含 sm_89（4090）**，且会校验 nvcc 版本与 torch 构建 CUDA 必须一致（13.3 ≠ 11.8/12.1 直接 RuntimeError）——务必保持默认关闭。
- 装 Uni-Core 时 pip 会解析 install_requires（numpy/lmdb/tqdm/torch>=2.0/ml_collections/scipy/tensorboardX/tokenizers/wandb），torch 已由 conda 满足，其余从 pip 拉，正常。

### 1.7 其它注意事项

- `conda env create` 半途失败（F1）后 env 是残缺的，重跑前必须 `conda env remove -n drugclip -y`（或 create 加 `--force`），否则报 prefix 已存在。
- Ubuntu 24.04 系统 gcc 13 不需要动（全 wheel 安装）；Uni-Core 纯 python 构建用 conda env 自带 setuptools/wheel 即可。
- 修正后的最小流程见第 3 节末尾。

---

## 2. `test.sh` vs `benchmark.sh` 差异对比

| 项 | `test.sh` | `benchmark.sh` | 影响 |
|---|---|---|---|
| 权重来源 | `./data/benchmark_weights/90.pt` | `./output/pretrain/2026-07-24_03-16-10/checkpoint_best.pt` | 见下 |
| `use_folds` | `True` | `False` | **死参数，无影响**（见下） |
| `TASK` | `PCBA`（注释提示 DUDE） | `PCBA` | 相同 |
| `batch_size` | 64 | 12 | 无影响（仅显存占用） |
| `results_path` | `./test` | `output/test_$TASK` | 无影响 |
| `CUDA_VISIBLE_DEVICES` | `"1"` | `"0"` | **test.sh 在本机必挂**（见下） |
| 其余参数 | 相同 | 相同 | — |

### 2.1 权重来源

- **`test.sh` 的 `./data/benchmark_weights/90.pt` 在官方数据集里不存在。** 已查 HuggingFace `THU-ATOM/DrugCLIP_data`（README 里的 `bgao95/DrugCLIP_data` 会重定向到此）文件清单，`benchmark_weights/` 下实际是：
  - `dude_ecfp_30/60/90.pt`、`dude_identity_0/30/60/90.pt`、`dude_scaffold.pt`、`litpcba_identity_90.pt`
  
  没有裸的 `90.pt`。按 TASK 选择：**PCBA → `litpcba_identity_90.pt`；DUDE → `dude_identity_90.pt`**（或 ecfp/scaffold 变体）。不改路径的话 `checkpoint_utils.load_checkpoint_to_cpu` 直接 FileNotFoundError。
- **`benchmark.sh` 的权重是提交者机器上的预训练产物**（2026-07-24 03:16 那次 run，见 commit `87b59f1` "add: benchmark"），新 clone 的仓库里没有 `./output/pretrain/...`。它是给"刚跑完自己 pretrain.sh"的人用的模板，直接跑必然找不到文件。
- 补充：README 说权重在 `model_weights.zip` 里解压到 `./data`，但该 zip 装的是 `6_folds/`、`8_folds/` 集成权重和 `unimol/` 预训练权重，与 `benchmark_weights/` 是**两回事**；`benchmark_weights/` 直接散在 HF repo 里（`data/benchmark_weights/*.pt` 逐个下载即可）。

### 2.2 `use_folds` 是死代码 —— 两脚本的 True/False 差异对结果零影响

`unimol/tasks/drugclip.py` 中：

- `test_dude()`（约 L1296）：循环体内**硬编码 `use_folds = False`**（L1323）覆盖了入参；
- `test_pcba()`（约 L1016）：函数第一行**硬编码 `use_folds = False`**（L1018）；
- `unimol/test.py` 里 `--use-folds` 还是 `type=str`（L74），即使传 `False` 字符串也是 truthy —— 双重 bug。

即：**CLI 传 `--use-folds True/False` 都不影响执行路径，永远走单模型 `test_*_target`；6 折集成函数 `test_dude_target_ensemble` / `test_pcba_target_ensemble`（依赖 `./data/model_weights/6_folds/fold_{0..5}.pt`）永远不会被这两个脚本触发。**
对复现的意义：当前脚本只能复现单权重（identity-90 等）的 AUC/BEDROC/EF 结果；论文中的 6-fold 集成结果需要改代码（删掉两处硬编码、把 `--use-folds` 改 `type=bool`/`store_true`）才能复现。

### 2.3 TASK 与数据目录

- 两脚本默认都是 `TASK="PCBA"`。`drugclip.py` 里数据路径**硬编码**：PCBA → `./data/lit_pcba/`（L1019），DUDE → `./data/DUD-E/`（L1065/L1168）。
- HF 压缩包名是 `LIT-PCBA.zip` 与 `DUD-E.zip`。`DUD-E` 与代码一致；`LIT-PCBA`（大写连字符）与 `lit_pcba` **不一致**，解压后需确认顶层目录名，必要时 `mv ./data/LIT-PCBA ./data/lit_pcba`，否则 `os.listdir("./data/lit_pcba/")` 抛 FileNotFoundError。
- 两个脚本命令行里都出现未定义的 `$data_path`（test.sh L9、benchmark.sh 没有），展开为空字符串——无害，因为测试数据路径全部硬编码在 drugclip.py 里；`"./dict"` 是真正的位置参数。

### 2.4 GPU 编号

- **本机只有 GPU 0**（nvidia-smi 实测单卡 4090）。`test.sh` 的 `CUDA_VISIBLE_DEVICES="1"` 会把唯一一张卡藏掉 → `torch.cuda.is_available()` 为 False → test.py 走 CPU 分支，而 `drugclip.py` 内 `unicore.utils.move_to_cuda` 无 CUDA 时不再搬数据，fp16 CPU 推理会极慢甚至某些 op 报错。**必须改成 `"0"`**（benchmark.sh 已经是 "0"）。
- 附带副作用：`test_dude` 结束会在 CWD 写 `Holo_RealPocket_GenPack.csv`。

---

## 3. 本机 conda 检查与安装建议

**检查结果：未安装任何 conda 系工具**（`which conda mamba micromamba` 全空）。需要全新安装。

### 3.1 推荐安装（与 Dockerfile 同款，规避 ToS 弹窗）

```bash
cd ~/Downloads
wget https://repo.anaconda.com/miniconda/Miniconda3-py39_24.5.0-0-Linux-x86_64.sh
bash Miniconda3-py39_24.5.0-0-Linux-x86_64.sh -b -p ~/miniconda3
~/miniconda3/bin/conda init bash
```

- 选 `Miniconda3-py39_*` 安装器而不是 `Miniconda3-latest`：Dockerfile 注释已记录原因——latest 的 meta-packages 会导致 `python=3.9` 解析问题，且新版安装器/conda 24.x+ 带 `conda-tos` 插件会有 Anaconda ToS 弹窗；py39_24.5.0 安装器没有该插件。
- WSL2 下无需装任何系统 CUDA toolkit：torch 的 conda 包自带 CUDA 运行时，驱动由 Windows 宿主侧提供，610.43 完全兼容 11.8/12.1。

### 3.2 安装后先配镜像（国内网络必需）

在 `~/.condarc` 里写入（照抄 Dockerfile 的配置）：

```yaml
channels:
  - defaults
show_channel_urls: true
channel_alias: https://mirrors.tuna.tsinghua.edu.cn/anaconda
default_channels:
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/pro
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2
custom_channels:
  conda-forge: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  msys2: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  bioconda: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  menpo: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  pytorch: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  simpleitk: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  nvidia: https://mirrors.sustech.edu.cn/anaconda-extra/cloud
```

pip 同理：`pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple`。

### 3.3 修正后的无 Docker 搭建流程（替代 build_conda_env.sh）

```bash
# 0) 修复 environment.yaml：docker/requirements.txt -> requirements.txt；可选删除 faiss-gpu、pytorch-cuda 改 12.1
# 1) 建环境（首次 ~2.7GB cuda 包；失败重跑前先 conda env remove -n drugclip -y）
conda env create -f environment.yaml
# 2) 激活（原脚本缺这一步）
conda activate drugclip
# 3) Uni-Core（--no-build-isolation 必须；保持默认不编 CUDA 扩展）
git clone https://github.com/dptech-corp/Uni-Core.git /tmp/Uni-core
pip install --no-build-isolation /tmp/Uni-core && rm -rf /tmp/Uni-core
# 4) 验证
python -c "import torch, unicore; print(torch.__version__, torch.cuda.is_available())"
```

### 3.4 数据准备与最小可复现路径

- 当前 `./data/` 只有 `targets.zip`；benchmark 需要再补：`DUD-E.zip`、`LIT-PCBA.zip`、`benchmark_weights/dude_identity_90.pt`（DUDE）或 `litpcba_identity_90.pt`（PCBA）；6-fold 集成需要 `model_weights.zip`。
- 最小验证（改 3 处就能跑）：
  1. `environment.yaml` 修 `-r requirements.txt`（并建好环境）；
  2. `test.sh`：`weight_path="./data/benchmark_weights/dude_identity_90.pt"`、`CUDA_VISIBLE_DEVICES="0"`、`TASK="DUDE"`；
  3. 确认 `./data/DUD-E/` 目录名正确（zip 解压后若为 `DUD-E` 则 OK）。
- 预期产物：各 target 的 AUC/BEDROC/EF 均值输出 + `Holo_RealPocket_GenPack.csv`。

---

## 4. 结论摘要（按严重度排序）

| # | 问题 | 位置 | 必现? |
|---|---|---|---|
| 1 | `-r docker/requirements.txt` 文件不存在 → env 创建 pip 阶段失败 | environment.yaml | 必现 |
| 2 | 脚本无 `conda activate drugclip` → Uni-Core 装错环境、`import torch` 失败 | build_conda_env.sh | 必现 |
| 3 | 本机无 conda | 本机 | 必现 |
| 4 | `test.sh` 权重 `90.pt` 不存在（HF 上是 `dude_identity_90.pt` / `litpcba_identity_90.pt` 等） | test.sh | 必现 |
| 5 | `benchmark.sh` 权重指向提交者本机 pretrain 产物，新 clone 无此文件 | benchmark.sh | 必现 |
| 6 | `CUDA_VISIBLE_DEVICES="1"` 但本机仅 1 张 GPU | test.sh | 必现 |
| 7 | `use_folds` 硬编码 False + `type=str` → 两脚本 True/False 均无效，6-fold 集成不可达 | drugclip.py L1018/L1323, test.py L74 | 逻辑 bug |
| 8 | 代码期望 `./data/lit_pcba/`，zip 名为 `LIT-PCBA` → 目录名需核对/改名（PCBA 时） | drugclip.py / 数据准备 | 大概率 |
| 9 | nvidia/pytorch 通道国内直连慢 → 需先配 TUNA/南科大镜像 | ~/.condarc | 环境相关 |
| 10 | faiss-gpu 1.8.0 无 cuda11.8 构建（只有 11.4.4 / 12.1.1）；且全仓无 faiss 引用 → 建议删除或改用 pytorch-cuda=12.1 | environment.yaml | 求解器风险 |

> 备注：以上均为静态分析结论；`faiss-gpu`/`pytorch-cuda` 的可用构建、`benchmark_weights` 文件名均已通过 conda 通道 repodata 与 HF API 实测核实。未执行任何安装或大文件下载。
