# DrugCLIP (Drug-The-Whole-Genome) 复现静态分析报告

> 生成方式：纯静态分析（代码阅读 + git 历史 + HuggingFace API 元数据），未下载大文件、未安装环境。
> 仓库根目录：`/home/lynchpin/Projects/Drug-The-Whole-Genome`（HEAD = 87b59f1 "add: benchmark"）

---

## 1. environment.yaml pip 段引用无效（无 Docker 路线会直接失败）

### 问题

`environment.yaml:16` 存在：

```yaml
pip:
  - -r docker/requirements.txt
```

但仓库中 **不存在 `docker/` 目录**。git 历史显示：

- `3fe8c92 "mod: 环境安装"`：`docker/requirements.txt → requirements.txt`、`docker/Dockerfile → Dockerfile`、`conda/environment.yaml → environment.yaml`，即文件被移到了仓库根目录，**但 `environment.yaml` 内的引用路径没有跟着改**。

因此无 Docker 路线（`build_conda_env.sh` → `conda env create -f environment.yaml`）会直接报错：

```
CondaValueError: Error from pip: -r docker/requirements.txt: cannot open file 'docker/requirements.txt': No such file or directory
```

### 修复建议（任选其一）

1. 修改 `environment.yaml:16`（推荐，改动最小）：
   ```yaml
   pip:
     - -r requirements.txt
   ```
2. 或保持引用并创建 `docker/requirements.txt` 符号链接指向根目录 `requirements.txt`（不推荐，容易再漂移）。

### 其他环境相关提示

- `environment.yaml` 用 `pytorch-cuda=11.8`，而 `Dockerfile:139` 用 `pytorch-cuda==12.1`，两条路线 CUDA 版本不一致，`faiss-gpu=1.8.0` 需与之一致。
- `requirements.txt:58` 注释中的 unicore wheel URL（`unicore-0.0.1+cu117torch1.13.1-cp39`）是 **torch 1.13.1/cu117** 构建，与本仓库的 torch 2.1.2 不兼容，切勿启用（见 §3 依赖分析）。

---

## 2. Benchmark 所需数据与权重（test.py / test.sh / benchmark.sh）

### 2.1 代码实际读取的路径（unimol/tasks/drugclip.py，写死）

| 任务 | 分子数据 | 口袋数据 | 备注 |
|------|---------|---------|------|
| DUDE | `./data/DUD-E/{target}/mols.lmdb`（test_dude_target，L1065） | `./data/DUD-E/{target}/pocket.lmdb`（L1108） | 目标目录来自 `os.listdir("./data/DUD-E/")`；缺文件则 `return None` 跳过 |
| PCBA | `./data/lit_pcba/{target}/mols.lmdb`（test_pcba_target，L931） | `./data/lit_pcba/{target}/pockets.lmdb`（L971） | 目录名**小写** `lit_pcba`；缺文件不跳过（None 会解包崩溃） |

- 词典：`./dict/dict_mol.txt`、`./dict/dict_pkt.txt`（由 test.sh 的 `"./dict"` 位置参数传入，`drugclip.py:175-176` 加载；两文件已在仓库中）。
- LMDB schema（`AffinityDataset`/`AffinityMolDataset`/`AffinityPocketDataset`）：mols.lmdb 每记录含 `atoms`、`coordinates`、`smi`、`label`；pocket.lmdb 含 `pocket_atoms`、`pocket_coordinates`、`pocket`。key 为 `"0","1",...` 序号（`lmdb_dataset.py:48`）。
- 注意：`test_dude`/`test_pcba` 内部都把 `use_folds` 硬编码为 `False`（drugclip.py:1018、1323），所以 `--use-folds` 参数实际被忽略，6 折集成权重在 benchmark 中**不会**被加载（仅 retrieval 用）。
- `test.py` 的 `--results-path` 未被使用；DUDE 结果以 `Holo_RealPocket_GenPack.csv` 写到当前目录（drugclip.py:1364）。

### 2.2 权重 `./data/benchmark_weights/90.pt` 从哪来 —— 路径是错的

`test.sh:4` 写死 `weight_path="./data/benchmark_weights/90.pt"`，仓库内、HuggingFace 上均**不存在**名为 `90.pt` 的文件。HuggingFace 数据集 `bgao95/DrugCLIP_data`（现重定向为 `THU-ATOM/DrugCLIP_data`）中 `benchmark_weights/` 目录实际包含：

```
benchmark_weights/dude_ecfp_30.pt       ECFP4 相似度 >0.3 去冗余
benchmark_weights/dude_ecfp_60.pt       ECFP4 相似度 >0.6 去冗余
benchmark_weights/dude_ecfp_90.pt       ECFP4 相似度 >0.9 去冗余
benchmark_weights/dude_scaffold.pt      骨架去冗余
benchmark_weights/dude_identity_0.pt    HMMER 序列一致性去冗余
benchmark_weights/dude_identity_30.pt   MMseqs2 identity >0.3
benchmark_weights/dude_identity_60.pt   MMseqs2 identity >0.6
benchmark_weights/dude_identity_90.pt   MMseqs2 identity >0.9
benchmark_weights/litpcba_identity_90.pt  LIT-PCBA MMseqs2 identity >0.9
```

`90.pt` 应理解为“90% 相似度分裂的权重”。**修复建议**：test.sh 里按 TASK 改为显式路径，例如：

```bash
TASK="PCBA"          # 或 DUDE
weight_path="./data/benchmark_weights/litpcba_identity_90.pt"   # PCBA
# weight_path="./data/benchmark_weights/dude_ecfp_90.pt"        # DUDE（或 dude_identity_90.pt）
```

这些权重含在 HF 的 `benchmark_weights.zip` 中（README 只提到 DUD-E.zip / LIT-PCBA.zip，需自行下载 benchmark_weights.zip，或按文件逐个下载）。

### 2.3 benchmark.sh 的权重

`benchmark.sh:6` 用 `weight_path="./output/pretrain/2026-07-24_03-16-10/checkpoint_best.pt"`——即**自训练** checkpoint（pretrain.sh 产出），路径含固定时间戳，新环境必然不存在。建议改为：

```bash
weight_path="$(ls -dt output/pretrain/*/ | head -1)checkpoint_best.pt"
```

### 2.4 HuggingFace 下载后目录如何组织

数据集：`https://huggingface.co/datasets/bgao95/DrugCLIP_data`（作者已更名 THU-ATOM，原链接 302 重定向，`downloads=1188`，总存储约 83.9 GB）。按代码写死的相对路径，解压后应形成：

```
./data/
├── DUD-E/                        # 来自 DUD-E.zip（解压后目录名需为 DUD-E）
│   └── {target}/                 # 如 0024, ABL1, ...（102 个靶点）
│       ├── mols.lmdb
│       └── pocket.lmdb
├── lit_pcba/                     # 来自 LIT-PCBA.zip —— 注意代码用小写 lit_pcba
│   └── {target}/                 # 15 个靶点
│       ├── mols.lmdb
│       └── pockets.lmdb          # 注意：LIT-PCBA 用复数 pockets.lmdb，DUDE 用单数 pocket.lmdb
├── benchmark_weights/            # 来自 benchmark_weights.zip（README 未列，需单独下）
│   ├── dude_ecfp_30.pt / 60 / 90
│   ├── dude_identity_0/30/60/90.pt
│   ├── dude_scaffold.pt
│   └── litpcba_identity_90.pt
├── model_weights/                # 来自 model_weights.zip（retrieval 用）
│   ├── 6_folds/fold_{0..5}.pt
│   ├── 8_folds/fold_{0..7}.pt
│   └── unimol/{mol_pre_no_h_220816.pt, pocket_pre_220816.pt}   # pretrain.sh 初始化用
├── encoded_mol_embs/             # 来自 encoded_mol_embs.zip（retrieval use_cache=True 用）
│   └── 6_folds/fold{0..5}.pkl    # 每 pkl: name_list=[hitid,SMILES], embedding_list(128维)
└── targets/                      # 来自 targets.zip（README 的虚拟筛选口袋，retrieval_all.sh 用）
    └── {target}/PDB/pocket.lmdb
```

注意事项：
- **大小写敏感**：代码访问 `./data/lit_pcba/`（小写）；若 zip 解出来是 `LIT-PCBA/`，Linux 下必须重命名（README 的“put inside ./data”表述不够精确）。
- 文件命名差异：DUDE 是 `pocket.lmdb`，LIT-PCBA 是 `pockets.lmdb`，两个任务不能混用。
- `WetLab_PDBs_and_LMDBs/`（含 5HT2AR/NET/TRIP12 的 pocket.lmdb）在仓库是文件夹而非 zip，供 retrieval.sh / wet-lab 用；benchmark 不需要。

---

## 3. import 与 requirements.txt 覆盖情况（缺失依赖清单）

方法：扫描全部 `*.py` 的一级 import，与 `requirements.txt` + `environment.yaml` pip 段 + conda 段比对（排除 stdlib 与仓库内部模块 `unicore` 相关、`unimol.*`、`docking_utils`、`screening_utils`）。

### 3.1 缺失依赖（必须补，按严重程度排序）

| 包 | 在哪被 import | 影响 | 建议 |
|----|--------------|------|------|
| `zstandard` | `unimol/data/lmdb_dataset.py:61`（**模块顶层 import**） | **致命**：`unimol/data/__init__.py` 无条件导入 lmdb_dataset → `import unimol.data` 即 `ModuleNotFoundError`，任何入口（test/retrieval/encode/pretrain）都会挂 | 加入 requirements.txt（`pip install zstandard`，纯 CPU 包） |
| `unicore` | 全部入口（`unicore.options/checkpoint_utils/tasks`） | **致命**：不在 requirements.txt；其中注释掉的 wheel 是 cu117/torch1.13.1 构建，与 torch 2.1.2 不兼容 | 保持 `build_conda_env.sh:2` 的 `git clone …/Uni-Core && pip install --no-build-isolation`（与 Dockerfile:165-168 一致）；把 wheel 注释行删掉避免误用 |
| `selfies` | `unimol/utils/decode_utils.py:10` | 中等：仅分子生成/解码工具用到；`unimol/utils/__init__.py` 为空，**不**影响 benchmark/retrieval | 用相关脚本时 `pip install selfies`；建议补入 requirements.txt |
| `torch_scatter` | `utils/screening_utils.py:3`（`scatter_max`） | 中等：仅分块筛选 `utils/screening_chunk.py` 用；且需与 torch 版本匹配的预编译包 | 用时安装对应 cu118 版 `torch-scatter`；建议补入 requirements.txt 并注释说明 |
| `numpy` | 几乎所有文件 | 低：未显式列出（由 conda pytorch 传递安装），单独 pip 重建环境时可能缺失 | 在 requirements.txt 显式加 `numpy`（Dockerfile:152 已显式装了，说明作者自己补过） |

### 3.2 已覆盖且无问题的依赖

`torch/torchvision/torchaudio`（conda）、`faiss-gpu`（conda，仓库代码实际未调用）、`lmdb`、`io_path`（即 iopath）、`ml_collections`、`tensorboardX`、`tqdm`、`tokenizers`、`typing-extensions`（environment.yaml pip 段）；`rdkit`、`biopython`（Bio）、`scikit-learn`（sklearn）、`pandas`、`scipy`、`h5py`（requirements.txt 已注释说明是补加的）、`IPython`（tasks/drugclip.py:4，仅导入 `embed` 未调用）、`matplotlib`、`einops`、`transformers` 等。

### 3.3 声明了但仓库代码未使用（无害，可忽略）

`lightning/torchmetrics`、`hydra-core/colorlog/optuna-sweeper`、`wandb`、`nanopq`、`biotite`、`biopandas`、`ray`、`jsonlines`、`python-dotenv`、`rich`、`pytest`、`sh`、`ipython`、`jupyterlab`、`pre-commit/black/isort/flake8/nbstripout`——均来自上游 Uni-Mol 模板的 requirements，与本次复现流程无关。

### 3.4 结论

修复顺序建议：① `environment.yaml` 的 `-r` 路径；② requirements.txt 补 `zstandard`（否则环境建完一步都跑不了）；③ 修 test.sh 的 `90.pt` → HF 实际权重名；④ 按 §2.4 组织数据目录（重点是大写 `DUD-E` / 小写 `lit_pcba` 与单复数 pocket 文件名的差异）。

---

## 附录：关键代码位置索引

- `environment.yaml:16` —— 失效的 `-r docker/requirements.txt`
- `requirements.txt:7-10` —— torch 需 conda 安装的说明；`:58` —— 不兼容的 unicore wheel 注释
- `test.sh:4` —— `./data/benchmark_weights/90.pt`（不存在的文件名）
- `benchmark.sh:6` —— 固定时间戳的自训练 checkpoint 路径
- `unimol/tasks/drugclip.py:1018,1323` —— `use_folds` 硬编码 False
- `unimol/tasks/drugclip.py:798,931,971,1065,1108,1168,1226` —— benchmark 数据路径
- `unimol/tasks/drugclip.py:812,1183,1474,1616,1688-1700` —— 6/8 折权重与缓存路径
- `unimol/data/lmdb_dataset.py:61` —— `import zstandard`（硬依赖）
- `utils/screening_utils.py:3` —— `torch_scatter`；`unimol/utils/decode_utils.py:10` —— `selfies`
