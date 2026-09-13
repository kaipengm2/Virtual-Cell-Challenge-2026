# VCC Atlas Transfer

这是比赛历史最高分 **v38 / FullAtlas BalancedCounts** 的独立实现，整理为可以安装、
配置、验证和复现的 CPU 基线。它通过多个公开图谱迁移已测量的扰动效应，再生成整数细胞计数。

历史公榜得分为 **0.1545618019**；2026 年 9 月 10 日 08:34:53（上海时间）的快照排名为
**82 / 839**。这是历史公开验证结果，不代表当前排名或私榜表现。

独立版已在冻结输入下完成全量复现：360,000 个细胞、2,437,685,571 个非零计数值、
稀疏索引、元数据及保存的预测数组，逐元素对照均为零差异。验证记录附在仓库中。

## 从这里开始

在本目录下运行：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
vcc-atlas demo --output runs/demo
pytest -q
```

示例完全使用合成数据，无需下载生物数据、登录账号或调用 API。它会经过完整推理流程，
生成 144 个细胞、24 个基因的预测文件；这个小示例不用于比赛提交或评价预测能力。

## 复现 v38

按照 [数据说明](docs/DATA.md) 准备输入，放在 `data/v38/`，然后运行：

```bash
vcc-atlas verify-inputs --manifest docs/v38-inputs.sha256.json --data-dir data/v38
vcc-atlas predict --config configs/v38.json --data-dir data/v38 --output runs/v38
```

每次使用新的输出目录。完整配置生成 A/B/C 三个背景、300 个目标、每组 400 个细胞，
共 360,000 × 18,533。输出包括原始整数计数、预测均值、覆盖权重、拟合误差、输入和代码
校验和。只有生成完成后才会出现 `run.json`。

五份来源统计文件合计约 395 MB；历史预测 H5AD 约 4.4 GiB。
历史生成耗时约 608 秒，不包含原始数据提取、压缩、官方校验或打分。运行不需要 GPU。
本地复现使用 Python 3.13；具体依赖见 `requirements-tested.txt`。

## 保留了什么

- K562、HCT116、HEK293T、H1 的 2:1:1:2 融合；CD4 三个条件合为权重 0.5 的一类来源。
- 平均 CPM 和汇总计数采用不同的表达单位、不同的效应幅度，分别预测。
- 在统计文件的完整目标面板上中心化，显式处理缺失目标和未测基因。
- 深度相近的对照细胞池化、双矩约束、整数舍入和列总量修复。
- v38 使用的 GENCODE v47 基因边界启动子先验。
- 数据获取与提取脚本、来源校验清单、无损压缩工具、数值对照工具和测试。

“完整来源面板”指每份统计文件保存的全部目标，并非原始图谱测量的所有扰动。
模型主要迁移其他细胞背景中已测量的效应，不应描述成对所有来源都未见过的新扰动的预测。
池化和计数修复可能改变异质性，均值拟合准确不意味着细胞分布或差异表达同样准确。

## 如何打包

```bash
python -m pip install -e '.[compact]'
python scripts/compact_prediction.py runs/v38/prediction.h5ad runs/v38/prediction_compact.h5ad
vcc prep runs/v38/prediction_compact.h5ad -g data/v38/gene_names.csv \
  --perts data/v38/pert_counts.csv -o runs/v38/prediction.vcc
```

无损压缩会改变文件存储方式和行排列，保留各组内部细胞顺序及计数。
完整矩阵压缩约需 39 GB 临时磁盘。内存较小的机器可用 [磁盘映射打包路径](docs/SUBMISSION.md)。
VCC CLI 需单独安装；仓库中的命令不会自动提交结果。

详细说明见 [方法](docs/METHOD.md)、[数据](docs/DATA.md)、
[复现证据](docs/REPRODUCIBILITY.md) 和 [外部来源](THIRD_PARTY.md)。
本仓库代码采用 MIT 许可证，外部数据遵循各自条款。仓库不附带真实细胞矩阵、API key 或账号配置。
