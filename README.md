# VCC v38

历史公榜 **0.1545618019**。固定参数的五来源迁移模型，CPU 运行，无训练。
同输入下，360,000 个细胞的预测计数与 v38 逐元素一致。

```bash
# Python 3.13；另需安装官方 vcc CLI
pip install -r requirements.txt
python predict.py --data-dir data --output prediction.h5ad
python compact.py prediction.h5ad prediction_compact.h5ad
python pack.py prediction_compact.h5ad --data-dir data --output prediction.vcc
```

`data/` 放以下文件（不随代码分发）：

```text
gene_names.csv                 # 官方基因顺序
pert_counts.csv                # 官方目标，每组 400 个细胞
context_A.h5ad                 # 官方对照
context_B.h5ad
context_C.h5ad
K562_GWPS_CPM_full_statistics.npz
HCT116_full_statistics.npz
HEK293T_full_statistics.npz
H1_2025_full_statistics.npz
CD4_DE_statistics.npz
official_pairs.csv
```

没有来源统计文件时，可从公开原始数据生成：

```bash
pip install slafdb
python prepare.py --data-dir data --raw-dir raw
# 单独准备示例：python prepare.py --source hct
```

原始数据约 117 GB，另需扫描 X-Atlas；来源地址和校验和见 `sources.json`。
来源面板不能缩减。X-Atlas 远程数据更新可能改变结果。
预测约 10 分钟；压缩、打包分别需要约 39 GB、29 GB 临时磁盘。
`pack.py` 使用已安装的 vcc-cli 0.2.0，并执行全部官方检查；其他版本使用 `vcc prep`。
代码 MIT；外部数据遵循各自条款。
