# RSTAR-DeCNN

在图像域融合完整 Stage II RSTAR-STAM 与逐相位 3D DeCNN。默认训练冻结整个 RSTAR-STAM，DeCNN 从目标相门控 FDK 和同次扫描的全投影 FDK 学习同相位重建。原版及 STAM 推理沿用相邻项目的 profile、强度变换和滑窗。DeCNN 按图 1 实现两套独立编码器、四级双路跳连和三维卷积；上采样默认取 nearest，BN 的 eps 和 momentum 分别为 1e-5 和 0.1。[DeCNN 原文](https://doi.org/10.1109/TRPMS.2021.3133510)

本地对照上游版本为 `57d61544a0929d56307eda0e5d0edde8ce245105`。实验记录还保存实际加载的上游 Python 文件摘要。安装可使用 `python -m pip install -e ../RSTAR-STAM -e .`。如果现有 GPU 环境已安装依赖，从本目录运行以下命令即可引用相邻源码。

```powershell
$env:PYTHONPATH = (Resolve-Path ../RSTAR-STAM).Path
python -m rstar_decnn --help
```

新检出时，可将两个分支放在相邻目录。

```powershell
git clone --branch rstar-stam --single-branch https://github.com/godspeedlucip/proj2 RSTAR-STAM
git clone --branch restar-decnn --single-branch https://github.com/godspeedlucip/proj2 RSTAR-DeCNN
Set-Location RSTAR-DeCNN
python -m pip install -e ../RSTAR-STAM -e .
```

## 准备数据并训练

```powershell
python -m rstar_decnn prepare --manifest $pairedSource --prior-mode phase_weighted_fdk --out runs/pig/paired.jsonl
python -m rstar_decnn train --config configs/frozen.yaml --manifest runs/pig/paired.jsonl --split $animalSplit --out runs/pig/frozen-2026
```

- 配对源清单采用上游 JSONL 合同。预处理只新写清单和全投影视图先验，源扫描 HDF5 保留原样。缺少投影文件时，提供原始相位源清单和呼吸周期曲线 `--raw-manifest`、`--cycle-csv`。原实验用 Poisson 噪声时同时提供 `--simulation-seed`。重放会核对逐相门控 FDK。
- 已有同扫描全投影先验时，`--prior-index` 接受包含 `case_id`、`path`、`sha256` 的 JSONL。HDF5 需要 `prior[Z,Y,X]` 和 `metadata_json`，其中记录病例、动物、源文件 SHA-256、先验模式、混合投影摘要、校准和坐标网格。两路使用同一现有先验的专门对照用 `--shared-prior` 和配置 `decnn_prior: shared`；同相输入对照设 `decnn_prior: self`。
- HDF5 采用 `cbct[T,Z,Y,X]`、`prior[Z,Y,X]`、`valid_mask[T,Z,Y,X]`，训练和评价另用 `target[T,Z,Y,X]`。如果配对数组标签顺序不同，可在 `target_metadata_json` 或 `valid_mask_metadata_json` 中记录各自的 `phase_labels`；读取时按 gated 标签对齐。独立记录的仿射、间距或强度合同发生冲突时会报病例级错误。推理可读取没有 target 的文件。
- `--split` 是含 `train`、`val`、`test` 的动物 ID JSON。训练前填写冻结配方中的 `backbone_checkpoint`，登记选定的 DeCNN 线性变换来源，并按正式预算调整步数。仓库配置的训练步数、学习率和暂定强度范围是启动设置。固定种子 2026 做预实验；正式配对配置可分别改为 2026、2027、2028。
- 四条配方分别为 `configs/frozen.yaml`、`configs/scratch_stage2.yaml`、`configs/tetris.yaml`、`configs/warmup_unfreeze.yaml`。冻结配方用验证动物等权的全体积、有效域 DeCNN 归一化 MSE 选模。联合阶段使用原猪心任务损失和 DeCNN 辅助 MSE，按有效体素数汇总并保持主干的完整相位序列。AMP 下任务损失使用 FP32。
- `summary.json` 记录历史最佳检查点路径。继续训练时使用原运行目录、配置、样本清单、动物划分和 `--resume $runDir/last.pt`。检查点包含两支权重和推理所需状态，推理机器可独立载入。用 `--stop-after N` 在第 N 次有效更新后暂停，便于验证转段和续训。
- 从零与 Tetris 联合阶段会更新随机初始化的 SAM。使用 PyTorch 2.3.1 时，这两条路线的严格逐张量续训验收使用 CPU，因为 CUDA 的 SAM 一维线性插值反向算子在该版本没有确定性实现。CUDA 训练配方使用 `deterministic: false`。

## 导出、标定与评价

```powershell
python -m rstar_decnn predict --checkpoint $best --manifest runs/pig/paired.jsonl --split $animalSplit --split-name val --cache --rstar $matchedRstar --baseline-design $baselineDesign --out runs/pig/val
python -m rstar_decnn predict --checkpoint $best --manifest runs/pig/paired.jsonl --split $animalSplit --split-name test --cache --rstar $matchedRstar --baseline-design $baselineDesign --out runs/pig/test
python -m rstar_decnn calibrate --manifest runs/pig/paired.jsonl --split $animalSplit --cache runs/pig/val/manifest.jsonl --annotations $lvAnnotations --extractor $lvExtractor --motion-tolerance-ml 0 --out runs/pig/validation-lock.json
python -m rstar_decnn evaluate --manifest runs/pig/paired.jsonl --split $animalSplit --cache runs/pig/test/manifest.jsonl --annotations $lvAnnotations --extractor $lvExtractor --lock runs/pig/validation-lock.json --out runs/pig/evaluation
```

- 正式基线设计 JSON 登记 `training_start`、`update_budget`、`animal_split`、`matched_stam_training_start`、`matched_stam_update_budget`。缓存记录三支物理域输出、模型和输入摘要、病例与相位标签、几何、校准、滑窗和运行耗时。在线预测省略 `--cache`，用 `--alpha` 传入需要复现的系数。0 和 1 分别旁路到 STAM 和 DeCNN。可用 `--mode full` 或分别设置两支的滑窗和重叠像素。DeCNN 滑窗和步长三轴应与 16 对齐。
- 标注使用上游 LV JSONL 格式，字段包括 `case_id`、`source_sha256`、真实 `phase_label`、HDF5 的 `path`、文件 `sha256` 和二值掩膜 `dataset`。ED 和 ES 通过 `role` 声明。提取器接收物理域 `[1,1,Z,Y,X]`，应返回同网格二值轮廓。验证锁固定输入和模型摘要、提取器、容许量、评价代码及系数。若运动证据不足，输出探索性边界标定以及待验证状态。
- `calibrate` 和 `evaluate` 可以按种子重复传入 `--cache`。评价输出 `comparison.csv`、`summary.json` 和逐扫描 `cases.jsonl`。已锁定的组合与两个原主干、DeCNN 单支共五行比较，按动物做配对和重采样。报告保留有效动物数、额外轮廓提取失败、全图 MAE、固定窗 SSIM 和一例基于参考运动的剖面。

## 验收入口

```powershell
python -m unittest discover -s tests -t . -v
python -m tests.engineering_smoke --out runs/engineering-recheck
python -m tests.resource_benchmark --checkpoint $engineeringBest --out runs/resource-recheck.json
```

工程脚本用相邻项目的体模检查点和合成输入验证流程。资源脚本使用 11 相位合成训练块，以及 10 或 11 相位的奇数空间尺寸推理。实际验收、同次猪心投影准备和正式实验缺项见[实施验收记录](docs/implementation-validation.md)。
