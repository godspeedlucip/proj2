# 3D nnU-Net + TAM 心脏 CT 分割

入口适用于单卡、单通道心脏 CT、互斥类别和 `3d_fullres` PlainConvUNet。两时相消融固定输入 `[phase1, phase5]`。使用 `--context-phases` 可训练三时相稀疏监督模型，逐一预测其余真实时相。令 `L` 为编码器总阶段数，阶段索引从零开始。

| 预设 | TAM 插入阶段 | 六阶段时的索引 |
| --- | --- | --- |
| C1 | 无 | 无 |
| C3 | `L−1` | 5 |
| C4 | `L−2, L−1` | 4, 5 |
| C5 | `L−3, L−2, L−1` | 3, 4, 5 |

每个插入位置的注意力有 8 个头，默认嵌入宽度等于阶段输出通道数。如需覆盖，可在共有 plans 的 `3d_fullres.architecture.arch_kwargs.tam.embed_dim_by_stage` 中加入字符串阶段索引与宽度，例如 `{"3": 64}`。宽度需为 8 的正整数倍。

## 准备一折数据

病例清单是 JSON 数组。相对路径从清单目录解析。同一患者多次采集应使用一致的 `patient_id`。预测清单可省略两期的 `label` 字段。

```json
[
  {
    "case_id": "acq_001",
    "patient_id": "patient_001",
    "phase1": {"image": "ct/acq_001_p1.nii.gz", "label": "mask/acq_001_p1.nii.gz"},
    "phase5": {"image": "ct/acq_001_p5.nii.gz", "label": "mask/acq_001_p5.nii.gz"}
  }
]
```

划分 JSON 的每一项是一折，`train`、`val` 和 `test` 各含病例 ID 数组。每折三个集合都需有病例，并完整覆盖清单。输入 `dataset.json` 使用 nnU-Net 的 CT 通道、标签映射和输出文件扩展名，例如 `channel_names: {"0": "CT"}`、`labels: {"background": 0, "heart": 1}`、`file_ending: ".nii.gz"`。

下面的 PowerShell 示例使用原生 nnU-Net planner，仅将选定折的训练患者注册为参考数据。为不同的规划折选用独立的参考 dataset ID。已有训练患者专属的参考 plans 时，可直接运行 `prepare`。将 `nnUNet_raw` 和 `nnUNet_preprocessed` 设为先前配置的 nnU-Net 路径，并在仓库根目录运行。

```powershell
python -m nnunetv2.run.paired_tam reference-dataset manifest.json splits.json dataset.json "$env:nnUNet_raw/Dataset990_TAMReference" --fold 0
nnUNetv2_extract_fingerprint -d 990
nnUNetv2_plan_experiment -d 990
python -m nnunetv2.run.paired_tam prepare manifest.json splits.json "$env:nnUNet_preprocessed/Dataset990_TAMReference/nnUNetPlans.json" dataset.json paired/Dataset990_TAMReference/nnUNetPlans/fold_0 --fold 0
```

每个 dataset、plans 版本和折使用独立的空输出目录。成对预处理冻结训练患者两期前景的 CT 统计，并保存原始几何、患者划分及源文件指纹。对输入文件的修改需要重新准备一折。

### 十期 CT 仅标注 phase1 和 phase5

为每例在同一清单中补齐 `phase2`、`phase3`、`phase4`、`phase6` 到 `phase10`。每个额外时相写成 `"phase7": {"image": "ct/acq_001_p7.nii.gz"}` 这样的图像条目。标注路径仍写在 `phase1` 和 `phase5`。独立预测清单可省略所有 `label` 项。

```powershell
$plans = "$env:nnUNet_preprocessed/Dataset990_TAMReference/nnUNetPlans.json"
$prepared = "paired/Dataset990_TAMReference/nnUNetPlans/sparse_fold_0"
python -m nnunetv2.run.paired_tam prepare manifest.json splits.json $plans dataset.json $prepared --fold 0 --context-phases phase2 phase3 phase4 phase6 phase7 phase8 phase9 phase10
python -m nnunetv2.run.paired_tam profile $prepared profiles/sparse_c5.json --preset C5 --device cuda
python -m nnunetv2.run.paired_tam train $prepared runs/sparse_c5 --preset C5 --fold 0 --seed 12345 --device cuda
python -m nnunetv2.run.paired_tam predict runs/sparse_c5 image_only_manifest.json predictions/sparse_c5 --fold 0 --device cuda
```

预处理将配置的图像和两份标注缓存到共同网格。每个训练 patch 使用 `[phase1, phase5, 目标期]`，目标期从配置的额外时相随机抽取。每次前向输出三份分割，两期监督损失等权平均。推理遍历额外时相，分别保留各组第三个输出，并从第一个组合取两期参考输出。以上命令每例导出十份原始网格分割。评估仅对两期人工掩膜计算 Dice 和 HD95。

这个模式采用论文表 2 的稀疏监督与表 1 的 C5 插入位置。把三维心脏 CT 编码阶段接入 nnU-Net，以及用三期一组遍历十期，是本项目的迁移设计。[TAM 论文](https://doi.org/10.1016/j.media.2026.103981)

## 预试并锁定训练设置

在同一 CUDA 环境先预试 C5，再用同一份 plans 检查全部四组。预试测量前向、反向和优化器更新，并对一个验证病例运行成对滑窗。JSON 中包含每步用时、显存峰值和实际配置。

```powershell
$prepared = "paired/Dataset990_TAMReference/nnUNetPlans/fold_0"
python -m nnunetv2.run.paired_tam profile $prepared profiles/c5.json --preset C5 --device cuda
python -m nnunetv2.run.paired_tam profile $prepared profiles/all.json --preset all --device cuda
```

若显存不足，先在参考 plans 的副本中减小成对 `batch_size`，再考虑减小 `patch_size`。保留目标间距和主干步长。每轴 patch 必须能被累计步长整除。给新配置设置新的 `plans_name` 与 `data_identifier`，再用新目录重新准备、预试并运行四组。正式比较使用同一折的同一份冻结 plans。

## 运行消融

```powershell
python -m nnunetv2.run.paired_tam ablate $prepared runs/tam_exp_01 --fold 0 --seed 12345 --device cuda
```

该命令为四组复用同一份计划、数据、训练预算与初始空间主干权重。每个 epoch 使用统一的采样种子策略。带种子的训练在单进程中增强，以保持病例采样与空间增强顺序一致。重复实验使用独立的运行目录和指定种子。中断后可用同一命令加 `--resume`，配方中锁定的参数需要一致。默认训练预算为 1000 epoch，每轮 250 个训练步和 50 个验证步。

`fold_0/ablation_summary.json` 包含四组测试汇总、参数量和共有设置。报告按患者、时相、结构配对计算 C3、C4、C5 相对于 C1 的差值，同时给出 C4 相对 C3 和 C5 相对 C4 的差值。差值为后者配置的指标减去参照配置的指标。每组还导出两期掩膜和 `test/summary.json`，其中包含各病例明细、最低 Dice 和最高 HD95 病例。先把同一患者多次采集取均值，再按患者汇总。

HD95 用两方向表面距离各自的第 95 百分位中的较大值，以毫米表示。任一掩膜为空时 HD95 记为 `null`，只有一侧为空时 Dice 为零，两侧均为空时 Dice 也记为 `null`。均值使用有定义的值，配对差值只使用两组都能定义该指标的患者及时相。明细保留空掩膜标记，汇总保留有效计数。

## 单组训练与独立预测

```powershell
python -m nnunetv2.run.paired_tam train $prepared runs/c5 --preset C5 --fold 0 --seed 12345 --device cuda
python -m nnunetv2.run.paired_tam evaluate $prepared runs/c5 --split test --fold 0 --device cuda
python -m nnunetv2.run.paired_tam predict runs/c5 predict_manifest.json predictions/c5 --fold 0 --device cuda
```

独立预测重载 checkpoint，在各时相的原始网格导出 `<case_id>_<phase>` 掩膜。输出时相由训练时的 plans 决定。
