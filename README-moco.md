# RSTAR-DPI-MoCo 运行说明

本分支在原样调用 RSTAR Stage II 的基础上估计三维运动场。实验配置和采纳条件见 [SPEC.md](SPEC.md)，实施顺序见 [PLAN.md](PLAN.md)。本地的公开权重与模拟体模可用于工程验收。历史 4／1／13 猪心试点使用规格登记的另一个 Stage II 权重和按扫描划分的清单。

## 本地体模验收

从项目根目录用包含 PyTorch、CUDA 和 TIGRE 的环境运行。Windows 本机已用 Conda `common` 环境验证。脚本从同级基线工程的已有体模读取输入，输出目录使用新路径。

```powershell
$py = 'D:/python/miniconda/env/env/common/python.exe'
& $py -m unittest discover -s tests -v
& $py -m validation.run_moco_checks --source ../RSTAR4D/runs/acceptance-20261116/demo_final --out runs/moco_check
```

验收包含 10／11 相位基线数值对照、配准完整体与分块接缝、两期断点续训、A/A2、混合投影和仅视角表重放、B/C/D、印刷符号和未归一化和，以及锁定配置后的留出体模评分。配准体模只做少量更新。体模的全角 FDK 参考、阈值和统计与猪心试点分开保存。

## 历史扫描清单与 A/A2

病例清单沿用 `validate-data` 的格式。通过扫描 ID 把病例明确分到训练、验证和测试三份。原始 `animal_id` 仍代表生物动物，`split-moco` 报告跨 split 的同动物扫描。生成的子清单保留相对路径、源摘要与 split 标签。若有缺相位，在源清单中补充 `expected_phase_labels`，用完整采集顺序列出真实标签。HDF5 的 `phase_labels` 记录实际存储的顺序。

```powershell
$p = 'runs/pilot_moco'
& $py -m rstar4d split-moco --manifest data/pilot/manifest.jsonl --scan-split data/pilot/scan_split.json --out "$p/splits"
$train = "$p/splits/train_cases.jsonl"
$val = "$p/splits/val_cases.jsonl"
$stage2 = 'path/to/verified/stage2/best.pt'
& $py -m rstar4d infer-moco --manifest $train --baseline-checkpoint $stage2 --out "$p/b_train"
& $py -m rstar4d infer-moco --manifest $val --split-name val --baseline-checkpoint $stage2 --out "$p/b_val"
$b = "$p/b_val/b/manifest.jsonl"
& $py -m rstar4d train-register --config configs/moco/a_learned.yaml --manifest $train --val-manifest $val --baseline "$p/b_train/b/manifest.jsonl" --val-baseline $b --out "$p/register_b"
$reg = "$p/register_b/checkpoints/best.pt"
& $py -m rstar4d infer-moco --config configs/moco/a_zero.yaml --manifest $val --out "$p/a_zero"
& $py -m rstar4d infer-moco --config configs/moco/a_learned.yaml --manifest $val --baseline $b --register-checkpoint $reg --out "$p/a_learned"
& $py -m rstar4d evaluate-moco --manifest $val --split-name val --out "$p/scores_a" --predictions "b=$b" --predictions "x=$p/a_zero/x/manifest.jsonl" --predictions "zero=$p/a_zero/d/manifest.jsonl" --predictions "learned=$p/a_learned/d/manifest.jsonl" --metric-config data/pilot/metrics.yaml
```

历史试点先对 Stage II 文件重算 SHA-256，按规格核对权重来源、物理强度和生成参数。可在试点 YAML 中设置 `baseline.checkpoint_sha256` 强制检查。已有增强图用 `--baseline` 提供，清单须保留病例、Stage II checkpoint 摘要和 HDF5 输入合同。历史增强图缺少 AMP 或设备记录时，该缺项保留为未知，无法与已记录不同推理设置的增强图合并为同一训练输入来源。

`train-register --resume` 使用 `checkpoints/last.pt`，并保持原始训练参数、数据清单、基线来源和运行目录。用 `--stop-after-epoch` 暂停以验证续训。A2 用 `a2_input_x.yaml` 单独训练和推理，同一预算下保留 A 的邻对规则和搬运图。配准训练只读几何支持和配准输入。输入 HDF5 可提供二值 `geometry_support[Z,Y,X]` 或 `geometry_support[T,Z,Y,X]`，并在元数据填写 `geometry_support_rule`。有登记的 FDK 几何时使用重建 FOV，其余配对数据以声明的体素网格为默认支持。配准使用每轴默认 4 体素位移上限，裁块时外扩空间上下文，在中心域求局部 NCC。推理分块可设置 `registration.inference_tile`，保持至少 64 体素上下文和全局步幅对齐。

## 校正图与 B/C/D

在正式试点配置中，从重复重建预先确定 `pigae.atol` 和 `pigae.rtol`，并将 `pigae.tolerance_evidence` 指向记录依据的文件。保留所选消融预设的 `outputs`、`fusion.source` 和 `pigae.mode`。配置、输入来源和误差日志随校正图保存。可用以下路径准备同源校正图。

- 混合投影使用原病例清单的 `projections`、`projections_sha256`。文件需有 `mixed_line_integral`、`projection_index`、`angles_rad`、`state_label` 和 `geometry_json`。如演算包含逐视角几何，`view_geometry` 组按独立 `projection_index` 连接 `offDetector`、`rotDetector`、`COR` 和显式的 FDK 乘性权重 `fdk_weight`。
- 模拟重放用带摘要的 `acquisition` 视角表和 `--raw-manifest` 指向原 CT。每期体积按原 SHA-256 验证，只投影已有索引分配给该相位的角度。旧 Poisson 数据须在原 CT 清单对应行提供生成时的 `simulation_seed`。
- 外部校正用 `--external-c` 提供预测格式的清单，每例包含 `case_id`、`animal_id`、`path`、`sha256`、`source_sha256`、`source_metadata_hash`。文件内为浮点 `correction[T,Z,Y,X]` 和同源 `metadata_json`，其中 `pigae` 登记 `mode`、`calibration_order: after_pigae`、`intensity_units: mm^-1`、`source_sha256`、输入文件摘要映射 `input_sha256s` 和物理来源 `provenance`。

```powershell
& $py -m rstar4d prepare-moco --config data/pilot/b_source_c.yaml --manifest $val --out "$p/c_val"
& $py -m rstar4d infer-moco --config data/pilot/b_source_c.yaml --manifest $val --baseline $b --register-checkpoint $reg --correction "$p/c_val/manifest.jsonl" --out "$p/b_source_c"
```

B 的并列比较把 `--manifest` 显式设为各组共同可用的扫描集合。使用同一个配准 checkpoint 和场缓存目录，默认取输出目录同级的 `fields`，也可传 `--field-cache`。C 的四份预设各自同时导出两个回注结果。选择 D 的 `d_infer_all.yaml` 复用原模型，`d_train_all.yaml` 则另训一个全对模型。`printed.yaml` 需要对应符号的校正产物。`raw_sum.yaml` 将未归一化分子保存在 `support/<case>/coverage.h5`，同时提供与归一化图的差值摘要。该文件也保存总权重、外部权重、近／远权重、贡献计数、几何与融合有效域。

每一运行目录有 `effective.yaml`、阶段摘要和预测清单。融合推理以目标相位发布带摘要的中间 HDF5，恢复时可跳过完整目标。缓存逐相位对保存 XYZ 体素位移与键，单独记录新场、命中、网络前向和已恢复目标。

## 评分和锁定测试

`evaluate-moco` 按病例、相位和强度域对齐各输出，生成 `phase_metrics.csv`、`cases.jsonl` 与 `summary.json`。参考有效域来自配对 HDF5。共同外部有效域是各融合输出外部贡献覆盖的交集。可用 `phase_labels` 指定预先登记的评分相位，默认使用声明的采集顺序。室壁评分通过 `annotations` 指定 JSON，血池曲线通过 `blood_pool_label` 选择配对 HDF5 中的参考 `roi_labels`。未设置重复测量证据时血池曲线 MAE 的允许增加量为零。标注 JSON 的最小结构如下，坐标顺序为扫描仪 ZYX 毫米。

```json
{
  "cases": {
    "case_id": {
      "source_sha256": "<paired.h5 SHA-256>",
      "wall_points": {
        "3": [{"id": "wall-1", "center_scanner_zyx_mm": [0, 2, 5], "normal_scanner_zyx": [0, 0, 1]}]
      },
      "landmarks": [
        {"id": "point-1", "source_phase": 2, "target_phase": 3,
         "fixed_scanner_zyx_mm": [0, 2, 5], "moving_scanner_zyx_mm": [0, 2, 6]}
      ]
    }
  }
}
```

在查看候选图前登记室壁测点、血池标签、相位范围和评分配置。用 `--split-name val` 对 `b` 和所有候选的 `y_single`、`y_multi` 评分。然后运行 `lock-moco --validation <scores_c/summary.json> --out <lock.json>`。锁定后以 `--split-name test --test-lock <lock.json>` 运行准备、推理与评分。正式测试所用的配置、模型来源和评分参数必须与锁相符。有效标注不足或没有候选满足条件时，测试锁选用冻结基线。
