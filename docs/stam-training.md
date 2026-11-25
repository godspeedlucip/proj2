# RSTAR4D Stage II 的 STAM 训练与验收

[实施计划](superpowers/plans/2026-11-25-rstar4d-stam-joint-block-implementation.md) · [设计 Spec](superpowers/specs/2026-11-25-rstar4d-stam-joint-block-design.md)

## RTX A6000 配方

两份主配置保持 batch 1、AMP、30 个 epoch、三种 ZYX 裁块和同一学习率里程碑。每次处理最多四个互相独立的粗尺度 Z 层，并关闭 STAM 激活重计算。低显存运行可在配对配置里同时改为 `z_chunk_size: 1` 和 `activation_checkpointing: true`。目标卡为 48 GB RTX A6000。 [NVIDIA RTX A6000 规格](https://www.nvidia.com/en-us/products/workstations/rtx-a6000/)

SAM 文件放在 `weights/sam_vit_b_01ec64.pth`。当前文件的 SHA-256 为 `ec2df62732614e57411cdcf32a23ffdf28910380d03139ee0f4fcbe91eb8c912`。Stage II 扩展检查点携带完整 SAM 第 5 块，供推理和恢复直接读取。 [官方 ViT-B 权重](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth) · [SAM 源码版本](https://github.com/facebookresearch/segment-anything/tree/6fdee8f2727f4506cfbbe553e23b895e27956588) · [Apache-2.0 许可证](../rstar4d/models/SAM-LICENSE.txt)

在项目根目录执行下列命令。变量 `$stage2Weights`、`$manifest` 和 `$split` 指向同划分、同预处理的猪心原版 Stage II 检查点、配对数据清单和动物划分。若起点是裸权重，用 `convert-weights --stage 2` 声明真实 profile、强度域、校准、归一化、`--split` 和 `--temporal-topology`。

```powershell
python -m rstar4d benchmark --stage 2 --config configs/train/pig_stam_frozen.yaml --phase-count 11 --steps 3 --weights $stage2Weights --out runs/a6000_frozen.json
python -m rstar4d benchmark --stage 2 --config configs/train/pig_stam_joint.yaml --phase-count 11 --steps 3 --weights $stage2Weights --out runs/a6000_joint.json

python -m rstar4d train --stage 2 --config configs/train/pig_stam_frozen.yaml --manifest $manifest --split $split --init-from-stage2 $stage2Weights --run-dir runs/D_2026 --stop-after-epoch 1
python -m rstar4d train --stage 2 --config configs/train/pig_stam_joint.yaml --manifest $manifest --split $split --init-from-stage2 $stage2Weights --run-dir runs/G_2026 --stop-after-epoch 1

python -m rstar4d train --resume runs/D_2026/checkpoints/last.pt
python -m rstar4d train --resume runs/G_2026/checkpoints/last.pt
```

| 组别 | 配置或来源 | 绝对 epoch |
| --- | --- | --- |
| A | 直接推理同一原版 Stage II 检查点 | 基线 |
| B | `pig_stam_temporal.yaml` | 1–10 |
| C | D 的 `checkpoints/warmup/best.pt` 和 `last.pt` | 1–10 |
| D | `pig_stam_frozen.yaml` | 1–30 |
| E | `pig_stam_warmup.yaml`，可由 C 的 `last.pt` 派生 | 1–10 冻结，11–30 联合 |
| F | `pig_stam_backbone_f.yaml`，从 A 初始化 | 11–30 |
| G | `pig_stam_joint.yaml` | 1–30 |
| H | `pig_stam_backbone_h.yaml`，从 A 初始化 | 1–30 |

E 使用 `--fork-from runs/D_2026/checkpoints/warmup/last.pt` 派生到新目录，继承 C 的历史最优检查点、Adam 状态和绝对 epoch。独立运行 E 时使用 `--init-from-stage2`。F 和 H 会先评估 A，将它纳入 best 选择。正式比较使用配对的 2026、2027、2028 种子。每个种子修改 YAML 的 `seed` 并使用新运行目录。

```powershell
python -m rstar4d infer --checkpoint runs/G_2026/checkpoints/best.pt --manifest $manifest --split $split --split-name test --mode tiled --tile 32 128 128 --overlap 16 64 64 --out runs/G_2026/predictions
python -m rstar4d evaluate --predictions runs/G_2026/predictions --manifest $manifest --lv-annotations $lvAnnotations --lv-extractor $lvExtractor --out runs/G_2026/evaluation
python -m rstar4d compare-stam --seed 2026 --candidate runs/G_2026/evaluation --reference runs/D_2026/evaluation --out runs/G_vs_D_2026.json
```

正式比较使用验证集锁定的统一 tile/overlap。多种子对比按相同顺序重复 `--seed`、`--candidate` 和 `--reference`。用 `--motion-tolerance-ml` 传入锁定的运动容许量，默认 0 mL。程序按匹配相位、扫描和种子计算动物级配对差，报告获益动物数和动物级 bootstrap 区间。

`--lv-annotations` 是可信标注 JSONL。每行包含 `case_id`、`source_sha256`、实际 `phase_label`、标注 HDF5 相对路径 `path`、文件 `sha256` 和三维二值掩膜的 `dataset`。有生理 ED/ES 标注时填写 `role`。各模型使用同一冻结 TorchScript 提取器。输入为原强度域 `[1,1,Z,Y,X]`，输出为同网格二值掩膜 `[Z,Y,X]` 或 `[1,1,Z,Y,X]`。报告统计缺标、空边界带和轮廓提取失败。

## 本地验收

本地测量使用 RTX 4070 Ti SUPER、公开 Stage II、官方 SAM ViT-B、11 相位合成张量、AMP 和原版训练损失。每个裁块完成一次有效更新，均无 AMP 溢出。耗时含首步开销，显存为 PyTorch 峰值分配量。

| 裁块 ZYX | A6000 配方冻结 GiB / 秒 | A6000 配方联合 GiB / 秒 | 低显存冻结 GiB / 秒 | 低显存联合 GiB / 秒 |
| --- | ---: | ---: | ---: | ---: |
| 16×128×128 | 5.57 / 2.18 | 7.12 / 3.33 | 4.47 / 2.42 | 6.02 / 3.33 |
| 32×96×96 | 6.03 / 2.03 | 7.72 / 3.39 | 4.95 / 2.16 | 6.64 / 3.58 |
| 64×64×64 | 5.27 / 1.80 | 6.74 / 2.97 | 4.36 / 1.95 | 5.82 / 3.14 |

[A6000 配方冻结记录](../validation/stam_gpu_a6000_config_frozen.json) · [A6000 配方联合记录](../validation/stam_gpu_a6000_config_joint.json) · [低显存冻结记录](../validation/stam_gpu_frozen.json) · [低显存联合记录](../validation/stam_gpu_joint.json)

修改前记录了 Stage II 在 10 和 11 相位奇数空间尺寸下的 FP32 输出哈希。回归共 44 项，覆盖初始输出、非零投影轴对照、参数量、首两步梯度、续训、派生和离线推理。小体模使用官方 SAM，冻结和联合各完成两次更新，随后通过 full/tiled infer、LV evaluate 和 compare-stam。CLI 验收结果由脚本写入指定输出目录的 `summary.json`。 [FP32 基线](../validation/stam_baseline_fp32.json) · [CLI 验收脚本](../validation/run_stam_smoke.py)

用公开 Stage II 在现有猪心病例的 17×129×129 裁块上做了跨域检查。tiled 全有效域 MAE 为 0.00310 mm⁻¹，SSIM 为 0.891。full/tiled 的共享中心、接缝 MAE 为 0.000653 和 0.000211 mm⁻¹。正式 A–H 结果需要同划分训练的猪心 Stage II 权重、锁定的多动物配对数据、可信 LV 与 ED/ES 标注和固定提取器。A6000 实机资源与动物级训练结果仍待测量。

```powershell
python -m unittest tests.test_model tests.test_training tests.test_inference tests.test_cli
python -m unittest discover -s tests
python -m validation.run_stam_smoke --out runs/stam_recheck --device cuda
```
