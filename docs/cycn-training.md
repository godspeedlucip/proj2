# CycN 训练与验收

## 运行入口

`cycn` 缺省关闭。两阶段均可启用。训练配方位于 [configs/train](../configs/train)。B、C、D 的主干初始学习率均为 1.5e-5。C、D 和冻结版的新增支路为 1.5e-4。第 10、20、25 轮结束后乘 0.5。五相模型的支路按病例的 `temporal_topology` 做取模或无重复端点反射，来源权重的 `model_profile` 单独决定主干填充。YAML 中显式关闭时写 `mode: 'off'`。

| 组别 | 配方 | 参数更新 |
|---|---|---|
| A | 来源 Stage II 检查点直接推理 | — |
| B | `pig_cycn_backbone.yaml` | 原版主干 |
| C | `pig_cycn_center_only.yaml` | 主干和中心复制支路 |
| D | `pig_cycn_five_phase.yaml` | 主干和真实五相支路 |
| 冻结 D | `pig_cycn_frozen.yaml` | 五相支路 |

使用原版 Stage II 基线检查点的 `model_profile`、强度域和归一化。检查点若附带原始数据快照及划分，训练入口会校验其一致性。固定验证裁块按病例平均 L1 选 best。恢复训练须保留同一个运行目录的 `checkpoints/last.pt` 与历史 `best.pt`。

```powershell
$py = 'D:\python\miniconda\env\env\common\python.exe'
& $py -m rstar4d train --stage 2 --config configs/train/pig_cycn_five_phase.yaml --manifest $manifest --split $split --init-from-stage2 $stage2Weights --run-dir runs/pig-D --stop-after-epoch 1
& $py -m rstar4d train --resume runs/pig-D/checkpoints/last.pt
& $py -m rstar4d infer --checkpoint runs/pig-D/checkpoints/best.pt --manifest $manifest --split $split --split-name val --mode tiled --compare-full --out runs/pig-D/val
& $py -m rstar4d evaluate --predictions runs/pig-D/val --manifest $manifest --metric-config configs/metrics/pig_attenuation_v1.yaml --lv-annotations $annotations --lv-extractor $frozenExtractor --out runs/pig-D/eval-val
```

正式验证前固定种子、动物划分、验证裁块及评估窗。B、C、D 完成 30 轮后，对照三组 `last.pt` 的有效步数及 `train.jsonl` 的 AMP 跳步，确认等预算后用各组 `best.pt` 和 A 的原版权重运行同设置的留出集推理。若验证阶段缺少重复勾画误差依据，预先采用平均运动绝对误差增量上限 0 mL。有独立依据时使用 `--motion-tolerance-ml` 和 `--motion-tolerance-source` 记录验证阶段锁定的值及来源。

```powershell
& $py -m rstar4d compare-cycn --a $evalA --b $evalB --c $evalC --d $evalD --seed 2026 --out runs/pig-comparison.json
```

可信 LV 标注采用 JSONL，每行包含 `case_id`、实际 `phase_label`、`source_sha256`、标注 HDF5 的 `path`、`sha256` 和同网格三维二值 `dataset`。生理 ED/ES 从病例元数据的 `ed_phase_label`、`es_phase_label` 或可信标注行的 `role` 取得。各组使用同一冻结 TorchScript 轮廓提取器，输入为原强度域 `[1,1,Z,Y,X]`，输出为同网格二值掩膜。若任一主要比较的有效配对运动动物少于两只，运动判定为未判定。

## 工程验收记录

完整回归共通过 41 项测试，数值对齐与资源测量见 [验收数据](../validation/cycn_acceptance.json)。

- 作者代码的对应两层与项目 3DBlock 共享同一组初始权重时，512×512 输入输出为 `[1,4,504,504]`，FP32 最大绝对差为 0。原块 710 个参数，XY 读出 192 个，图像端消融共新增 722 个。比较容限为 `rtol=1e-5`、`atol=1e-6`。[作者代码](https://raw.githubusercontent.com/shaohua-zhi/N-Net_and_CycNet/master/CycN-Net/model_CycNnet.py)
- Stage II 的 232 个主干参数键和固定 FP32 输出对齐。关闭扩展或零初始化投影时最大绝对差均为 0。启用非零投影后，`z_chunk=1` 和 `16` 的整网最大绝对差为 7.75e-7。
- 10 相 cyclic 和 11 相 ordered_states 均通过两轮 CLI 训练、验证、full/tiled 推理和左室评价，使用非连续相位标签和奇数空间尺寸。连续与暂停后恢复的五相实验均完成 8 次有效更新，模型权重、Adam 状态、学习率和历史 best 完全一致。
- RTX 4070 Ti SUPER 上按三种 ZYX 裁块、10/11 相、三种训练策略共测 18 项 CUDA AMP 组合，每项两次有效更新，均无 AMP 跳步。关闭扩展的最高已分配显存为 6.587 GiB，联合训练为 6.733 GiB，冻结为 6.721 GiB。合成病例的 A/B/C/D 在统一 tile 与 overlap 上另测 CUDA 推理成本和接缝误差。
- 猪心 7569 的 11 相真实裁块使用公开权重进行跨强度域的工程检查。CUDA AMP 下 tiled 为 1.088 s、0.333 GiB，full 为 0.239 s、0.752 GiB。以有效 FOV 计，内部 tile 边缘前后四体素的 full/tiled MAE 为 2.078e-4 mm⁻¹，其余区域为 2.068e-4 mm⁻¹。

复跑合成工程验收时使用新输出目录。

```powershell
& $py -m unittest discover -s tests
& $py -m validation.run_cycn_acceptance --out runs/cycn-check-new
& $py -m rstar4d benchmark --config configs/train/pig_cycn_five_phase.yaml --stage 2 --phase-count 11 --temporal-topology ordered_states --steps 2 --weights RSTAR/githubcode/checkpoints/checkpoints_Tetris2.pth --out runs/benchmark-new.json
```

## 猪心正式对照

已完成 M1、M2 和任务 6 的工程链路。M3 等待以下研究输入。

- **来源权重与来源契约**。本地公开 Stage II 权重 SHA-256 为 `02dd0a7f9e854d36f4d0ae05f53e31dbc017a961a01a4a3554439a0b995ce42d`。另有 `demo_final` 的合成数据检查点和已有猪心预测预览。尚未找到携带可核实训练来源、归一化、强度域和数据快照的猪心原版 Stage II 检查点。
- **多动物配对清单和动物划分**。已准备的真实配对清单只有动物 7569，形状为 TZYX `[11,467,256,256]`，数据 SHA-256 为 `35218b2ff0a93a586767742019827a15df6c25666b2572a8b5f7dbff8a90d2ec`，拓扑为 `ordered_states`，强度域为 `calibrated_mu`，归一化为 scale 0.02、offset 0。库存的划分模板未指定验证或留出动物。
- **可信 LV 轮廓和 ED/ES**。上述病例的配对 HDF5 未附 LV 掩膜或真实 ED/ES 标签。尚未找到覆盖正式留出集、可追溯到同一源网格的可信 LV 标注和冻结提取器。正式 A/B/C/D 的边界改善和运动保真均待判定。
