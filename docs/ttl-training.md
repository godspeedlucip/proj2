# 冻结 RSTAR-STAM 后训练 TTL

Stage II 在 `x2` 跳连上加入一个 TTL。`--init-from-stam` 严格载入父检查点并冻结全部继承参数。Adam 只有 `ttl` 参数组，共 1,887,136 个可训练参数。`best.pt` 按固定验证样本上的 `validation_l1` 严格下降更新，第零步可入选。父模型的 STAM 模式、分块与重计算设置从检查点继承。TTL 算子与注意力偏置初始化已与作者公开的固定版本核对。[TT U-Net notebook](https://github.com/ZihengD/TT-U-Net/blob/51d4f05afee68058664c31fddfe1079bf37ec928/code/TTUNet/TTUNet_demo.ipynb)

## 父检查点与预检

在项目根目录运行。`$parentCkpt`、`$manifest`、`$split`、`$lvAnnotations`、`$lvExtractor` 分别指向已验证的获胜 STAM 检查点、配对 HDF5 清单、动物划分、相位标记的左室标注和冻结 TorchScript 提取器。用既有消融记录里的 `$expectedParentSha256` 核对文件身份。

```powershell
$parentHash = (Get-FileHash -LiteralPath $parentCkpt -Algorithm SHA256).Hash.ToLowerInvariant()
if ($parentHash -ne $expectedParentSha256) { throw 'STAM parent SHA-256 mismatch' }
Get-Content -LiteralPath $split
python -m rstar4d validate-data --manifest $manifest --split $split
```

从 `configs/train/pig_ttl_full.yaml` 创建每个种子的配置。按获胜检查点核对 `model_profile`、`loss`、`normalization` 和 `temporal_roll`。训练入口还会核验数据快照、动物划分、强度域、校准方式及相位拓扑。省略 `stam.mode` 即继承父模式，显式指定时须与父检查点一致。

模板使用与 nnU-Net 相同形式的逐轮多项式衰减。第 1 轮使用初始学习率 1.5e-4，已完成 t 轮时下一轮的学习率为 1.5e-4 × (1 − t/N)^0.9，N 取 Stage II 的总轮数。模板默认 N=30，开跑前可将 `epochs["2"]` 改为其他预算，例如 60。使用 `--stop-after-epoch 30` 暂停、再从同一运行的 `last.pt` 续训时，日程继续按最初锁定的总轮数执行。[nnU-Net PolyLR](https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunetv2/training/lr_scheduler/polylr.py)

测试前锁定 TTL 学习率、日程、有效更新预算、精度、tile/overlap、评价配置与运动容许量。示例使用 AMP 和 tile `32 128 128`、overlap `16 64 64`。FP32 运行时，各组推理统一添加 `--fp32`。

## 原版 RSTAR + TTL

这条路径使用 `configs/train/pig_rstar_ttl_full.yaml`，其中 `stam.mode: 'off'`。指定原版 RSTAR Stage II 检查点后，训练只更新 `x2` 跳连上的 TTL，使用 Adam 和相同的逐轮 PolyLR。第零步验证、父 SHA-256、`best.pt`、`last.pt` 和续训沿用 TTL 的保存规则。

```powershell
python -m rstar4d train --stage 2 --config configs/train/pig_rstar_ttl_full.yaml --manifest $manifest --split $split --init-from-stage2 $rstarStage2Ckpt --run-dir runs/rstar_ttl
python -m rstar4d train --resume runs/rstar_ttl/checkpoints/last.pt
```

`$rstarStage2Ckpt` 须为原版 Stage II 权重。若原文件只有参数，先用 `convert-weights --stage 2` 记录实际 profile、强度域、校准、归一化、动物划分和相位拓扑。资源预检使用同一配置，将 `benchmark --weights` 指向原版检查点。

评估 TTL 的独立增益时，用同一个原版检查点直接推理作为基线，再用 `runs/rstar_ttl/checkpoints/best.pt` 预测相同病例，沿用相同推理与评价协议。

## 资源预检

RTX A6000 模板使用 48 GB 显存配置、batch 1、FP16 AMP 和每步 1 个裁块。TTL 的 "z_chunk_size: 0" 让全部独立轴位层在一批中计算，三种裁块保持原方案。[RTX A6000 规格](https://www.nvidia.com/en-us/products/workstations/rtx-a6000/)

```powershell
python -m rstar4d benchmark --stage 2 --config $ttlConfig --phase-count 11 --steps 3 --weights $parentCkpt --out runs/ttl_benchmark.json
```

命令对每种裁块用同一个合成输入测量父模型和 TTL 模型。父模型仅做前向。TTL 先预热一次有效更新，再记录计时更新、AMP 溢出、训练峰值显存和两组推理成本。用 `ttl.mode: local3` 进行资源测试时，另传 `--temporal-topology ordered_states` 或 `--temporal-topology cyclic`。

RTX 4070 Ti SUPER、PyTorch 2.3.1+cu118、AMP、11 相位、`ttl.z_chunk_size: 0` 的本地工程预检使用 `stam-acceptance/phantom/frozen` 检查点（SHA-256 `51e4fc11fc243085387e1de5e4b8b60280a54e601bf0c15be3eac59c0f17f862`）。每种裁块均完成 3 次计时更新，预热和计时阶段均无 AMP 跳步，父张量逐值相等。

| 裁块 Z×Y×X | B 训练 GiB | B 秒/更新 | A/B 推理秒 | A/B 推理 GiB |
|---|---:|---:|---:|---:|
| 16×128×128 | 3.256 | 0.203 | 0.098 / 0.092 | 0.632 / 0.656 |
| 32×96×96 | 3.579 | 0.204 | 0.095 / 0.092 | 0.691 / 0.710 |
| 64×64×64 | 3.147 | 0.168 | 0.076 / 0.077 | 0.629 / 0.647 |

同一工程检查点还通过了 CPU FP32 小体模的派生、两次更新、续训、推理和左室配对评价。第 2 轮完成时共有 4 次有效更新。full 与 tiled 在重叠边界的 MAE 为 A 0.001439、B 0.000994 mm⁻¹。

## 正式 A/B

A 使用同一份获胜检查点推理一次。只有父 SHA、病例、相位、归一化、推理和评价协议均可核验时才复用结果。

```powershell
python -m rstar4d infer --checkpoint $parentCkpt --manifest $manifest --split $split --split-name test --mode tiled --tile 32 128 128 --overlap 16 64 64 --out runs/A/pred
python -m rstar4d evaluate --predictions runs/A/pred --manifest $manifest --lv-annotations $lvAnnotations --lv-extractor $lvExtractor --out runs/A/eval

# 示例使用 2026。为 2027 和 2028 分别修改配置的 seed 和运行目录。
python -m rstar4d train --stage 2 --config $ttlConfig --manifest $manifest --split $split --init-from-stam $parentCkpt --run-dir runs/B_2026
python -m rstar4d infer --checkpoint runs/B_2026/checkpoints/best.pt --manifest $manifest --split $split --split-name test --mode tiled --tile 32 128 128 --overlap 16 64 64 --out runs/B_2026/pred
python -m rstar4d evaluate --predictions runs/B_2026/pred --manifest $manifest --lv-annotations $lvAnnotations --lv-extractor $lvExtractor --out runs/B_2026/eval

python -m rstar4d compare-stam --seed 2026 --candidate runs/B_2026/eval --reference runs/A/eval --seed 2027 --candidate runs/B_2027/eval --reference runs/A/eval --seed 2028 --candidate runs/B_2028/eval --reference runs/A/eval --motion-tolerance-ml 0 --out runs/B_vs_A.json
```

中断后从同一运行目录的 `checkpoints/last.pt` 续训。检查点内含完整父模型和 TTL、优化器、日程、AMP 与 RNG 状态，可独立加载。完成后复查父文件 SHA-256。

```powershell
python -m rstar4d train --resume runs/B_2026/checkpoints/last.pt
```

报告配对的左室边界带 MAE 与 ED−ES 容积差绝对误差、有效与获益动物数、95% 动物 bootstrap 区间以及缺标和提取失败。全有效域 MAE、SSIM 与推理成本从相同协议的评价报告提取。若第零步入选，报告该种子未优于冻结起点。

可选 C 只将 `ttl.mode` 改为 `local3`，使用相同种子、初始权重、采样顺序和训练预算。按病例的 `ordered_states` 或 `cyclic` 决定端点是否环绕。用对应种子的 B 与 C 配对以评估远相位直接注意力的贡献。
