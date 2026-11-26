# Stage II 的 PIDA-Net 二维 DA

A/B 从同一个完整的 STAM Joint Block Stage II 检查点起步。A 继续训练，B 在 `conv5_2` 前加入 PIDA-Net 图 3 的二维可变形注意力。每个 `(batch, phase, z)` 只搜索本平面的静态先验特征。B 增加 67,995 个 DA 参数和 61,440 个融合权重。原解码器权重、偏置和新增通道的零初值使 A/B 的第零步输出在 FP32 容差内等价。 [PIDA-Net 原文](https://doi.org/10.1016/j.apradiso.2025.112276)

复现采用余弦点积选全平面 Top-1，同分时取行优先的首个位置，再从 FP32 原始点积的全平面 softmax 读取该位置的权重。两条 3×3 卷积分支分别预测 18 个偏移分量和 9 个掩膜值。最终在原始 V 的匹配中心周围做 3×3 双线性调制采样，再乘匹配权重。`query_chunk_size` 限制每批查询，`plane_batch_size` 限制每批平面，设为 0 时相应维度整批处理。

## 启动与续训

两份配对配置取自已验收的联合训练配方，只在 `pida_da.enabled` 上有区别。正式实验须用获胜父检查点的 `model_profile`、归一化、训练策略和学习率配置同步替换两组设置。`--init-from-stam` 核验父权重的 Stage II、STAM 模式、强度与校准、动物划分及数据文件哈希，严格读取完整权重，再为两组分别创建优化器和训练计数。完整的 STAM SAM 权重随检查点携带。

```powershell
python -m rstar4d train --stage 2 --config configs/train/pig_stam_pida_control.yaml --manifest $manifest --split $split --init-from-stam $stamWeights --run-dir runs/pida_A --stop-after-epoch 1
python -m rstar4d train --stage 2 --config configs/train/pig_stam_pida_da.yaml --manifest $manifest --split $split --init-from-stam $stamWeights --run-dir runs/pida_B --stop-after-epoch 1
python -m rstar4d train --resume runs/pida_A/checkpoints/last.pt
python -m rstar4d train --resume runs/pida_B/checkpoints/last.pt
```

每组在 `run_identity.json` 和检查点中记录父权重 SHA-256。`train.jsonl` 按 epoch 记录成功更新次数、AMP 跳步和裁块序列哈希。检查点和预测元数据记录 DA 版本、端口、打分规则与 `prior_mode`。旧数据缺这一字段时记为 `unspecified`。恢复使用同一运行目录的 `checkpoints/last.pt`，保留历史 `best.pt`。

只训练 DA 时，将 B 配置的 `pida_da.train_policy` 设为 `pida_only`。这一模式把零初值的 DA 输出加到三级跳接，保留原解码器输入宽度，共训练 67,995 个参数。F0 直接从同一父检查点推理，F1 用 `--init-from-stam` 训练后推理。

## 推理和比较

在验证集锁定相同 `tile`、`overlap`、精度和选模规则后，用 A/B 各自的 `best.pt` 推理测试集。`full` 模式搜索整张特征平面，`tiled` 模式搜索各裁块内的特征平面，可能选到不同参考位置。以下示例中的切块是待锁定的候选设置。

```powershell
foreach ($group in @('A', 'B')) {
    python -m rstar4d infer --checkpoint "runs/pida_$group/checkpoints/best.pt" --manifest $manifest --split $split --split-name test --mode tiled --tile 32 128 128 --overlap 16 64 64 --out "runs/pida_$group/predictions"
    if ($LASTEXITCODE -ne 0) { throw "Inference failed for $group" }
    python -m rstar4d evaluate --predictions "runs/pida_$group/predictions" --manifest $manifest --lv-annotations $lvAnnotations --lv-extractor $lvExtractor --out "runs/pida_$group/evaluation"
    if ($LASTEXITCODE -ne 0) { throw "Evaluation failed for $group" }
}
python -m rstar4d compare-stam --seed 2026 --candidate runs/pida_B/evaluation --reference runs/pida_A/evaluation --motion-tolerance-ml 0 --out runs/pida_B_vs_A.json
```

`compare-stam` 输出逐动物 B−A、可评价动物数、获益动物数与动物级 bootstrap 95% 区间。主指标为 LV 边界带 MAE，运动指标为 ED–ES 左室容积变化的绝对误差，并附全有效域 MAE、固定窗 SSIM 与推理耗时。多种子按配对次序重复 `--seed`、`--candidate` 和 `--reference`。

## 工程验收

在独立进程中严格加载旧 STAM 小体模检查点，10 和 11 相位的奇数 Z/Y/X 输入均通过 FP32 对照。DA 第零步最大绝对差分别为 `5.36e-7`、`7.75e-7`。CPU 小体模的 A、B 和 F1 各完成两次有效更新，分别验证 epoch 恢复、full/tiled 离线推理与 LV 评估。B 和 F1 的连续训练与中断恢复逐张量一致。F1 的宿主参数和缓冲区变化数为零。

| 裁块 ZYX | A 峰值 GiB / 秒每步 | B 峰值 GiB / 秒每步 |
| --- | ---: | ---: |
| 16×128×128 | 7.12 / 1.90 | 8.61 / 2.27 |
| 32×96×96 | 7.72 / 1.92 | 9.34 / 2.36 |
| 64×64×64 | 6.74 / 1.73 | 8.16 / 2.24 |

RTX 4070 Ti SUPER 上使用 11 相位合成张量、AMP 和同一小体模 STAM 父权重。每组每种裁块完成两次有效更新，AMP 跳步为零。计时含首次算子开销，显存为 PyTorch 峰值分配量。 [数值与资源验收记录](../validation/pida_validation.json)

```powershell
python -m unittest discover -s tests
python -m validation.run_stam_smoke --out runs/pida_smoke --device cpu --sam $samWeights --pida
python -m rstar4d benchmark --stage 2 --phase-count 11 --steps 2 --config configs/train/pig_stam_pida_da.yaml --weights $stamWeights --out runs/pida_gpu.json
```

正式动物级实验待提供同划分的获胜 STAM 检查点、对应配对清单和划分、可信 LV/ED–ES 标注与冻结提取器。上述小体模结果用于工程验收。
