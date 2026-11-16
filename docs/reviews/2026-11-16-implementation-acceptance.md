# RSTAR4D 完整实现验收 — 2026-11-16

已实现 Spec 的两条数据入口、TIGRE 仿真、完整两阶段网络训练、断点恢复、滑窗推理、导出、图像/投影评价与全部 CLI。默认算法以公开 Notebook/MATLAB 为准。当前交付证明代码和数据链路能够运行；尚未执行真实动物数据的完整 60/30 epoch 实验，也不声称复现论文表格数值。

## 交付位置与默认算法

- [运行说明](D:/desktop/课题2/proj/RSTAR4D/README-training.md)，[安装配置](D:/desktop/课题2/proj/RSTAR4D/pyproject.toml)，[可安装 wheel](D:/desktop/课题2/proj/RSTAR4D/dist/rstar4d_training-0.1.0-py3-none-any.whl)。包使用平铺 `rstar4d/`，可直接 `python -m rstar4d`，无需安装到外部环境。
- [网络](D:/desktop/课题2/proj/RSTAR4D/rstar4d/models/model.py)：保留公开权重的 232 项键和 2,118,693 参数；circular 时间边界；Stage I 跳过并冻结 z；Stage II 按公开代码进行 Dirac/零初始化。
- [默认训练配置](D:/desktop/课题2/proj/RSTAR4D/configs/train/pig_local_16gb.yaml)：L1+0.1×(1−原 SSIM)，Adam 1.5e-4，60/30 epoch，原里程碑衰减；两阶段全 T。猪域固定除以 0.02，空间块缩小是明确的适配。L1/reflect 仅作显式可选方案。
- [仿真](D:/desktop/课题2/proj/RSTAR4D/rstar4d/simulation/pipeline.py)：真实 TIGRE Siddon/FDK；GT 为每相位全角重建，CBCT 为该状态分配角度重建，默认 prior 是混合投影各状态 FDK 按视角数加权；同一 post-FDK 校准、有效掩膜和输出坐标合同。
- [训练器](D:/desktop/课题2/proj/RSTAR4D/rstar4d/training/trainer.py)：动物均匀抽样、固定验证块、梯度累积、AMP、best/last、完整 RNG/优化器/scheduler/配置/数据指纹；运行归属检查和 epoch 边界恢复。
- [评价](D:/desktop/课题2/proj/RSTAR4D/rstar4d/evaluation/report.py)：原强度域 MAE/RMSE、固定窗 SSIM、ROI、病例/动物等权统计；满足物理条件时计算并汇总投影指标。

原 Notebook、公开权重、外部生成脚本和原始数据均未修改。8 个已登记源文件的 SHA256 再次核对全部一致：[来源及环境证据](D:/desktop/课题2/proj/RSTAR4D/runs/acceptance-20261116/implementation_provenance.json)。用户有意采用不同配置的历史 GT/CBCT 不属于本次故障检查或修改对象。

## 实际执行的检查

环境：Windows 10，Python 3.9.23，Torch 2.3.1+cu118，TIGRE 3.0.0，NumPy 1.26.3，RTX 4070 Ti SUPER 16 GiB。实际 Python 为 `D:/python/miniconda/env/env/common/python.exe`。环境 YAML 是重建参考；未新建另一套 Conda 环境验证安装。

完整 unittest 共 **36 项通过，无跳过**。日志：[implementation-tests-final.log](D:/desktop/课题2/proj/RSTAR4D/validation/implementation-tests-final.log)。wheel 已使用 `--no-deps --no-build-isolation` 构建并检查包内容；构建日志在 [package-build.log](D:/desktop/课题2/proj/RSTAR4D/validation/package-build.log)。

另保存两条入口独立的完整 CLI 产物：真实小体模生成/已有 HDF5 导入 → 两阶段训练 → best 重载 → 滑窗 → HDF5/TIFF/NIfTI → 图像及投影评价。每阶段 1 epoch、8 draw；测试动物与训练、验证动物独立。结果目录：[demo_final](D:/desktop/课题2/proj/RSTAR4D/runs/acceptance-20261116/demo_final)，[完成状态](D:/desktop/课题2/proj/RSTAR4D/runs/acceptance-20261116/demo_final/acceptance_status.json)。这些短运行未收敛，预测指标可能差于输入，不作为恢复性能证据。

复跑并保存同类产物：

```powershell
& 'D:/python/miniconda/env/env/common/python.exe' -m validation.run_delivery_checks --out runs/my_acceptance
```

## Spec 验收对应

| ID | 证据与实际范围 |
|---|---|
| A01 | 实际 inventory 发现 63 系列、10 个无歧义动物 ID；Round1 为 T11，其余为 T10；泄漏划分被拒绝。 |
| A02 | 原函数对照覆盖非对称数据、正负偏移、奇偶网格、crop/zoom/wide-FOV；不转置原 NIfTI 数组。 |
| A03 | HU→mu、水/空气、padding 逐值、无效掩膜及梯度测试通过。 |
| A04 | 用户 cycle.csv 的 600 views、末角 299.5°、T10/T11 分配计数通过。 |
| A05 | 两条实际仿真流程及代码数据流检查；prior 只来自动态混合投影；真实病例保存 600 条投影。 |
| A06 | 实际 CUDA 水球 Ax/FDK；中心 mu≈0.02007918，偏差约 0.40%；另直接调用原重建函数，六种偏移/网格组合在 rtol=1e-4、atol=1e-6 内一致。 |
| A07 | 原函数 CPU 有效掩膜与 GPU 重建比较覆盖 legacy crop、zoom、wide-FOV 和 padding；投影输入不被 FDK 修改。 |
| A08 | 重建后校准零保持、系数及应用阶段检查；训练 cohort 强度/校准不一致拒绝。 |
| A09 | 新导入的相位顺序、校准系数、spacing/geometry/来源合同必须匹配；空 geometry/缺少 sidecar 拒绝；共同重采样有同步测试。 |
| A10 | 两份原权重严格加载；与原 Notebook CPU 前向及完整有效域 loss 对照；T10/T11。 |
| A11 | circular/reflect 端点行为分别测试。 |
| A12 | Stage I/II 的 z 梯度、初始化、继承与 pipeline 再进入测试。 |
| A13 | 非恒定体模：本次 Stage I 60 步、Stage II 95 步达到末20步均值≤初20步50%；均小于300步，无 NaN。 |
| A14 | CPU 连续2epoch与1+resume+1模型/Adam/scheduler相同；包含同动物多病例和manifest重排；配置变化拒绝，迁移完整运行目录成功。 |
| A15 | mask 梯度、空窗口、补齐、无法采到合格块的明确失败；不会重选动物/病例/块尺寸；累积最后不足一组的权重测试；NaN不保存checkpoint。 |
| A16 | 恒等/已知偏差的原单位指标、空ROI/null、病例与动物投影汇总测试。 |
| A17 | T10/T11、奇数空间尺寸、恒等预测器滑窗正权重全覆盖；真实网络 CLI 导出与重载。 |
| A18 | 原始体模和配对 HDF5 两条入口均实际完成两阶段训练、推理与评价，产物保存在 demo_final。 |
| A19 | T11、原公开损失、fp16 三种块均完成 forward/backward/Adam；峰值如下。代码明确传播 CUDA/OOM/损坏权重错误；未人为耗尽显存触发系统级 OOM。 |
| A20 | 未校准完整FOV自投影误差接近数值精度；另执行有真实源偏移的 Ax→FDK→输出重投影，正确输出坐标误差小于重复加源偏移；校准域/不完整FOV明确跳过。 |

部分条目由运行测试与静态数据流检查共同支持，不表示穷举全部参数组合或跨版本 CUDA 数值一致。

## 真实猪病例与显存

本次完整生成 `round1__Control_1_7569__Heart_with_contrast_back`：

- GT/CBCT/valid：`[11,467,256,256]`，prior：`[467,256,256]`。
- spacing：`[0.375,0.4605,0.4605]` mm；强度域 `calibrated_mu`。
- 混合投影：`[600,474,648]`；相位视角数 `[28,62,60,60,61,58,61,60,60,62,28]`。
- paired/projections 总大小约 3.09 GB；[manifest](D:/desktop/课题2/proj/RSTAR4D/runs/acceptance-20261116/real_prepared/manifest.jsonl)、[摘要](D:/desktop/课题2/proj/RSTAR4D/runs/acceptance-20261116/real_case_summary.json)、[最终校验](D:/desktop/课题2/proj/RSTAR4D/runs/acceptance-20261116/real_validation.log)。

长时间真实仿真在边界审查修复前启动，其生成元数据保留当时实现指纹。生成算法未在此次修复中改变；最终代码重新通过其完整数据校验和训练块读取。缓存严格绑定代码指纹，因此用新实现重新生成时应使用新输出目录，不改写历史指纹。

最终真实数据块检查：Stage I `[1,2,11,1,256,256]`、Stage II `[1,2,11,16,128,128]`，每阶段两次有效更新。首次 Stage I 的 AMP scale 从65536降至16后稳定更新；短检查需要足够 draw，不能把跳过更新算作成功。公开 GradScaler 行为保留，逐次记录在 [real_patch_check.json](D:/desktop/课题2/proj/RSTAR4D/runs/acceptance-20261116/demo_final/real_patch_check.json)。这是同一病例的执行检查，没有把同动物拆成训练/测试来报告效果。

| Stage II 空间块 ZYX，T=11 | allocated GiB | reserved GiB | 两步时间 s |
|---|---:|---:|---:|
| 16×128×128 | 5.98 | 6.42 | 3.63 |
| 32×96×96 | 6.59 | 7.03 | 3.76 |
| 64×64×64 | 5.78 | 6.19 | 3.30 |

数据来自 [benchmark.json](D:/desktop/课题2/proj/RSTAR4D/runs/acceptance-20261116/benchmark.json)，包含真实参数更新。它衡量网络训练块，不代表整幅真实体积推理或大型仿真的总资源占用。

## 独立审查与边界修复

[独立报告](D:/desktop/课题2/proj/RSTAR4D/docs/reviews/implementation-final-review.md) 的初始结论为需要修改。六项 Important 均已用先失败后通过的回归验证并修复：完整配对合同、运行归属保护、manifest 顺序规范化、固定抽样单位的空间重试、先筛选再校验病例身份、生成任务失败时清除过期完成状态。投影汇总已补齐；原重建函数和真实偏移链的证据也已补齐。最终由主实现者复核修改边界并执行全套测试；没有把原始审查报告冒充修改后的独立批准。

## 尚未执行的研究实验

没有生成全部63系列、没有运行真实动物全预算60/30epoch、没有报告真实留出动物性能或论文表格复现。正式实验前仍需填写真实动物 train/val/test；一个嵌套目录的候选动物ID为7550/7596，需通过显式 identity_map 确定。该歧义不阻止选择其他确定病例。具体路径见 [inventory摘要](D:/desktop/课题2/proj/RSTAR4D/runs/acceptance-20261116/pig_inventory/summary.json)。
