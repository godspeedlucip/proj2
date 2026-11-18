# rstar-advance 训练与评价验收

2026-11-18 在 Windows、Python 3.9.23、PyTorch 2.3.1+cu118、TensorBoard 2.16.2、TIGRE 3.0.0 环境执行。GPU 为 RTX 4070 Ti SUPER。两阶段训练、推理、指标及诊断均使用合成数据。正式动物数据待提供。

## 结果

| 验收项 | 观测 |
| --- | --- |
| 全套回归 | `python -m unittest discover -s tests -v` 执行 88 项，全部通过，0 项跳过。包括实际 CUDA AMP 更新、TIGRE 投影和 FDK 重建。 |
| 裁块预检 | 已知位移体模的整幅和默认裁块各有 4096 个有效运动窗口，损失在 FP32 容差内一致。过短的读取范围在前向前报告缺少支撑的轴、侧和体素数。真实扫描边缘按有效域截断。 |
| 八组短程消融 | baseline、M、P、B、MP、MB、PB、MPB 共用一份 Stage I 权重。各组第 1 轮结束后退出，从 checkpoint 续到第 2 轮。八组两轮的采样哈希逐轮相同。实际读回 17 份轮次归档，涵盖共享 Stage I 的 1 轮和 Stage II 的 16 轮。 |
| A+B 联合布局 | 对同一模拟病例的相位 2 采用两套实际投影视角，分别重建并以独立样本 ID 训练、推理、评价。清单另具独立验证动物。联合清单的两份预测共用同一病例参考。该病例有 2 个生理相位、4 个相位与布局组合，报告病例数和动物数均为 1。性能基准也能从联合清单取块并完成 Adam 更新。 |
| 指标分母 | 两相位样例中 MAE 有效 2 相，SSIM 因缺少完整窗口仅有效 1 相。布局覆盖不齐时，轮廓先按病例和相位平均布局，再按动物汇总。轮廓回归样例的两个相位均值为 1 和 0，动物均值为 0.5。图像统计保留有效体素或窗口权重。 |
| prior 错配诊断 | 原对照的 288 个目标位置中，2 个位置满足两侧单粗候选条件，均未通过细采样支撑。连续支撑体模得到 8 个可比点，两个相位各 4 个。错配减匹配的修正量 RMS 为 +1.9923 × 10⁻⁶。本结果验证筛选和计量链路，真实错配抑制效果待实验。 |
| 续训 | 两阶段确定性 FP32 的连续训练与新进程续训逐值一致。已核对模型、Adam、调度器、随机状态、采样顺序及历史最优。CUDA AMP 在恢复后的下一次有效更新上模型、优化器与 scaler 一致。故障注入覆盖归档写入中断、别名滞后、日志末行损坏和首轮重跑。 |
| TensorBoard | 对八组事件文件逐个读回训练和验证曲线、固定切片、误差图和轮廓叠加。第 1、2 轮各出现一次，缺失事件可由归档恢复。独立评价写出两种结构的 Surface Dice、HD95、ASSD 和有效动物数。另测了日志目录扫描期间未提交事件的可见性和可选标注 SHA-256 校验。 |

合成体模上 MPB 的整幅与滑窗预测在重叠区的 MAE 为 1.123 × 10⁻⁹，最大绝对差为 9.779 × 10⁻⁹。这里的窗口支撑、允许位移、轮廓距离和容差均以输出网格体素为单位。真实八组训练、真实错配诊断及留出动物的独立轮廓评价，需待正式数据与动物划分就绪后执行。

## 复现

从仓库根目录，在训练环境中使用新输出目录运行以下命令。

```powershell
python -m unittest discover -s tests -v
python -m validation.cardiac_smoke --out runs/cardiac-demo
python -m validation.prior_controls --smoke runs/cardiac-demo --out runs/cardiac-prior-demo
python -m unittest tests.test_layout_metrics.LayoutMetricTests.test_joint_fdk_train_infer_and_independent_contour_evaluation
python -m tensorboard.main --logdir runs/cardiac-demo
```

八组的采样与事件核对结果写入 `smoke.json`，配对结果写入 `report/summary.json`，prior 各级筛选数和门控读数写入诊断目录的 `diagnostics.json`。本机验收产物位于 `runs/acceptance-rstar-advance-20261118/verified/`。
