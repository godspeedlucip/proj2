# PNMC 包内运行说明

本分支根目录只包含 `pnmc/`。PN-Net 是需要训练权重的增强网络；配准与运动补偿 OS-SART 按病例运行，不另行训练。完整顺序是投影数据 → FDK → PN-Net 推理 → 相位配准 → MC-OSSART。当前实现是项目的 PNMC-adapted 版本，不宣称逐参数复现论文。

## 环境与输入

在仓库根目录运行 `python -m pnmc`。环境需要 PyTorch、NumPy、SciPy、nibabel、tifffile、SimpleITK 和可工作的 TIGRE/CUDA 安装；`reconstruct --check-only` 会检查物理阶段依赖和 CUDA 设备。包中的 `models/legacy_priornet.py` 保留 matched PriorNet 对照所需的旧网络定义。

完整重建需要：

- **物理病例 manifest**：`CaseManifest` JSON，包含 `schema_version`、病例/动物/采集 ID、非空 `phase_ids`、原始投影 `projections`、实际 `angles_rad`、每视角 `phase_id_per_view` 和 `view_id`、扫描 `geometry`、`projection_volume_grid`、`reconstruction_grid`、`normalization` 与 `provenance`。投影 `ArrayRef` 应含 `path`、`key`、`axes=view,v,u`、`unit=line_integral`、`sha256`。格式由 `pnmc/data/manifest.py::load_case_manifest` 严格验证。重建不需要参考真值图像。
- **PN-Net checkpoint**：训练输出的 `best.pt`，须含本项目的格式版本、权重、架构、归一化和 `train_grid_hashes`。文件名不能证明它是 fullgrid 权重；训练网格指纹必须与病例重建网格一致。
- **网络配置 JSON**：训练该 checkpoint 时使用的 `NetworkConfig`，特别是通道、融合方式、attention、query chunk 等架构项。`pnmc/configs/reconstruction.json` 是已审计的项目默认重建配置，不代表适用于每个扫描。

## 命令

将下列路径替换为本机实际文件；先检查，再用同样的输入运行重建：

```bash
python -m pnmc reconstruct \
  --case /path/to/physical_case.json \
  --checkpoint /path/to/best.pt \
  --network-config /path/to/network_config.json \
  --config pnmc/configs/reconstruction.json \
  --output /path/to/pnmc_run \
  --device cuda:0 --check-only

python -m pnmc reconstruct \
  --case /path/to/physical_case.json \
  --checkpoint /path/to/best.pt \
  --network-config /path/to/network_config.json \
  --config pnmc/configs/reconstruction.json \
  --output /path/to/pnmc_run \
  --device cuda:0
```

若输出目录中已有可验证的完整阶段，第二条命令可加 `--resume`。恢复按**阶段或完整目标相位**进行，不保证从某一轮 MC 更新中间继续。`--case` 可重复提供多个候选病例；正常重建按 `case_id` 审计尝试，`--check-only` 则只接受一个病例。

单独查看 PN-Net 增强效果可用 `python -m pnmc infer --manifest ... --checkpoint ... --split test --output ... --device cuda:0`。已有增强体积也可用 `python -m pnmc register --case ... --enhanced ... --config pnmc/configs/reconstruction.json --output ... --device cuda:0` 单独执行配准。

## 结果与参数边界

重建结果包含增强阶段、DVF 与 QC、各目标相位 MC 体积、`mc_report.json` 的投影残差和迭代历史、阶段哈希及总报告。只有实际病例运行完成才能报告完整 PNMC 效果；代码测试或 checkpoint 存在不能替代真实病例验证。对照真值的图像指标应在重建后单独计算。

`epochs` 是 PN-Net 训练轮次；`sweeps` 是 MC-OSSART 对所有相位子集的迭代轮次。`lambda_initial`、`lambda_decrement` 和 `lambda_minimum` 控制每轮修正步长，它们可用验证病例选择，但不是网络训练。当前 TIGRE `matched` 反投影与 Siddon 前投影在本机是近似配对；项目不把先前的 `1e-3` 点积误差设为运行硬门槛。
