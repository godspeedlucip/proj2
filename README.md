# RSTAR4D：猪数据训练与 TIGRE 仿真

基于 [ZihengD/RSTAR](https://github.com/ZihengD/RSTAR) 公开实现，提供可执行的两阶段训练工程。算法默认遵循公开代码：时空网络、circular 时间填充、Tetris 阶段切换、L1＋SSIM 和原学习率调度。`RSTAR/` 保留原始参考代码、权重、说明和示例图；新增工程位于 `rstar4d/`。

支持两条入口：

- 原始多相位 CT → TIGRE 投影/FDK → GT、CBCT、prior、有效掩膜 → 两阶段训练。
- 已有配对 HDF5/TIFF/NIfTI → 明确的相位、坐标与强度合同 → 同一训练流程。

提供断点恢复、滑窗推理、HDF5/TIFF/NIfTI 导出、原强度域图像指标及符合适用条件时的投影评价。

## 快速开始

从仓库的 `rstar` 分支检出并进入项目根目录。参考 `environment-train.yml` / `environment-sim.yml` 建立环境；真实投影需要兼容 CUDA 的 [CERN TIGRE](https://github.com/CERN/TIGRE)。开发机验证环境为 Python 3.9、PyTorch 2.3.1+cu118、TIGRE 3.0.0。

```bash
python -m pip install -e .
python -m rstar4d doctor
python -m validation.run_delivery_checks --out runs/my_acceptance
```

最后一条命令创建三个独立合成动物，用真实 TIGRE 算子运行两条数据入口的完整小规模训练、推理与评价。无需真实猪数据；输出目录必须为空。仅检查已有配对数据的训练环境时，可使用 `doctor --skip-tigre`。

正式猪数据训练示例：

```bash
python -m rstar4d pipeline --config configs/train/pig_local_16gb.yaml --manifest data/pig_prepared/manifest.jsonl --split data/animal_split.json --run-dir runs/pig_001
```

原始猪数据路径、逐系列 settings、cycle 与校准 CSV 通过配置指定。`configs/simulation/pig_*.yaml` 中的 `D:/...` 是开发机路径，需要按实际环境修改；原始数据和实验生成产物不随仓库发布。动物划分必须显式提供，不能按切片或相位拆分同一动物。

## 文档与验证

- [完整运行说明](README-training.md)
- [实施 Spec](docs/superpowers/specs/2026-11-16-rstar4d-pig-tigre-training-design.md)
- [实现验收报告](docs/reviews/2026-11-16-implementation-acceptance.md)
- [推送前测试日志](validation/pre-push-tests.log)
- [参考代码来源记录](rstar4d/provenance.json)

```bash
python -m unittest discover -s tests -v
```

开发机执行 36 项测试全部通过，无跳过；包含公开网络/损失对照、真实 TIGRE、两条入口闭环及恢复一致性。部分原始脚本对照测试需要开发机外部文件，其他机器缺少这些文件时会明确跳过。报告中的本地 `D:/...` 和 `runs/...` 链接是验收产物位置，不是仓库下载链接。

当前结果证明核心方法及工程链路可运行。未完成全部真实动物的 60/30 epoch 实验，不声称复现论文最终性能。猪域强度归一化、较小空间块和有效域处理属于显式适配。
