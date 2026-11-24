# DPI-MoCo

从带呼吸相位标签的锥束投影生成训练体、训练 RestoreNet、RegisNet 和 DenoiseNet，再对新扫描重建全相位体。数据和体输出使用 `[Z,Y,X]`，体素灰度以 `mm⁻¹` 保存。

## 运行

以下命令在训练机的项目根目录执行。从 `requirements.txt` 安装 Python 依赖，并安装与 GPU 兼容的 TIGRE。联调使用 PyTorch 2.3.1、CUDA 11.8、TIGRE 3.0.0 和 TensorBoard 2.16.2。换到其他机器时，把扫描清单中的相对路径改为该环境的数据位置。

```powershell
python -m dpi_moco prepare --manifest data/train.json --config configs/paper_equations.yaml --out runs/paper/train
python -m dpi_moco train restore --prepared runs/paper/train --out runs/paper/restore
python -m dpi_moco train register --prepared runs/paper/train --restore runs/paper/restore/G.pt --out runs/paper/register
python -m dpi_moco train denoise --prepared runs/paper/train --out runs/paper/denoise
```

训练默认每阶段 50 轮。2D 训练默认 `128×128`、batch 32，配准默认整对体、batch 1。清单含多个扫描仪域时，用 `--domain` 指定训练域。RestoreNet 可先用 `--loss mse` 训练，再在新目录通过 `--init-weights <G.pt>` 启动完整损失训练。`--max-batches` 可缩短联调运行。

## TensorBoard 与断点续练

```powershell
python -m tensorboard.main --logdir runs --host 127.0.0.1 --port 6006
```

每个训练目录下有 `tensorboard/`、`checkpoints/epoch_0001.pt` 等逐轮断点，以及供推理加载的 `G.pt`、`R.pt` 或 `O.pt`。TensorBoard 记录分步及逐轮损失、学习率、epoch、实际样本数、更新数，以及输入、预测和目标的并排预览。有验证集时也记录验证损失和图像。

`--epochs` 是完整学习率日程。用 `--stop-after-epoch` 在指定轮结束后主动暂停，再用 `--resume` 从最近一个完整 epoch 接续。下例第 6 轮延续原 50 轮学习率日程。

```powershell
python -m dpi_moco train restore --prepared runs/paper/train --out runs/paper/restore_pause --epochs 50 --stop-after-epoch 5
python -m dpi_moco train restore --prepared runs/paper/train --out runs/paper/restore_pause --resume
```

亦可用 `--resume <epoch_XXXX.pt>` 在新输出目录从指定轮继续。断点保存模型和判别器、Adam、学习率调度器、epoch 和全局步、采样器、Python/NumPy/PyTorch CPU 和 CUDA 随机数状态，并校验训练数据、配置及配准依赖的 `G.pt`。在轮内退出后，会重跑该未完成轮。TensorBoard 从最近一次提交恢复并清除未提交的时间点。逐轮断点完整写入后才发布，且同一训练目录带进程锁。

## 新病例推理与评价

依照 `data/train.json` 为新病例编写扫描清单。`views` 使用 `view_id,angle_rad,phase_id`，也可用 `angle_deg`，训练准备另需 `cycle_id`。可用 `valid` 剔除无效投影。相位 ID 默认为零基，原始标签从 1 开始时，设置 `phase_base: 1`。探测器偏移可给常量或逐投影数组。训练清单中的同一患者必须属于同一划分。

```powershell
python -m dpi_moco infer --manifest data/test.json --config configs/paper_equations.yaml --restore runs/paper/restore/G.pt --register runs/paper/register/R.pt --denoise runs/paper/denoise/O.pt --out runs/paper/test --keep-intermediates
python -m dpi_moco evaluate --pred runs/paper/test --reference data/test_reference.json --protocol configs/metrics.yaml --stages x b d dpri
```

`predictions.json` 带相位、网格、配置及权重 SHA-256。使用 `--keep-intermediates` 保存先验、FDK、校正体、增强体、融合体及加噪融合体。`--motion-qc` 保存统一窗宽窗位的三平面相位图、位移分位数和折叠比例。`--motion-ablation` 用同一批位移场另融合 `x` 和 `b`，同时保存 `c` 的零场融合结果。评价参考由单独的 JSON 提供，参照预测清单的 `scans[].phases[]`，每相位提供相同网格、单位和相对路径。

默认 `paper_equations` 使用 `printed + sum`，`engineering` 使用 `static_consistent + mean`。局部 FDK 使用 TIGRE Ram–Lak、等角近似和 `2π/n_views` 的角积分尺度。样例用 `[100,100]` 探测器补边。源生成器在重建后施加圆形 FOV 遮罩，本实现保存完整网格，中心 ROI 用于数值核对。

## 验证

```powershell
python -m unittest discover -s tests -v
python scripts/validate.py --out runs/validation_new
```

体模脚本覆盖 TIGRE 前向、FDK 和从准备到评价的 CLI 全流程。三个训练阶段均比较两轮连续训练与第一轮后恢复的最终状态，并检查 TensorBoard 在恢复后清除未提交的日志。确定性逐张量比较使用 CPU，样例重建使用 CUDA。
