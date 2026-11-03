# 600 投影、均匀 20 周期的模拟数据验证

本项目从现有 `generate_round1_contrast_recon.py`、`generate_round1_contrast_gt.py`、
`generate_pig_first_phase_prior.py`、`utils/tool.py` 和 `utils/phase_mapping.py` 复制并局部修改，
项目中的三个主文件命名为 `uniform_recon.py`、`uniform_gt.py` 和 `uniform_prior.py`，
保留 HU→线性衰减校准、TIGRE 锥束正投影、FDK、FOV、TIFF/NIfTI GT 等现有数值流程。
新实验只用固定周期表分配投影相位，不读取真实投影的周期检测结果。

## 实验定义

- 600 个投影，索引 `0..599`，总扫描角 `300°`，角度为 `index × 0.5°`。
- 20 个等长周期，第 `k` 个周期覆盖 `[30(k-1), 30k)`，恰好 30 个投影。
- 周期起点是相位最小值，中点 `30(k-1)+15` 是相位最大值，终点 `30k` 是下一周期的起点。
- 沿用原项目的「最邻近相位」规则，将两段各 15 个投影映射到 CT 相位。默认使用 Round 6 的 `IM00001..IM00010`，最小/最大相位分别为 1/10。
- `settings/cycle.csv` 是原映射器使用的边界表；`settings/projection_phase_mapping.csv` 是可直接核查的逐投影表。共同边界的周期标签按半开区间归入下一周期，相位标签与原映射器一致。

## 环境与运行

在本机已安装 TIGRE 的 `common` Conda 环境中，可从本目录运行：

```powershell
D:\python\miniconda\env\env\common\python.exe run.py prepare
D:\python\miniconda\env\env\common\python.exe run.py list
D:\python\miniconda\env\env\common\python.exe run.py all
D:\python\miniconda\env\env\common\python.exe validate.py
```

`prepare` 生成两张周期/相位表；`list` 检查默认 CT 系列，不执行 GPU 工作；
`all` 对默认 `Control_1_7569/Heart_with_contrast_back` 完成正投影、全投影先验 FDK、
各相位稀疏 FDK 和对应 GT。也可用 `recon` 或 `gt` 分开执行。已有结果采用设置和周期表签名跳过；使用 `--overwrite` 重算。
`validate.py` 检查 600 行映射、每相位投影数量、投影数组有限值以及先验/重建/GT 的形状和间距，
并写出 `outputs/validation_report.json`。检查其他系列时可传 `--recon-dir`、`--gt-dir`；
其他相位数可传 `--n-phases`。

若使用其他环境，安装 `requirements.txt` 中的依赖并确保 TIGRE 能使用 CUDA GPU。
当前工作区的 Python 3.14 环境没有 `tigre`，不能用于正投影和 FDK；`prepare`、`list` 和测试可以运行。

CT 默认读取相邻目录 `../data/pig/AHFP-Round-6`；体数据本身较大，没有复制到本项目。
可通过 `--round-root`、`--settings-csv`、`--calibration-csv` 修改输入。
`--series <猪ID或相对系列路径>` 选择单个或多个系列，`--all-series` 执行设置表中所有系列。
其他轮次若使用 11 相 CT，同时传 `--n-phases 11` 和该轮的设置表；请用不同的
`--recon-root`、`--gt-root` 保存不同实验，以免同名系列覆盖。

## 输出

默认结果保存在 `outputs/recon/<系列相对路径>/` 和 `outputs/gt/<系列相对路径>/`：

- `projections.npy`：`float32`，形状 `[600, 探测器行, 探测器列]`，按扫描索引排列。可用 `np.load(path, mmap_mode="r")` 低内存读取。
- `prior_image.tif/.nii.gz`：600 投影的 FDK 先验。
- `phase_XX_reconstruction.tif/.nii.gz`：相位 `XX` 对应投影的稀疏 FDK。
- `phase_XX_gt.tif/.nii.gz`：与重建网格匹配的每相位 CT 真值。
- `reconstruction_config.json`、`gt_config.json`、两套 `manifest.json`：输入与设置来源、处理状态。
- `outputs/validation_report.json`：结果校验摘要。

单个完整系列会占用数 GB 磁盘和较多 GPU 显存。运行测试：

```powershell
python -m pytest tests -q
```
