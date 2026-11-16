> GitHub distribution: source, configs, tests and documentation only. Local runs/, source PDFs, generated measurements, volumes, checkpoints and ZIP artifacts referenced in the historical reports are not uploaded. Regenerate demo data with `python -m bird.demo`; use your own study root for the pig adapter.

# BIRD 动态 CT 网络复现

以 Du et al., PMB 71 (2026) 105003 的**期刊版**为基准，提供完整的独立 PyTorch 参考实现。**不是作者官方源码；未声称复现论文表1/2数值。** 未确认公开官方 GitHub 仓库，检索记录见 [SOURCES.md](SOURCES.md)。

- [方法论详解](METHOD_CN.md)：设计逻辑、公式和手算例子、消融证据、适用边界。
- [网络复现 spec](SPEC.md)：张量/单位、模块、梯度归属、参数、数据合同、验收要求与补全假设。
- [验证报告](VALIDATION.md)：实际运行环境、测试、完整配置检查与合成训练结果。
- [猪心数据 + TIGRE 规格](PIG_CBCT_SPEC.md)：按用户代码制定的600视角、300°、10秒CBCT、真实采集时间、全源体FOV和GT网格。
- [TIGRE 运行说明](PIG_RUN.md) / [GPU 验证报告](PIG_VALIDATION.md)：数据生成、TIGRE训练/恢复/导出及实际验证记录。
- [论文一致性与完整运行复核](REPRODUCTION_AUDIT.md)：逐式审查、33项测试、原始paper配置与16级TIGRE网络的运行证据及差异范围。

**先验图像不需要。** BIRD 的参考空间/拓扑保持图像由本次扫描投影共同学习。猪心相位 CT 用来模拟测量，不作为重建网络的先验输入或 GT 监督。

本机已在 `D:/python/miniconda/env/env/common/python.exe` 验证 TIGRE + CUDA。首个真实猪心序列的完整600视角数据位于 `runs/pig/round1_control1_10s/`。两套GT配置按用户原用途保留；本次代码验证不要求统一两套配置。

## 已实现

3D/1D 多分辨率 hashgrid、三路时空融合、拼接 shortcut 预测器、独立双向 DVF、TP 与 FF 双特征、双输出投影监督、排序复数 FFT 距离、cycle Jacobian 正则、仅前向时间二阶正则、仅前向参数更新的图像配准、世界毫米单位射线积分。包含训练、断点恢复、任意时刻体数据与 DVF 导出。

## 快速运行

在本目录打开 PowerShell 或终端。已有依赖时无需重复安装。

```powershell
python -m pip install -r requirements.txt
python -m bird.demo --out runs/my_demo.npz
python -m bird.train --data runs/my_demo.npz --config configs/smoke.json --out runs/my_smoke --device cpu --threads 2
python -m bird.infer --checkpoint runs/my_smoke/checkpoint.pt --out runs/my_smoke/volume.npz --times 0 0.5 1 --resolution 24 24 24 --chunk 2048 --threads 2
python -m bird.evaluate --checkpoint runs/my_smoke/checkpoint.pt --data runs/my_demo.npz --out runs/my_smoke/projection_fit.json
```

`my_demo/my_smoke` 为新运行目录名，避免覆盖本次已验证的 `runs/demo_data.npz` 与 `runs/smoke/`。生成器和推理拒绝覆盖已有输出，训练拒绝覆盖已有 checkpoint。

运行测试：

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = "1"
python -m pytest -q -p no:cacheprovider
```

环境变量只用来隔离机器上与本项目无关的 pytest 插件。常规干净环境可直接 `python -m pytest -q`。

## 配置和真实投影

| 配置 | 用途 |
|---|---|
| `configs/smoke.json` | CPU 流程验证：3级编码、100步、64射线、24采样/射线、每项正则32点 |
| `configs/paper.json` | 期刊公开参数及明确补全：16级、宽32、2048射线、64000步、lambda_TP=.5 |
| `configs/xcat.json` | lambda_TP=.75，空间384×384×320、时间540 |
| `configs/clinical_simulation.json` | lambda_TP=.1，空间512×512×293、时间2500 |
| `configs/verify_tigre.json` | TIGRE 16级/宽32、64×96×96，20步完整短周期验证；不是论文64000步实验 |
| `configs/paper_cuda_chunks.json` | 原paper配置仅扩大执行分块；RTX4070Ti SUPER上完整优化步已验证 |

后两者只是网络/训练设置模板，不包含原论文数据。改变空间网格、时间采样数、毫米范围或强度单位会影响有限差分和正则尺度，须按 [SPEC.md](SPEC.md) 核对。`samples_per_ray=512`、中间维数、激活等是补全假设，不是作者确认值。

真实数据保存为 NPZ，支持平面探测器几何或通用逐射线坐标，字段见 spec 第3节。输入是负对数投影 `-log(I/I0)`，不要直接输入光子计数、PNG 或重建 HU。scanner 几何、暗场/平场、坏点、散射、束硬化等预处理由数据提供方完成；这里不推测私有 scanner 格式。盒范围必须包含待解释的全部衰减物体。

```powershell
python -m bird.train --data path/to/scan.npz --config configs/paper.json --out runs/scan01 --device cuda
```

此命令需要可用 CUDA 版 PyTorch。最初验证使用系统 CPU 环境；随后发现并验证了上述独立 CUDA/TIGRE 环境，见 PIG_VALIDATION。纯 PyTorch 编码和有限差分适合研究参考，完整临床规模运行会明显慢于论文的 tiny-cuda-nn 实现；论文的约6–8小时速度不能直接套用。

## 断点恢复与内存

```powershell
python -m bird.train --data runs/my_demo.npz --config configs/smoke.json --out runs/interrupted --stop-after 50
python -m bird.train --data runs/my_demo.npz --config configs/smoke.json --out runs/interrupted --resume
```

恢复时校验配置和数据 SHA-256；恢复模型、Adam、cosine scheduler、CPU/CUDA RNG 与抽样 generator。`--stop-after` 不修改总学习率周期。崩溃后日志超过 checkpoint 的部分会先留存 `history.before-recovery-*.jsonl`，再移除未提交记录。

训练用 `ray_chunk` 和 `reg_chunk` 控制激活内存；射线 checkpoint 重算保留完整 batch FFT。保持总 batch 不变时可缩小 chunk；随之变化的随机正则分块抽样顺序不承诺逐位相同。不要用“每个 chunk 分别 FFT”替代完整 batch 损失。

NPZ 载入占用主存；推理虽然按 chunk 生成坐标，仍同时保存输出数组。`mu`、`mu_tp`、三分量 DVF 合计约每体素20字节，不含压缩临时内存。高分辨率多相位请每次只导出一个 `--times` 时刻；当前不是超大体数据流式存储实现。

## 输出内容与评价

- `checkpoint.pt`：模型与优化状态，仅加载可信的项目 checkpoint；使用 `weights_only=True`。
- `resolved.json`：实际配置、扫描边界/时间、数据 SHA-256、运行环境。
- `history.jsonl`：四项损失、总损失、每步学习率和耗时。
- 推理 NPZ：`mu[T,Z,Y,X]`、`mu_tp[T,Z,Y,X]`、`backward_dvf_mm[T,Z,Y,X,3]`、`times_s`、`bounds_mm`、`voxel_size_mm`。衰减单位 mm^-1，DVF 最后一维是 x/y/z。不是 HU；转换 HU 需要经标定的 mu_water。
- `bird.evaluate`：比较同一扫描全部射线的初始/训练后 RMSE 与 MAE，仅为训练投影拟合诊断。
- `bird.evaluate.volume_metrics(prediction,target,data_range,ssim=True)`：输入 `[Z,Y,X]`，显式指定动态范围，报告 MAE/RMSE/PSNR 和轴向切片平均 SSIM。SSIM 使用 scikit-image 7×7 默认窗口，不声称与作者未公开的 SSIM 参数逐项相同。多相位逐相位调用后再汇总；ROI 与对齐规则应独立声明。

## 目录

```text
bird/
  config.py       参数及验证
  encoding.py     1D/3D hashgrid、时空融合
  model.py        DVF、双图像头、预测器
  geometry.py     盒相交、中点积分、分块查询
  losses.py       双投影监督和三个运动约束
  data.py         NPZ 数据与几何适配
  train.py        单病例优化和恢复
  infer.py        任意时间体数据推理
  demo.py         非周期合成三维场景
  evaluate.py     投影拟合与体图像指标
  io.py           输出文件路径规范化
  pig_cbct.py      按本地猪心脚本生成TIGRE测量和合同
  tigre_backend.py TIGRE Ax/matched Atb、自定义autograd和分块VJP
  train_tigre.py   完整体网格的TIGRE训练入口
  infer_pig.py     按当前GT生成几何导出raw mu
  check_tigre.py   GPU投影/近似伴随检查
  audit_pig_gt.py  现存GT元数据兼容性审计
configs/          论文、合成验证及猪心TIGRE smoke/pilot配置
tests/            数学、梯度、几何、恢复和错误边界测试
runs/             本次运行结果；不属于原论文数据
sources/          本地阅读用提取资料；不收入纯代码压缩包
```

`BIRD_reproduction_TIGRE.zip` 是更新后的代码/规格/小型验证记录包；不含原始猪CT、生成的投影大文件、体数据、checkpoint或论文全文。完整生成结果保留在本机 `runs/pig/`。

再次核验后的最新包是 `BIRD_reproduction_verified.zip`，包含新增独立oracle测试、运行证据与复核报告；同样不打包大型数据和权重。当前纯PyTorch实现的实际速度与论文不同，详见 REPRODUCTION_AUDIT。
