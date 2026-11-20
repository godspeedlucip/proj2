# RSTAR4D 完整训练工程

## 猪心轮廓消融（`rstar-advance`）

- 用 `inventory-pig` 生成 `series.jsonl`，按真实相位核对 mask。标注声明为 JSONL，每行用 `case_id`、`animal_id`、`phase_label` 和 `structure` 定位结构。结构取 `heart_outer` 或 `lv`。每行显式填写 `available`。确认可用后提供 `trusted_domain` 作为二值 NIfTI，或填写 `whole_fov_confirmed: true`。可选字段 `mask` 覆盖同系列默认 mask，`correspondence_domain` 指定可靠的双端对应区域，`closed` 允许对闭合轮廓报告体积。
- 用 `python -m rstar4d prepare-cardiac-annotations --inventory INVENTORY --manifest PAIRED_MANIFEST --annotations DECLARATIONS --split SPLIT --out PREPARED` 构建带 SHA256 的边界缓存和覆盖表。导入的旧 NIfTI 若缺 TIGRE 源网格信息，在声明中提供从输出 ZYX 索引到 NIfTI 世界坐标的 `output_to_world` 矩阵。已经对齐的 `roi_labels` 可直接使用。
- 用 `python -m experiments.cardiac_ablation write-matrix --base configs/train/pig_cardiac_ablation.yaml --out CONFIG_DIR` 生成本轮八组配置，`write-controls` 派生机制对照。用同一 Stage I checkpoint 分别运行 `train --stage 2 --init-from-stage1`。两个心脏消融基础 YAML 现均使用论文的实际输入尺寸，按 Z×Y×X 等概率选择 16×256×256、32×144×144 或 64×128×128，保留该扫描全部相位。已有运行和此前导出的 YAML 保留当时的配置。整轮显存取决于相位数与模型分支，可用 `benchmark --stage 2 --manifest MANIFEST --config CONFIG --memory-limit-gib 13.5` 实测。
- Stage II 的训练位置按 70% 轴向中部、25% 全视野均匀、5% X 末端贴边采样。中部在 X、Y 合法起点范围的中间一半内取整点，Z 均匀；末端分支只固定 X。三项是选择方式的概率，同一个物理位置可以被多种方式抽到。各组共享随机种子、位置及尺寸，重试保留原分支。固定验证块使用全视野均匀采样。`cardiac_samples.jsonl` 记录每次采用的 `sampling_mode` 和实际 `window_shape`。
- `fixed_input` 模式下，主干和 M/P 分支都只读指定尺寸。M 的训练损失可在块外读取参考灰度、有效域及启用的表面标签，以评价跨过裁切面的位移；这些数据仅用于计算训练损失。真实扫描外的位移沿用原来的越界处理。
- `infer` 按 checkpoint 恢复 M/P 模式和滑窗上下文。对每组重建用同一冻结分割器或盲法勾画。评价清单每行提供 `case_id`、`phase_label`、`structure`、`prediction_sha256`、独立二值 ZYX HDF5 的 `path`、`sha256`、`dataset` 和 `delineation`。冻结分割器的 `delineation` 含 `method: frozen_segmenter`、`version`、`weights_sha256`、`threshold`，盲法用 `method: blinded` 与 `batch`。
- 在指标 YAML 的 `contours.surface_dice_tolerance_vox` 中为两个结构填写锁定容差，用 `evaluate --contours CONTOURS --metric-config METRICS` 输出轮廓指标。最后用 `python -m experiments.cardiac_ablation compare --runs RUNS --out REPORT` 按动物报告配对差值。
- `angle-layouts` 从同一混合扫描按两个等量角度子集重建一个状态，并分别导出实际几何和交换几何诊断清单。`diagnose-prior` 接收验证病例对照清单，每行含 `case_id`、`animal_id`、`path`、`sha256`，指向同一网格的 NPZ，含已归一化的 `dynamic[T,Z,Y,X]`、`matched[Z,Y,X]`、`mismatched[Z,Y,X]`、`source_fov[T,Z,Y,X]`、`prior_fov[Z,Y,X]` 及几何模式的 `geometry[T,14]`。

[设计规格](docs/superpowers/specs/2026-11-18-rstar4d-cardiac-contour-ablation-design.md) · [实施计划](docs/superpowers/plans/2026-11-18-rstar4d-cardiac-contour-ablation.md) · [训练、续训及八组体模验收记录](docs/reviews/2026-11-18-rstar-advance-training-acceptance.md)

`angle-layouts` 同时导出 `manifest-joint.jsonl`，其中 A、B 都使用各自实际投影的角度摘要。两行共享 `animal_id` 和原始 `case_id`，各有唯一 `sample_id`，推理输出使用样本 ID 命名。多布局预测的独立轮廓清单也要填写对应的 `sample_id`。将联合清单中的训练样本与独立验证动物一起写入训练清单，split 仍按动物指定。训练按动物、原病例、布局依次采样，同病例增加布局时维持每轮预算。

`metrics_by_phase.csv` 保存每相位的全图指标和有效分母；`compare` 的 `paired_phase_summary.csv` 和 `paired_by_phase.csv` 分别列出逐相位配对的汇总和明细。组间默认在相同布局上配对，显式指定 `--layout-pair B A` 可比较候选组 B 布局与参考组 A 布局。同一相位只有在双方该项指标都有定义时才进入有效相位数。旧评价目录只有病例级指标时，相位数为 `null`，可复用原预测重新执行 `evaluate`。

匹配与错配 prior 的诊断按相位报告从有效目标、单粗候选、正细采样支撑、等细点数到值幅匹配的留存数。可以在下文的八组体模运行结束后执行 `python -m validation.prior_controls --smoke runs/cardiac-demo --out runs/cardiac-prior-demo`，对比原零点体模与具有连续插值支撑的正对照。两组门控和修正量的差值保留实际方向。

本工程以公开Notebook为算法依据：原网络/state_dict、circular时间填充、Stage I跳过z、Stage II原Dirac/零初始化、L1＋0.1×(1−SSIM)、Adam 1.5e-4、60/30轮及原分段学习率。原Notebook和权重保留在 `RSTAR/githubcode/`。

猪数据使用TIGRE；HU→mu、几何偏移、FDK、padding、FOV和有效掩膜沿用用户生成代码，副本及SHA256记录在 `rstar4d/provenance.json`。运行时不依赖工作区外的Python脚本。猪域强度固定除以0.02、16GB配置缩小空间块，均为明确适配。当前默认prior按公开MATLAB的相位视角数加权FDK；`prior_mode: all_views_fdk`可选，对应用户原prior生成方式。两种prior均只使用动态混合投影。

## 1. 环境与检查

当前已经验证的Python：

```powershell
$py = 'D:\python\miniconda\env\env\common\python.exe'
& $py -m rstar4d doctor --out runs/doctor.json
& $py -m rstar4d --help
```

在项目根目录可以直接运行，无需修改外部Conda环境。也支持在选定环境中 `python -m pip install -e .`。新环境参考 `environment-train.yml` / `environment-sim.yml`；TIGRE需另行从 [CERN TIGRE](https://github.com/CERN/TIGRE) 构建，不能用同名无关的 `pip install tigre`。训练环境可独立于仿真环境；纯训练检查用 `doctor --skip-tigre`。未提供CUDA时仿真明确失败，训练可显式设置 `device: cpu, amp: false`。

## 2. 先跑真实算子的完整小体模链路

下列命令生成3个独立合成动物，包含T10/T11、实际TIGRE投影和FDK，再从随机初始化训练完整网络。输出目录必须为空；不会覆盖已有产物。

```powershell
& $py -m rstar4d make-phantom --out data/smoke_raw --views 240
& $py -m rstar4d simulate-pig --manifest data/smoke_raw/raw_manifest.jsonl --config data/smoke_raw/simulation.yaml --out data/smoke_prepared
& $py -m rstar4d validate-data --manifest data/smoke_prepared/manifest.jsonl --split data/smoke_raw/split.json
& $py -m rstar4d pipeline --config configs/train/smoke.yaml --manifest data/smoke_prepared/manifest.jsonl --split data/smoke_raw/split.json --run-dir runs/smoke
& $py -m rstar4d infer --checkpoint runs/smoke/stage2/checkpoints/best.pt --manifest data/smoke_prepared/manifest.jsonl --tile 8 32 32 --overlap 4 16 16 --out runs/smoke/predictions --export h5 tif nii
& $py -m rstar4d evaluate --predictions runs/smoke/predictions --manifest data/smoke_prepared/manifest.jsonl --metric-config configs/metrics/pig_attenuation_v1.yaml --out runs/smoke/evaluation
```

每阶段8个draw，给原GradScaler的初始尺度校准留出空间；它只验证程序闭环，不代表训练收敛或真实猪数据效果。默认数据单位是mm⁻¹。结果有HDF5、逐相位TIFF/NIfTI、带固定窗的 `comparison.png`、JSON和CSV指标。NIfTI sidecar明确array ZYX→scanner ZYX约定，不冒称动物LPS/RAS坐标。

## 3. 从原始猪CT生成训练数据

```powershell
& $py -m rstar4d inventory-pig --data-root 'D:/desktop/课题2/data/pig' --settings-root 'D:/desktop/课题2/get_data/settings/heart_only' --out data/pig_inventory
& $py -m rstar4d validate-settings --config configs/simulation/pig_heart_post_fdk.yaml --inventory data/pig_inventory/series.jsonl
& $py -m rstar4d simulate-pig --config configs/simulation/pig_heart_post_fdk.yaml --manifest data/pig_inventory/series.jsonl --limit 1 --out data/pig_prepared
```

先读取 `data/pig_inventory/summary.json`。目录里出现多个动物ID的系列需要提供CSV `identity_map`，列为 `series_relative_path,animal_id`，通过 `inventory-pig --identity-map ...` 使用；不自动把最外层目录当成动物。`--case-id`可重复指定精确case_id。移除 `--limit 1` 可生成全部所选病例。源图像逐相位加载，投影默认每64角一批，600views/300°等参数以逐系列CSV为准。

模拟GT是该状态全部视角FDK，输入是该状态分配视角FDK；可选Poisson噪声只作用于混合扫描并由CBCT/prior共用。`calibration.mode: post_fdk`将同一多项式施加于重建图像，精确0背景保持0。无校准使用 `configs/simulation/pig_no_calibration.yaml`。

缓存绑定源内容、有效配置、cycle、校准、实现与TIGRE后端指纹。完整病例原子提交；失败保留 `<case>.partial/error.json`，需要检查后换输出目录或手工处理该失败产物。配置/源内容改变时不会覆盖旧缓存。`generation_summary.json`记录本次请求病例集与已完成病例；扩展生成失败会保持`complete:false`。

## 4. 已有配对数据入口

如果已经有规范paired HDF5，建立JSONL，每行：

```json
{"case_id":"case_001","animal_id":"animal_01","paired_h5":"cases/case_001.h5"}
```

```powershell
& $py -m rstar4d import-paired --manifest data/paired_sources.jsonl --out data/imported
& $py -m rstar4d validate-data --manifest data/imported/manifest.jsonl --split data/animal_split.json
```

TIFF/NIfTI等图像入口使用 `cbct`、`target`、`valid_mask` 的逐相位路径列表，以及单个 `prior` 路径；HDF5单数组用 `{"path":"case.h5","dataset":"ct"}` 明确映射旧键。另提供 `metadata` 和 `source_metadata`（分别包含cbct/prior/target/valid_mask的完整空间、相位和强度合同），不能从文件名或shape猜测。完整示例见 `examples/paired_sources.example.jsonl` 与 `examples/paired_metadata.example.json`。示例中的64×256×256网格仅演示格式，必须填写实际shape、spacing、中心和4×4 voxel-to-scanner affine。TIFF的几何信息必须由sidecar或显式参数补充。已有故意不同配置的旧产物无需用作这一入口的训练对。

默认保留GT网格。如明确需要统一重采样，在来源行增加 `"resample":{"shape":[Z,Y,X],"spacing":[dz,dy,dx]}`；输入、prior、GT与mask使用同一中心/物理变换，ROI用最近邻，mask要求全部插值来源有效。新元数据记录变换，FOV物理投影资格保守关闭。

prepared HDF5数组：`cbct/target/valid_mask [T,Z,Y,X]`、`prior [Z,Y,X]`，可选整数 `roi_labels`；强度/spacing/相位标签/校准/来源等存于 `metadata_json` 属性。无GT推理可省target，但训练必须有target与有效掩膜。`manifest.jsonl`包含内容SHA256；修改数据后必须新建数据快照，不能直接续训旧运行。

## 5. 训练和续训

划分文件按动物ID，不按slice或phase：

```json
{"train":["animal_01","animal_02"],"val":["animal_03"],"test":["animal_04"]}
```

同动物所有轮次、姿态必须落在同一split。inventory生成的是待编辑模板（初始全部列入train），正式训练需要明确的非空train和val。用户决定真实动物划分；程序不会替研究者随意划分。

```powershell
& $py -m rstar4d pipeline --config configs/train/pig_local_16gb.yaml --manifest data/pig_prepared/manifest.jsonl --split data/animal_split.json --run-dir runs/pig_001
& $py -m rstar4d train --resume runs/pig_001/stage2/checkpoints/last.pt
```

`pipeline`自动Stage I best→Stage II初始化→Stage II训练。对同一run重新运行会恢复已完成epoch，不会重复初始化z或覆盖已完成Stage II。单阶段使用 `train --stage 1`；Stage II指定 `--init-from-stage1 ...` 或 `--init-from-stage2 ...`，二者与 `--resume` 互斥。公开裸state_dict只可用于显式初始化，不能作为resume。

两个 Stage 都在每轮完成训练和验证后保存独立的完整归档 `checkpoints/epoch-0001.pt`、`epoch-0002.pt` 等。`last.pt` 用于继续最新训练，`best.pt` 对应验证病例等权 L1 最低的轮次。归档含权重、Adam、学习率调度器、AMP GradScaler、随机状态、已完成轮次、累计更新数、固定验证样本及数据、划分和配置签名。从较早轮次继续时指定新的实验目录。例如从第 3 轮的状态启动第 4 轮：

```powershell
& $py -m rstar4d train --resume runs/pig_001/stage2/checkpoints/epoch-0003.pt --run-dir runs/pig_stage2_from_epoch3
```

恢复以完整轮次为边界。第 8 轮中途退出时，第 8 轮从第一个样本重算。迁移训练时复制整个 Stage 目录。恢复会核对训练身份、历史最优、数据签名和算法版本，确定性 CPU 设置可用于逐值核验连续与恢复训练。

### TensorBoard 观测

正常训练默认启用 TensorBoard，并在每个 Stage 或消融组的 `tensorboard/` 下发布逐轮事件。可用合成体模观察完整流程。在仓库目录运行：

```powershell
& $py -m validation.cardiac_smoke --out runs/cardiac-demo
& $py -m tensorboard.main --logdir runs/cardiac-demo
```

使用新的体模输出目录。TensorBoard 的 Scalars 支持叠加 baseline、M、P、B 和联合组的同名曲线。横轴是完成的 epoch。`train/`、`validation/`、`optimizer/` 和 `resources/` 显示损失、有效更新、AMP 溢出、学习率及资源使用。`loss/` 显示按有效支撑汇总的辅助损失，配合 `support/` 中的窗口数、命中块数和体素数观察；无有效监督时只报告计数。Images 固定验证块、相位、切片及窗宽，每轮显示稀疏图、prior、预测、参考图和绝对误差。可信表面存在时在预测上以橙色绘整心外表面、青色绘 LV。独立轮廓评价在 `evaluation/tensorboard/` 写入各结构指标和每种指标的有效动物数。

同一轮的事件文件随完整 checkpoint 发布，续训时可从归档补回缺失事件。关闭训练可视化可设置 `tensorboard: false`。训练日志、事件文件和模型权重均写到本地 run 目录。

Stage I默认每epoch的draw数为训练病例z层数之和，Stage II为15×病例数；先均匀选动物，再选病例和空间位置。全部T保留，batch=1。病例顺序规范化，manifest重排不改变抽样。每个draw固定动物/病例/块尺寸，只重试空间位置；32次仍不满足有效比例则报错，不静默跳过动物。验证使用固定样本，best按病例等权L1选择，同时记录动物等权L1。`small_volume: error`默认拒绝过小体积；显式`pad`只在高索引端复制填充输入，并将padding的valid设为0。

原SSIM的Stage I/II张量布局和data_range=255被保留；T=10/11时库对时间维的处理可能不同，这属于公开代码行为。部分有效掩膜时使用同一SSIM公式并只统计全有效窗口，invalid不参与梯度。fp16溢出采用原GradScaler跳步/降尺度并逐次记录；全epoch无参数更新、非有限loss/fp32梯度直接报错。真实T11样本初始尺度可能需要超过8个draw才稳定，smoke配置仅用于小体模；真实训练使用正式配置的完整epoch。整epoch无有效SSIM窗口也报错。可显式选 `loss.kind: l1` 或 `physical_ssim` 做消融，但不是默认公开训练。

## 6. 推理、指标和显存

```powershell
& $py -m rstar4d infer --checkpoint runs/pig_001/stage2/checkpoints/best.pt --manifest data/pig_prepared/manifest.jsonl --mode tiled --out runs/pig_001/predictions
& $py -m rstar4d evaluate --predictions runs/pig_001/predictions --manifest data/pig_prepared/manifest.jsonl --out runs/pig_001/evaluation
& $py -m rstar4d benchmark --config configs/train/pig_local_16gb.yaml --stage 2 --phase-count 11 --weights RSTAR/githubcode/checkpoints/checkpoints_Tetris2.pth --out runs/benchmark.json
```

默认只预测checkpoint划分的test动物；其他数据可显式传 `--split ...` 或 `--split-name all`。默认滑窗32×128×128、overlap16×64×64，保留全T，使用严格正Gaussian权重融合；xy起点对齐8、Stage II z对齐2。奇数尺寸通过补齐并回裁。`--mode full`是显式整幅模式，OOM不会自动更改形状或切换模式。

指标在原强度域计算MAE/RMSE；SSIM使用固定[0,0.05]窗、11×11 Gaussian、sigma1.5、population covariance。ROI名称由metrics配置 `roi_labels: {'1': 'heart'}` 指定，不套用论文肺/肿瘤名称。报告同时包含输入和预测、病例等权/动物等权均值及样本SD；空ROI和缺GT为null及原因。

`evaluate --projection`仅对未校准mu、保存了混合投影、且输出FOV已由源非零支持域验证完整时运行。重投影使用输出网格，不重复添加source offOrigin；其他情况明确返回not_applicable。投影指标单独按病例/动物等权汇总，包含不适用原因、常量视角计数和null。论文表格的原始病例/ROI和数值不在本工程交付中。

## 7. 验证

```powershell
& $py -m unittest discover -s tests -v
```

测试覆盖公开权重、前向和损失对照、动物划分、mask、T10/T11、TIGRE 投影与 FDK、两条 CLI 入口、续训、滑窗和指标。CUDA 不可用时会跳过 GPU 测试。本机此次 88 项通过、0 项跳过，详见[训练可靠性与体模验收记录](docs/reviews/2026-11-18-rstar-advance-training-acceptance.md)。早期基线见[实现验收报告](docs/reviews/2026-11-16-implementation-acceptance.md)。可用 `python -m validation.run_delivery_checks --out runs/my_acceptance` 执行并保存两条入口的小体模产物。

配置YAML内路径相对YAML所在目录，manifest路径相对manifest所在目录，CLI路径相对当前工作目录。未知配置字段、损坏权重、缺失元数据、非有限值会明确失败。
