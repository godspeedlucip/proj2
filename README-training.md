# RSTAR4D 完整训练工程

## 猪心轮廓消融（`rstar-advance`）

- 用 `inventory-pig` 生成 `series.jsonl`，按真实相位核对 mask。标注声明为 JSONL，每行用 `case_id`、`animal_id`、`phase_label` 和 `structure` 定位结构。结构取 `heart_outer` 或 `lv`。每行显式填写 `available`。确认可用后提供 `trusted_domain` 作为二值 NIfTI，或填写 `whole_fov_confirmed: true`。可选字段 `mask` 覆盖同系列默认 mask，`correspondence_domain` 指定可靠的双端对应区域，`closed` 允许对闭合轮廓报告体积。
- 用 `python -m rstar4d prepare-cardiac-annotations --inventory INVENTORY --manifest PAIRED_MANIFEST --annotations DECLARATIONS --split SPLIT --out PREPARED` 构建带 SHA256 的边界缓存和覆盖表。导入的旧 NIfTI 若缺 TIGRE 源网格信息，在声明中提供从输出 ZYX 索引到 NIfTI 世界坐标的 `output_to_world` 矩阵。已经对齐的 `roi_labels` 可直接使用。
- 用 `python -m experiments.cardiac_ablation write-matrix --base configs/train/pig_cardiac_ablation_16gb.yaml --out CONFIG_DIR` 派生八组，`write-controls` 派生机制对照。用同一 Stage I checkpoint 分别运行 `train --stage 2 --init-from-stage1`。CUDA 试配用 `benchmark --stage 2 --manifest MANIFEST --config CONFIG --memory-limit-gib 13.5`，观察实际 C/W 和监督覆盖。原设计尺寸保存在 `configs/train/pig_cardiac_ablation.yaml`。
- `infer` 按 checkpoint 恢复 M/P 模式和滑窗上下文。对每组重建用同一冻结分割器或盲法勾画。评价清单每行提供 `case_id`、`phase_label`、`structure`、`prediction_sha256`、独立二值 ZYX HDF5 的 `path`、`sha256`、`dataset` 和 `delineation`。冻结分割器的 `delineation` 含 `method: frozen_segmenter`、`version`、`weights_sha256`、`threshold`，盲法用 `method: blinded` 与 `batch`。
- 在指标 YAML 的 `contours.surface_dice_tolerance_vox` 中为两个结构填写锁定容差，用 `evaluate --contours CONTOURS --metric-config METRICS` 输出轮廓指标。最后用 `python -m experiments.cardiac_ablation compare --runs RUNS --out REPORT` 按动物报告配对差值。
- `angle-layouts` 从同一混合扫描按两个等量角度子集重建一个状态，并分别导出实际几何和交换几何诊断清单。`diagnose-prior` 接收验证病例对照清单，每行含 `case_id`、`animal_id`、`path`、`sha256`，指向同一网格的 NPZ，含已归一化的 `dynamic[T,Z,Y,X]`、`matched[Z,Y,X]`、`mismatched[Z,Y,X]`、`source_fov[T,Z,Y,X]`、`prior_fov[Z,Y,X]` 及几何模式的 `geometry[T,14]`。

[设计规格](docs/superpowers/specs/2026-11-18-rstar4d-cardiac-contour-ablation-design.md) · [实施计划](docs/superpowers/plans/2026-11-18-rstar4d-cardiac-contour-ablation.md)

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

Checkpoint包含模型、Adam、scheduler、GradScaler、RNG、固定验证块、数据/划分指纹和配置。保证epoch边界的CPU确定性恢复；中途batch中断会从上个完整epoch开始。不保证跨GPU/软件版本逐位相同。除device/workers/pin_memory/log_every等运行选项和经哈希匹配的路径迁移外，改变算法或数据会拒绝resume。恢复必须指向目标run自己的`checkpoints/last.pt`，并匹配`run_identity.json`与历史`best.pt`；迁移时复制整个run目录。不能把单个checkpoint恢复到另一非空run。需要修改实验时选择新run并显式warm-start。

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

不依赖pytest。测试包含公开权重/前向/loss对照、动物泄漏、mask、T10/T11、真实TIGRE、两条入口的完整CLI流程、续训一致性和滑窗/指标。CUDA不可用时仅GPU测试跳过，不能据此声称TIGRE验证通过。本机36项测试通过且无跳过。具体范围见 [实现验收报告](D:/desktop/课题2/proj/RSTAR4D/docs/reviews/2026-11-16-implementation-acceptance.md)。可用 `python -m validation.run_delivery_checks --out runs/my_acceptance` 一次执行并保存两条入口的完整小体模产物。

配置YAML内路径相对YAML所在目录，manifest路径相对manifest所在目录，CLI路径相对当前工作目录。未知配置字段、损坏权重、缺失元数据、非有限值会明确失败。
