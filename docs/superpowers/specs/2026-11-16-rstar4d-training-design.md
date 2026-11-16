# RSTAR4D 完整仿真与网络训练系统 Spec

> 本文件为历史 v1，已被 [v2 猪数据与 TIGRE 规格](D:/desktop/课题2/proj/RSTAR4D/docs/superpowers/specs/2026-11-16-rstar4d-pig-tigre-training-design.md) 替代。后续实施只以 v2 为准；本页的 ASTRA、固定10相位、HU标签和统一90层网格不再适用于当前用户任务。

版本：1.0 · 日期：2026-11-16 · 状态：已完成设计与可实施性审查，尚未实现本 spec 中的新代码；除用户明确的范围外，默认技术选择均为本文提出的工程约定。

## 1. 目标与交付边界

交付一个独立于 Notebook 的 Python 工程，支持两条入口，并汇合到相同的训练、推理和评估流程：

1. 原始 10 相位 4D CT + 呼吸分相信号 → 动态锥束投影仿真 → 噪声 → 分相 FDK / 平均图像 → 配对数据 → Tetris Stage I → Stage II → 推理 → 评估。
2. 已有配对 CT、CBCT、prior → 格式检查与转换 → 同一套两阶段训练、推理、评估。

“原始 4D CT 仿真”和“配对数据训练”均为用户明确要求的必交付功能，不得只实现其中一条。没有真实数据时，系统必须能生成带已知运动的合成体模，跑通整个链路；这属于软件验收，不算复现论文医学效果。

本项目首先交付完整、可追溯、可继续训练的研究代码。保留公开网络的计算行为与权重兼容性，补全数据流和实验管理。默认支持单张 NVIDIA GPU；训练支持 CPU 小尺寸检查，ASTRA 仿真要求 CUDA。新代码不得覆盖现有 Notebook、MATLAB 脚本或两个公开权重。

不把缺少的患者数据、扫描仪校准和论文离线评估流程假设为已知。系统不自动下载院内数据，不承诺恢复论文全部表格数值。临床原始投影的 Amsterdam Shroud 呼吸提取、商用扫描仪半扇束校准、其余论文基线的重实现、分布式多 GPU 训练不属于第一版交付范围。

## 2. 设计依据、方案与复现口径

### 2.1 已确认的事实

- 本地论文为 arXiv v4，2024-09-29；公开代码标注后续 TRPMS 版本。
- 模型使用 xy / z / t 三分支相加、连续两组的卷积模块，时间循环填充，U-Net 编解码与残差连接。
- Stage I 跳过 z 分支；Stage II 启用 z 分支，首层 z 下采样 2 倍、末层最近邻恢复 2 倍。
- 两份公开权重各有 232 个 state-dict 条目；现有模型总参数量为 2,118,693。论文表 4 为 2.35 M，这一差异保留为版本差异记录。
- 原代码已通过小尺寸 CPU 前向与 GPU 单步训练检查，尚未验证原始病例完整训练或整幅推理。
- 仿真脚本输出 `416×416×180×10`；训练期望更小尺寸，且脚本没有完成四维真值汇总、训练文件保存与两阶段数据准备。
- 作者确认公开仿真省略探测器偏移；最终 SSIM 使用 MATLAB 离线计算，不等于训练日志 SSIM。[仿真说明](https://github.com/ZihengD/RSTAR/issues/6#issuecomment-4022042854)、[指标说明](https://github.com/ZihengD/RSTAR/issues/2#issuecomment-3771558208)。

### 2.2 采用的方案及替代方案

本设计建议采用 **Python/PyTorch 训练 + Python ASTRA CUDA 仿真 + HDF5 中间数据**。这是设计者的工程选择，用户已确认完整链路范围，未指定具体库版本。

| 方案 | 优点 | 代价 | 决策 |
|---|---|---|---|
| MATLAB 仿真 + Python 训练 | 与原仿真脚本接近 | 双语言与许可证依赖，仍需补格式转换 | 原脚本保留作参照，不作为主入口 |
| Python ASTRA + PyTorch | 单一代码语言、可自动测试、可分环境运行 | 需核对轴顺序、物理单位、重建尺度 | 采用 |
| 重写新架构或直接换 3D 网络 | 更自由地优化显存 | 失去公开权重和方法对应关系 | 第一版不采用 |

### 2.3 三种实验口径

必须分别记录 `model_profile`、`simulation_profile`、`training_profile`、`metric_profile`，不能仅以一个“reproduce=true”代替。

- `official_v1`：网络结构与公开权重兼容；只证明结构/前向兼容。
- `paper_l1`：遵循论文 L1 目标，论文未写明的调度与阶段轮数使用本 spec 的显式工程默认值；名称和报告不能写成“精确论文配置”。
- `public_code`：尽量复刻 Notebook 的训练超参数和旧 SSIM 行为，服务于对照；仍不能声明完整实验重现。
- `local_16gb`：从 `paper_l1` 派生，缩小 Stage II 训练块；报告必须显示块尺寸变化。

所有运行都生成 `provenance.json`，区分 `source_reported`、`source_code_observed`、`engineering_default`、`user_override` 四种参数来源。不得把未知的半扇束偏移量、旧权重强度归一化或数据划分填成“论文默认”。

## 3. 完成定义

交付物必须包括：可安装 Python 包、仿真与训练 CLI、配置与 schema、数据导入器、公开权重转换器、两阶段训练器、断点续训、推理与指标、体模、必要测试、安装说明和运行示例。

软件完成须同时满足：

1. 给定合成体模和固定 seed，自动生成 10 相位真值、投影、CBCT、prior 和元数据。
2. 原始数据入口与配对数据入口生成同一种规范化磁盘格式。
3. 两阶段独立运行和自动衔接都可用；训练产物真实保存，重新启动后可恢复。
4. 加载公开权重采用严格 key/shape 检查；CPU 前向与原 Notebook 数值对齐。
5. 原始病例在 train/val/test 间无患者交叉；呼吸信号和仿真派生关系可审计。
6. 推理输出 HU 图像和物理元数据；有真值时生成病例级及总体指标。
7. 没有真值或 ROI 时明确输出缺失原因，不伪造标签或指标。
8. 完整仿真冒烟测试和单病例过拟合测试有日志、配置及输出证据。

真实数据的效果验收单独进行：至少在固定留出病例上比较输入 FDK 与模型输出，并报告所有病例，不预设必须达到论文 SSIM 数值。

## 4. 技术栈、平台与环境

第一版支持 Windows PowerShell 和 Linux。路径包含中文、空格时必须可运行；禁止硬编码用户盘符、环境路径和 `cuda:0`。

训练基线依赖：Python 3.9–3.11、PyTorch 2.3.1、NumPy、h5py、PyYAML、SciPy、scikit-image、SimpleITK、tqdm；复刻旧损失使用 `pytorch-msssim==1.0.0`。TensorBoard 为可选 extra。新训练代码不依赖 torchvision、prefetch_generator 或 MATLAB。

仿真基线依赖：Python 3.10、`astra-toolbox==2.4.1`、NumPy、h5py、SimpleITK、PyYAML。后续版本需通过仿真标定测试后才能更新锁文件。

可以提供两个独立环境 `environment-train.yml`、`environment-sim.yml`，通过磁盘 HDF5 / manifest 交接；必须支持在两个进程中完成链路，不强迫两个 GPU 库同进程共存。统一环境可以作为附加安装方式，但须实际验证后再文档化。

本机已有环境分别确认过 PyTorch 2.3.1 + CUDA 11.8 和 ASTRA 2.4.1 CUDA 可用；尚未验证上述全部依赖在新环境共同安装。实施时锁定实际通过测试的 Python/包/CUDA/驱动版本并保存 `environment.json`。

`doctor` 必须检查导入、GPU 可用性、自由显存、磁盘空间、权重存在性；不因机器有 GPU 就断言训练环境的 torch 支持 CUDA。

## 5. 数据契约与患者划分

### 5.1 统一轴、坐标与强度

- 磁盘四维图像：`[T,Z,Y,X]`；三维 prior：`[Z,Y,X]`。
- 网络张量：`[B,C,T,Z,Y,X]`。默认 `T=10`，第一版模型配置拒绝其他 T。
- 物理坐标为 mm；图像元数据用 `origin_xyz_mm`、`spacing_xyz_mm`、`direction_xyz`，明确其是 xyz 而非数组 zyx。
- 导入时转为统一 LPS 方向、共用物理网格；不得仅凭 shape 相同就认为 CT/CBCT 配准一致。
- 磁盘强度统一为 float32 HU。进入网络前用 `x=(HU+1000)/1000`，输出反变换 `HU=1000*x-1000`。这属于新训练默认值，不是已证实的旧权重原始归一化。
- 不逐病例 min-max，不默认截断输出 HU，不根据验证/测试数据拟合归一化统计。
- 旧数据转换必须明确指定 `intensity_encoding=hu|hu_plus_1000|normalized_density` 和 `axis_order`；没有元数据时拒绝猜测。旧权重推理必须提供其归一化声明；若未经样例核实，输出标记 `normalization_verified=false`。

### 5.2 原始 CT 导入

必须支持以下输入：

1. 十个 3D NIfTI 文件，按 manifest 显式指定相位顺序。
2. 单个 HDF5，`ct` 数据集为四维，输入轴顺序和强度编码显式给定。
3. 十个 DICOM Series：manifest 显式列出相位和 SeriesInstanceUID；用 SimpleITK 读取并应用 DICOM 强度标定，拒绝混合序列和重复/缺失空间位置。

不按文件名猜测 DICOM 切片顺序或呼吸相位。相位数组的索引 0–9 对应指定的有序呼吸周期。方向、间距和视野不一致时执行记录在案的物理重采样；无法确定配准关系时终止导入。

**患者坐标到扫描坐标：** 默认扫描坐标的 +x/+y/+z 分别沿 LPS 的左/后/上方向；首相位重采样后的共同网格物理中心作为等中心，所有相位使用同一中心。manifest 可显式给出 `isocenter_lps_mm=[L,P,S]` 覆盖。默认关系为 `p_scan=p_lps-isocenter_lps`，不自动按每相位的体区/肿瘤质心重新居中。第一版不接受未实现的额外扫描旋转矩阵；若输入方向不一致，先完成 LPS 重采样。保存两坐标系间的4×4变换和每个网格 affine；ASTRA后端负责匹配其数组轴的正负方向，由非对称体模检查验证。

**视野与未知区域：** 正投影网格超出任一相位的有效原 CT 支持域时，默认 `outside_ct=error`。用户可显式选择 `outside_ct=air`，将该区域设为 -1000 HU 且 valid=0，记录填充体积和假设；不能把这一选择标为已知解剖真值。正投影网格若裁掉有效 CT 中 `HU>-950` 的区域，默认 `truncated_anatomy=error`，提示调整网格/等中心；只能显式设为 `allow` 作为截断实验并记录体素数。固定配置下不自动扩大网格、移动中心或替换几何。网络目标区包含无效 CT 时，后续 loss/metric 继续使用有效掩膜。

### 5.3 文件与字段

HDF5 分为 `data_kind=raw_ct|paired` 两类。原始 CT 导入产物只要求 `ct`、`valid_mask` 和元数据；下表为 `paired` 数据合同，不要求 raw_ct 提前存在 CBCT/prior。每个仿真扫描保存一个 `case_id.h5`，attributes 必须包括 `schema_version=1`、`data_kind`、`patient_id`、`case_id`、`split`、`phase_count`、`axis_order`、强度单位、物理网格、来源文件 SHA256、配置哈希、seed、呼吸信号 ID（非仿真配对数据显式为 null）。

| 数据集 | 形状 | 约束 |
|---|---|---|
| `ct` | `[10,Z,Y,X]` | 训练/验证必需，推理可缺失 |
| `cbct` | `[10,Z,Y,X]` | 必需，与 ct 对齐 |
| `prior` | `[Z,Y,X]` | 必需，且注明生成方式 |
| `valid_mask` | `[Z,Y,X]` 或 `[10,Z,Y,X]` | 必需；有效原始支持域为 1，填充/不支持区域为 0 |
| `body_mask` | `[10,Z,Y,X]` | 有 ct 时生成，定义为有效域内 `ct > -950 HU`，只用于明确命名的体区指标；无真值时可缺失 |
| `lung_mask`、`tumor_mask` | `[10,Z,Y,X]` | 可选；二值，必须记录来源与变换 |

动态肺/肿瘤掩膜必须逐相位提供；三维静态掩膜不能静默复制成动态真值。由 RTSTRUCT 导入的轮廓需先匹配参考序列，在图像物理空间栅格化。

投影单独存于 `case_id.projections.h5`：`line_integral [V,N,U]`、`angles_rad [N]`、`phase_id [N]`、扫描几何及原始/重建网格元数据；这里 `[V,N,U]` 是 ASTRA Python 投影布局。默认保存，可显式关闭，但关闭后不能进行投影域评估。

### 5.4 Manifest

使用 UTF-8 JSONL，一行一个扫描，至少包括：

```json
{"case_id":"p001_r01","patient_id":"p001","split":"train","path":"cases/p001_r01.h5","resp_signal_id":"r01","parent_ct_id":"p001_ct","simulation_config_hash":"..."}
```

路径相对 manifest 所在目录解析，也支持绝对路径。`raw_manifest.jsonl` 包含 `ct_id,patient_id,split,source_format`、`phase_paths` 或 `source_path`、`axis_order`、`intensity_encoding`、DICOM 系列映射与可选 ROI 引用。导入产物 manifest 保留上述来源、split 和以下信号列表。

每个原始 CT 条目必须声明非空 `resp_signals` 数组，每项含 `id`、`type=phase_id|phase_fraction|synthetic`；文件型信号还含 `path` 和需要的 `index_base`，合成信号含 `period_s`、`initial_phase_fraction`。`simulate` 为列表中每个信号生成一次扫描，case_id 固定为 `<ct_id>__<signal_id>`。全局 seed 与 case_id 通过 SHA256 派生病例 seed，不使用进程相关的 Python hash。以合成信号作为真实病例的替代时必须由用户在 manifest 明确选择，不能自动回退。第一版不自动创建额外几何增强副本。

`paired_sources.jsonl` 每行包含 `case_id,patient_id,split`，以及单一 HDF5 的 `source_path`/dataset 名映射，或 `ct_path,cbct_path,prior_path`（推理入口可无 ct_path）、各字段轴顺序、强度编码、共同物理元数据和可选 ROI 路径。适配器先转 HU 和 `[T,Z,Y,X]`，再执行第7节的共同重采样。

数据切分发生在仿真、裁块和增强之前，按患者 ID；同一患者的全部扫描、呼吸信号组合与增强副本必须在同一 split。`split.json` 显式列出患者 ID，不自动套用论文 40/5/11 数量。使用真实呼吸信号时另记录信号库划分，测试不得复用训练信号；同一合成正弦生成器可以跨 split 用于软件体模，但必须标记 synthetic，不声称具有真实信号独立性。

HDF5 按相位和 z 分块读取，不每次把整个数据集转换为 float64。worker 内独立打开文件；Windows 默认 `num_workers=0`。

## 6. 动态 CBCT 仿真

### 6.1 物理单位与计算链路

每个相位的 CT 转换为线性衰减系数：

$$\mu=\max(0,(HU+1000)/1000)\times\mu_{water},\quad \mu_{water}=0.01707\;\mathrm{mm}^{-1}.$$

例如水为 0 HU，衰减系数为 0.01707 mm⁻¹；10 mm 水的线积分为 0.1707。这是与现有脚本常数对应的单能近似，不引入散射、束硬化或电子噪声模型。

对投影视角 n，使用其相位 `phase_id[n]` 的三维体积计算正投影。必须先在较大共同视野生成投影，再在目标重建视野重建；不能先把胸部裁成网络小块再模拟整次扫描。

噪声可关闭；启用时：`counts ~ Poisson(I0 * exp(-line_integral))`，默认 `I0=50000`，`counts_safe=max(counts,1)`，然后 `p=-log(counts_safe/I0)`。记录光子数与 seed，不对每个病例另外拟合强度尺度。

FDK 的结果单位为 mm⁻¹，再使用 `HU=1000*(mu/mu_water-1)` 恢复 HU。噪声和离散重建可能产生负衰减估计，不在重建后静默裁剪。水体模标定未通过时禁止开始批量生成数据。

### 6.2 扫描几何默认值

`simulation_profile=full_fan_reference` 为完整可运行的第一版默认。以下为参考公开脚本的工程几何，并非声称重建作者私有扫描协议。

| 项 | 默认 |
|---|---|
| 扫描时长 | 60 s |
| 投影视角数 N | 680 |
| 旋转 | `theta[n] = -2*pi*n/N`，n=0..N−1，不重复端点 |
| SOD / ODD | 1000 / 500 mm；SDD=1500 mm |
| 探测器 `[V,U]` | `[384,512]` |
| 像素间距 `[v,u]` | `[0.776,1.552]` mm |
| 探测器偏移 | 0 mm |
| 正投影网格 `[Z,Y,X]` | `[320,512,512]`，1 mm 各向同性 |
| FDK 网格 `[Z,Y,X]` | `[180,416,416]`，1 mm 各向同性，与扫描中心一致 |
| 滤波器 / ShortScan | `ram-lak` / false |
| 每批投影视角 | 32，可配置 |

默认直接以降采样后的探测器生成投影，区别于原脚本先 1024 列投影后缩为 512 列；该差异必须写入 provenance。保留 `detector_sampling=legacy_resize` 对照选项，使用原 1024 列、0.776 mm，之后逐视角用 SimpleITK 线性插值重采样到同一物理覆盖范围的 512 列、1.552 mm 网格，v 方向保持，抗混叠关闭；该选项模拟“先投影后降采样”的流程，不声称逐值等于 MATLAB imresize3，也不是默认模式。

ASTRA 的 FDK 目前要求各向同性重建体素；输入可先重采样，网络预处理阶段允许不同 xyz 间距。使用 `create_vol_geom(Y,X,Z,...)` 明确物理范围，图像数组仍为 `[Z,Y,X]`。[ASTRA 几何说明](https://astra-toolbox.com/docs/geom3d.html)

第一版可验证、可训练的仿真范围为无偏移全扇束圆轨迹。`detector_offset_mm != 0` 必须在仿真前抛出 `UnsupportedGeometryError`，不得让未验证的偏移 FDK 默默运行。半扇束扩展需要单独实现偏移几何、截断/冗余权重及体模标定，再解除限制。该限制不妨碍原始 CT → 仿真 → 训练的完整全扇束链路。

### 6.3 呼吸信号与分相

必须支持：

- `phase_id` CSV：每行与一个投影视角对应，显式 `index_base=0|1`，归一到 0..9，长度等于 N。
- 连续相位 CSV：`time_s,phase_fraction`，phase_fraction∈[0,1)，先 unwrap，再按投影时间线性插值，最后取模和分箱；禁止跨周期直接对 0.99→0.01 作普通线性插值。
- 合成周期信号：默认周期 4 s，初始相位 0；`phase_id=floor(10*frac(t/4))`。测试配置可使用不同周期和初始相位。

暂不从无分支信息的呼吸振幅曲线自动推断十相位，也不自动从临床投影提取信号。提供振幅数据时应要求用户给出已分相映射。

检查所有相位均有视角；存在空相位时终止该病例，不填零图替代。写出每相位视角数量、角度分布与分相可视化。

### 6.4 正投影、FDK 与 prior

使用 ASTRA `FP3D_CUDA` 或对应封装；每个新投影缓冲区必须清零，因为该算法将结果累加到现有投影对象。按相位、角度块串行处理，释放所有 ASTRA ID，异常路径也必须释放。[ASTRA FP3D 文档](https://astra-toolbox.com/docs/algs/FP3D_CUDA.html)

默认 `gating_mode=phase_subset`：按相位抽取投影和对应角度，调用与公开脚本一致的子集 FDK；不偷偷把非均匀角度插值为均匀角度。使用库规定的 FDK 尺度，不额外乘 10。相位采样不均匀造成的伪影属于训练输入，但尺度必须通过静态体模检查。

prior 提供两个明确模式：

- `all_views_fdk`（新数据默认）：用真实动态扫描的全部投影做一次 FDK。
- `weighted_phase_fdk`（公开脚本对照）：各相位 FDK 在衰减系数域按 `n_phase/N` 加权求和，再转 HU。

不声称两种 prior 在任何采样条件下完全相同；写入文件和 checkpoint 的 data contract，并拒绝训练/推理间静默变更。FDK 使用圆轨迹与明确滤波参数。[ASTRA FDK 文档](https://astra-toolbox.com/docs/algs/FDK_CUDA.html)

必须逐相位保留经过同一物理变换的完整 CT 真值，而不是只保留最后一次循环的 `P_gt`。仿真生成 `ct,cbct,prior` 后检查形状、有限值和物理网格一致。

### 6.5 资源与失败恢复

每个病例采用临时文件写入，成功校验后原子改名并写入完成记录；失败记录为 JSONL，可单独重试。缓存身份键包含规范 CT 数据及元数据内容 SHA256、呼吸文件内容 SHA256 或合成信号参数、有效仿真配置、病例 seed、实现源码版本/哈希及 ASTRA 版本。只有身份键一致且既有输出内容校验通过才能跳过；仅 CT ID、信号 ID、文件名或配置相同不够。同路径输入内容变化必须报冲突，要求新输出目录或显式的新扫描 ID，不覆盖。

同一 GPU 上仿真和训练默认分进程顺序运行。无 CUDA 时给出可操作的错误，不以 CPU 近似投影器替代后仍标记同一 profile。

## 7. 预处理、数据集与增强

### 7.1 网络网格

默认把 FDK 重建结果和相同物理范围的 CT 真值共同重采样至 `[T,Z,Y,X]=[10,90,256,256]`。保持重建体素覆盖范围与中心，spacing 随尺寸改变；默认范围为 416×416×180 mm，网络 spacing_xyz 为 `[1.625,1.625,2.0]` mm。

该 spacing 是本设计默认，论文未充分给出对应信息。不得对三种图像各自执行独立裁剪或非一致缩放。图像线性插值，掩膜最近邻；保存原始与目标 affine、插值方式和有效支持域。仿真投影数据仍对应原始物理网格。

已有配对数据必须先验证共同网格；重采样策略与仿真入口一致，用户可显式禁用重采样保留已有尺寸。

### 7.2 两阶段样本

- Stage I：取同一 z 位置的所有 10 相位，得到 `[1,10,1,Y,X]` 的单通道 CBCT / CT；prior 为 `[1,1,Y,X]`，模型入口扩展后拼成两个通道。
- Stage II：取同一空间位置的完整 10 相位块；CBCT、CT、prior、有效域和 ROI 同步裁块。
- 单批 batch_size=1；不同 shape 不在同一批直接 stack。Dataset 返回张量及 case_id、裁块坐标、相位偏移、空间变换记录。
- Stage I 每 epoch 的样本数默认是训练 manifest 中各扫描 z 层数之和。Stage II 默认 `15 × 训练扫描数`。这只是与论文样本规模接近的可重现采样预算，不是恢复原始裁块清单。
- 按患者均匀抽样，再在患者内均匀抽扫描、裁块位置；使用固定 seed 与 epoch/draw_id 派生的确定性随机数。不能由仿真副本多的患者主导所有 batch。
- 验证使用生成并保存的固定裁块清单，无随机增强、无 shuffle；测试患者不参与检查点选择。

**小体积与无效裁块：** 启动前检查所有病例能容纳该阶段所有配置块，默认 `small_volume_policy=error`，不悄悄删除大块或重分配 1/3 概率。显式 `pad` 模式允许高索引端 replicate 填充到所需块大小，同时 valid=0；要求原始有效 voxel 占抽样块至少 `min_valid_fraction=0.5`（工程默认）。随机裁块最多按确定性 RNG 重抽32次，仍无法满足就报出 case_id、shape、有效比例并终止，不换另一个病例掩盖问题。验证裁块清单生成时执行相同检查，空有效域不进入 loss。`public_code/legacy_notebook` 训练只接受完整有效、无需填充的合法块，禁止 pad，以便保持旧损失对照的含义。

训练集可启用相位循环平移；同样平移 CT/CBCT/动态掩膜，prior 不变。不默认反转相位顺序或交换 z 与 xy 轴。默认空间翻转/旋转关闭；启用时只作用在共同物理变换下，记录实验偏离。

## 8. 网络结构与公开权重兼容

模型接口为 `RSTAR4DNet(stage: int, base_channels: int = 16)`；标准结构固定 2 输入通道、1 输出通道，不引入 BatchNorm、InstanceNorm、注意力模块或输出激活。

### 8.1 卷积块

对 `[B,C,T,Z,Y,X]` 分别 reshape 后使用 Conv3d 实现：

- xy 分支核 `[1,3,3]`，xy 零填充；
- z 分支核 `[1,3,1]`，z replicate 填充；
- t 分支核 `[3,1,1]`，t circular 填充。

第一组三分支相加，再过第二组三分支相加，最后 LeakyReLU(0.2)；中间不增加激活。Stage I 只保留 xy+t 两分支的计算路径。模型内部的局部残差、跳连位置必须逐一保持 Notebook 行为，不根据论文示意图另加残差。

编码通道为 16→32→64→128，三次 xy 下采样 2 倍；解码反向恢复，最近邻插值后卷积。每级对应原命名 `conv1...conv8`、`conv*_2`、`conv*_3`，保留 state-dict key，或者提供无歧义的一一映射并在测试中验证。

Stage I 首层 z stride=1、末层 z scale=1；Stage II 首层 z stride=2、末层 z scale=2。两阶段都保持 T 不变。

### 8.2 尺寸、冻结与切换

网络核心要求 xy 为 8 的倍数；Stage II z 为偶数。外部输入适配器在高索引端 replicate 填充到合法尺寸，并返回裁剪描述；输出回裁到原 shape。填充像素不参与损失和指标；Stage I 的 z=1 不能按 Stage II 规则填为 2。

Stage I 的全部 z 卷积 `requires_grad=false`，且不进入优化器。Stage II 从 Stage I 初始化时：

1. 严格加载 Stage I 的全部权重。
2. 下采样包装层 `conv1/2/3/4.conv2` 的 z 权重做 Dirac 初始化、bias 清零。
3. 所有可分离块中 z 分支的两组权重和 bias 清零。
4. 打开 z 分支梯度，并建立全新的 Stage II optimizer / scheduler / scaler。

不能保证切换瞬间输出逐体素等于 Stage I，因为 Stage II 引入 z 下采样；测试只验证其他权重保持、初始化值正确及有效梯度。

`--resume` Stage II 或加载 Stage II 预训练模型时，绝不能再次执行 z 初始化。`--resume`、`--init-from-stage1`、`--init-from-stage2` 必须互斥。

公开权重结构验收：232 个条目、总参数 2,118,693、strict load 无缺失/多余 key。模型前向与原 Notebook 使用相同权重、输入和 stage 在 CPU float32 上 `atol=1e-6, rtol=1e-5` 对齐。

## 9. 损失、训练配置与运行逻辑

### 9.1 默认目标

默认使用有效域内的平均 L1，训练张量使用上述密度归一化。例：真值 `[1.0,1.2]`、预测 `[0.9,1.3]`，L1=0.1，对应平均绝对误差 100 HU。按有效 voxel 数归一，不能把补零区计入分母。

可选现代对照损失为 `L1 + lambda_ssim*(1-SSIM2D)`，SSIM 逐相位逐轴向切片计算，图像与窗口明确取 HU `[-1000,500]` → `[0,1]`，data_range=1、Gaussian 11×11、sigma=1.5。用于 SSIM 的裁剪不影响 L1。该选项不标记为论文原始训练方案。

`public_code` 的 `legacy_notebook` 损失必须单独实现并标记：normalized 图像乘 1000、data_range=255、pytorch-msssim 1.0.0；Stage I 训练按原 `[B,1,T,Y,X]` 布局，Stage II 按 `[B,T,Z,Y,X]` 布局。保留其窗口尺寸限制行为；不把这个量命名为四维 SSIM，也不用于论文最终指标。可用归一化随机张量与 Notebook 的 loss 逐值对照。

### 9.2 必交付配置

以下尺寸顺序统一为 `[Z,Y,X]`，所有块均包含完整 T=10。

| 参数 | `paper_l1` | `public_code` | `local_16gb` |
|---|---|---|---|
| Stage I epochs | 30，工程约定 | 60 | 30 |
| Stage II epochs | 30，工程约定 | 30 | 30 |
| Adam lr | 1e-4 | 1.5e-4 | 1e-4 |
| betas / weight_decay | (0.9,0.999) / 0 | 同左 | 同左 |
| scheduler | 每阶段 cosine 至 1e-5，工程约定 | S1 在 20/30/40、S2 在 10/20/25 epoch 后 ×0.5 | 同 paper_l1 |
| 损失 | L1 | L1+0.1×旧 SSIM 项 | L1 |
| Stage I shape | `[1,256,256]` | 同左 | 同左 |
| Stage II shapes | `[16,256,256]`、`[32,144,144]`、`[64,128,128]` | 同左 | `[16,128,128]`、`[32,96,96]`、`[64,64,64]` |
| 块形状概率 | 各 1/3 | 各 1/3 | 各 1/3 |
| AMP / batch / accumulation | fp16 / 1 / 1 | 同左 | 同左 |

论文只报告 30 epochs 和 lr 从 1e-4 降到 1e-5，没有给出明确的两阶段分配或完整 schedule，因此 `paper_l1` 的 30+30 与 cosine 必须带 `engineering_default` 标签。

默认启动配置为 `local_16gb`，它是显存友好的起始建议，不保证任何 16 GB 卡都能容纳。不得 OOM 后静默改变块大小、相位数或网络通道数。

### 9.3 训练器职责

- 使用单一训练循环处理两个阶段；数据准备、model、loss、metrics 分离。
- CUDA fp16 使用 GradScaler；CPU 使用 float32，不能调用 CUDA 专用函数。CUDA bf16 可显式选用并记录，默认不启用。
- 支持 gradient accumulation；梯度按真实累积 batch 数缩放，epoch 最后不足一组也正确更新。
- gradient clipping 默认关闭；启用时在 unscale 后执行，保存阈值。
- 每个 batch 检查 loss / 输出有限性；发现 NaN/Inf 则记录 case_id、shape、强度范围和配置，终止本次运行，不继续写 best。
- 默认关闭 early stopping，跑满配置预算；每 epoch 固定验证并保存 last/best。
- 标准 validation selection 为固定验证样本有效域 L1 的病例等权均值，不能按 batch loss 求和选择 best；同病例多个块先合并其 L1 分子/分母，随后病例等权。
- 训练日志记录 L1、SSIM（如启用）、总 loss、lr、epoch、optimizer_step、样本数、耗时、显存峰值、跳过/失败情况。禁止原 Notebook 的任意常数倍 loss 输出。

### 9.4 复现与恢复

默认 seed=2026；保存 Python、NumPy、Torch CPU/CUDA RNG 状态。`deterministic=true` 时关闭 cudnn.benchmark，启用确定性约束；不支持的算子报错，不静默降级。

第一版承诺 **epoch 边界恢复**，不承诺任意 batch 中断的逐位恢复。`last.pt` 保存完成 epoch 后的状态与 next_epoch；异常中断从最近完整 epoch 重新开始，并在日志说明丢失的未提交步骤。checkpoint 写入前后的 scheduler 与 RNG 顺序由测试固定。

恢复前必须匹配模型/阶段、所有训练语义配置（包括总epochs和调度）、数据合同、患者划分、固定验证清单，以及每个参与 prepared 病例的内容哈希。训练以不可变 prepared 数据快照为输入，来源 CT/呼吸哈希保留为历史 provenance，不要求训练机器同时持有全部原始文件；若显式指定重新生成/校验当前源文件，则其变化必须走新仿真与新 prepared 快照，不能冒充原实验。任一训练输入身份不匹配时拒绝 `--resume`；如需改变实验，使用新的 run_id 和显式模型 warm-start，重建优化器并记录母 checkpoint。

允许恢复时变更的纯运行项仅为 `device,num_workers,pin_memory,log_every` 与显式 `path_map` 路径迁移；迁移后逐文件内容哈希、逻辑病例 ID 和划分必须相同。记录这些覆盖项，设备或worker改变时不承诺逐位一致。不能通过修改 YAML 悄悄更换块尺寸、归一化、loss或训练患者。checkpoint 保存的 manifest 身份以逻辑条目和内容哈希为准，不把迁移后的绝对路径误认为新数据。

相同环境、CPU 确定性测试中，连续训练 2 epoch 与 1 epoch 后恢复再训练 1 epoch 的样本序列、lr、优化器状态及最终参数必须一致。不同 GPU、驱动或库版本的逐位相同不在保证范围内。

## 10. 检查点、阶段串联与实验目录

checkpoint schema 至少含：`schema_version, architecture, model_state, stage, completed_epoch, next_epoch, global_step, optimizer_state, scheduler_state, scaler_state, best_metric, rng_states, resolved_config, data_contract, split_hash, manifest_hash, source_hashes, software_versions`。

只依赖公开 `.pth` 时可将其转换为带元数据的推理 checkpoint；没有优化器状态的文件不能标记为可恢复训练 checkpoint。加载文件采用 weights-only 安全路径，RNG 等元数据存为可安全反序列化的基础类型/张量。

每次运行独占一个目录：

```text
runs/<run_id>/
  resolved_config.yaml
  provenance.json
  environment.json
  input_manifest.jsonl
  split.json
  validation_samples.jsonl
  train.jsonl
  metrics.csv
  stage1/checkpoints/{last,best}.pt
  stage2/checkpoints/{last,best}.pt
  predictions/
  evaluation/
```

默认 pipeline 串联：Stage I 从头训练 → 选择 Stage I best → 初始化 Stage II → Stage II 从头优化 → 保存 Stage II best → 在 val 上生成检查报告。测试集评估须显式执行命令，不在每个 epoch 反复查看。

pipeline 状态必须有 `stage1_complete / stage2_initialized / stage2_complete` 标志与关联 checkpoint 哈希。重新启动不得从新一次 Stage I 覆盖已有 Stage II，也不得对 Stage II resume 再做 z 初始化。

## 11. 推理与定量评估

### 11.1 推理

加载 checkpoint 的模型、归一化、prior 规则；验证输入合同，不使用随手改过的训练配置覆盖它。

- `mode=full`：一次输入完整 `[1,2,10,Z,Y,X]`，使用 inference_mode 和选定 AMP；保留输入输出原尺寸和物理信息。
- `mode=tiled`：空间滑窗，保留完整 10 相位；重叠区域使用严格正的 Gaussian 权重归一融合，最后回裁。默认 tile `[32,128,128]`、overlap `[16,64,64]`，最终覆盖必须包含末端且所有 voxel 的权重和大于 0。
- tile 起点与下采样格点对齐：z 为 2 倍数，xy 为 8 倍数；先填充整幅合法网格，再生成 tile。
- 不自动在 full OOM 后切换 tiled。提示可用方案，用户/配置选择后创建独立运行记录。
- tiled 因感受野边界与整幅结果不完全等价，报告标记模式，不能据此复现论文 whole-volume 时延。

Stage I 推理按 z 层运行，用于消融；Stage II 使用完整体积/块。输出 HDF5 HU、可选逐相位 NIfTI、固定 HU 显示窗的 QC 图。真实输入无 ct 时只产出预测与输入对照，图像真值指标为 missing。

### 11.2 本项目标准图像指标

默认 `metric_profile=project_v1`，区别于作者未公开的 MATLAB 流程：

- RMSE：在原 HU、未截断预测上，按指定有效 ROI 的 voxel 计算。
- SSIM：逐相位逐轴向切片；GT/pred 都裁剪到 [-1000,500] HU，data_range=1500，Gaussian window=11、sigma=1.5、K=(0.01,0.03)、population covariance。使用 scikit-image 并锁定验证版本。
- ROI SSIM：先在完整切片算 SSIM map，再在 ROI ∩ valid 内的有效窗口中心取均值；不把 ROI 外设为零后求整图 SSIM。切片边缘和不完整有效窗口不计入。
- RMSE 的 ROI 可以比 SSIM 的有效窗口中心更多，因此各自记录有效 voxel/center 数。空 ROI 或不足窗口时为 null + reason，不记 0。
- `whole_fov` 指 valid_mask；`body` 指已声明的体区 mask；`lung/tumor` 指对应外部 mask。禁止把 whole_fov 自动标为论文 SSIM-Global。
- 病例 RMSE 先对全部有效相位/voxel 汇总平方误差和与 voxel 数，再开方；不是先算每片 RMSE 后平均。病例 SSIM 汇总全部相位/切片有效窗口中心的 SSIM map 值并除以中心总数。病例间等权求均值与样本标准差（ddof=1）；病例数<2 时 SD 为 null。显式记录病例数和有效 ROI 数。

每次评估同时输出输入 CBCT 和模型结果，以病例配对比较改善幅度。输出 `per_case.csv`、`summary.json`、指标参数、掩膜来源与预测哈希。

作者 MATLAB 协议未来可作为 `author_matlab` 插件接入；未提供原脚本、ROI 与设置前，不创建这个 profile 的“通过”结果。

### 11.3 投影域评估

仅当存在投影、几何、相位映射和足够视野的重建体时可运行。完整物理体积按原 phase_id 正投影，与观测 line_integral 比较；不加第二次模拟噪声。

网络输出视野小于投影物体视野时，默认跳过并报告 `insufficient_reconstruction_fov`。不得把裁剪外一律补空气，再将投影差异解释为网络质量。可选择独立的“外部区域来自何种图像”的实验，但须显式配置和命名，第一版不将其列为论文指标。

project_v1 的投影指标：每视角计算 NCC；SSIM 使用部署前在训练投影上确定的固定 clip_min/max 和 data_range，写入 metric_config，不按每张测试投影重新拉伸。常量投影 NCC 未定义时记 null。先视角平均到病例，再病例等权；不得声称等同论文表 5。

## 12. CLI 与接口

包名 `rstar4d`，统一通过 `python -m rstar4d <command>` 调用。下面都是待实现接口示例，不是当前已存在命令。

```text
doctor          检查训练/仿真依赖与资源
make-phantom    生成三名独立体模病例和显式 train/val/test 划分
import-ct       原始 NIfTI/HDF5/DICOM → 统一四维 HU 数据
import-paired   已有 CT/CBCT/prior 或旧 MATLAB HDF5 → 配对格式
simulate        CT + 信号 + 几何 → 投影/FDK/prior/四维真值
prepare         同步重采样、生成训练 manifest 与验证样本清单
validate-data   schema、物理网格、强度、split 与掩膜检查
convert-weights 公开纯 state_dict → 带规范元数据的权重
train           单阶段训练或恢复
pipeline        顺序执行两阶段训练
infer           整幅或滑窗推理
evaluate        图像域/满足条件时的投影域指标
benchmark       显存与时间检查，不更新实验配置
```

核心 Python 接口：

```python
import_ct(raw_entry: dict, config: dict) -> str  # 输出 HDF5 路径
simulate_case(ct_path: str, signal: dict, config: dict, out_dir: str) -> dict
prepare_case(case_path: str, config: dict, out_dir: str) -> dict
validate_manifest(path: str, split_path: str, strict: bool = True) -> dict
build_model(stage: int, profile: str = "official_v1") -> torch.nn.Module
initialize_stage2(model: torch.nn.Module, stage1_state: dict) -> dict
train_stage(config: dict, resume: Optional[str] = None) -> dict
infer_case(checkpoint: str, case_path: str, config: dict) -> dict
evaluate_case(prediction_path: str, case_path: str, config: dict) -> dict
```

类型标注中的 Optional 从 typing 导入，兼容 Python 3.9。函数返回的 dict 必须符合各命令 JSON schema，包含 status、产物路径、配置哈希和错误上下文，不靠解析控制台文字衔接。

完整命令示例：

```powershell
# 仿真环境
python -m rstar4d doctor --component simulation
python -m rstar4d import-ct --manifest data/raw_manifest.jsonl --out data/imported
python -m rstar4d simulate --manifest data/imported/manifest.jsonl --config configs/simulation/full_fan_reference.yaml --out data/simulated
python -m rstar4d prepare --manifest data/simulated/manifest.jsonl --config configs/data/default.yaml --out data/prepared

# 也可从配对数据开始；此命令直接生成同一种 prepared manifest
python -m rstar4d import-paired --manifest data/paired_sources.jsonl --config configs/data/default.yaml --out data/prepared

# 训练环境
python -m rstar4d validate-data --manifest data/prepared/manifest.jsonl --split data/split.json
python -m rstar4d benchmark --config configs/train/local_16gb.yaml --mode train --stage 2
python -m rstar4d pipeline --config configs/train/local_16gb.yaml --manifest data/prepared/manifest.jsonl --split data/split.json --run-dir runs/experiment_001
python -m rstar4d train --resume runs/experiment_001/stage2/checkpoints/last.pt
python -m rstar4d infer --checkpoint runs/experiment_001/stage2/checkpoints/best.pt --manifest data/prepared/manifest.jsonl --split-name test --mode tiled --out runs/experiment_001/predictions
python -m rstar4d evaluate --predictions runs/experiment_001/predictions --manifest data/prepared/manifest.jsonl --metric-config configs/metrics/project_v1.yaml --out runs/experiment_001/evaluation
```

配置 schema 必须拒绝未知字段、相互矛盾的 init/resume、非法形状/相位数。命令的路径相对于当前工作目录；YAML 内文件路径相对于 YAML 目录；resolved_config 保存绝对路径。CLI override 必须全部写入 resolved_config。

## 13. 建议文件布局

新工程放在当前工作区根目录，现有 `RSTAR/` 作为只读参照保留：

```text
pyproject.toml
environment-train.yml
environment-sim.yml
README-training.md
configs/
  data/default.yaml
  simulation/{full_fan_reference,smoke}.yaml
  train/{paper_l1,public_code,local_16gb,smoke}.yaml
  metrics/project_v1.yaml
src/rstar4d/
  __main__.py, cli.py, config.py, schemas.py
  data/{importers,manifest,preprocess,datasets,sampling}.py
  simulation/{geometry,respiration,phantom,astra_backend,pipeline}.py
  models/{rstar4d,legacy_weights}.py
  training/{losses,trainer,checkpoint,pipeline}.py
  inference/{predict,sliding_window}.py
  evaluation/{image_metrics,projection_metrics,report}.py
  utils/{seed,logging,provenance,io,resources}.py
tests/
  test_data_contract.py
  test_geometry_units.py
  test_simulation_phantom.py
  test_model_compatibility.py
  test_stage_transition.py
  test_resume.py
  test_metrics.py
  test_inference.py
  test_pipeline_smoke.py
docs/data-format.md
docs/reproduction-differences.md
```

模块名可在实现时小幅调整，但职责、外部命令和数据合同保持稳定；不能把整个工程重新堆入一个训练脚本。

## 14. 必须通过的验收检查

| ID | 场景 | 通过标准 |
|---|---|---|
| A01 | 轴顺序与几何 | 非对称体模在物理 xyz 的已知位置导入/重采样/导出后误差≤0.5目标体素；测试 x/y/z 交换能被发现 |
| A02 | 强度单位 | HU→mu→HU、HU→network→HU 的浮点误差≤1e-3 HU；10 mm 水线积分符合 0.1707，离散容差单独说明 |
| A03 | ASTRA 无噪声标定 | 水球体模中心 ROI 平均 HU 误差≤50 HU；另用非对称多结构体模比较360视角与30视角，前者重建RMSE更低；不得用事后拟合比例修正测试 |
| A04 | 动态模拟 | 每视角只从对应相位投影；10相位标签全保存；固定 signal/seed 可重建相同投影噪声 |
| A05 | 患者隔离与输入身份 | 同患者跨split立即失败；派生扫描不能绕过检查；同ID的CT/呼吸内容更新使仿真缓存失效，prepared数据内容变化使resume报冲突；测试信号复用按实际/synthetic规则检查 |
| A06 | 公开权重兼容 | 两份权重严格加载；参数数目一致；每阶段与 Notebook CPU 输出在规定容差内一致 |
| A07 | Stage I/II | S1 z 参数无梯度且不更新；切换只改指定 z 参数；S2 z 参数有梯度且更新；resume 不重置 z |
| A08 | 形状与填充 | 奇数z、非8倍数xy、小于tile的图像推理后恢复原shape；小于训练块按error/pad规则处理；全无效区域在32次内明确失败；填充不进入loss/metrics |
| A09 | 单病例过拟合 | 固定非恒定小体模块、L1、关闭增强，最多300次更新后最后20步平均loss≤初始20步的50%；分别检查两阶段，无NaN |
| A10 | checkpoint恢复 | CPU确定性2 epoch连续训练与1+resume+1状态一致；包含 lr、Adam、样本序列和参数 |
| A11 | 指标正确性 | 相同图像RMSE=0、SSIM≈1；整体+10HU时RMSE=10；空ROI为null；ROI外误差不改变ROI-RMSE |
| A12 | 滑窗 | 恒等/常量预测器重建无接缝、权重覆盖无零；真实网络模式与差异有记录，不要求tiled与full相等 |
| A13 | 两入口与端到端 | 同一体模经仿真输出和配对导入获得一致规范数据；两阶段、保存、重载、推理、指标均通过 |
| A14 | 失败路径 | 空相位、单位不明、缺字段、无CUDA、OOM、损坏权重、输出路径冲突均报明确错误，无伪完成标志 |
| A15 | 显存 | 三种Stage II块各测试forward/backward/optimizer；记录GPU和峰值；16GB目标若不通过则如实报告并显式调整新profile |

合成冒烟配置使用三名不同几何/运动参数的体模，显式 train/val/test 各一；10 相位，网络尺寸 `[10,16,32,32]`，模拟使用小型各向同性网格、足够覆盖物体的探测器、至少200视角，每阶段仅2步。它用于快速验证管线，不用于验收去伪影提升。

投影域评估另测：完整视野无噪声自投影一致；裁剪视野自动输出不可评估原因；物理单位与 phase_id 错误必须被检测。

## 15. 实施顺序与阶段产物

1. **固定合同与网络。** 完成 schema、规范化、模型、公开权重兼容与阶段切换；产物是可加载/前向/反向的模块及检查记录。
2. **完成原始 CT 仿真。** 完成导入、物理网格、呼吸映射、投影、噪声、FDK、prior、体模标定与持久化；产物是可读的真实配对格式。
3. **完成配对数据与训练。** 完成导入适配、采样、两阶段循环、日志、best/last、恢复与串联；产物是可恢复的训练实验。
4. **完成推理与评估。** 完成整幅/滑窗、HU输出、ROI指标与条件满足时的投影指标；产物是病例级结果和数据流可追溯报告。
5. **完成端到端验收。** 跑两入口体模、单病例过拟合、16GB显存检查，锁定环境并完成文档；真实病例效果另附实验报告。

每阶段完成后才能依赖其产物推进，不能用全零假数据替代未实现的仿真函数后宣称端到端完成。

## 16. 实施时必须保留的限制与待核实事项

- 原论文精确半扇束偏移、患者清单、呼吸信号配对和最终 MATLAB 指标协议未获得；不影响本 spec 的全扇束完整训练系统交付，但影响论文数值复现。
- 公开权重的强度约定尚未使用作者样例核实；结构兼容可以独立验收，恢复医学图像质量必须补做真实样例验证。
- 本设计的网络 spacing、默认两阶段轮数、cosine 调度、16GB块形状、project_v1指标均为显式工程约定。
- 真实 DICOM/CT、呼吸信号与肺/肿瘤掩膜需由用户合法提供；缺失时使用体模完成软件验收，并在报告中区分数据来源。
- 分块推理、全扇束几何和不同训练块形状均可能改变效果，结果报告必须展示这些条件。

## 17. 可追溯来源

- [本地论文](D:/desktop/课题2/proj/RSTAR4D/RSTAR4D.pdf)：第9–11页网络与训练，第12–14页数据/训练设置，第17页参数和消融。
- [Stage I 源代码](D:/desktop/课题2/proj/RSTAR4D/RSTAR/githubcode/Training_Tetris1.ipynb)：模型、停用z、loss、60epoch配置。
- [Stage II 源代码](D:/desktop/课题2/proj/RSTAR4D/RSTAR/githubcode/Training_Tetris2.ipynb)：z采样、初始化、loss、30epoch配置。
- [仿真脚本](D:/desktop/课题2/proj/RSTAR4D/RSTAR/githubcode/data_simulation.m)：扫描几何、噪声、FDK、prior加权。
- [公开仓库](https://github.com/ZihengD/RSTAR)：代码发布背景与样例链接。
- ASTRA 官方几何、FP3D_CUDA、FDK_CUDA 文档见第6节对应链接；参考文档当前版本可能高于本机锁定版本，实施以锁定版本能力测试为准。

本次审查时的源文件 SHA256 如下。实施时再次检查并写入 source manifest；来源更新不能静默改变已有实验的版本归属。

| 文件 | SHA256 |
|---|---|
| RSTAR4D.pdf | `2d05eb3cb5387c8160787a2594ad7e051cfb8d413bacc6fe6893efff30916891` |
| Training_Tetris1.ipynb | `072327feb27719a5cb0ecbb510232f01eab0d111e5a231ef700ed502b866cab2` |
| Training_Tetris2.ipynb | `66ba6dba58a87c2e72dcbc9beac9d951093b9108a0c305b66c98dc606c5f47df` |
| data_simulation.m | `193d38d20e70e98d031122ded288970f0452e5fe709a03e8481c3f1a35d6330f` |

## 18. 设计审查记录

2026-11-16 独立 builder check 指出三项影响实施的缺口：扫描等中心与 CT 支持域规则、输入内容变化时的缓存/续训判定、已有小体积与空有效裁块处理。现已分别在§5.2、§6.5/§9.4、§7.2补齐，并增加对应验收。审查未发现网络结构、阶段切换和公开权重兼容的其他实施阻碍。上述审查针对规格可实施性，不代表未来程序已通过验收。
