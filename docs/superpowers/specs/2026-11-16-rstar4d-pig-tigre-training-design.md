# RSTAR4D 猪数据训练与 TIGRE CBCT 仿真 Spec

版本：2.2 · 日期：2026-11-16 · 状态：实施规格；代码位于rstar4d，运行方式见README-training.md，实际验收以实现验收报告为准。

本文件替代 v1 的 ASTRA、固定10相位、HU标签与统一90层预处理方案。用户明确要求使用 TIGRE，数据来自 `D:/desktop/课题2/data/pig/`，GT 定义以 `D:/desktop/课题2/generate_round1_contrast_gt.py` 及其依赖为依据。未由用户指定的默认值在本文作为工程选择写明，不等同于论文已报告设置。

用户已说明旧GT与CBCT由不同配置有意生成，本次复现与运行审查不处理这组历史文件的配对，也不以此判定程序不能运行。历史记录仅作背景；新训练输入的空间合同仍按所选数据方案定义。

**最新用户要求优先：具体算法以公开代码为准，完整实现本Spec。** 默认采用公开网络、circular时间边界、原L1+0.1×SSIM损失与stage张量布局、60/30轮、Adam1.5e-4及原分段衰减。猪域强度缩放、较小空间块和有效掩膜是显式适配。prior默认采用公开MATLAB的按视角数加权相位FDK，all_views_fdk保留为用户原流程选项。本文此前的30+30/cosine/L1默认已被此要求替代。目录使用平铺可安装rstar4d包，允许在当前工作区直接python -m rstar4d，无需先修改外部环境。

## 1. 目标与范围

交付可安装的 Python/PyTorch 工程，支持：

1. 原始猪多相位 CT → TIGRE 正投影 → 相位混合扫描 → 稀疏相位 FDK、prior 和逐相位全视角 FDK GT → 配对数据 → Tetris 两阶段训练 → 推理与评估。
2. 已有 GT / CBCT / prior / 有效掩膜 → 元数据与一致性检查 → 同一训练管线。

必须同时交付两条入口。保留原始数据和外部生成脚本不变；新输出写入 RSTAR4D 工作区，默认不得覆盖外部 `outputs`。当前目标是适配猪数据的 RSTAR4D 方法，不声称重现原论文全部肺部实验。

必交付：数据发现与manifest、TIGRE仿真、旧数据适配、模型、两阶段训练、best/last checkpoint、续训、整幅/滑窗推理、强度域正确的指标、体模与测试、环境锁定和运行文档。临床原始投影解析、真实呼吸提取、半扇束设备校准、其他论文基线、多GPU训练不属于第一版。

## 2. 论文是否需要 prior

**论文正式实验和公开实现都使用 prior。** 论文§2.3.4把 average image 与 degraded image 拼接输入 RSTAR4D-Net；公开网络首层为两个输入通道，训练时复制 prior 到各个相位后拼接。

prior 提供同一次动态扫描的平均结构信息和运动模糊。默认按公开MATLAB脚本累加相位FDK：`prior=sum(FDK_t*N_t/N)`；显式`prior_mode=all_views_fdk`则对全部动态混合投影做一次FDK，与用户原prior代码一致。两者都仅使用输入投影，但不能未经角度权重、滤波和几何对照就宣称数值完全相同。prior不是额外采集的静态CT或逐相位GT平均；使用GT构造训练输入会改变可观测信息，第一版禁止。

新系统默认保留两个通道：相位稀疏 CBCT + 扫描 prior。训练和推理都需要 prior；已有数据未提供 prior 时，只有原始相位混合投影及其几何足够完整才可重建它，否则报错。移除 prior 是另一个1通道消融模型，不兼容公开首层权重，不在主线中隐式启用。

## 3. 本次代码与数据核查结论

### 3.1 原始数据清单

发现规则与用户 `discover_series` 一致：在 `AHFP-Round-*` 下递归查找增强系列中的 `IMxxxxx_ct.nii.gz`，排除 `without_contrast` 和 `*_mask`。以下是文件清单统计，不代表全部图像已通过内容检查：

| Round | 增强系列数 | 每系列相位数 |
|---|---:|---:|
| 1 | 10 | 11 |
| 2 | 10 | 10 |
| 3 | 10 | 10 |
| 4 | 9 | 10 |
| 5 | 10 | 10 |
| 6 | 14 | 10 |

共63个系列；系列/姿态/轮次不能视为63个独立动物。按稳定 animal_id 划分，同一动物所有 Round、姿态和派生数据必须属于同一 split。

### 3.2 必须纠正的 v1 假设

- **GT定义：** 当前脚本对每个相位独立使用全部600个角度进行投影和FDK，`gt_generation=single_phase_full_angle_fdk_v1`。直接采样源CT仅用于生成有效掩膜，不提供最终GT强度。
- **全视角不是360°：** 当前设置600 views覆盖0°至299.5°，总跨度300°。不能把 full-angle GT 写成360°无缺失角CT真值。
- **相位数：** 支持T=10和T=11；已验证原网络两阶段都能加载原权重并对11相位小输入前向计算，未证明猪数据上的恢复质量。
- **强度：** 原CT为HU，`TigreTool.hu_to_mu`使用 `mu_water=0.02`；FDK输出为衰减系数，校准可在FDK后施加多项式。已保存的GT不能直接当HU除以1000。
- **坐标：** 当前脚本把 nibabel 返回的数组/spacing直接按zyx送入TIGRE。抽查文件的NIfTI affine为对角矩阵，但脚本语义是首轴切片；不得仅依赖NIfTI常规xyz解释自动转置或LPS重定向。
- **深度：** 抽查GT有467层、640层，不能无条件缩到论文90层；默认保留匹配GT网格，用空间块训练和滑窗推理。
- **历史输出说明：** Round1 Control_1曾抽查到GT与CBCT采用不同FOV/偏移配置。用户确认这是有意生成的不同产物，不要求本次处理，不列为当前复现或运行阻塞项。
- **生成版本漂移：** 抽查旧sidecar记录 `polynomial_valid_source_only_v2`，当前脚本记录 `polynomial_post_fdk_zero_preserved_v1` 或 `hu_to_mu_valid_source_only_v1`。不能把旧输出自动归为当前生成版本。

### 3.3 采用的方案

采用 **Python TIGRE + PyTorch + HDF5 + 显式元数据**。先把用户现有的几何、相位映射和重建后处理整理为项目内的独立适配层，再验证与外部脚本的结果一致。不得只将旧文档中的ASTRA名字替换为TIGRE而保留错误布局、单位和网格。

保留两种模型profile：`official_v1`用于复现主线及公开权重/前向对照，默认采用论文的circular时间填充；`pig_ordered_v1`仅作为有序状态轴的可选reflect消融，区别见§9。数据的状态顺序与模型的边界假设分别记录。

## 4. 依据、配置来源与环境

以下外部文件只读参考，不直接修改：

- `generate_round1_contrast_gt.py`：GT生成与valid mask。
- `generate_round1_contrast_recon.py`：动态投影、相位稀疏FDK和prior。
- `generate_pig_first_phase_prior.py::reconstruct_prior`：投影padding、重建网格、FOV mask、裁剪/缩放。
- `utils/tool.py::TigreTool`：几何、角度、HU→mu、Ax、FDK。
- `utils/phase_mapping.py`：cycle.csv到投影视角/状态标签映射；`moni/phase_mapping.py`只是重导出。
- `series_calibration.py`：按系列取多项式、FDK后校准、零背景保持。
- `get_data/settings/heart_only/round{r}_heart_gt_settings.csv`、`round{r}_heart_calibration.csv` 与 `cycle.csv`：推荐的新生成配置来源，与 `get_data/README.md` 一致。

**配置优先规则：**

1. 复用现有GT时，以该GT的 `gt_config.json`、manifest、实际文件共同确定几何、校准及生成版本；输入信息矛盾则拒绝，不从另一份默认CSV补猜。
2. 全新生成时，显式指定每轮settings CSV和 `calibration.mode=post_fdk|none`。推荐 `pig_heart_post_fdk` 使用heart_only逐轮settings与逐系列calibration；必须在resolved config中列出实际路径和行内容。
3. 三者GT/CBCT/prior必须使用同一series settings与校准模式。旧GT若使用无法重现的旧校准流程，先隔离，不与当前流程混用；可保留已有匹配的三者作为单独legacy cohort。
4. settings有`expand`等当前SeriesSettings未消费的字段时，审计报告列出“源脚本忽略字段”，不能自行赋予含义。

训练环境目标Python3.9–3.11，PyTorch2.3.1；TIGRE使用当前可导入的 `pytigre` 分发版本3.0.0作首个兼容基线，同时记录实际模块/扩展哈希。新环境版本须实际测试后锁定，不凭分发版本推定上游commit。

依赖：numpy、scipy、nibabel、tifffile、h5py、PyYAML、scikit-image、tqdm、torch、TIGRE；TensorBoard可选，旧loss对照才需要pytorch-msssim1.0.0。不再要求ASTRA或MATLAB。TIGRE源项目为[CERN/TIGRE](https://github.com/CERN/TIGRE)，新安装不得使用会安装同名无关包的 `pip install tigre`。

同环境或分离的sim/train环境均可；仿真和训练默认顺序运行。安装清单与`doctor`检查真实CUDA扩展调用，不仅检查import。本轮已用现有TIGRE完成32³水球的120-view GPU Ax/FDK小检查，尚未执行完整猪体积投影；具体证据见§18。

## 5. 数据身份、轴与存储合同

### 5.1 身份与划分

`case_id=round__animal_id__series_name__scan_variant`；保存完整series_relative_path。禁止按round、slice、phase随机划分训练集。同一动物在不同轮次/姿态归入同一split。目录嵌套出现多个动物ID时要求 `identity_map.csv` 显式映射；例如Round5有 `Stent_3_7550/Stent_2_7596/...`，不能以最外层目录静默判为7550。

首次运行生成inventory与候选split，真实训练必须读取明确的`split.json`。数量少时可提供按动物的grouped cross-validation配置；不编造论文40/5/11划分。测试动物/标签不能参与模型选择或拟合新校准系数。

逐系列校准系数若来自该动物的图像统计或另行测量，必须记录来源；它是数据生成条件，不能宣称测试病例完全无个体校准。训练阶段不读取GT重新拟合这些系数。

### 5.2 数组与坐标

- 原始猪CT：`axis_order=ZYX`、`coordinate_convention=pig_source_array`，保留nibabel数组值和原header spacing顺序；不自动canonicalize。
- 磁盘图像：`[T,Z,Y,X]`，prior为`[Z,Y,X]`；模型为`[B,C,T,Z,Y,X]`。
- 投影：TIGRE布局`[N,V,U]`，N=view、V=row、U=column，绝不沿用ASTRA的`[V,N,U]`。
- 默认保留原系列T，保存`phase_labels`显式顺序；Round1 `[1..11]`，其余按文件和CSV核实，不能丢一相位凑10。
- 保存 `source_spacing_zyx_mm`、`used_spacing_zyx_mm`、`output_spacing_zyx_mm`、原始affine、source-array到scanner的映射、settings行和FOV采样描述。
- 本适配层只宣称扫描/数组坐标一致，不凭旧对角NIfTI affine宣称已恢复动物真实LPS坐标。通用标准NIfTI/DICOM适配器使用独立的坐标模式，不能混入本profile。

TIFF通常不含足够空间元数据：导入必须伴随 `gt_config.json` / `reconstruction_config.json` / manifest，或完整的人工显式参数；缺失则拒绝训练。不同格式的同一图像若均存在，校验shape/dtype/值与元数据，只选一个数据源，不能重复计样本。

### 5.3 强度与归一化

`intensity_domain` 必须为 `hu|mu|calibrated_mu`，并有 `mu_water`、`calibration_application`、系数/来源哈希和`normalization`。

- 原CT HU→mu：`max(0,0.02*(1+HU/1000))`；恰为-2048 HU的源padding设mu=0且valid=0，不能当作有效空气。
- 无校准FDK保存为mu，单位mm⁻¹；理论上可以用`HU_equivalent=1000*(mu/0.02-1)`输出附加量，但须明确为该单能假设下的等效HU。
- FDK后多项式校准的输出保存为`calibrated_mu`；保留经验标定域，默认不将其逆映射或直接改名为HU。
- 网络输入/目标统一 `x=value/0.02`，输出乘0.02还原原强度域。这个固定缩放是本项目工程默认；不逐病例min-max，不根据测试集拟合尺度。
- `hu`配对数据可在显式profile中按`(HU+1000)/1000`归一化，但不能和mu数据在同一训练运行静默混合。
- 两份公开权重仅证明架构可加载，其原始强度约定和猪域效果未核实。新训练默认Stage I随机初始化；如显式warm-start必须记录域迁移与归一化假设。

### 5.4 HDF5与Manifest

raw CT与paired HDF5分开schema。raw保存`source_ct_hu [T,Z,Y,X]`与valid；paired必需：

| 字段 | 形状/含义 |
|---|---|
| `cbct` | `[T,Z,Y,X]`，各状态稀疏FDK |
| `prior` | `[Z,Y,X]`，同一混合扫描全部views的FDK |
| `target` | `[T,Z,Y,X]`，逐状态全视角FDK；无真值推理可缺 |
| `valid_mask` | `[T,Z,Y,X]`，GT逐相位有效采样掩膜 |
| `roi_labels` | 可选同shape整数分割；不预设类别为肺/肿瘤 |

使用`target_kind=single_phase_full_angle_fdk_v1`，避免把target字段暗示成原始无误差CT。`ct`仅在旧数据适配时映射为target，必须声明其target_kind。

metadata必需：schema_version、case/animal/round/series、split、phase_labels、temporal_topology、强度与坐标合同、所有shape/spacing、扫描/重建几何、原始来源内容哈希、settings、校准、cycle及实现版本、有效域规则、noise配置。

投影单独存HDF5：`mixed_line_integral [N,V,U]`、`angles_rad [N]`、`state_label [N]`、`branch [N]`、`cycle [N]`、`projection_index [N]`和几何。保存未padding、未做图像域多项式校准的真实模拟投影。默认保存，不要求保存全部T×N GT投影。

UTF-8 JSONL manifest每行一个case，所有相对路径以manifest目录解析。每个case包括paired文件内容哈希及源文件列表。大图像按T/z分块读float32，避免float64整幅副本；worker各自打开HDF5，Windows默认num_workers=0。

## 6. TIGRE CBCT与GT的共同定义

令原始第t个状态CT为 `C_t`，其未校准衰减体为 `mu_t`；`A_n` 是第n个视角的TIGRE投影算子，`s(n)`由cycle映射决定。`R(P,angles;settings)`表示共享FDK、padding、FOV和输出网格流程；`Q`为可选FDK后标定。

三种图像严格定义为：

1. **混合扫描投影：** `p[n]=A_n(mu_{s(n)})`。
2. **网络输入CBCT_t：** `Q(R(p[state_label=t], angles[state_label=t]))`。
3. **先验prior：** 默认`Q(sum_t(N_t/N * R(p[state_label=t],angles[state_label=t])))`；显式all_views_fdk为`Q(R(p,angles))`。两者都只使用同一混合扫描的600个views。
4. **监督target_t：** 独立计算该状态的 `p_gt_t[n]=A_n(mu_t)`，使用全部600 views做 `Q(R(p_gt_t,angles))`，并按GT脚本有效掩膜置零。

prior不使用target平均、不使用第一相位全视角CT替代。GT额外投影视角仅进入监督路径，不能回流到输入CBCT/prior。默认保持当前脚本无新增噪声；可选Poisson噪声只加到混合扫描p，且CBCT/prior共享同一噪声实现，GT保持无新增噪声。噪声profile与无噪声结果分别记录。

## 7. 几何、相位映射与重建细节

### 7.1 按系列settings建立几何

典型基础值如下，实际以指定CSV行或已有GT元数据为准：

| 项目 | 当前猪流程 |
|---|---|
| views / angle span | 600 / 300°，endpoint=False，最后299.5° |
| DSO / DSD | 750 / 1250 mm；源到探测器距离是DSD，不是ODD |
| detector `[V,U]` | `[474,648]` |
| detector spacing | `[0.616,0.616]` mm |
| 正投影体shape | 原始CT shape，按zyx，不先缩成网络块 |
| used spacing | 源spacing × CSV `voxel_spacing_multiplier` |
| offDetector / COR / rotDetector | 零，保持当前脚本 |
| 正投影方式 | `tigre.Ax`，显式固定Siddon以匹配当前默认 |
| 角度batch | 64，最后不足64保留原顺序 |
| detector padding | 行/列各侧通常100 pixel，按CSV |
| 输出xy | 通常256×256，按CSV |
| 输出z/spacing/FOV | 由该病例的重建函数决定，不统一90层 |

`voxel_spacing_multiplier`是对体素物理尺度的修改，不是图像重采样；投影与GT必须采用同样值，不“修正”为1。物体投影偏移沿zyx定义：

`offOrigin=[0, -(down-up)*used_dY, -(right-left)*used_dX]`。

重建时共享函数创建居中的输出几何，offOrigin回到0；源体已在投影中偏移，因此不能在重建时再叠加相同offset。保存源采样偏移和输出网格中心，使GT有效掩膜、CBCT及分割共用同一映射。

当前TIGRE允许本流程的非各向同性dVoxel；不得保留旧ASTRA规格的“FDK必须各向同性”限制。也不把物体offOrigin误认为半扇束offDetector。非零detector offset仍需独立验证，第一版显式拒绝未验证的半扇束profile。

### 7.2 cycle.csv映射

复用/移植 `build_projection_phase_mapping` 的精确语义：文件行给cycle、max、min，首周期从0开始，之后从上周期min开始；状态顺序从min到max，在返回支上反转；最近状态采用 `floor(fraction*(K-1)+0.5)`。600是最终排他边界，实际index是0..599，共享边界只输出一次。

必须完整保存branch和cycle，不只保存state_label。对于已核对的cycle.csv：

- T=11的views计数为 `[28,62,60,60,61,58,61,60,60,62,28]`。
- T=10的views计数为 `[32,66,67,67,67,67,67,67,66,34]`。

每个索引恰好归属一个状态；空状态、重复索引、超界、最终cycle边界与N不一致均报错，不重采样信号掩盖差异。只要cycle与代码哈希相同，新结果必须逐行匹配上述来源映射。

同一标签的上行/下行views在当前流程中合并重建；不在主线中改成20/22个方向相位。标签轴是`ordered_states`，并不自动等价于原论文的闭合时间相位环，模型边界处理见§9。

### 7.3 共享FDK后处理

实现一个共同的重建函数，CBCT、prior、GT全部调用，不能复制三份略有差别的处理：

1. 输入检测为 `[N,V,U]` float32，几何与角度数匹配。
2. 精确保留用户`pad_linatech`：沿目标轴拼接反向的前padding段、原数据、原顺序的后padding段；分别处理两探测器轴。这不等于通用的对称reflect，不能随手替换。
3. 同步增加detector shape为原尺寸加各侧padding；物理原始FOV仍按未padding detector算，不能将外推padding视为真实测量扩展FOV。
4. 计算object_fov=`nDetector*dDetector*DSO/DSD`，重建候选shape为floor(FOV/used_spacing)，限制不超过source shape。
5. 若 `fov_source_pixels>0`，直接在xy固定输出尺寸上重建，xy spacing乘 `fov_source_pixels/fov_output_pixels`。请求FOV不能超过未padding detector FOV。
6. 在上述重建网格执行FDK，随后先使用该网格的原FOV mask及`fov_mask_shrink_pixels`规则，将FOV外值设0；不能把padding造成的额外视野纳入有效域。
7. 然后按原函数center crop或SciPy zoom(order=1, mode=nearest, prefilter=False)到xy输出大小，保留z。wide-FOV已是目标大小时该步为相同大小的裁取。固定相同中心和奇偶尺寸行为，不在最终网格另套二值FOV mask替代“先mask再缩放”。
8. GT另外使用 `make_phase_gt(...,return_valid_mask=True)` 的全插值源有效且在FOV内规则；取样数据只用于mask，不替换FDK强度。
9. 需要标定时，三者均在FDK后应用同一多项式，逐series系数一致，输入恰为0的位置保持0；GT先按valid mask清零再做这个操作。

TIGRE版本、Siddon、FDK滤波与角度归一化必须锁定。当前安装的FDK实现默认ram_lak且内部parker=False；兼容profile保持实际来源行为，不因扫描300°就自行添加Parker权重。若未来改权重或滤波，GT/CBCT/prior全部重新生成并给新profile。不得额外乘T或N/n_phase补亮度。

### 7.4 校准版本和投影一致性

默认新生成使用当前脚本的“源HU→mu→投影→FDK→多项式”顺序。旧 `polynomial_valid_source_only_v2` 不等同于当前post-FDK；二次多项式与投影/FDK不交换。复用旧标签时不能只更改sidecar字符串使它通过。

post-FDK校准后的图像也不能直接当物理mu正投影，并声称与原始p一致。第一版的物理投影域评估只支持未经多项式标定的mu输出且具有完整物体FOV；校准域或heart-only裁剪输出默认报告不适用。需要逆标定/外部体积补全时是独立实验，不能在本管线中自动猜测。

### 7.5 批处理、缓存与资源

逐series、逐phase、64-view块处理，避免把T个大CT和T套GT投影同时放入内存。GT各相位使用全部views，完成重建后释放投影。保留混合600-view数组供prior与分相重建；输入投影不能被FDK原地改变，使用测试固定此合同。

每病例临时文件写完、数值/合同通过后原子提交；失败保留诊断不写complete。缓存key包含源CT内容与元数据、settings有效行、calibration有效行及应用位置、cycle内容、随机seed、实现/TIGRE版本、生成profile。只比较文件size/mtime的旧签名不够作为新系统缓存依据。

源data/script在工作区外，只读；默认新产物目录 `data/generated/pig_tigre/`。已存在不同哈希结果报冲突，不覆盖。与训练共用GPU时顺序运行，不在后台自动同时占用显存。

## 8. 配对导入、预处理和抽样

已有产物支持TIFF、NIfTI、HDF5，映射文件名 `phase_XX_gt`、`phase_XX_gt_valid_mask`、`phase_XX_reconstruction`、`prior_image`；从明确manifest指定根目录，不从多个outputs目录自动拼接“看起来最近”的文件。

进入训练前必须逐病例核对：T与labels、shape、真实spacing、FOV/采样位置、source shape/spacing、voxel倍率、offOrigin、padding、滤波/版本、校准模式/系数、生成版本、valid mask来源。same shape不是配准证据。参数不匹配的三者拒绝入库；不能靠resize让其表面同形。

默认 `preprocess=preserve_gt_grid`：保留每个匹配病例的z深度与xy大小/spacing。若显式选共同重采样，三者和掩膜共享同一变换，记录新profile；不把已生成heart-only FOV恢复成未见过的胸部全FOV。

训练仍使用局部空间块但保留该病例所有T状态：

- Stage I每样本取同一z、全部T，网络输入 `[1,2,T,1,Y,X]`。
- Stage II使用全部T的空间块；target、cbct、prior、valid、ROI同步裁块。
- batch_size=1，允许相邻batch具有不同T或空间shape，不做时间插值，不把T=10 padding成11。
- 先均匀选animal，再选其病例与空间位置。Stage I每epoch预算默认为训练病例z层数总和；Stage II为15×训练病例数。预算、抽样次序与seed写入日志。
- `ordered_states`默认关闭相位循环平移、时间反转和状态丢弃；论文cyclic profile才允许同步相位roll。
- 小于配置块默认预检报错；显式pad时高索引端replicate填充，valid=0。要求有效voxel比例≥0.5，最多确定性重抽32次，失败报出case，不静默改变块概率或跳过病例。
- 验证使用预先保存的固定样本列表、不shuffle、不增强；测试从不参与best选择。

猪ROI依据已有整数分割标签及用户提供的label_map；第一版没有证据将其叫作“肺”或“肿瘤”，不按论文名称套指标。缺ROI可计算有效域指标，不能假造ROI。

## 9. 网络与Tetris阶段切换

### 9.1 保持的主体

沿用现有U-Net：输入2通道输出1通道、base_channels=16、编码16→32→64→128、三次xy下采样、最近邻上采样、原局部残差与跳连。无新增归一化/注意力/输出激活。

每个可分离块：xy核`[1,3,3]`、z核`[1,3,1]`、t核`[3,1,1]`并行相加，再重复一组，最后LeakyReLU(0.2)。不在两组间增加激活。xy零padding、z replicate；t边界由profile决定。

Stage I跳过z分支并冻结其参数；首层z stride=1，末层z scale=1。Stage II启用z，首层z stride=2、末层z最近邻×2。核心xy要求8倍数、Stage II z为偶数；外层适配器高端填充并回裁，填充不进loss/metric。

### 9.2 状态轴边界

`official_v1` 使用原时间circular padding，作为论文架构基线；T=10/11均可运行，公开state-dict key/shape不变。

`pig_ordered_v1`（可选消融，非默认）把t padding改为reflect：最小/最大状态相邻信息来自端点内侧，适配当前min→max→min的有序状态生成方式，不把最小与最大直接接为环。其余网络、权重key与参数量保持。

这项变化是适配提案，不属于论文原始配置；反射邻接也不使状态轴自动成为均匀物理时间。结果必须对照记录 `temporal_topology=ordered_states`、`temporal_padding=reflect`。不能用“strict load成功”宣称与原circular网络前向等价。至少保留circular/reflect的可配置对照入口。

### 9.3 权重与初始化

`official_v1`两阶段均应有232个state条目、2,118,693参数。保留原key或完整一一映射，严格加载；CPU相同输入/权重与Notebook `atol=1e-6,rtol=1e-5`。对T=10和11都检查shape/有限值。

Stage I→II：加载Stage I全部模型权重；`conv1/2/3/4.conv2` z权重Dirac、bias=0；所有可分离块两组z权重/bias=0；解冻z并新建Stage II optimizer/scheduler/scaler。其他权重不变。z下采样引入后不要求切换瞬间输出相等。

Stage II resume或加载Stage II权重时绝不能再做z初始化。`resume`、`init-from-stage1`、`init-from-stage2`互斥。默认第一阶段从头训练，阶段二继承本次Stage I best。

## 10. 损失与训练配置

默认损失以公开代码为准：valid域L1 + 0.1×(1−原SSIM)，猪归一化域为value/0.02。全有效输入逐值对照pytorch-msssim1.0.0，Stage I/II保留原输入布局及乘1000、data_range255；部分有效输入扩展为只统计完整有效SSIM窗口，invalid不参与梯度。没有有效voxel即报错。例：target=0.020、pred=0.018 mm⁻¹，归一化L1=0.1，原域绝对误差=0.002 mm⁻¹；若是calibrated_mu，则只报告该标定域误差，不称为100 HU。

可选SSIM损失使用固定物理强度窗，不逐图拉伸。默认猪project window=[0,0.05]，归一到[0,1]后逐状态逐轴向切片算2D Gaussian SSIM map，window11、sigma1.5、data_range1；窗裁剪只影响SSIM不影响L1。只有11×11窗口全部valid且不越界的中心参与 `mean(1-SSIM_map)`，invalid/padding不参与该项梯度。组合为 `L1 + lambda_ssim*loss_ssim`，猪默认lambda_ssim=0，显式启用时建议0.1并记录。某batch无有效SSIM窗口时该项不计算、有效权重置0并记录skipped_ssim，不能伪报SSIM=1；整个epoch均无窗口但lambda>0则报配置错误。这个窗是工程默认，必须用训练集QC核查和记录覆盖率，再固定用于val/test。

| 配置 | 默认猪`pig_local_16gb` | `paper_l1_reference` | `public_code_reference` |
|---|---|---|---|
| 模型/t边界 | official_v1 / circular | official_v1 / circular | official_v1 / circular |
| epochs S1/S2 | 60/30，公开代码 | 30/30，工程约定 | 60/30 |
| Adam lr | 1.5e-4 | 1e-4 | 1.5e-4 |
| betas、weight decay | (0.9,0.999)、0 | 同左 | 同左 |
| scheduler | S1在20/30/40后×0.5，S2在10/20/25后×0.5 | 每阶段cosine至1e-5 | 同猪默认 |
| loss | L1+0.1×原SSIM，有效域扩展 | L1 | L1+0.1×原SSIM，旧强度/布局对照 |
| Stage I空间块ZYX | `[1,256,256]` | 同左 | 同左 |
| Stage II空间块ZYX | `[16,128,128]`、`[32,96,96]`、`[64,64,64]` | `[16,256,256]`、`[32,144,144]`、`[64,128,128]` | 同paper块 |
| 三种块概率 | 各1/3 | 同左 | 同左 |
| batch/accumulation/AMP | 1/1/fp16 | 同左 | 同左 |

上述块均保留各病例的T。16GB只是起始配置，必须对T=11实测峰值，不能自动OOM后减相位数、通道数或块形状。

原论文仅报告L1、30epochs、lr由1e-4降到1e-5，未明确两个阶段的预算/schedule；参考配置不称为精确论文设置。

原SSIM用pytorch-msssim1.0.0、归一化值乘1000、data_range255及原Stage I/II维度布局；T=11会改变该库对时间长度与11窗口的处理。依据用户最新要求保留这一公开行为，并在README明确说明；最终图像评价仍使用独立的原强度域指标，不能将该训练SSIM当作论文表格指标。

训练器要求：单一两阶段循环；CUDA fp16+GradScaler、CPU float32；支持梯度累积及最后不足组的正确缩放；默认不clip梯度、不early-stop；每epoch验证并保存best/last。记录各loss、lr、step、样本数、T/shape分布、耗时、显存、有效比例。遇NaN/Inf记录case/范围并终止，不保存新best。

best依据固定验证块的L1：同case先合并误差和与有效数，再case等权平均；同时报告animal等权验证指标，禁止以较长病例的更多slice掩盖动物差异。模型选择规则在运行开始固定。

## 11. 断点、随机性与实验追溯

seed=2026作为工程默认，保存Python/NumPy/Torch CPU/CUDA RNG。deterministic模式关闭cudnn.benchmark，遇不支持算子报错。样本随机数由seed、epoch、draw_id稳定派生，不使用Python进程hash。

第一版保证epoch边界续训；中断后从最后完成epoch重启，不声称任意batch逐位恢复。连续2epoch与1epoch+resume+1epoch的CPU确定性小测试应得到相同样本顺序、lr、Adam和模型状态。

checkpoint包含schema_version、模型profile/t padding、state_dict、stage、completed/next_epoch、step、optimizer/scheduler/scaler、best、RNG、resolved配置、数据/强度/坐标合同、animal split、样本清单及全部内容哈希、软件与源代码版本。

恢复时必须匹配模型、总epoch/schedule、loss、归一化、几何/校准数据快照、split及验证列表。prepared数据是不可变训练输入，不要求训练机同时持有全部原始CT。只有device、workers、pin_memory、log_every和经内容哈希验证的路径迁移可覆盖；其他变更必须新run_id并显式warm-start。更换GPU/版本不保证逐位相同。

只含公开state_dict的权重不得冒充可恢复的完整checkpoint；weights-only安全加载，元数据/RNG用基础类型与张量序列化。checkpoint原子写入，测试真实重载而非只检查文件存在。

pipeline按Stage I best→Stage II初始化→Stage II训练运行，保存阶段完成/初始化标志和父权重哈希。重新启动不能覆盖已有Stage II或重复初始化z分支。

## 12. 推理、评估与数据有效性

默认猪数据推理使用空间滑窗并保留全T；原始GT深度467/640时不承诺16GB整幅推理。提供显式full模式，不在OOM后静默换模式。

默认tile ZYX=`[32,128,128]`，overlap=`[16,64,64]`；z起点对齐2、xy对齐8，包含尾部；严格正Gaussian权重融合，全部voxel分母>0，最后回裁。tiled因边界上下文不同不保证等于full，报告模式。Stage I按z切片推理用于消融。

输出HDF5/TIFF及metadata，单位保持mu或calibrated_mu；NIfTI导出必须显式说明其scanner/array affine约定，不能丢掉sidecar。固定显示窗生成输入/target/预测对照，显示窗不改变保存值。

`metric_profile=pig_attenuation_v1`：

- RMSE/MAE在原强度域、valid/ROI内部计算，未截断预测；名称为RMSE_mu或RMSE_calibrated_mu，不能写RMSE_HU。
- SSIM逐状态逐轴向切片，GT/pred固定裁到[0,0.05]，Gaussian11、sigma1.5、K=(0.01,0.03)、population covariance、data_range=0.05。
- ROI SSIM先整片计算map，再对ROI∩完整有效窗口中心取均值，不把ROI外清零后求整图SSIM。
- 病例RMSE先汇总所有有效voxel的平方误差和/总数再开方；SSIM按有效窗口中心总数聚合。病例间等权，并额外按animal先聚合病例再跨animal统计，防止多轮同动物过度加权。
- SD使用ddof=1，n<2为空；空ROI、无有效窗口、无GT明确记null+原因。
- 同时计算输入CBCT与预测，报告每例改善与失败，不只报告最好病例。

物理投影指标满足§7.4才能启用。heart-only视野缺少外围物体时跳过，不以补空气结果冒充模型一致性；标定域不自动逆多项式。论文MATLAB口径未获得，不用本指标宣称复现论文表格。

允许投影评估时，按 `phase_labels` 查找 `state_label[n]` 对应的预测体，计算 `p_hat[n]=Ax(pred_mu[state_label[n]], geometry_output, angles[n])`。geometry_output使用预测输出的nVoxel、dVoxel和已记录的输出网格到scanner变换；detector恢复保存的未padding几何。输出中心由实际FDK/crop/zoom变换得出，不能复用源体nVoxel/dVoxel或再次叠加原source offOrigin。若输出变换缺失、无法证明覆盖各状态全部非零物体支持域，则报不适用。

与保存的未padding混合line-integral逐射线比较，残差为 `r=p_hat-p_observed`，不再加噪声。病例projection_MAE=mean(abs(r))，projection_RMSE=sqrt(mean(r²))，均为无量纲line-integral误差；relative_L2=norm(r)/norm(p_observed)，分母为0时null。可选NCC逐视角计算并平均，常量视角null并报告计数；不把这些量命名为图像HU误差。病例等权并另报animal等权。阳性测试用完整FOV、无标定、无噪声体模及同一几何自投影，要求残差接近数值误差；另对非零source offOrigin案例检查输出几何不出现二次位移。

## 13. 工程结构与CLI

新包为`rstar4d`，源代码不依赖外部工作目录的同名`utils`包。迁移核心函数时记录外部源哈希，并以相同小输入与原函数比较；运行期读取外部数据/settings可通过配置路径指定。

```text
rstar4d/  # 平铺可安装包，支持直接 python -m rstar4d；下列模块为职责划分
  cli.py, config.py, schemas.py
  data/{pig_inventory,paired_import,contracts,datasets,sampling}.py
  simulation/{tigre_backend,pig_geometry,phase_mapping,reconstruction,targets,pipeline,phantom}.py
  models/{rstar4d,legacy_weights}.py
  training/{losses,trainer,checkpoint,pipeline}.py
  inference/{predict,sliding_window}.py
  evaluation/{image_metrics,projection_metrics,report}.py
  utils/{io,seed,logging,provenance,resources}.py
configs/{data,simulation,train,metrics}/
tests/
README-training.md
environment-train.yml
environment-sim.yml
```

统一接口：

```text
doctor            检查torch/TIGRE真实CUDA调用与资源
inventory-pig     扫描Round/动物/系列/相位/CSV覆盖及歧义
validate-settings 检查几何、cycle、校准与已有GT版本
make-phantom      创建已知运动、多动物划分的小数据
simulate-pig      同一配置生成target、混合投影、CBCT、prior及valid
import-paired     严格导入已有匹配产物
validate-data     检查数据/元数据/强度/坐标/动物划分
convert-weights   公开权重转换，显式注明模型profile和强度假设
train             单阶段训练或恢复
pipeline          两阶段训练串联
infer             full或tiled预测
evaluate          原强度域图像指标/满足条件时的投影指标
benchmark         测T=10/11的显存与时间，不自动改配置
```

CLI接口已按下述形式提供，实际执行验收见实现报告：

```powershell
python -m rstar4d inventory-pig --data-root 'D:/desktop/课题2/data/pig' --settings-root 'D:/desktop/课题2/get_data/settings/heart_only' --out data/pig_inventory
python -m rstar4d validate-settings --config configs/simulation/pig_heart_post_fdk.yaml --inventory data/pig_inventory/series.jsonl
python -m rstar4d simulate-pig --config configs/simulation/pig_heart_post_fdk.yaml --manifest data/pig_inventory/series.jsonl --out data/prepared
# 上面的simulate-pig与下面的import-paired为二选一入口，不能往同一非空目录依次导入两套数据
python -m rstar4d import-paired --manifest data/paired_sources.jsonl --out data/prepared
python -m rstar4d validate-data --manifest data/prepared/manifest.jsonl --split data/animal_split.json
python -m rstar4d benchmark --config configs/train/pig_local_16gb.yaml --stage 2 --phase-count 11
python -m rstar4d pipeline --config configs/train/pig_local_16gb.yaml --manifest data/prepared/manifest.jsonl --split data/animal_split.json --run-dir runs/pig_001
python -m rstar4d train --resume runs/pig_001/stage2/checkpoints/last.pt
python -m rstar4d infer --checkpoint runs/pig_001/stage2/checkpoints/best.pt --manifest data/prepared/manifest.jsonl --split-name test --mode tiled --out runs/pig_001/predictions
python -m rstar4d evaluate --predictions runs/pig_001/predictions --manifest data/prepared/manifest.jsonl --metric-config configs/metrics/pig_attenuation_v1.yaml --out runs/pig_001/evaluation
```

simulate-pig须输出规范paired manifest；可直接用于训练。import-paired是已有产物的另一入口；上述两条生成路径不能被误认为必须先后执行。所有接口返回机器可读summary，不靠解析控制台拼接。

YAML路径相对YAML，manifest路径相对manifest，CLI路径相对cwd；resolved配置保存实际路径。配置schema拒绝未知字段、矛盾init/resume、缺失校准选择或非法phase labels；CSV额外源字段按§4审计规则处理。

运行目录保存resolved_config、source/data hashes、environment、inventory/split、validation_samples、train.jsonl、metrics.csv、两阶段best/last、predictions、evaluation与错误记录。

## 14. 必须通过的验收

| ID | 检查 | 通过标准 |
|---|---|---|
| A01 | 数据发现与动物身份 | Round1识别11相位，其余按实际识别；嵌套ID需显式映射；同动物跨split失败 |
| A02 | 原始坐标/spacing | 非对称体模、offOrigin正负、奇偶网格中心与原代码一致；不按NIfTI常规xyz静默转置 |
| A03 | HU/mu与padding | 0HU→0.02；-1000HU→0；-2048设valid=0；有效空气不能被误排除 |
| A04 | cycle映射 | 当前cycle.csv生成600行、角度终点299.5°、T10/11计数匹配§7.2；边界无重复 |
| A05 | 三种图像的数据来源 | GT每相位全600views；prior仅混合600views；CBCT仅对应状态views；无GT到prior的信息回流 |
| A06 | TIGRE小体模 | Ax/FDK真实CUDA运行，float32与[N,V,U]正确；相同参数/版本对原函数RMSE/相对误差在记录容差内 |
| A07 | 重建后处理 | padding逐值、FOV mask、wide-FOV与legacy裁剪两分支、spacing与原函数对齐；不增加第二次offOrigin |
| A08 | 校准 | post-FDK同一多项式作用于三者且零保持；pre-FDK旧版本混入当前cohort被拒绝 |
| A09 | 配对校验 | 人工构造的同shape不同spacing/FOV训练对被拒绝；缺sidecar的TIFF被拒绝；不以用户有意采用不同配置的历史产物作为本次故障样本 |
| A10 | 权重与模型 | official_v1两权重严格加载、232条目/2,118,693参数；CPU与原前向atol1e-6/rtol1e-5；T10/11通过 |
| A11 | 状态边界 | official circular与pig reflect分别有端点测试；后者端点不读取对侧极值状态；权重key相同但不宣称前向等价 |
| A12 | Tetris切换 | Stage I z无梯度且不更新；Stage II只初始化规定z参数并出现梯度；resume不重新初始化 |
| A13 | 单例过拟合 | 非恒定固定小体模、无增强，300步内末20步L1均值≤初20步50%，两阶段分别检查，无NaN |
| A14 | 可恢复训练 | CPU确定性连续2epoch等于1+resume+1；prepared内容/split/loss变化拒绝resume |
| A15 | 有效域与小体积 | invalid不计L1/SSIM梯度；无SSIM窗口按约定记录/报错；pad/32次重抽正确；无效域不发生除零；T不被填充/裁掉 |
| A16 | 指标单位 | 相同值RMSE=0/SSIM≈1；整体+0.002时原域RMSE=0.002；calibrated_mu不标HU；空ROI=null |
| A17 | 滑窗与输出 | 正权重全覆盖、恒等预测器无接缝、还原原shape/单位；不同T、奇数z、末端tiles通过 |
| A18 | 全流程 | 原始体模入口和已有配对入口均跑通两阶段、保存、重载、推理、评估，产物元数据闭环 |
| A19 | 显存与失败 | T11三种训练块测forward/backward/step峰值；无CUDA/OOM/损坏权重明确失败，无静默减形状 |
| A20 | 投影指标与适用性 | 完整FOV未校准体模自投影残差接近数值误差，非零source offOrigin不被重复施加；calibrated_mu、heart-only不完整FOV明确跳过 |

A06在CPU准备相同输入后调用锁定GPU后端，基础geometry/projection/recon数组默认比较rtol=1e-4、atol=1e-6（mu域）；若GPU版本差异超出，记录并调查，不随意放宽。另用未校准、360°无噪声水球中心ROI检查mu误差≤5%，用于单位标定；这项物理标定不能替代300°猪profile与原脚本的行为对照。

冒烟体模：三个独立合成animal，显式train/val/test；T10/T11都覆盖，小体积约16×32×32、至少200views，每阶段1epoch、8个draw（允许原GradScaler完成初始缩放校准）。它验证程序闭环，不作为医学恢复性能证据。真实效果须在留出动物上评价输入与输出全部病例，不设虚构的论文数值门槛。

## 15. 实施顺序

1. 整理inventory、坐标/强度/相位/目标合同和配置；对现有输出做配对审计，不重建所有数据。
2. 移植并验证TIGRE几何、cycle、padding/FDK、post校准与GT有效掩膜；先体模，再一个小规模真实病例。
3. 在同一配置下生成一个真实病例的GT/CBCT/prior，核对原代码及空间位置后，才批量生成。
4. 完成网络、阶段初始化、数据抽样、loss、checkpoint/恢复，先单病例过拟合。
5. 完成两阶段pipeline、滑窗、原强度域指标，执行T11显存检查与全流程测试。
6. 锁定数据划分/配置/环境，在真实留出动物上报告效果与限制。

不得用假投影、全零target或仅随机tensor通过来宣称整条仿真训练链完成。

## 16. 设计选择、限制与版本证据

- 数据来源和TIGRE由用户明确指定；heart_only逐系列settings、固定0.02缩放、16GB小块、固定评价mu-SSIM窗是工程选择。默认loss、60/30epoch、学习率调度、circular和prior加权遵循公开代码；reflect、paper_l1_reference仅作显式消融。
- 输入是猪增强心脏区域数据，论文关于膈肌驱动运动与旋转伪影的证据不能自动推广；需要实验验证，不以论文肺部指标命名本数据ROI。
- 当前cycle将回程与去程相同状态合并，是原生成代码的设定；第一版保留，不声称区分两个运动方向。
- 全视角300° FDK GT仍有重建算子和有限角度误差，训练目标是接近该参考，不是直接恢复原始CT的所有细节。
- 现有历史产物可能不完整且来自旧强度流程；先核对逐病例sidecar，不能从总manifest的complete或文件名独立推断可训练。
- 本次没有批量生成/覆盖CBCT，没有修改外部数据或脚本，亦未执行完整训练。

实现时锁定外部生成脚本、依赖helper、phase_mapping、settings/calibration有效行、cycle以及本地TIGRE Python/CUDA扩展哈希。对新移植函数保存来源路径/哈希，不让外部文件后续变动静默改变实验。

## 17. 可核查来源

- [论文](D:/desktop/课题2/proj/RSTAR4D/RSTAR4D.pdf)：§2.3.4的average image输入；网络与Tetris见§2.2。
- [公开Stage II](D:/desktop/课题2/proj/RSTAR4D/RSTAR/githubcode/Training_Tetris2.ipynb:371)：prior扩展与双通道输入。
- [用户GT生成代码](D:/desktop/课题2/generate_round1_contrast_gt.py:235)：逐相位全部角度FDK、有效掩膜及post校准。
- [CBCT/prior生成代码](D:/desktop/课题2/generate_round1_contrast_recon.py:344)：相位混合投影、prior、稀疏重建。
- [TIGRE与图像工具](D:/desktop/课题2/utils/tool.py:87)：HU→mu、坐标、角度、projection/reconstruction与文件保存。
- [重建后处理](D:/desktop/课题2/generate_pig_first_phase_prior.py:162)：padding、FOV与输出spacing。
- [相位映射](D:/desktop/课题2/utils/phase_mapping.py:149)、[校准实现](D:/desktop/课题2/series_calibration.py:93)。
- [现有数据生成说明](D:/desktop/课题2/get_data/README.md)、[Round1心脏settings](D:/desktop/课题2/get_data/settings/heart_only/round1_heart_gt_settings.csv)、[cycle](D:/desktop/课题2/data/cycle.csv)。
- [CERN TIGRE 官方项目](https://github.com/CERN/TIGRE)：算子、环境与安装来源。

## 18. 设计阶段的验证与独立审查（历史记录）

- 已读取用户GT、CBCT/prior、几何、重建后处理、校准和cycle映射代码，并只读抽查原始NIfTI header及现有输出sidecar。
- 已复算当前cycle.csv的T10/T11逐视角映射，600行计数和末角度符合§7.2。
- 已使用两份公开权重和原网络做T11小输入CPU前向：两阶段shape正确、数值有限；未测试猪数据恢复效果。
- 使用现有 `TigreTool` 和 `reconstruct_prior` 做32×32×32、mu=0.02的水球，120个360°views、64×64 detector、无padding：投影/FDK均由TIGRE CUDA执行、全部有限，中心4³ ROI均值为0.02007918 mm⁻¹，约0.40%偏差。不是600-view真实猪病例验收。
- 独立builder check确认主体prior/GT、校准、T10/11与动物划分设计可实施，提出三项细节：mask先于crop/zoom、SSIM loss有效窗口、重投影输出几何与指标定义。已分别修订§7.3、§10、§12及验收表。

外部源代码的本次内容指纹如下；实施前再次核对，若变化则重新审查相应合同：

| 来源 | SHA256 |
|---|---|
| generate_round1_contrast_gt.py | `fb531eb5caaa3fc101f833e9336da52c760b680bfdfa40b8a1b45866b34cbe59` |
| generate_round1_contrast_recon.py | `76b8eb86921bb6379f1e04b846d5bc678fc77483655002327cfd855891d5e901` |
| generate_pig_first_phase_prior.py | `f45c547051b888f9f65812e896c9e42ea5f9389e1ea4faf71459d63f44491235` |
| utils/tool.py | `f3bdbe1c84c7b1eaa9c9bdae7d7fb536f25af85e5e08cb152f43bbb46ec6b027` |
| utils/phase_mapping.py | `460dd4794b4aee9b00bc688db880f63aa358aa0855d892bc500fbc5e9e50737c` |
| series_calibration.py | `ce47f520095d690bc43d50d880ba84cfaaa0d6fbbedebe2c69637014af34c5e1` |
| data/cycle.csv | `578343dc38d823795f0c2a85a7cb20750701458254ef40cbdec20c9d65d5de5e` |

### 18.1 后续运行复核（2026-11-16）

详见[论文一致性与可运行性验证报告](D:/desktop/课题2/proj/RSTAR4D/docs/reviews/2026-11-16-reproduction-validation.md)。本次独立验证脚本已完成11相位、600views/300°的小动态体模TIGRE正投影/FDK，接入原Notebook数据类和两阶段训练函数，各执行2步，检查z分支梯度、权重继承及保存重载推理一致。

三种Spec训练块在T=11、原Stage II/circular网络、fp16/L1/Adam下实际完成两次参数更新：峰值allocated约4.37、4.76、4.17GiB；reserved约5.06、5.64、4.99GiB。该结果支持网络部分的16GB起始配置，不覆盖真实猪体积仿真、完整epoch、生产checkpoint恢复或滑窗。

本次发现并确认原Notebook的依赖/数据缺失、保存被注释和未定义writer；当时Spec中的CLI尚未实现；本次已完成`python -m rstar4d`，以实现验收报告为准。默认模型已改回论文circular，reflect只作为消融。以上是部件与合成数据联通证据，不代表本Spec的A01–A20全部完成或论文数值已复现。

## 19. 完整实现与验收

已实现双数据入口、两阶段训练、恢复、滑窗、评价和CLI。公开代码为算法优先依据；具体文件对应、A01–A20证据、独立审查修复以及实际运行范围见 [实现验收报告](D:/desktop/课题2/proj/RSTAR4D/docs/reviews/2026-11-16-implementation-acceptance.md)。历史章节不代表当前代码交付状态。
