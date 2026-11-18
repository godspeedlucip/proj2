# RSTAR4D 猪心轮廓重建消融实验 Implementation Plan

> **For agentic workers:** 按任务逐项勾选。执行时使用 `superpowers:executing-plans`；若选择分任务委派，使用 `superpowers:subagent-driven-development`。

**Goal**　在同一 Stage I 权重、动物划分和裁块序列下，独立训练 M/P/B 的八种 Stage II 组合，以留出动物的整心外表面、LV 表面和全图重建质量衡量增益。

**Architecture**　先统一采集数据、训练监督和全局网格坐标。Stage II 在扩展块 `W` 前向，M 与 P 独立计算修正，在 `conv3_3` 残差之后合并一次；深层编码和同尺度跳连共享结果。训练器在中心块 `C` 计重建和边界损失，运动损失保留邻状态在 `W` 的源支撑。

**Tech Stack**　沿用 Python 3.9+、PyTorch、NumPy、SciPy、h5py、NiBabel、PyYAML、TIGRE、`unittest`。仓库内所有相对路径均以项目根目录为准。

**Spec**　[v1.4 设计规格，第 1–7 节](../specs/2026-11-18-rstar4d-cardiac-contour-ablation-design.md)

## Global Constraints

- 原输入 `[B,2,T,Z,Y,X]`、输出 `[B,1,T,Z,Y,X]`，按实际 `phase_labels` 处理整条序列。复用 `official_v1` 主干及其时间填充，M 单独按有序相邻状态配对。Stage I 使用原网络、采样、损失和权重。
- 所有已纳入相位均用对应全视角 FDK 监督。推理上下文只含采集可得数据。人工表面、可信标注域、参考图和参考描述只进入训练监督或独立评价。
- 对齐完成后，距离统一为 `output_voxel`，ZYX 三轴间距为 `(1,1,1)`。物理 affine/spacing 用于配准并随结果留存。旧毫米缓存须从已对齐数据重新生成。
- 八组统一 `stage2_crop_mode: core_with_context`，中心块为 `[16,128,128]`、`[32,96,96]`、`[64,64,64]`。Stage II 全状态输入 `W`，只对 `C` 计重建损失。旧 YAML 缺裁块设置时走原路径。
- 八组固定 `surface_weight=0`、`grad_weight_vox=0`、`half_width_vox=2`、`w_H=w_V=1`。主配方沿用 `pig_local_16gb.yaml` 的 60+30 轮、Adam、归一化及 `L1 + 0.1 × (1−SSIM)`。只用训练/验证动物选择损失权重、轮廓容差和诊断设置，锁定后进入测试。

## Review Focus

1. 不连续相位 ID、10/11 相位和跨零扫描弧，角度、标注及邻居必须指向同一个状态。见任务 2、3、5。
2. 对应点在 `C` 外、真实 FOV 内时仍应能读取。漏读真实源支撑须在损失前报错；越出真实 FOV 则保留分母和回拉梯度。见任务 4、7。
3. 局部 LV 标注、裁切面和带交叠可能制造假表面或改变结构权重。见任务 2、8。
4. P 只有一个有效粗候选时，相对权重恒为 1；原始有符号余弦仍应进入门控。零粗候选或无效细采样点应无贡献。见任务 6、12。
5. 全关旧权重、扩展模型断点和无 GT 滑窗分别经过不同入口，须核对结构、中心坐标、输出覆盖及选优规则。见任务 9、10。

## 文件与执行顺序

| 任务 | 主要文件 | 可验收结果 |
|---|---|---|
| 1 配置 | `rstar4d/config.py`，新增 `configs/train/pig_cardiac_ablation.yaml` | 解析旧配置，生成主实验的统一配方 |
| 2 标注 | `data/inventory.py`、`contracts.py`、`paired.py`，新增 `data/cardiac_annotations.py` | 从真实相位文件到 FDK 网格的可信双表面 |
| 3 角度 | 新增 `data/acquisition.py`，复用 `simulation/pipeline.py` 的采集记录 | 每相位 14 维，有来源可核对 |
| 4 网格 | 新增 `data/context.py`、`models/sampling_grid.py`，修改 `data/sampling.py` | 训练、验证、推理共用 C/W 及拉回坐标 |
| 5–6 分支 | 新增 `models/motion.py`、`models/prior_fusion.py` | M、P 各自在体模上验证 |
| 7–8 损失 | 新增 `training/motion_losses.py`，扩展 `training/losses.py` | 跨块运动、可选表面、双边界及梯度对照可单测 |
| 9 训练 | `models/model.py`、`official_stage2.py`、`training/trainer.py`、`checkpoint.py`、`pipeline.py` | 八组前后向、验证和严格恢复 |
| 10 部署与显存 | `inference/predict.py`、`sliding_window.py`、`cli.py` | 全幅/滑窗走同一模型，三种实际 W 完成资源验收 |
| 11 评价 | 新增 `evaluation/contours.py`，扩展 `evaluation/report.py` | 独立轮廓按结构、动物配对汇总 |
| 12 实验 | 新增 `experiments/cardiac_ablation.py`，扩展 `simulation/pipeline.py` | 派生八组、机制对照和可追溯的报告 |

表中包内短路径均位于 `rstar4d/`。依赖顺序为 1 → {2,3,4}，4 → 5，3+4 → 6，2+4+5 → 7，2+4 → 8，1–8 → 9 → 10；11 使用 2 和 10 的产物，12 使用全部结果。以下命令在已激活的项目 Python 环境中运行。每项先加指名的测试并运行，确认失败原因对应待实现行为，再实现并用同一命令复验至 `OK`。在 Git 工作副本中按任务单独提交；当前项目根目录无 Git 元数据，可直接用测试和产物记录进度。

### 任务 1　锁定兼容配置与实验开关

**文件与接口**　修改 `rstar4d/config.py` 的 `train_defaults/validate_train`；新增基准 YAML 和 `tests/test_cardiac_config.py`。新增 `resolve_stage2(config: dict, stage: int) -> dict` 返回可序列化的架构、上下文配方、预设系数与生效系数。训练、推理、benchmark 共用这一解释。

- [ ] 写 `test_legacy_and_eight_modes`，断言旧 YAML → `legacy/off/off/False`；八组只变 `motion.mode/prior.mode/boundary.enabled`；全关的裁块配方仍和全开相同。`test_invalid_combinations` 逐项覆盖负数、非有限值、非法枚举、非正结构权重或半带宽、B 开且目标权重为零、非 learned 却请求正表面权重。
- [ ] 运行 `python -m unittest -v tests.test_cardiac_config`。把 Spec §5 YAML 纳入现有 `merge_strict`，字符串 `"off"` 加引号。M 为 `off/identity/learned`，P 为 `off/image_only/geometry`。位移上限 ZYX 为 `[8,8,8]`，结构、平滑、B 起始权重为 `0.05/0.01/0.5`，B 线性升权 5 轮，可选表面截断 5 vox。Stage I 生效系数全零，允许同一 YAML 供两阶段 pipeline 使用。
- [ ] 把 LNCC 半径、支撑余量、网格对齐、分支种子及编码版本纳入序列化配方。**实施初值**用 LNCC 半径 `[2,2,2]`（5³ 窗），局部方差阈值和边缘阈值在归一化参考描述上定义并随 run 保存；支撑余量按任务 4 推导。把 Spec 的起始值与验证后锁定值分别记录。复验旧 YAML 和本任务测试。

### 任务 2　从原始相位清单生成可信标注与距离缓存

**文件与接口**　修改 `rstar4d/data/{inventory,contracts,paired}.py`、`rstar4d/cli.py`；新增 `rstar4d/data/cardiac_annotations.py`、`tests/test_cardiac_annotations.py`。通过新增 `prepare-cardiac-annotations --inventory ... --manifest ... --annotations ... --out ...` 读取逐例人工确认清单，输出新 manifest，关联 `annotations` 路径和 `annotations_sha256`；保留原 paired 文件。模拟和导入的 paired manifest 共用此入口。

- [ ] 写 `test_inventory_partial_labels`、`test_nested_labels_on_shifted_grid`、`test_truncated_faces_and_cache_identity`。用编号 2、5、11 的 CT 和只有 5 的 mask 检查配对；LV 单独可信时 H 不可用；已知地标在移位/缩放后落入指定输出索引。平面内侧表面体素中心作距离零点，距面 2 vox 为真、3 vox 为假。重叠 C 给出逐元素相同的带。
- [ ] 运行 `python -m unittest -v tests.test_cardiac_annotations`。在 `inventory_pig` 按同系列实际 `IM<编号>_ct.nii.gz` 匹配 `IM<编号>_mask.nii.gz`，保留 ID、路径、个数。明确输入清单的主键 `(animal_id,case_id,phase_label,structure)`，含标签语义、可用性、可信区域及可选的双端对应区域。复用 `roi_labels` 时同样要求来源和可信信息；缺失标注时该状态仍有重建监督。
- [ ] 实现 `prepare_annotations(record, inventory_row, declarations, out_path) -> dict`。先以源 mask 和 CT affine 对齐，再按本次 TIGRE 源数组→扫描坐标→输出网格的实际变换（包括 `source_offOrigin_zyx_mm`、有效 spacing、末次 crop/zoom）最近邻重采样；复用 `simulation/backend.py:output_grid` 和 `simulation/reference.py:phase_grid` 的约定。导入数据使用其显式网格变换。以中心地标、边缘地标和真实病例三向切片叠加核对，歧义病例记录原因。
- [ ] 将 `annotation_available[T,2]`、逐结构可信域、可靠配对区域、表面、距离与来源写入 HDF5。H 为 `mask∈{1,2}`，V 为 `mask==2`；在整幅上沿 6 邻域找有可信内外端点的前景表面，`1↔2` 只计 LV。排除 FOV 截断、未知和裁切面。全幅读取有效域内两侧 EDT 距离，单位步长 1，空表面显式不可用。缓存键含标签和 FOV 来源哈希、语义映射、可用性/可信域、输出网格、`output_voxel`、半带宽及算法版本；临时文件校验后原子发布。
- [ ] 扩展 manifest 路径解析/相对写回、按需校验及 `snapshot`。修改可信域或源文件须使缓存失效；只改 spacing 元数据且已对齐数组相同，重新计算的距离和带数值相同。复跑本任务和 `python -m unittest -v tests.test_data`。按 split 汇总实际文件、结构和可信相位覆盖；Spec 的 63 个系列/640 个 CT 相位/149 个配对 mask 仅用于对照清单版本。

### 任务 3　从实际采集记录编码几何

**文件与接口**　新增 `rstar4d/data/acquisition.py`、`tests/test_cardiac_geometry.py`，连接 `rstar4d/data/contracts.py`。提供 `angle_summary(angles_rad, state_label, phase_labels, *, selected_indices=None, total_views=None) -> np.ndarray` 和 `load_geometry(record, phase_labels) -> np.ndarray`，返回 float32 `[T,14]`。

- [ ] 写 `test_equal_counts_different_angles`、`test_cross_zero_acquisition_order`、`test_optional_geometry_io`。两组同数异角的 histogram 不同，数量分量相等；350°→0°→10° 的弧内最大间隔为 `10/360`。打乱相位 ID 和输入标签顺序检查输出行；关和纯图像病例无投影文件也能读取，geometry 缺来源时错误包含 case ID。
- [ ] 运行 `python -m unittest -v tests.test_cardiac_geometry`。geometry 仅读 `angles_rad/state_label` 两个 HDF5 数据集，或校验过来源哈希及编码版本的摘要。统一零点和正方向，按采集轨迹展开角度，12 个 30° 箱按相位内视角数归一化，另加该相位投影数/共同扫描总数及弧内最大相邻间隔/360°。检查数组长度、有限值、缺失相位与索引归属。按病例缓存摘要；`image_only` 由模型送固定 14 维常量。
- [ ] 先在完整采集轨迹上展开角度，再按相位及实际选中索引取差分；两个稀疏远隔视角也要保留真实弧内间隔。测试用 HDF5 访问记录确认未索引 `mixed_line_integral`。显式 `validate-data` 保留完整性检查；正常训练/推理按需访问对应附属文件，几何摘要绑定原始采集哈希。相位增强同步置换摘要和所有时序字段。复跑本任务测试。

### 任务 4　实现中心块、扩展块与共用采样网格

**文件与接口**　新增 `rstar4d/data/context.py`、`rstar4d/models/sampling_grid.py`、`tests/test_cardiac_context.py`；修改 `rstar4d/data/sampling.py` 的 `PatchDataset/validation_samples`。共用 `plan_window(volume_shape, core_origin, core_shape, recipe) -> Window`、`check_support(window, recipe, source_fov, case_id) -> None` 和 `pull3d(source, coords_zyx, valid) -> (values, valid_samples)`。Window 用整数三元组表示半开区间，batch 中用整型张量存端点，避免默认 DataLoader 拼接 Python slice。

- [ ] 写 `test_aligned_window_and_replay`、`test_missing_halo_vs_real_fov`、`test_trilinear_support_and_units`。覆盖奇数空间尺寸、扫描边缘、`small_volume=pad`、人工 padding、FOV 孔洞。整数索引在最后一个合法格点可用；小数采样只要任一非零权重点无效就屏蔽。故意少读一层合法源支撑，错误包含病例、轴、缺失宽度。
- [ ] 运行 `python -m unittest -v tests.test_cardiac_context`。沿用现有 `(seed,epoch,index,stage)` 确定动物、状态序列及 C，并在 C 上检验 `min_valid_fraction`。按轴取单侧 `ceil(max_displacement_vox + lncc_radius_vox + support_margin_vox)`；由全八组最大读域推导余量，覆盖参考梯度差分、插值、M 取用的 x2 编码感受野、P 粗搜及七点细采样/编码需求。用算子核半径乘所在层步长累加，平行路径取最大，把逐轴推导数值写入 YAML 和测试。W 起止向外对齐 `(2,8,8)` 并截到体积；对齐后仍不足时区分真实 FOV 外与漏读。采样前用允许的**全部**位移范围检查支撑，与本轮模型输出无关。
- [ ] 固定 batch 合同：`input[2,T,*W]`；`context` 含相位顺序、全幅尺寸、W 原点、中心切片端点、逐状态 `source_fov[T,*W]`、按生成规则确定的 `prior_fov[*W]`、按需 `geometry[T,14]`。共享 FOV 病例记录采用的共同域。训练另给 `target/valid[1,T,*W]`、按需参考梯度描述与有效位、两带/可信表面及源距离支撑。参考描述在完整网格计算或用带差分余量的等价读取；记录实际 C/W、左右宽度。验证样本存 C 且可精确还原 W。已有 temporal roll 必须同步相位 ID、角度、FOV、标注、配对与监督，默认猪数据继续关闭。
- [ ] 统一 FP32 采样坐标，为 `align_corners=False` 按轴用 `2*(i+0.5)/n-1` 转归一化 XYZ；`x2` 有效位取原图中心 `(2z,4y,4x)`。场用 W 的全局原点和步长索引，上采样只改变采样密度。对各方向注入常量位移，检查 `(2,4,4)` vox ↔ `(1,1,1)` 特征格，返回原图仍为 `(2,4,4)`。复跑本任务及原数据测试。

支撑检查使用病例的**全幅** FOV 或其等价查询索引；前向 context 只传 W 内有效位。为任务 7 的监督数据提供病例/相位到全幅 FOV 的可重开句柄或路径，避免将整幅域复制到每个 GPU batch。`validation_samples` 只对中心块有效比例不足重采样；支撑缺失、标注损坏、几何缺失须保留具体病例错误，避免被现有的通用 `except ValueError: continue` 吞掉。

### 任务 5　实现 M 的相邻状态拉回与门控

**文件与接口**　新增 `rstar4d/models/motion.py`、`tests/test_cardiac_motion.py`。`AdjacentMotionFusion.forward(dynamic, F, context, return_aux=False) -> (delta_m, aux)`。输入动态图为 `[B,1,T,*W]`，主干特征为 `[B,64,T,*S]`，修正与 F 同形。共用任务 4 的 `pull3d`。

- [ ] 写 `test_directed_pairs_and_no_wrap`、`test_identity_and_masked_neighbors`、`test_learned_pull_gradient`。T=2/10/11 产生 2/18/20 个有向场，首尾各一个邻居；不同轴的非重复纹理验证 `x+u` 的方向。目标无效或无合法邻居时 `delta_m==0`；双邻居有效且初始化时 `a_self≈0.965`。
- [ ] 运行 `python -m unittest -v tests.test_cardiac_motion`。按 Spec M1–M5 实现：动态图逐状态转 `[B*T,1,*W]`，3³ 编码 `1→8→16`，stride `(2,2,2)/(1,2,2)`；`cat(E_t,E_s,abs(E_t-E_s))` 经 `48→24→16→3`，小权重、零偏置的线性末层接逐轴 `max_disp*tanh`。除以 `(2,4,4)` 后从邻状态原始 F 插值。共享 1³ `64→8` 压缩，两端及差值经 1³ `24→8→1` 打分，末层权重 0、偏置 −4；自身 logit=0，屏蔽无效邻居后 softmax。输出 `Σ a_s*(A_s-F_t)`。新增隐藏层用 `LeakyReLU(0.2)`，3³ padding=1，1³ padding=0。
- [ ] `identity` 跳过位移预测，按零场拉回，保留同一借用门控；不计位移损失。`learned` 按有序标签位置定义固定 pairs 顺序，每对同时记录两个方向；需要 aux 时返回 `pairs[N,2]`、`flow_vox[B,N,3,*S]`、采样有效位、全局原点/步长；每例全空源域在运动监督配对时排除。复验有限反向梯度、两端不同方向独立估计及无 aux 路径。

### 任务 6　实现 P 的粗匹配、七点采样和角度门控

**文件与接口**　新增 `rstar4d/models/prior_fusion.py`、`tests/test_cardiac_prior.py`。`GeometryPriorFusion.forward(dynamic, static_prior, context) -> delta_p`，静态输入 `[B,1,*W]`，输出 `[B,64,T,*S]`。P 内动态和静态共享空间编码参数，M 使用独立实例。

- [ ] 写 `test_single_candidate_keeps_raw_cosine`，构造原始余弦 +0.9、−0.8 两个单候选，断言两次 `s_rel==1` 且用 forward hook 读到 P5 的第 57 个通道分别为 +0.9、−0.8。`test_zero_and_partial_support` 验证零粗候选返回零、细点越界只屏蔽该点；`test_prior_encoded_once` 计静态编码次数，比较分块与未分块的小张量结果及梯度。
- [ ] 运行 `python -m unittest -v tests.test_cardiac_prior`。按 Spec P1–P3 复用 M1 的层规格，静态 K/V 每块计算一次。独立 1³ Q/K `16→8` 并 L2 归一化，3³ 搜索 27 个候选；argmax 取 `p*`，保留胜者有符号余弦 `r` 和温度 0.2 的有效候选 softmax 权重 `s_rel` 的计算图。V 为 `16→16`。以匹配两端及绝对差形成 `H[48]`，经 3³ `48→16`、1³ `16→21`、tanh，从中心及 ±Z/±Y/±X 七点加学习偏移采样，微调上限 `(2,4,4)` vox。
- [ ] 实现 P4–P6：`14→16→8` MLP；将 `H,e_t,r` 拼成 57 通道，经 1³ `57→16→7` 和 sigmoid 得 `m_j`。七组逐通道 `w_j[16]` 初值 1/7，对有效 `V_j` 求 `s_rel*Σ(m_j*w_j*V_j)`，用小幅非零初始化的无偏置 1³ `16→64` 输出。几何只进入 P5；`image_only` 给同一 MLP 全零常量且不读取角度。对无效目标清零。按状态/空间分批做局部索引，带足邻域，避免全序列展开 27 份值特征。复验单候选、零候选无 NaN、梯度与静态编码复用。

### 任务 7　实现保留分母与位移梯度的运动监督

**文件与接口**　新增 `rstar4d/training/motion_losses.py`、`tests/test_cardiac_motion_losses.py`。`motion_terms(aux, supervision, context, recipe) -> dict` 输出 `struct/smooth/surface` 的可微标量、各自计数与失败类别。三个项读取同一全局场、同一 pairs 顺序，所有坐标和归约用 FP32。

- [ ] 写 `test_known_shift_across_core`、`test_outside_fov_preserves_denominator_and_gradient`、`test_flat_source_and_empty_pair`。在各向异性、非重复纹理体模上把真实对应放在 C 外、W 和 FOV 内，向 M3 和损失采样器注入同一真值场。断言 M3 能读到该点、`d_out==0`、`L_struct(true)<L_struct(zero)`；全幅和 W 使用同一参考描述、中心集合和场，窗口数严格相等，采样与损失用 `atol=1e-5, rtol=1e-4`。整窗移出真实 FOV 后分母相等、损失高于恒等对齐，沿独立平移参数的负梯度指向合法域。
- [ ] 运行 `python -m unittest -v tests.test_cardiac_motion_losses`。以全幅参考 1 vox 三轴梯度幅值和有效差分域构成描述。先校验 W 的全部支撑，再按目标有效域、边缘及窗口方差固定 C 内窗中心。用每个目标窗点的场拉回源描述；完整合法且有对比度时用有符号 `1−clip(LNCC,−1,1)`，源视野越界或源低方差均计 2 并保留窗口。只在目标无窗或源全空时跳过对应配对。把低对比度、真实越界、跨 C 有效、缺失支撑及合格窗数分开记录。
- [ ] 实现 `outside_distance_vox(points_global, source_domain) -> Tensor`。合法采样为 0，非法位置相对于**完整源域**（含内部孔洞）查询最近合法体素中心 q。可按病例/相位缓存全幅有效中心的空间索引，对失败点分批精确查询；停止 q 的梯度，保留 `norm(p-q)` 对 p 的梯度。每窗额外加 `0.1*mean(d_out)/2`。用棋盘孔洞和非整数查询与小体积穷举最近点对照；单改 spacing 不改变代价。
- [ ] 平滑项仅在两端均在 C 且有效的 x2 相邻格点上求位移绝对差，各轴除以 `2/4/4`。可选表面项先按同结构可靠双端对应、目标 C、源可信区域和 W 支撑固定每方向点集，然后分别从全幅生成的源距离图采样，截断 `[0,d]`。越界计 `d+0.1*d_out`，保留分母；双向均值再平均，缺任一方向跳过。用常量场测试 smooth=0、线性场验证轴步长，整面移出仍受罚；单端标注只供 B。复跑本任务。

### 任务 8　实现双边界结构平均和梯度对照

**文件与接口**　修改 `rstar4d/training/losses.py`，新增 `tests/test_cardiac_boundary.py`。`boundary_terms(pred_c, target_c, bands_c, structure_weights, grad_weight_vox) -> dict`；`boundary_scale(epoch, ramp_epochs, weight) -> float`。输入网络归一化强度和 `[B,2,T,*C]` 的可信带；统计另保留病例、动物和相位标识。

- [ ] 写 `test_structure_mean_and_overlap`、`test_signed_gradients_and_sparse_pairs`、`test_epoch_ramp_and_unlabelled_gradient`。用大小不等且不交叠的带分别填常数误差 1、3，断言总带损失为 2；再覆盖交叠、单带、双空。预测等于参考时 `G_k=0`，模糊阶跃后 `G_k>0`，孤立带点保留灰度项；未标相位的重建梯度仍存在。
- [ ] 运行 `python -m unittest -v tests.test_cardiac_boundary`。用 C、真实 FOV、标注可用性及可信域与 `distance<=2` 取交集。每个结构先在本 batch 的有效体素上做均值，再按有效结构权重归一化；交叠点进入各自均值，B 的总系数只乘一次。双空返回同设备可微零。可选 `G_k` 只用同相位同结构中两端都有效的 Z/Y/X 正向相邻体素对，比较有符号差分，每步 1 vox。
- [ ] Stage II 第 1–5 轮系数依次为目标值的 `0, 0.25, 0.5, 0.75, 1`，以后固定；断点用实际 epoch。逐结构记录有效相位、体素、梯度对数、灰度/梯度原值与贡献，单列交叠和空监督。复验权重 0.5 的序列为 `[0,.125,.25,.375,.5,.5]`。

### 任务 9　接入正常训练、验证、迁移与断点恢复

**文件与接口**　修改 `rstar4d/models/{model,official_stage2,__init__}.py`、`rstar4d/training/{trainer,losses,checkpoint,pipeline}.py` 和 `rstar4d/cli.py`，新增 `tests/test_cardiac_training.py`。保留 `create_model(stage,profile)`、`initialize_stage2(model,state)` 旧语义，新增 `attach_stage2_extensions(model, resolved) -> None`、`model_from_checkpoint(ck) -> nn.Module`。扩充 `forward_volume(model,x,*,context=None,return_aux=False)`，默认返回张量，请求 aux 时返回 `(pred,aux)`。

- [ ] 写 `test_off_matches_notebook_loss_and_gradients`，用现有 `tests/test_model.py` 的 Notebook 和公开 state_dict 作独立对照。在同设备确定性 FP32、同输入/权重下对照 Stage I、Stage II 输出、原损失、一次反向后的所有具名梯度；Stage II 全关仍有 232 个 state_dict 键、2,118,693 个参数。另将原主干在同一 W 前向、同一 C 计分，与八组的全关入口比较。B 单开参数键和参数数均与全关相同。
- [ ] 运行 `python -m unittest -v tests.test_cardiac_training tests.test_model`。在官方 Stage II 的 `x2=conv3_3(conv3_2(x2))+x2` 后保存 F，分别用原始动态图/F 和单份静态 prior 计算 `enhanced=F+ΔM+ΔP`。将同一 enhanced 传给 `conv4` 及 `cat([conv5_output,enhanced])`；测试用 hook 核对两处。仅启用分支时注册模块。保留原末端 replicate padding 和输出裁切，空间 context 同步补假、aux 记录真实网格原点/步长、有效范围；默认张量返回及 T≥2 行为可回归。
- [ ] 新训从 Stage I 时先在裸 Stage II 上 `initialize_stage2` 严格转移并完成原 Z 初始化，再按固定的独立 M/P 种子挂载，最后创建 Adam。加载同构断点时先按其架构创建完整模型，严格还原模型、优化器、scheduler、scaler、RNG、next_epoch、best 和验证样本。跨结构只走显式 `--init-from-stage2` 新 run，主干严格校验，共享且兼容的分支按名称/版本/形状加载，清单列出加载、跳过和新建参数。旧 checkpoint 缺扩展字段按全关构造；继续执行既有签名及 run 身份校验。
- [ ] 在 train 和 validate 共用 W 输入准备及 C 输出切片。原 `compute_loss(pred_c,target_c,valid_c,stage,loss_config)` 处理所有相位；仅有实际使用位移的损失才请求 aux。按生效系数组合任务 7、8 的项。验证仍按各病例有效 C 上 L1 再病例等权选 `best.pt`，同时保留 animal L1；全空辅助监督的轮次打印 0 和实际计数。日志含原值、计数及加权贡献，按结构/动物/相位汇总，含 H、LV、两者均命中及总中心块数。
- [ ] checkpoint 保存有效架构及版本（通道、搜索窗、温度、位移上限、角度编码、`distance_unit: output_voxel`）、配置与分支种子；训练签名包含生效配方、源/附属文件哈希、split 和已有代码指纹。实施 `test_all_eight_update_reload_and_resume`：每组实际跑一次参数更新、存盘、严格重载并比较输出；比较 epoch 边界续跑与连续训练。验证换组被 resume 拒绝、M 无标注、P 纯图像无角度、同名分支初始化跨组相等、有有效样本时梯度有限且分支可更新。再跑 `python -m unittest -v tests.test_cardiac_training tests.test_training tests.test_cli tests.test_model`，用短配方覆盖 `train` 和两阶段 `pipeline`。

### 任务 10　接入整幅与滑窗推理，实测三档显存

**文件与接口**　修改 `rstar4d/inference/{predict,sliding_window}.py`、`rstar4d/cli.py`；新增 `tests/test_cardiac_inference.py`。扩充 `predict_volume(..., *, context=None, recipe=None)`；`predict_manifest` 从 checkpoint 构造模型及裁块策略。为现有 `benchmark` 增加 `--manifest/--split` 真实病例模式，共用训练的 batch、loss 和优化器路径。

- [ ] 写 `test_tiled_core_coverage_on_odd_volume`、`test_full_tiled_and_prediction_only_manifest`。用带全局坐标的恒等体模验证四边、最后一个 tile、内部和接缝，所需真实体素覆盖权重均大于零，恢复顺序和全幅一致。用实际模型在小病例分别输出 full/tiled，记录中心内部和重叠接缝的 MAE 与最大绝对误差。B 单开只用稀疏序列、prior 及基本采集元数据；无 ROI、无 target 的 manifest 应能导出重建。
- [ ] 运行 `python -m unittest -v tests.test_cardiac_inference tests.test_inference`。full 与 tiled 均读每块全部状态、真实 FOV 和按需的同例角度摘要，调用 `forward_volume(...,return_aux=False)`。新裁块模式把 tile 解释为 C，复用训练的 `plan_window` 读 W，仅把 C 对应的真实区域按既有 `gaussian_weight` 拼接。新模式未显式给 tile 时使用 checkpoint 记录的中心块及合法重叠；旧 checkpoint 保留原 CLI 默认和旧路径。输出物理强度 `[T,Z,Y,X]`，保存配方、模型版本与来源。先验证坐标体模精确覆盖，再分析实际网络接缝误差。
- [ ] 用 `benchmark --stage 2 --config ... --manifest ... --split ... --steps 2 --out ...` 在 16 GB GPU 以实际训练配方逐一测三种 C 及对应的 W。选含有效监督的真实中心，覆盖最长纳入序列和内部大 W，记录 case、T、C/W、各侧 halo、启用参数量、AMP、优化器更新、峰值 allocated/reserved。每档使用真实前向、完整损失、反向及 Adam 更新，并在预热后同步 CUDA 计时。单序列推理另按相同设备/模式计时。旧 benchmark 的随机中心块结果只作历史参照。
- [ ] 三档及全开组均须给出实测结果。若 OOM，优先分状态/空间处理局部匹配、复用 prior K/V，检查分批实现的前后向等价与实际内存；仍超限时统一缩小 C、保留必需 halo，检查核心非空且有有效监督，重新锁定并重跑全部比较组。记录失败尺寸及最终可运行尺寸。复跑本任务测试及 `python -m unittest discover -s tests -v`，记录 CUDA/TIGRE 跳过项和实测设备。

新配方的默认推理中心块取锁定列表第一档，重叠取各轴约一半并对齐 Stage II 网格，记录实际取值。显存测试让 B 达到目标权重，并核对 M 的有效参考窗及两条带的计数；确无两带同时命中的病例时，分样本覆盖并随结果列出计数。

### 任务 11　建立逐结构、按动物配对的独立轮廓评价

**文件与接口**　新增 `rstar4d/evaluation/contours.py`、`tests/test_cardiac_evaluation.py`；修改 `rstar4d/evaluation/report.py`、`rstar4d/config.py`、`rstar4d/cli.py`。给 `evaluate` 增加可选 `--contours` 独立预测轮廓清单；以 `(case_id,phase_label,structure,prediction_sha256)` 关联重建，记录分割器版本/权重哈希与阈值，或盲法勾画批次。评价参考来自任务 2。默认评价继续输出原图像指标。

- [ ] 写 `test_unit_surface_metrics_and_empty_predictions`、`test_pair_by_animal_and_phase`。用相同表面得 Dice=1、HD95=ASSD=0；平行平面相差一格，两个距离均为 1 vox；改变已对齐数组的 spacing 元数据后结果相同。加入多病例同动物、缺相位、局部可信表面、空预测和无参考的样例，核对配对键、失败数和分母。
- [ ] 运行 `python -m unittest -v tests.test_cardiac_evaluation`。提供 `surface_metrics(pred_surface, ref_surface, tolerance_vox) -> dict`，在双方有可信覆盖的完整输出网格上取表面，剔除未知与 FOV 截断面。按单位体素中心定义三项指标并版本化：Surface Dice 为双向距离 ≤ 各结构预设容差的表面点数占比；HD95 取两个方向各自 95 分位的较大值，ASSD 取双向距离总和除以总表面点数。有可信非空参考而预测空记失败，Dice=0、距离记 null，并计入失败分母；缺可信参考记不可评价。将此口径和两结构容差在看测试集前写入 metric 配置。
- [ ] 输出 `contours_by_phase.csv`、`contours_by_animal.csv` 并扩展 `summary.json`。分别报告 H、LV 的 Surface Dice、HD95、ASSD、带内 MAE，全图有效域 MAE/SSIM 复用既有评价；附容差、原 spacing、`distance_unit: output_voxel`、有效动物/状态数和失败数。先按共同动物/病例/相位/结构对齐各组，逐动物汇总配对差值，再动物等权汇总，同时列明无法配对的原因。有完整闭合标签才报告体积，多期只计算实际标注相位间变化。核对来源哈希避免将人工参考直接充作预测轮廓。复跑本任务测试。

### 任务 12　执行八组实验与预先定义的机制对照

**文件与接口**　新增 `experiments/cardiac_ablation.py`、`tests/test_cardiac_experiments.py`；扩展 `rstar4d/simulation/pipeline.py` 的混合投影稀疏 FDK 路径。实验脚本生成配置、对照数据和配对报告；实际训练、推理、评价均调用正常入口。各组用独立 run 目录。

| 组 ID | motion.mode | prior.mode | boundary.enabled |
|---|---|---|---|
| baseline | off | off | false |
| M | learned | off | false |
| P | off | geometry | false |
| B | off | off | true |
| MP | learned | geometry | false |
| MB | learned | off | true |
| PB | off | geometry | true |
| MPB | learned | geometry | true |

- [ ] 写 `test_eight_configs_change_three_fields_only`，断言 8 个唯一 ID、8 个输出目录、八份预设 YAML 除三开关外相同；固定种子回放 C/W、相序与训练/验证抽样记录逐项一致。`test_angle_layout_changes_one_phase` 验证两子集数量相等、布局不同且每个索引均属所选相位，prior、target、其余状态输入及共同扫描分母相同。运行 `python -m unittest -v tests.test_cardiac_experiments`。
- [ ] 实现 `write-matrix --base ... --out ...` 与按共同键比较任务 11 产物的 `compare --runs ... --out ...`。锁定同一 Stage I best 的 SHA256、split、种子、prior 类型、主干时间拓扑、样本数/顺序、C/W、轮数、优化器、验证样本及选优准则。训练/验证预实验选定非零结构、平滑、B 权重并记录损失和梯度量级；所有组使用同一锁定版本。保存每组的配置和数据哈希。
- [ ] 先在体模跑八组短流程，验证训练 → 恢复 → 推理 → 独立轮廓评价和比较可闭环。正式实验复用一次 Stage I，各组从该权重新建 Stage II。下例假定已按本计划生成训练清单、动物划分、Stage I 权重及独立轮廓清单，`$manifest/$split/$stage1/$metrics/$contours` 为相应路径，`$contours` 对应当前组预测。PowerShell 中依次运行：

  ```powershell
  python -m experiments.cardiac_ablation write-matrix --base configs/train/pig_cardiac_ablation.yaml --out runs/cardiac/configs
  $group = 'MPB'
  $run = "runs/cardiac/$group"
  python -m rstar4d train --stage 2 --init-from-stage1 $stage1 --config "runs/cardiac/configs/$group.yaml" --manifest $manifest --split $split --run-dir $run
  python -m rstar4d infer --checkpoint "$run/checkpoints/best.pt" --manifest $manifest --split $split --split-name test --mode tiled --out "$run/predictions"
  python -m rstar4d evaluate --predictions "$run/predictions" --manifest $manifest --metric-config $metrics --contours $contours --out "$run/evaluation"
  ```

  对表中八个组分别执行后三条命令。每组 infer 完成后，从该组重建用同一冻结分割器及阈值生成轮廓，或完成同一盲法流程，然后传入该组的 `$contours`。中断后使用同组的 `train --resume "$run/checkpoints/last.pt"`。先报告 MPB−baseline、M−baseline、MPB−PB，再报告完整模型减去 P、B 的配对比较及其他组合。各组附参数量、训练峰值显存、单序列推理耗时和监督覆盖。

- [ ] **M 对照。** 固定 P/B，比较 off、identity、learned。若需区分学习位移和辅助监督，再训练 learned 且 `struct_weight=smooth_weight=0`。可靠双端标注充足时单列 `surface_weight=0` 与非零，保持训练样本及三个主开关相同。首轮结论限于重建与轮廓；固定解剖、只改造影的组织运动实验置于后续。
- [ ] **P 对照。** 固定 M/B，比较 image_only 与 geometry。按所选相位从同一混合投影池取两个等量异角布局（可重叠），沿用共同扫描的 prior/参考/其他状态，用相同 FDK 及校准仅重建该状态。分别保存选中索引、来源哈希、布局 ID 和分母；几何只统计实际进入 FDK 的索引。用多布局训练、留出布局及动物测试；在同一已训练几何模型交换两份摘要作为诊断。常规整套采集测试各用本次采集得到的 prior。
- [ ] **P 单候选诊断。** 在验证集对同一训练后模型固定目标、几何和单候选域，替换为解剖匹配/明显错配的 prior 局部块，配平有效细点数和值特征幅度。每对保存 `r,s_rel`、有效门控均值及 `RMS(ΔP)`，按动物报告差值。错配修正未减小也如实记录；用此项检验错配抑制假设。
- [ ] **B 对照。** 复用 baseline 与 B，新增 B 单开 + 带内梯度差一组；用训练/验证动物选定正的 `grad_weight_vox` 后锁定。保持两结构等权、2 vox、相同升权日程和样本；按动物分别比较 H/LV 的 Surface Dice、HD95 和全图质量。不同带宽、边界重采样及双端表面另列实验。
- [ ] 复跑本任务和全套测试。按 `compare --runs runs/cardiac --out runs/cardiac/report` 汇总。交付所有组的生效配置、checkpoint 来源、动物配对表、失败/缺失数及运行资源；留出动物的独立分割或盲法轮廓缺失时，将该结构标记为待评价，单列已可计算的图像指标。

## 最终验收索引

| Spec §7 | 核对位置与证据 |
|---|---|
| 1 基准等价 | 任务 9，Notebook 同权重前向/损失/逐参数梯度，旧裁块及共用 C/W 各一套 |
| 2 开关独立 | 任务 5、6、8、9、10，八组更新与完整重载、缺附属文件、推理无 GT |
| 3 空间正确 | 任务 3、4、5、7，异角、三轴平移、无效候选、固定分母与回拉梯度 |
| 4 双边界 | 任务 2、7、8、11，可信表面、2/3 vox、结构归约、可选项、spacing 不变性 |
| 5 正常入口与资源 | 任务 9、10，pipeline/train/断点/full/tiled，三档 C/W 真实训练峰值及接缝结果 |
| 6 研究结论 | 任务 11、12，两结构留出动物配对表、纯图像/真实角度对照和解释范围 |
| 7 prior 单候选错配 | 任务 6 确认原始余弦进入 P5；任务 12 提交训练后配对诊断 |
| 8 跨块运动 | 任务 4、7，逐值、逐窗对照全幅；分别验证真实域外和 W 缺支撑 |

**编写计划时已验证**　现有 `python -m unittest -v tests.test_model` 的 4 项测试通过。上文各项复验、GPU 测量和动物实验均为实施阶段的验收工作。
