# RSTAR4D 全相位运动补偿设计规格

**基线与范围。** 分支 `rstar-dpi-moco` 从猪心训练工程 `proj2` 的本地 `rstar@a64867abe027fc6ba8b63a61630ccba40124613d` 建立。在该分支已有的 RSTAR4D Stage II、TIGRE 仿真、动物划分和 HDF5 推理流程上扩展输出端。A 组直接运行工程基线。设计日期为 2026-11-25。下文标为拟新增的功能与实验均待实现。

**代码依据。** 基线的 [训练及推理说明](README-training.md)、[模型入口](rstar4d/models/model.py)、[预测清单与 checkpoint 校验](rstar4d/inference/predict.py) 和 [滑窗推理](rstar4d/inference/sliding_window.py) 定义了增强图的数值行为。原作者 Notebook 位于 [RSTAR/githubcode](RSTAR/githubcode/Training_Tetris2.ipynb)。运动模块参考同一工作区 `proj/DPI-MoCo` 的本地复现提交 `b709c2807b026c192f4194203bb6595fe74af930`。

## 1. 研究问题与输出

原版 RSTAR 用整段稀疏 FDK 和同次扫描的静态先验生成逐相位恢复图。以第 3 相位为例，RSTAR 可能保住左室内缘的位置，同时把一段细小分支平滑掉。DPI-MoCo 提供了检验这一现象的方法：在第 2、3 相位的恢复图上找对应位置，再从第 2 相位保留更多细节的观测中取值。对每个目标相位重复这一过程。是否有可恢复的细节、位移是否可信、融合后目标相位的血池灰度是否准确，分别评价。[RSTAR4D 论文](https://arxiv.org/abs/2403.16361) · [DPI-MoCo 论文，§III-C/D、式（13）–（14）、图 14](https://doi.org/10.1109/TMI.2024.3483451)

同一扫描交付三组同网格、同强度单位的四维图，以便直接比较。

| 输出 | 定义 | 用途 |
|---|---|---|
| $b_t$ | 冻结的原版 RSTAR Stage II 对完整序列预测的第 $t$ 相位 | 原版基线，同时给 RegisNet 提供配准图像 |
| $d_t$ | 将各源相位的 PIGAE 校正图 $c_s$ 拉回目标网格后聚合 | 单独衡量 DPI-MoCo 式图像域补偿，并定位配准或细节源的失败 |
| $y_t$ | 以 $b_t$ 为基础，加入 $d_t$ 相对 $b_t$ 的局部细节修正 | 本方案最终重建，同时评价移动边界和逐相位造影强度 |

DPI-MoCo 的增强图由 RestoreNet 生成。此处由 RSTAR 直接提供 $b$，复用已训练的主干和原输入。主实验训练一个共享的三维 RegisNet；伪平均分箱、RestoreNet 的 GAN/对比损失和末端加噪 DenoiseNet 属于原论文其他阶段。$y_t$ 的融合规则是本方案的实验选择。

## 2. 从同一次扫描构造两路图像

**RSTAR 路径。** 对病例保留有序相位表。$x_t=\mathrm{FDK}(P_t)$ 是第 $t$ 相位的稀疏重建。将该基线使用的先验 $PI_R$ 在相位维复制，与 $x_{1:T}$ 沿通道拼接，输入布局为 `[B,2,T,Z,Y,X]`，输出为 `[B,1,T,Z,Y,X]`。主干按原权重的归一化约定推理，再把 $b$ 转回统一物理强度。每次空间裁块均保留所有相位。主比较中的各组共用同一份缓存 $b$、同一模型权重以及相同的输入和有效域。

**细节路径。** 用同一扫描全部有效混合投影重建先验 $PI_D$。对每个相位读取其真实视角索引 $V_t$、锥束几何和 FDK 配置。对 $PI_D$ 仅在这些视角上做正投影，再用同一组视角重建：

$$
q_t=\mathrm{FDK}_{V_t}(A_{V_t} PI_D),\qquad
c_t=x_t+(PI_D-q_t).
$$

主实验取本地 DPI-MoCo 的 `static_consistent` 约定。静态同几何例中 $x_t=q_t$ 时应回到 $PI_D$。印刷式（8）、（10）对应 $c_t=x_t-(PI_D-q_t)$，登记为 `printed` 复核组。DPI-MoCo 原文的式（14）对贡献求和；本方案基线的 FDK 是逐相位定标的强度图，主结果按有效贡献数归一化。另导出印刷式的未归一化和，并记录贡献数，供核对幅值。不同尺度的产物先标明标定方式再评价。

**对接先验与强度。** 原 RSTAR 的 MATLAB 示例按每相位的视角数加权相位 FDK 得到先验；本地 DPI-MoCo 在全部有效投影上做一次 FDK。清单分别记录 $PI_R$、$PI_D$ 的方法和标识。两者确由同一方法、相同配置生成时可以共用文件。$x_t,q_t,PI_D$ 必须使用相容的坐标、视角权重、补边、裁剪、FOV 与强度标定。若猪数据有 FDK 后的非线性校准，在校准前的物理衰减域计算 $q,c$，再对输出施加与 $x,b$ 相同的后处理；不得把后处理后的强度直接当作线性投影输入。对复用的旧体，先用同源投影重建 $x_t$ 核对，再接受其 $c_t$。已有同源 $x_t,PI_D$ 且保留逐视角几何、分箱及 FDK 配置时，也可直接从 $PI_D$ 计算 $q_t$；已有经过验证的 $c_t$ 则可直接导入。只有图像且无法确定采样算子或校正图来源时，单列 $b$ 与变形 $x$ 的消融。

## 3. 逐对估场与全相位补偿

训练和推理均在**同一扫描**内配对。对于每个目标 $t$ 和每个源 $s\ne t$，计算 $u_{t\leftarrow s}=R(b_s,b_t)$，调用顺序为 `RegisNet3D(moving=b_s, fixed=b_t)`。$u$ 定义在目标网格，表示从目标体素到源图采样坐标的偏移。体数组顺序为 `[Z,Y,X]`，场为 `[B,3,Z,Y,X]`，三个通道依次是 $(dx,dy,dz)$，单位为对应轴的体素。以目标到源的拉回定义

$$
W(v_s,u)(p)=v_s(p+u(p)).
$$

例如源图血管在 $x=44$、目标在 $x=40$，目标位置 40 应预测 $dx=+4$ 并读取源位置 44。复用本地 `warp_pull` 的三线性插值和 `align_corners=False`。每个方向单独前向。自身相位取恒等场，恰好贡献一次。10 相位包含 90 个非自身有序对，11 相位为 110 个。只取相邻相位可作为造影数据的对照，采用有序前后邻居并记录首尾处理；主实验使用全部相位。

令 $m_{ts}(p)$ 表示目标体素有效、源采样点在界内且有完整有效插值支撑。以零填充采样源有效域并检查坐标，避免本地 `padding_mode="border"` 将越界复制值计作证据。目标有效域内，

$$
N_t(p)=\sum_s m_{ts}(p),\qquad
d_t(p)=\frac{\sum_s m_{ts}(p) W(c_s,u_{t\leftarrow s})(p)}{N_t(p)} .
$$

保留 $N_t$ 图与目标有效掩膜；目标域外导出填充值并从评价中排除。正常有效点的自身贡献保证分母至少为 1；若输入缺失自身相位，报告病例和相位并停止该病例。按目标和源逐对计算，在线累加分子、分母，默认只保存配准质控摘要；检查某一配对时可单独导出 DVF 与变形图。

## 4. 怎样把细节加回 RSTAR

猪心造影中，即使室壁完全对齐，血池衰减也可能随状态改变。因而分别保留纯补偿图 $d_t$ 和以当前状态为基础的 $y_t$。首轮使用可复现的固定滤波与单个系数，不引入额外可训练的融合网络。

定义 $G_\ell$ 为物理空间标准差 $\ell=1.5$ mm 的三维高斯低通，按 spacing 换算各轴核宽，截断于 $3\ell$。在目标有效且 $N_t\ge2$ 的支撑 $S$ 上，以 $G_\ell*(Sr)/(G_\ell*S)$ 计算归一化低通。令 $r=d_t-b_t$，输出为

$$
y_t(p)=b_t(p)+\alpha\big[r(p)-G_{\ell,S}(r)(p)\big],\quad p\in S .
$$

目标有效而缺少其他相位支撑时取 $y_t=b_t$。$\alpha=0$ 精确复现原 RSTAR。$\alpha$ 从 $\{0,0.25,0.5,0.75,1\}$ 在验证动物上统一选一次。低通差分使均匀区域的常量强度偏差不进入输出；边界附近仍可能传入造影锋面或残余条纹，因此灰度曲线与空间轮廓均参与验收。高通规则、带宽和系数是本项目新增的融合假设，$d_t$ 是独立对照。

## 5. 训练、推理与工程接口

1. **准备病例。** 登记 `animal_id/patient_id, scan_id, split, phase_labels, view_id, angle, geometry, shape_zyx, spacing_xyz_mm, origin, direction, units, calibration, valid_mask`。真实相位标签可为 1 基或非连续值，显式映射到内部索引并保留逆映射。检查每个使用相位视角非空。RSTAR 与全相位配准均支持按实际 $T$ 编排；10、11 相位分别验收。配准训练只用真实分箱，现有 DPI 准备器可按 `training=False` 生成这条路径，无需伪平均分箱的 `cycle_id`。
2. **冻结并缓存增强图。** 按动物划分训练、验证和测试，同一动物的轮次、姿态、派生视角进入同一 split。所有组复用同一份 RSTAR 基线；若需在猪域训练 RSTAR，用训练动物的逐相位参考训练，记录域、模型和强度变换。导出的 $x,c,b,d,y$ 均为 float32，与病例登记的 `mu` 或 `calibrated_mu` 强度域一致（mm⁻¹）；这两个域分别汇总。各网络的输入变换由各自 checkpoint 约定；逐病例 min–max 会破坏跨相位曲线比较。
3. **单独训练配准。** 取冻结 $b$ 的有序相位对；优化 $L=\mathrm{NCC}_{loss}(W(b_s,u),b_t)+\sigma\,\mathrm{Smooth}(u)$。沿用本地实现的负局部平方归一化互相关（$9^3$ 窗）与三轴位移一阶差分平方均值，$\sigma=4$ 为论文起点。Adam、batch 1、50 轮，学习率 $10^{-4}$ 降至 $10^{-5}$。每轮打乱训练配对，只在验证集选权重。原损失含边界填充，另记有效视野内的匹配程度、超界率与折叠率，以便判断损失下降的来源。模拟参考及人工轮廓只按其声明的训练或评价用途加载，推理输入限于同次扫描的可获取数据。
4. **适配现有代码。** DPI 本地实现 `dpi_moco/models.py::RegisNet3D` 负责估场，`losses.py::{warp_pull,local_ncc,field_smoothness}` 负责重采样与损失，`physics.py::{TigreProjector,pigae}`、`prepare.py::prepare_scan` 可提供真实分箱校正体。`training.py::Samples`、`inference.py::{enhanced_paths,load_weights,fuse_target}` 目前把增强图来源绑定到 RestoreNet 权重；新分支改为读取 RSTAR 增强图清单，把权重依赖校验改为 RSTAR 提供者及其指纹。融合函数增加有效贡献数及 $y_t$。移植或显式固定版本依赖，运行时按配置定位资源。旧的 RestoreNet 配套配准权重不能直接当作已在 RSTAR 输出上训练的权重。
5. **结果可追溯。** 按病例保存 $b,d,y$、相位顺序、有效域、贡献数和质控摘要。缓存键包含扫描/几何、视角映射、先验与校准方式、网格、源数据、模型及配置指纹。增强图缓存变化后重新校验配准权重的来源。按目标相位写临时产物，完整后再发布，便于中断续跑。若完整体积超出显存，先沿用已核对的分块或低分辨率方案；分块需共享全局坐标并留足采样上下文，重采样场时同步换算各轴位移单位。同一病例和小体模上核对整幅/分块有效内区和接缝后再比较质量。

### 在工程分支上的接入位置

| 环节 | 已有入口与数据 | 拟新增的最小衔接 |
|---|---|---|
| 准备输入 | `simulate-pig` 或 `import-paired` 生成病例，`validate-data` 校验。每例 `paired.h5` 包含 `cbct[T,Z,Y,X]`、`prior[Z,Y,X]`、`valid_mask` 及网格元数据。开启 `save_projections` 时还有 `projections.h5`。 | 拟新增 `rstar4d prepare-moco`。按病例与原始相位标签连接源数据，逐视角使用 `mixed_line_integral, angles_rad, state_label, projection_index`。用同次扫描计算 $c$，写入有源清单指纹的独立产物。已校准的配对数据若缺少同源物理输入，可按 §2 的条件导入校正体。 |
| 生成增强图 | `rstar4d infer` 的 `predict_manifest` 以版本化 Stage II checkpoint 加载权重，调用 `predict_volume`；输出每例 `prediction.h5/prediction` 和全批次 `manifest.jsonl`。 | 直接把这组预测作为 $b$。分别用同一 checkpoint 和已登记的动物划分执行 `--split-name train`、`val`、`test`，写到各自目录。缓存记录原有的 checkpoint、病例与元数据哈希，以及 full/tiled、tile/overlap、AMP 等推理配置。 |
| 训练运动场 | 原 `pipeline` 负责 Stage I/II，猪域模型按其强度合同推理。 | 拟新增 `rstar4d train-register`，从训练/验证增强图清单读取 $b$。注册模型独立保存配准参数与断点，依赖字段标记 RSTAR 提供者、Stage II 权重指纹和增强配置。 |
| 运动补偿与评价 | `rstar4d evaluate` 接受含 `prediction` 数据集的预测 HDF5，核对病例数据、源元数据与产物哈希，再汇总 MAE、RMSE、SSIM。 | 拟新增 `rstar4d infer-moco`，复用 $b$ 与校正体，分别为 $d$、$y$ 写与现有评价器兼容的预测清单和 HDF5，保留 $b$ 原清单。另存每目标相位的贡献数、有效域和场质控；在共享区域计算跨组比较，并按 §6 增加曲线、轮廓及 PSNR 评价。 |

**尺度与几何。** 增强路径使用工程 checkpoint 的 `model_profile, normalization, intensity_domain, calibration_application`；`predict_volume` 按 $(v+\mathrm{offset})/\mathrm{scale}$ 归一化，输出端执行逆变换。RegisNet 读取物理域的 $b$ 后使用自己的强度变换。猪域默认参数为 scale 0.02、offset 0。对非零 offset 的模型分别遵从两个实现的正负号，跨模块以物理图交接。注册训练所需的 `B,1,Z,Y,X` 从相位栈中按标签提取。[数据字段和网格](rstar4d/data/contracts.py) · [工程推理实现](rstar4d/inference/sliding_window.py)

PIGAE 读取采样几何时，以 [仿真产物](rstar4d/simulation/pipeline.py) 和 [工程 TIGRE 算子](rstar4d/simulation/backend.py) 为准。使用工程 FDK 的每个相位至少需要两个有效视角。输入源体网格和裁切后的输出网格可能不同，正投影须为先验实际所在的物理网格构造几何，避免重复加入源体偏移。若输出 FOV 截断显著结构，优先从混合投影生成覆盖这些结构的校准前先验，再将 $q_t$ 与 $x_t$ 按相同流程重建到目标网格；仅有截断先验的结果单列，并用静态同源体模量化误差。跨组统一采用基线的补边、裁剪和 FOV 标记。

基线命令示例（已有配对清单和工程 Stage II 权重时）：

```powershell
python -m rstar4d infer --checkpoint runs/pig_001/stage2/checkpoints/best.pt --manifest data/pig_prepared/manifest.jsonl --split-name test --out runs/moco/b_test
python -m rstar4d evaluate --predictions runs/moco/b_test/manifest.jsonl --manifest data/pig_prepared/manifest.jsonl --out runs/moco/eval_b
```

## 6. 实验矩阵与验收

所有组固定动物划分、RSTAR 权重、基线先验、使用视角、场的方向与配对范围。在各方法共同有效的目标区域计分，并报告每组覆盖率。几何或有效域差异不得通过缩小评价区域掩盖。

| 组 | 用什么估场 | 变形什么 | 实际问题 |
|---|---|---|---|
| A | 无 | 无，直接评价 $b$ | 原 RSTAR 达到什么水平？ |
| B | 零场 | $c$ | 同样的多相位混合在无配准时有何效果？ |
| C | 用 $x$ 训练及推理的独立 RegisNet | $c$ | 与 D 使用同样初始化、损失、配对及预算，测增强图对估场的贡献 |
| D | 用 $b$ 训练及推理的 RegisNet | $c$ | 全相位纯补偿 $d$；D 对 B 检验运动，D 对 C 检验增强图 |
| E1 / E2 | 逐对复用 D 的同一 DVF | $x$ / $b$ | 与 D 比较原始、轻度校正、深度恢复三种细节源 |
| F | 复用 D | $b$ 加入 $d-b$ 的局部高频 | 最终 $y$；与 A、D 比较保留 RSTAR 基础图是否有收益 |

在 B–E 上先比较纯补偿图；需要比较最终融合时，对每组应用同一带宽和锁定的系数，避免为每组另行挑选融合方式。附加配对范围对照只改全部源为有序邻相位。DPI-MoCo 表 V 中 3.37 dB 的增量来自其 RestoreNet 基线及模拟测试病例，不能作为 RSTAR 验收阈值。

| 验收对象 | 可执行检查 |
|---|---|
| 原版回归 | 相同输入、checkpoint、dtype、设备与推理方式，$b$ 与原 Stage II 输出对齐；CPU float32 小例建议 `atol=1e-6, rtol=1e-5`。关闭补偿或取 $\alpha=0$ 时，域内 $y=b$。比较按相位进行。 |
| 算子与方向 | 静态同算子样例令 $x=q$，校正后应得到 $PI_D$。已知 $(dx,dy,dz)$ 的非对称三维平移用标记点验证方向、轴序、spacing，另测零场、奇数尺寸和采样越界。 |
| 融合与时序 | 所有相位同值时，归一化 $d$ 等于该值；印刷式的和在完整支撑处为 $T$ 倍。一个邻居超界应少计一个贡献，自身只出现一次。固定解剖、仅改变血池强度的体模，记录虚假边界位移和衰减曲线误差。 |
| 图像与运动 | 同网格逐相位参考比较全图/心脏 ROI 的 MAE、PSNR、SSIM，所有组固定强度范围。用可用可信轮廓或独立勾画测左室边界 HD95（mm）和边界带 MAE；同相位参考血池 ROI 计算平均衰减曲线、峰值偏差和曲线 MAE。按动物汇总配对差值，列出有效动物、相位及缺标数量。独立解剖对应点存在时另报 TRE。 |
| 推理规模 | 10/11 相位核对 90/110 个非自身场，每个目标保留自身贡献；记录每扫描实际耗时、峰值显存、折叠率及源越界比例。 |

**主判断。** 在验证集按相对 A 的血池曲线 MAE 不增大这一条件，从候选 $\alpha$ 中选择左室边界 HD95 最低者，并列时取较小系数；两项采用同一组有参考的验证动物并报告例数。若缺少足够的验证轮廓或曲线参考，保留 A、D 与预先列明的系数对照，标注选择证据不足。锁定后在留出动物报告 F 对 A、D 的配对结果。若只见全图 PSNR 上升，边界或造影曲线变差，记录各阶段收益和代价。只有 A、D、F 的留出比较才能回答额外补偿是否值得保留。
