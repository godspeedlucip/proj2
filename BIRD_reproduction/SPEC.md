# BIRD 网络复现规格

版本：1.0，2026-11-16。基准：Du et al., *Physics in Medicine & Biology* 71 (2026) 105003，DOI 10.1088/1361-6560/ae639c。公式号均来自本地期刊 PDF。

补充：猪心数据使用用户指定的 TIGRE 投影与10秒扫描；具体覆盖项、数据合同及 voxel-grid adaptation 见 [PIG_CBCT_SPEC.md](PIG_CBCT_SPEC.md)。下文射线中点积分对应原始参考入口 `bird.train`；猪心 `bird.train_tigre` 明确使用 TIGRE Ax/matched Atb。

## 1. 交付与证据边界

本项目交付可运行的独立 PyTorch 参考实现：完整网络、可微射线积分、四项损失、逐病例训练、checkpoint 恢复、任意时刻体数据推理、合成数据及测试。不是作者官方代码，也未完成论文数据集上的定量复现。

以期刊版为唯一实现基准。2025 年 arXiv v1 的解析运动补偿重建模块不属于此版网络输出流程，不能把该模块与期刊版公式混拼。本地旧摘要只作线索，原文优先。未获得独立补充材料、作者训练脚本、原始扫描数据或权重。

检索截至 2026-11-16 未确认官方 BIRD GitHub 仓库；不将同名项目或 tiny-cuda-nn 当作 BIRD 源码。若未来发现官方仓库，应记录 URL、commit、许可证并另存到 `vendor/`，重新审计差异。

## 2. 技术选择及可替换假设

选择可在 CPU/CUDA 上运行的纯 PyTorch 多分辨率 hashgrid，实现完整算法路径。此选择便于阅读、单元测试和 Windows 运行，但不会复现 tiny-cuda-nn 的 CUDA 性能、哈希索引或逐位结果。没有用位置编码或普通 MLP 偷换 hashgrid。

| 项目 | 原文事实 | 本实现的决定 |
|---|---|---|
| 编码 | 16 levels，2 features/level，base=16，hashmap=2^19 | 论文配置保留；小型配置仅用于验证 |
| 最高分辨率 | 约为目标重建分辨率两倍 | 空间使用 2×max(Nx,Ny,Nz)，时间使用 2×Nt；后者及各轴选择属假设 |
| 4D 编码 | 3D/1D 编码、细化、独立及乘积三支路、MLP 融合 | 各 latent 输出 32；各小 MLP 一隐藏层，宽度 32 |
| 预测块 | 两个单隐藏层 MLP，输入到第二个的拼接 shortcut，式23 | 第一个输出 32，第二个输出 1 或 3 |
| 激活和初始化 | 未完整报告 | hidden ReLU、latent identity、衰减系数 softplus、DVF identity；DVF 小权重初始化。均在配置/代码中显式说明 |
| 数值微分 | 步长等于目标重建分辨率 | 中心差分；空间 h=1/N，时间 h=1/(Nt−1)；只在内部采样，避免时间越界 |
| 射线积分 | 基于射线采样 | 盒相交后的等距中点求积；采样数可配，非作者公开的精确投影器 |
| 优化 | Adam，cosine，hash LR=.03，其他=.003，64000 steps，2048 rays | betas=(.9,.999)、eps=1e−8、eta_min=0、无 weight decay；这些缺失项是明确假设 |
| 损失平均 | 公式给出范数，批次约减未充分说明 | 对每点求 Frobenius/向量平方和，然后对点平均；registration 标量绝对差平均 |

采用完整可读的纯 PyTorch 参考实现，最初在系统CPU环境验证；新增TIGRE入口随后在本机独立CUDA环境验证。临床规模编码加速仍需后续工作，移植后须重跑验收测试。

## 3. 坐标、单位和输入合同

- 世界坐标顺序为 `(x,y,z)`，单位 mm；投影为 `p=-log(I/I0)`，无量纲。不能直接输入原始 photon counts 或 HU 图像。
- `bounds_mm` 为 `[2,3]`，第一行为下界，第二行为上界。归一化坐标 `x=(world−lower)/(upper−lower)`。网络 DVF 单位为该归一化坐标；导出时乘每轴 extent 得 mm。
- `time_bounds_s=[t0,t1]`，要求 `t1>t0`；网络时间 `(time_s−t0)/(t1−t0)`，范围 [0,1]，禁止按周期取模。
- 积分仍使用世界坐标路径长度 mm，不使用归一化步长代替，防止衰减单位错配。盒外衰减为 0；盒必须覆盖所有待解释的衰减物体，截断躯干且不处理外部积分会产生偏差。
- 所有输入必须有限；盒轴长度、射线方向模长、数据数量须为正。所有时间必须在时间范围内。负的有限 post-log 值可保留，不静默裁剪。
- 推理数组顺序 `[time,z,y,x]`；采样体素中心 `(i+0.5)/N`。空间分辨率配置顺序 `[Nx,Ny,Nz]`。
- 形变后的参考坐标可超出 [0,1]，hashgrid 按扩展网格求值，不 clamp，以保留坐标梯度；这是需要结合实际 FOV 检查的实现选择。

支持两种 `.npz`，`allow_pickle=False`：

1. 通用射线：`origins_mm[N,3]`、`directions[N,3]`、`times_s[N]`、`projections[N]`、`ray_ids[N]`（唯一非负整数，按 view→detector-row→detector-column 的全局排序键）、`bounds_mm[2,3]`、`time_bounds_s[2]`。方向自动单位化。
2. 平面探测器几何：`projections[V,H,W]`、`sources_mm[V,3]`、`detector_centers_mm[V,3]`、`detector_u_mm[V,3]`、`detector_v_mm[V,3]`、`times_s[V]`，以及上述 bounds。u/v 向量是一像素的物理位移；像素中心为 `center+(col−(W−1)/2)u+(row−(H−1)/2)v`。只为抽样索引构造射线，不展开全部几何。逐视角 source 和 detector 可以来自轴扫或螺旋轨迹；弧形探测器须由外部预处理为精确通用射线，不能假装平面探测器。

所有数组载入主存，NPZ 不是超大原始数据的流式存储后端。主存不足时需另外实现 memmap/HDF5 适配，不能靠 GPU chunk 设置解决主存问题。

## 4. 网络和张量合同

输入 `x[N,3]`、`t[N,1]`；允许 `t` 为标量广播。

| 路径 | 运算 | 输出 |
|---|---|---|
| backward DVF | `e4D_b(x,t)` → shortcut predictor | `Db[N,3]` |
| 参考坐标 | `xs=x+Db`，式8 | `[N,3]` |
| TP feature | `hash3D_tp(xs)`，式10 | `[N,32]`（论文配置） |
| TP head | `pred_tp(Ftp)`，式11 | `mu_tp[N,1]` |
| FF feature | `e4D_ff(x,t)`，式13 | `[N,32]` |
| final head | `pred_final(cat(Ftp,Fff))`，式14 | `mu[N,1]` |
| forward DVF | 独立的 `e4D_f(xs,t)` → predictor，式17 | `Df[N,3]` |

三个 4D encoder 不共享参数。3D/1D hash 输出先分别经 MLP 得 hx/ht，再分别计算 `MLP_x(hx)`、`MLP_t(ht)`、`MLP_xt(hx*ht)`；拼接后经融合 MLP。不是简单 concat 或乘法替换整个融合结构。

最终图像是融合特征的非线性预测，不是 `mu_tp + mu_ff`，也不要求自由支路输出显式 residual。静态参考空间是潜在的，未强制等于 t=0。

## 5. 损失及梯度所有权

### 投影双监督，式15–16

两路均沿同一组射线积分。按 ray_ids 的升序排列实测与预测，计算完整一维 DFT：`rho=mean(abs(fft(pred_sorted)-fft(measured_sorted)))`。FFT 使用 `norm="backward"`，复数误差取模；不是仅比较频谱幅度，也不是 `rfft` 半谱，不能对随机乱序的 batch 直接 FFT。

`Lfdl=lambda_tp*rho(tp,p)+(1-lambda_tp)*rho(final,p)`。

### 空间正则，式19–20

`E(xs,t)=Df(xs,t)+Db(xs+Df(xs,t),t)`；对静态 xs 求数值 Jacobian，损失为 `mean(sum(J(E)^2, dims=(output_xyz,input_xyz)))`。这一项更新两方向 DVF，不能改成普通 DVF 平滑或仅 `||E||²`。论文使用一个 static→dynamic→static cycle，未另加对称第二项。

### 时间正则，式21

`a=(Df(xs,t+h)-2*Df(xs,t)+Df(xs,t-h))/h²`，`Lts=mean(sum(a²,xyz))`。只更新 forward DVF。h 的单位是归一化时间，权重也对应此约定。

### 配准，式22

`xs=stopgrad(xd+Db(xd,t))`，`xhat=xs+Df(xs,t)`；`Lrgt=mean(abs(mu_tp_frozen(xhat,t)-stopgrad(mu_tp(xd,t))))`。

freeze 意味着图像模块及 backward DVF 参数不求梯度，但 `mu_tp_frozen(xhat,t)` 对输入 xhat 的导数必须保留。不能将整个重采样放在 `no_grad()` 内，否则 forward DVF 收不到正确梯度。该项只更新 forward DVF；图像目标随当前模型每次迭代重新计算。

总损失：`L=Lfdl+10*Ldm+0.1*Lts+10*Lrgt`。一般 lambda_tp=.5，XCAT .75，含自由变化的临床模拟 .1。

三项正则各自独立均匀抽样；默认每项点数为本批射线采样点数的 1/4。可显式覆盖以做资源实验，但必须记录偏离。小型配置显式覆盖为 32。

## 6. 训练、内存与恢复

- Adam 将 hashgrid table 参数和其余参数分组，学习率分别 .03/.003；不使用真实动态体或 DVF 标签。
- 射线分块计算图像查询，checkpoint recomputation 用于限制保留激活；完整 batch 的所有投影拼接后才计算全局排序 FFT，不把 FFT 按 chunk 独立计算。
- 正则分块后按点数加权求和并及时反传；累积的是同一步参数下的梯度，再执行一次 optimizer step。注册梯度所有权保持不变。
- 每步记录各分项、总损失、学习率及步数；拒绝非有限损失或梯度。
- 保存模型、optimizer、scheduler、resolved config、数据 SHA-256、torch CPU/CUDA RNG、射线抽样 generator 状态及步数。
- resume 必须检查同一配置与数据 SHA-256，不允许把其他扫描当作同一优化轨迹恢复。结束步数由配置确定；`--stop-after` 仅为可恢复中断，不改变原有 cosine horizon。
- 输出存在 checkpoint 时，新训练拒绝覆盖，用户需指定新输出目录或明确使用 resume。

## 7. 验收标准

1. 常数衰减射线积分等于 `mu×相交长度(mm)`；平行/不相交/盒内起点不产生 NaN。
2. 1D/3D hashgrid 对特征参数和坐标可微；低分辨率网格插值有手算测试。
3. FFT loss 对连同 key 一起打乱的 batch 不变，已知小向量的数值正确；复数相位误差不能丢失。
4. 仿射互逆映射使空间正则近零；非互逆缩放有确定非零值。常数 cycle 偏移可使该正则为零，文档须说明它不保证严格逆映射。
5. 时间二次 DVF 的二阶差分结果正确，线性轨迹为零。
6. 单独 `Lrgt.backward()` 只为 forward DVF 产生梯度，且非平坦图像时梯度非零。
7. 两条图像路径和 backward DVF 均受 Lfdl 监督；全损失可反传。
8. 同一模型分块/不分块投影与梯度一致（容差内）。
9. 合成 3D 动态 phantom 可生成、训练、保存和推理；loss 下降只证明工程流程可优化，不证明解剖正确性或论文指标。
10. 不间断训练与中断恢复在同一 CPU/配置/种子下得到一致状态；坏形状、非有限数据、时间越界须早报错。
11. 推理支持指定扫描时间内任意时刻，输出 `[T,Z,Y,X]` 及配置、物理边界；时间越界拒绝。

## 8. 实验复现另需条件

要声称复现论文表1/2，需原始数据/生成器、scanner 几何与预处理、确切坐标/强度尺度、射线采样方案、作者缺失的网络细节及指标动态范围。需用同一数据和基线预算重跑，报告多种随机种子及误差。本文图6的可视化与软逆一致性不能代替 Jacobian determinant、folding fraction 或真实 DVF 验证。论文真实患者仅一个回顾性数据集，未验证低剂量条件。
