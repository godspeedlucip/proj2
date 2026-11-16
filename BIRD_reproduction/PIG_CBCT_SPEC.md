# Pig 数据与 TIGRE CBCT 接入规格

2026-11-16；此规格补充 SPEC.md。用户明确要求 TIGRE 投影，确认扫描时长为 **10 s**。

## 1. 无先验图像的实验合同

BIRD 的参考空间与图像 INR 从投影共同学习，不读取 `prior_image`、phase GT 或稀疏 FDK 作为训练输入。猪 CT 相位图仅用于模拟测量；`phase_*_gt` 与 valid mask 仅用于重建后评价。训练输入是动态混合投影、scanner 几何及采集时间。

## 2. 以用户生成代码为准

只读取 `D:/desktop/课题2/generate_round1_contrast_gt.py` 及其依赖；不修改原始数据/生成脚本。Round 1–6 使用各自 `data/roundN_contrast_recon_settings.csv`，不能把第一轮的空间分辨率复制到全部轮次。

- 数组轴遵循现有 `TigreTool` 的 `[z,y,x]`；不按 NIfTI affine 自动转置/重定向。
- HU→mu：`max(0.02*(1+HU/1000),0)`，单位 mm^-1；HU=-2048 的 source padding 为0。预投影不应用拟合多项式或 air threshold。
- 本次检查的常用设置：600视角、300°、DSO=750 mm、DSD=1250 mm；探测器 `[v,u]=[474,648]`、像素 `[.616,.616]` mm。
- 每个序列自己的体素间距乘 `voxel_spacing_multiplier`；默认1.5。投影源体保持完整源体 FOV。
- `offOrigin_zyx=[0,-(down-up)*dY,-(right-left)*dX]`。正负号沿用原脚本。
- `theta_i=i*300/600` 度，`time_i=i*10/600` 秒，i=0..599。视角和时间都不重复终点。网络 t=time/10，最后采样 t=599/600。
- `cycle.csv` 的非等长8个周期生成 view→phase 映射；使用原 `build_projection_phase_mapping`，最近相位取样。该模拟是分段恒定动态场，不能说成连续形变真值。
- 相位编号只决定生成哪个 CT 的投影，不作为模型时间。来自不同周期的同一相位保留不同采集时刻。

GT生成对每个静态相位单独使用全部600个角度，再 FDK；这与动态训练测量是两个数据流程。后者每个角度只来自映射到的一个相位，绝不拼入“每相位600角度”的 GT 测量。

## 3. 数据产品

`bird.pig_cbct` 提供只读 plan 与独立 generate 命令。保存：

- `plan.json`：源路径/体素信息、settings、cycle映射、10s时间、完整训练体范围、GT评价网格、源代码与CSV指纹。
- `projections.npy`：TIGRE Ax(Siddon) 生成的 float32 `[selected_views,v,u]`，memmap写入。
- `case.json`：带测量 SHA-256 的训练合同；只有全部写入成功、有限性检查通过才发布。
- `view_mapping.csv`：原始全局view index、角度、秒数、相位、cycle。允许显式选小量 views 进行接入冒烟测试，但合同标明不是完整600视角数据。

发现旧的第一轮 Control_1 缓存 `reconstruction_config.json` 使用 `polynomial_valid_source_only_v2`，而当前 GT 脚本为 `hu_to_mu_valid_source_only_v1` 或 post-FDK 校准。因此**不直接复用旧737MB投影**，新测量从原始相位生成并记录当前强度约定。

现存 `outputs/round1_contrast_gt/Control_1_7569/Heart_with_contrast_back/gt_config.json` 也不兼容：旧值 shift_up=18、shift_left=10、fov_source_pixels=128，而当前CSV三项均为0；旧强度是预投影多项式处理。新导出的“GT网格”指**当前生成代码/CSV合同的网格**，不代表已与磁盘旧GT对齐。`bird.audit_pig_gt` 只读核对并输出差异。要评价，应按同一设置生成新的 raw-mu GT，或明确构建并验证另一套匹配的数据合同。

仅将路径/元数据写入本项目 `runs/pig/`；默认不批量生成所有猪的投影，不自动启动64000步训练。

## 4. TIGRE 训练路径及与论文的差别

两路图像 INR 在采集时刻的完整三维网格上求值，分别调用 TIGRE `Ax(...,projection_type='Siddon')`；反传显式使用 `Atb(...,backprojection_type='matched')`。**不能使用默认 FDK 权重当作训练梯度**。TIGRE matched backprojector 是近似转置，必须报告数值 adjoint 测试误差，不声称严格精确梯度。

标准 TIGRE 接口接收显式三维数组，此模式是 view-based voxel-grid adaptation，改变了原论文直接按射线查询的效率与离散化。BIRD 的双特征网络和三个正则仍保留。旧 ray renderer 作为独立参考入口保留；pig 入口只用 TIGRE，不静默回退。

为避免整幅 INR 激活常驻显存，采用两遍计算：

1. no_grad 分块查询两路体数据，令物化体作为可微叶子，经 TIGRE 与投影损失求体梯度；
2. 同一模型参数下按块重新查询 INR，将体梯度乘回输出，累积网络 VJP；
3. 累积三个正则项，一次 optimizer.step。

梯度应与不分块的原生 autograd 查询一致。GT 不参与这两遍计算。每次可选少量 views、从其 detector pixels 抽样，完整抽样 batch 统一排序再 FFT。不能每个视角分别优化一次后才合并损失。

训练必须覆盖**完整投影源体**，不能直接以中心256×256 GT裁剪体解释穿过躯干的完整投影。可显式降低网格分辨率，但保持源体总物理尺寸和 offOrigin 不变，并标注为 coarse pilot；正式试验再提高分辨率。网络空间编码最高级和有限差分步长随所选网格更新。

默认物化网格从配置 `spatial_resolution=[Nx,Ny,Nz]` 取得；`--volume-shape Z Y X` 显式覆盖。`temporal_resolution=n_views+1=601` 使差分步长为1/600，即10/600秒。正则点数需显式给定，不能照搬原射线采样点数。`samples_per_ray` 与 `ray_chunk` 在此入口不参与TIGRE投影；由 `--voxel-chunk` 控制INR分块。

载入case时校验 view ID、固定角度/时间公式、完整扫描覆盖标志、体素shape×spacing=size、bounds与offOrigin/size一致，以及投影SHA256/维度/有限性。断点绑定case和测量指纹、配置、物化网格及抽样选项；恢复模型、Adam、scheduler和随机状态。

## 5. 评价网格

GT 是完整相位的全视角 FDK 结果，严格说是评价参考而非无伪影真实衰减场。评价先在与 FDK 相同的居中 FOV 网格上查询网络，再按原函数执行圆柱 mask 与 transverse crop/resize；有效mask只在评价阶段使用。原生猪源体与最终GT不能仅凭数组尺寸直接比较。

若 GT 使用 post-FDK 多项式校准，必须显式对齐评价强度域并记录系数；不能改变训练的物理 post-log 投影以迁就校准后的 GT。默认导出 raw mu，未指定校准时不自动套用旧 compare.csv。

## 6. 验收

- 600view→phase全覆盖、时间单调；原始第599view=9.983333s/299.5°；重复phase不得重复时间。
- 依用户代码核验 shape/spacing/offset、HU padding与转换、GT网格逻辑。
- memmap投影可按view读取，sha256/shape/有限性错误拒绝；生成中断不产生可训练 case。
- `Ax/Atb matched` 在真实 GPU 上运行，测相对内积误差；只声称近似伴随。
- 分块物化/VJP 与完整查询的参数梯度在数值容差内一致。
- 以真实猪相位生成少量视角，完成低分辨率训练步和物理网格推理，所有数值有限；此测试不代表完成猪数据重建质量验证。
- 记录已有 GPU 环境，兼容其 Python3.9、PyTorch2.3.1+cu118、TIGRE3.0.0；不安装或修改该环境。
