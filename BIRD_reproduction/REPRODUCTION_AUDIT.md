# BIRD 论文一致性与完整运行核验

核验日期：2026-11-16。基准是本地2026年期刊版 *Nonperiodic dynamic CT reconstruction using backward-warping implicit neural representation with diffeomorphism regularization*，PMB71 105003，DOI10.1088/1361-6560/ae639c。直接核对正文Eq.8–27、§2.3–2.4和图1–3；没有用旧版arXiv或本项目spec替代原文。用户两套GT配置是有意设置，本次不将差异作为问题，不更改两套数据。

## 结论的含义

核心网络和损失可以称为**依据期刊公开公式完成的独立参考实现**。通过运行检查说明训练、反传、保存、恢复和推理链路可执行；不能将这一结论写成“已经得到论文表1/2的结果”或“与作者源码数值等价”。没有确认的作者源码可作逐项比对，论文也未报告所有实现细节。

## 逐项核对

| 原文 | 应实现的内容 | 代码与核验 |
|---|---|---|
| Eq.8–12 | 后向DVF将动态坐标映射到静态空间；静态坐标查询TP特征及图像 | `model.py::BIRD.forward/topology`；路径与图1一致 |
| Eq.13–14 | FF编码动态坐标；TP/FF特征融合生成最终图像 | `ff_encoder`和`final_head`；不是两幅图像简单相加 |
| Eq.15–16 | 两路投影监督；按view/pixel排序的1D复数DFT距离 | `losses.py::data_fidelity/frequency_distance`；完整batch统一FFT；独立稠密DFT定义验证值和梯度 |
| Eq.17 | 独立前向DVF | 两套DVF不共享参数，前向场仅辅助约束 |
| Eq.19–20 | 前后向cycle误差的空间Jacobian平方Frobenius范数 | 中心差分覆盖九个元素；非对角仿射解析值及A/B梯度验证通过 |
| Eq.21 | 前向DVF时间二阶差分 | 差分除以dt²后平方；只约束前向场；二次时间函数解析值通过 |
| Eq.22 | 图像配准项仅训练前向DVF | 目标及初始后向坐标停止梯度；预测侧冻结参数并保留坐标导数；已有梯度不会被改写 |
| Eq.23 | 两个单隐藏层MLP，第二个接收输入shortcut | `ShortcutPredictor`按concat实现 |
| Eq.24–26 | 空间/时间分别编码及细化，空间/时间/乘积三支路融合 | `SpaceTimeEncoder`；3个独立4D编码器 |
| Eq.18、27 | 四项损失组合；各正则独立采样、累积后一次更新 | 两个训练入口均成立；分块按点数加权，不对每块独立FFT |
| §2.4 | 16级、每级2特征、base16、hash容量上限2^19、隐藏宽32 | 实例化检查7个hashgrid、26个单隐藏层MLP，全部隐藏宽32 |
| §2.4 | Adam、cosine、hash学习率.03、其他.003、2048射线、64000步 | `paper.json`原配置保持；所有参数恰好进入一个optimizer组 |
| §2.4 | 正则点数为射线采样点数1/4 | 原始ray入口默认2048×512/4=262144点/正则；TIGRE入口改为显式数量，属于适配差异 |

独立只读reviewer复核以上映射，未发现已证实的核心数学错误或阻断运行的代码缺陷。新增的三项独立oracle测试存于 `tests/test_paper_oracles.py`；本轮完整测试为 **33 passed**。

## 与作者实现的边界

1. **hashgrid后端不同**：使用纯PyTorch，论文使用tiny-cuda-nn。结构相同不代表哈希索引、边界行为、逐级分辨率取整、浮点结果和性能相同。
2. **未报告细节采用显式补全**：ReLU/softplus、latent32、初始化、512点中点积分、坐标归一化、中心差分及内部采样、Adam缺省值等。它们不能被标为作者确认设置。
3. **TIGRE是按用户要求的适配**：Ax(Siddon)需要显式体网格，当前入口先抽view再抽pixel；论文在全扫描随机抽ray并直接查询INR。两者投影离散化、batch联合分布及计算复杂度不同，matched Atb也只是近似伴随。
4. **试验设置不同**：猪心600view/10s/300°与论文四组实验不是同一数据；TIGRE pilot使用粗网格、显式2048正则点。本次20步配置仅用于完成一次短验证周期。
5. **结果尚未复现**：未完成64000步收敛或论文原始数据/基线/PSNR/SSIM对照，不据训练loss下降推断图像质量或严格微分同胚。

初始化补充：TP初始图像极平坦时，float32配准差可能量化为0。这不意味着计算图断开。真实BIRD采用非平坦内容时，float32配准损失和前向梯度均非零；双精度及已有梯度情况下的梯度归属测试也通过。

## 本次命令与证据位置

GPU环境：`D:/python/miniconda/env/env/common/python.exe`，Python3.9.23、PyTorch2.3.1+cu118、pytigre3.0.0，RTX4070 Ti SUPER。系统Python为CPU测试环境。所有命令从本项目目录运行，设置 `PYTHONDONTWRITEBYTECODE=1`，输出使用新的目录名避免覆盖。

```powershell
$pigPython = 'D:\python\miniconda\env\env\common\python.exe'
& $pigPython -m bird.train --data runs/demo_data.npz --config configs/paper.json --out runs/my_verify/paper --device cuda:0 --threads 2 --stop-after 1
& $pigPython -m bird.train --data runs/demo_data.npz --config configs/paper.json --out runs/my_verify/paper --device cuda:0 --threads 2 --resume --stop-after 2
& $pigPython -m bird.infer --checkpoint runs/my_verify/paper/checkpoint.pt --out runs/my_verify/paper/volumes.npz --times 0 0.5 1 --resolution 24 24 24 --chunk 4096 --device cuda:0 --threads 2

& $pigPython -m bird.train_tigre --case runs/pig/round1_control1_10s/case.json --config configs/verify_tigre.json --out runs/my_verify/tigre --views-per-step 2 --voxel-chunk 8192 --device cuda:0 --stop-after 10
& $pigPython -m bird.train_tigre --case runs/pig/round1_control1_10s/case.json --config configs/verify_tigre.json --out runs/my_verify/tigre --views-per-step 2 --voxel-chunk 8192 --device cuda:0 --resume
```

只从更新压缩包开始时，大型数据未打包：`runs/demo_data.npz` 可用 `bird.demo` 生成，猪心case需按PIG_RUN从原CT生成。完整本机工作目录已有这些数据。

原始paper配置参数量44,335,916，TIGRE 64×96×96配置参数量32,882,166；两者均为16级/宽32。参数量不同来自目标网格与hash表分配，不能把TIGRE粗网格测试称为44M参数测试。原始paper配置已独立验证。

结构/梯度/源文件指纹：`runs/verification_20261116/structure_and_gradients.json`。逐步日志、checkpoint、推理及运行摘要分别位于同目录 `paper_exact/`、`tigre_20/`。最终数值见本报告的实际运行结果及对应JSON。

## 实际运行结果

| 检查 | 结果 | 证据 |
|---|---|---|
| 未修改paper.json完整优化步 | 44,335,916参数；2048×512射线采样；每项正则262144点；有限前向/反传/Adam更新及checkpoint | `paper_exact/one_step_profile.json` |
| paper恢复及推理 | 从1步恢复到2步；216个参数张量均有Adam状态；0、.5、1秒的mu/TP/DVF全部有限 | `paper_exact/pipeline_check.json` |
| 16级TIGRE完整短周期 | 32,882,166参数、64×96×96物化网格、2048射线和每项2048正则点；20/20步，10步处中断恢复；cosine学习率到0 | `tigre_20/pipeline_check.json` |
| TIGRE运行稳定性 | 20步所有损失有限，参数全部有限；六个主要分支均发生参数更新 | 同上及`history.jsonl` |
| TIGRE推理 | 0、5、9.983333333秒；mu/TP=[3,64,96,96]，DVF=[3,64,96,96,3]，全部有限 | `tigre_20/volumes.npz`及pipeline检查 |
| 实际猪心几何的伴随检查 | 64×96×96物化网格、两角度，正值随机测试内积相对误差0.000437206，约0.0437% | `tigre_20/pipeline_check.json`；不是任意信号/几何的误差界 |

在**完全相同的固定投影输入**上进行训练前后检查：

| 路径 | 固定评价输入 | 初始RMSE | 当前RMSE |
|---|---|---:|---:|
| 原始ray入口、paper.json | 合成扫描全部6144条射线 | 1.53556089 | 1.49522917（2步后） |
| TIGRE入口、16级网络 | 猪扫描view0/200/400、全部921456像素 | 5.32057992 | 4.26116034（20步后） |

这支持“网络确实更新并改善已测量投影拟合”，不是保留集或GT重建精度。没有比较两套GT，也没有将这两个RMSE跨数据相互比较。TIGRE的训练、检查只读取投影和几何，不读取GT图像。

## 时间与可实际使用的范围

原始paper配置一完整优化步（含启动/保存）为68.19秒，PyTorch峰值分配910837760字节，约0.848GiB。第二步日志约65秒。此处不是只查询4个点或缩小正则点数的网络前向测试。

新增 `configs/paper_cuda_chunks.json` **只改变执行分块**：ray_chunk由16增至64，reg_chunk由1024增至8192；网络、总batch、射线采样点数、正则点数、学习率、64000步周期等均不变。相同机器/数据的单步实测13.26秒，PyTorch峰值分配941484032字节，约0.877GiB；完整反传和保存通过。首步FDL均为21.67174339，分块数值容差已有独立测试覆盖。这是单步测量，不承诺固定加速比，亦不统计TIGRE/驱动的全部显存。

```powershell
& $pigPython -m bird.train --data runs/demo_data.npz --config configs/paper_cuda_chunks.json --out runs/my_verify/paper_chunks --device cuda:0 --threads 2 --stop-after 1
```

TIGRE 20步累计训练时间约308.22秒，平均15.41秒/步，范围13.09–20.63秒/步，不含另行推理与固定投影评价。原始源分辨率560×512×512没有被这次64×96×96测试覆盖。

**当前版本可执行，但仍是较慢的参考实现。** 按13.26秒/步粗略外推64000步约9.8天；这不是完整训练实测，也不是TIGRE原始源网格的预测。论文采用tiny-cuda-nn并报告约6–8小时，不能把该速度套用到当前纯PyTorch实现。若实际实验要求一天内跑完，需要进一步加速编码/微分与采样，且重新验证数值一致性。

## 最终判定

- 核心网络/损失的论文对应关系：本轮核验通过，独立审查无已确认核心bug。
- 完整执行链路：本轮实际跑通，包括原始44M配置及16级TIGRE配置、恢复、优化器/调度器状态和多时刻推理。
- 64000步完整训练完成、最终图像质量、作者源码等价、论文表格与运行时间复现：**尚未证明**，不得由短验证推断。

本轮没有修改生产算法源码；新增3个独立数学回归测试、2个验证/分块配置和记录。包内源码指纹与实际运行代码对应。更新代码与小型记录包为 `BIRD_reproduction_verified.zip`，大型投影、checkpoint和体数据保留本机目录。
