# RSTAR4D 论文一致性与可运行性验证

日期：2026-11-16。验证对象：本地论文 `RSTAR4D.pdf`、官方Notebook与权重、当前TIGRE猪数据Spec。用户有意用不同配置生成的历史GT/CBCT不在本次审查范围，也不计为运行阻塞。

## 1. 判定

| 要判断的内容 | 本次结论 |
|---|---|
| 官方模型是否实现论文的核心方法 | 可分离4D卷积、双通道prior输入、circular时间填充、Tetris两阶段均存在，模型与发布权重兼容；公开参数量与论文表格仍有差异 |
| 核心部件能否运行 | 可以；已用真实TIGRE算子和合成动态体模验证仿真、数据读取、两阶段各2步训练、权重保存重载、推理 |
| 原仓库能否不作适配直接完整运行 | 不可以；缺依赖和示例数据，保存代码被注释，Stage II结尾引用未定义writer |
| Spec要求的完整训练系统是否已经交付 | 没有；尚无生产用rstar模块、CLI、在线采块、完整checkpoint/恢复、滑窗和指标管线 |
| 是否复现论文实验数值 | 没有；未获得原训练/验证病例、完整划分与预处理，未执行完整训练和相同测试评价 |

**适当表述：论文核心网络有可运行的公开实现，TIGRE与两阶段训练的最小验证链路通过；完整猪数据训练工程仍待实现。** 本次验证脚本是审查工具，不能将其称为交付版训练工程。

## 2. 与论文逐项对照

论文依据为本地PDF：§2.2.1（第8–10页）、§2.2.2（第10–11页）、§2.3.1–2.3.5（第11–14页）、Table 4（第17页）。与最终发表版本如有差异，需要另作版本对照；本报告未将两个版本自动视为相同。

| 项目 | 论文 | 官方代码 / 当前Spec | 判断 |
|---|---|---|---|
| 主体结构 | U-Net，xy二维、z一维、t一维卷积 | 两Notebook均提供，原模型类未经改写用于本次测试 | 核心结构一致 |
| 先验输入 | average image与degraded image拼接 | 网络首层2通道；prior广播到各相位后拼接 | 一致 |
| 时间边界 | circular padding | 原Notebook为circular；Spec v2.0曾默认提议reflect | v2.1已改回circular主线，reflect仅作消融 |
| Stage I | 单层2D+T，跳过并冻结z卷积 | z分支在forward中不调用；测试z参数无梯度、不改变 | 一致；原代码未显式设置requires_grad=False，但跳过分支达到本次测试的不更新效果 |
| Stage II | 多尺寸4D块，启用z卷积 | 原Notebook恢复z分支，提供Dirac/零初始化；原数据加载器只读已经准备好的块 | 网络/切换具备；在线块生成与采样尚未交付 |
| 训练块 | ZYX为16×256×256、32×144×144、64×128×128，全T=10 | 公开Notebook说明列出相同尺寸；猪Spec使用更小块，并支持T=11 | 猪默认是显存适配，不是相同实验配置 |
| loss | pixelwise L1 | Notebook实际为L1+0.1×(1−SSIM)；猪Spec默认L1 | 公开训练设置与论文不同 |
| lr、epoch | Adam，1e-4逐渐到1e-5，30epochs；未明确两阶段如何分配 | Notebook为1.5e-4，Stage I实际60epochs、Stage II30epochs；全局EPOCHS=50并未控制循环 | 存在差异；Spec30+30和cosine仅为明确标注的工程默认 |
| 参数量 | Table 4：2.35M | 实测2,118,693参数，232个state_dict条目 | 有待解释的版本/统计差异，不能声称完全相同 |
| 投影与GT | 680views/360°、论文半扇束设置；原4D CT作为测试GT | 用户要求TIGRE、猪配置600views/300°、逐相位全视角FDK作为GT | 用户指定的数据适配；不能用相同表格数值判定 |
| prior具体算法 | 文中说明average image输入 | 官方MATLAB按各相位views比例加权相位FDK；用户代码对全部混合views做一次FDK | 输入信息来源一致，但未验证不同后端的数值等价 |
| 推理 | 24GB RTX3090，完整256×256×90×10一次推理 | 本机16GB；Spec拟用保留全T的空间滑窗 | 推理方式适配，滑窗尚未实现/验证 |
| 指标 | 测试病例SSIM/RMSE及肺/肿瘤ROI | Notebook只有训练/验证损失打印；无论文指标导出和ROI评估实现 | 尚未复现实验评价 |

原脚本自己也注明仿真代码可能与论文版本略有不同。因此，“作者发布的实现”“论文描述的设置”和“适配猪数据的新工程”必须分别陈述。

## 3. 原样运行的实际阻塞

1. 两Notebook的导入单元均实际抛出 `ModuleNotFoundError: No module named 'prefetch_generator'`。本次使用现有CUDA环境，未安装或修改依赖。
2. 四个硬编码的 `example_data/train_data_stage1`、`eval_data_stage1`、`train_data_stage2`、`eval_data_stage2` 目录均不存在。用户的原始NIfTI不能直接由这些HDF5数据类读取。
3. 两Notebook的 `torch.save` 均被注释；没有完整optimizer、scheduler、scaler及RNG checkpoint。Stage II硬编码载入发布的Stage I权重，不会自动接上用户刚训练出的Stage I。
4. Stage II末行 `writer.close()` 中的writer在Notebook中未定义；独立执行该语句实际产生NameError。
5. Spec中的 `python -m rstar.cli --help` 实际返回 `ModuleNotFoundError: No module named 'rstar'`。本次新增的validation目录只是审查脚本，不填补该生产实现。
6. Notebook缺在线混合尺寸采块、患者/动物级划分管理、续训、滑窗、原单位输出与论文指标统计。MATLAB脚本依赖外部输入，也没有形成到两个训练数据目录的完整导出管线。

代码证据：[Stage I保存处](D:/desktop/课题2/proj/RSTAR4D/RSTAR/githubcode/Training_Tetris1.ipynb:430)、[Stage II加载处](D:/desktop/课题2/proj/RSTAR4D/RSTAR/githubcode/Training_Tetris2.ipynb:427)、[Stage II末行](D:/desktop/课题2/proj/RSTAR4D/RSTAR/githubcode/Training_Tetris2.ipynb:555)、[prior加权累加](D:/desktop/课题2/proj/RSTAR4D/RSTAR/githubcode/data_simulation.m:90)。

## 4. 本次实际执行的验证

环境：`D:/python/miniconda/env/env/common/python.exe`，Python3.9、PyTorch2.3.1+cu118、RTX4070 Ti SUPER 16GB、现有TIGRE CUDA后端。

### 4.1 权重与前向

- 两阶段各自的发布权重均strict加载成功。
- Stage I分别运行T=10/11、Z=1、XY=32；Stage II分别运行T=10/11、Z=16、XY=32。
- 四项测试输出shape正确，所有值有限。
- 每模型2,118,693参数；state_dict共232条目。没有以T11成功推断其已在猪数据上获得有效恢复。

### 4.2 TIGRE至训练的合成数据链路

采用32³水样椭球和移动高衰减球组成11个状态，从HU调用用户工具中的HU→mu变换。DSO/DSD=750/1250mm，探测器64²、间距1mm，600views/300°，使用实际cycle.csv分配视角，逐状态计数为 `[28,62,60,60,61,58,61,60,60,62,28]`。

对每状态进行TIGRE正投影：全部600views用于该状态FDK参考；选中views拼入同一混合扫描；混合扫描再生成共享prior与各状态稀疏FDK。重建调用用户的 `reconstruct_prior`，使用小体模的padding=4、输出32²、fov_source_pixels=32。所有图像与投影有限，CBCT/target为 `[11,32,32,32]`，prior为 `[32,32,32]`。

此项使用合成体、无噪声、无多项式标定；未运行完整真实猪病例或用户GT脚本的全部有效掩膜/校准逻辑。输出保存在 `validation/artifacts/2026-11-16/tigre_phantom_mu.h5`。

为调用未经改写的Notebook数据类，测试用HDF5将值写为 `mu/0.02*1000`，再由其原有 `/1000` 得到 `mu/0.02`。这是验证脚本显式做的强度适配，不是证明原Notebook已经支持猪数据。

### 4.3 原训练函数、阶段切换、保存重载

提取原Notebook的模型类、数据类、train1/eval1和Stage II初始化单元原文执行。因导入依赖缺失，外围使用标准DataLoader，未执行原Notebook所有单元。

| 项目 | Stage I | Stage II |
|---|---:|---:|
| 输入B,C,T,Z,Y,X | 1,2,11,1,32,32 | 1,2,11,16,32,32 |
| 训练步数 | 2 | 2 |
| 发生更新的参数张量数 | 152 | 232 |
| 具有梯度的z参数张量数 | 0 | 80 |
| 保存重载后输出最大绝对差 | 0 | 0 |

Stage I从发布Stage I权重继续训练，Stage II继承此次更新后的Stage I，执行原Dirac/零初始化，且断言所有非z权重在初始化时保持不变。模型及梯度有限，两个阶段均确认发生参数更新。

保存/重载由审查脚本补上，只保存state_dict；未测试生产用完整断点恢复。训练和eval使用同一合成病例，只有两步，结果不用于证明泛化能力、从头收敛或论文性能。JSON中的RMSE仅为诊断数值。

### 4.4 三种Spec训练块的显存

单独进程中，原Stage II/circular架构、发布Stage II权重、batch=1、T=11、fp16+GradScaler、L1+Adam，每种块执行两次forward/backward/optimizer step。断言梯度有限、模型参数实际改变、GradScaler未因溢出降尺度。

| Z,Y,X | PyTorch峰值allocated（GiB） | PyTorch峰值reserved（GiB） |
|---|---:|---:|
| 16,128,128 | 4.37 | 5.06 |
| 32,96,96 | 4.76 | 5.64 |
| 64,64,64 | 4.17 | 4.99 |

以上是网络训练部分的实测值，不是完整猪CT的TIGRE投影显存、总进程显存或整幅推理显存；不包含尚未实现的数据缓存和完整验证管线。未测论文原始大块的T11训练。

## 5. 运行证据与复查

- [验证脚本](D:/desktop/课题2/proj/RSTAR4D/validation/audit_reproduction.py)
- [体模、模型和两阶段结果JSON](D:/desktop/课题2/proj/RSTAR4D/validation/artifacts/2026-11-16/smoke_results.json)
- [16×128×128显存](D:/desktop/课题2/proj/RSTAR4D/validation/artifacts/2026-11-16/memory_16x128x128.json)
- [32×96×96显存](D:/desktop/课题2/proj/RSTAR4D/validation/artifacts/2026-11-16/memory_32x96x96.json)
- [64×64×64显存](D:/desktop/课题2/proj/RSTAR4D/validation/artifacts/2026-11-16/memory_64x64x64.json)

JSON记录Notebook、验证脚本和外部依赖源码的SHA256。本次4条脚本命令退出码均为0；这表示审查脚本完成、各正向测试通过，**不表示原Notebook原样运行通过**。preflight明确记录 `ready_to_run_unmodified=false` 及其错误。

在项目根目录复查：

```powershell
& 'D:\python\miniconda\env\env\common\python.exe' 'validation\audit_reproduction.py' --mode smoke
& 'D:\python\miniconda\env\env\common\python.exe' 'validation\audit_reproduction.py' --mode memory --shape 16 128 128
& 'D:\python\miniconda\env\env\common\python.exe' 'validation\audit_reproduction.py' --mode memory --shape 32 96 96
& 'D:\python\miniconda\env\env\common\python.exe' 'validation\audit_reproduction.py' --mode memory --shape 64 64 64
```

同一路径下的审查产物会被重新生成。真实猪数据、外部脚本和官方Notebook/权重未修改。

## 6. 完整运行验收尚缺什么

按已更新的Spec实现可安装rstar模块、两个数据入口、采样与数据划分、两阶段调度、完整checkpoint/恢复、空间滑窗、指标和命令行；随后执行真实小病例的完整链路、固定病例过拟合、断点恢复等验收。实际训练/测试完成后，才能评价猪数据上的方法效果。

若目标是复现论文表格，另需匹配论文的数据、预处理、扫描协议、训练设置、ROI和评价口径。使用用户要求的猪数据/TIGRE流程可以验证方法迁移，但本身不足以验证论文的肺部实验数值。
