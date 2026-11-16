# 猪心 TIGRE 实际验证记录

日期：2026-11-16。本报告补充初版CPU验证，不将短测试描述为完成论文指标复现。

## 环境与数据

- GPU：NVIDIA GeForce RTX4070 Ti SUPER，16GB。
- 已安装环境：`D:/python/miniconda/env/env/common/python.exe`；Python3.9.23、torch2.3.1+cu118、pytigre3.0.0、NumPy1.26.3、SciPy1.13.1、nibabel5.2.1。本次没有安装/修改环境。
- 源：Round1 `Control_1_7569/Heart_with_contrast_back`，11相位CT；源网格ZYX=560×512×512，使用间距约0.375×0.921×0.921 mm，完整物理尺寸210×471.552×471.552 mm。
- 测量：10秒/300°/600view，detector=474×648，pitch=.616mm，DSO750/DSD1250mm；当前CSV该例offOrigin=0。
- 已从原始CT重新生成**全部600视角**，存于 `runs/pig/round1_control1_10s/`；不使用不兼容旧投影。
- 测量全部有限，min=0.0317528173、max=10.3692827225；SHA256=`258635c1b6b3ed70446813fe04c573861a76e33053cbd8b108a7462a3bd64674`。

## 实际通过的检查

1. **30项CPU测试**：原16项加猪心时间/HU/网格、GT后处理、分块VJP、损坏case合同和扫描终点检查。命令：`python -m pytest -q -p no:cacheprovider`，使用 `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`、`PYTHONDONTWRITEBYTECODE=1`。
2. **真实TIGRE GPU算子**：`python -m bird.check_tigre`；固定seed2026、12³小体、3角度、20×20探测器，内积相对误差0.00224949417（约0.225%）；包装器返回梯度与显式matched Atb最大差0。证明包装器路径一致，**不证明Ax/Atb严格互为离散转置**，也不是所有几何的误差界。原始报告 `runs/pig/tigre_operator_check.json`。
3. **真实猪投影训练**：24³粗物化网格保持完整源体FOV，3级小编码，4步、每步2视角/128像素、每项正则32点。两路图像均用TIGRE。各步全损失有限，初始与训练后参数最大绝对变化0.0731309801。日志 `runs/pig/tigre_smoke/history.jsonl`。
4. **GPU恢复一致性**：连续4步 vs 先2步再恢复到4步，所有模型张量最大绝对差 **0.0**。只陈述本机这次实测，不保证其他GPU/版本逐位一致。
5. **源体推理**：0、5、9.983333333秒，mu/mu_tp=[3,24,24,24]、DVF=[3,24,24,24,3]，全部有限。
6. **当前GT合同的网格导出**：5秒，输出467×256×256，有限，值域[0,0.0202147532]；没有GT监督/valid mask/多项式校准。该文件是4步网络预测，不是新生成的GT，亦不是重建精度证明。

记录汇总：`runs/pig/run_summary.json`。不同步骤抽到不同视角，不能用这4条batch loss计算同一固定扫描上的改善幅度。

## 独立审查及修复

独立reviewer只读核对原生成代码与新模块，确认轴顺序、offset、真实时间、无GT输入、完整FOV、matched反投影和VJP关键路径。发现并修复两项：

- 未传shape时忽略配置并意外物化完整源体：现在默认配置XYZ→ZYX，显式CLI可覆盖。
- 冗余case元数据可不一致而被接受：增加ID范围、时间/角度公式、完整覆盖、shape×spacing=size、bounds=(offset±size/2)反轴和几何尺度检查；训练和猪心导出共用验证。7个回归测试先观察失败、修复后通过。

另为通用导出补充TIGRE扫描终点排除测试。reviewer复核两项修复通过；GPU运行由主进程验证。

## 旧GT差异和未验证范围

`existing_gt_audit.json` 判定原有GT不兼容：当前shift_up/left/fov_source_pixels=0/0/0，旧GT=18/10/128；强度旧值为 `polynomial_valid_source_only_v2`。没有把这套旧GT用作训练或质量基准，也没有改写原文件。

未运行：其余62序列的测量、16级pilot的1000步训练、完整源分辨率或64000步训练、猪心图像质量指标、真实scanner散射/噪声/束硬化实验、与作者源码的数值对照。新测量是理想单能线积分模拟，当前cycle最近相位映射是分段恒定采样。粗网格与原始源体分辨率不同，存在有意的离散化误差。
