# PriorNet 心脏 mask 两阶段训练

本分支提供 `train_prior_with_mask.py`。模型仍使用 `Dense2D(input, prior)`；mask 只在第二阶段计算损失和验证指标。当前工作区没有实际 NPZ、数据清单或 checkpoint，下面的路径是运行示例。

## 1. 准备 NPZ 与清单

每个样本保留二维 `input`、`label`、`prior`。约 20% 已标注的样本可额外存储二维 `mask`，形状与 `label` 的空间尺寸相同。`0` 是背景，默认所有大于 `0` 的有限数值（如 `1`、`2`、`255`）都合并为参与损失的区域。mask 不乘以图像读取时使用的 50 倍系数；负值和非有限值会报错。转换后须同时有前景和背景像素。

若只有部分标签代表心脏区域，可创建 JSON 文件，例如 `mask-labels.json` 内容为 `[2]`，并在微调命令中加入 `--mask-labels-file mask-labels.json`。也可写 `[1, 2]`。文件必须是非空 JSON 数组，元素是大于 `0` 的有限数值。未列出的标签（包括 `0`）按背景处理；如果所选标签在某个样本中不存在，读取器会报告该 NPZ 路径。标签选择同时用于微调训练和验证，所选标签会记录在 checkpoint 的 `config.mask_labels` 中。不传该参数即使用所有正值。

标注精度与 mask 的像素标签值是两回事。精度可由 NPZ 中的标量整数 `mask_quality` 提供，也可放在微调清单的第二列：

| 值 | 含义 | 第二阶段损失 |
| --- | --- | --- |
| `1` | 已确认的精确心脏分割 | 心脏区域、轮廓内外窄带和可选 Sobel 梯度项。 |
| `0` | 粗略心脏范围 | 平滑的区域权重。 |
| 缺失 | 质量未知 | 按粗略范围处理。 |

清单为 UTF-8 文本，每行一个绝对或相对于运行目录的 NPZ 路径。第二列为可选的 mask 质量，第三列为可选的病例 ID。例如：

```text
/data/case01_p0_slice1.npz	1
/data/case02_p3_slice8.npz	0
/data/case03_p1_slice5.npz
/data/nonstandard_name.npz		patient04
```

验证清单必须能确定病例 ID：文件名为 `<病例ID>_p<相位>_slice<切片>.npz` 时自动提取；其他命名必须在第三列提供。若文件名中的前缀不是实际患者 ID，务必在第三列填写真实患者 ID，避免把同一人的多个采集误当独立病例。可用空第二列直接填写第三列。NPZ 与清单同时给出质量且不一致时，读取器会报错。第一阶段清单可含无 mask 样本；第二阶段清单只能列 mask 已确认与 `label` 对齐的样本。代码检查 mask 数值与形状，但无法从数组本身证明病例、相位、切片和方向对齐。

准备四份清单：基础训练集、完整验证集、有 mask 的训练子集、有 mask 的验证子集。后两份分别是前两份中已确认对齐的子集；按**病例**划分训练与验证，不能把同一病例的不同切片或相位分在两侧。

在训练前记录四份清单的样本数和独立病例数，并按相位及 `exact/rough/unknown` 统计有 mask 样本。抽取每类标注叠加在 `label` 上检查。精确 mask 应对应同一呼吸相位的真实边界；错位样本须退出带 mask 清单。若验证集没有独立病例的精确 mask，区域指标仍可计算，但不能据此声称心脏边界得到改善。

## 2. 基础训练：100 个训练循环

```text
python train_prior_with_mask.py --stage base --base-train-list txt/sim_train_v2.txt --base-valid-list txt/sim_valid_v2.txt --output-dir runs/heart_mask --device cuda --batch-size 40 --num-workers 8
```

基础阶段每轮随机抽取训练集约 10%，仅使用整图 MSE，完全忽略 mask。因此这里的 100 个循环约相当于每个样本平均参与 10 次，不等于完整遍历训练集 100 次。日志中的 `cumulative_samples_seen` 记录累计训练样本次数。`runs/heart_mask/base/latest.pt` 是第二阶段所有对照实验的共同初始化 checkpoint。完整验证集关闭随机强度缩放。训练器会更新 SophiaG 的曲率估计；这修正了旧训练脚本未调用 `update_hessian()` 的问题，因此与旧脚本的优化轨迹不完全一致。

若已有兼容 `Dense2D` 的旧 checkpoint，可跳过基础阶段，直接用 `--stage finetune --base-checkpoint <旧checkpoint路径>`。旧 checkpoint 至少需要 `state_dict` 字段。

## 3. 有 mask 微调：最多 50 轮

先运行普通 MSE 对照组，再逐项增加 mask 损失。每个命令使用**同一个**基础 checkpoint、相同清单与随机种子，输出到 `runs/heart_mask/finetune/<模式>/`：

```text
python train_prior_with_mask.py --stage finetune --base-checkpoint runs/heart_mask/base/latest.pt --base-valid-list txt/sim_valid_v2.txt --masked-train-list txt/masked_train.tsv --masked-valid-list txt/masked_valid.tsv --output-dir runs/heart_mask --device cuda --batch-size 40 --num-workers 8 --loss-mode full
```

把末尾 `--loss-mode full` 分别改为下表中的模式，依次运行：

| 组别 | 模式 | 损失 | 作用 |
| --- | --- | --- | --- |
| B | `full` | 整图 MSE | 判断额外 50 轮和仅使用 20% 有 mask 样本本身带来的变化。 |
| C | `roi` | B + 精确或粗略区域损失 | 单独判断 mask 区域权重的作用。 |
| D | `boundary` | C + 精确 mask 的边界带 MSE | 单独判断精确边界附近的重建收益。 |
| E | `gradient` | D + 精确边界带 Sobel-L1 | 判断梯度项能否改善边缘，而不引入明显噪声或伪影。 |

E 组需额外传入正数 `--lambda-gradient`，例如先试 `0.01`。日志会记录未加权和加权的训练损失分项；据其实际量级调整系数，使训练初期加权梯度项约为整图 MSE 的 5%～20%。论文中的系数不能直接照搬到本项目的图像强度尺度。`boundary`、`gradient` 模式要求训练清单至少有一个明确标为 `1` 的精确 mask；未知质量不会触发边界项。

第二阶段每轮遍历全部有 mask 训练样本，默认最多 50 轮，连续 8 轮目标验证指标没有改善则早停。进入第二阶段时重设随机种子，使 `both` 与单独运行 `finetune` 在相同基础 checkpoint 和参数下使用相同的抽样随机流。微调使用新建的优化器；checkpoint 的 `config` 保存基础 checkpoint 的绝对路径和 SHA-256。每组写入 `latest.pt`、`best.pt` 和 `metrics.jsonl`；已有同名 checkpoint 时程序拒绝覆盖，重新实验请使用新的输出目录。

## 4. 判断结果

每轮分别在完整验证集记录全图 MSE/MAE，在有 mask 验证集记录区域指标、精确 ROI MSE/MAE、精确边界带 MSE/MAE 和 Sobel 梯度 MAE。每项指标先在病例内部平均，再对有该项指标的病例平均；日志同时记录样本数、病例数及精确、粗略、质量未知的 mask 数。精确验证 mask 存在时，`best.pt` 按边界带 MSE 选择；不存在时按粗略 ROI MSE 选择，并在日志标为 `boundary_assessed=false`。

重点比较 C 对 B、D 对 C、E 对 D。边界带误差衡量轮廓附近的**图像保真度**，不是预测分割轮廓的 Dice 或 Hausdorff 距离。保留固定病例和相位的预测、参考及误差局部放大图，检查是否出现过锐化、噪声或其他组织退化。第一版建议仅在独立病例上边界指标改善，且完整验证集 MSE 相对 B 不增加超过 5% 时接受；5% 是实验门槛，不是临床阈值。

如果试点确有稳定收益，再为剩余图像补粗略 mask。新增标注按 `0` 处理，原先有证据的精确标注继续按 `1` 处理；全量标注后需重新运行相同数据、相同步数的普通 MSE 对照，确认新增粗略 mask 是否带来增量收益。
