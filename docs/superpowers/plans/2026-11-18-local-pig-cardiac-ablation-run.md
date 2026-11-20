# 本机猪心 CT 数据与 RSTAR-Advance 消融训练方案

供审核。代码以 `/data1/liuyang/proj_2/rstar-advance` 的 `b2a7f7f` 为起点。新数据、图表、训练归档与测试结果集中存入 `/data2/liuyang/proj_2/rstar-advance`。

## 数据来源与可比性

| 来源 | 检查结果 | 拟用方式 |
| --- | --- | --- |
| `/data1/liuyang/proj_2/data/Pig_4D_Cardiac_CT` | 6 个轮次、10 头猪、63 个增强扫描、640 个 CT 相位。10 个扫描为 T11，53 个为 T10。149 个相位有同名 mask。 | 核对病例身份和原始标注，新建 TIGRE 仿真。 |
| `/data1/liuyang/proj_2/prior/data/heart_main` | 63 套数据均通过转换器的源文件、配对和输出网格检查。逐套核对了 cycle 文件哈希、原始与当前代码生成的角度数组和相位投影索引，计数均与原重建清单一致。 | 将现有 CBCT、全视角 FDK 参考、静态 prior 和 FOV 有效域重新打包为新 HDF5。 |
| `/data1/liuyang/proj_2/prior/data/真实数据/heart_only` | 1 个实测 CBCT 扫描、11 个相位、输出网格 128×256×256，并附静态 prior。原始几何目录有实测逐视角参数，配相沿用同一 cycle 的假设。 | 用于固定切片预览和实测扫描推理。保持现有 FDK 强度值，GT 缺失项留空。 |
| `/data1/liuyang/proj_2/prior/gen_data/proj2/settings/heart_only` | 当前分支用该目录的逐轮 settings、calibration 和 `cycle.csv` 检查 63 个病例，全部通过。 | 作为两套数据的几何、周期和强度校准来源。 |
| `/data2/liuyang/proj_2/rstar` | 旧 baseline 的划分单位是扫描，共 44/6/13。清单用 scan ID 填训练接口的 `animal_id`，真猪 ID 另存为 `biological_animal_id`。10 头猪中有 9 头跨集合，例如 `7569` 的 round 1、2 在训练集，round 3 在测试集。 | 复用旧扫描名单，让测试扫描与历史结果一一对应。主消融仍从头训练 Stage I，并在同一配方下补训全关组。 |

模拟训练集用 `heart_main` 的 CT 前投影与 FDK 产物。原配置生成 600 个等角视角，跨度 300°，相邻 0.5°。按 `cycle.csv` 的心动周期把全部投影分配给最近相位，各相位角度间隔不固定。T10 相位视图数为 32、66、67、67、67、67、67、67、66、34，T11 为 28、62、60、60、61、58、61、60、60、62、28。全视角参考使用同一相位的全部 600 个角度重建。

主实验用带 GT 的模拟训练集完成八组。各组先在同一批模拟测试扫描上做配对评价，再对实测扫描推理。新增 TIGRE 体模用于验证算子。另加泊松噪声的模拟压力测试可按需进行，作为独立配置记录。

## 扫描划分

按要求以完整扫描为单位，训练 70%、验证 10%、测试 20%。沿用旧 baseline 的准确名单及取整结果。每个模拟扫描的所有相位和衍生版本进入同一个集合。猪 ID 仍保存在 `biological_animal_id`，训练清单按扫描标识填 `animal_id`。同一头猪的不同轮次或体位可出现在不同集合。

| 集合 | 扫描 | 涉及猪数 | CT 相位 | 有 mask 的相位 |
| --- | --- | ---: | ---: | ---: |
| 训练 | 44 | 10 | 448 | 92 |
| 验证 | 6 | 3 | 61 | 16 |
| 测试 | 13 | 8 | 131 | 41 |

模拟数据共用 `training_data/split.json`。先在训练和验证扫描上做图像质检、选配方与选择 checkpoint，参数冻结后推理留出扫描。报告明确测试单位是扫描，另外保留猪 ID 供汇总。实测扫描单独管理。

## 输出目录

```text
/data2/liuyang/proj_2/rstar-advance/
  configs/                      # 仿真、训练八组、固定评价参数
  training_data/
    inventory/                  # 病例清单、来源核对和标注声明
    split.json
    real/heart_only/            # 实测扫描输入、prior、角度与来源信息
    simulated/heart_main/{paired,source_masks,annotated}/
    simulated/phantom/
  visualizations/data/          # 抽样 PNG、采样角度图、图页索引
  training_output/simulated/
    shared-stage1/
    ablation/{MPB,PB,MB,MP,B,M,P,baseline}/
  test_results/{real,simulated}/
    <组名>/{predictions,evaluation}/
    paired_report/
  preflight/                    # 环境、数据验收、显存试配、短训练和续训演练
  logs/                         # 数据生成和各训练任务的日志、PID 及退出状态
```

旧 baseline 的训练数据约 196 GiB。本次用两个猪病例测实际 HDF5、角度和标注缓存大小，再加入八组推理产物和全部逐轮归档估算占用。检查时 `/data2` 可用约 1.9 TiB。全量序列默认保存 HDF5，抽样导出图像以控制空间。

## 1. 生成数据并交付抽样图

- [ ] 用 `inventory-pig` 生成清单，按真实动物 ID、轮次、体位和相位核对现有 GT、recon 和 mask。存档旧扫描 ID 与当前病例 ID 的一一映射，以及 split、生成配置、命令、源码版本和来源哈希。
- [x] **模拟训练样本。** 63 组配对 HDF5 已通过全量校验并发布到 `training_data/simulated/heart_main/paired`。输入使用 `phase_*_reconstruction.tif`，目标使用 `phase_*_gt.tif`，有效域使用 `phase_*_gt_valid_mask.tif`，静态 prior 使用 `prior_image.tif`。保持原有 FDK 后校准和 `all_views_fdk` prior。按 `cycle.csv` 和原角度生成函数保存 `angles_rad`、`state_label` 和来源哈希供 P 读取。
- [x] **实测推理样本。** 已按原网格打包 11 个相位与 prior，并从实测 `angles_h.raw` 及其原配相来源恢复角度标签。验证了 `[11,128,256,256]` 输入、`[11,14]` 角度特征和源文件哈希。保留实测用 Hann 滤波、原物理网格和现有强度值。
- [x] **独立体模。** 已使用 `make-phantom` 和 `simulate-pig` 生成 3 个 T10/T11 体模，记录投影、角度及相位，并通过 `validate-data`。
- [x] **标注对齐。** 149 个相位已依据原 NIfTI affine 和 FDK 输出网格做最近邻对齐。对齐产物存放在 `source_masks/`，逐相位记录输入、输出哈希及有效 FOV 内与原 TIFF 标注的 Dice。
- [ ] **正式边界缓存。** 设计约定标签 1∪2 为整心、2 为 LV。确认结构语义及可信范围，为两结构分别记录 `available`。经确认的整幅标注使用 `whole_fov_confirmed`。缺标相位仍用于图像重建。用 `prepare-cardiac-annotations --half-width-vox 2` 保存边界和覆盖率。
- [ ] **可视化与验收。** 模拟域抽取至少 2 个扫描，覆盖 T10/T11、不同体位及已标/未标相位。另导出实测扫描的切片。交付带病例、相位和切片索引的 PNG、角度分布和索引表。用 `validate-data` 校验模拟清单的网格、校准、哈希、FOV 和扫描互斥，单列缺标相位及两结构覆盖率。

源数据抽查图已保存到 `preflight/source-preview/`，含验证集 T11 仰卧和训练集 T10 俯卧。在各自有效 FOV 内，最近邻映射与已有 TIFF 标注的标签 1∪2 Dice 为 0.996、0.999，标签 2 Dice 为 0.990、0.995。原转换清单中 149 个相位均保有标签 1 和 2。正式标注需逐例确定可信域。

## 2. 本机环境与短训练

- [ ] 优先使用已检查到依赖的 `/data2/liuyang/proj_2/rstar/venv/bin/python`，从新代码目录用 `-m` 调用。该环境有 Python 3.9.25、PyTorch 2.3.1+cu121、pytigre 3.0.0、H5Py 和 TensorBoard。正式运行前在授权环境重查 `nvidia-smi` 与现有 PID，按空闲显存和利用率选卡，用 `CUDA_VISIBLE_DEVICES` 限定单卡。方案核查时六张 RTX A6000 都有计算进程。选卡后用 `doctor` 核对 CUDA 前后向和 TIGRE 水球 Ax/FDK。
- [ ] 完成小体模闭环、数据检查和短训练。先在独立预检目录训练 1 轮 Stage I，再使用模拟训练集和短训权重，从 MPB 开始试配 Stage II 三种中心块。评估实际扩展块、T11 最重病例、显存峰值和步耗时，并用含标注的训练块核对整心/LV 命中及 M 的有效 LNCC 窗口。几何开关须读到非空且与相位一一对应的角度。MPB 首轮应有有限的优化更新，随后从 `last.pt` 续训并跑通滑窗推理。
- [ ] Stage II 从 `pig_cardiac_ablation_16gb.yaml` 试配。三种中心块为 `[8,16,16]`、`[8,24,24]`、`[12,24,24]`，激活检查点开启，位移上限、LNCC 半径、读取支撑和对齐规则按仓库配置。首轮可试 `--memory-limit-gib 13.5`，经基准复测后锁定同域全部八组的裁块和训练预算。若显存或监督覆盖无法满足，调整共同配置并重跑预检。

## 3. 正式训练与恢复

拟采用 Stage I 60 epoch、Stage II 30 epoch，Adam 初始学习率 1.5e-4，按仓库配置分段降低，归一化尺度 0.02。每轮拟用 512/256 个训练块（Stage I/II），每个验证扫描拟固定 2 个训练验证样本，参考旧本机配置。Stage I 每次读单层 Z 和全部相位，Stage II 用多层 3D 块及全部相位。八组共享同一份 Stage I `best.pt`。用预检的步耗时和验证耗时估算总时长，用固定验证块的扫描等权 L1 选 checkpoint。

| 顺序 | 用户组名 | 代码配置名 | 模块 |
| --- | --- | --- | --- |
| 1 | BMP | `MPB` | B 边界 ＋ M 运动 ＋ P 几何 prior |
| 2 | BP、BM、PM | `PB`、`MB`、`MP` | 对应双模块 |
| 3 | B、M、P | `B`、`M`、`P` | 对应单模块 |
| 4 | 全关对照 | `baseline` | 原 RSTAR4D 主干，使用同域同配方 |

先训练 MPB，再完成双模块组，然后单模块组与全关组。各组从共同 Stage I 权重起跑。B 的半带宽为 2 vox，整心与 LV 等权，权重在前 5 轮升到 0.5。M 的结构匹配权重为 0.05、平滑权重为 0.01。P 使用与重建输入对应的 `geometry` 模式。

入口示例使用已经生成并锁定的配置和清单。`GPU` 取运行前确认的物理编号。示例在正式 Stage I 结束后用正式权重复核 MPB 的显存和更新。

```bash
cd /data1/liuyang/proj_2/rstar-advance
PY=/data2/liuyang/proj_2/rstar/venv/bin/python
OUT=/data2/liuyang/proj_2/rstar-advance
DOMAIN=simulated
: "${GPU:?先设为本次选择的物理 GPU 编号}"
export CUDA_VISIBLE_DEVICES="$GPU"
DATA="$OUT/training_data/$DOMAIN/heart_main/annotated/manifest.jsonl"
SPLIT="$OUT/training_data/split.json"
RUNS="$OUT/training_output/$DOMAIN"
CONFIG="$OUT/configs/$DOMAIN"

"$PY" -m experiments.cardiac_ablation write-matrix --base "$CONFIG/base.yaml" --out "$CONFIG/arms"
"$PY" -m rstar4d train --stage 1 --config "$CONFIG/base.yaml" --manifest "$DATA" \
  --split "$SPLIT" --run-dir "$RUNS/shared-stage1"
"$PY" -m rstar4d benchmark --stage 2 --manifest "$DATA" --split "$SPLIT" \
  --config "$CONFIG/arms/MPB.yaml" --weights "$RUNS/shared-stage1/checkpoints/best.pt" \
  --memory-limit-gib 13.5 --out "$OUT/preflight/${DOMAIN}_MPB.json"
"$PY" -m rstar4d train --stage 2 --config "$CONFIG/arms/MPB.yaml" \
  --manifest "$DATA" --split "$SPLIT" --init-from-stage1 "$RUNS/shared-stage1/checkpoints/best.pt" \
  --run-dir "$RUNS/ablation/MPB"
```

正式调用之前先查任务是否已运行，记录每个任务的 GPU、日志、PID 和退出状态。每轮保存完整 epoch 归档、`last.pt`、`best.pt`、`train.jsonl` 和 TensorBoard。因中断需要续训时，确认同组 `checkpoints/last.pt` 可读后用 `train --resume <last.pt>`，从下一未完成轮继续。短程验证用独立目录，正式组保持同一数据和配置快照。检查每轮的抽样哈希、更新数、SSIM 有效窗、AMP 溢出、M 窗口及 B 边界带命中。

TensorBoard 的每轮图像固定两张完整轴位切片。实测默认相位 6、切片第 70 张（零基 69），展示稀疏输入与本轮模型重建。模拟验证使用固定扫描的中心相位和中心层，展示重建、GT、绝对误差及有符号残差，即重建减 GT，红色为正、蓝色为负。通过 `tensorboard_real.manifest` 指向实测清单，并设置 `phase_label: 6`、`axial_slice: 69`。图像随逐轮归档保存，以供续训恢复事件。

## 4. 测试与最终交付

- [ ] 用各组的验证最优 checkpoint，在同一批模拟测试扫描上执行 `infer --split-name test --mode tiled`。滑窗和 overlap 先在验证集锁定。每组保存 HDF5、抽样 PNG、`evaluate` 的 MAE/RMSE/SSIM 和逐扫描/逐相位表，同时报告稀疏输入的分数。实测扫描保存重建体和抽样预览。
- [ ] 轮廓评价需有冻结分割器或盲法勾画。可从训练扫描的可信标注建立分割器，用验证扫描定阈值，冻结版本和权重再统一应用于每组重建。若采用人工勾画，用同一盲法批次与协议。当前已核查的工程和 baseline 产物尚未确认可复用的独立轮廓模型。可先完成图像指标，轮廓结果以独立预测清单和可信参考齐备为验收条件。
- [ ] 拟在测试前为两结构各锁定 Surface Dice 容差 2 个输出体素。按组报告整心和 LV 的 Surface Dice、HD95、ASSD、边界带 MAE、原始体素间距、有效相位数和扫描数。空预测作为失败记录，距离未定义保留空值。用同一参考和独立轮廓协议比较各组。每组训练目录用相对链接映射到测试结果中的 `evaluation/`，使现有 `compare --runs` 同时读到评估、训练日志和初始化记录。
- [ ] 交付模拟测试总表、逐扫描配对差值及分母、猪 ID 汇总、参数量、训练显存/耗时、推理耗时和实测重建。核对八个正式组的最终轮次、权重、预测与评价文件，检查各组有相同训练划分、Stage I 权重和逐轮样本顺序。模拟测试为 13 个扫描，涉及 8 头猪。

请审核主实验保留原来的模拟采样和训练预算，以及独立轮廓采用哪套分割模型或勾画协议。
