# RSTAR-Advance 本机预实验总结

2026-11-18，北京时间。代码基于 `b2a7f7f`，本地增加了 TensorBoard 整幅切片预览。按当前要求，本轮只进行了数据准备、短训练和显存预检。正式 Stage I 和八组消融尚未启动。

## 已完成的工作

| 项目 | 结果 |
| --- | --- |
| 扫描划分 | 沿用旧 baseline 的准确扫描名单，按 70% / 10% / 20% 取整为 **44 训练 / 6 验证 / 13 测试**。一个扫描的相位全部进入同一集合。10 头猪中有 9 头跨集合，因此指标以留出扫描解释。 |
| 模拟训练数据 | 63 个增强扫描、640 个 CT 相位已全部打包、通过 `validate-data` 并发布。独立 HDF5 包含 `heart_main` 的稀疏重建、全视角 FDK 参考、prior 和 FOV mask。各相位角度用原周期表和原重建索引复核，全部通过。 |
| 实测扫描 | 将 `真实数据/heart_only` 的 11 相位、prior 和实测角度单独打包，输入为 `[11,128,256,256]`，角度特征为 `[11,14]`。保留原 FDK 强度及配相来源，用于推理和 TensorBoard。 |
| 标注审核 | 核对 149 个标注相位的源 CT 与 mask 网格、配套 TIFF 网格及标签记录，全部符合预检条件。分布为训练 92、验证 16、测试 41。有标注文件的相位仍需确定可信监督范围。 |
| 体模与软件 | 本机 RTX A6000 通过 PyTorch CUDA 前后向及 TIGRE 水球 Ax/FDK。3 个 T10/T11 体模通过数据校验。CPU 回归 89 项中 78 项通过，11 项跳过。 |

所有数据格点、角度特征、扫描划分、标注哈希和样图来源均已通过最终验收。来源和结果目录见 [全量数据验收](/data2/liuyang/proj_2/rstar-advance/training_data/data_audit.json)、[模拟训练清单](/data2/liuyang/proj_2/rstar-advance/training_data/simulated/heart_main/paired/manifest.scan.jsonl)、[扫描划分](/data2/liuyang/proj_2/rstar-advance/training_data/split.json)、[模拟打包日志](/data2/liuyang/proj_2/rstar-advance/logs/real-pack.kChjwYX5.log)、[标注网格审核](/data2/liuyang/proj_2/rstar-advance/training_data/inventory/annotation_grid_audit.json) 和 [实测输入清单](/data2/liuyang/proj_2/rstar-advance/training_data/real/heart_only/manifest.jsonl)。

**标注对齐。** 149 个相位均完成最近邻对齐并逐个复核源文件身份、输出哈希和 FDK 网格。训练、验证、测试分别为 92、16、41 个相位。与旧 TIFF 在有效 FOV 内比较，整心 Dice 的中位数和最小值为 0.9986、0.9957，LV 为 0.9946、0.9838。两种结构在 149 个相位均有可计算的 Dice。Dice 用于核验几何偏移，正式 B 的可信域仍按结构和相位声明。 [对齐清单](/data2/liuyang/proj_2/rstar-advance/training_data/simulated/heart_main/source_masks/manifest.jsonl)

## 实测预实验

**TensorBoard。** 使用正式划分中的一个模拟训练扫描、一个模拟验证扫描和实测 `heart_only`，完成 Stage I 一轮短训、两次优化更新。已回读 256×256 的 TensorBoard 图像。实测固定相位 6、第 70 张轴位切片，展示输入和重建。验证固定同一扫描、相位及切片，显示输入、重建、GT、绝对误差和有符号残差。残差为重建减 GT，红色表示偏高，蓝色表示偏低。合成回归验证了续训可从轮次归档重发缺失事件。

[实际切片预览](/data2/liuyang/proj_2/rstar-advance/visualizations/data/tensorboard_preflight_slices.png) · [TensorBoard 日志](/data2/liuyang/proj_2/rstar-advance/preflight/tensorboard-actual.xWeOgk0w/stage1/tensorboard) · [模拟与实测角度图](/data2/liuyang/proj_2/rstar-advance/visualizations/data/phase_view_angles.png)

**模拟训练数据抽样。** 从新打包的 HDF5 各抽取一个 T10 和 T11 扫描。每例对照同一轴位层的有标注、无标注相位，展示稀疏输入、静态 prior、FDK 参考图和输入绝对误差。有标注相位叠加两种结构的边界。 [T10 训练样图](/data2/liuyang/proj_2/rstar-advance/visualizations/data/simulated_Control_3_7602_t10_sample.png) · [T11 验证样图](/data2/liuyang/proj_2/rstar-advance/visualizations/data/simulated_Stent_3_7550_t11_sample.png) · [图册索引](/data2/liuyang/proj_2/rstar-advance/visualizations/data/data_gallery.json)

**BMP 显存。** 用训练集 T11 扫描 `round1__Control_1_7569__Heart_with_contrast_back`、AMP 和 13.5 GiB PyTorch 分配限额测试三种中心块。每种尝试 5 次 Adam 更新，3 次实际更新。表中的每步时间取基准程序的计时步，含 AMP 校准过程，显存为 PyTorch 记录值。

| 中心块 Z×Y×X | 实际读取块 | 峰值分配 / 预留 GiB | 秒/步 |
| --- | --- | ---: | ---: |
| 8×16×16 | 92×120×120 | 11.21 / 13.50 | 7.66 |
| 8×24×24 | 92×128×128 | 12.76 / 13.46 | 4.65 |
| 12×24×24 | 96×128×128 | 11.36 / 13.39 | 3.25 |

[T11 基准记录](/data2/liuyang/proj_2/rstar-advance/preflight/bmp-memory-t11.c7cQs2w1/benchmark.json)

**BMP 边界与运动。** T11 显存试配的随机块未命中边界带，因此另用训练集 T10 扫描 `round5__Control_3_7602__Heart_with_contrast_stomach` 的已检查切片生成试用缓存。从该相位取一个同时命中两结构的中心块，GPU 验证了 119 个整心和 55 个 LV 边界带体素、36,864 个运动匹配窗口、实际非零 B 损失和 3 次有效更新。P 读取了 10 相位的角度特征。该次峰值分配为 12.91 GiB、预留 13.36 GiB，约 6.16 秒/步。两张试用标注只覆盖各自已检查的轴位切片，正式实验仍需确认全量可信域和边界命中率。

[边界覆盖](/data2/liuyang/proj_2/rstar-advance/preflight/bmp-paired.FRf8PxiF/annotated/coverage.json) · [带双边界监督的 GPU 记录](/data2/liuyang/proj_2/rstar-advance/preflight/bmp-paired.FRf8PxiF/supervised_benchmark.json)

**体模短训与续训。** 用同一套 TIGRE 体模做一轮共享 Stage I，从该权重启动 BMP，第一轮退出后从 `last.pt` 续到第二轮。两轮共完成 4 次有效更新，整心、LV 和运动窗口均有非零覆盖。标准滑窗入口在单个留出体模上写出预测、逐相位指标，TensorBoard 保留两轮参考图。仅更新 4 步时，预测 MAE 为 0.00625，稀疏输入 MAE 为 0.00216，说明仍处于训练初期。本项验收对应启动、续训和推理链路。 [体模验收记录](/data2/liuyang/proj_2/rstar-advance/preflight/bmp-phantom-loop.W2Bn4PUr/audit.json) · [体模指标](/data2/liuyang/proj_2/rstar-advance/preflight/bmp-phantom-loop.W2Bn4PUr/evaluation/summary.json)

## 正式消融的启动顺序

1. 全量模拟清单已发布，包含 63 个扫描。按 `annotation_candidates.jsonl` 确认结构和可信域，填写正式标注声明，再生成八组共用的 annotated manifest。旧 baseline 以原扫描名单对照，本轮所有组共用一份新训练的 Stage I 权重。
2. 复查本机 `nvidia-smi` 和在运行的任务，选择物理 GPU。预检覆盖了现有三个中心块，建议先保留同一配方。Stage I 拟 60 轮、512 块/轮，Stage II 拟 30 轮、256 块/轮。按本次实测每步耗时估计，BMP 的 7,680 次训练迭代约 7–16 小时，另加验证、预览、检查点和磁盘读取时间。
3. 先跑 Stage I，然后从同一份 `best.pt` 启动 **BMP（`MPB`）**。可以先在正式配置下跑到第 1 轮检查验证、TensorBoard、AMP 更新和 B 覆盖，再从 `last.pt` 续训。BMP 完成后依次启动 **BP（`PB`）、BM（`MB`）、PM（`MP`）**，然后 **B、M、P**，最后跑同配方全关组。
4. 对每组验证最优权重推理同一批 13 个模拟测试扫描，再用实测扫描生成推理结果。模拟域汇总 MAE、RMSE、SSIM 和逐扫描配对差值。正式轮廓比较使用同一个冻结分割器或同一盲法勾画协议，实测扫描报告重建和可视化。

### 命令参考

准备正式标注声明后，从仓库根目录执行。声明包含扫描 ID、标注相位、结构和可信域。下面的全量训练命令用于正式启动时。

```bash
cd /data1/liuyang/proj_2/rstar-advance
set -euo pipefail
OUT=/data2/liuyang/proj_2/rstar-advance
PY=/data2/liuyang/proj_2/rstar/venv/bin/python
SRC="$OUT/training_data/simulated/heart_main"
SPLIT="$OUT/training_data/split.json"
CFG="$OUT/configs/simulated"
RUN="$OUT/training_output/simulated"
LOG="$OUT/logs"
DATA="$SRC/annotated/manifest.jsonl"

test -f "$SRC/paired/publish_status.json"
test -s "$OUT/training_data/inventory/annotation_declarations.jsonl"
"$PY" -m rstar4d prepare-cardiac-annotations \
  --inventory "$OUT/training_data/inventory/series.jsonl" \
  --manifest "$SRC/paired/manifest.scan.jsonl" \
  --annotations "$OUT/training_data/inventory/annotation_declarations.jsonl" \
  --split "$SPLIT" --out "$SRC/annotated"
"$PY" -m rstar4d validate-data --manifest "$DATA" --split "$SPLIT"

nvidia-smi
: "${GPU:?先将 GPU 设为当前选定的物理 GPU 编号}"
export CUDA_VISIBLE_DEVICES="$GPU"
mkdir -p "$LOG"

run_logged() {
  local tag="$1"
  shift
  (
    flock -n 9
    printf '%s\n' "$BASHPID" > "$LOG/$tag.pid"
    printf '%s\n' "$CUDA_VISIBLE_DEVICES" > "$LOG/$tag.gpu"
    trap 'result=$?; printf "%s\n" "$result" > "$LOG/$tag.exit"' EXIT
    "$@" 2>&1 | tee -a "$LOG/$tag.log"
  ) 9>"$LOG/$tag.launch.lock"
}

run_logged stage1 "$PY" -m rstar4d train --stage 1 --config "$CFG/base.yaml" \
  --manifest "$DATA" --split "$SPLIT" --run-dir "$RUN/shared-stage1"

STAGE1="$RUN/shared-stage1/checkpoints/best.pt"
test -s "$STAGE1"
run_logged MPB "$PY" -m rstar4d train --stage 2 --config "$CFG/arms/MPB.yaml" \
  --manifest "$DATA" --split "$SPLIT" --init-from-stage1 "$STAGE1" \
  --run-dir "$RUN/ablation/MPB" --stop-after-epoch 1
```

检查 BMP 的 `train.jsonl`、TensorBoard、`epoch-0001.pt` 以及 `last.pt` 后，在同一环境续训。B 的权重从第 1 轮的 0 爬升到第 5 轮的 0.5，第 1 轮用两个结构的边界带命中数检查抽样覆盖。经确认的公共配置和数据清单保留在各运行目录。

```bash
run_logged MPB "$PY" -m rstar4d train --resume "$RUN/ablation/MPB/checkpoints/last.pt"
for arm in PB MB MP B M P baseline; do
  run_logged "$arm" "$PY" -m rstar4d train --stage 2 \
    --config "$CFG/arms/$arm.yaml" --manifest "$DATA" --split "$SPLIT" \
    --init-from-stage1 "$STAGE1" --run-dir "$RUN/ablation/$arm"
done
```

输出路径和评估约定见 [训练方案](../superpowers/plans/2026-11-18-local-pig-cardiac-ablation-run.md)。预实验标注清单保存在 `preflight/bmp-paired.FRf8PxiF`，正式启动使用全量清单。全域可信标注、独立轮廓协议及实测全量推理输出仍待落实。
