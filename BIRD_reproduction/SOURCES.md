# 来源、版本和公开代码核验

核验日期：2026-11-16（Asia/Shanghai）。

## 主依据

- 本地期刊原文：`../Nonperiodic dynamic CT reconstruction using backward-warping implicit neural representation with diffeomorphism regularization.pdf`。
- 题名：Nonperiodic dynamic CT reconstruction using backward-warping implicit neural representation with diffeomorphism regularization。
- Muge Du et al., Physics in Medicine & Biology 71 (2026) 105003；发表日期以 PDF 为准：2026-05-18；接受日期2026-04-22。
- DOI：[10.1088/1361-6560/ae639c](https://doi.org/10.1088/1361-6560/ae639c)。本次 web 工具未能打开 DOI 落地页，所以正文依据来自用户本地 PDF。
- PDF SHA-256：`ec95bc8d2654b63ded90512ef6525dd5669d9379ae466434cfecd221389cc209`。
- 本地提取文本：`sources/bird_journal.txt`。正文27页，另含下载封面；无附录章节。独立补充文件未获得。
- 原有 `../paper_summary.md` 的 BIRD 部分与期刊版交叉核对；不以摘要替代原文。

## 预印本版本差异

[arXiv:2505.03463v1](https://arxiv.org/abs/2505.03463) 发布于2025-05-06，[HTML全文](https://arxiv.org/html/2505.03463v1)。期刊首页注明其为实质性修订版。

| 项目 | 预印本v1 | 本次实现的期刊版 |
|---|---|---|
| 方法描述 | 后向 INR、微分同胚正则及解析重建增强 | 双特征图像 INR、后向查询、双向正则 |
| 最终输出流程 | 有独立解析运动补偿细节增强模块 | 式14的融合特征预测输出 |
| 对齐策略 | 仅作版本差异参考 | 公式8–27、图1–3为实现合同 |

因此没有把预印本的解析增强模块额外塞入期刊网络。需要复现预印本时应建立单独版本。

## BIRD 官方仓库检索结果

**未确认可公开下载的官方 BIRD 仓库，因此没有官方源码下载记录，也没有捏造 commit 或许可证。** 这表示本次检索没有找到，不是证明仓库不存在。

核验范围：

1. 本地全文检索 `code/github/available/supplement/appendix`。唯一实现库链接为 tiny-cuda-nn；没有 BIRD repository URL。
2. arXiv 摘要和 HTML 的代码链接与 implementation details；无已确认的作者 BIRD 仓库。
3. GitHub 连接器 repository search：`BIRD CT`、`"backward-warping" "CT"`、`"BIRD" "reconstruction"`、`"Nonperiodic dynamic"`、`BIRD diffeomorphism`、`"Muge Du"`。题名/方法专有查询返回空或无关同名项目；未确认作者仓库。
4. web 检索完整题名结合 `github/code`，以及 `2505.03463 github` 等。第三方索引只是发现线索，不作为算法事实的主要证据。未收到作者提供的私有链接。

期刊 Data availability statement 说明原始研究数据因敏感个人信息未公开，可向作者合理请求。它没有提供可直接下载的数据链接。本次没有向任何作者发送邮件或消息。

## 依赖与独立实现说明

- 原文使用 [NVlabs/tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn)，这不是 BIRD 官方仓库。
- 本实现 `bird/encoding.py` 自行实现1D/3D多分辨率哈希网格；未复制 tiny-cuda-nn 源码、未将其伪装成 BIRD 下载产物。表容量、级数等遵循论文公开设定，哈希/网格定位/浮点精度及速度不保证相同。
- 使用 [PyTorch FFT 接口](https://docs.pytorch.org/docs/stable/generated/torch.fft.fft.html)，完整 complex FFT、`norm="backward"`。
- 本次写入的代码和 spec 为独立研究实现。用户提供的论文与提取文本保持本地来源地位，不包含在纯代码交付压缩包内。

## 公式到代码映射

新增TIGRE接口依据已安装的pytigre3.0.0源码以及 CERN/TIGRE 官方 [PyTorch bindings](https://github.com/CERN/TIGRE/blob/master/Python/tigre/utilities/pytorch_bindings.py)、[Ax](https://github.com/CERN/TIGRE/blob/master/Python/tigre/utilities/Ax.py)、[Atb](https://github.com/CERN/TIGRE/blob/master/Python/tigre/utilities/Atb.py)。本机Atb默认是FDK，本实现明确传 `backprojection_type='matched'`；未依赖新版bindings的签名。近似伴随误差以本机真实GPU报告为准。

猪心合同依据用户本地脚本 `generate_round1_contrast_gt.py`、`generate_round1_contrast_recon.py`、`generate_pig_first_phase_prior.py`、`utils/tool.py`、`utils/phase_mapping.py` 以及每轮settings和cycle CSV。它们的路径/SHA256写入case的plan.json；10秒时长来自用户本次明确答复，不是论文或原脚本已给出的事实。

| 原文 | 内容 | 文件 |
|---|---|---|
| 式1–3 | 物理线积分 | `bird/geometry.py` |
| 式8–14 | backward、TP、FF、融合图像 | `bird/model.py` |
| 式15–16 | 双监督与排序 DFT | `bird/losses.py` |
| 式17 | forward DVF | `bird/model.py` |
| 式19–20 | cycle Jacobian | `bird/losses.py` |
| 式21 | forward 时间二阶导 | `bird/losses.py` |
| 式22 | 图像配准及仅 forward 更新 | `bird/losses.py` |
| 式23 | 两个单隐藏层 MLP 与拼接 shortcut | `bird/model.py` |
| 式24–26、图3 | 分解时空 hashgrid 与三路融合 | `bird/encoding.py` |
| 式27、§2.4 | 参数分组、总损失与优化 | `bird/train.py`、`configs/` |
