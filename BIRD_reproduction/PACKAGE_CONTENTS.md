# 更新包范围

最新 `BIRD_reproduction_verified.zip` 包含独立复现源码、配置、测试、规格、方法解释、运行/验证记录，以及runs中小型JSON/JSONL/CSV元数据。包内路径统一位于BIRD_reproduction目录。此前TIGRE包保留为快照。

不包含：原始猪CT、投影npy、推理体npz/npy、checkpoint、论文PDF或sources提取内容。完整数据和测试权重留在本机原项目 `BIRD_reproduction/runs/`；case元数据本身不能代替被排除的投影数据，迁移后须重新生成。

`MANIFEST.sha256` 仅覆盖最新更新包所含文件（不含清单自身），不是整个本机runs/sources目录的清单。旧版 `BIRD_reproduction.zip` 及 `BIRD_reproduction_TIGRE.zip` 保留为此前交付快照；本次更新使用带verified后缀的压缩包。
