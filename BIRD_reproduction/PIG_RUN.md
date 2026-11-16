# 猪心数据与 TIGRE 运行说明

本机已完成第一轮 `Control_1_7569/Heart_with_contrast_back` 的600视角动态测量生成；不是从旧校准投影复制。当前扫描合同：10秒、300°、600视角，不包含终点。训练输入只有投影、几何和真实时间；不读先验或GT。

## 环境

使用已安装且验证通过的环境，无需重新安装：

```powershell
Set-Location 'D:\desktop\课题2\ref'
Push-Location BIRD_reproduction
$pigPython = 'D:\python\miniconda\env\env\common\python.exe'
$env:PYTHONDONTWRITEBYTECODE = '1'
```

设置不写pyc是为了保持父项目只读。Python3.9.23、torch2.3.1+cu118、pytigre3.0.0、NumPy1.26.3、SciPy1.13.1、nibabel5.2.1；GPU为RTX4070 Ti SUPER 16GB。系统默认 `python` 是另一个CPU环境；它能跑测试但不能替代这里的GPU环境。TIGRE需要已编译且匹配CUDA的安装，不能简单把普通PyPI同名包当成TIGRE。

## 现成数据与重新生成

已生成的数据合同：`runs/pig/round1_control1_10s/case.json`。投影为同目录 `projections.npy`，float32 `[600,474,648]`，约737MB。第599视角是9.983333333秒、299.5°。所有相位读取自用户原始CT，轴顺序、倍率、offset与当前CSV/脚本一致。

要重新生成一个新目录（示例名称与已交付结果不同）：

```powershell
& $pigPython -m bird.pig_cbct generate --project-root 'D:\desktop\课题2' --round 1 --series 'Control_1_7569/Heart_with_contrast_back' --duration 10 --out runs/pig/my_round1_control1_10s --batch-views 16 --gpu 0
```

`--round` 支持现有1–6轮对应的目录与CSV；`--series` 是该轮目录下的精确相对路径。只有 `with_contrast` 序列按原脚本规则选择。省略 `--views` 才生成全600视角；`--views 0 300` 只用于小型接入测试，case会标记为非完整扫描。生成器拒绝覆盖非空输出目录；中断后不发布可训练case，请换新目录重新运行。

只做几何和时间计划，不读取相位体素/不调用GPU：

```powershell
& $pigPython -m bird.pig_cbct plan --project-root 'D:\desktop\课题2' --round 1 --series 'Control_1_7569/Heart_with_contrast_back' --duration 10 --out runs/pig/my_plan.json
```

## 训练及恢复

4步完整链路检查；保留完整源体物理范围，物化网格降为24³。文件中的空间配置是XYZ，命令行 `--volume-shape` 是ZYX：

```powershell
& $pigPython -m bird.train_tigre --case runs/pig/round1_control1_10s/case.json --config configs/pig_tigre_smoke.json --out runs/pig/my_smoke --views-per-step 2 --voxel-chunk 4096 --device cuda:0 --gpu 0 --stop-after 2
& $pigPython -m bird.train_tigre --case runs/pig/round1_control1_10s/case.json --config configs/pig_tigre_smoke.json --out runs/pig/my_smoke --views-per-step 2 --voxel-chunk 4096 --device cuda:0 --gpu 0 --resume
```

物化网格默认采用配置的 `spatial_resolution`；训练几何的sVoxel和offOrigin始终来自完整源体。更大的**试验模板**为16级编码、1000步、2048条抽样射线、2048个正则点、64×96×96物化网格：

```powershell
& $pigPython -m bird.train_tigre --case runs/pig/round1_control1_10s/case.json --config configs/pig_tigre_pilot.json --out runs/pig/my_pilot --volume-shape 64 96 96 --views-per-step 2 --voxel-chunk 8192 --device cuda:0 --gpu 0
```

此1000步pilot尚未完整运行；后续复核已完成同网格/16级网络的20步短验证周期，见 REPRODUCTION_AUDIT。它不是已调参的猪心方案或64000步复现结果。TIGRE每个抽样时刻要求整幅体，分块VJP可降低INR激活内存，但完整体/梯度仍要驻存；增加网格尺寸会明显增加时间与内存。首例原始投影源体为560×512×512，不能把粗网格测试的速度外推到原始网格。

这条入口的图像预测和拓扑保持分支都通过TIGRE Ax；反向明确使用matched Atb，属于近似伴随。直接沿射线查询的原始参考入口 `bird.train` 与该入口的离散化不同。不得将 `.npy` case投影交给要求 `.npz` 数据的原入口。

## 推理

导出完整源体范围内的mu、mu_tp和毫米DVF：

```powershell
& $pigPython -m bird.infer --checkpoint runs/pig/my_smoke/checkpoint.pt --out runs/pig/my_smoke/source_volume.npz --times 0 5 9.983333333333333 --resolution 24 24 24 --chunk 4096 --device cuda:0 --threads 2
```

按**当前GT生成代码/CSV**的FOV网格、圆柱mask、裁剪/缩放规则导出raw mu：

```powershell
& $pigPython -m bird.infer_pig --checkpoint runs/pig/my_smoke/checkpoint.pt --case runs/pig/round1_control1_10s/case.json --out runs/pig/my_smoke/current_gt_grid --times 5 --chunk 65536 --device cuda:0
```

首例输出 `[467,256,256]`，单时刻float32约122MB。网络在10秒范围内按时间连续查询，但该模拟测量来自最近离散相位；不能把网络插值视为已验证的连续运动真值。有效时间为 `0 <= t < 10`，不把相位号当作秒数。

## GT兼容性与评价

现存Control_1 GT的 `gt_config.json` 与当前CSV不同：shift_up为18而非0、shift_left为10而非0、fov_source_pixels为128而非0，且强度为旧预投影多项式处理。文件形状相同也不代表物理网格或强度相同。已有只读报告：`runs/pig/existing_gt_audit.json`。

```powershell
& $pigPython -m bird.audit_pig_gt --plan runs/pig/round1_control1_10s/plan.json --gt-config 'D:\desktop\课题2\outputs\round1_contrast_gt\Control_1_7569\Heart_with_contrast_back\gt_config.json' --out runs/pig/my_gt_audit.json
```

应先按当前设置/原始mu生成对应GT，或明确重建另一套匹配的合同。当前导出不应用旧compare.csv，不应用GT valid mask，也不计算与旧GT的PSNR/SSIM。正式评价需明确相位与采集时刻、valid mask、共同强度域和data_range；同一相位在多个周期有多个实际采集时间，不能只按一个phase编号评价整次非周期扫描。

## 算子检查与交付文件

```powershell
& $pigPython -m bird.check_tigre --out runs/pig/my_operator_check.json --gpu 0
Pop-Location
```

`check_tigre` 检查真实GPU的Ax/matched Atb内积关系及autograd包装梯度，不证明任意几何下精确伴随。`PIG_VALIDATION.md` 给出本次结果。更新压缩包仅含代码、文档和小型记录；复制压缩包到其他机器后，须按原始CT重新生成投影，不能只凭打包的case元数据训练。
