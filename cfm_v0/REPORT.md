CFM v0 实测报告

代码、主训练和固定 final panel 已实际完成。CFM v0 能生成带猫特征的多种材料图，并有明确的条件响应，但尚未实现可靠、独立且视觉合格的 s/a 控制：640 张随机样本仅 244 张联合属性命中；固定 40 张视觉检查中，仅 5 张同时满足基本连贯的猫形和属性命中。

已完成原始 224×224 单通道 CFM：NVIDIA GeForce RTX 4090，11,529,633 参数，batch 16，20,000 steps / 320,000 training draws（50.0 个等效 epochs），主训练 33.42 分钟。上限预先设为 20,000 steps 或 45 分钟；实际停止原因为 step_budget。仅按 validation velocity MSE 选择第 20,000 step 的 EMA checkpoint：validation=0.11360，训练结束后一次性 test（800 张）MSE=0.11140。峰值 PyTorch allocated GPU memory 6.39 GiB。

原数据 8,000 张全部通过 PNG/NumPy/CSV 行序与 pixel SHA256 核对；0=黑色硬材料，1=白色软材料，白色不是孔洞。原文件 hash 保持不变，未扩充数据。划分为 6,400/800/800，按粗粒度 s/a 分层；所有标准化、采样权重和 support density 仅来自 training set。沿用原 Q4 plane-stress、40×40×2 mm、Esoft=1/Ehard=10、横向自由拉伸 FEM。四张 training 图复算标签的最大相对偏差为 1.09e-09，另一次 full-resolution direct-solver 对照偏差为 1.89e-09。未重新求解原始全部 8,000 张。

条件为 $s=0.5[\log(E_x/E_\mathrm{soft})+\log(E_y/E_\mathrm{soft})]$、$a=\log(E_x/E_y)$。随机初始化 conditional U-Net，通过每个 residual block 的 FiLM 注入标准化 s/a 与时间。使用 $x=2b-1$、$x_t=(1-t)\epsilon+tx$、目标 velocity $x-\epsilon$ 的 MSE，t 均匀采样；无 augmentation、无 surrogate。training 8×12 格计数的 inverse-square-root 权重与 uniform 各混合一半，先 cap 4 再均值归一化，最终权重 0.716–4.009。所有生成采用固定 midpoint ODE 64 steps（128 NFE）、阈值 x≥0 为软相，未作清理或几何变换。

Training-standardized s/a 的 k=20 kNN support threshold 为 0.282075，取 validation 距离的第 95 分位；它只是经验支持指标，不证明物理可达。预先固定 20 个目标、5 条受支持的扫描区间，含正/负 a 与多个 s 水平。s≈1.02 的 ±0.05 邻域只有 83 张 training 图，s≈1.83 则有 797 张；低刚度正 a 的部分区间不满足 support 要求，未强行补成矩形网格。目标、seeds、阈值及 hash 均在 final generation 前锁定。

![support_and_targets](support_and_targets.png)

Final panel 为 20×32=640 张，所有目标共用 seeds 730000–730031；全部最终 binary maps 已实际 FEM。FEM 有效 640/640，失败率 0.0%。单图双方向 FEM smoke 中位 0.90 秒；final sampling 用时 2.93 分钟，8-worker final FEM 用时 3.65 分钟。命中阈值始终为 |Δs|≤log(1.05)、|Δa|≤log(1.10)。随机生成联合命中率 38.1%；|Δs| median/P90=0.0422/0.0971，|Δa|=0.0832/0.2046，总体偏差 Δs=0.0216、Δa=0.0043。

下表响应斜率理想为 1，交叉斜率理想为 0；括号为配对 seed bootstrap 95% CI。误差分位数只用有效 FEM；失败仍计入未命中。

| 扫描 | CFM 响应斜率 | CFM 交叉斜率 | Retrieval 响应/交叉 |
|---|---:|---:|---:|
| 固定 s=1.018，扫 a | 0.642 (0.478, 0.825) | 0.062 (-0.022, 0.143) | 0.984/-0.044 |
| 固定 s=1.827，扫 a | 0.869 (0.827, 0.908) | 0.024 (-0.022, 0.066) | 0.949/0.014 |
| 固定 s=2.037，扫 a | 0.599 (0.558, 0.642) | 0.056 (0.016, 0.099) | 0.887/-0.034 |
| 固定 a=-0.191，扫 s | 1.025 (1.016, 1.035) | -0.006 (-0.023, 0.011) | 0.996/0.019 |
| 固定 a=0.191，扫 s | 0.956 (0.916, 0.999) | -0.153 (-0.234, -0.072) | 0.977/-0.027 |

s 的响应斜率接近 1；a 的响应斜率只有 0.599–0.869，尤其 s≈1.02 和 2.04 时明显响应不足。固定 a=+0.191 扫 s 时，a_hat 随 s 增大而下降，交叉斜率 −0.153，CI 不含 0。固定 a=−0.191 时平均交叉斜率虽接近 0，却有 U 形漂移：逐 seed 的 a_hat 扫描极差中位数为 0.226；正 a 扫描为 0.202，均超过两倍 a 命中容差。固定 s 的三条扫描中，s_hat 逐 seed 极差中位数分别为 0.052、0.075、0.048。因此平均交叉斜率小不能证明独立控制，当前更接近有偏差且有离散性的统计控制。

![controllability](controllability.png)

联合 support 密度的 dense/medium/sparse 分界来自 training r20 的第 50/80 分位，和单独的 s 稀疏程度不同。

| 联合区域 | 目标/样本数 | 联合命中率 | Δs 绝对误差 median/P90 | Δa 绝对误差 median/P90 |
|---|---:|---:|---:|---:|
| dense | 4/128 | 58.6% | 0.0352/0.0836 | 0.0441/0.1121 |
| medium | 1/32 | 31.2% | 0.0369/0.0963 | 0.1215/0.1865 |
| sparse | 15/480 | 33.1% | 0.0450/0.1007 | 0.0915/0.2365 |

数据较充足的联合区域能较常生成属性接近的样本，但 58.6% 的命中率仍不够可靠；稀疏区域平均只有 33.1%。最困难的是 T16/T17（a=+0.191，s≈1.30/1.53），各只有 3/32 命中；s≈1.02 的 T00–T03 也只有 18.8%–28.1%，且 a_hat 的 seed 标准差达 0.179–0.199。稀疏不等于完全失败：T12 和 T15 分别有 20/32、18/32 命中。它们说明部分稀疏条件可生成成功样本，但不足以支持稳定使用。密集/稀疏差异是相关证据，未用消融证明数据密度是唯一原因。

实际查看每个目标前两个固定 seeds，共 40 张，不按力学结果过滤：20 张判为可辨且基本连贯的猫，19 张有猫特征但形态失真，1 张难以辨猫。其中“基本连贯的猫且属性联合命中”为 5/40。这是 assistant 对实际像素的视觉检查，不是 human confirmation，也不是 640 张的全面视觉合格率。高刚度正 a 及部分轮廓风格出现拉长、触边或碎裂；同一 noise seed 在不同条件下仍保留明显的姿态/风格关联。基本连贯不等于没有伪影或具备制造可行性。

![generated_examples](generated_examples.png)

Retrieval 仅搜索 6,400 张 training 图，按 $\sqrt{(\Delta s/\log1.05)^2+(\Delta a/\log1.10)^2}$ 选最近者，20/20 个目标命中；|Δs| median/P90=0.0038/0.0111，|Δa|=0.0083/0.0271。若对 CFM 做 FEM 筛选后的 best-of-32（最小上述 L2），有 20/20 个目标命中；这不代表随机生成性能。

全 training set 近邻检查：exact pixel duplicates=0/640，复用既有 strict visual predicate 的匹配数=0。最近 training 图的像素差比例 median=0.148，去位置/尺度并允许反射的 shape pooled RMS median=0.246。每目标 32 图两两像素差的 median 在 0.133–0.508 之间。这些仅为 memorization/diversity diagnostics，不能据此声称有意义的新结构。本轮已展示同一目标下多个非完全复制的候选，以及同一 seed 随条件变化的连续形态关联；但 training 邻域本身已有 16–369 张落在各目标的命中框内。当前没有证明生成模型在准确性、视觉质量或有用的新结构上优于 retrieval，它的价值暂限于候选提议与条件生成的可复现实验基础。

证据更支持模型在有限训练下对形态与条件分布的联合学习仍不足，且稀疏覆盖与较大误差相伴：同一 development seeds 从第 1,000 step 的碎片改善到第 10,000 step 的猫特征，说明训练进程确实影响视觉质量；但对应 8 张属性联合命中由 8/8 降为 5/8，velocity MSE 不能充当视觉或 FEM 指标。条件响应不足和非线性交叉漂移已被真实 FEM 证实；是训练预算、模型容量还是 conditioning 机制的主要限制，未做消融，不能因果定论。数据行序、相位含义和 solver 抽查未发现错误。二值阈值偏置不太可能单独解释大多数失败：619/640 张在 (−0.1,0.1) 内没有任何终态像素，其中 381 张仍未命中，因此在该区间改变阈值不会改变这些 binary maps；没有运行其他阈值的 FEM 或声称排除了所有离散化效应。本轮未追加训练、扩数据、surrogate 或新方法；本结论限于既定预算和受 training 支持的扫描区间。

复现入口：[run_commands.ps1](run_commands.ps1)；隔离环境依赖见 [requirements.txt](requirements.txt)。中断主训练可用 `python cfm_v0/train.py train --batch 16 --steps 20000 --minutes 45 --resume`，结束后用 `python cfm_v0/finish.py` 完成固定评估（应使用本目录 .venv 中的 Python）。新复现请设置 `CFM_V0_RUN_DIR` 为新目录，避免覆盖本次输出。外部接口示例：`python cfm_v0/generate.py --ex 4 --ey 5 --count 4 --output custom_4_5`，也可用 `--s ... --a ...`；自定义目标不纳入本报告。

完整 [逐样本 CSV](run/final/samples.csv)、[20 目标汇总](run/final/target_summary.csv)、[metrics/paired slopes](run/final/metrics.json)、[binary arrays](run/final/binary.npy)、[FEM diagnostics](run/final/fem.jsonl)、[冻结协议与 seeds](run/protocol.json)、[splits](run/splits.json)、[训练配置](run/train_config.json)、[checkpoint](run/checkpoints/best.pt)、[实际评估命令和用时](run/completion_commands.json) 保留为机器可读文件。最后 4 张展示例是明确标注的 FEM-selected illustrations，整体指标始终来自全部 640 张。
