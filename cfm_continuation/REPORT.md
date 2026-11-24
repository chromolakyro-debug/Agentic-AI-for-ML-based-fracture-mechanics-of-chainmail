# CFM v0 continuation 实测报告

本轮从原 20,000-step `last.pt` 恢复 model、AdamW moments、EMA 与 RNG，累计训练到 125,000 steps。本轮 trainer 实际耗时 6.192 小时，累计处理 3,680,000 个训练样本（575.0 等效 epochs；加权有放回抽样，不等于每张图均匀访问该次数）。停止原因：用户接受最近20k约0.5182%的改善已足够接近平台；在下一个完整125k checkpoint停止，未达到8小时上限，也不宣称原双窗口<0.5%规则严格满足。

原 8,000 张数据、6,400/800/800 split、11,529,633 参数网络、FiLM conditioning、224×224 分辨率和 FEM solver 均保持不变。无 augmentation。材料定义为 0=黑色硬固体、1=白色软固体，`s=0.5[log(Ex/Esoft)+log(Ey/Esoft)]`，`a=log(Ex/Ey)`；所有力学结果来自阈值 0 后严格二值图的真实 FEM。

RTX 4090 上 batch 16/32/64 的实测吞吐分别为 150.1/158.5/45.2 images/s，peak allocated 显存分别为 6.48/12.37/23.08 GiB。选择 batch 32；batch64 的可用 device memory 为 0，吞吐反而下降。本轮训练峰值 allocated/reserved 为 12.37/12.74 GiB，含validation/save的平均吞吐约 150.7 images/s。约123k后瞬时吞吐降至56 images/s，原因未查明，环境快照与完整速度日志已保留；没有更改batch或训练配置。

LR 在 20k→22k 从 `2e-5` 线性升至 `1e-4`，随后按独立 continuation schedule 缓慢 cosine 下降，计划在 200k 达到 `3e-5`。保留原 optimizer state，没有重启原始 20k cosine。除这次预定 warm restart 外，未发生额外 LR restart。最终 LR 为 `5.64395e-05`。

固定 800 张 validation、固定 noise/time、固定 validation batch16，每 5k 计算 EMA velocity MSE。20k 重新计算值 0.1135988，best 0.0997982（120,000 steps），last 0.0998719。跨进程 BF16/cuDNN 的 20k 重算差异约 2e-6，已记录；远小于 0.5% 平台判据。平台状态：改善明显放缓，但严格原判据尚未满足。125k前两个20k窗口的best-loss相对改善为0.330%与0.501%；用户已明确认可约0.52%足够接近平台，因此停止。不能据此证明充分收敛；LR仍为5.64e-5，没有因极小LR停滞而宣称收敛。

最终评估选用 125,000-step last EMA；last 和 best-validation 权重分别保留，路径、大小与 SHA256 见 `run/checkpoint_inventory.json`。以视觉质量优先：125k相对120k修复了development中严重头身异常，连贯猫形24/30→28/30，联合命中16/30→22/30；validation仅相对升高0.074%。该选择在final生成前提交。best.pt只是复用旧sampler所需的125k采样别名，真正最低validation权重位于best-validation.pt（120k）。

## 视觉与力学结果

Development panel 为 15 个有经验支持的 targets × 2 个固定 seeds，每个 milestone 的全部 30 张图均展示并进行 FEM。最终视觉检查使用 20 个 targets 的前两个固定 seeds，共 40 张，未经视觉或 FEM 筛选。20k 原记录中基本连贯猫形为 20/40，本轮 AI 主观评级为 26/40；同时满足该视觉标准和两个力学容差的样本从 5/40 变为 17/40。该数字不是 human ground truth，也不能代表全部 640 张的视觉合格率。最终固定40图中，silhouette的腿与头身、部分pixel和纹理猫有所改善，但提升有限且不均匀。s≈1.018的pixel图仍常贴边，高正a的部分几何/纹理图仍横向拉长；T08 seed730001出现新的不可辨识白色块。Development连贯猫形为12→23→26→28→25→24→28/30（20k到125k），说明validation近乎持续下降与视觉质量并不逐点一致。

Final panel 复用上一轮固定 20 targets × 32 noise seeds，属于已使用过的历史配对 benchmark，不是全新 holdout。固定命中阈值仍为 `|Δs|≤log(1.05)` 与 `|Δa|≤log(1.10)`。本轮 640/640 FEM 有效，失败率 0.00%；失败计入未命中，误差分位数仅对有效结果计算。Final生成耗时 176.8 秒，8 workers FEM耗时 200.7 秒；另完成7个development checkpoints共210张图的FEM。

| 指标 | 20k baseline | Continuation |
|---|---:|---:|
| 联合命中率 | 38.12% | 62.81% |
| `abs(Δs)` median / P90 | 0.0422 / 0.0971 | 0.0268 / 0.0784 |
| `abs(Δa)` median / P90 | 0.0832 / 0.2046 | 0.0493 / 0.1452 |
| s / a bias | +0.0216 / +0.0043 | +0.0139 / -0.0036 |

同 target/seed 的联合命中率提高 24.69 个百分点，按 32 个 seed clusters bootstrap 的 95% 区间为 [18.28, 31.56] 个百分点；不将相关的 640 个结果视为完全独立样本。

| 扫描 | 主响应斜率 20k→本轮 | 交叉斜率 20k→本轮 | held 属性极差 median 20k→本轮 |
|---|---:|---:|---:|
| 固定 s=1.018，扫描 a | 0.642→0.854 | +0.062→-0.001 | 0.052→0.072 |
| 固定 s=1.827，扫描 a | 0.869→0.978 | +0.024→+0.030 | 0.075→0.061 |
| 固定 s=2.037，扫描 a | 0.599→0.822 | +0.056→+0.033 | 0.048→0.036 |
| 固定 a=-0.191，扫描 s | 1.025→1.007 | -0.006→+0.006 | 0.226→0.152 |
| 固定 a=0.191，扫描 s | 0.956→0.995 | -0.153→-0.001 | 0.202→0.127 |

固定a扫描s时，主斜率均接近1；原正a扫描的系统性cross slope从−0.153降到−0.001，是明确改善。但固定负/正a时seed内a极差median仍为0.152/0.127，单个seed仍可能漂移。固定s扫描a时主响应提高，s≈1.827最接近理想，s≈1.018和2.037仍压缩到0.854和0.822。s≈1.018的held-s极差反而从0.052升至0.072，低中刚度稀疏区独立控制仍不足；仅凭平均cross slope不能宣称完全解耦。平均 cross slope 接近零不能排除非单调漂移，应同时读 held range 与逐 seed 曲线。

| 区域（train-only support density） | 目标数 / 样本数 | 联合命中率 20k→本轮 |
|---|---:|---:|
| dense | 4 / 128 | 58.59%→85.16% |
| medium | 1 / 32 | 31.25%→50.00% |
| sparse | 15 / 480 | 33.12%→57.71% |

Dense联合命中85.16%，sparse为57.71%，均改善但差距仍明显。稀疏并非一律不可生成：T04(s≈1.827,a≈−0.373)与T12(s≈0.299,a≈−0.191)均为29/32命中；最差是s≈1.018扫描的T00–T03，仅4–8/32命中，T16(s≈1.300,a≈+0.191)为11/32。Development未覆盖s<1.16，这是本轮视觉选择对这些困难final条件覆盖不足的限制。局部覆盖与风格/尺寸/条件相关可能参与失败，但未通过消融确立因果。Support 是 train-standardized s/a 空间的 20-NN 经验指标，threshold 仅由 validation 固定；不是物理可达性证明。

Training-only retrieval 仍在 20/20 个目标找到满足两个容差的已有图像，`|Δs|` median=0.0038、`|Δa|` median=0.0083。CFM提供同一条件下32个不同seed的候选与可配对扫描的连续条件接口，但当前属性准确性仍明显低于retrieval，未证明比检索更有价值的新力学结构。实用上不能用CFM整体随机生成替代已有猫的高精度检索；best-of-32虽20/20目标可命中，它经过FEM筛选，不能替代62.81%的普通随机命中率。

生成图对全部 6,400 张 training images 的最近像素 Hamming 距离 median 从 0.1483 变为 0.1365，normalized shape RMS median 从 0.2460 变为 0.2044。本轮 exact pixel duplicates=0，既有 strict visual predicate matches=0。每个target仍有32个不同binary hash，within-target pairwise Hamming median范围从0.133–0.508变为0.142–0.530，未见整体diversity collapse。生成图同时更接近training形状；实际查看最相似示例T09 seed730017与training #2760，其姿态及纹理高度相似（normalized shape RMS=0.071），不能排除模板或局部形态记忆。跨条件同seed保持相似姿态本身也不等于独立样本多样性。像素不同不能单独证明有意义的新结构；风格相关和有无 memorization 也不能只靠一个自动指标判断。

## 证据边界与下一步

已证实：同模型/同数据续训组合改善了总体力学误差和小样本视觉质量，说明20k结果并非已用尽现有模型的能力；但同时改变了batch、LR和训练时长，不能分离各自贡献。训练集在20个目标都有命中检索例，故这些失败不能简单归因于目标根本无数据支持。Validation MSE不是视觉或FEM指标，120k低MSE仍发生严重视觉失败。639/640连续输出没有任何像素落在阈值±0.1内，其中237张仍力学未命中，因此小幅挪阈值不能修复绝大多数剩余失误。Conditioning对局部形态的映射不足、风格与力学耦合、训练覆盖及优化不足仍是待验证解释；没有证据确认必须扩数据、增参数或换生成方法。

先让独立人工盲评同40图及近邻极端例，确认猫形完整性和近似复用程度；再优先审查T00–T03的training邻域、风格/尺寸与单seed响应，把视觉与力学失败对应起来。本轮不再改final panel、checkpoint或阈值，也不自动追加训练、数据、surrogate或新方法。

训练曲线见 `training_curves.png`；各 milestone 全部固定样本见 `run/development_*/fixed_samples.png`；最终属性对照与示例见 `controllability_comparison.png`、`generated_examples.png`。逐样本结果、FEM diagnostics、binary arrays、配置、splits、seeds、完整日志和实际命令都保留。审计 ZIP 不含完整 training images 或模型权重；可独立复核存储结果和重新 FEM，但完整续训需另取原数据与权重。
