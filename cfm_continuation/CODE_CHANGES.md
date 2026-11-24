# 相对 CFM v0 的关键差异

原 `cfm_v0/*.py`、原始 checkpoint/log 和 `mechanics.py` 均不修改；开始与结束的 SHA256 记录用于核验。

| 项目 | 原 v0 | 本轮 |
|---|---|---|
| 入口 | `cfm_v0/train.py` 严格绑定 steps=20000 config | 新 `training.py` 独立不可变 continuation config，从旧 full-state checkpoint 迁移一次，后续严格检查新 config |
| 模型 / 目标 | 11.53M FiLM U-Net，linear Gaussian→image velocity MSE | 完全复用 `model.py`，无参数量或 loss 变化 |
| Optimizer / EMA | AdamW foreach，weight decay .01，grad clip1，EMA .999 | 保留所有 moments、global step、EMA、CPU/CUDA/NumPy/Python RNG |
| Batch | 16 | benchmark16/32/64 后按冻结规则选择32；具体测量见JSON |
| LR | 200-step warmup，2e-4 cosine降至2e-5@20k | 20k→22k升至1e-4，22k→200k慢降至3e-5；可选至多一次额外温和restart，实际是否发生以记录为准 |
| Validation | 每1k、固定800行/seed/batch16 | 每5k，直接调用原 validation_loss，数据/noise/time与batch16保持一致 |
| 停止 | 20k或45分钟预算 | 最早80k；两段20k窗口best val改善均<0.5%并结合实际视觉/FEM；200k或额外8小时硬上限 |
| Development | 原4条件×2seeds | 新15条件×2seeds，包含正负a与dense/sparse，每20k生成全部30图并真实FEM |
| Final | 固定20×32 | 复用同targets/seeds作历史配对对照，结束后只评一次，不称新holdout |
| Audit | 原独立审阅包 | 新ZIP含续训、milestones与原20k对照，不附巨大weights/full training images |

新的 `check_resume.py` 对同一连续下一步更新进行了 full-state save/reload 精确比对，检查 model、EMA、optimizer tensor 与 RNG digest。生产 continuation 在20k首次 validation 重算时出现约2e-6的BF16/cuDNN跨进程差异，初始1e-6 assertion在任何optimizer update之前停止；将对照容差设为1e-5后继续，并保留差异记录。固定 validation 的数据/noise/time 没有改动。不能据此宣称跨硬件或跨进程 bitwise reproducibility。

采样器、binary阈值、FEM solver、原始labels、train-only normalization/weights、support threshold、retrieval候选范围都没有改变。Development 的 connected-component/edge诊断仅用于辅助比较，不是 semantic classifier；最终仍进行实际图片检查。CPU image-neighbor分析继续复用原 NumPy 实现，避免混用Windows OpenMP runtime。

120k 检查后，用户补充“0.52%的MSE差别已经可以了”，认可最近20k约0.5182%的改善已足够接近平台。本轮因此改为在下一个完整checkpoint（125k）停止；这是一项有记录的用户停止指令，不是事后改写0.5%阈值或宣称原双窗口规则已满足。`user_stop_request.json`、`controlled_stop.json` 与 `stop_after_checkpoint.py` 记录过程。监控脚本确认原trainer已原子保存125k完整model/EMA/optimizer/RNG后，停止该task-owned进程；parent workflow的非零退出属于这个受控停止，原样保留。最终保留状态以125k checkpoint为准；保存与终止之间可能存在少量未保存更新，未计入报告的step/sample count。

最终选择last125k EMA，而不是最低validation的120k EMA：125k固定development修复了120k严重头部异常，主观连贯猫形24/30→28/30，FEM联合命中16/30→22/30；validation仅相对升高约0.074%。该选择在本轮final生成前提交。`best-validation.pt`始终保存真正的120k validation optimum；`last.pt`保存125k完整训练状态。为了直接复用旧final sampler的固定文件名约束，`best.pt`在选择结束后成为所选125k EMA的采样别名，不能只凭该文件名推断它是validation最优。

封存后增加了`training_summary.json`存在时禁止继续写同一个run的保护检查，避免后续训练混入本轮final/audit。未来恢复应在新目录迁移完整last.pt，并分别保留best-validation和采样选择；这一保护在训练结束后加入，不影响本轮已完成的optimizer updates，实际执行版本保存在code_snapshots中。
