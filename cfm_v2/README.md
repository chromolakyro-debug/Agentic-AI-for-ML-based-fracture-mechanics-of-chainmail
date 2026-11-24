# CFM v2：no-data-augmentation ablation

本实验对照已完成的 CFM v1，仅关闭训练图的 D4 augmentation（随机 CCW rot90 与 horizontal flip），原图与物理 Ex/Ey 不变。从 seed 20261117 随机初始化，不加载 v1 权重。

完整沿用 curated_v1 的 7,857 张图、train/validation/test=6285/786/786、v1 的训练集 normalization 和逐样本 oversampling weights。沿用 11,529,633 参数 Conditional U-Net、s/a FiLM、linear flow、velocity MSE、AdamW、batch32、bfloat16、EMA、gradient clip。配置直接由 v1 最终配置生成，机器校验差异仅为 version、augmentation 和 sampling.order 三项。

训练仍消耗 v1 的 rotation/flip 随机抽样，但丢弃结果，不执行图像或标签变换，以对齐原图抽样与 noise/time RNG。保留 v1 的 cuDNN benchmark 和 TF32；不声称跨进程历史训练能 bitwise 复现。checks.py 会验证实际输入为未增强原图、条件未交换、初始化及 RNG 对齐、restart 边界完整恢复、固定 validation 重复一致。

LR 每20k updates restart，首峰2e-4、200-update warmup，后续峰1e-4、谷3e-5。validation 每5k，固定786张、顺序与noise/time，EMA velocity MSE选择 checkpoint。80k 后完整周期改善连续2次<1%启动 evaluation；连续3次<0.5%后再完成一个完整周期仍无有效改善才停止。采用 v1 最终上限300k/18小时 active训练时间，evaluation和setup不计。相同停止规则可能产生不同实际训练步数；报告会分别展示相同步数和各自停止后的结果，不将其混为固定训练量比较。

last 每1000 updates保存完整状态。相邻两次 validation MSE 的绝对相对变化 `abs(current-previous)/max(abs(previous),1e-12)` 首次严格小于1%时，启用留存机制；只在全局step为20k整数倍时独立保存完整 checkpoint，激活后不关闭。例如35k触发则从40k保存；40k触发则当步保存。该状态写入last随训练恢复，与plateau停止规则无关。最终用 best_val_mse 执行原有24 targets × 10 seeds、64-step midpoint、阈值0、真实FEM evaluation，复用 v1 的采样与retrieval配置。这个固定panel已有历史使用，不是新holdout。test不参与训练选择。这是一个training seed的配对消融。

另设固定训练量的 FEM 对照：在本轮80.6k、尚未看到任何v2 FEM结果时，选定160k作为matched-update比较点，因为这是v1已有完整评估的最早checkpoint。主训练完成后，若周期评估已覆盖同一160k EMA及相同评估设置，直接复用；否则对留存的160k EMA补做同一24×10评估。若训练未达到160k，则明确记录该比较不可用。此对照只用于结果分析，训练与best_val_mse选择继续遵循上述规则。预选记录及hash在run/matched_budget_evaluation_protocol.json及.sha256中。

首次运行（prepare仅一次）：

```powershell
.\cfm_v0\.venv\Scripts\python.exe cfm_v2/prepare.py
.\cfm_v0\.venv\Scripts\python.exe cfm_v2/checks.py
.\cfm_v0\.venv\Scripts\python.exe cfm_v2/workflow.py
```

中断后仅运行 workflow.py，自动从本轮 last 恢复。结果集中在本目录；evaluation 页面使用独立 cfm_v2_step_XXXXXX run ID。training_curves.png 与 ablation_curves.png 显示训练、验证和 LR；REPORT.md 与 cfm_v2_ai_review.zip 在最终评估后生成。原始逐步记录、配置差异及 v1 参考配置在 run/。不自动执行 Git 保存。
