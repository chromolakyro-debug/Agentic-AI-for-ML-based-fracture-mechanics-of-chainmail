# CFM v1 continuation 1

来源：cfm_v1/run/checkpoints/last.pt，280000 updates；完整恢复model/EMA/AdamW/scheduler/RNG。原模型、数据、划分、增强、权重及normalization不变。

新停止条件：每20k完整周期与此前best validation比较，改善<0.1%连续4次后arm，再完整经过一个restart周期仍无≥0.1%改善才停止。旧0.5%/3次判据停止计数清零，历史保留。evaluation已激活，每20k进行24targets×10seeds真实FEM；阈值及targets/seeds沿用v1，最终仅best-validation选择。最早因plateau停止为380k，安全上限480k或额外12小时active训练时间，先到先停。

每1k更新last，每20k保留step_XXXXXX.pt；best_val_mse继承父模型最佳并仅在新validation改善时更新。LR沿原绝对step，下一280001正好是自然restart边界，lr=1e-4。没有新warmup、moment reset或scheduler重启。

实际恢复验证：固定validation两次均0.0960818052，与父记录0.0960817882差1.7e-8（低于预设1e-7），不据此更改训练判据。

启动/中断恢复：

```powershell
.\cfm_v0\.venv\Scripts\python.exe cfm_v1_cont1/workflow.py
```

不重复bootstrap。训练曲线前280k为复制的父历史，后续为本次新增记录。evaluation输出统一放在evaluation/runs/cfm_v1_cont1_step_XXXXXX；训练配置和cache仍在本目录；复用evaluation代码，仅增添通用output_directory配置，不修改原评估结果。代码快照在run/code_snapshot。
