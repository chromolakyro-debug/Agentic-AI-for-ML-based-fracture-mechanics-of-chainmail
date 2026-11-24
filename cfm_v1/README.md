# CFM v1

新的随机初始化baseline，保留11.53M Conditional U-Net、s/a FiLM、Gaussian-to-image linear flow、velocity MSE、AdamW(weight_decay=.01)、EMA=.999、gradient clip=1、bfloat16，以及续训阶段已验证的batch32。没有加载旧checkpoint。

新数据7,857张，沿用6285/786/786划分。原数据与旧实验只读。新训练集重新拟合z-score mean/std（用户已明确同意），保持物理s/a定义；oversampling严格复用旧8×12分箱、逆平方根频数/均匀混合和截断规则，只针对原始训练样本计算。类别不做均匀化：基础概率按样本数量，之后叠加旧规则的属性稀疏权重。按权重抽取原图后才做D4增强。

训练增强：uniform CCW rot90，再独立p=.5左右翻转；精确torch数组索引，无插值、缩放或反色。奇数个90°在原始Ex/Ey上交换，然后重算s/a并标准化；翻转不交换标签。validation不增强，786张固定顺序、固定CUDA noise/time序列，每5000 updates计算EMA velocity MSE；test不参与选择。MSE是每图每像素速度场的平方误差，不是FEM机械误差。

Scheduler按optimizer update更新，周期20,000 updates（约101.83个按累计样本数计的equivalent epochs），周期不增长。首周期max_lr=2e-4，200-update初始warmup；后续峰值降一次至1e-4并保持，min_lr=3e-5。restart仅改变LR，保留optimizer moments。模型、EMA、optimizer、scheduler、plateau状态、所有RNG和训练进度写入last，每1000 updates保存。

80k后比较完整周期内best validation与此前best；连续2周期改善<1%启动周期性evaluation；连续3周期改善<0.5%仅armed，至少再完成1个完整restart周期仍无≥0.5%改善才停止。epsilon=1e-12；有效改善重置严格patience。最大200k updates或12小时active train/validation/checkpoint时间，不含evaluation和setup；预算与平台停止分别记录。

启动或恢复统一workflow：

```powershell
.\cfm_v0\.venv\Scripts\python.exe cfm_v1/workflow.py
```

如仅需恢复训练到下一个周期边界（示例40000，需根据last.step选择）：

```powershell
.\cfm_v0\.venv\Scripts\python.exe cfm_v1/train.py --until 40000 --resume
```

主训练前执行prepare.py及checks.py；正式运行后不要再次prepare。checks包含八种精确变换、物理量条件重算、restart边界全状态恢复、两次固定validation一致性、plateau状态机和当前batch32吞吐验证。测试权重丢弃，主模型重新初始化。

周期和最终evaluation复用evaluation/run.py，以及此前用户冻结的24个targets、10个seeds。绑定新6,285张训练样本，使用immutable best_val快照。相同checkpoint/config依照现有内容校验cache复用。每次全部240张二值图跑原FEM；validation MSE只负责训练选择，机械结果不会反向选择best_val。

训练曲线与最终中文REPORT.md、AI审计ZIP集中本目录；主评估页面位于evaluation/runs/cfm_v1_step_XXXXXX/index.html。run/内保留完整逐step原始训练记录、validation、LR、恢复状态、配置、哈希、实际命令及milestone信息。

用户在约143k时新增要求：从150k开始，每10,000 updates独立保留完整状态checkpoint，路径为`run/checkpoints/step_XXXXXX.pt`。独立留存进程复制训练原子保存的last，核验内部step、完整状态和SHA256；不改变训练配置、随机序列、optimizer或LR。留存策略及hash清单分别为`run/checkpoint_retention_policy.json`、`run/retained_checkpoints.json`。中断后恢复时，在另一终端先运行以下命令，再运行workflow；它会自动跳过已保存快照，并在训练终止后退出。此前已经覆盖的历史checkpoint不能补回。大型权重不进入审计ZIP。

```powershell
.\cfm_v0\.venv\Scripts\python.exe cfm_v1/retain_checkpoints.py
```

后续用户在约177k将留存频率调整为20,000 updates，从180k开始生效；已保存的150k/160k/170k不删除。用户也授权若validation仍下降可延长预算：将在原200k边界复核，拟将上限扩至300k/18小时累计active训练时间，plateau判据不变。授权与实际是否延长分别记录，不将拟议预算当作已执行训练。

200k复核后实际扩展至300k/18小时：本周期validation改善0.633974%，尚未满足plateau停止。extend_budget.py仅迁移last的config.limits，并逐字段核验其余完整状态完全一致；原200k权重及配置保留。记录见run/budget_extension_applied.json。继续使用原plateau与LR周期。
