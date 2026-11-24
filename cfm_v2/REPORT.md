# CFM v2 无 data augmentation 消融

实际训练至 180,000 updates，active训练/validation/checkpoint耗时 9.861 小时。停止原因 `convergence_criterion`；最佳 EMA 在 165,000，validation velocity MSE=0.093990512。实际 LR restart 8 次。最终评估仅按 validation 选择。

与 CFM v1 相比，训练干预仅为关闭 D4 图像旋转/镜像及对应 Ex/Ey 交换。完全复用相同 7,857 张数据、6285/786/786 划分、训练集 normalization、逐样本 oversampling weights、seed 20261117、11.53M Conditional U-Net、batch32、AdamW、EMA、bfloat16、loss、LR 和 plateau 判据。从随机初始化开始。配置差异机器校验仅 version、augmentation、sampling.order；原数据和 v1 未改动。

保留未使用的 rotation/flip RNG 抽样；诊断update检查证实模型实际输入为原图、物理条件未交换，初始化、样本抽样及noise/time随机流与v1对齐。合成restart边界的恢复接线检查得到完全一致的权重、下一批index/RNG和LR（该诊断未完整训练20k）；诊断模型的固定validation重复差异为0。实际训练每20k从last的完整状态续接。cuDNN benchmark继续沿用v1，不宣称历史GPU运算bitwise完全复现。这是单个training seed的配对实验，不是多seed统计结论。

每5k对全部786张固定validation计算EMA velocity MSE，不做增强；loss是像素均值 `(v_theta-(x-epsilon))²`，不是FEM误差。test未用于训练、选择或停止。LR每20k一个cosine周期，首峰2e-4、200-step warmup，后续峰1e-4、谷3e-5，不重置optimizer moments。

从80k起（含80k），在20k整数倍的完整周期末，比较周期最低validation与此前历史最低值；连续2周期改善<1%启动evaluation，连续3周期<0.5%进入待停止，再完成至少一个完整restart周期仍无≥0.5%改善才停止；有效改善重置严格patience。采用v1最终300k/18小时active预算。阶段A触发step=100000。训练停止步数可随数据增强改变，不将各自停止后的比较称作固定训练量比较。

cycle改善率定义为 `(cycle开始前历史最低MSE - cycle内最低MSE) / max(abs(cycle开始前历史最低MSE), 1e-12)`。本次在160k完成连续第三个<0.5%的cycle、进入待停止；160k–180k完整确认cycle的最佳MSE为0.093990512，相比此前0.094306915改善0.3355%，因此在180k停止。这是按约定plateau规则停止，不是已证明完全收敛。周期evaluation从100k启用，此后每20k评估当时validation最优EMA；FEM指标不反向选择模型或决定停止。

checkpoint独立留存遵循用户修正后的规则：相邻validation的绝对相对变化首次<1%时激活，仅在20k整数倍steps保存。触发step=25000，实际留存steps=[40000, 60000, 80000, 100000, 120000, 140000, 160000, 180000]；完整状态和SHA256核验通过。last仍每1k更新，best_val_mse按最优validation保存。

| best EMA checkpoint | validation MSE | mean(d_mech) | s/a联合命中 |
| --- | --- | --- | --- |
| 100,000 | 0.0947206 | 0.3606 | 100/240 |
| 120,000 | 0.0945608 | 0.3499 | 105/240 |
| 125,000 | 0.0945602 | 0.3430 | 102/240 |
| 150,000 | 0.0943069 | 0.3775 | 111/240 |
| 165,000 | 0.0939905 | 0.3606 | 109/240 |

最终固定24 targets × 10 seeds全部使用64-step midpoint及阈值0二值化，真实FEM有效240/240、失败0，s/a联合命中109/240，Ex/Ey各±5%联合命中65/240。s/a阈值分别为log(1.05)、log(1.10)。FEM失败计入命中率分母，mean(d_mech)仅统计有效求解。

v1最佳step=280000、validation MSE=0.096081788、mean(d_mech)=0.320352、s/a命中120/240；v2最佳step=165000、mean(d_mech)=0.360630、s/a命中109/240。paired_validation.csv提供相同训练步数的validation比较；paired_final_mechanics.csv按target/seed配对各自最佳模型。完整曲线见ablation_curves.png。固定panel已有历史使用，不是新holdout，机械结果未反向选择checkpoint。

完整生成图及train-only retrieval见 `evaluation/runs/cfm_v2_step_165000/index.html`。固定seed全24目标、不筛选的图见run/fixed_seed_165000.png。数值审计不等于猫形视觉质量确认；不从机械误差直接推断视觉质量、创新性或memorization。retrieval设置沿用v1，包含其允许的轴交换规则。

恢复入口 `python cfm_v2/workflow.py`，实际命令在run/commands.jsonl。审计ZIP包含配置、代码、完整原始loss/LR、逐样本FEM、最终HTML、binary FEM输入、v1参考结果与hash清单，不包含权重或全训练图。无自动Git保存。

固定160k的补充对照已完成，详细结论见ABLATION_COMPARISON.md。v1→v2：validation MSE 0.098273841→0.094560572，mean(d_mech) 0.368404→0.354885，s/a命中109→124/240；两组均240/240 FEM有效。该step在首次v2 FEM结果之前选定，不影响最终165k模型的选择。各自最佳终点的机械结果则是v1更好；两种比较不可混为同训练量结论。

实际视觉检查已记录于run/final_visual_observations.md和run/matched_160000/visual_observation文件。固定首seed中T12、T18、T23/T24等仍有明确视觉失败；没有进行全部240张的blind评分或memorization审计。
