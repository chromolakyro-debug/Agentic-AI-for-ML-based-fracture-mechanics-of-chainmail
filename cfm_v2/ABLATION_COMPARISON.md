# CFM v1 / v2 augmentation ablation 对照结论

在事先选定的160k、同updates比较中，关闭D4的v2在本次固定panel上的平均机械误差和命中率更好；按各自validation最优模型比较最终结果，v1的机械指标更好。两种比较回答不同问题，不能合并为“augmentation一定有益/有害”。这是一个training seed的实验，固定panel已在历史评估中使用，不是新的holdout。

| 固定160k EMA | 原始validation MSE | mean(d_mech)，越低越好 | s/a命中 | Ex/Ey命中 | FEM有效 |
| --- | --- | --- | --- | --- | --- |
| v1，D4 augmentation | 0.098273841 | 0.368404 | 109/240 (45.42%) | 72/240 | 240/240 |
| v2，无augmentation | 0.094560572 | 0.354885 | 124/240 (51.67%) | 84/240 | 240/240 |

固定160k下，v2 validation MSE下降3.78%，mean(d_mech)下降3.67%，s/a命中增加15个，即6.25个百分点；Ex/Ey命中增加12个。按target/seed配对，126个样本的机械距离降低，114个升高；新增52个s/a命中，同时失去37个原有命中。这里的240个输出不代表240个独立训练模型。

160k在v2尚为80,600 steps、第一次FEM结果出现之前已选定，因为这是v1已有完整FEM评估的最早step；协议及SHA256见run/matched_budget_evaluation_protocol.json与.sha256。补充evaluation在主训练结束后执行，使用保存的160k完整checkpoint中的EMA。上表MSE来自恰好160k的原始validation记录，不使用该checkpoint里指向此前历史最佳的best_val_mse字段。它不参与选择、early stopping或延长训练。

两组使用完全相同的24 targets × 10 seeds、64-step midpoint、阈值0二值化、物理FEM参数、normalization、retrieval及tolerances。初始noise数组SHA256一致，evaluation代码hash一致，GPU均为RTX4090，PyTorch 2.1.2+cu121、CUDA12.1、cuDNN8801一致。逐样本binary/PNG hash、机械距离和命中标记的算术复核通过。该复核不是另一次独立FEM重算；evaluation本身调用既有真实FEM流程。

| 各自validation最优模型，训练量不同 | 训练停止step | best EMA step | best validation MSE | mean(d_mech) | s/a命中 | Ex/Ey命中 |
| --- | --- | --- | --- | --- | --- | --- |
| v1 | 280,000 | 280,000 | 0.096081788 | 0.320352 | 120/240 | 80/240 |
| v2 | 180,000 | 165,000 | 0.093990512 | 0.360630 | 109/240 | 65/240 |

在这个终点比较中，v2 best validation MSE低2.18%，但mean(d_mech)高12.57%，s/a命中少11个。v2只训练180k，v1训练280k，所以该表反映相同停止/选择流程下的终点，不是固定训练预算效果。最终模型仍按validation选择165k，未因为160k的FEM命中更高而替换它。

关闭D4既减少图像方向变化，也删除了奇数次90度旋转带来的Ex/Ey交换（a变号）。训练源图和逐源图oversampling weights相同，但有效condition覆盖会改变。固定evaluation的6×6粗网格中，T23/T24所在格的v2原方向训练样本数分别为0/1；考虑v1可交换方向的源图数分别为5/14。这是粗格子覆盖诊断，不代表物理不可达，也不能单独证明某个输出失败的原因。该ablation未把方向多样性和condition覆盖的贡献进一步拆开。

固定160k结果也不是各区域一致改善：负a半区mean(d_mech)从0.375205降到0.307622，正a半区从0.361603升到0.402148。极端正a的20个样本从0.738679升到0.887760，s/a命中2降到1；极端负a的平均距离虽下降，命中仍只有1/20。原始validation MSE更低不直接保证目标条件全域更可控。

实际查看了固定首seed 730000、全部24个targets的v1/v2 160k图，以及最终v2 165k图，未筛选。清晰的轮廓、纹理、几何猫形均存在；v2也保留T12细线退化、T18主体过小或破碎、T23/T24压扁或截断等失败。方向分布不同，旋转本身不作为失败。本次没有全部240张的blind人工评分，也没有visual nearest-neighbor或memorization审计，因此不宣称整体视觉更好或模型没有记忆训练图。

数据入口：run/matched_160000/comparison.json、paired_mechanics.csv、v1/v2的results.json和fem_inputs.npz；两个first_seed.png及visual_observation文件记录实际图像检查。run/paired_validation.csv对齐所有共同validation steps；run/paired_final_mechanics.csv对齐各自最佳模型。ablation_curves.png不将v2外推到180k以后的steps。

完整生成结果在工作区evaluation/runs/cfm_v2_matched_step_160000/index.html；审计ZIP中对应matched_160k_v2_evaluation/index.html，v1对应matched_160k_v1_evaluation/index.html。最终选定模型另见final_evaluation/index.html。
