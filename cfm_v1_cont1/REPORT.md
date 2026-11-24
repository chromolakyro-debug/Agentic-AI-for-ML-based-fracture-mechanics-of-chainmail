# CFM v1 continuation 1

从v1 280k完整last恢复，训练至480,000，新增200,000 updates，本轮active训练/validation/checkpoint耗时11.092小时；停止原因`update_budget`。原v1数据、checkpoint和日志保留。新输出集中cfm_v1_cont1。

模型、AdamW moments、EMA、scheduler及RNG恢复检查逐项一致；恢复后的固定validation与原280k差约1.7e-8，两次复算完全一致。11.53M U-Net、batch32、bfloat16、增强、train-only归一化及oversampling沿用v1，固定划分6285/786/786不变。固定validation noise/time，每5k计算EMA velocity MSE，定义为`mean[(v_theta-(x-epsilon))²]`，不是FEM力学误差。

停止门槛由每20k周期改善<0.5%连续3次，改为<0.1%连续4次，之后仍需额外完整restart周期；≥0.1%改善清空严格计数。只清空旧停止计数，不删除旧历史或重置训练状态。续训从300k周期开始判定；已激活evaluation，每20k检查并评估当时的best-validation，完全相同的checkpoint/config复用缓存；本轮7个不同checkpoint各完成240样本评估，并每20k留存完整训练checkpoint。LR按原绝对step继续，峰1e-4、谷3e-5，每20k restart一次，不清空optimizer moments。本轮安全上限额外200k updates或12小时active时间。

最终按best-validation选择step 480,000，MSE=0.094092295；原280k为0.096081788。最终240张真实二值FEM有效240、失败0，mean(d_mech)=0.306281，s/a联合命中133/240、Ex/Ey ±5%联合命中86/240。`d_mech=sqrt((Ex_FEM-Ex_target)^2+(Ey_FEM-Ey_target)^2)`；s/a容限为log(1.05)、log(1.10)。原v1 280k对应mean(d)=0.320352、s/a=120/240。该24×10 panel为复用历史panel，不是全新holdout，FEM不用于调参或选择checkpoint。

逐阶段结果见run/evaluation_history.json，所有图像与两组train-only机械retrieval在`evaluation/runs/cfm_v1_cont1_step_480000/index.html`。曲线保留原280k历史并继续追加。480k与原280k的同一seed730000全24targets已实际并排查看：480k的T05有更清楚的面部，T09增加眼睛细节，T11仍是完整可辨识的坐姿猫，T14/T15/T20保留像素猫特征。T07变得很细长、细节减少，T13扁平且有分离片段，T10姿态倒转并触边，T12/T18仍明显截断，T16/T17仍触边，T01/T04/T19/T23仍呈圆钝的简化轮廓。因此有局部改善，也有退化及风格变化，不能据这24张认定整体视觉质量明显提高。最终240张FEM全部有效，mean(d)=0.306280854，s/a联合命中133，Ex/Ey联合命中86；相对原280k平均机械距离降低约4.39%，联合命中数提高，但各milestone并非单调改善。 固定样本为全部24个targets的第一枚预定seed730000，未经质量筛选；这些观察不等同于全部240图的人工评分，没有完成memorization或多样性的量化检验。

恢复：`cfm_v0/.venv/Scripts/python.exe cfm_v1_cont1/workflow.py`（未完成时从last继续；完成后仅复核最终评估）。配置、命令、hash与完整状态均在run中；审计ZIP不含大权重和训练图片。

## 本轮核验补充

本轮实际新增LR restart 10次；累计restart 23次。原v1 checkpoint的SHA256再次核验未变。最终HTML数值、480个retrieval位置的train-only去重/轴交换/排序和FEM query选择已独立核对通过。

| Best-validation checkpoint | validation MSE | mean(d_mech) | s/a联合命中 | Ex/Ey联合命中 |
|---|---:|---:|---:|---:|
| 280k | 0.096081788 | 0.320352 | 120/240 | 80/240 |
| 300k | 0.095631428 | 0.311856 | 137/240 | 92/240 |
| 320k | 0.095431266 | 0.331425 | 128/240 | 88/240 |
| 340k | 0.095314828 | 0.330176 | 133/240 | 87/240 |
| 360k | 0.094884390 | 0.306678 | 138/240 | 91/240 |
| 415k | 0.094672950 | 0.327107 | 121/240 | 81/240 |
| 440k | 0.094465749 | 0.338907 | 130/240 | 84/240 |
| 480k | 0.094092295 | 0.306281 | 133/240 | 86/240 |

MSE改善与机械指标并非单调一致，最终模型始终按validation MSE选择，未按此表中的FEM表现挑选。逐周期停止判定见run/plateau_history.json；完整原始曲线、固定seed图、逐样本FEM值及失败状态均保留。

本轮按用户明确确认的480k上限停止；最后完整周期的validation改善仍为0.395%，严格平台期计数为0，未满足更严格的收敛停止条件，不能声称已经充分收敛。训练与最终评估已结束，不再自动延长。
