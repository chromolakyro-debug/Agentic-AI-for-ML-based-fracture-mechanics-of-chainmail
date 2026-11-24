# CFM v1 实际训练与评估

训练至280,000 updates，累计active训练/validation/checkpoint时间15.460小时，8,960,000次样本呈现，等效1425.62 epochs（有放回采样）。停止原因`convergence_criterion`；最佳EMA在280,000，validation velocity MSE=0.096081788。最终评估按best_val_mse选择，没有按FEM换模型。

沿用11,529,633参数Conditional U-Net、s/a conditioning、Gaussian linear flow matching、AdamW、EMA、gradient clip及batch32/bfloat16。GPU RTX4090，实际每步吞吐/显存完整保存在training.jsonl。新数据7,857张，固定train/validation/test=6285/786/786；逐图关联和整类映射检查通过。随机初始化新模型。z-score mean/std仅用新训练集重算；原数据不改。oversampling沿用原图s/a的8×12训练集分箱、逆平方根频数与均匀混合、权重截断后归一，不做类别均衡，也不对增强后的分布二次加权。

每次先抽原图，再等概率CCW旋转0/90/180/270度，独立50%左右翻转；严格先旋转后翻转。奇数90度先交换物理Ex/Ey，再计算`s=0.5[log(Ex/Esoft)+log(Ey/Esoft)]`、`a=log(Ex/Ey)`及z-score。不插值、不反色、不生成永久副本。validation不增强，786张固定顺序、noise/time，每5k用EMA评估。MSE是像素均值`(v_theta-(x-epsilon))²`，不是FEM误差。

LR每20k updates一个cosine周期，首峰2e-4、200-step warmup，后续峰1e-4、谷3e-5、周期不变；实际restart 13次，仅改变LR。80k后完整周期比较，连续2次改善<1%启动评估，连续3次<0.5%后还必须再过一个完整restart周期；有效改善重置停止计数。最近四个窗口：220k: 0.369%、240k: 0.226%、260k: 0.351%、280k: 0.200%。用户在训练中授权延长预算；200k时仍改善0.634%，故上限由200k/12小时调整为300k/18小时。仅更新预算字段，其余恢复状态逐项完全一致，证据在budget_extension_applied.json；不是重新初始化训练。

阶段A在160k启动。所有评估复用既有24 targets、10 seeds、midpoint 64步采样、阈值0二值化，真实原FEM求解；该panel已多次使用，不能称为新holdout。validation与FEM分开记录。

| checkpoint | validation MSE | mean(d_mech) | s/a联合命中 | Ex/Ey ±5%联合命中 |
| --- | --- | --- | --- | --- |
| 160,000 | 0.098274 | 0.3684 | 109/240 | 72/240 |
| 180,000 | 0.097811 | 0.3449 | 119/240 | 67/240 |
| 200,000 | 0.097191 | 0.3368 | 120/240 | 81/240 |
| 220,000 | 0.096832 | 0.3353 | 119/240 | 79/240 |
| 240,000 | 0.096613 | 0.3307 | 120/240 | 89/240 |
| 260,000 | 0.096275 | 0.3386 | 125/240 | 85/240 |
| 280,000 | 0.096082 | 0.3204 | 120/240 | 80/240 |

`d_mech=sqrt((Ex_FEM-Ex_target)^2+(Ey_FEM-Ey_target)^2)`；s/a命中阈值固定为`|Δs|≤log(1.05)`且`|Δa|≤log(1.10)`。最终FEM有效240/240，失败0计入分母。|Δs| median/P90=0.0324/0.0914，|Δa|=0.0652/0.1452。

旧125k同panel的mean(d)=0.3783、s/a命中97/240；本轮为0.3204、120/240。新旧数据、划分、训练量和增强均不同，不能当作augmentation单因素消融。新训练集机械retrieval top1的mean(d)=0.0954，与随机生成分开比较；页面第三行query经过FEM best-of-10筛选，不能替代随机生成成绩。

极端各向异性`|a_target|>0.8`命中7/40，中部`|a_target|<0.3`命中73/120。各格训练覆盖、误差和逐seed扫描斜率见training_target_bin_coverage.json、final_scan_slopes.json；理想响应斜率1、交叉斜率0，仅作描述性诊断，不声称统计独立或物理可达性证明。

固定中部a扫描s，平均响应斜率为1.020/1.028，逐seed交叉斜率绝对值均值为0.029/0.023，总体刚度跟随较好。固定s扫描a，响应斜率范围0.686–1.007，最低/最高s两列存在各向异性响应不足。极端a格T01/T02分别只有5/14张训练图可经轴交换落入该格，覆盖稀疏与困难区域重合，但尚不能证明因果。

固定seed全24目标的图像已逐milestone查看，猫形与失败例并存，局部轮廓、风格会随checkpoint改变；validation下降没有对应所有条件下单调视觉改善。极端条件反复出现拉长/圆钝，几何和像素结构存在触边、局部残缺。完整观察记录见run/visual_observations.md；这里只做限定样本的实际视觉检查，没有宣称全部图人工合格。多样性与memorization留给完整HTML及训练图人工对照，机械retrieval不是视觉近邻，不能仅凭像素不相同证明创新。

现有证据确认训练后期收益减小、误差及视觉缺陷依赖条件；覆盖不足、conditioning难度与二值化各自贡献尚未被隔离验证，不把猜测当结论。本轮未扩数据、加参数或引入surrogate。下一步宜先人工审阅失败条件和训练覆盖，再决定单一受控改动。

完整结果页面：`evaluation/runs/cfm_v1_step_280000/index.html`；训练曲线training_curves.png。独立审计已重算训练集归一化/权重、逐step LR、validation选择、plateau状态、FEM输入hash及距离/样本方差；两组共480个retrieval位置另用brute-force oracle核验train-only、来源去重和轴交换，HTML内嵌数据与结果表一致。用户曾要求10k留存，后改20k；150k/160k/170k保留，180k起每20k完整快照，均有SHA256。恢复命令、实际命令、配置版本和权重路径见README.md、run/commands.jsonl、run/checkpoint_inventory.json。审计包不含完整训练图或模型权重，包含原始loss/LR、各milestone固定图与逐样本FEM、最终HTML和exact FEM binary arrays。
