# 可复现 Evaluation Pipeline

本目录只评估，不启动训练。全部新代码、配置、cache及输出位于evaluation/。旧数据/metadata/splits/checkpoints只读。Python沿用项目已有环境：`cfm_v0/.venv/Scripts/python.exe`。

正式24格已根据用户补充的红圈图冻结：R1/R6选C3/C4，R2/R5选C2–C5，R3/R4选C1–C6。原图保存在`configs/user_selected_cells.png`。精确bin edges来自`analysis/curated_augmented_distribution/grid_v1_equal_width/s_a/counts.json`；不采用后来保留作展示的5×5网格。row 1在最下方，column 1在最左侧。`configs/targets.json`记录精确中点，不从截图估计坐标。程序拒绝把诊断targets用作正式评估。

## 一次运行及断点恢复

在项目根目录运行以下命令。所有步骤默认校验缓存；中断后运行同一命令。正式target被冻结后，一次完整流程产生24×10张图。

```powershell
.\cfm_v0\.venv\Scripts\python.exe evaluation/run.py --config evaluation/configs/old_20k.json
.\cfm_v0\.venv\Scripts\python.exe evaluation/run.py --config evaluation/configs/old_125k.json
```

缺失24格时可独立验证4个以前已有的development targets；它们覆盖正负a及不同s，仅用于模块检查：

```powershell
.\cfm_v0\.venv\Scripts\python.exe evaluation/run.py --config evaluation/configs/old_20k.json --diagnostic
.\cfm_v0\.venv\Scripts\python.exe evaluation/run.py --config evaluation/configs/old_125k.json --diagnostic
```

阶段入口：`--stage audit|target_retrieval|generate|fem|fem_retrieval|statistics|report|all`，默认all。报告和统计阶段读取已存在的前置结果。`--retry-failed`显式重试FEM失败，旧失败cache另存带时间戳文件。常规运行保留失败，不静默排除；整体命中率分母包含全部seeds。

## 冻结用户指定targets

将24个选中格子按需要的target顺序写成JSON数组，例如`[[row,col],...]`，保存到`evaluation/configs/selected_cells.json`。这不是随分布自动选点。该文件必须来自用户指定的红圈，不能用示例任意24点替代。

```powershell
.\cfm_v0\.venv\Scripts\python.exe evaluation/targets.py --cells evaluation/configs/selected_cells.json
```

程序验证24个不同合法格子，精确使用bin中点，保存target_id、row、col、s、a、Ex、Ey与grid/source hash。冻结后拒绝覆盖，后续checkpoint共享该文件与seeds.json。`Ex=Esoft*exp(s+a/2)`、`Ey=Esoft*exp(s-a/2)`。模量沿用原FEM数值单位，Esoft=1，没有凭空指定MPa。

## 数据与模型迁移

配置中的路径相对项目根目录，`--root`可指定新根目录。旧20k与125k EMA均绑定旧8,000张数据和同一个6,400/800/800划分；125k是之前最终选择的best.pt，120k才是best-validation.pt。不会用新数据划分为旧模型检索。

`adapters/dataset.py`通过metadata ID、NPZ sample_ids（若存在）、逐图像素hash、NPY和标签一致性建立关联。旧NPZ无ID时，完整验证原filename的数值ID=row、每行PNG=NPY、metadata模量=NPZ，禁止未经核对按行号关联。新整理数据已提供`configs/curated_dataset_adapter.json`，可独立读取；新checkpoint必须声明其真正训练数据/划分的hash，不能替换数据路径后继续假用旧checkpoint。

新模型适配器接口：`Adapter(model_config,sampling_config,dataset,root)`；提供`info`、`initial_noise(seeds)`及`generate(target,seeds)->(raw_float32_NHW,standardized_condition)`。额外conditioning写入model.extra_conditioning；旧CFM adapter拒绝额外风格条件。当前pipeline记录s/a标准化值；如新模型采用不同条件结构，需扩展该记录字段，不得隐式遗漏。新FEM通过`initializer(config)`和`solve((key,binary))`适配；必须返回实际求解结果或明确失败。

## 设置与cache

64步midpoint，CUDA bfloat16网络、float32积分状态，batch10；每个seed单独创建torch CUDA Generator，同checkpoint不同target与同架构不同checkpoint共享同一初始噪声，实际噪声数组与hash保存。没有resize、crop、augmentation或反相。raw>=0得到soft=1，hard=0；224×224正方形。FEM沿用原Q4 plane-stress有限试件，两方向边界正向拉伸、横向自由，非周期均匀化，tol=1e-9。

距离为`d(q,p)=sqrt(((Ex_q-Ex_p)/1)^2+((Ey_q-Ey_p)/1)^2)`，尺度固定。对于retrieval，候选原轴序及交换轴序取较小距离；对生成误差，始终用其固定全局坐标下的真实Ex/Ey，不允许通过交换来降低模型误差。同source只占一名额；严格同距优先原轴，排序并列按dataset_index稳定处理。旧训练未使用augmentation，但本轮遵照要求采用轴交换可用的retrieval基线。

target retrieval cache包含训练集内容/索引/来源ID、所有PNG内容hash、targets、距离及代码hash，不含checkpoint。因此两模型共享；必要输入改变会重建。生成缓存验证checkpoint、模型/采样代码、软件、noise、target、预处理和NPZ内容。FEM cache验证实际二值像素、形状、dtype、FEM配置及solver代码，允许真正相同材料场跨checkpoint复用。cache内容自身也带校验和。run_id已使用后若输入配置变化则要求新run_id，避免混写。

报告保留raw float32与实际binary arrays，每张PNG与FEM像素逐一比对；人工页面直接嵌入results.json的数值。检索图逐字节复制原图，保留样本类别、原文件路径及原始/匹配轴序。所有统计与连线均使用未扰动的真实FEM值，无jitter。页面旋转镜像仅是显示功能。

## 验证及审计

```powershell
.\cfm_v0\.venv\Scripts\python.exe evaluation/tests_core.py
.\cfm_v0\.venv\Scripts\python.exe evaluation/validate_physics.py
.\cfm_v0\.venv\Scripts\python.exe evaluation/verify_report.py evaluation/runs/old_20k_diagnostic
```

核心测试覆盖轴交换、train范围、来源去重、固定尺度、cache失效及损坏、精确bin中点、ddof=1与失败分母。真实FEM集成检查重新求解一张高各向异性训练图及其90°旋转/左右镜像，比较metadata并故意送入非法相位检查失败处理。该检查不替代生成样本FEM。

`runs/<run_id>/index.html`可以离线直接打开。审计时提供REPORT.md、manifest/config/targets/seeds/grid/model_info、results/summary/retrieval表、figures、HTML与其images/retrieval_images，以及code_hashes和code_snapshot即可。需要重算FEM时追加arrays（包含raw及binary）和initial_noise.npy；不必提供整个训练集或模型权重。模型权重路径及SHA256在manifest。人工评级通过页面导出JSON，未评级不能宣称已人工确认。
