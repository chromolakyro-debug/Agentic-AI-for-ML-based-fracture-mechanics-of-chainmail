# Cat-shaped Mechanical Metamaterials：CFM continuation 审阅入口

本包用于向外部 AI 讨论下一步；它是实测证据包，不是模型分发包。请先读 `cfm_continuation/REPORT.md`，再看 `training_curves.png`、按 step 排序的 `run/development_*/fixed_samples.png`、`controllability_comparison.png` 和 `generated_examples.png`。不要把本说明中的实验规程当作结果；实际训练规模和结论只以报告及机器记录为准。

项目用 8,000 张 224×224 binary cat material maps 训练 Conditional Flow Matching。0 是黑色硬固体，1 是白色软固体，两相均不是孔洞；Esoft=1、Ehard=10。条件为 `s=0.5[log(Ex/Esoft)+log(Ey/Esoft)]` 与 `a=log(Ex/Ey)`。给定 Gaussian noise，经固定 midpoint 64-step ODE 和阈值 0 得到 binary image，再调用真实 224×224 Q4 plane-stress FEM。这里 Ex/Ey 是横向自由、有限试样边界条件下的 apparent moduli，不是 periodic homogenization。

本轮只对同一个 11,529,633 参数 Conditional U-Net 续训，从原 20k 完整 checkpoint 恢复权重、optimizer、EMA 与 RNG。没有扩数据、增加参数、改 FEM 或引入 surrogate。原代码和结果保留，本轮代码、配置、输出位于 `cfm_continuation/`。

建议重点审阅：

1. Warm restart、sample count、optimizer/EMA 恢复、validation 固定 noise/time 与停止判据是否严谨？是否触发安全上限而非收敛？
2. 同 targets/seeds 的全体 development 图中，猫头身、姿态、碎片化和截断是否随训练改善？AI 评级是主观观察，不能当作 human ground truth。
3. s/a 的主响应、cross slope 与 held-attribute range 是否同时改善？平均 cross slope 小不能排除非单调漂移。注意 dense/sparse 条件及正负 a 的差异。
4. 视觉改善是否以力学准确性或 diversity 为代价？近邻图、pixel Hamming、normalized shape RMS 与 strict predicate 给出了哪些 memorization 证据，又有哪些局限？
5. 与 training-only retrieval 相比，CFM 提供的价值是否足够？请区分数据覆盖、训练预算、conditioning 和二值化等问题中的已证实证据与猜测，优先讨论现有方法的下一步验证。

数据和选择纪律：split 固定为 6400/800/800；条件标准化、采样权重和 support density 只用 train；support threshold 由原 validation 的 k=20 距离分位数固定。Development 是本轮生成前冻结的 15 targets×2 seeds。Final 复用 v0 的 20 targets×32 seeds，是已使用过的历史 benchmark，不是全新未接触 holdout；只在本轮 checkpoint 选择结束后生成。每个 target 共用同一组 seeds。命中阈值固定 `|Δs|≤log(1.05)`、`|Δa|≤log(1.10)`，FEM 失败计入未命中。图例里明确标出的 FEM-selected 示例不能替代普通随机生成的整体统计。

主要文件：

| 路径 | 用途 |
|---|---|
| `cfm_continuation/REPORT.md` | 中文实测结论、限制和下一步 |
| `cfm_continuation/run/complete_training_history.csv` | 原训练与续训的完整已记录历史；training 为100-step窗口均值，validation 为完整800样本 |
| `cfm_continuation/run/training.jsonl` | 本轮原始训练/validation/LR/sample count/显存/速度/时间日志 |
| `cfm_continuation/run/development_*/` | milestone 全部固定30张 binary arrays、contact sheet、FEM、逐样本值、实际视觉观察 |
| `cfm_continuation/run/final/` | 640张 final binary arrays、manifest、FEM diagnostics、metrics、逐样本 CSV/JSON、40图视觉观察、NN diagnostics |
| `cfm_v0/run/final/` | 原20k同640个 target/seed 的比较基准 |
| `cfm_continuation/run/comparison.json` | 原20k与本轮的配对力学、density、diversity 对照 |
| `cfm_continuation/run/protocol.json` | 冻结 targets、seeds、support、采样与二值化规程 |
| `cfm_continuation/run/data.npz` / `splits.json` | 全8000行属性、split、train-only统计与权重；不含训练图像 |
| `context/dataset_properties_8000.csv` | 易读属性/风格/原始seed/材料相/拆分表；64-bit seed必须按整数或字符串读取 |
| `cfm_continuation/run/checkpoint_inventory.json` | 所有本轮权重的实际保存路径、step、schema、大小与 SHA256 |
| `cfm_continuation/run/commands.jsonl` / `setup_commands.json` | 实际命令、时间、退出状态；失败启动也保留 |
| `cfm_continuation/CODE_CHANGES.md` / `REPRODUCE.md` | 关键代码/配置差异和复现说明 |

本包不含完整8,000张 training images、训练环境、巨大模型权重或可再生 descriptor cache。因此仅凭 ZIP 不能重新训练或生成新样本；需要按 hash 找回本地原始数据和 checkpoint。可以直接核对现有统计、binary identity，以及在安装 FEM 依赖后重新求解已提供的 binary outputs。

`context/DATASET_README.md` 是数据发布时的历史说明，其中“没有训练模型/没有划分”的句子描述的是当时状态；当前CFM split与训练状态以本包 protocol、config、日志和报告为准。本轮未使用test subset做训练、checkpoint选择或调参。

完整性检查：`python VERIFY_ARCHIVE.py cfm_continuation_ai_review.zip`。解压后数值复核：`python cfm_continuation/verify_results.py --run cfm_continuation/run`（需要 NumPy/SciPy，不需要 GPU/weights）。这些检查验证文件与存储结果，不会自动提供人工猫形确认，也不证明新结构的物理可制造性。
