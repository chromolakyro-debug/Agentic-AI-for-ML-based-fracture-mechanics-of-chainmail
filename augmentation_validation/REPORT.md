# 旋转与镜像 augmentation 力学标签验证

结论：160组中160/160组满足Ex/Ey映射的预定混合判据；原图重算与metadata 160/160组一致。主验证320次图像FEM（每次含x/y两种加载），求解失败0次。固定精度复核另做20次图像FEM。当前Ex/Ey标签可按本报告规则用于on-the-fly augmentation。

每类无放回随机20张，共160张；seed=2026111601，抽样和变换使用两个独立PCG64随机流。抽样覆盖现有全部split，仅做验证，不迁移集合。每张均匀抽取k∈{0,1,2,3}，独立抽取50% flip，先np.rot90(k)，再np.flip(axis=1)。未为凑齐变换重新抽样；恒等变换也实际重复求解。样本及变换在求解前冻结于samples.json/config.json。

图像列随物理x增大，图像行向下；mechanics.py用flipud将材料赋到y向上的网格。np.rot90的正角为物理逆时针，axis=1 flip为物理左右镜像。输入输出都是224×224 uint8、严格0/1、连续内存，未反色、插值或重编码源PNG；每对相位像素数和体积分数严格一致。

适用物理前提：40×40×2mm正方形试件，hx=hy=40/224mm；每像素一个全积分Q4 plane-stress单元，两相各向同性linear elasticity，Esoft=1、Ehard=10、相同nu=0.3。x拉伸左右边规定ux=0/0.04mm、横向自由，上下traction-free；y拉伸交换方向。Ex=Fx/(Ly·thickness·strain)，Ey=Fy/(Lx·thickness·strain)，都在固定全局坐标定义。旋转/镜像只是网格与材料的置换；没有周期边界配对。原点单个横向pin只消除刚体平移，换角点不改变应变能或模量；反向加载经刚体平移/符号变换等价。未发现破坏该映射的方向性材料或非对称变形约束，未改原力学问题。

原图与增强图均从各自像素重新装配刚度矩阵并实际求解，没有用交换输出代替增强FEM。生产设置为float64、AMG-CG、tol=1e-9、maxiter=1200；复用网格/预条件器，不复用材料矩阵或解。

求解前固定：abs(actual−expected)≤1e-8+1e-7·abs(expected)，atol以当前Esoft=1的模量单位计；相对误差分母max(abs(expected),1e-12·Esoft)。当前两相实心且同nu，模量能量界为[1,10]，尺度下限不实际触发。依据为既有AMG/direct回归阈值1e-7，以及既有224分辨率对照误差1.89e-09；残差容差本身不等于模量误差。本轮每类预先固定1对样本以tol=1e-11复核，并对01、05类各1对做独立spsolve直接解。没有事后放宽容差。

| CCW旋转 | 左右翻转 | 样本数 | 映射通过 | 最大相对误差（Ex/Ey合并） |
|---|---|---:|---:|---:|
| 0° | 否 | 18 | 18 | 0.000e+00 |
| 0° | 是 | 20 | 20 | 6.868e-08 |
| 90° | 否 | 14 | 14 | 3.821e-08 |
| 90° | 是 | 24 | 24 | 1.904e-08 |
| 180° | 否 | 15 | 15 | 3.313e-08 |
| 180° | 是 | 22 | 22 | 4.796e-08 |
| 270° | 否 | 32 | 32 | 6.029e-08 |
| 270° | 是 | 15 | 15 | 3.567e-08 |

| 量 | 绝对误差 median / P90 / max | 相对误差 median / P90 / max |
|---|---|---|
| Ex | 9.089e-09 / 9.511e-08 / 3.225e-07 | 2.104e-09 / 2.079e-08 / 6.868e-08 |
| Ey | 1.262e-08 / 8.948e-08 / 1.939e-07 | 2.269e-09 / 1.776e-08 / 6.029e-08 |

原图重算对metadata最大相对误差：Ex 2.024e-10、Ey 1.811e-09。按类统计见summary.json；失败样本完整保存在failures.json（当前0条），全部原始求解诊断在三个*_fem.jsonl中。

精度复核：10对中10对映射通过，生产解对高精度解最大相对变化2.830e-08；高精度映射最大相对误差1.943e-10。这支持误差来自迭代/浮点计算而非标签映射错误。此次没有超差后修复或重跑，也未细化网格：所验映射是同一正方形Q4网格的对称置换，不是对连续体绝对模量离散精度的证明。

0°/180°和任意左右flip下，Ex/Ey保持；90°/270°和任意左右flip下，Ex/Ey交换，均以本轮抽样和预定容差为限。可在training样本读取时调用augmentation.py的random_augmentation，先对0/1图及未标准化Ex/Ey变换，再执行原有条件转换和标准化；不直接置换已标准化条件。本次只验证Ex/Ey，不默认Poisson ratio、剪切量、方向性场、姿态metadata等也遵循同一规则；未修改CFM训练代码或启动训练。

局限：160张抽样并非全量7857张穷举；使用全部split进行的本次物理不变性验证不构成训练或模型选择。该结论限于当前正方形、等间距、各向同性、对称加载协议，不能直接推广至矩形区域、方向性材料、其他边界条件或任意角度旋转。没有发现当前Ex/Ey实现的未解决问题。

主验证实测wall time 118.6s（8 CPU workers），单图含两向加载的中位求解时间2.83s；高精度迭代9.1s、直接解14.0s。三阶段wall time之和141.6s；后两阶段并行，此和不是端到端用时。所有源数据、metadata、split和现有solver/loader文件hash未变。只保留变换配方和一张对照图，没有永久增强训练集副本。

复现（项目根目录、现有环境）：

```powershell
.\cfm_v0\.venv\Scripts\python.exe augmentation_validation\run.py prepare  # 首次冻结；已有结果时禁止重复prepare
.\cfm_v0\.venv\Scripts\python.exe augmentation_validation\run.py main
.\cfm_v0\.venv\Scripts\python.exe augmentation_validation\run.py precision
.\cfm_v0\.venv\Scripts\python.exe augmentation_validation\run.py direct
.\cfm_v0\.venv\Scripts\python.exe augmentation_validation\analyze.py
```

main/precision/direct支持按配置hash、像素hash与原图/增强图独立key断点续算；已有失败记录不会被静默覆盖。要独立从零重跑，应复制脚本到另一个项目同级实验目录，保留本次冻结结果。
