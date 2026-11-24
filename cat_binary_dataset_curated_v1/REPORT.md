# 数据集整理报告

完整原始8,000张、64个requested_style细类已逐一核验。按用户指定的精确标签映射，51个细类合并为八组、保留7,857张；13个细类整体排除143张。没有未知标签、数量差异、单图例外或视觉筛选。原数据和既有实验结果/checkpoint保留。

| 编号 | 新类别 | 数量 | Train | Validation | Test |
|---|---|---:|---:|---:|---:|
| 01 | 写实点画 / realistic_stippling | 750 | 600 | 75 | 75 |
| 02 | 写实笔绘 / realistic_ink_drawing | 1,500 | 1,200 | 150 | 150 |
| 03 | 几何三角 / triangular_geometry | 750 | 600 | 75 | 75 |
| 04 | 几何破碎 / fragmented_geometry | 750 | 600 | 75 | 75 |
| 05 | 像素线描 / pixel_line_art | 759 | 607 | 76 | 76 |
| 06 | 像素实心 / solid_pixel_art | 766 | 613 | 77 | 76 |
| 07 | 卡通 / cartoon | 1,500 | 1,200 | 150 | 150 |
| 08 | 剪影 / silhouette | 1,082 | 865 | 108 | 109 |
| 合计 | | 7,857 | 6,285 | 786 | 786 |

图片位于`images/01_realistic_stippling/`至`images/08_silhouette/`。每类按原始数值ID升序，从`cat_000001.png`连续编号；全局唯一`new_id`使用`01_cat_000001`形式。NPY、NPZ、CSV、JSONL和split已按同一新索引同步重排。

所有保留图片逐文件SHA256与原图一致，原PNG格式、尺寸、像素和黑白相位未改。按原始ID与pixel hash建立关联，再逐样本核对原CSV和冻结CFM数据的Ex/Ey及s/a；所有力学数值原样复制，没有重新FEM。

原metadata全部124个字段与原值保存在`metadata.jsonl`的`original_metadata`中，原CSV还以原字节完整存入`provenance/original_dataset/metadata.csv`。有效`metadata.csv`只改写原`filename`为新images目录下路径，其余原字段不变，另增加新旧ID、路径、类别与split追溯字段。完整删除记录在`provenance/deleted_samples.*`；8,000条操作映射在`provenance/all_sample_actions.csv`。

根据用户对冲突条款的明确答复，采用按八类重新分层80/10/10，seed=20261116；整数取整后为6,285/786/786。三个新集合互斥且覆盖全部保留样本。相对旧split，2,653张保留样本发生集合迁移；因此不宣称满足已被替代的“只过滤、不迁移”条款。旧split原件及其过滤后版本均保留在provenance，且不作为有效split。

原CFM mean/std、物理定义和数值保持不变。mean/std来自原6400张training set，不是按新split拟合；旧checkpoint也已见过部分新validation/test，因此新split不能直接当作旧模型的全新holdout。旧sampling weights与support统计完整归档，并按原ID保存保留样本的历史值，没有把缺失旧权重补造给新train，也未重新计算采样权重或support。

验收通过：逐图字节/像素一致、原metadata无损、标签对应正确、新ID与路径有效、删除类不在有效索引/split中、原文件保留。实际调用现有`cfm_v0.train.setup(data_dir=..., run_dir=...)`读取全部7,857行，检查八类和split，并构造16张CPU输入batch；默认读取原8,000张仍通过。唯一既有代码改动是为该loader增加可选路径参数，默认行为不变；修改前代码已归档。

实测源核验约14.5秒，整理约33.1秒，完整读回验收约38.6秒。未重新生成图片、未重新计算力学性质、未训练模型。八类对照图见`eight_classes.png`，每类4张固定随机样本；完整检查记录为`validation.json`与`loader_validation.json`。
