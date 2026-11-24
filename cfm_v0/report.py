"""Build the concise Chinese report from completed measurements and explicit visual review."""
from collections import Counter
import numpy as np
from common import HERE, OUT, read_json


def main():
    final=OUT/'final'
    train=read_json(OUT/'training_summary.json');cfg=read_json(OUT/'train_config.json')
    metrics=read_json(final/'metrics.json');overall=metrics['overall']
    manifest=read_json(final/'manifest.json');fem=read_json(final/'fem_summary.json')
    test=read_json(OUT/'test_once.json');smoke=read_json(OUT/'smoke_fem.json')
    visual=read_json(final/'visual_inspection.json');interpretation=read_json(final/'interpretation.json')
    neighbors=read_json(final/'image_neighbors.json');samples=read_json(final/'samples.json')
    sample_lookup={r['key']:r for r in samples}
    assert len(visual['samples'])==40
    assert {r['key'] for r in visual['samples']}=={r['key'] for r in samples if r['seed'] in manifest['seeds'][:2]}
    for row in visual['samples']:assert row['sha256_pixels']==sample_lookup[row['key']]['sha256_pixels']
    counts=Counter(r['category'] for r in visual['samples'])
    joint_visual=sum(r['category']=='recognizable_cat' and sample_lookup[r['key']]['hit_joint'] for r in visual['samples'])
    nn=neighbors['results'];div=neighbors['diversity'];protocol=read_json(OUT/'protocol.json')
    text=[]
    text.append('CFM v0 实测报告\n')
    text.append(interpretation['opening']+'\n')
    text.append(f"已完成原始 224×224 单通道 CFM：{cfg['gpu']}，{cfg['model_parameters']:,} 参数，batch {cfg['batch']}，"
                f"{train['steps']:,} steps / {train['images_seen']:,} training draws（{train['nominal_epochs']:.1f} 个等效 epochs），"
                f"主训练 {train['elapsed_seconds']/60:.2f} 分钟。上限预先设为 20,000 steps 或 45 分钟；实际停止原因为 {train['stop_reason']}。"
                f"仅按 validation velocity MSE 选择第 {train['best_step']:,} step 的 EMA checkpoint：validation={train['best_validation_loss']:.5f}，"
                f"训练结束后一次性 test（800 张）MSE={test['velocity_mse']:.5f}。峰值 PyTorch allocated GPU memory {train['peak_memory_gb']:.2f} GiB。\n")
    text.append('原数据 8,000 张全部通过 PNG/NumPy/CSV 行序与 pixel SHA256 核对；0=黑色硬材料，1=白色软材料，白色不是孔洞。'
                '原文件 hash 保持不变，未扩充数据。划分为 6,400/800/800，按粗粒度 s/a 分层；所有标准化、采样权重和 support density 仅来自 training set。'
                '沿用原 Q4 plane-stress、40×40×2 mm、Esoft=1/Ehard=10、横向自由拉伸 FEM。'
                f"四张 training 图复算标签的最大相对偏差为 {max(r['label_max_relative_error'] for r in smoke['real_samples']):.2e}，"
                f"另一次 full-resolution direct-solver 对照偏差为 {smoke['direct_relative_discrepancy']:.2e}。未重新求解原始全部 8,000 张。\n")
    text.append('条件为 $s=0.5[\\log(E_x/E_\\mathrm{soft})+\\log(E_y/E_\\mathrm{soft})]$、$a=\\log(E_x/E_y)$。'
                '随机初始化 conditional U-Net，通过每个 residual block 的 FiLM 注入标准化 s/a 与时间。'
                '使用 $x=2b-1$、$x_t=(1-t)\\epsilon+tx$、目标 velocity $x-\\epsilon$ 的 MSE，t 均匀采样；无 augmentation、无 surrogate。'
                'training 8×12 格计数的 inverse-square-root 权重与 uniform 各混合一半，先 cap 4 再均值归一化，最终权重 0.716–4.009。'
                '所有生成采用固定 midpoint ODE 64 steps（128 NFE）、阈值 x≥0 为软相，未作清理或几何变换。\n')
    text.append(f"Training-standardized s/a 的 k=20 kNN support threshold 为 {protocol['support']['threshold']:.6f}，取 validation 距离的第 95 分位；"
                '它只是经验支持指标，不证明物理可达。预先固定 20 个目标、5 条受支持的扫描区间，含正/负 a 与多个 s 水平。'
                's≈1.02 的 ±0.05 邻域只有 83 张 training 图，s≈1.83 则有 797 张；低刚度正 a 的部分区间不满足 support 要求，未强行补成矩形网格。'
                '目标、seeds、阈值及 hash 均在 final generation 前锁定。\n')
    text.append('![support_and_targets](support_and_targets.png)\n')
    text.append(f"Final panel 为 20×32={overall['n']} 张，所有目标共用 seeds 730000–730031；全部最终 binary maps 已实际 FEM。"
                f"FEM 有效 {overall['valid']}/{overall['n']}，失败率 {overall['failure_rate']:.1%}。"
                f"单图双方向 FEM smoke 中位 {smoke['median_solve_seconds']:.2f} 秒；final sampling 用时 {manifest['invocation_wall_seconds']/60:.2f} 分钟，"
                f"8-worker final FEM 用时 {fem['invocation_wall_seconds']/60:.2f} 分钟。命中阈值始终为 |Δs|≤log(1.05)、|Δa|≤log(1.10)。"
                f"随机生成联合命中率 {overall['hit_joint']:.1%}；|Δs| median/P90={overall['abs_delta_s']['median']:.4f}/{overall['abs_delta_s']['p90']:.4f}，"
                f"|Δa|={overall['abs_delta_a']['median']:.4f}/{overall['abs_delta_a']['p90']:.4f}，总体偏差 Δs={overall['bias_s']:.4f}、Δa={overall['bias_a']:.4f}。\n")
    text.append('下表响应斜率理想为 1，交叉斜率理想为 0；括号为配对 seed bootstrap 95% CI。误差分位数只用有效 FEM；失败仍计入未命中。\n')
    text.append('| 扫描 | CFM 响应斜率 | CFM 交叉斜率 | Retrieval 响应/交叉 |\n|---|---:|---:|---:|')
    for r in metrics['scans']:
        ci=r['cross_slope_95ci'];gainci=r['main_slope_95ci']
        text.append(f"| 固定 {r['held']}={r['fixed_value']:.3f}，扫 {r['varying']} | {r['main_slope_mean']:.3f} ({gainci[0]:.3f}, {gainci[1]:.3f}) | "
                    f"{r['cross_slope_mean']:.3f} ({ci[0]:.3f}, {ci[1]:.3f}) | {r['retrieval_main_slope']:.3f}/{r['retrieval_cross_slope']:.3f} |")
    text.append('\n'+interpretation['control']+'\n')
    text.append('![controllability](controllability.png)\n')
    text.append('联合 support 密度的 dense/medium/sparse 分界来自 training r20 的第 50/80 分位，和单独的 s 稀疏程度不同。\n')
    text.append('| 联合区域 | 目标/样本数 | 联合命中率 | Δs 绝对误差 median/P90 | Δa 绝对误差 median/P90 |\n|---|---:|---:|---:|---:|')
    for r in metrics['density']:
        text.append(f"| {r['density']} | {r['targets']}/{r['n']} | {r['hit_joint']:.1%} | {r['abs_delta_s']['median']:.4f}/{r['abs_delta_s']['p90']:.4f} | "
                    f"{r['abs_delta_a']['median']:.4f}/{r['abs_delta_a']['p90']:.4f} |")
    text.append('\n'+interpretation['density']+'\n')
    text.append(f"实际查看每个目标前两个固定 seeds，共 40 张，不按力学结果过滤："
                f"{counts['recognizable_cat']} 张判为可辨且基本连贯的猫，{counts['cat_like_but_distorted']} 张有猫特征但形态失真，"
                f"{counts['unrecognizable']} 张难以辨猫。其中“基本连贯的猫且属性联合命中”为 {joint_visual}/40。"
                '这是 assistant 对实际像素的视觉检查，不是 human confirmation，也不是 640 张的全面视觉合格率。'+visual['summary']+'\n')
    text.append('![generated_examples](generated_examples.png)\n')
    text.append(f"Retrieval 仅搜索 6,400 张 training 图，按 $\\sqrt{{(\\Delta s/\\log1.05)^2+(\\Delta a/\\log1.10)^2}}$ 选最近者，"
                f"{overall['retrieval_hit_targets']}/20 个目标命中；|Δs| median/P90={overall['retrieval_abs_delta_s']['median']:.4f}/{overall['retrieval_abs_delta_s']['p90']:.4f}，"
                f"|Δa|={overall['retrieval_abs_delta_a']['median']:.4f}/{overall['retrieval_abs_delta_a']['p90']:.4f}。"
                f"若对 CFM 做 FEM 筛选后的 best-of-32（最小上述 L2），有 {overall['targets_with_fem_selected_best_hit']}/20 个目标命中；这不代表随机生成性能。\n")
    hamming=np.array([r['pixel_hamming_fraction'] for r in nn])
    shape=np.array([r['shape_pooled_rms'] for r in nn if r['descriptor_valid']])
    strict=sum(bool(r.get('strict_visual_matches_training_indices')) for r in nn)
    text.append(f"全 training set 近邻检查：exact pixel duplicates={sum(r['exact_pixel_duplicate'] for r in nn)}/{len(nn)}，"
                f"复用既有 strict visual predicate 的匹配数={strict}。最近 training 图的像素差比例 median={np.median(hamming):.3f}，"
                f"去位置/尺度并允许反射的 shape pooled RMS median={np.median(shape):.3f}。"
                f"每目标 32 图两两像素差的 median 在 {min(r['pairwise_hamming_median'] for r in div):.3f}–{max(r['pairwise_hamming_median'] for r in div):.3f} 之间。"
                '这些仅为 memorization/diversity diagnostics，不能据此声称有意义的新结构。'+interpretation['value']+'\n')
    text.append(interpretation['diagnosis']+'\n')
    text.append('复现入口：[run_commands.ps1](run_commands.ps1)；隔离环境依赖见 [requirements.txt](requirements.txt)。'
                '中断主训练可用 `python cfm_v0/train.py train --batch 16 --steps 20000 --minutes 45 --resume`，'
                '结束后用 `python cfm_v0/finish.py` 完成固定评估（应使用本目录 .venv 中的 Python）。'
                '新复现请设置 `CFM_V0_RUN_DIR` 为新目录，避免覆盖本次输出。'
                '外部接口示例：`python cfm_v0/generate.py --ex 4 --ey 5 --count 4 --output custom_4_5`，也可用 `--s ... --a ...`；自定义目标不纳入本报告。\n')
    text.append('完整 [逐样本 CSV](run/final/samples.csv)、[20 目标汇总](run/final/target_summary.csv)、'
                '[metrics/paired slopes](run/final/metrics.json)、[binary arrays](run/final/binary.npy)、'
                '[FEM diagnostics](run/final/fem.jsonl)、[冻结协议与 seeds](run/protocol.json)、'
                '[splits](run/splits.json)、[训练配置](run/train_config.json)、[checkpoint](run/checkpoints/best.pt)、'
                '[实际评估命令和用时](run/completion_commands.json) 保留为机器可读文件。'
                '最后 4 张展示例是明确标注的 FEM-selected illustrations，整体指标始终来自全部 640 张。\n')
    output=HERE if OUT==(HERE/'run').resolve() else OUT
    (output/'REPORT.md').write_text('\n'.join(text),encoding='utf-8')
    print('Wrote',output/'REPORT.md')


if __name__=='__main__':main()
