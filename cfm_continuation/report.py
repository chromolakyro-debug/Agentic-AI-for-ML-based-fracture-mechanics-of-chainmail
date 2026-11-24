"""Render concise Chinese report from measured files and explicit final observations."""
import paths
import numpy as np
from common import OUT,read_json


def main():
    s=read_json(OUT/'training_summary.json');c=read_json(OUT/'continuation_config.json');comparison=read_json(OUT/'comparison.json')
    old=comparison['old'];new=comparison['new'];notes=read_json(OUT/'conclusions.json');selected=read_json(OUT/'checkpoint_selection.json')
    history=read_json(OUT/'complete_training_history.json');val=[r for r in history if r['phase']=='continuation' and 'validation_velocity_mse' in r]
    visual=read_json(OUT/'final'/'visual_review.json');oldvisual=read_json(paths.ORIGINAL/'final'/'visual_inspection.json')
    generation=read_json(OUT/'final'/'manifest.json');fem=read_json(OUT/'final'/'fem_summary.json')
    oldcat=sum(r['category']=='recognizable_cat' for r in oldvisual['samples']);newcat=visual['counts'].get('2',0)
    loss0=val[0]['validation_velocity_mse'];best=s['best_validation_loss'];last=val[-1]['validation_velocity_mse']
    percent=lambda x:f'{x*100:.2f}%'
    lines=['# CFM v0 continuation 实测报告','',
      f"本轮从原 20,000-step `last.pt` 恢复 model、AdamW moments、EMA 与 RNG，累计训练到 {s['steps']:,} steps。本轮 trainer 实际耗时 {s['continuation_elapsed_seconds']/3600:.3f} 小时，累计处理 {s['images_seen']:,} 个训练样本（{s['equivalent_epochs']:.1f} 等效 epochs；加权有放回抽样，不等于每张图均匀访问该次数）。停止原因：{s['stop_explanation'].rstrip('。')}。",'',
      f"原 8,000 张数据、6,400/800/800 split、11,529,633 参数网络、FiLM conditioning、224×224 分辨率和 FEM solver 均保持不变。无 augmentation。材料定义为 0=黑色硬固体、1=白色软固体，`s=0.5[log(Ex/Esoft)+log(Ey/Esoft)]`，`a=log(Ex/Ey)`；所有力学结果来自阈值 0 后严格二值图的真实 FEM。",'',
      f"RTX 4090 上 batch 16/32/64 的实测吞吐分别为 {c['benchmarks'][0]['images_per_second']:.1f}/{c['benchmarks'][1]['images_per_second']:.1f}/{c['benchmarks'][2].get('images_per_second',float('nan')):.1f} images/s，peak allocated 显存分别为 {c['benchmarks'][0]['peak_allocated_gib']:.2f}/{c['benchmarks'][1]['peak_allocated_gib']:.2f}/{c['benchmarks'][2].get('peak_allocated_gib',float('nan')):.2f} GiB。选择 batch {c['batch']}；batch64 的可用 device memory 为 0，吞吐反而下降。本轮训练峰值 allocated/reserved 为 {s['peak_allocated_gib']:.2f}/{s['peak_reserved_gib']:.2f} GiB，含validation/save的平均吞吐约 {(s['images_seen']-320000)/s['continuation_elapsed_seconds']:.1f} images/s。约123k后瞬时吞吐降至56 images/s，原因未查明，环境快照与完整速度日志已保留；没有更改batch或训练配置。",'',
      "LR 在 20k→22k 从 `2e-5` 线性升至 `1e-4`，随后按独立 continuation schedule 缓慢 cosine 下降，计划在 200k 达到 `3e-5`。保留原 optimizer state，没有重启原始 20k cosine。" + ('本轮另外进行过一次有记录的温和 LR restart，见 `run/additional_lr_restart.json`。' if s['additional_lr_restart'] else '除这次预定 warm restart 外，未发生额外 LR restart。')+f"最终 LR 为 `{s['last_lr']:.6g}`。",'',
      f"固定 800 张 validation、固定 noise/time、固定 validation batch16，每 5k 计算 EMA velocity MSE。20k 重新计算值 {loss0:.7f}，best {best:.7f}（{s['best_step']:,} steps），last {last:.7f}。跨进程 BF16/cuDNN 的 20k 重算差异约 2e-6，已记录；远小于 0.5% 平台判据。平台状态：{notes['plateau']}。",'',
      f"最终评估选用 {selected['step']:,}-step {selected['selected']}；last 和 best-validation 权重分别保留，路径、大小与 SHA256 见 `run/checkpoint_inventory.json`。{notes['selection']}。",'',
      '## 视觉与力学结果','',
      f"Development panel 为 15 个有经验支持的 targets × 2 个固定 seeds，每个 milestone 的全部 30 张图均展示并进行 FEM。最终视觉检查使用 20 个 targets 的前两个固定 seeds，共 40 张，未经视觉或 FEM 筛选。20k 原记录中基本连贯猫形为 {oldcat}/40，本轮 AI 主观评级为 {newcat}/40；同时满足该视觉标准和两个力学容差的样本从 {comparison['joint_visual_mechanical']['old_coherent_and_property_hit']}/40 变为 {comparison['joint_visual_mechanical']['new_coherent_and_property_hit']}/40。该数字不是 human ground truth，也不能代表全部 640 张的视觉合格率。{notes['visual']}。",'',
      f"Final panel 复用上一轮固定 20 targets × 32 noise seeds，属于已使用过的历史配对 benchmark，不是全新 holdout。固定命中阈值仍为 `|Δs|≤log(1.05)` 与 `|Δa|≤log(1.10)`。本轮 {new['valid']}/{new['n']} FEM 有效，失败率 {percent(new['failure_rate'])}；失败计入未命中，误差分位数仅对有效结果计算。Final生成耗时 {generation['invocation_wall_seconds']:.1f} 秒，8 workers FEM耗时 {fem['invocation_wall_seconds']:.1f} 秒；另完成7个development checkpoints共210张图的FEM。",'',
      '| 指标 | 20k baseline | Continuation |','|---|---:|---:|',
      f"| 联合命中率 | {percent(old['hit_joint'])} | {percent(new['hit_joint'])} |",
      f"| `abs(Δs)` median / P90 | {old['abs_delta_s']['median']:.4f} / {old['abs_delta_s']['p90']:.4f} | {new['abs_delta_s']['median']:.4f} / {new['abs_delta_s']['p90']:.4f} |",
      f"| `abs(Δa)` median / P90 | {old['abs_delta_a']['median']:.4f} / {old['abs_delta_a']['p90']:.4f} | {new['abs_delta_a']['median']:.4f} / {new['abs_delta_a']['p90']:.4f} |",
      f"| s / a bias | {old['bias_s']:+.4f} / {old['bias_a']:+.4f} | {new['bias_s']:+.4f} / {new['bias_a']:+.4f} |",'',
      f"同 target/seed 的联合命中率提高 {comparison['paired_hit_change']['mean']*100:.2f} 个百分点，按 32 个 seed clusters bootstrap 的 95% 区间为 [{comparison['paired_hit_change']['seed_bootstrap_95ci'][0]*100:.2f}, {comparison['paired_hit_change']['seed_bootstrap_95ci'][1]*100:.2f}] 个百分点；不将相关的 640 个结果视为完全独立样本。",'',
      '| 扫描 | 主响应斜率 20k→本轮 | 交叉斜率 20k→本轮 | held 属性极差 median 20k→本轮 |','|---|---:|---:|---:|']
    for scan in comparison['scans']:
        x,y=scan['old'],scan['new'];held='s' if scan['varying']=='a' else 'a'
        lines.append(f"| 固定 {held}={scan['fixed']:.3f}，扫描 {scan['varying']} | {x['main_slope_mean']:.3f}→{y['main_slope_mean']:.3f} | {x['cross_slope_mean']:+.3f}→{y['cross_slope_mean']:+.3f} | {x['held_range_median']:.3f}→{y['held_range_median']:.3f} |")
    lines += ['',notes['control']+'。平均 cross slope 接近零不能排除非单调漂移，应同时读 held range 与逐 seed 曲线。','',
              '| 区域（train-only support density） | 目标数 / 样本数 | 联合命中率 20k→本轮 |','|---|---:|---:|']
    for x,y in zip(comparison['density_old'],comparison['density_new']):lines.append(f"| {y['density']} | {y['targets']} / {y['n']} | {percent(x['hit_joint'])}→{percent(y['hit_joint'])} |")
    nn=comparison['neighbors_new'];nn0=comparison['neighbors_old']
    lines += ['',notes['regions']+'。Support 是 train-standardized s/a 空间的 20-NN 经验指标，threshold 仅由 validation 固定；不是物理可达性证明。','',
      f"Training-only retrieval 仍在 20/20 个目标找到满足两个容差的已有图像，`|Δs|` median={new['retrieval_abs_delta_s']['median']:.4f}、`|Δa|` median={new['retrieval_abs_delta_a']['median']:.4f}。{notes['retrieval']}。",'',
      f"生成图对全部 6,400 张 training images 的最近像素 Hamming 距离 median 从 {nn0['nearest_hamming_median']:.4f} 变为 {nn['nearest_hamming_median']:.4f}，normalized shape RMS median 从 {nn0['nearest_shape_rms_median']:.4f} 变为 {nn['nearest_shape_rms_median']:.4f}。本轮 exact pixel duplicates={nn['exact_pixel_duplicates']}，既有 strict visual predicate matches={nn['strict_visual_matches']}。{notes['diversity']}。像素不同不能单独证明有意义的新结构；风格相关和有无 memorization 也不能只靠一个自动指标判断。",'',
      '## 证据边界与下一步','',notes['evidence']+'。', '',notes['next_step']+'。', '',
      '训练曲线见 `training_curves.png`；各 milestone 全部固定样本见 `run/development_*/fixed_samples.png`；最终属性对照与示例见 `controllability_comparison.png`、`generated_examples.png`。逐样本结果、FEM diagnostics、binary arrays、配置、splits、seeds、完整日志和实际命令都保留。审计 ZIP 不含完整 training images 或模型权重；可独立复核存储结果和重新 FEM，但完整续训需另取原数据与权重。']
    (paths.HERE/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(paths.HERE/'REPORT.md')

if __name__=='__main__':main()
