"""Paired historic benchmark comparison and threshold/neighbor diagnostics."""
import paths
import numpy as np
from common import OUT,read_json,save_json,S_TOL,A_TOL


def main():
    old=read_json(paths.ORIGINAL/'final'/'metrics.json');new=read_json(OUT/'final'/'metrics.json')
    a=read_json(paths.ORIGINAL/'final'/'samples.json');b=read_json(OUT/'final'/'samples.json')
    prior={r['key']:r for r in a};assert set(prior)=={r['key'] for r in b}
    changes=[]
    for r in b:
        o=prior[r['key']];assert (r['s'],r['a'],r['seed'])==(o['s'],o['a'],o['seed'])
        changes.append(dict(key=r['key'],target_id=r['target_id'],seed=r['seed'],old_hit=o['hit_joint'],new_hit=r['hit_joint'],
            paired_abs_error_s_change=abs(r['delta_s'])-abs(o['delta_s']) if r['fem_valid'] and o['fem_valid'] else None,
            paired_abs_error_a_change=abs(r['delta_a'])-abs(o['delta_a']) if r['fem_valid'] and o['fem_valid'] else None))
    seeds=sorted({r['seed'] for r in b});differences=np.array([np.mean([int(r['new_hit'])-int(r['old_hit']) for r in changes if r['seed']==seed]) for seed in seeds])
    rng=np.random.default_rng(2026111511);boot=differences[rng.integers(0,len(seeds),(10000,len(seeds)))].mean(1)
    nnb=read_json(OUT/'final'/'image_neighbors.json');nna=read_json(paths.ORIGINAL/'final'/'image_neighbors.json')
    def neighbors(m):
        r=m['results'];d=m['diversity'];shape=[v['shape_pooled_rms'] for v in r if v['descriptor_valid']]
        return dict(exact_pixel_duplicates=sum(v['exact_pixel_duplicate'] for v in r),strict_visual_matches=sum(bool(v.get('strict_visual_matches_training_indices')) for v in r),
            nearest_hamming_median=float(np.median([v['pixel_hamming_fraction'] for v in r])),nearest_hamming_min=float(min(v['pixel_hamming_fraction'] for v in r)),
            nearest_shape_rms_median=float(np.median(shape)) if shape else None,
            within_target_pairwise_hamming_median_range=[min(v['pairwise_hamming_median'] for v in d),max(v['pairwise_hamming_median'] for v in d)],
            within_target_shape_rms_median_range=[min(v['pairwise_normalized_shape_rms_median'] for v in d if v['pairwise_normalized_shape_rms_median'] is not None),max(v['pairwise_normalized_shape_rms_median'] for v in d if v['pairwise_normalized_shape_rms_median'] is not None)],
            unique_count_per_target=[v['unique_binary_hashes'] for v in d])
    margin={}
    for key in ('fraction_near_threshold_0p1','fraction_near_threshold_0p25'):
        zero=[r for r in b if r[key]==0]
        margin[key]=dict(zero_near_threshold_samples=len(zero),joint_misses_with_no_near_threshold_pixels=sum(not r['hit_joint'] for r in zero),
                        caveat='Only a bound on changing this same continuous output by small threshold shifts; not proof that binary topology or training representation never matters.')
    scans=[]
    for x,y in zip(old['scans'],new['scans']):
        assert x['group']==y['group']
        scans.append(dict(group=x['group'],varying=x['varying'],fixed=x['fixed_value'],
            old={k:x.get(k) for k in ('main_slope_mean','cross_slope_mean','held_range_median')},
            new={k:y.get(k) for k in ('main_slope_mean','cross_slope_mean','held_range_median')}))
    old_visual=read_json(paths.ORIGINAL/'final'/'visual_inspection.json')['samples']
    new_visual=read_json(OUT/'final'/'visual_review.json')['samples']
    assert {r['key'] for r in old_visual}=={r['key'] for r in new_visual}
    current={r['key']:r for r in b}
    joint_visual=dict(n=len(new_visual),old_coherent=sum(r['category']=='recognizable_cat' for r in old_visual),
        new_coherent=sum(r['rating']==2 for r in new_visual),
        old_coherent_and_property_hit=sum(r['category']=='recognizable_cat' and prior[r['key']]['hit_joint'] for r in old_visual),
        new_coherent_and_property_hit=sum(r['rating']==2 and current[r['key']]['hit_joint'] for r in new_visual),
        scope='Same fixed first two seeds at each final target (40); subjective AI visual classification, not human truth or all-640 visual rate.')
    save_json(OUT/'comparison.json',dict(old=old['overall'],new=new['overall'],density_old=old['density'],density_new=new['density'],scans=scans,
        paired_hit_change=dict(mean=float(differences.mean()),seed_bootstrap_95ci=np.quantile(boot,[.025,.975]).tolist(),
            gained_hits=sum(not r['old_hit'] and r['new_hit'] for r in changes),lost_hits=sum(r['old_hit'] and not r['new_hit'] for r in changes),
            method='10,000 bootstrap resamples of 32 noise-seed clusters; all 20 paired target outcomes remain together within a seed.'),
        neighbors_old=neighbors(nna),neighbors_new=neighbors(nnb),threshold_diagnostics=margin,joint_visual_mechanical=joint_visual,
        caveat='Historical panel already used in v0; this is paired regression comparison, not a fresh unseen holdout.'))
    save_json(OUT/'paired_sample_changes.json',changes)
    print({'old_hit':old['overall']['hit_joint'],'new_hit':new['overall']['hit_joint'],'paired_ci':np.quantile(boot,[.025,.975]).tolist()})

if __name__=='__main__':main()
