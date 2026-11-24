"""Append targeted, FEM-labelled cats without changing the original 1,000 maps.

Stages: prepare-pilot -> select-pilot -> visual review -> approve-pilot;
prepare-full -> select-full -> visual review -> export -> validate.
Visual approval is bound to exact binary pixel hashes, never inferred from labels.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import io
import json
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace

import numpy as np
from PIL import Image

import candidate_generation as gen
import targeted_candidates as targeted
import augmentation_selection as augselect
from evaluate_mechanics import label_items, mechanics_fingerprint
from release_workflow import contact_sheet, phase_connectivity

REFERENCE_AUGMENTATION_REVIEW = {}
REFERENCE_BASE_PIXELS_SHA256 = "fe0d506f26f5fbf0ca75dc27e121adc1721c6bbba1422677220a7262b80f7d43"
REFERENCE_BASE_LABELS_SHA256 = "685c4e7b79fc717106e1281b6e6aa466b8c41a1cb2a9c392649dac801d51b71b"


def sample_key(sample, stage):
    stage=sample.diagnostics.get('source_stage',stage)
    return f"{stage}:{sample.spec.seed}:{sample.diagnostics.get('variant_id', 0)}"


def pixel_sha(pixels):
    return hashlib.sha256(np.asarray(pixels, dtype=np.uint8).tobytes()).hexdigest()


def original_labels_sha(rows):
    identities=[[r['filename'],r['sha256_pixels'],str(r['seed']),float(r['Ex_app']),float(r['Ey_app'])]
                for r in rows[:1000]]
    return hashlib.sha256(json.dumps(identities,separators=(',',':')).encode()).hexdigest()


def fixed_identity(samples,labels):
    identity=[]
    for sample,label in zip(samples,labels):
        pixels=sample['pixels'] if isinstance(sample,dict) else sample.pixels
        identity.append([pixel_sha(pixels),float(label['Ex_app']),float(label['Ey_app'])])
    return hashlib.sha256(json.dumps(identity,separators=(',',':')).encode()).hexdigest()


def load_base(base):
    samples, labels, rows = augselect.load_base(base)
    pixels = np.load(Path(base)/"dataset.npy", allow_pickle=False)
    if pixels.shape != (1000,224,224) or pixel_sha(pixels) != REFERENCE_BASE_PIXELS_SHA256:
        raise ValueError("This augmentation requires the original 1000-map v6 dataset.")
    if original_labels_sha(rows)!=REFERENCE_BASE_LABELS_SHA256:
        raise ValueError("Original mechanical labels or identity changed.")
    return samples, labels, rows, pixels


def review(args):
    path = args.work_dir/"augmentation_visual_review.json"
    result = json.loads(path.read_text(encoding="utf-8")) if path.exists() else dict(REFERENCE_AUGMENTATION_REVIEW)
    return set(result.get("passed_hashes", [])), result.get("excluded_hashes", {})


def prepare(args, stage):
    load_base(args.base)
    if stage in ("full","styles"):
        pilot = json.loads((args.work_dir/"pilot_selection.json").read_text(encoding="utf-8"))
        if pilot.get("visual_review") != "pass" or pilot.get("selected_count") != 50:
            raise ValueError("A fully reviewed 50-image pilot is required before full generation.")
    if stage=='styles':
        import style_candidates as engine
    else:
        engine=targeted
    sources = engine.acquire(stage, args.sources, args.work_dir, seed=args.seed, batch_size=args.batch_size)
    variants = []
    for source in sources:
        variants.extend(engine.variants_for_sample(source, count=3))
    # Store the exact complete proposed pool before screening, including rejected sources.
    gen.cache_batch(args.work_dir/f"{stage}_variants.npz", variants)
    valid = [s for s in variants if not engine.rejection(s)]
    keys = [sample_key(s, stage) for s in valid]
    if len(set(keys)) != len(keys):
        raise ValueError("Variant keys are not unique.")
    labels = label_items(list(zip(keys, [s.pixels for s in valid])),
                        args.work_dir/f"{stage}_labels.jsonl", args.workers)
    by_key = dict(zip(keys, labels))
    manifest = dict(stage=stage, source_count=len(sources), variant_count=len(variants),
                    automatic_valid_variants=len(valid),
                    automatic_valid_sources=len({s.spec.seed for s in valid}),
                    rejection_counts=dict(Counter(engine.rejection(s) for s in variants if engine.rejection(s))),
                    seed=args.seed, batch_size=args.batch_size, mechanics_protocol=mechanics_fingerprint(),
                    labelled_variants={k:by_key[k]["sha256_pixels"] for k in keys})
    (args.work_dir/f"{stage}_prepared.json").write_text(json.dumps(manifest,indent=2), encoding="utf-8")
    print(json.dumps({k:v for k,v in manifest.items() if k != "labelled_variants"}), flush=True)


def prepared_pool(args, stage):
    samples = gen.read_batch(args.work_dir/f"{stage}_variants.npz")
    labels_by_key = {}
    protocol = mechanics_fingerprint()
    for line in (args.work_dir/f"{stage}_labels.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row.get("protocol_fingerprint") == protocol:
            labels_by_key[(row["key"],row["sha256_pixels"])] = row
    if stage=='styles':
        import style_candidates as engine
    else:
        engine=targeted
    samples = [s for s in samples if not engine.rejection(s)]
    labels = [labels_by_key[(sample_key(s,stage),pixel_sha(s.pixels))] for s in samples]
    if stage=='full' and (args.work_dir/'styles_prepared.json').exists():
        extra,extra_labels=prepared_pool(args,'styles')
        samples+=extra;labels+=extra_labels
    return samples, labels


def selected_pool(args, stage, require_review=True):
    manifest = json.loads((args.work_dir/f"{stage}_selection.json").read_text(encoding="utf-8"))
    samples, labels = prepared_pool(args, stage)
    by_key = {sample_key(s,stage):(s,l) for s,l in zip(samples,labels)}
    selected = [by_key[k] for k in manifest["selected_keys"]]
    if [pixel_sha(s.pixels) for s,l in selected] != manifest["selected_hashes"]:
        raise ValueError("Selected pixels changed.")
    passed, excluded = review(args)
    if require_review and any(pixel_sha(s.pixels) not in passed or pixel_sha(s.pixels) in excluded for s,l in selected):
        raise ValueError("Not all selected pixels have passed AI visual review.")
    return [s for s,l in selected], [l for s,l in selected], manifest


def select_stage(args, stage):
    fixed, fixed_labels, base_rows, base_pixels = load_base(args.base)
    if stage == "full":
        pilot_samples, pilot_labels, pilot_manifest = selected_pool(args,"pilot")
        if pilot_manifest.get("visual_review") != "pass":
            raise ValueError("Pilot gate not complete.")
        fixed = list(fixed)+pilot_samples
        fixed_labels = list(fixed_labels)+pilot_labels
    candidates, candidate_labels = prepared_pool(args,stage)
    passed, excluded = review(args)
    eligible = [(s,l) for s,l in zip(candidates,candidate_labels) if pixel_sha(s.pixels) not in excluded]
    candidates, candidate_labels = [s for s,l in eligible], [l for s,l in eligible]
    target = 50 if stage == "pilot" else args.additions-50
    if stage=='full' and args.style_additions:
        style_ids=[i for i,s in enumerate(candidates) if s.diagnostics.get('source_stage')=='styles']
        core_ids=[i for i,s in enumerate(candidates) if s.diagnostics.get('source_stage')!='styles']
        if not style_ids:raise ValueError('The requested style cohort must be prepared before full selection.')
        style_selected=[];ss={};sp={}
        families=('binary_illustration','pixel_art','graphic_portrait')
        for family_number,family in enumerate(families):
            quota=args.style_additions//3+int(family_number<args.style_additions%3)
            ids=[i for i in style_ids if candidates[i].diagnostics.get('style_family')==family]
            si,fstats,fpairs=augselect.select_additions(list(fixed)+[candidates[i] for i in style_selected],
                list(fixed_labels)+[candidate_labels[i] for i in style_selected],
                [candidates[i] for i in ids],[candidate_labels[i] for i in ids],target=quota,seed=args.seed+family_number)
            if len(si)<quota:
                raise ValueError(f'Only {len(si)}/{quota} distinct acceptable {family} sources; generate more style candidates.')
            style_selected.extend(ids[i] for i in si);ss[family]=fstats;sp[family]=fpairs
        ci,cs,cp=augselect.select_additions(list(fixed)+[candidates[i] for i in style_selected],
                                          list(fixed_labels)+[candidate_labels[i] for i in style_selected],
                                          [candidates[i] for i in core_ids],[candidate_labels[i] for i in core_ids],
                                          target=target-len(style_selected),seed=args.seed)
        indices=style_selected+[core_ids[i] for i in ci]
        stats=dict(style_selection=ss,core_selection=cs,selected_count=len(indices),
                   target=target,style_count=len(style_selected),target_met=len(indices)==target and len(style_selected)==args.style_additions)
        pair_metrics=dict(style_audit=sp,core_audit=cp)
    else:
        indices, stats, pair_metrics = augselect.select_additions(fixed,fixed_labels,candidates,candidate_labels,
                                                                 target=target,seed=args.seed)
    chosen = [candidates[i] for i in indices]
    labels = [candidate_labels[i] for i in indices]
    hashes = [pixel_sha(s.pixels) for s in chosen]
    if len(chosen) < target:
        print(f"Only {len(chosen)}/{target} valid additions available; no padding.", flush=True)
    captions = [f"{sample_key(s,stage)}\nB={s.black_fraction:.1%} Ex={l['Ex_app']:.2f} Ey={l['Ey_app']:.2f}"
                for s,l in zip(chosen,labels)]
    # One replaceable 5x10 pilot sheet. Full review sheets stay only in scratch.
    if stage == "pilot":
        contact_sheet(chosen,args.work_dir/"pilot_contact_sheet.png",columns=10,tile=224,labels=captions)
    else:
        for start in range(0,len(chosen),50):
            contact_sheet(chosen[start:start+50],args.work_dir/f"full_review_{start//50:02d}.png",
                          columns=10,tile=224,labels=captions[start:start+50])
    result = dict(stage=stage,selected_count=len(chosen),target=target,fixed_base_identity=fixed_identity(fixed,fixed_labels),
                  selected_keys=[sample_key(s,stage) for s in chosen],selected_hashes=hashes,
                  selected_seeds=[int(s.spec.seed) for s in chosen],
                  visual_review="pass" if all(h in passed for h in hashes) and len(chosen)==target else "pending",
                  technical_validation="pass",statistics=stats,pair_metrics=pair_metrics,
                  selected=[dict(key=sample_key(s,stage),sha256_pixels=pixel_sha(s.pixels),
                                 candidate=int(s.spec.candidate),seed=int(s.spec.seed),
                                 variant=s.diagnostics.get("variant_id",0),Ex=l["Ex_app"],Ey=l["Ey_app"],
                                 black_fraction=s.black_fraction,group=s.diagnostics.get("target_group"),
                                 style=s.spec.style) for s,l in zip(chosen,labels)])
    (args.work_dir/f"{stage}_selection.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps({k:result[k] for k in ("stage","selected_count","target","visual_review")}), flush=True)


def approve_pilot(args):
    samples,labels,result=selected_pool(args,"pilot")
    if len(samples)!=50:
        raise ValueError("Pilot must contain 50 selected distinct specimens.")
    result["visual_review"]="pass"
    (args.work_dir/"pilot_selection.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print("Pilot: 50/50 binary, FEM, duplicate and exact-pixel AI review checks passed.",flush=True)


def new_row(sample, lab, stage, index, args):
    actual_stage=sample.diagnostics.get('source_stage',stage)
    row=dict(filename=f"cat_{index:04d}.png",method="independent targeted local diffusion; binary preprocessing; per-source FEM-guided framing search; joint-coverage selection",
             source=gen.SOURCE,pipeline_version=7,candidate_id=sample.spec.candidate,seed=sample.spec.seed,
             generation_master_seed=sample.diagnostics.get('generation_master_seed',args.seed),generation_batch_size=args.batch_size,
             requested_geometry_family=sample.spec.geometry_family,requested_pose=sample.spec.pose,
             requested_view=sample.spec.view,requested_morphology=sample.spec.morphology,requested_tail=sample.spec.tail,
             requested_style=sample.spec.style,requested_orientation=sample.spec.orientation,
             requested_composition=sample.spec.composition,extent_mode=sample.spec.extent_mode,
             background_value=sample.spec.background,target_foreground_fraction=sample.spec.target_fraction,
             prompt=sample.spec.prompt,black_pixels=int((sample.pixels==0).sum()),black_fraction=sample.black_fraction,
             Ex_app=lab["Ex_app"],Ey_app=lab["Ey_app"],Ex_over_Es=lab["Ex_app"],Ey_over_Es=lab["Ey_app"],
             Ex_over_Ey=lab["Ex_app"]/lab["Ey_app"],E_soft=1.,E_hard=10.,nu_soft=.3,nu_hard=.3,
             specimen_width_mm=40.,specimen_height_mm=40.,specimen_thickness_mm=2.,nominal_strain=.001,
             mechanical_protocol="finite_plane_stress_lateral_free_tension_Q4_224",
             solver_protocol_sha256=lab["protocol_fingerprint"],siglip_cat_logit=sample.cat_logit,
             siglip_cat_margin=sample.cat_margin,cat_bbox_aspect=sample.geometry[0],cat_bbox_max_side_px=sample.geometry[1],
             cat_bbox_center_x=sample.geometry[2],cat_bbox_center_y=sample.geometry[3],
             sha256_pixels=pixel_sha(sample.pixels),validation="pass",ai_visual_review="pass",
             generation_stage=f"augmentation_{actual_stage}",style_family=sample.diagnostics.get('style_family','targeted_anatomy'),target_group=sample.diagnostics.get("target_group",""),
             target_Ex=sample.diagnostics.get("target_Ex",""),target_Ey=sample.diagnostics.get("target_Ey",""),
             source_id=sample.diagnostics.get("source_id",sample.spec.seed),
             variant_id=sample.diagnostics.get("variant_id",0),candidate_variant_key=sample_key(sample,stage))
    for key in ('pixel_art_grid','pixel_block_side','semantic_protocol','semantic_evaluation'):
        row[key]=sample.diagnostics.get(key,'')
    for key in ("residual_x","residual_y","energy_relative_error_x","energy_relative_error_y",
                "force_balance_relative_error_x","force_balance_relative_error_y","fem_valid"):
        row[key]=lab[key]
    for key in ("hard_edge_left","hard_edge_right","hard_edge_top","hard_edge_bottom",
                "foreground_boundary_contact","foreground_fraction","forced_border_pixels"):
        row[key]=sample.diagnostics[key]
    row.update(phase_connectivity(sample.pixels))
    return row


def export(args):
    base_samples,base_labels,rows,base_pixels=load_base(args.base)
    pilot,plabels,pmanifest=selected_pool(args,"pilot")
    full,flabels,fmanifest=selected_pool(args,"full")
    added=pilot+full
    labs=plabels+flabels
    if fmanifest.get('fixed_base_identity')!=fixed_identity(list(base_samples)+pilot,list(base_labels)+plabels):
        raise ValueError('The pilot or base changed since full selection.')
    if len(added)!=args.additions or len({s.spec.seed for s in added})!=len(added):
        raise ValueError("Selected addition count or independent-source contract failed.")
    audit_indices,audit_stats,final_audit=augselect.select_additions(base_samples,base_labels,added,labs,
                                                                  target=len(added),seed=args.seed)
    if set(audit_indices)!=set(range(len(added))):
        raise ValueError('Final joint addition set failed near-duplicate or source audit.')
    (args.work_dir/'final_pair_audit.json').write_text(json.dumps(final_audit,indent=2),encoding='utf-8')
    combined=np.concatenate([base_pixels,np.stack([s.pixels for s in added])])
    n=len(combined)
    args.output.mkdir(parents=True,exist_ok=True)
    image_dir=args.output/"images"
    image_dir.mkdir(exist_ok=True)
    expected={f"cat_{i:04d}.png" for i in range(n)}
    if any(p.name not in expected for p in image_dir.glob("*.png")):
        raise ValueError("Unexpected PNGs in release; use a fresh output directory.")
    for i in range(1000):
        shutil.copy2(args.base/"images"/rows[i]["filename"],image_dir/rows[i]["filename"])
        rows[i]["generation_stage"]="original_v6"
    for j,(sample,lab) in enumerate(zip(added,labs)):
        gen.save_png(sample.pixels,image_dir/f"cat_{1000+j:04d}.png")
        rows.append(new_row(sample,lab,"pilot" if j<len(pilot) else "full",1000+j,args))
    np.save(args.output/"dataset.npy",combined,allow_pickle=False)
    bins=[(min(5,int((float(r['Ex_app'])-1)/1.5)),min(5,int((float(r['Ey_app'])-1)/1.5))) for r in rows]
    counts=Counter(bins)
    preview_ids=np.random.default_rng(args.seed+991).choice(n,100,replace=False).tolist()
    preview_cells={idx:cell for cell,idx in enumerate(preview_ids)}
    for i,(row,cell) in enumerate(zip(rows,bins)):
        row.update(property_bin_x=cell[0],property_bin_y=cell[1],property_bin_count=counts[cell],
                   balanced_sampling_weight=n/(len(counts)*counts[cell]),preview_cell=preview_cells.get(i,""))
    fields=list(dict.fromkeys(key for row in rows for key in row))
    with (args.output/"metadata.csv").open("w",encoding="utf-8",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    contact_sheet([SimpleNamespace(pixels=combined[i]) for i in preview_ids],args.output/"preview.png",
                  labels=[f"cat_{i:04d} B={float(rows[i]['black_fraction']):.1%}\nEx={float(rows[i]['Ex_app']):.2f} Ey={float(rows[i]['Ey_app']):.2f}" for i in preview_ids])
    (args.work_dir/"release_manifest.json").write_text(json.dumps(dict(
        original_count=1000,added_count=len(added),total_count=n,original_pixels_sha256=pixel_sha(base_pixels),
        dataset_pixels_sha256=pixel_sha(combined),added_hashes=[pixel_sha(s.pixels) for s in added],
        preview_indices=preview_ids,pilot=pmanifest,full=fmanifest),indent=2),encoding="utf-8")
    write_report(args,rows,pmanifest,fmanifest)
    validate(args)


def write_report(args,rows,pilot,full):
    values=np.array([[float(r['Ex_app']),float(r['Ey_app'])] for r in rows])
    fractions=np.array([float(r['black_fraction']) for r in rows])
    coarse=np.linspace(1,10,7)
    table=np.histogram2d(values[:,0],values[:,1],bins=[coarse,coarse])[0].T.astype(int)
    baseline=np.histogram2d(values[:1000,0],values[:1000,1],bins=[coarse,coarse])[0].T.astype(int)
    fine=np.arange(1,10.01,.5)
    beforefine=np.histogram2d(values[:1000,0],values[:1000,1],bins=[fine,fine])[0]
    afterfine=np.histogram2d(values[:,0],values[:,1],bins=[fine,fine])[0]
    new_rows=rows[1000:]
    style_counts=Counter(r['style_family'] for r in new_rows)
    style_mask=np.array([r['style_family']!='targeted_anatomy' for r in new_rows])
    style_values=values[1000:][style_mask]
    style_fine=np.histogram2d(style_values[:,0],style_values[:,1],bins=[fine,fine])[0]
    novel_style_cells=int(np.count_nonzero((style_fine>0)&(beforefine==0)))
    target_errors=np.array([max(abs(float(r['Ex_app'])-float(r['target_Ex'])),
                               abs(float(r['Ey_app'])-float(r['target_Ey']))) for r in new_rows])
    audit=json.loads((args.work_dir/'final_pair_audit.json').read_text(encoding='utf-8'))
    support=table>0
    cv_before=float(baseline[support].std()/baseline[support].mean())
    cv_after=float(table[support].std()/table[support].mean())
    sparse=sorted((int(table[iy,ix]),ix,iy) for iy,ix in zip(*np.nonzero((table>0)&(table<=15))))
    sparse_text='; '.join(f'Ex[{coarse[ix]:g},{coarse[ix+1]:g}), Ey[{coarse[iy]:g},{coarse[iy+1]:g}): {n}'
                         for n,ix,iy in sparse[:4])
    def summary(v):
        counts=np.histogram2d(v[:,0],v[:,1],bins=[coarse,coarse])[0]
        p=counts[counts>0]/len(v)
        return f"Ex {v[:,0].min():.4f}–{v[:,0].max():.4f}; Ey {v[:,1].min():.4f}–{v[:,1].max():.4f}; occupied 6x6 cells {np.count_nonzero(counts)}; effective occupied cells {np.exp(-sum(p*np.log(p))):.2f}; Ex>=8.5 {(v[:,0]>=8.5).sum()}, Ey>=8.5 {(v[:,1]>=8.5).sum()}, both>=8.5 {((v[:,0]>=8.5)&(v[:,1]>=8.5)).sum()}"
    text=f"""# Targeted binary cat dataset quality report

**{len(rows)} valid specimens = original 1000 + {len(rows)-1000} additions.** Original filenames, pixel maps and Ex/Ey labels are preserved. No training, train/test split or conditional-model evaluation has been performed.

All PNGs are one-bit 224x224 and reload identically to dataset.npy. Array shape=({len(rows)},224,224), dtype=uint8, values={{0,1}}; no blank maps. 0=black hard solid; 1=white soft solid. Unique pixel hashes and diffusion seeds are verified. Current hard-fraction range {fractions.min():.2%}–{fractions.max():.2%}; high-end targeted candidates may exceed the previous95% cutoff, up to99% subject to visual review.

**Mechanical labels:** same independent40x40x2mm specimen, Esoft=1,Ehard=10,nu=.3 in both phases, Q4 plane stress and lateral-free0.1% tensile tests. Every new image has two FEM load cases solved using AMG-preconditioned CG, residual/energy/reaction checks and fixed-fraction Reuss–Voigt bounds. These are protocol-dependent apparent moduli, not periodic homogenized constants.

**Before:** {summary(values[:1000])}.

**After:** {summary(values)}.

Occupied0.5x0.5 modulus cells: {np.count_nonzero(beforefine)} -> {np.count_nonzero(afterfine)}. This counts observed data cells, not the entire feasible set. No unobserved area is claimed achievable or impossible merely from these counts.

Coarse-bin count CV on the same final observed support: {cv_before:.3f} -> {cv_after:.3f} (lower is more even, but bin-dependent). Gains are concentrated in the high-modulus tail. Remaining sparse observed regions: {sparse_text}. Anisotropic coverage is still incomplete; a zero count is not a proof of physical impossibility.

Final6x6 counts, rows increasing Ey, columns increasing Ex; edges1,2.5,4,5.5,7,8.5,10:
```
{chr(10).join(' '.join(f'{x:4d}' for x in row) for row in table)}
```

**Generation and selection:** independent new local Z-Image-Turbo sources, targeted morphology and drawing-style strata, followed by up to three parameterized framing candidates per source. All valid variants receive FEM; at most one final map is retained per source. Selection targets missing mechanical neighborhoods and geometric novelty, rather than treating variants as independent training examples. The 50-image targeted pilot passed before the full stage. Final added images have exact-pixel AI visual review; no human inspection is claimed.

Added style strata: {', '.join(f'{k}={v}' for k,v in sorted(style_counts.items()))}. User references inspired binary illustration with internal anatomy, pixel art, and graphic cat portraits; their pixels were not copied. Pixel-art samples have verified constant2x2 or4x4 blocks. Portraits intentionally include the head, chest and front paws. Styles are requested strata with visual screening, not a claim of perfectly separable categories; some portraits also appear pixelated or contain fine fur details.

The {int(style_mask.sum())} new-style maps occupy {novel_style_cells} previously unobserved0.5x0.5 modulus cells relative to the original1000. Their demonstrated contribution is different visual/internal geometry within observed mechanical regions; style diversity alone has not established a wider mechanical domain at this resolution.

Actual target accuracy: {int((target_errors<=.25).sum())}/{len(new_rows)} additions satisfy both requested moduli within0.25 Esoft units; median L-infinity target error {np.median(target_errors):.3f}. Target coordinates guide exploration, not guaranteed inverse design.

**Duplicates:** exact duplicate count0. Audited {audit['audited_new_base_pairs']} new-to-original and {audit['audited_new_new_pairs']} new-to-new pairs; {audit['local_near_duplicate_pairs_involving_new']} local near-duplicate conflicts remain. Checks combine same phase polarity, black-fraction difference<=.025, max(|deltaEx|,|deltaEy|)<=.12, and reflection-aware silhouette IoU>=.91 or IoU>=.85 with pHash distance<=6. Source variants are mutually exclusive. Different geometry at similar performance is retained. These thresholds do not guarantee perceptual dissimilarity of every pair.

**Limitations:** fixed mechanical targets are generation/search intentions; actual achieved Ex/Ey labels must be used. Some targeted regions remain sparse or unattained. Modulus support is correlated; it is not uniform across[1,10]^2. Anatomical prompts may be missed. Very high hard fractions can reduce visible soft-cat size or require outline drawings. No experimental,3D,nonlinear,stress-concentration or manufacturing validation is claimed. Small solver residuals do not certify continuum discretization error. No assertion of sufficient data for flow matching has been made.
"""
    refinement=args.work_dir/"new_refinement.json"
    if refinement.exists():
        result=json.loads(refinement.read_text(encoding="utf-8"))
        text+=f"\nPixel-preserving refinement: {result['count']} new specimens checked at224 and448 elements/side; maximum observed modulus change {result['maximum_relative_change']:.3%}. Sampled evidence only, not a universal error bound.\n"
    reproduction=args.work_dir/"reproduction_check.json"
    if reproduction.exists():
        result=json.loads(reproduction.read_text(encoding="utf-8"))
        text+=f"\nSame-workstation regeneration check: {result.get('checked_sources')} sources; exact binary agreement={result.get('all_exact')}. Cross-platform determinism is not guaranteed.\n"
    style_reproduction=args.work_dir/'style_reproduction_check.json'
    if style_reproduction.exists():
        result=json.loads(style_reproduction.read_text(encoding='utf-8'))
        text+=f"\nNew-style regeneration check: {result.get('checked_sources')} sources; exact binary agreement={result.get('all_exact')}.\n"
    (args.output/"quality_report.md").write_text(text,encoding="utf-8")


def validate(args):
    output=args.output
    array=np.load(output/"dataset.npy",allow_pickle=False)
    with (output/"metadata.csv").open(newline="",encoding="utf-8-sig") as f:
        rows=list(csv.DictReader(f))
    n=len(rows)
    if array.shape!=(n,224,224) or array.dtype!=np.uint8 or set(np.unique(array))!={0,1} or n<1000:
        raise ValueError("Dataset contract failed.")
    names=[f"cat_{i:04d}.png" for i in range(n)]
    if sorted(p.name for p in (output/"images").glob("*.png"))!=names or [r['filename'] for r in rows]!=names:
        raise ValueError("Image/metadata ordering or count failed.")
    if pixel_sha(array[:1000])!=REFERENCE_BASE_PIXELS_SHA256:
        raise ValueError("Original 1000 pixel maps changed.")
    if original_labels_sha(rows)!=REFERENCE_BASE_LABELS_SHA256:
        raise ValueError("Original 1000 mechanical labels or identities changed.")
    protocol=mechanics_fingerprint()
    counts=Counter((int(r['property_bin_x']),int(r['property_bin_y'])) for r in rows)
    hashes=[]
    for i,r in enumerate(rows):
        pixels=gen.load_png(output/"images"/r['filename'])
        if not np.array_equal(pixels,array[i]):raise ValueError("PNG/NumPy mismatch")
        digest=pixel_sha(pixels);hashes.append(digest)
        f=float((pixels==0).mean());x=float(r['Ex_app']);y=float(r['Ey_app'])
        if digest!=r['sha256_pixels'] or set(np.unique(pixels))!={0,1}:raise ValueError("Pixel hash/blank failure")
        if abs(f-float(r['black_fraction']))>1e-12 or int((pixels==0).sum())!=int(r['black_pixels']):raise ValueError("Fraction failure")
        if not .05<=f<=.99:raise ValueError("Recognizability fraction range failure")
        if r.get('pixel_art_grid'):
            grid=int(r['pixel_art_grid'])
            if grid:
                if grid not in (56,112):raise ValueError('Unknown pixel art grid')
                block=224//grid
                if not np.array_equal(pixels,np.repeat(np.repeat(pixels[::block,::block],block,0),block,1)):
                    raise ValueError('Pixel art block structure failed')
        lo=1/(1-.9*f);hi=1+9*f
        if min(x,y)<lo-1e-6 or max(x,y)>hi+1e-6:raise ValueError("FEM necessary bounds failed")
        if r['solver_protocol_sha256']!=protocol:raise ValueError("FEM implementation/protocol mismatch")
        params=dict(E_soft=1,E_hard=10,nu_soft=.3,nu_hard=.3,specimen_width_mm=40,specimen_height_mm=40,specimen_thickness_mm=2,nominal_strain=.001)
        if any(float(r[k])!=v for k,v in params.items()):raise ValueError("Physical parameters changed")
        for axis in ('x','y'):
            if float(r[f'E{axis}_over_Es'])!=float(r[f'E{axis}_app']):raise ValueError("Condition mismatch")
            for prefix,limit in [('residual',1e-7),('energy_relative_error',1e-6),('force_balance_relative_error',1e-7)]:
                diagnostic=float(r[f'{prefix}_{axis}'])
                if not np.isfinite(diagnostic) or not 0<=diagnostic<=limit:raise ValueError("FEM diagnostic failed")
        if r['fem_valid']!='True' or r['validation']!='pass' or r['ai_visual_review']!='pass':raise ValueError("Incomplete validation")
        cell=(min(5,int((x-1)/1.5)),min(5,int((y-1)/1.5)))
        if cell!=(int(r['property_bin_x']),int(r['property_bin_y'])) or int(r['property_bin_count'])!=counts[cell]:raise ValueError("Bin count mismatch")
        if not np.isclose(float(r['balanced_sampling_weight']),n/(len(counts)*counts[cell]),rtol=1e-12):raise ValueError("Weight mismatch")
    if len(set(hashes))!=n or len({r['seed'] for r in rows})!=n:raise ValueError("Duplicate map/source")
    cells=[(int(r['preview_cell']),i) for i,r in enumerate(rows) if r['preview_cell']!='']
    if sorted(c for c,i in cells)!=list(range(100)):raise ValueError("Preview membership")
    with Image.open(output/"preview.png") as preview:
        if preview.size!=(2306,2646):raise ValueError("Preview size")
        for c,i in cells:
            x=6+(c%10)*230;y=6+(c//10)*264
            if not np.array_equal(np.asarray(preview.crop((x,y,x+224,y+224)),dtype=np.uint8),array[i]):raise ValueError("Preview pixels")
    print(json.dumps(dict(valid_images=n,shape=list(array.shape),dtype=str(array.dtype),values=[0,1],
                          original_1000_preserved=True,exact_duplicates=0,preview_cells=100)),flush=True)


def extract_base(args):
    if args.output.resolve()==args.base.resolve():
        raise ValueError('Use a separate output directory when extracting the original prefix.')
    array=np.load(args.base/'dataset.npy',allow_pickle=False)[:1000]
    with (args.base/'metadata.csv').open(encoding='utf-8-sig',newline='') as f:
        rows=list(csv.DictReader(f))[:1000]
    if pixel_sha(array)!=REFERENCE_BASE_PIXELS_SHA256 or original_labels_sha(rows)!=REFERENCE_BASE_LABELS_SHA256:
        raise ValueError('Input does not contain the original 1000-specimen prefix.')
    images=args.output/'images'
    images.mkdir(parents=True,exist_ok=True)
    expected={r['filename'] for r in rows}
    if any(p.name not in expected for p in images.glob('*.png')):
        raise ValueError('Output contains images outside the original prefix; use a fresh directory.')
    for row in rows:shutil.copy2(args.base/'images'/row['filename'],images/row['filename'])
    counts=Counter((int(r['property_bin_x']),int(r['property_bin_y'])) for r in rows)
    for r in rows:
        cell=(int(r['property_bin_x']),int(r['property_bin_y']))
        r.update(property_bin_count=counts[cell],balanced_sampling_weight=1000/(len(counts)*counts[cell]),preview_cell='')
    np.save(args.output/'dataset.npy',array,allow_pickle=False)
    with (args.output/'metadata.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(json.dumps(dict(extracted=1000,output=str(args.output),original_pixels_and_labels_preserved=True)))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['extract-base','prepare-pilot','select-pilot','approve-pilot','prepare-full','prepare-styles','select-full','export','validate'])
    parser.add_argument('--work-dir',type=Path,default=Path(tempfile.gettempdir())/'binary-cat-augmentation')
    parser.add_argument('--base',type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--sources',type=int,default=100)
    parser.add_argument('--additions',type=int,default=250)
    parser.add_argument('--style-additions',type=int,default=75)
    parser.add_argument('--seed',type=int,default=1143186)
    parser.add_argument('--batch-size',type=int,default=4)
    parser.add_argument('--workers',type=int,default=8)
    args=parser.parse_args()
    args.work_dir.mkdir(parents=True,exist_ok=True)
    args.base=args.base or args.work_dir/'base'
    args.output=args.output or args.work_dir/'release'
    if args.mode=='extract-base':extract_base(args)
    elif args.mode.startswith('prepare-'):prepare(args,args.mode.split('-')[1])
    elif args.mode.startswith('select-'):select_stage(args,args.mode.split('-')[1])
    elif args.mode=='approve-pilot':approve_pilot(args)
    elif args.mode=='export':export(args)
    else:validate(args)
    return args


if __name__=='__main__':
    main()
