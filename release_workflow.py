"""Mechanics-aware selection, export and reproducible validation."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import candidate_generation as gen
import mechanics_selection as selection
from evaluate_mechanics import label_items, mechanics_fingerprint

REFERENCE_REVIEW = {}  # Populated by the release builder, after AI inspection.
REFERENCE_BASELINE = {}
REFERENCE_SOLVER_CHECKS = {}


def pool_hash(samples):
    digest=hashlib.sha256()
    for sample in samples:
        digest.update(str(sample.spec.seed).encode("ascii"))
        digest.update(sample.pixels.tobytes())
    return digest.hexdigest()


def load_pool(directory):
    samples=[]
    for path in sorted(Path(directory).glob("batch_*.npz")):
        samples.extend(gen.read_batch(path))
    return samples


def contact_sheet(samples, destination, columns=10, tile=224, labels=None):
    rows=(len(samples)+columns-1)//columns
    gap,label_height=6,34
    sheet=Image.new("1",(gap+columns*(tile+gap),gap+rows*(tile+label_height+gap)),1)
    draw=ImageDraw.Draw(sheet)
    # Fixed Pillow bitmap font keeps the sheet portable and pixel reproducible.
    font=ImageFont.load_default()
    for index,sample in enumerate(samples):
        x=gap+(index%columns)*(tile+gap)
        y=gap+(index//columns)*(tile+label_height+gap)
        image=Image.fromarray(sample.pixels.astype(bool)).resize((tile,tile),Image.Resampling.NEAREST)
        sheet.paste(image,(x,y))
        draw.rectangle((x-1,y-1,x+tile,y+tile),outline=0)
        label=labels[index] if labels else f"candidate {sample.spec.candidate:04d}\nblack {sample.black_fraction:.1%}"
        draw.multiline_text((x+3,y+tile+3),label,fill=0,font=font,spacing=2)
    sheet.save(destination)


def review_for(pool, mode, work_dir):
    path=Path(work_dir)/f"{mode}_visual_review.json"
    supplied=path.exists()
    review=json.loads(path.read_text(encoding="utf-8")) if supplied else REFERENCE_REVIEW.get(mode,{})
    if review and review.get("pool_sha256") != pool_hash(pool):
        if supplied:
            raise ValueError("Visual-review manifest is not for these exact candidate pixels.")
        return {}  # A different hardware/seed run needs fresh review, not a borrowed reference verdict.
    return review


def eligible_pool(pool, review):
    exclusions={int(k):v for k,v in review.get("excluded",{}).items()}
    reviewed=set(review.get("reviewed_candidates",[]))
    rejected=Counter()
    valid=[]
    for sample in pool:
        reason=gen.rejection(sample)
        if not reason and sample.spec.candidate in exclusions:
            reason="ai_visual_review"
        if not reason and reviewed and sample.spec.candidate not in reviewed:
            reason="not_visually_reviewed"
        if reason:
            rejected[reason]+=1
        else:
            valid.append(sample)
    return valid,dict(rejected)


def get_labels(samples, work_dir, mode, workers):
    return label_items([(str(s.spec.candidate),s.pixels) for s in samples],
                       Path(work_dir)/f"{mode}_labels.jsonl",workers)


def technical_check(samples):
    hashes=[]
    for sample in samples:
        pixels=sample.pixels
        if pixels.shape!=(224,224) or pixels.dtype!=np.uint8 or set(np.unique(pixels))!={0,1}:
            raise ValueError(f"Invalid binary image {sample.spec.candidate}")
        if not .05 <= (pixels==0).mean() <= .95:
            raise ValueError("Black fraction outside the reference recognizability range")
        stream=io.BytesIO()
        gen.save_png(pixels,stream)
        stream.seek(0)
        if not np.array_equal(gen.load_png(stream),pixels):
            raise ValueError("PNG round trip changed the binary map")
        hashes.append(gen.fingerprint(pixels))
    if len(set(hashes))!=len(hashes):
        raise ValueError("Exact duplicate maps")
    if len({s.spec.seed for s in samples})!=len(samples):
        raise ValueError("Multiple outputs share a diffusion source seed")


def run_pilot(args):
    pool=gen.acquire("pilot",args.candidates or 80,args.work_dir,args.seed,args.batch_size)
    review=review_for(pool,"pilot",args.work_dir)
    valid,rejected=eligible_pool(pool,review)
    labels=get_labels(valid,args.work_dir,"pilot",args.workers)
    indices,stats,pair_metrics=selection.select(valid,labels,target=50,seed=args.seed)
    if not stats["target_met"]:
        raise RuntimeError(f"Insufficient independent pilot samples: {stats['selected_count']}/50")
    chosen=[valid[i] for i in indices]
    technical_check(chosen)
    reviewed=set(review.get("reviewed_candidates",[]))
    selected_reviewed=bool(chosen) and all(s.spec.candidate in reviewed for s in chosen)
    captions=[f"pilot {s.spec.candidate:04d} B={s.black_fraction:.1%}\nEx={labels[i]['Ex_app']:.3f} Ey={labels[i]['Ey_app']:.3f}"
              for s,i in zip(chosen,indices)]
    contact_sheet(chosen,args.work_dir/"pilot_contact_sheet.png",tile=180,labels=captions)
    result=dict(valid_images=50,candidates=len(pool),screened=len(valid),rejections=rejected,
                generation_config=dict(seed=args.seed,batch_size=args.batch_size,pipeline_version=gen.PIPELINE_VERSION),
                selected_candidates=[s.spec.candidate for s in chosen],pool_sha256=pool_hash(pool),
                statistics=stats,near_duplicates=pair_metrics,
                visual_review="AI-reviewed" if selected_reviewed else "pending",
                technical_validation="pass",fem_validation="pass",
                limitations="Observed sample coverage is not the complete physically achievable region.")
    (args.work_dir/"pilot_result.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps({key:result[key] for key in ("valid_images","candidates","screened","rejections","visual_review","technical_validation","fem_validation")}),flush=True)


def phase_connectivity(pixels):
    from scipy import ndimage
    labels,count=ndimage.label(pixels==0)  # face-connected hard elements; corner contact is not a finite-width path.
    left=set(labels[:,0])-{0};right=set(labels[:,-1])-{0}
    top=set(labels[0])-{0};bottom=set(labels[-1])-{0}
    areas=np.bincount(labels.ravel());areas[0]=0
    black=pixels==0
    return dict(hard_components_4_connected=int(count),hard_spans_x=bool(left&right),hard_spans_y=bool(top&bottom),
                hard_largest_component_share=float(areas.max()/black.sum()),
                minimum_hard_column_fraction=float(black.mean(axis=0).min()),
                minimum_hard_row_fraction=float(black.mean(axis=1).min()))


def export_release(args,pool,valid,labels,indices,stats,pair_metrics,review,rejections):
    if len(indices)!=1000 or not stats["target_met"]:
        raise RuntimeError("Insufficient independent samples for a 1000-image release")
    output=args.output
    output.mkdir(parents=True,exist_ok=True)
    images=output/"images"
    images.mkdir(exist_ok=True)
    extra=[p for p in images.glob("*.png") if p.name not in {f"cat_{i:04d}.png" for i in range(1000)}]
    if extra:
        raise ValueError("Output images folder contains unexpected PNGs; use a fresh output folder")
    chosen=[valid[i] for i in indices]
    reviewed=set(review.get("reviewed_candidates",[]))
    if any(s.spec.candidate not in reviewed for s in chosen):
        raise RuntimeError("Every selected final image needs review of its actual pixels before release")
    technical_check(chosen)
    array=np.stack([s.pixels for s in chosen])
    np.save(output/"dataset.npy",array,allow_pickle=False)
    preview_indices=np.random.default_rng(args.seed+991).choice(1000,100,replace=False).tolist()
    preview_cells={index:cell for cell,index in enumerate(preview_indices)}
    reviewed=set(review.get("reviewed_candidates",[]))
    bins=[]
    for i in indices:
        ex,ey=labels[i]["Ex_app"],labels[i]["Ey_app"]
        bins.append((min(5,max(0,int((ex-1)/1.5))),min(5,max(0,int((ey-1)/1.5)))))
    counts=Counter(bins)
    active=len(counts)
    rows=[]
    for index,(sample,label_index,cell) in enumerate(zip(chosen,indices,bins)):
        name=f"cat_{index:04d}.png"
        gen.save_png(sample.pixels,images/name)
        if not np.array_equal(gen.load_png(images/name),array[index]):
            raise ValueError("Exported image disagrees with dataset")
        lab=labels[label_index]
        row=dict(filename=name,method="independent local diffusion; binary threshold; isotropic framing; finite Q4 FEM; joint-property selection",
                 source=gen.SOURCE,pipeline_version=gen.PIPELINE_VERSION,candidate_id=sample.spec.candidate,
                 seed=sample.spec.seed,generation_master_seed=args.seed,generation_batch_size=args.batch_size,
                 requested_geometry_family=sample.spec.geometry_family,requested_pose=sample.spec.pose,
                 requested_view=sample.spec.view,requested_morphology=sample.spec.morphology,
                 requested_tail=sample.spec.tail,requested_style=sample.spec.style,
                 requested_orientation=sample.spec.orientation,requested_composition=sample.spec.composition,
                 extent_mode=sample.spec.extent_mode,background_value=sample.spec.background,
                 target_foreground_fraction=sample.spec.target_fraction,prompt=sample.spec.prompt,
                 black_pixels=int((sample.pixels==0).sum()),black_fraction=sample.black_fraction,
                 Ex_app=lab["Ex_app"],Ey_app=lab["Ey_app"],Ex_over_Es=lab["Ex_app"],Ey_over_Es=lab["Ey_app"],
                 Ex_over_Ey=lab["Ex_app"]/lab["Ey_app"],E_soft=1.,E_hard=10.,nu_soft=.3,nu_hard=.3,
                 specimen_width_mm=40.,specimen_height_mm=40.,specimen_thickness_mm=2.,nominal_strain=.001,
                 mechanical_protocol="finite_plane_stress_lateral_free_tension_Q4_224",
                 solver_protocol_sha256=lab["protocol_fingerprint"],
                 property_bin_x=cell[0],property_bin_y=cell[1],property_bin_count=counts[cell],
                 balanced_sampling_weight=1000/(active*counts[cell]),
                 siglip_cat_logit=sample.cat_logit,siglip_cat_margin=sample.cat_margin,
                 cat_bbox_aspect=sample.geometry[0],cat_bbox_max_side_px=sample.geometry[1],
                 cat_bbox_center_x=sample.geometry[2],cat_bbox_center_y=sample.geometry[3],
                 sha256_pixels=gen.fingerprint(sample.pixels),validation="pass",
                 ai_visual_review="pass" if sample.spec.candidate in reviewed else "not_performed",
                 preview_cell=preview_cells.get(index,""))
        row.update({key:lab[key] for key in ("residual_x","residual_y","energy_relative_error_x","energy_relative_error_y",
                    "force_balance_relative_error_x","force_balance_relative_error_y","fem_valid")})
        row.update({key:sample.diagnostics[key] for key in ("hard_edge_left","hard_edge_right","hard_edge_top","hard_edge_bottom",
                    "foreground_boundary_contact","foreground_fraction","forced_border_pixels")})
        row.update(phase_connectivity(sample.pixels))
        rows.append(row)
    with (output/"metadata.csv").open("w",newline="",encoding="utf-8") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    captions=[f"cat_{i:04d} B={rows[i]['black_fraction']:.1%}\nEx={rows[i]['Ex_app']:.2f} Ey={rows[i]['Ey_app']:.2f}" for i in preview_indices]
    contact_sheet([chosen[i] for i in preview_indices],output/"preview.png",labels=captions)
    report=build_report(pool,chosen,rows,stats,pair_metrics,rejections,review,args.work_dir)
    (output/"quality_report.md").write_text(report,encoding="utf-8")
    own_path=Path(__file__).resolve()
    if own_path.name=="generate_dataset.py":
        for name in ("generate_dataset.py","README.md"):
            source=own_path.parent/name
            destination=output/name
            if source.exists() and source.resolve()!=destination.resolve():
                shutil.copy2(source,destination)
    manifest=dict(seed=args.seed,candidates=len(pool),pool_sha256=pool_hash(pool),selected_candidates=[s.spec.candidate for s in chosen],
                  statistics=stats,pair_metrics=pair_metrics,rejections=rejections,preview_indices=preview_indices,
                  dataset_sha256=hashlib.sha256(array.tobytes()).hexdigest())
    (args.work_dir/"release_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    print(json.dumps({"output":str(output),"count":len(rows),"distribution":stats["after"]}),flush=True)


def build_report(pool,chosen,rows,stats,pair_metrics,rejections,review,work_dir):
    values=np.array([[r["Ex_app"],r["Ey_app"]] for r in rows])
    fractions=np.array([r["black_fraction"] for r in rows])
    after=stats["after"]
    before=stats["before"]
    lines=["# Binary cat mechanical dataset quality report", "",
           f"**{len(rows)} valid images**, selected from {len(pool)} independent local diffusion candidates. "
           "Every PNG is one-bit 224 x 224 and reloads exactly as the corresponding dataset.npy slice. "
           "Array shape=(1000,224,224), dtype=uint8, values={0,1}; no blank maps. 0=hard/black, 1=soft/white.", "",
           f"Black fraction min/median/max: **{fractions.min():.2%} / {np.median(fractions):.2%} / {fractions.max():.2%}**. "
           "The old30–70% constraint and compulsory3px background frame were removed. No filler, noise, dilation, anatomical cropping or repeated transformed sources are used.", "",
           "| Quantity (normalized by E_soft) | Minimum | Median | Maximum |",
           "|---|---:|---:|---:|",
           f"| Ex_app | {values[:,0].min():.5f} | {np.median(values[:,0]):.5f} | {values[:,0].max():.5f} |",
           f"| Ey_app | {values[:,1].min():.5f} | {np.median(values[:,1]):.5f} | {values[:,1].max():.5f} |",
           f"| Ex_app/Ey_app | {(values[:,0]/values[:,1]).min():.4f} | {np.median(values[:,0]/values[:,1]):.4f} | {(values[:,0]/values[:,1]).max():.4f} |", "",
           f"Observed joint6x6 cells: {after['observed_nonempty_bins']}/36. "
           f"Effective occupied cells (exp entropy): {before['effective_occupied_bin_count']:.2f} in screened candidates -> {after['effective_occupied_bin_count']:.2f} after selection. "
           f"Cell-count CV over observed candidate support: {before['count_cv_over_reference_support']:.3f} -> {after['count_cv_over_reference_support']:.3f}. "
           "This is approximate balance over observed support, not a claim that all36 cells or every point inside an occupied cell are physically attainable.", "",
           "Final cell counts (rows: increasing Ey; columns: increasing Ex; shared edges1,2.5,4,5.5,7,8.5,10):", "",
           "```text"]
    lines += [" ".join(f"{n:4d}" for n in line) for line in after["bin_counts_y_rows_x_columns"]]
    lines += ["```", "", "Black fraction counts in10%-wide bins from0% to100%: "
              + ", ".join(map(str,after["black_fraction_histogram_counts"])) + ". "
              "Property coverage takes priority over a flat phase-fraction histogram. CSV balanced_sampling_weight compensates remaining cell-count imbalance; it does not create missing examples.", "",
              "**Mechanical labels:** independent finite40x40x2mm specimen; two perfectly bonded isotropic linear elastic phases; E_soft=1,E_hard=10,nu_soft=nu_hard=.3. "
              "Fully integrated Q4 plane stress, one element per pixel, two0.1% nominal-strain tests. Loaded edges prescribe only axial displacement; transverse movement is free apart from one rigid-translation pin. "
              "Moduli are total reaction/(cross-section x nominal strain). They are apparent values for this protocol, not periodic homogenized constants or experimentally calibrated materials.", "",
              f"All2000 final load cases passed algebraic residual, energy/reaction consistency and homogeneous-modulus bounds. Maximum relative residual: "
              f"{max(max(r['residual_x'],r['residual_y']) for r in rows):.2e}. "
              "Solver checks cover homogeneous phases, analytical nu=0 laminates, axis exchange, linearity, common modulus scaling, and sparse direct-solver agreement. "
              "Pixel-preserving refinement results are reported below when available; they are sampled discretization checks, not certified errors for every image.", "",
              "**Duplicates and geometry:** exact duplicates0;1000 different diffusion seeds. "
              "Near-duplicate filtering combines mirror-aware object-normalized silhouette IoU/pHash with actual phase fraction and FEM distance; opposite phase polarity is retained. "
              "A pair conflicts if silhouette IoU>=.91, or IoU>=.85 and pHash distance<=6, together with |delta black fraction|<=.025 and max(|delta Ex|,|delta Ey|)<=.12. "
              "Different shapes with similar performance are intentionally retained. Pair statistics: "
              + json.dumps({k:v for k,v in pair_metrics.get("selected",{}).items() if k!="most_shape_similar_pairs"},ensure_ascii=False) + ".", "",
              "Requested geometry-family counts: " + ", ".join(f"{k}:{v}" for k,v in sorted(Counter(s.spec.geometry_family for s in chosen).items())) + ". "
              "These are prompt strata, not verified anatomical labels. Both phases can contact specimen boundaries;4-connected hard spanning is only a descriptor because the soft solid also carries load.", "",
              "Measured bbox aspect ratio p05/median/p95: "
              + "/".join(f"{v:.3f}" for v in np.quantile([r['cat_bbox_aspect'] for r in rows],[.05,.5,.95]))
              + f". Black/white backgrounds: {sum(r['background_value']==0 for r in rows)}/{sum(r['background_value']==1 for r in rows)}. "
              + f"Hard phase spans x/y by4-connectivity: {sum(r['hard_spans_x'] for r in rows)}/{sum(r['hard_spans_y'] for r in rows)}.", "",
              f"**Recognizability:** {sum(r['ai_visual_review']=='pass' for r in rows)}/1000 final maps have AI visual review, in addition to SigLIP screening. "
              "Review used contact sheets with enlargement of ambiguous cases. No human visual inspection is claimed. "
              "A separate50-image pilot passed binary, loading, duplicates, FEM and AI visual checks before full generation. The final preview is one seeded random100-image10x10 sheet.", "",
              "Screening rejections: " + json.dumps(rejections,sort_keys=True) + ".", "",
              "**Limitations:** some requested poses/views are missed; silhouettes and stylized limbs can repeat familiar cat forms. "
              "Extreme phase fractions can mean small cats or outlines. Performance support remains correlated and cannot cover the whole square; sparse boundary regions still contain fewer independent examples. "
              "Linear elasticity, perfect bonding and plane stress need future physical/3D validation; stress hotspots, failure, nonlinear response and print resolution are not validated. "
              "1000 images are a starting dataset, not evidence that a conditional generator will generalize. Final generated binary maps must be re-simulated."]
    refinement=Path(work_dir)/"final_refinement.json"
    if refinement.exists():
        data=json.loads(refinement.read_text(encoding="utf-8"))
        lines += ["",f"Refinement: {data['count']} selected extreme/narrow-path specimens were checked at224 and448 elements per side, preserving original pixel geometry. "
                  f"Maximum observed relative modulus change: {data['maximum_relative_change']:.3%}. No universal continuum-error bound is claimed."]
    baseline=REFERENCE_BASELINE
    baseline_path=Path(work_dir)/"baseline_summary.json"
    if baseline_path.exists():baseline=json.loads(baseline_path.read_text(encoding="utf-8"))
    if baseline:
        lines += ["", "Original dataset comparison (same solver/protocol, all1000 original maps): "
                  f"Ex={baseline['Ex_min']:.4f}–{baseline['Ex_max']:.4f}; Ey={baseline['Ey_min']:.4f}–{baseline['Ey_max']:.4f}; "
                  f"observed6x6 cells={baseline['occupied_bins']}; effective cells={baseline['effective_bins']:.2f}. "
                  "Original0072/0083: Ex2.127904/2.147694,Ey1.831056/1.597203; original0075/0120: Ex2.298520/2.262684,Ey1.681299/2.050866. "
                  "These references are old filenames; the rebuilt dataset has new ordering."]
    similar_fraction=(fractions>=.475)&(fractions<=.525)
    if similar_fraction.sum()>=2:
        comparable=values[similar_fraction]
        lines += ["",f"Geometry effect at similar composition: {int(similar_fraction.sum())} final maps have47.5–52.5% black. "
                  f"Within this narrow fraction band, Ex ranges {comparable[:,0].min():.4f}–{comparable[:,0].max():.4f} "
                  f"and Ey {comparable[:,1].min():.4f}–{comparable[:,1].max():.4f}; labels are not assigned from fraction alone."]
    reproduction=Path(work_dir)/"reproduction_result.json"
    if reproduction.exists():
        check=json.loads(reproduction.read_text(encoding="utf-8"))
        lines += ["",f"Reproduction spot check: regenerated {check['checked_candidates']} candidates in their original4-image batches, "
                  f"including {check['retained_images']} retained images; pixel-exact agreement={check['all_pixels_exact']}. "
                  "This is sampled same-workstation evidence, not a cross-platform determinism guarantee."]
    return "\n".join(lines)+"\n"


def run_full(args):
    pilot_path=args.work_dir/"pilot_result.json"
    if not pilot_path.exists():
        raise RuntimeError("Run and inspect the 50-image pilot before full generation")
    pilot=json.loads(pilot_path.read_text(encoding="utf-8"))
    if pilot.get("technical_validation")!="pass" or pilot.get("fem_validation")!="pass":
        raise RuntimeError("Pilot has not passed objective checks")
    if pilot.get("visual_review")!="AI-reviewed":
        raise RuntimeError("The selected pilot images need visual review before full generation")
    expected=dict(seed=args.seed,batch_size=args.batch_size,pipeline_version=gen.PIPELINE_VERSION)
    if pilot.get("generation_config")!=expected:
        raise RuntimeError("Run a pilot with this seed, batch size and pipeline version first")
    pool=gen.acquire("full",args.candidates or 1600,args.work_dir,args.seed,args.batch_size)
    review=review_for(pool,"full",args.work_dir)
    valid,rejections=eligible_pool(pool,review)
    labels=get_labels(valid,args.work_dir,"full",args.workers)
    indices,stats,pair_metrics=selection.select(valid,labels,target=1000,seed=args.seed)
    export_release(args,pool,valid,labels,indices,stats,pair_metrics,review,rejections)


def validate_export(output):
    output=Path(output)
    array=np.load(output/"dataset.npy",allow_pickle=False)
    if array.shape!=(1000,224,224) or array.dtype!=np.uint8 or set(np.unique(array))!={0,1}:
        raise ValueError("Array contract failed")
    expected=[f"cat_{i:04d}.png" for i in range(1000)]
    files=sorted(p.name for p in (output/"images").glob("*.png"))
    if files!=expected:
        raise ValueError("Exactly 1000 correctly named PNGs required")
    with (output/"metadata.csv").open(newline="",encoding="utf-8") as stream:
        rows=list(csv.DictReader(stream))
    if [r["filename"] for r in rows]!=expected:
        raise ValueError("Metadata/file order mismatch")
    cells_by_row=[(int(r["property_bin_x"]),int(r["property_bin_y"])) for r in rows]
    bin_counts=Counter(cells_by_row)
    parameters=dict(E_soft=1.,E_hard=10.,nu_soft=.3,nu_hard=.3,specimen_width_mm=40.,
                    specimen_height_mm=40.,specimen_thickness_mm=2.,nominal_strain=.001)
    hashes=[]
    protocol=mechanics_fingerprint()
    for i,row in enumerate(rows):
        image=gen.load_png(output/"images"/row["filename"])
        if not np.array_equal(image,array[i]):
            raise ValueError("PNG/NumPy mismatch")
        digest=gen.fingerprint(image)
        hashes.append(digest)
        if digest!=row["sha256_pixels"] or int(row["black_pixels"])!=int((image==0).sum()):
            raise ValueError("Metadata pixel validation failed")
        if not .05 <= float(row["black_fraction"]) <= .95:
            raise ValueError("Occupancy limit failed")
        if abs(float(row["black_fraction"])-float((image==0).mean()))>1e-12:
            raise ValueError("Metadata black fraction mismatch")
        for key,value in parameters.items():
            if float(row[key])!=value:
                raise ValueError(f"Unexpected physical parameter {key}")
        if row["mechanical_protocol"]!="finite_plane_stress_lateral_free_tension_Q4_224":
            raise ValueError("Unknown mechanical protocol")
        if row["solver_protocol_sha256"]!=protocol:
            raise ValueError("The solver/material/protocol implementation differs from these labels")
        for axis in ("x","y"):
            if not 1-1e-6 <= float(row[f"E{axis}_app"]) <= 10+1e-6:
                raise ValueError("Modulus outside homogeneous bounds")
            for prefix,limit in (("residual",1e-7),("energy_relative_error",1e-6),("force_balance_relative_error",1e-7)):
                diagnostic=float(row[f"{prefix}_{axis}"])
                if not np.isfinite(diagnostic) or not 0<=diagnostic<=limit:
                    raise ValueError(f"Failed FEM {prefix} check")
            if float(row[f"E{axis}_over_Es"])!=float(row[f"E{axis}_app"]):
                raise ValueError("Normalized condition mismatch")
        ex,ey=float(row["Ex_app"]),float(row["Ey_app"])
        if not np.isclose(float(row["Ex_over_Ey"]),ex/ey,rtol=1e-12):
            raise ValueError("Anisotropy ratio mismatch")
        cell=(min(5,max(0,int((ex-1)/1.5))),min(5,max(0,int((ey-1)/1.5))))
        if cell!=cells_by_row[i] or int(row["property_bin_count"])!=bin_counts[cell]:
            raise ValueError("Property bin/count mismatch")
        expected_weight=1000/(len(bin_counts)*bin_counts[cell])
        if not np.isclose(float(row["balanced_sampling_weight"]),expected_weight,rtol=1e-12):
            raise ValueError("Balanced sampling weight mismatch")
        if row["fem_valid"]!="True":
            raise ValueError("Invalid FEM flag")
        if row["validation"]!="pass" or row["ai_visual_review"]!="pass":
            raise ValueError("Incomplete technical or visual validation")
    if len(set(hashes))!=1000 or len({r['seed'] for r in rows})!=1000:
        raise ValueError("Duplicate pixels or source seeds")
    cells=[(int(r["preview_cell"]),i) for i,r in enumerate(rows) if r["preview_cell"]!=""]
    if sorted(c for c,i in cells)!=list(range(100)):
        raise ValueError("Preview membership invalid")
    with Image.open(output/"preview.png") as preview:
        if preview.size!=(2306,2646):
            raise ValueError(f"Wrong preview dimensions {preview.size}")
        for cell,index in cells:
            x=6+(cell%10)*230;y=6+(cell//10)*264
            if not np.array_equal(np.asarray(preview.crop((x,y,x+224,y+224)),dtype=np.uint8),array[index]):
                raise ValueError("Preview cell does not match exported image")
    print(json.dumps(dict(valid_images=1000,shape=list(array.shape),dtype=str(array.dtype),values=[0,1],
                          exact_duplicates=0,preview_cells=100,metadata_matches=True)),flush=True)


def workflow_main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode",choices=("pilot","full","validate"))
    parser.add_argument("--work-dir",type=Path,default=Path(tempfile.gettempdir())/"binary-cat-mechanics")
    parser.add_argument("--output",type=Path,default=Path(__file__).resolve().parent/"cat_binary_dataset")
    parser.add_argument("--seed",type=int,default=gen.MASTER_SEED)
    parser.add_argument("--batch-size",type=int,default=4)
    parser.add_argument("--candidates",type=int)
    parser.add_argument("--workers",type=int,default=8)
    args=parser.parse_args()
    args.work_dir.mkdir(parents=True,exist_ok=True)
    if args.mode=="pilot":run_pilot(args)
    elif args.mode=="full":run_full(args)
    else:validate_export(args.output)


if __name__=="__main__":
    workflow_main()
