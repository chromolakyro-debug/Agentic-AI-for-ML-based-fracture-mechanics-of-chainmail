"""Independent binary-illustration, pixel-art and cat-portrait candidates.

The supplied reference images inform style descriptions only; their pixels are
never read by this generator. Every source uses a new seed. Multiple local
framing candidates are FEM search alternatives, not independent dataset images;
retain at most one final image per source seed.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
from pathlib import Path
import io
import json

import numpy as np
from PIL import Image
from scipy import ndimage
from skimage.filters import threshold_otsu

import candidate_generation as base

VERSION = 1
MASTER_SEED = 1144186
STYLE_FAMILIES = ("binary_illustration", "pixel_art", "graphic_portrait")
TARGET_GROUPS = ("high", "x_strong", "y_strong", "explore")
TARGETS = {
    "high": ((8.7,8.7),(9.0,8.5),(8.5,9.0),(8.2,8.2)),
    "x_strong": ((4.6,2.45),(5.6,2.9),(6.4,3.6),(7.2,4.5)),
    "y_strong": ((3.5,7.5),(4.6,8.1),(6.8,8.7),(2.2,4.8)),
    "explore": ((5.1,7.4),(7.4,5.1),(5.0,3.4),(3.4,5.0),(6.2,4.0)),
}
STYLES = {
    "tabby_linocut": ("binary_illustration", "A bold black-and-white linocut illustration of a domestic tabby cat. Clearly drawn eyes, nose, whiskers, broad natural fur markings and internal curves separating chest, belly and paws. Use large clean ink regions and bold readable strokes."),
    "fur_woodcut": ("binary_illustration", "A striking binary woodcut illustration of a fluffy domestic cat. Large expressive eyes, pointed ears, a furry chest and several coarse angular fur tufts inside the body. Strong black and white anatomical regions, with clear paw and tail details."),
    "anatomical_ink": ("binary_illustration", "A detailed but clean black ink illustration of a domestic cat, with bright eyes, a nose, long clearly drawn whiskers and broad internal fur and limb details. Use a few wide ink strokes and large white gaps, not fine crosshatching."),
    "tuxedo_graphic": ("binary_illustration", "A crisp black-and-white illustration of a tuxedo domestic cat. A broad white chest bib, white muzzle and paws contrast with connected black head and body regions. Clear expressive eyes and a few long whiskers; detailed recognizable feline anatomy."),
    "pixel_tabby": ("pixel_art", "Pure monochrome pixel art of a domestic tabby cat, constructed from large clearly visible square pixel blocks. Stepped jagged edges, square eyes, blocky ears, paws and a bent tail. Use broad black-and-white tabby fur patches that follow the cat anatomy."),
    "pixel_tuxedo": ("pixel_art", "A black-and-white retro pixel-art domestic tuxedo cat, with an expressive blocky face, square eyes, a white chest bib and blocky paws. Visible large square pixels and stepped contours throughout, no smooth vector curves."),
    "pixel_fluffy": ("pixel_art", "A monochrome pixel-art fluffy domestic cat built from chunky square blocks, with a wide stepped chest ruff, large square eyes, pointed stair-step ears and a thick blocky tail. Distinct body and four connected paws, simple readable pixel clusters."),
    "pixel_round": ("pixel_art", "A playful monochrome pixel-art domestic cat with a round blocky belly, large square eyes, stair-step pointed ears, tiny connected block paws and a large angular curling tail. Every contour has clearly visible square-pixel steps."),
    "portrait_tabby_stencil": ("graphic_portrait", "A bold black-and-white stencil portrait of a domestic tabby cat showing its entire pointed-eared head, shoulders, chest and two front paws. Large expressive feline eyes, a small triangular nose, whiskers and broad natural forehead and cheek markings."),
    "portrait_pixel_tabby": ("graphic_portrait", "A striking monochrome pixel-art portrait of a domestic tabby cat, with its complete head, pointed ears, chest and two front paws. Large square-pixel eyes and blocky natural tabby markings on forehead, cheeks and chest. Clearly visible pixel steps and connected anatomy."),
    "portrait_fluffy_ink": ("graphic_portrait", "A crisp binary ink portrait of a fluffy domestic cat showing the whole head, both ears, broad furry chest and two front paws. Expressive eyes, clear nose and whiskers, wide angular fur tufts and large clean black-and-white regions."),
    "portrait_tuxedo": ("graphic_portrait", "A bold graphic black-and-white tuxedo domestic cat portrait, complete head and ears above a white chest bib and two separate front paws. Large feline eyes, a triangular nose, long whiskers and asymmetrical natural facial markings."),
}
ANATOMIES = (
    ("long_recline", "lying", ("side","three_quarter"), "lying at full length with a deep long horizontal torso, forepaws reaching forward and hind paws folded beside the haunches"),
    ("curved_turn", "looking_back", ("three_quarter","top_down"), "turning a long curved torso while looking back over one shoulder, broad hips offset from shoulders and a curling tail beside its body"),
    ("broad_sit", "sitting", ("front","three_quarter"), "sitting with broad rounded haunches, a narrow waist and two well-separated forepaws beneath an expressive cat face"),
    ("upright_reach", "stretching", ("front","three_quarter"), "reaching upward on its hind paws, one forepaw held high and the other bent near its chest, a long upright torso and clear pointed-eared cat head"),
    ("crouching_arch", "crouching", ("side","three_quarter"), "crouching with a broad low chest, an arched narrow waist, raised haunches and a tail curving closely around its flank"),
    ("side_sprawl", "playing", ("top_down","three_quarter"), "sprawled on its side with a broad belly, two bent forepaws near its face and two hind paws extending at different angles"),
    ("alert_step", "walking", ("side","three_quarter"), "taking one long step with a low deep horizontal torso, a broad clear cat head and four connected paws spread along its body"),
    ("open_curl", "curling", ("top_down","three_quarter"), "curled into an open C shape with face and pointed ears visible, a broad curved torso and paws tucked inside the natural curve"),
)
PORTRAIT_POSITIVES = (
    "a black and white portrait of a domestic cat showing its head chest and front paws",
    "a monochrome pixel art portrait of a tabby pet cat with pointed ears and whiskers",
    "a detailed black ink illustration of a domestic cat face and furry chest",
)
PORTRAIT_NEGATIVES = (
    "a black and white portrait of a dog",
    "a monochrome pixel art portrait of a fox",
    "a portrait of a rabbit with long ears",
    "a black and white owl face",
    "a portrait of a tiger or lion, a wild big cat",
    "a drawing of a raccoon face",
    "a drawing of a human face",
    "an abstract black and white symbol or lettering",
    "a drawing of more than one animal",
)
STYLE_POSITIVES = (
    "a detailed black and white ink illustration of one whole domestic cat",
    "a monochrome pixel art drawing of a complete domestic cat with paws and a tail",
    "a black and white woodcut picture of a full body tabby cat",
)


def source_plan(spec):
    family = STYLES[spec.style][0]
    group = TARGET_GROUPS[spec.action_variant//100]
    target = TARGETS[group][spec.action_variant%100]
    grid = (56 if spec.seed%2 else 112) if family == "pixel_art" else (56 if spec.style == "portrait_pixel_tabby" else 0)
    return dict(source_stage="styles",style_family=family,target_group=group,
                target_Ex=float(target[0]),target_Ey=float(target[1]),source_id=int(spec.candidate),
                source_seed=int(spec.seed),generation_master_seed=MASTER_SEED,
                style_pipeline_version=VERSION,pixel_art_grid=grid)


def specs_for(mode, total, master_seed=MASTER_SEED):
    if mode not in ("styles","pilot","full") or total < 1:
        raise ValueError("Expected mode styles/pilot/full and positive count")
    rng = np.random.default_rng([int(master_seed), {"styles":741,"pilot":742,"full":743}[mode]])
    specs = []
    while len(specs)<total:
        families = rng.permutation(np.tile(np.arange(3),4))
        for family_index in families:
            family = STYLE_FAMILIES[int(family_index)]
            style = str(rng.choice([name for name,value in STYLES.items() if value[0]==family]))
            probabilities = (.20,.15,.30,.35) if family=="graphic_portrait" else (.25,.25,.25,.25)
            group_index = int(rng.choice(4,p=probabilities))
            group = TARGET_GROUPS[group_index]
            target_index = int(rng.integers(len(TARGETS[group])))
            if family == "graphic_portrait":
                geometry = "head_chest_front_paws"
                pose = "portrait"
                view = str(rng.choice(("front","front","three_quarter")))
                anatomy = str(rng.choice((
                    "a broad-cheeked adult with a wide furry chest and short rounded front paws",
                    "a slender narrow-faced cat with tall ears, a long chest and long front paws",
                    "a round-faced kitten with very large eyes, a compact chest and broad paws",
                    "an angular-faced adult with heavy brows, wide shoulders and one front paw slightly raised")))
                camera = "Frontal portrait, looking toward the viewer." if view=="front" else "Slight three-quarter portrait with both eyes and both front paws visible."
                body_request = "The complete head, both ears, entire chest and both front paws are inside the image. No surrounding emblem or circular backdrop."
            else:
                pool = list(ANATOMIES)
                if group == "x_strong":
                    pool = [a for a in pool if a[0] in ("long_recline","crouching_arch","alert_step","side_sprawl")]
                elif group == "y_strong":
                    pool = [a for a in pool if a[0] in ("upright_reach","broad_sit","curved_turn")]
                geometry,pose,views,anatomy = pool[int(rng.integers(len(pool)))]
                view = str(rng.choice(views))
                camera = base.CAMERAS[view]
                body_request = "One complete cat, entire head, ears, torso, visible paws and tail fully inside the image."
            morph = str(rng.choice(base.MORPHS)) if family!="graphic_portrait" else anatomy
            tail = str(rng.choice(base.TAILS)) if family!="graphic_portrait" else "tail not requested in chest-up portrait"
            orientation = str(rng.choice(("leftward","rightward"))) if view!="front" else "front_facing"
            background = int(rng.random() >= (.75 if group!="explore" else .5))
            composition = str(rng.choice(("center","center","left","right","upper","lower")))
            if group=="high":
                target_fraction = float(rng.uniform(.045,.16))
                extent = "fraction"
            elif family=="graphic_portrait":
                target_fraction = float(rng.uniform(.25,.55))
                extent = "max_extent"
            elif group in ("x_strong","y_strong"):
                target_fraction = float(rng.uniform(.27,.55))
                extent = "max_extent"
            else:
                target_fraction = float(rng.uniform(.15,.48))
                extent = "fraction" if rng.random()<.5 else "max_extent"
            subject = f"One {morph}, {anatomy}, with {tail}." if family!="graphic_portrait" else f"The subject is {anatomy}."
            direction = f"The head turns {orientation}." if orientation!="front_facing" else ""
            prompt = (f"{STYLES[style][1]} {camera} {subject} {direction} {body_request} "
                      "Pure black and pure white only, empty white background. Every mark belongs to the cat's anatomy or natural coat. "
                      "No gray, shading gradients, stars, moons, circles, frames, text, scenery, floor or other objects.")
            specs.append(base.Spec(400000+len(specs),int(rng.integers(0,2**63-1)),pose,view,morph,tail,
                                   style,orientation,composition,group_index*100+target_index,background,
                                   target_fraction,prompt,geometry,extent))
            if len(specs)==total:
                break
    return specs


def _finish_pixels(pixels, spec, details):
    grid = source_plan(spec)["pixel_art_grid"]
    if grid:
        # A genuine coarse binary grid with exact 2x2 or 4x4 constant blocks.
        small = Image.fromarray(pixels).resize((grid,grid),Image.Resampling.NEAREST)
        pixels = np.asarray(small.resize((224,224),Image.Resampling.NEAREST),dtype=np.uint8)
    fore = pixels != spec.background
    if not fore.any():
        return pixels,dict(details,foreground_width=0,foreground_height=0),np.zeros(4)
    x0,y0,x1,y1 = base.bbox(fore)
    w,h=x1-x0,y1-y0
    black=pixels==0
    details=dict(details,resized_bbox=[x0,y0,x1,y1],foreground_width=w,foreground_height=h,
                 foreground_fraction=float(fore.mean()),black_fraction=float(black.mean()),pixel_art_grid=grid,
                 pixel_block_side=224//grid if grid else 1,
                 hard_edge_left=float(black[:,0].mean()),hard_edge_right=float(black[:,-1].mean()),
                 hard_edge_top=float(black[0].mean()),hard_edge_bottom=float(black[-1].mean()),
                 foreground_boundary_contact=bool(fore[0].any() or fore[-1].any() or fore[:,0].any() or fore[:,-1].any()))
    geometry=np.array([w/h,max(w,h),(x0+w/2)/224,(y0+h/2)/224],dtype=float)
    return pixels,details,geometry


def binarize(image, spec):
    gray=np.asarray(image.convert("L"))
    threshold=float(np.clip(threshold_otsu(gray),130,195))
    raw=gray<threshold
    labels,count=ndimage.label(raw,structure=base.CONNECTIVITY)
    if not count:
        return np.full((224,224),spec.background,np.uint8),1-float(spec.background),"blank",np.zeros(4),{},raw.astype(np.uint8)
    areas=np.bincount(labels.ravel());areas[0]=0
    clean=np.isin(labels,np.flatnonzero(areas>=10))
    if not clean.any():
        return np.full((224,224),spec.background,np.uint8),1-float(spec.background),"blank",np.zeros(4),{},raw.astype(np.uint8)
    # Group nearby ink components for object localization only. Dilation is
    # NEVER included in output pixels; disconnected fur/eye/whisker marks are
    # natural anatomical detail rather than automatic fragmentation failures.
    grouped,n=ndimage.label(ndimage.binary_dilation(clean,structure=base.CONNECTIVITY,iterations=6),structure=base.CONNECTIVITY)
    grouped_ink=np.bincount(grouped[clean],minlength=n+1);grouped_ink[0]=0
    # Preserve ALL substantial source components. A face, paw, whisker or fur
    # region must not disappear merely because it is disconnected from the
    # largest ink component. Multi-object/background failures are semantic and
    # visual-review exclusions, rather than silent anatomical deletions.
    keep=clean
    x0,y0,x1,y1=base.bbox(keep)
    issue=""
    grouped_share=float(grouped_ink.max()/clean.sum())
    if min(x0,y0,gray.shape[1]-x1,gray.shape[0]-y1)<2:
        issue="source_clipped"
    elif grouped_share<.15 and int((areas>=10).sum())>80:
        issue="fragmented_objects"
    pixels,details,geometry=base.frame_mask(keep,spec)
    details.update(threshold=threshold,source_main_component_share=float(areas.max()/keep.sum()),
                   source_grouped_component_share=grouped_share,source_grouping_radius=6,
                   retained_source_components=int((areas>=10).sum()),
                   removed_source_pixels=int(raw.sum()-keep.sum()))
    pixels,details,geometry=_finish_pixels(pixels,spec,details)
    fraction=float((pixels==0).mean())
    details.update(source_plan(spec),variant_id=-1,source_candidate_id=spec.candidate)
    if not .05<=fraction<=.99:
        issue=issue or "black_fraction_out_of_range"
    return pixels,fraction,issue,geometry,details,keep.astype(np.uint8)


class LocalModels(base.LocalModels):
    def __init__(self):
        super().__init__()
        texts=PORTRAIT_POSITIVES+PORTRAIT_NEGATIVES+STYLE_POSITIVES
        with self.torch.inference_mode():
            tokens=self.processor(text=list(texts),padding="max_length",return_tensors="pt").to("cuda")
            self.style_text_features=self.normalize(self.siglip.get_text_features(**tokens).float())

    def generate(self,specifications):
        torch=self.torch
        generators=[torch.Generator("cpu").manual_seed(s.seed) for s in specifications]
        with torch.inference_mode():
            images=self.pipe([s.prompt for s in specifications],width=512,height=512,generator=generators,
                             num_inference_steps=base.STEPS,guidance_scale=0.,max_sequence_length=256).images
        processed=[binarize(image,s) for image,s in zip(images,specifications)]
        semantic=[base.semantic_image(r[0],s.background) for r,s in zip(processed,specifications)]
        with torch.inference_mode():
            inputs=self.processor(images=semantic,return_tensors="pt").to("cuda")
            features=self.normalize(self.siglip.get_image_features(**inputs).float())
            logits=(features@self.text_features.T*self.logit_scale+self.logit_bias).cpu().numpy()
            style_logits=(features@self.style_text_features.T*self.logit_scale+self.logit_bias).cpu().numpy()
            embeddings=features.cpu().numpy()
        samples=[]
        for i,(spec,result) in enumerate(zip(specifications,processed)):
            pixels,fraction,issue,geometry,details,source_mask=result
            if source_plan(spec)["style_family"]=="graphic_portrait":
                cat_logit=float(style_logits[i,:len(PORTRAIT_POSITIVES)].max())
                # Deliberately do not use the old "cat head without a body"
                # alternative for a requested head/chest/front-paw portrait.
                negatives=style_logits[i,len(PORTRAIT_POSITIVES):len(PORTRAIT_POSITIVES)+len(PORTRAIT_NEGATIVES)]
                alternatives=PORTRAIT_NEGATIVES
                details["semantic_protocol"]="portrait_positive_vs_animal_portraits_v1"
            else:
                cat_logit=float(max(logits[i,:len(base.CAT_TEXT)].max(),style_logits[i,-len(STYLE_POSITIVES):].max()))
                negatives=logits[i,len(base.CAT_TEXT):len(base.CAT_TEXT)+len(base.ALTERNATIVES)]
                alternatives=base.ALTERNATIVES
                details["semantic_protocol"]="whole_cat_and_binary_style_positive_v1"
            strongest=int(negatives.argmax())
            png=io.BytesIO();base.save_png(pixels,png);png.seek(0)
            if not np.array_equal(base.load_png(png),pixels):
                raise ValueError("PNG roundtrip failed")
            samples.append(base.Sample(spec,pixels,embeddings[i],cat_logit,cat_logit-float(negatives[strongest]),
                                       alternatives[strongest],logits[i,len(base.CAT_TEXT)+len(base.ALTERNATIVES):],
                                       fraction,issue,geometry,details,source_mask))
        return samples


def rejection(sample):
    p=sample.pixels
    if p.shape!=(224,224) or p.dtype!=np.uint8 or not np.isin(p,[0,1]).all():
        return "invalid_array"
    if not .05<=float((p==0).mean())<=.99:
        return "black_fraction_out_of_range"
    if sample.issue:
        return sample.issue
    if max(sample.diagnostics.get("foreground_width",0),sample.diagnostics.get("foreground_height",0))<84:
        return "cat_too_small_for_style_details"
    if not np.isfinite(sample.cat_logit) or not np.isfinite(sample.cat_margin):
        return "invalid_semantic_score"
    if sample.cat_logit < -3 or sample.cat_margin < 0:
        return "low_cat_score"
    grid=sample.diagnostics.get("pixel_art_grid",0)
    if grid:
        block=224//grid
        if not np.array_equal(p,np.repeat(np.repeat(p[::block,::block],block,axis=0),block,axis=1)):
            return "pixel_grid_contract_failed"
    return ""


def variants_for_sample(sample,count=3):
    if not 1<=count<=3:
        raise ValueError("At most three local FEM variants per independent source")
    if sample.issue not in ("","black_fraction_out_of_range") or not sample.source_mask.any():
        return []
    spec=sample.spec
    plan=source_plan(spec)
    # Correct when a non-default master seed was used; cached source is truth.
    plan["generation_master_seed"]=sample.diagnostics.get("generation_master_seed",MASTER_SEED)
    if plan["target_group"]=="high":
        options=[("fraction",spec.target_fraction,spec.composition),
                 ("fraction",max(.018,spec.target_fraction*.70),"center"),
                 ("fraction",min(.28,spec.target_fraction*1.45),spec.composition)]
    else:
        _,full_details,_=base.frame_mask(sample.source_mask.astype(bool),replace(spec,extent_mode="max_extent"))
        smaller=full_details["foreground_fraction"]*.90**2
        alternate={"left":"right","right":"left","upper":"lower","lower":"upper"}.get(spec.composition,"left")
        options=[(spec.extent_mode,spec.target_fraction,spec.composition),
                 ("fraction",smaller,"center"),("max_extent",spec.target_fraction,alternate)]
    result=[];seen=set()
    for variant_id,(extent,fraction,composition) in enumerate(options[:count]):
        variant_spec=replace(spec,candidate=spec.candidate*4+variant_id,extent_mode=extent,
                             target_fraction=float(fraction),composition=composition)
        pixels,details,geometry=base.frame_mask(sample.source_mask.astype(bool),variant_spec)
        pixels,details,geometry=_finish_pixels(pixels,variant_spec,details)
        digest=base.fingerprint(pixels)
        if digest in seen:continue
        seen.add(digest)
        details=dict(sample.diagnostics,**details)
        details.update(plan,variant_id=variant_id,source_candidate_id=spec.candidate,
                       semantic_evaluation="source_object_normalized; final actual-scale AI review required")
        actual=float((pixels==0).mean())
        issue="" if sample.issue=="black_fraction_out_of_range" else sample.issue
        if not .05<=actual<=.99:issue=issue or "black_fraction_out_of_range"
        result.append(base.Sample(variant_spec,pixels,sample.embedding,sample.cat_logit,sample.cat_margin,
                                  sample.alternative,sample.view_scores,actual,issue,geometry,details,sample.source_mask))
    return result


def iter_acquire(mode,total,work_dir,seed=MASTER_SEED,batch_size=4):
    specifications=specs_for(mode,total,seed)
    cache=Path(work_dir)/f"styles-v{VERSION}-{mode}-{seed}-b{batch_size}"
    cache.mkdir(parents=True,exist_ok=True)
    models=None;passed=0
    for start in range(0,total,batch_size):
        batch=specifications[start:start+batch_size]
        path=cache/f"batch_{start:05d}_{len(batch)}.npz"
        if path.exists():
            samples=base.read_batch(path)
            if [s.spec for s in samples]!=batch:raise ValueError(f"Style source cache mismatch: {path}")
        else:
            if models is None:models=LocalModels()
            samples=models.generate(batch)
            for sample in samples:sample.diagnostics["generation_master_seed"]=int(seed)
            base.cache_batch(path,samples)
        passed+=sum(not rejection(s) for s in samples)
        if (start+len(batch))%20==0 or start+len(batch)==total:
            print(f"styles {mode}: {start+len(batch)}/{total} independent sources; {passed} source auto-pass",flush=True)
        yield samples


def acquire(mode,total,work_dir,seed=MASTER_SEED,batch_size=4):
    return [sample for batch in iter_acquire(mode,total,work_dir,seed,batch_size) for sample in batch]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode",choices=("styles","pilot","full"),default="styles",nargs="?")
    parser.add_argument("--count",type=int,default=120)
    parser.add_argument("--seed",type=int,default=MASTER_SEED)
    parser.add_argument("--batch-size",type=int,default=4)
    parser.add_argument("--work-dir",type=Path,required=True)
    args=parser.parse_args()
    sources=acquire(args.mode,args.count,args.work_dir,args.seed,args.batch_size)
    variants=[v for s in sources for v in variants_for_sample(s)]
    print(json.dumps(dict(sources=len(sources),variants=len(variants),valid_variants=sum(not rejection(v) for v in variants),
                          source_styles=dict(Counter(s.diagnostics["style_family"] for s in sources))),indent=2))


if __name__=="__main__":main()
