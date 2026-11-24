"""Independent sources in the user's nine retained reference styles.

Reference pixels are never copied into generated maps. Reference filename/SHA
identify the user-supplied style examples. Reference 05 (old flat silhouette)
is deliberately absent. Every source yields ONE candidate, without augmentation
copies. All rendering remains binary at 224px; no FEM or final selection here.
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import hashlib
import io
import json

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
from skimage.filters import threshold_otsu

import candidate_generation as base

VERSION = 2
MASTER_SEED = 20261114
STYLE_FAMILIES = ("triangular_lattice", "binary_pen", "outline_pixel", "whiskered_sprite",
                  "woodcut", "comic", "geometric", "halftone", "inverted_line")
EXTENTS = (160, 180, 202, 224)
REFERENCE_FILES = {
    "triangular_lattice": ("02-retained-triangular-lattice.png", "a0e1dee8a9c0b84e73fcc9d46b7a3688c716b045d9027f773af9153390a3dc6c"),
    "binary_pen": ("03-binary-pen-illustration.png", "c61e7ab71a5cc7fd2ce34414268e3a882deced616464d0b088a30a7e3321c6a1"),
    "outline_pixel": ("04-outline-pixel-art.png", "a4c0316055b01665180832234b6ab84314ac590e8ac375c2b60ebe2f731e1f4d"),
    "whiskered_sprite": ("06-whiskered-pixel-sprite.png", "8ff989030cda0e5c4877e098292cc7d8a11257389d9afa7a4834b3df0d0160ca"),
    "woodcut": ("07-woodcut-engraving.png", "64b50dc862a560b119788c52d48dc575601ac9c0a7f3d01309c1ee94046b1566"),
    "comic": ("08-comic-cartoon.png", "9db7b26186e79dbe979ee595a9cad804e57cdc1c7870190a7741e562ac459711"),
    "geometric": ("09-geometric-abstract.png", "b0f2c60790be8913d0d9d51f0ad18fa444d2c299bc67f9c41e34dd897316851f"),
    "halftone": ("10-dithered-halftone.png", "dfdb7588604495a42d69a6295305b26b62f8bbb1d111be48594267909179f224"),
    "inverted_line": ("11-moonlit-inverted-line-art.png", "ab85c0b321a39d36ec5f17379fcdb61057a1d3aca0735d767c8f899756d5e0b0"),
}
STYLE_TEXT = {
    "triangular_lattice": "A recognizable domestic cat constructed as a bold triangular lattice. Its feline outer contour, pointed ears, cat head, paws and tail are clear. Inside its torso are several large triangular white openings separated by thick connected black triangular struts; the struts follow and support the cat anatomy.",
    "binary_pen": "A detailed binary pen illustration of a domestic cat. Expressive eyes, nose, whiskers and natural tabby fur markings. Numerous short readable black pen strokes describe its chest, belly and limbs, with clear white gaps. Keep strong anatomical contours and bold facial features that remain legible when reduced.",
    "outline_pixel": "A monochrome outline pixel-art domestic cat built from visible large square pixels. A stepped black contour surrounds a mostly white body, with large square eyes, stair-step ears, block paws and an angular curled tail. Maintain conspicuous chunky pixel steps, not smooth vector curves.",
    "whiskered_sprite": "A monochrome retro pixel sprite of a domestic cat. Chunky filled black pixel clusters, big square white eyes, visible inner-ear blocks, broad rectangular whiskers sticking out from both cheeks and separated block paws. Strong stair-step contours and a blocky cat tail make the animal unmistakable.",
    "woodcut": "A black-and-white woodcut engraving of a domestic cat with expressive eyes, a clear nose and long whiskers. Broad carved white fur cuts alternate with connected black ink regions across the face, chest and body. Coarse directional engraving strokes follow anatomy, with clear internal paw and facial detail.",
    "comic": "An expressive black-and-white comic-book domestic cat with large lively eyes, nose, mouth, whiskers, visible paw pads and jagged fur tufts. Bold variable-width ink contours and white anatomical highlights separate the chest, belly and limbs. Dynamic readable character drawing, not a plain filled silhouette.",
    "geometric": "A clearly recognizable domestic cat assembled from connected geometric anatomy: triangular ears, a circular face, rectangular chest regions, quarter-circle haunches and an arcing segmented tail. Alternating black and white geometric regions divide the face and torso; large simple eyes and whiskers remain clear.",
    "halftone": "A recognizable domestic cat drawn entirely with black halftone dots and small clustered ink marks on white. Dot density describes its fur, round body and tabby markings; eyes, nose, pointed ears and paws stay crisp. All dots belong to the cat, with a clean empty background and no gray wash.",
    "inverted_line": "A bold monochrome tabby-cat cartoon stencil with broad solid BLACK cheeks, chest, torso, paws and tail. Cut large clean WHITE tabby bands into the forehead, cheeks, shoulders and tail. Big white eyes, a white muzzle, a clear nose, thick whiskers and separated toes. Most of the cat body is filled ink, with substantial contrasting natural coat markings.",
}
SUBSTYLES = {
    "triangular_lattice": ("broad triangular ribs and a solid expressive cat head", "unequal triangle sizes following shoulders and haunches", "an open truss torso with stout legs and an arcing tail", "slanted internal struts around a narrow waist"),
    "binary_pen": ("broad striped tabby fur and a white chest bib", "short broken fur strokes and a furry ruff", "long directional pen strokes on a slender coat", "dense facial fur detail with broad white belly regions"),
    "outline_pixel": ("a wide angular curling tail and large square eyes", "a short block tail and chunky stepped paws", "tall stair-step ears and a zigzagging tail", "an asymmetric stepped torso and a large square muzzle"),
    "whiskered_sprite": ("oversized eyes and three thick whisker bars per cheek", "narrow eyes and two long rectangular whiskers per cheek", "large ear interiors and a broad blocky chest bib", "round blocky cheeks and strongly separated square paws"),
    "woodcut": ("broad swept fur cuts and a white muzzle", "short carved marks and a heavy dark chest ruff", "long flowing carved fur strokes around the shoulders", "an asymmetric tabby mask and sharply engraved paws"),
    "comic": ("a surprised expression and broad white paw pads", "a mischievous grin and thick cheek whiskers", "a curious expression with one eyebrow raised", "a sleepy expressive face and long jagged fur tufts"),
    "geometric": ("asymmetrical quarter-circle chest and hip regions", "large triangular shoulder regions and rectangular paws", "a half-black half-white circular face and segmented tail", "offset geometric torso panels and broad rounded haunches"),
    "halftone": ("large clustered dots and coarse dark tabby patches", "small separated dots with crisp dark eyes and nose", "variable-density dot clusters following a furry coat", "broad white chest regions and dense dotted haunches"),
    "inverted_line": ("a broad white chest bib, thick white cheek bands and a filled black torso", "large white curved tabby bands across solid black shoulders and a thick filled tail", "a bold white forehead blaze, large white muzzle and contrasting solid cheek regions", "asymmetrical broad white face markings, white paw pads and a filled black chest"),
}
PORTRAIT_RATE = {"triangular_lattice":0., "binary_pen":.30, "outline_pixel":.15,
                 "whiskered_sprite":.60, "woodcut":.55, "comic":.25,
                 "geometric":.35, "halftone":.25, "inverted_line":.55}
STYLE_PROTOTYPES = {
    "triangular_lattice": "a cat whose body is a triangular truss lattice with large triangular openings",
    "binary_pen": "a detailed black and white pen illustration of a cat with fur strokes and whiskers",
    "outline_pixel": "a chunky square-pixel outline drawing of a cat with a white interior and square eyes",
    "whiskered_sprite": "a monochrome pixel-art cat sprite with square eyes and thick rectangular whiskers",
    "woodcut": "a detailed black and white woodcut engraving of a furry cat with carved fur lines",
    "comic": "an expressive ink comic cartoon cat with facial details whiskers paw pads and jagged fur",
    "geometric": "a cat made of connected black and white geometric shapes with divided circular face and torso",
    "halftone": "a cat depicted using visible halftone dots and clustered ink stippling",
    "inverted_line": "a bold black and white filled cartoon tabby cat stencil with thick contrasting stripes expressive eyes a white muzzle and chest markings",
}
OLD_STYLE_PROTOTYPES = (
    "a plain solid filled cat silhouette with no internal anatomical detail",
    "a simple smooth empty outline of a cat with an empty body and almost no facial details",
    "a minimal flat vector cat icon with a solid body and only tiny eye dots",
)
PORTRAIT_POSITIVES = (
    "a black and white domestic cat portrait showing its whole head chest and two front paws",
    "a monochrome portrait of a pet cat with pointed ears feline eyes nose and whiskers",
    "a stylized domestic cat face with its chest and front paws",
)
PORTRAIT_NEGATIVES = (
    "a portrait of a dog", "a portrait of a fox", "a portrait of a rabbit",
    "an owl face", "a portrait of a tiger or lion", "a raccoon face",
    "a human face", "an abstract symbol or lettering", "more than one animal",
)
EXTRA_CAT_POSITIVES = (
    "a whole domestic cat rendered in geometric black and white art",
    "a full body pixel art pet cat with its paws and tail",
    "a detailed whole domestic cat illustration with fur and whiskers",
)


def cache_dir(mode, work_dir, seed=MASTER_SEED, batch_size=4):
    return Path(work_dir)/f"reference-v{VERSION}-{mode}-{seed}-b{batch_size}"


def verify_reference_files(directory):
    directory=Path(directory)
    for family,(filename,digest) in REFERENCE_FILES.items():
        path=directory/filename
        if hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
            raise ValueError(f"Reference asset changed: {family}: {path}")
    return True


def _family(spec):
    return spec.style.split("__",1)[0]


def source_plan(spec):
    family=_family(spec)
    filename,digest=REFERENCE_FILES[family]
    mode="pilot" if spec.candidate<5000000 else "full"
    pixel_grid=0
    if family in ("outline_pixel","whiskered_sprite"):
        pixel_grid=(32,56,56,112)[spec.seed%4]
    return dict(source_stage=f"reference_{mode}",source_mode=mode,source_id=int(spec.candidate),
                source_seed=int(spec.seed),style_family=family,reference_style_id=filename[:2],
                reference_style_name=family,reference_filename=filename,reference_sha256=digest,
                reference_used_as="style description only; no reference pixels copied",
                style_prompt=STYLE_TEXT[family],style_subtype=SUBSTYLES[family][int(spec.style.split("__",1)[1])],
                style_subtype_id=int(spec.style.split("__",1)[1]),
                requested_extent_px=EXTENTS[spec.action_variant],portrait=spec.pose=="portrait",
                pixel_art_grid=pixel_grid,generation_master_seed=MASTER_SEED,
                reference_pipeline_version=VERSION,variant_id=0,
                requested_target_foreground_fraction=None,
                composition_anchor=list(base.ANCHORS[spec.composition]))


def specs_for(mode, total, master_seed=MASTER_SEED):
    if mode not in ("pilot","full") or total<1:
        raise ValueError("Expected pilot/full and positive source count")
    rng=np.random.default_rng([int(master_seed),811 if mode=="pilot" else 812])
    offset=4000000 if mode=="pilot" else 5000000
    combinations=[(family,background,extent) for family in STYLE_FAMILIES for background in (0,1) for extent in range(4)]
    result=[]
    while len(result)<total:
        for ci in rng.permutation(len(combinations)):
            family,background,extent=combinations[int(ci)]
            subindex=int(rng.integers(4))
            subtype=f"{subindex:02d}"
            portrait=bool(rng.random()<PORTRAIT_RATE[family])
            morph=str(rng.choice(base.MORPHS))
            tail=str(rng.choice(base.TAILS))
            direction=str(rng.choice(("leftward","rightward")))
            if portrait:
                pose="portrait"
                view=str(rng.choice(("front","front","three_quarter")))
                geometry="head_chest_front_paws"
                anatomy=str(rng.choice((
                    "a broad expressive head, thick furry shoulders, a white chest region and two short front paws",
                    "a slender angular head, tall ears, a long narrow chest and two long front paws",
                    "a large rounded kitten head, oversized eyes, compact chest and broad front paws",
                    "a mature cat face with a broad muzzle, offset shoulders and one front paw slightly raised")))
                subject=f"Portrait of {morph}, with {anatomy}. Complete head, both ears, chest and both front paws inside the image."
                tail="tail not requested in head-chest-front-paws portrait"
            else:
                geometry,pose,views,anatomy=base.GEOMETRY_FAMILIES[int(rng.integers(len(base.GEOMETRY_FAMILIES)))]
                views=tuple(v for v in views if v!="rear") or ("three_quarter",)
                view=str(rng.choice(views))
                subject=f"One {morph}, {anatomy}, with {tail}. Complete head, ears, torso, paws and tail inside the image."
            camera=base.CAMERAS[view]
            direction_text=f"The head turns {direction}." if view not in ("front","rear") else ""
            composition=str(rng.choice(("center","center","center","left","right","upper","lower","upper_left","lower_right")))
            prompt=(f"{STYLE_TEXT[family]} Use {SUBSTYLES[family][subindex]}. {camera} {subject} {direction_text} "
                    "Pure black marks on empty pure white background. One cat only. "
                    "No moon, stars, halo, emblem, frame, text, floor, scenery or other objects. Preserve all cat anatomy.")
            requested_orientation=direction if view not in ("front","rear") else f"{view}_facing"
            result.append(base.Spec(offset+len(result),int(rng.integers(0,2**63-1)),pose,view,morph,tail,
                                    family+"__"+subtype,requested_orientation,composition,extent,int(background),None,
                                    prompt,geometry,"reference_extent"))
            if len(result)==total:break
    return result


def _clustered_threshold(side=4):
    yy,xx=np.mgrid[:side,:side]
    distance=(xx-(side-1)/2)**2+(yy-(side-1)/2)**2
    order=np.argsort(distance.ravel(),kind="stable")
    rank=np.empty(side*side,dtype=float);rank[order]=np.arange(side*side)
    return (rank.reshape(side,side)+.5)/(side*side)


def _lattice(fore, spec):
    """Cut only fully inscribed triangular windows, preserving feline contour.

    This constructive rendering is expressly part of the requested lattice
    style. It never adds marks outside the independently generated cat.
    """
    output=fore.copy()
    x0,y0,x1,y1=base.bbox(fore)
    yy,xx=np.mgrid[:224,:224]
    if spec.view in ("front","rear"):
        protected=yy<y0+.28*(y1-y0)
    elif spec.orientation=="leftward":
        protected=(xx<x0+.27*(x1-x0))&(yy<y0+.66*(y1-y0))
    else:
        protected=(xx>x1-.27*(x1-x0))&(yy<y0+.66*(y1-y0))
    distance=ndimage.distance_transform_edt(fore)
    side=(28,34,40)[spec.seed%3]
    height=side*np.sqrt(3)/2
    strut=(3.,4.,5.)[(spec.seed//3)%3]
    cut=0
    for row in range(-1,int(224/height)+2):
        for col in range(-1,int(224/side)+2):
            left=col*side+(side/2 if row%2 else 0)
            top=row*height
            triangles=(((left,top),(left+side,top),(left+side/2,top+height)),
                       ((left+side,top),(left+1.5*side,top+height),(left+side/2,top+height)))
            for points in triangles:
                v=np.asarray(points,float);centre=v.mean(axis=0)
                v=centre+(v-centre)*(1-2*strut/side)
                patch=Image.new("1",(224,224),0)
                ImageDraw.Draw(patch).polygon([tuple(p) for p in v],fill=1)
                cells=np.asarray(patch,dtype=bool)
                if cells.sum()<65 or protected[cells].any():continue
                if np.all(distance[cells]>=2):
                    output[cells]=False;cut+=1
    return output,cut


def _details(pixels,spec,details):
    fore=pixels!=spec.background
    if not fore.any():return details,np.zeros(4)
    x0,y0,x1,y1=base.bbox(fore);w,h=x1-x0,y1-y0
    black=pixels==0
    filled=ndimage.binary_fill_holes(fore)
    holes=filled&~fore
    hole_labels,hole_count=ndimage.label(holes,structure=base.CONNECTIVITY)
    hole_areas=np.bincount(hole_labels.ravel());hole_areas[0]=0
    _,components=ndimage.label(fore,structure=base.CONNECTIVITY)
    details.update(resized_bbox=[x0,y0,x1,y1],foreground_width=w,foreground_height=h,
                   foreground_fraction=float(fore.mean()),black_fraction=float(black.mean()),
                   interior_hole_count=int((hole_areas>=3).sum()),interior_hole_fraction=float(holes.sum()/max(1,filled.sum())),
                   final_ink_components=int(components),background=spec.background,forced_border_pixels=0,
                   hard_edge_left=float(black[:,0].mean()),hard_edge_right=float(black[:,-1].mean()),
                   hard_edge_top=float(black[0].mean()),hard_edge_bottom=float(black[-1].mean()),
                   foreground_boundary_contact=bool(fore[0].any() or fore[-1].any() or fore[:,0].any() or fore[:,-1].any()))
    return details,np.array([w/h,max(w,h),(x0+w/2)/224,(y0+h/2)/224],float)


def binarize(image,spec):
    family=_family(spec)
    gray=np.asarray(image.convert("L"))
    threshold=float(np.clip(threshold_otsu(gray),130,200))
    raw=gray<threshold
    labels,count=ndimage.label(raw,structure=base.CONNECTIVITY)
    areas=np.bincount(labels.ravel());areas[0]=0
    minimum_component=1 if family=="halftone" else (2 if family in ("binary_pen","woodcut") else 6)
    keep=np.isin(labels,np.flatnonzero(areas>=minimum_component))
    details=source_plan(spec)
    if not keep.any():
        return np.full((224,224),spec.background,np.uint8),1-float(spec.background),"blank",np.zeros(4),details,keep.astype(np.uint8)
    x0,y0,x1,y1=base.bbox(keep);w,h=x1-x0,y1-y0
    issue="source_clipped" if min(x0,y0,gray.shape[1]-x1,gray.shape[0]-y1)<2 else ""
    scale=EXTENTS[spec.action_variant]/max(w,h)
    dims=(max(1,round(w*scale)),max(1,round(h*scale)))
    coverage=np.asarray(Image.fromarray(keep[y0:y1,x0:x1].astype(np.uint8)*255).resize(dims,Image.Resampling.LANCZOS))
    ax,ay=base.ANCHORS[spec.composition]
    ox,oy=round((224-dims[0])*ax),round((224-dims[1])*ay)
    fore=np.zeros((224,224),bool)
    fore[oy:oy+dims[1],ox:ox+dims[0]]=coverage>=128
    details.update(source_bbox=[x0,y0,x1,y1],isotropic_scale=float(scale),
                   minimum_source_component_pixels=minimum_component,removed_source_pixels=int(raw.sum()-keep.sum()),
                   retained_source_components=int((areas>=minimum_component).sum()),
                   source_main_component_share=float(areas.max()/keep.sum()),
                   target_foreground_fraction=None)
    if family=="halftone":
        crop=gray[y0:y1,x0:x1]
        mid_fraction=float(((crop>48)&(crop<208)).mean())
        details["source_gray_mid_fraction"]=mid_fraction
        if mid_fraction>.12:
            # Encode source tonal information only within its own cat extent.
            # Pure white background stays pure white; no independent noise.
            gray_crop=np.asarray(Image.fromarray(crop).resize(dims,Image.Resampling.LANCZOS))
            tone=np.ones((224,224),float)
            tone[oy:oy+dims[1],ox:ox+dims[0]]=gray_crop/255.
            rank=np.tile(_clustered_threshold(4),(56,56))
            fore=(1-tone)>rank
            details["halftone_encoding"]="deterministic_4x4_clustered_dots_of_source_gray"
        else:
            details["halftone_encoding"]="source_binary_dots_preserved"
    if family=="triangular_lattice" and fore.any():
        fore,cut=_lattice(fore,spec)
        details.update(triangle_cells_cut=int(cut),triangle_renderer="inscribed_voids_in_generated_cat_ink; outer anatomy preserved")
    grid=details["pixel_art_grid"]
    if grid:
        low=Image.fromarray(fore.astype(np.uint8)).resize((grid,grid),Image.Resampling.NEAREST)
        fore=np.asarray(low.resize((224,224),Image.Resampling.NEAREST),dtype=bool)
        details["pixel_block_side"]=224//grid
    pixels=np.full((224,224),spec.background,np.uint8);pixels[fore]=1-spec.background
    details,geometry=_details(pixels,spec,details)
    fraction=float((pixels==0).mean())
    if not .05<=fraction<=.97:issue=issue or "black_fraction_out_of_range"
    return pixels,fraction,issue,geometry,details,keep.astype(np.uint8)


def _style_view(pixels,background):
    fore=pixels!=background
    if not fore.any():return Image.fromarray(pixels*255).convert("RGB")
    x0,y0,x1,y1=base.bbox(fore)
    crop=Image.fromarray(pixels[y0:y1,x0:x1]*255)
    scale=208/max(crop.size)
    crop=crop.resize(tuple(max(1,round(v*scale)) for v in crop.size),Image.Resampling.NEAREST)
    canvas=Image.new("L",(224,224),background*255)
    canvas.paste(crop,((224-crop.width)//2,(224-crop.height)//2))
    return canvas.convert("RGB")


class LocalModels(base.LocalModels):
    def __init__(self):
        super().__init__()
        texts=PORTRAIT_POSITIVES+PORTRAIT_NEGATIVES+EXTRA_CAT_POSITIVES+tuple(STYLE_PROTOTYPES[s] for s in STYLE_FAMILIES)+OLD_STYLE_PROTOTYPES
        with self.torch.inference_mode():
            tokens=self.processor(text=list(texts),padding="max_length",return_tensors="pt").to("cuda")
            self.reference_text_features=self.normalize(self.siglip.get_text_features(**tokens).float())

    def generate(self,specifications):
        torch=self.torch
        generators=[torch.Generator("cpu").manual_seed(s.seed) for s in specifications]
        with torch.inference_mode():
            images=self.pipe([s.prompt for s in specifications],width=512,height=512,generator=generators,
                             num_inference_steps=base.STEPS,guidance_scale=0.,max_sequence_length=256).images
        processed=[binarize(image,s) for image,s in zip(images,specifications)]
        n=len(specifications)
        semantic=[base.semantic_image(r[0],s.background) for r,s in zip(processed,specifications)]
        style_views=[_style_view(r[0],s.background) for r,s in zip(processed,specifications)]
        with torch.inference_mode():
            inputs=self.processor(images=semantic+style_views,return_tensors="pt").to("cuda")
            features=self.normalize(self.siglip.get_image_features(**inputs).float())
            base_scores=(features[:n]@self.text_features.T*self.logit_scale+self.logit_bias).cpu().numpy()
            ref_scores=(features@self.reference_text_features.T*self.logit_scale+self.logit_bias).cpu().numpy()
            embeddings=features[:n].cpu().numpy()
        p=len(PORTRAIT_POSITIVES);a=len(PORTRAIT_NEGATIVES);e=len(EXTRA_CAT_POSITIVES)
        style_start=p+a+e
        samples=[]
        for i,(spec,result) in enumerate(zip(specifications,processed)):
            pixels,fraction,issue,geometry,details,mask=result
            if spec.pose=="portrait":
                cat=float(ref_scores[i,:p].max());negative=ref_scores[i,p:p+a];alternatives=PORTRAIT_NEGATIVES
                semantic_protocol="cat_portrait_vs_other_animal_portraits"
                threshold_logit,threshold_margin=-6.,2.
                screen_protocol="portrait_margin_v2"
            else:
                cat=float(max(base_scores[i,:len(base.CAT_TEXT)].max(),ref_scores[i,p+a:p+a+e].max()))
                negative=base_scores[i,len(base.CAT_TEXT):len(base.CAT_TEXT)+len(base.ALTERNATIVES)];alternatives=base.ALTERNATIVES
                semantic_protocol="whole_domestic_cat_with_style_positive"
                threshold_logit,threshold_margin=-3.,0.
                screen_protocol="whole_body_absolute_and_margin_v1"
            # Scores from both material polarities are not thresholded differently.
            # Use the better canonical/original-polarity score for each style,
            # which describes rendering rather than hard/soft material identity.
            style_scores=np.maximum(ref_scores[i,style_start:style_start+9],ref_scores[n+i,style_start:style_start+9])
            old_scores=np.maximum(ref_scores[i,style_start+9:],ref_scores[n+i,style_start+9:])
            requested=float(style_scores[STYLE_FAMILIES.index(_family(spec))])
            best_new=float(style_scores.max());best_old=float(old_scores.max())
            details.update(semantic_protocol=semantic_protocol,
                           semantic_threshold_logit=threshold_logit,semantic_threshold_margin=threshold_margin,
                           semantic_screen_protocol=screen_protocol,
                           style_scores={k:float(v) for k,v in zip(STYLE_FAMILIES,style_scores)},
                           old_style_scores={str(j):float(v) for j,v in enumerate(old_scores)},
                           old_style_descriptions=list(OLD_STYLE_PROTOTYPES),
                           requested_style_score=requested,best_new_style_score=best_new,best_old_style_score=best_old,
                           requested_style_margin_vs_old=requested-best_old,
                           best_new_style_margin_vs_old=best_new-best_old,
                           best_matching_reference_style=STYLE_FAMILIES[int(style_scores.argmax())],
                           style_screen_note="Raw SigLIP comparisons, not a guarantee of visual style adherence")
            strongest=int(negative.argmax())
            buffer=io.BytesIO();base.save_png(pixels,buffer);buffer.seek(0)
            if not np.array_equal(base.load_png(buffer),pixels):raise ValueError("Binary PNG roundtrip failed")
            samples.append(base.Sample(spec,pixels,embeddings[i],cat,cat-float(negative[strongest]),alternatives[strongest],
                                       base_scores[i,len(base.CAT_TEXT)+len(base.ALTERNATIVES):],fraction,issue,geometry,details,mask))
        return samples


def rejection(sample):
    p=sample.pixels;d=sample.diagnostics;family=_family(sample.spec)
    if p.shape!=(224,224) or p.dtype!=np.uint8 or not np.isin(p,[0,1]).all():return "invalid_array"
    if not .05<=float((p==0).mean())<=.97:return "black_fraction_out_of_range"
    if sample.issue:return sample.issue
    if max(d.get("foreground_width",0),d.get("foreground_height",0))<112:return "cat_too_small"
    if not np.isfinite(sample.cat_logit) or not np.isfinite(sample.cat_margin):return "invalid_semantic_score"
    # Pilot-v1 showed clear pixel portraits at -4 logits but +4 to +7 species
    # margins. Portraits use their separate text prototypes and require a
    # stronger relative species margin; raw scores and both thresholds are
    # recorded. This is a screening rule, never a substitute for visual review.
    threshold_logit,threshold_margin=(-6.,2.) if sample.spec.pose=="portrait" else (-3.,0.)
    if sample.cat_logit < threshold_logit or sample.cat_margin < threshold_margin:return "low_cat_score"
    if not np.isfinite(d.get("requested_style_score",np.nan)):return "missing_style_scores"
    if d["best_new_style_margin_vs_old"] < -.35:return "old_style_dominates"
    if d["requested_style_margin_vs_old"] < -1.25:return "requested_style_weak"
    grid=d.get("pixel_art_grid",0)
    if grid:
        k=224//grid
        if not np.array_equal(p,np.repeat(np.repeat(p[::k,::k],k,0),k,1)):return "pixel_grid_failed"
    if family=="triangular_lattice" and d.get("triangle_cells_cut",0)<2 and d.get("interior_hole_count",0)<3:
        return "insufficient_lattice_structure"
    if family=="halftone" and d.get("halftone_encoding")=="source_binary_dots_preserved" and d.get("final_ink_components",0)<12 and d.get("interior_hole_count",0)<12:
        return "halftone_detail_missing"
    return ""


def variants_for_sample(sample,count=1):
    if count<1:raise ValueError("count must be positive")
    # Exactly one candidate, irrespective of caller's older three-variant default.
    return [sample]


def iter_acquire(mode,total,work_dir,seed=MASTER_SEED,batch_size=4):
    specifications=specs_for(mode,total,seed)
    cache=cache_dir(mode,work_dir,seed,batch_size);cache.mkdir(parents=True,exist_ok=True)
    models=None;passed=0
    for start in range(0,total,batch_size):
        batch=specifications[start:start+batch_size];path=cache/f"batch_{start:05d}_{len(batch)}.npz"
        if path.exists():
            samples=base.read_batch(path)
            if [s.spec for s in samples]!=batch:raise ValueError(f"Source cache mismatch: {path}")
        else:
            if models is None:models=LocalModels()
            samples=models.generate(batch)
            for sample in samples:sample.diagnostics["generation_master_seed"]=int(seed)
            base.cache_batch(path,samples)
        passed+=sum(not rejection(s) for s in samples)
        if (start+len(batch))%20==0 or start+len(batch)==total:
            print(f"reference {mode}: {start+len(batch)}/{total}; {passed} pass automatic cat/style/binary screening",flush=True)
        yield samples


def acquire(mode,total,work_dir,seed=MASTER_SEED,batch_size=4):
    return [sample for batch in iter_acquire(mode,total,work_dir,seed,batch_size) for sample in batch]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode",choices=("pilot","full"))
    parser.add_argument("--count",type=int,default=144)
    parser.add_argument("--seed",type=int,default=MASTER_SEED)
    parser.add_argument("--batch-size",type=int,default=4)
    parser.add_argument("--work-dir",type=Path,required=True)
    args=parser.parse_args()
    samples=acquire(args.mode,args.count,args.work_dir,args.seed,args.batch_size)
    print(json.dumps(dict(sources=len(samples),automatic_pass=sum(not rejection(s) for s in samples),
                          requested_styles=dict(Counter(_family(s.spec) for s in samples)),
                          rejections=dict(Counter(rejection(s) for s in samples if rejection(s)))),indent=2))


if __name__=="__main__":main()
