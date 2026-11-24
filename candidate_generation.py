"""Independent cat candidates for finite two-material specimen design.

This module is intentionally self-contained so it can be included in the final
standalone dataset generator. It does not perform FEM or final dataset selection.
Every candidate has a unique diffusion seed and produces one material map.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import tempfile
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage
from skimage.filters import threshold_otsu

SIZE = 224
DIFFUSION_SIZE = 512
STEPS = 9
MASTER_SEED = 1142114
PIPELINE_VERSION = 6
GENERATOR = ("Tongyi-MAI/Z-Image-Turbo", "f332072aa78be7aecdf3ee76d5c247082da564a6")
SIGLIP = ("google/siglip-base-patch16-224", "7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed")
SOURCE = f"{GENERATOR[0]}@{GENERATOR[1]}"
CONNECTIVITY = np.ones((3, 3), dtype=bool)
CAT_TEXT = (
    "a black and white drawing of one whole domestic cat",
    "a simple black ink illustration of a full body pet cat",
    "a full-body silhouette of a domestic cat",
)
ALTERNATIVES = (
    "a cat face or cat head without a body",
    "a drawing of more than one cat",
    "a black and white drawing of a dog",
    "a drawing of a tiger or other wild cat",
    "a black and white drawing of a fox",
    "a black and white drawing of a rabbit",
    "a black and white drawing of a squirrel",
    "an abstract or unrecognizable black and white symbol",
    "black writing and lettering on a white page",
    "a framed emblem, badge or logo with a cat inside",
    "a drawing of a kitten with extra or malformed legs",
)
VIEW_TEXT = (
    "a full-body drawing of a cat viewed from the side in profile",
    "a full-body drawing of a cat seen from the front, facing the viewer head on",
    "a full-body drawing of a cat seen from behind, with its back to the viewer",
    "a full-body drawing of a cat in a three-quarter view at an oblique angle",
    "a full-body drawing of a cat from a top-down overhead bird's-eye view",
)
POSE_TEXT = (
    "a cat sitting upright on its haunches", "a cat standing on all four legs",
    "a cat walking with one paw raised", "a cat running in a fast gallop with extended legs",
    "a cat jumping through the air", "a cat sleeping with its head resting down",
    "a cat lying down on its belly or side", "a cat stretching with its front legs extended and hips high",
    "a cat crouching low ready to pounce", "a playful cat on its back with paws raised or rearing to swat the air",
    "a cat curled up in a ball", "a cat looking backward over its shoulder",
)


@dataclass(frozen=True)
class Spec:
    candidate: int
    seed: int
    pose: str
    view: str
    morphology: str
    tail: str
    style: str
    orientation: str
    composition: str
    action_variant: int
    background: int
    target_fraction: float
    prompt: str
    geometry_family: str
    extent_mode: str


@dataclass
class Sample:
    spec: Spec
    pixels: np.ndarray
    embedding: np.ndarray
    cat_logit: float
    cat_margin: float
    alternative: str
    view_scores: np.ndarray
    black_fraction: float
    issue: str
    geometry: np.ndarray
    diagnostics: dict
    source_mask: np.ndarray


# These strata change anatomy and contours in the generative prompt. No image
# is stretched, rotated, mirrored, or multiplied to fill the strata.
GEOMETRY_FAMILIES = (
    ("tall_reach", "stretching", ("front", "rear", "side", "three_quarter"),
     "standing tall on its hind paws with both forelegs extended high above the head; an elongated upright torso connects broad hips and shoulders; the whole figure is distinctly taller than wide"),
    ("long_recline", "lying", ("side", "top_down", "three_quarter"),
     "reclining at full length with a long continuous broad torso, front paws extended far ahead and rear paws pointing backward; a distinctly horizontal, low figure"),
    ("compact_coil", "curling", ("top_down", "side", "three_quarter"),
     "curled tightly into a nearly round ball with head tucked against haunches and a thick tail encircling the body; pointed ears and a cat face remain clear"),
    ("radial_sprawl", "playing", ("top_down", "front", "three_quarter"),
     "sprawled on its back with belly visible and four broad bent paws pointing in different directions around a short central torso"),
    ("arched_bridge", "stretching", ("side", "three_quarter"),
     "arching its back into a pronounced high bridge, long forelegs straight down and bent hind legs well separated from the front pair; the space below the belly is wide and open"),
    ("pear_sit", "sitting", ("front", "rear", "three_quarter"),
     "sitting upright with very wide rounded haunches, a narrow waist, compact shoulders and a small triangular cat head, both forepaws visible together below the chest"),
    ("long_spine", "standing", ("top_down", "rear", "three_quarter"),
     "seen along a long gently S-curved spine, broad shoulders connected to narrower hips, all four short paws visible beside the body; the head is far from the tail base"),
    ("wide_crouch", "crouching", ("front", "three_quarter", "side"),
     "crouching extremely low with wide powerful shoulders, head close to the ground and broad haunches spread sideways; a wide squat outline with short thick paws"),
    ("diagonal_pounce", "jumping", ("side", "three_quarter"),
     "pouncing along a steep diagonal, long muscular torso extended between the forward forepaws and backward rear paws, head leading the gesture"),
    ("open_curl", "looking_back", ("top_down", "side", "three_quarter"),
     "bending its long body into an open C curve, turning its head backward toward its own flank while the forepaws and hind paws remain clearly separated"),
    ("broad_chest", "sitting", ("front", "rear", "three_quarter"),
     "sitting upright with a very broad heavy chest, narrow long lower forelegs and compact haunches, an oversized rounded cat head and two tall pointed ears"),
    ("low_stride", "walking", ("side", "three_quarter"),
     "taking a long low step with a horizontal deep torso, narrow waist, broad thighs and four clearly readable paws separated along the ground line, with no drawn ground"),
)
STYLES = {
    "paper_cut": "A clean cut-paper cat illustration with broad connected black regions and a few large white anatomical details.",
    "rounded_cartoon": "A simple rounded cartoon cat, bold smooth filled shapes, expressive eyes, clean closed paws.",
    "angular_graphic": "A clean angular graphic cat, broad geometric shoulders and haunches, connected wedge-shaped legs, clearly pointed ears.",
    "solid_silhouette": "A recognizable full-body domestic cat silhouette with a clear feline head, pointed ears, complete paws and a tail; only small white eyes and a muzzle detail.",
    "bold_ink": "A bold black ink illustration of a domestic cat with connected smooth body shapes and a few broad white curves distinguishing its limbs.",
    "soft_folk": "A simple folk-art domestic cat with large smooth rounded body regions, an expressive pointed-eared cat head, complete rounded paws, no ornaments.",
}
MORPHS = (
    "a slender adult with a long narrow waist and large paws",
    "a stocky cat with broad ribs and short thick legs",
    "a fluffy cat with a broad chest ruff and small face",
    "a long-legged cat with a small wedge-shaped head",
    "a kitten with a very large round head and compact body",
    "a muscular adult with narrow shoulders and heavy haunches",
    "a round-bellied cat with short paws and large ears",
    "a smooth long-bodied cat with narrow hips and a broad head",
)
TAILS = (
    "a thick tail held straight upward", "a long tail reaching horizontally behind the hips",
    "a broad tail wrapped close around the haunches", "a short rounded bobtail",
    "a long tail curling in a wide open arc", "a narrow tail bent close along the flank",
    "a fluffy tail sweeping low beside the paws", "a broad tapering tail pointing diagonally upward",
)
ANCHORS = {
    "center": (.5, .5), "upper_left": (0., 0.), "upper_right": (1., 0.),
    "lower_left": (0., 1.), "lower_right": (1., 1.), "left": (0., .5),
    "right": (1., .5), "upper": (.5, 0.), "lower": (.5, 1.),
}
CAMERAS = {
    "front": "Front view: face and chest toward the viewer.",
    "rear": "Rear view: back and hindquarters toward the viewer, pointed ears visible above the head.",
    "side": "Side profile: head, chest, belly and hindquarters clearly separated in profile.",
    "three_quarter": "Three-quarter view with clearly readable feline head and body anatomy.",
    "top_down": "Directly overhead view: the top of the head, shoulders, back, hips and paws are visible in the image plane.",
}


def specs_for(mode: str, total: int, master_seed: int = MASTER_SEED) -> list[Spec]:
    """Prefix-stable independent seeds, with balanced 48-candidate blocks."""
    rng = np.random.default_rng([master_seed, 61 if mode == "pilot" else 62])
    specifications = []
    block_size = 48
    while len(specifications) < total:
        families = rng.permutation(np.tile(np.arange(len(GEOMETRY_FAMILIES)), 4))
        polarities = rng.permutation(np.tile([0, 1], block_size // 2))
        modes = rng.permutation(np.array(["max_extent"] * 12 + ["fraction"] * 36))
        style_ids = rng.permutation(np.tile(np.arange(len(STYLES)), block_size // len(STYLES)))
        # Uniform foreground targets span broad low/high hard fractions once polarity is assigned.
        targets = rng.permutation(np.linspace(.055, .475, block_size) + rng.uniform(-.004, .004, block_size))
        for j in range(block_size):
            family, pose, views, anatomy = GEOMETRY_FAMILIES[int(families[j])]
            view = views[int(rng.integers(len(views)))]
            style = tuple(STYLES)[int(style_ids[j])]
            morph = MORPHS[int(rng.integers(len(MORPHS)))]
            tail = TAILS[int(rng.integers(len(TAILS)))]
            composition = tuple(ANCHORS)[int(rng.integers(len(ANCHORS)))]
            direction = ("leftward", "rightward")[int(rng.integers(2))]
            direction_text = f"The head points {direction}." if view not in ("front", "rear") else ""
            prompt = (
                f"{STYLES[style]} {CAMERAS[view]} One {morph}, {anatomy}, with {tail}. "
                f"{direction_text} One complete cat with entire ears, head, body, all visible paws and tail inside the image. "
                "Make the specified body proportions and pose obvious. Solid black ink on an empty pure white background. "
                "Clean unbroken anatomy, no cut-off limbs. No frame, lettering, symbols, furniture, floor, shadows, pattern or other objects."
            )
            specifications.append(Spec(
                len(specifications), int(rng.integers(0, 2**63 - 1)), pose, view, morph, tail,
                style, direction, composition, int(families[j]), int(polarities[j]), float(targets[j]),
                prompt, family, str(modes[j]),
            ))
            if len(specifications) == total:
                break
    return specifications


def bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def canonical(pixels: np.ndarray, background: int) -> np.ndarray:
    """Rendering-only normalization; never infer phase polarity from boundary pixels."""
    return pixels if background == 1 else 1 - pixels


def semantic_image(pixels: np.ndarray, background: int) -> Image.Image:
    """Object-normalized semantic view; physical coordinates remain in pixels."""
    normalized = canonical(pixels, background)
    ink = normalized == 0
    if not ink.any():
        return Image.fromarray(normalized * 255).convert("RGB")
    x0, y0, x1, y1 = bbox(ink)
    cropped = Image.fromarray(normalized[y0:y1, x0:x1] * 255)
    scale = 208 / max(cropped.size)
    cropped = cropped.resize(tuple(max(1, round(v * scale)) for v in cropped.size), Image.Resampling.NEAREST)
    canvas = Image.new("L", (SIZE, SIZE), 255)
    canvas.paste(cropped, ((SIZE-cropped.width)//2, (SIZE-cropped.height)//2))
    return canvas.convert("RGB")


def frame_mask(keep: np.ndarray, spec: Spec) -> tuple[np.ndarray, dict, np.ndarray]:
    """Isotropic, uncropped framing; no forced border and no fabricated geometry."""
    x0, y0, x1, y1 = bbox(keep)
    width, height = x1-x0, y1-y0
    maximum_scale = SIZE / max(width, height)
    desired_scale = np.sqrt(spec.target_fraction * SIZE * SIZE / keep.sum())
    scale = maximum_scale if spec.extent_mode == "max_extent" else min(maximum_scale, desired_scale)
    dims = (max(1, min(SIZE, round(width*scale))), max(1, min(SIZE, round(height*scale))))
    coverage = Image.fromarray(keep[y0:y1, x0:x1].astype(np.uint8)*255).resize(dims, Image.Resampling.LANCZOS)
    foreground = np.asarray(coverage) >= 128
    h, w = foreground.shape
    ax, ay = ANCHORS[spec.composition]
    ox, oy = round((SIZE-w)*ax), round((SIZE-h)*ay)
    pixels = np.full((SIZE, SIZE), spec.background, dtype=np.uint8)
    pixels[oy:oy+h, ox:ox+w][foreground] = 1-spec.background
    black = pixels == 0
    fore = pixels != spec.background
    details = {
        "source_bbox": [x0, y0, x1, y1], "resized_bbox": [ox, oy, ox+w, oy+h],
        "isotropic_scale": float(scale), "target_foreground_fraction": spec.target_fraction,
        "foreground_fraction": float(fore.mean()), "black_fraction": float(black.mean()),
        "background": spec.background, "forced_border_pixels": 0,
        "hard_edge_left": float(black[:, 0].mean()), "hard_edge_right": float(black[:, -1].mean()),
        "hard_edge_top": float(black[0].mean()), "hard_edge_bottom": float(black[-1].mean()),
        "foreground_width": w, "foreground_height": h,
        "foreground_boundary_contact": bool(fore[0].any() or fore[-1].any() or fore[:,0].any() or fore[:,-1].any()),
        "target_was_limited_by_extent": bool(desired_scale > maximum_scale),
    }
    geometry = np.array([w/h, max(w,h), (ox+w/2)/SIZE, (oy+h/2)/SIZE], dtype=float)
    return pixels, details, geometry


def binarize(image: Image.Image, spec: Spec):
    gray = np.asarray(image.convert("L"))
    threshold = float(np.clip(threshold_otsu(gray), 130, 195))
    raw = gray < threshold
    labels, count = ndimage.label(raw, structure=CONNECTIVITY)
    if count == 0:
        return np.full((SIZE,SIZE), spec.background, np.uint8), 1-float(spec.background), "blank", np.zeros(4), {"threshold": threshold}, raw.astype(np.uint8)
    areas = np.bincount(labels.ravel()); areas[0] = 0
    main = labels == int(areas.argmax())
    x0,y0,x1,y1 = bbox(main)
    reach = max(10, round(max(x1-x0,y1-y0)*.06))
    nearby = np.unique(labels[max(0,y0-reach):y1+reach, max(0,x0-reach):x1+reach])
    keep = np.isin(labels, nearby[areas[nearby] >= 14])
    x0,y0,x1,y1 = bbox(keep)
    issue = ""
    if min(x0,y0,gray.shape[1]-x1,gray.shape[0]-y1) < 2:
        issue = "source_clipped"
    elif areas.max()/keep.sum() < .72:
        issue = "fragmented"
    pixels, details, geometry = frame_mask(keep, spec)
    details.update(threshold=threshold, source_main_component_share=float(areas.max()/keep.sum()),
                   removed_source_pixels=int(raw.sum()-keep.sum()))
    fraction = float((pixels == 0).mean())
    if not .05 <= fraction <= .95:
        issue = issue or "black_fraction_out_of_range"
    return pixels, fraction, issue, geometry, details, keep.astype(np.uint8)


def save_png(pixels: np.ndarray, destination) -> None:
    Image.fromarray(pixels.astype(bool)).save(destination, format="PNG")


def load_png(path) -> np.ndarray:
    with Image.open(path) as png:
        if png.format != "PNG" or png.mode != "1" or png.size != (SIZE,SIZE):
            raise ValueError(f"Expected one-bit {SIZE}x{SIZE} PNG: {path}")
        return np.asarray(png, dtype=np.uint8)


class LocalModels:
    def __init__(self):
        import torch
        import torch.nn.functional as functional
        from diffusers import ZImagePipeline
        from huggingface_hub import snapshot_download
        from transformers import AutoProcessor, SiglipModel
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("Generation requires a CUDA GPU that supports bfloat16.")
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        self.torch = torch
        self.normalize = functional.normalize
        model_path = snapshot_download(GENERATOR[0], revision=GENERATOR[1], allow_patterns=[
            "model_index.json", "scheduler/*", "text_encoder/*", "tokenizer/*", "transformer/*", "vae/*"])
        self.pipe = ZImagePipeline.from_pretrained(model_path, torch_dtype=torch.bfloat16, use_safetensors=True).to("cuda")
        self.pipe.set_progress_bar_config(disable=True)
        self.pipe.vae.enable_slicing()
        self.pipe.vae.to(memory_format=torch.channels_last)
        self.processor = AutoProcessor.from_pretrained(SIGLIP[0], revision=SIGLIP[1], use_fast=False)
        self.siglip = SiglipModel.from_pretrained(SIGLIP[0], revision=SIGLIP[1], dtype=torch.float16, use_safetensors=True).to("cuda").eval()
        with torch.inference_mode():
            tokens = self.processor(text=list(CAT_TEXT+ALTERNATIVES+VIEW_TEXT+POSE_TEXT), padding="max_length", return_tensors="pt").to("cuda")
            self.text_features = self.normalize(self.siglip.get_text_features(**tokens).float())
            self.logit_scale = self.siglip.logit_scale.float().exp()
            self.logit_bias = self.siglip.logit_bias.float()

    def generate(self, specifications: list[Spec]) -> list[Sample]:
        torch = self.torch
        generators = [torch.Generator("cpu").manual_seed(spec.seed) for spec in specifications]
        with torch.inference_mode():
            images = self.pipe([spec.prompt for spec in specifications], width=DIFFUSION_SIZE, height=DIFFUSION_SIZE,
                               generator=generators, num_inference_steps=STEPS, guidance_scale=0., max_sequence_length=256).images
        processed = [binarize(image,spec) for image,spec in zip(images,specifications)]
        semantic = [semantic_image(result[0],spec.background) for result,spec in zip(processed,specifications)]
        with torch.inference_mode():
            inputs = self.processor(images=semantic,return_tensors="pt").to("cuda")
            features = self.normalize(self.siglip.get_image_features(**inputs).float())
            scores = features @ self.text_features.T * self.logit_scale + self.logit_bias
            embeddings,logits = features.cpu().numpy(), scores.cpu().numpy()
        samples = []
        for i,(spec,result) in enumerate(zip(specifications,processed)):
            pixels,fraction,issue,geometry,details,source_mask = result
            png = io.BytesIO(); save_png(pixels,png); png.seek(0)
            if not np.array_equal(load_png(png),pixels):
                raise ValueError(f"PNG roundtrip failed for {spec.candidate}")
            cat_logit = float(logits[i,:len(CAT_TEXT)].max())
            other = logits[i,len(CAT_TEXT):len(CAT_TEXT)+len(ALTERNATIVES)]
            strongest = int(other.argmax())
            samples.append(Sample(spec,pixels,embeddings[i],cat_logit,cat_logit-float(other[strongest]),
                ALTERNATIVES[strongest],logits[i,len(CAT_TEXT)+len(ALTERNATIVES):],fraction,issue,geometry,details,source_mask))
        return samples


def rejection(sample: Sample) -> str:
    if sample.pixels.shape != (SIZE,SIZE) or sample.pixels.dtype != np.uint8 or not np.isin(sample.pixels,[0,1]).all():
        return "invalid_array"
    if not .05 <= float((sample.pixels==0).mean()) <= .95:
        return "black_fraction_out_of_range"
    if sample.issue:
        return sample.issue
    if not np.isfinite(sample.cat_logit) or not np.isfinite(sample.cat_margin):
        return "invalid_semantic_score"
    if sample.cat_logit < -3 or sample.cat_margin < 0:
        return "low_cat_score"
    return ""


def fingerprint(pixels: np.ndarray) -> str:
    return hashlib.sha256(pixels.tobytes()).hexdigest()


def cache_batch(path: Path, samples: list[Sample]) -> None:
    rows = [dict(spec=asdict(s.spec),cat_logit=s.cat_logit,cat_margin=s.cat_margin,alternative=s.alternative,
                 black_fraction=s.black_fraction,issue=s.issue,diagnostics=s.diagnostics) for s in samples]
    with tempfile.NamedTemporaryFile(dir=path.parent,prefix=path.stem,suffix=".tmp",delete=False) as tmp:
        pending=Path(tmp.name)
    try:
        with pending.open("wb") as stream:
            np.savez_compressed(stream,rows=json.dumps(rows),pixels=np.stack([s.pixels for s in samples]),
                embeddings=np.stack([s.embedding for s in samples]),views=np.stack([s.view_scores for s in samples]),
                geometry=np.stack([s.geometry for s in samples]),source_masks=np.stack([s.source_mask for s in samples]))
        os.replace(pending,path)
    finally:
        pending.unlink(missing_ok=True)


def read_batch(path: Path) -> list[Sample]:
    with np.load(path,allow_pickle=False) as data:
        rows=json.loads(data["rows"].item())
        return [Sample(Spec(**row["spec"]),pixels,embedding,row["cat_logit"],row["cat_margin"],row["alternative"],
                       views,row["black_fraction"],row["issue"],geometry,row["diagnostics"],source_mask)
                for row,pixels,embedding,views,geometry,source_mask in zip(rows,data["pixels"],data["embeddings"],
                    data["views"],data["geometry"],data["source_masks"])]


def acquire(mode: str,total: int,work_dir: Path,seed: int=MASTER_SEED,batch_size: int=4) -> list[Sample]:
    specifications=specs_for(mode,total,seed)
    cache=Path(work_dir)/f"v{PIPELINE_VERSION}-{mode}-{seed}-b{batch_size}"
    cache.mkdir(parents=True,exist_ok=True)
    models=None
    samples=[]
    for start in range(0,total,batch_size):
        batch=specifications[start:start+batch_size]
        path=cache/f"batch_{start:05d}_{len(batch)}.npz"
        if path.exists():
            results=read_batch(path)
            if [s.spec for s in results]!=batch:
                raise RuntimeError(f"Cache does not match specifications: {path}")
        else:
            if models is None:
                models=LocalModels()
            results=models.generate(batch)
            cache_batch(path,results)
        samples.extend(results)
        if len(samples)//20 != start//20 or len(samples)==total:
            print(f"{mode}: {len(samples)}/{total}, {sum(not rejection(s) for s in samples)} pass automatic screening",flush=True)
    return samples


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode",choices=["pilot","full"])
    parser.add_argument("--count",type=int,default=80)
    parser.add_argument("--seed",type=int,default=MASTER_SEED)
    parser.add_argument("--batch-size",type=int,default=4)
    parser.add_argument("--work-dir",type=Path,required=True)
    args=parser.parse_args()
    if args.count<=0 or args.batch_size<=0:
        parser.error("count and batch-size must be positive")
    samples=acquire(args.mode,args.count,args.work_dir,args.seed,args.batch_size)
    valid=[s for s in samples if not rejection(s)]
    summary=dict(candidate_count=len(samples),valid_count=len(valid),seed=args.seed,pipeline_version=PIPELINE_VERSION,
                 rejected=dict(Counter(rejection(s) for s in samples if rejection(s))),
                 black_fraction_range=[min(s.black_fraction for s in valid),max(s.black_fraction for s in valid)] if valid else [],
                 exact_duplicate_count=len(valid)-len({fingerprint(s.pixels) for s in valid}),
                 requested_geometry_families=dict(Counter(s.spec.geometry_family for s in valid)),
                 foreground_touches_boundary=sum(s.diagnostics["foreground_boundary_contact"] for s in valid),
                 visual_review="not yet performed",mechanical_validation="not yet performed")
    (args.work_dir/f"{args.mode}_generation_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(json.dumps(summary,indent=2),flush=True)


if __name__=="__main__":
    main()
