"""Independent, mechanically targeted additions to the existing cat collection.

This module never changes the existing dataset. Each diffusion source has one
seed and may be evaluated at up to three framing settings; at most ONE setting
per source may enter a released dataset. Geometry diversity comes from new
independent generations and anatomy/style strata, not these local alternatives.
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

import candidate_generation as base

VERSION = 1
MASTER_SEED = 1143186
TARGET_GROUPS = ("high", "x_strong", "y_strong", "explore")
TARGETS = {
    "high": ((8.7, 8.7), (9.1, 8.7), (8.7, 9.1), (9.3, 9.3), (7.0, 8.7), (8.7, 7.0)),
    "x_strong": ((4.6, 2.45), (5.6, 2.9), (6.4, 3.6), (7.2, 4.5), (8.1, 5.5)),
    "y_strong": ((3.5, 7.5), (4.6, 8.1), (6.8, 8.7), (2.2, 4.8), (3.0, 7.1)),
    "explore": ((5.1, 7.4), (7.4, 5.1), (5.0, 3.4), (3.4, 5.0), (8.2, 8.0), (6.2, 4.0)),
}

# Anatomically meaningful empty regions and connected contours provide structure
# different from only filled white silhouettes. All are requested in the source
# image itself; the preprocessing does not fabricate holes, ribs or patterns.
TARGET_STYLES = {
    "clean_outline": "A simple full-body cat line drawing with a clear connected bold black contour, white interior, two large simple eyes and only a few broad curved lines separating its paws. A consistent medium-width black stroke, no shading, no hatching.",
    "contour_and_fill": "A clean black-and-white cat drawing with an outlined white torso, solid black ears and paws, and a connected thick curved tail. Broad simple anatomical regions; no texture or decorative markings.",
    "hollow_ink": "A full-body domestic cat drawn with thick smooth black ink contours enclosing large clean white anatomical regions. The chest, belly and four paws remain easy to distinguish. No hatching, no texture, no stripes.",
    "sculpted_cutpaper": "A connected cut-paper domestic cat, broad black shoulders and haunches, white negative spaces naturally separating the paws, belly and curved tail. A simple clear feline face and pointed ears. No arbitrary geometric holes.",
    "broad_folk": "A simple folk-art domestic cat with broad smooth connected black anatomy, a few large white anatomical details, rounded paws and a pointed-eared expressive cat head. No ornaments or texture.",
    "angular_contour": "A clean angular domestic cat illustration with a few bold black connected contours and broad filled regions. Triangular ears, wedge-shaped paws, expressive cat eyes, clear full-body anatomy. No separate polygons or decoration.",
}

# Added for the full augmentation after the pilot showed that hollow outlines
# often preserve high stiffness in both directions instead of reaching the
# intended anisotropic wings. Kept separate so pilot prompt sampling is exact.
FILLED_TARGET_STYLES = {
    "filled_silhouette": "A clear full-body domestic cat silhouette made of a solid broad black torso, head, four connected paws and tail. Only small white eyes and a muzzle detail; no outlined or hollow torso.",
    "filled_cartoon": "A simple full-body cartoon cat with a solid black chest, belly, head and connected thick paws. Large pointed ears and small white eyes. Broad continuous filled anatomy, no hollow body or texture.",
    "filled_rounded_ink": "A rounded black ink domestic cat with broad solid connected body regions and clear whole-body anatomy. Only a few small white facial details. Keep the belly and torso filled black.",
    "filled_angular_cutpaper": "A clean angular cut-paper domestic cat with a solid black torso, broad connected shoulders and haunches, complete paws, pointed ears and a tail. Small white eyes only; no internal geometric holes.",
}

# Name, pose, permitted views, anatomical description. None are applied as image
# transforms; they are independent diffusion prompts with independent seeds.
FAMILIES = {
    "folded_loaf": ("lying", ("three_quarter", "side", "front"), "resting with its forepaws folded beneath a broad low torso; its head turns to one side and its thick tail bends back along the flank"),
    "crescent_reach": ("stretching", ("top_down", "three_quarter"), "its long torso curves into a broad open crescent, one forepaw reaching far outward, the other bent close to its chest, hind paws resting at the opposite end of the crescent"),
    "low_arch": ("stretching", ("side", "three_quarter"), "stretching in a low long arch with forepaws extended horizontally, a narrow waist rising to broad haunches, and a tail hooked back toward its shoulders"),
    "side_roll": ("playing", ("top_down", "side", "three_quarter"), "rolling on its side with a long broad torso, two bent forepaws near its face and two hind paws extending in different directions; its body axis is mostly horizontal"),
    "wide_turn": ("looking_back", ("top_down", "three_quarter"), "turning its long torso around a wide corner, its head looking backward, broad hips offset from its shoulders, with all visible paws anatomically connected"),
    "upright_s_curve": ("sitting", ("front", "rear", "three_quarter"), "sitting tall with a gently S-curved upright torso, broad shoulders, a pinched waist and offset rounded haunches; its tail curves beside the body without enclosing a frame"),
    "reaching_offset": ("stretching", ("front", "rear", "three_quarter"), "standing on its hind paws with one foreleg extended high above its head and the other reaching diagonally to the side; an elongated upright torso connects a narrow waist to wide hips"),
    "long_neck_sit": ("sitting", ("front", "rear", "three_quarter"), "sitting erect with a long narrow chest above very broad haunches, a small clear cat head and two separate long forepaws; its tail curls upward beside one flank"),
    "bent_hindquarters": ("standing", ("rear", "three_quarter", "top_down"), "standing with a long upright body axis in the image, broad shoulders offset to one side, hindquarters bent toward the other side, four thick paws clearly joined to the body"),
    "open_body_curl": ("curling", ("top_down", "side", "three_quarter"), "curling its body into an open U shape, with its pointed-eared head at one end and its haunches at the other; a large natural space remains between head and hips"),
    "asymmetric_sprawl": ("playing", ("top_down", "front"), "sprawled with its belly visible, one hind leg tucked in, one hind leg stretched sideways and both forepaws bent at different angles; a broad asymmetric connected body"),
    "raised_haunches": ("crouching", ("side", "three_quarter"), "crouching with a deep broad chest close to its extended forepaws, a narrow arched waist and raised rounded haunches; its tail makes a broad curve beside the body"),
}
GROUP_FAMILIES = {
    "high": tuple(FAMILIES),
    "x_strong": ("folded_loaf", "crescent_reach", "low_arch", "side_roll", "wide_turn", "raised_haunches"),
    "y_strong": ("upright_s_curve", "reaching_offset", "long_neck_sit", "bent_hindquarters", "open_body_curl"),
    "explore": tuple(FAMILIES),
}
OUTLINE_STYLES = {"clean_outline", "contour_and_fill", "hollow_ink"}


def source_plan(spec: base.Spec) -> dict:
    group = TARGET_GROUPS[spec.action_variant // 100]
    target = TARGETS[group][spec.action_variant % 100]
    return {"target_group": group, "target_Ex": float(target[0]), "target_Ey": float(target[1]),
            "source_id": int(spec.candidate), "source_seed": int(spec.seed),
            "targeted_pipeline_version": VERSION, "foreground_style": spec.style}


def specs_for(mode: str, total: int, master_seed: int = MASTER_SEED) -> list[base.Spec]:
    """Prefix-stable 40-source blocks: high/x/y/explore = 40/30/20/10%."""
    if mode not in ("pilot", "full") or total < 1:
        raise ValueError("mode must be pilot/full and total positive")
    rng = np.random.default_rng([int(master_seed), 731 if mode == "pilot" else 732])
    source_offset = 200000 if mode == "pilot" else 300000
    result = []
    group_occurrences = Counter()
    while len(result) < total:
        groups = rng.permutation(np.array([0]*16+[1]*12+[2]*8+[3]*4))
        for group_index in groups:
            group_index = int(group_index)
            group = TARGET_GROUPS[group_index]
            group_ordinal = group_occurrences[group]
            group_occurrences[group] += 1
            target_index = int(rng.integers(len(TARGETS[group])))
            family = str(rng.choice(GROUP_FAMILIES[group]))
            pose, views, anatomy = FAMILIES[family]
            view = str(rng.choice(views))
            if mode == "full" and group in ("x_strong", "y_strong"):
                # Deterministic 85% filled quota per 20 members of each wing,
                # preserving prefix stability while preventing random small-
                # batch outline excess or loss of the non-filled comparison.
                style_pool = (list(FILLED_TARGET_STYLES) if group_ordinal % 20 < 17
                              else ["clean_outline", "hollow_ink", "contour_and_fill"])
            elif group == "high":
                # Larger outline cats can provide high hard fractions without
                # making all high-modulus examples tiny filled silhouettes.
                style_pool = ["clean_outline", "contour_and_fill", "hollow_ink",
                              "clean_outline", "sculpted_cutpaper", "broad_folk"]
            else:
                style_pool = list(TARGET_STYLES)
            style = str(rng.choice(style_pool))
            morphology = str(rng.choice(base.MORPHS))
            tail = str(rng.choice(base.TAILS))
            if mode == "full" and group == "x_strong":
                tail = ("a long low tail reaching horizontally behind the hips",
                        "a thick tail folded close along the long lower flank",
                        "a short rounded bobtail",
                        "a broad tail curled beside the haunches")[int(rng.integers(4))]
            elif mode == "full" and group == "y_strong":
                tail = ("a thick tail held upward close beside the upright torso",
                        "a narrow tail curved close along the long flank",
                        "a short rounded bobtail",
                        "a broad tail wrapped closely around the lower haunches")[int(rng.integers(4))]
            direction = str(rng.choice(("leftward", "rightward")))
            background = 0 if group != "explore" or rng.random() < .75 else 1
            if group == "high":
                target_fraction = float(rng.uniform(.020, .085))
                extent_mode = "fraction"
                composition = str(rng.choice(("center", "center", "upper_left", "lower_right", "left", "right")))
            elif group == "x_strong":
                target_fraction = float(rng.uniform(.28, .57))
                extent_mode = "max_extent"
                composition = str(rng.choice(("center", "upper", "lower")))
            elif group == "y_strong":
                target_fraction = float(rng.uniform(.20, .52))
                extent_mode = "max_extent"
                composition = str(rng.choice(("center", "left", "right")))
            else:
                target_fraction = float(rng.uniform(.12, .48))
                extent_mode = "max_extent" if rng.random() < .5 else "fraction"
                composition = str(rng.choice(tuple(base.ANCHORS)))
            axis_request = {
                "high": "Keep the complete cat anatomically clear, with a distinct head, ears, torso and tail.",
                "x_strong": "The complete cat has a broad predominantly horizontal silhouette, with substantial connected torso width and clearly readable limbs.",
                "y_strong": "The complete cat has a predominantly upright silhouette, with substantial connected torso height and clearly readable limbs.",
                "explore": "Emphasize the unusual asymmetric body proportions and large natural spaces between limbs while preserving recognizable feline anatomy.",
            }[group]
            if mode == "full" and group == "x_strong":
                axis_request = ("The complete cat is at least twice as wide as tall, with a long deep horizontal torso "
                                "continuously joining the chest and haunches. Keep broad filled body anatomy and readable connected paws.")
            elif mode == "full" and group == "y_strong":
                axis_request = ("The complete cat is at least twice as tall as wide, with a slender long upright torso "
                                "continuously joining chest and haunches. Keep a complete recognizable cat head, pointed ears and connected paws.")
            style_text = TARGET_STYLES.get(style, FILLED_TARGET_STYLES.get(style))
            prompt = (f"{style_text} {base.CAMERAS[view]} One {morphology}, {anatomy}, with {tail}. "
                      f"Its head points {direction}. {axis_request} "
                      "One complete domestic cat only, with entire ears, head, torso, all visible paws and tail inside the image. "
                      "Pure black ink on a completely empty pure white background. No cut-off limbs. "
                      "No frame, letters, symbols, furniture, floor, shadows, pattern or other objects.")
            result.append(base.Spec(source_offset+len(result), int(rng.integers(0, 2**63-1)),
                                    pose, view, morphology, tail, style, direction, composition,
                                    group_index*100+target_index, background, target_fraction,
                                    prompt, family, extent_mode))
            if len(result) == total:
                break
    return result


def targeted_binarize(image: Image.Image, spec: base.Spec):
    result = base.binarize(image, spec)
    pixels, fraction, issue, geometry, diagnostics, mask = result
    # The inherited range check is superseded by this augmentation's explicit
    # high-end range; all other source clipping/fragmentation checks remain.
    if issue == "black_fraction_out_of_range" and .05 <= fraction <= .99:
        issue = ""
    diagnostics.update(source_plan(spec), variant_id=-1, source_candidate_id=spec.candidate)
    return pixels, fraction, issue, geometry, diagnostics, mask


class LocalModels(base.LocalModels):
    def generate(self, specifications: list[base.Spec]) -> list[base.Sample]:
        torch = self.torch
        generators = [torch.Generator("cpu").manual_seed(spec.seed) for spec in specifications]
        with torch.inference_mode():
            images = self.pipe([spec.prompt for spec in specifications], width=base.DIFFUSION_SIZE,
                               height=base.DIFFUSION_SIZE, generator=generators,
                               num_inference_steps=base.STEPS, guidance_scale=0., max_sequence_length=256).images
        processed = [targeted_binarize(image, spec) for image, spec in zip(images, specifications)]
        semantic = [base.semantic_image(result[0], spec.background) for result, spec in zip(processed, specifications)]
        with torch.inference_mode():
            inputs = self.processor(images=semantic, return_tensors="pt").to("cuda")
            features = self.normalize(self.siglip.get_image_features(**inputs).float())
            scores = features @ self.text_features.T * self.logit_scale + self.logit_bias
            embeddings, logits = features.cpu().numpy(), scores.cpu().numpy()
        samples = []
        for i, (spec, result) in enumerate(zip(specifications, processed)):
            pixels, fraction, issue, geometry, details, source_mask = result
            png = io.BytesIO()
            base.save_png(pixels, png)
            png.seek(0)
            if not np.array_equal(base.load_png(png), pixels):
                raise ValueError(f"PNG roundtrip failed for source {spec.candidate}")
            cat_logit = float(logits[i, :len(base.CAT_TEXT)].max())
            other = logits[i, len(base.CAT_TEXT):len(base.CAT_TEXT)+len(base.ALTERNATIVES)]
            strongest = int(other.argmax())
            samples.append(base.Sample(spec, pixels, embeddings[i], cat_logit, cat_logit-float(other[strongest]),
                                       base.ALTERNATIVES[strongest], logits[i, len(base.CAT_TEXT)+len(base.ALTERNATIVES):],
                                       fraction, issue, geometry, details, source_mask))
        return samples


def rejection(sample: base.Sample) -> str:
    pixels = sample.pixels
    if pixels.shape != (224,224) or pixels.dtype != np.uint8 or not np.isin(pixels, [0,1]).all():
        return "invalid_array"
    if not .05 <= float((pixels == 0).mean()) <= .99:
        return "black_fraction_out_of_range"
    if sample.issue:
        return sample.issue
    if max(sample.diagnostics.get("foreground_width", 0), sample.diagnostics.get("foreground_height", 0)) < 56:
        return "cat_too_small"
    if not np.isfinite(sample.cat_logit) or not np.isfinite(sample.cat_margin):
        return "invalid_semantic_score"
    if sample.cat_logit < -3 or sample.cat_margin < 0:
        return "low_cat_score"
    return ""


def variants_for_sample(sample: base.Sample, count: int = 3) -> list[base.Sample]:
    """Local, reproducible FEM search settings; ONE output/source is allowed.

    Semantic scores describe the independent source after object normalization.
    Every selected output additionally needs visual review at its actual scale.
    """
    if count < 1 or count > 3:
        raise ValueError("Use between one and three local variants per source")
    spec = sample.spec
    plan = source_plan(spec)
    group = plan["target_group"]
    if sample.issue not in ("", "black_fraction_out_of_range"):
        return []
    if not sample.source_mask.any():
        return []
    # The source was generated once. Parameter searches alter its physically
    # relevant extent / boundary position without clipping or adding content.
    if group == "high":
        targets = [spec.target_fraction, max(.0105, spec.target_fraction*.65), min(.14, spec.target_fraction*1.45)]
        options = [("fraction", targets[0], spec.composition),
                   ("fraction", targets[1], "center"),
                   ("fraction", targets[2], spec.composition)]
    else:
        # The first candidate may touch both loaded specimen boundaries. The
        # second decreases overall extent while the third redistributes side
        # ligaments. This is local response search, never dataset multiplicity.
        full_spec = replace(spec, extent_mode="max_extent")
        _, full_details, _ = base.frame_mask(sample.source_mask.astype(bool), full_spec)
        smaller = float(full_details["foreground_fraction"]*.90**2)
        alternate = {"upper":"lower", "lower":"upper", "left":"right", "right":"left"}.get(
            spec.composition, "upper" if group == "x_strong" else "right")
        options = [(spec.extent_mode, spec.target_fraction, spec.composition),
                   ("fraction", smaller, "center"),
                   ("max_extent", spec.target_fraction, alternate)]
    variants = []
    hashes = set()
    for variant_id, (extent, target, composition) in enumerate(options[:count]):
        variant_spec = replace(spec, candidate=spec.candidate*4+variant_id,
                               extent_mode=extent, target_fraction=float(target), composition=composition)
        pixels, details, geometry = base.frame_mask(sample.source_mask.astype(bool), variant_spec)
        fingerprint = base.fingerprint(pixels)
        if fingerprint in hashes:
            continue
        hashes.add(fingerprint)
        details = dict(sample.diagnostics, **details)
        details.update(plan, variant_id=variant_id, source_candidate_id=spec.candidate,
                       variant_operation="isotropic_uncropped_extent_and_placement_search",
                       semantic_evaluation="source_object_normalized; selected output requires actual-scale visual review")
        issue = sample.issue if sample.issue != "black_fraction_out_of_range" else ""
        fraction = float((pixels == 0).mean())
        if not .05 <= fraction <= .99:
            issue = issue or "black_fraction_out_of_range"
        variants.append(base.Sample(variant_spec, pixels, sample.embedding, sample.cat_logit, sample.cat_margin,
                                    sample.alternative, sample.view_scores, fraction, issue, geometry, details,
                                    sample.source_mask))
    return variants


def iter_acquire(mode: str, total: int, work_dir: Path, seed: int = MASTER_SEED, batch_size: int = 4):
    specs = specs_for(mode, total, seed)
    cache = Path(work_dir) / f"targeted-v{VERSION}-{mode}-{seed}-b{batch_size}"
    cache.mkdir(parents=True, exist_ok=True)
    models = None
    accepted = 0
    for start in range(0, total, batch_size):
        batch = specs[start:start+batch_size]
        path = cache / f"batch_{start:05d}_{len(batch)}.npz"
        if path.exists():
            samples = base.read_batch(path)
            if [s.spec for s in samples] != batch:
                raise RuntimeError(f"Cached source specifications disagree: {path}")
        else:
            if models is None:
                models = LocalModels()
            samples = models.generate(batch)
            base.cache_batch(path, samples)
        accepted += sum(not rejection(s) for s in samples)
        if (start+len(batch)) % 20 == 0 or start+len(batch) == total:
            print(f"targeted {mode}: {start+len(batch)}/{total} independent sources; {accepted} source auto-pass", flush=True)
        yield samples


def acquire(mode: str, total: int, work_dir: Path, seed: int = MASTER_SEED, batch_size: int = 4):
    return [s for batch in iter_acquire(mode, total, work_dir, seed, batch_size) for s in batch]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("pilot", "full"))
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=MASTER_SEED)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--work-dir", type=Path, required=True)
    args = parser.parse_args()
    sources = acquire(args.mode, args.count, args.work_dir, args.seed, args.batch_size)
    variants = [v for s in sources for v in variants_for_sample(s)]
    print(json.dumps({"sources": len(sources), "variants": len(variants),
                      "valid_variants": sum(not rejection(v) for v in variants),
                      "source_rejections": dict(Counter(rejection(s) for s in sources if rejection(s))),
                      "groups": dict(Counter(s.diagnostics["target_group"] for s in sources)),
                      "one_final_output_per_source_required": True}, indent=2))


if __name__ == "__main__":
    main()
