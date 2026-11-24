"""Unscreened first frozen seed at all 24 targets, exact FEM-input PNGs."""
import argparse
from PIL import Image, ImageDraw
from common import ROOT, RUN, read, save


def render(step):
    folder = ROOT / 'evaluation' / 'runs' / f'cfm_v2_step_{step:06d}'
    targets = read(folder / 'targets.json')
    if isinstance(targets, dict):
        targets = targets['targets']
    seed = 730000  # First seed in the previously frozen panel, never quality selected.
    canvas = Image.new('RGB', (6 * 224, 4 * 252), '#eeeeee')
    draw = ImageDraw.Draw(canvas)
    selected = []
    for i, target in enumerate(targets):
        tid = target['target_id']
        path = folder / 'images' / f'{tid}_{seed}.png'
        assert path.exists(), path
        im = Image.open(path).convert('RGB')
        assert im.size == (224, 224)
        canvas.paste(im, ((i % 6) * 224, (i // 6) * 252 + 24))
        draw.text(((i % 6) * 224 + 8, (i // 6) * 252 + 6), f'{tid} seed{seed} | {step//1000}k', fill='black')
        selected.append(str(path.relative_to(ROOT)))
    assert len(selected) == 24
    canvas.save(RUN / f'fixed_seed_{step:06d}.png')
    save(RUN / f'fixed_seed_{step:06d}.json', dict(step=step, seed=seed, selection='all 24 targets, first frozen seed, unscreened', source_images=selected))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--step', type=int, required=True)
    render(parser.parse_args().step)
