"""One replaceable working contact sheet; never adds previews to the release."""
from pathlib import Path
import argparse
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import candidate_generation as gen


def read_pool(directory):
    samples = []
    for path in sorted(Path(directory).glob("batch_*.npz")):
        samples.extend(gen.read_batch(path))
    return samples


def make_sheet(samples, path, columns=10, tile=160, labels=None):
    rows = math.ceil(len(samples) / columns)
    gap, label_height = 6, 24
    sheet = Image.new("RGB", (gap+columns*(tile+gap), gap+rows*(tile+label_height+gap)), "white")
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 13)
    except OSError:
        font = ImageFont.load_default()
    for index, sample in enumerate(samples):
        x, y = gap+(index%columns)*(tile+gap), gap+(index//columns)*(tile+label_height+gap)
        image = Image.fromarray(sample.pixels*255).resize((tile,tile), Image.Resampling.NEAREST)
        sheet.paste(image, (x,y))
        draw.rectangle((x-1,y-1,x+tile,y+tile), outline="gray")
        label = labels[index] if labels is not None else f"{sample.spec.candidate:04d}  b={sample.black_fraction:.2f}"
        draw.text((x+2,y+tile+3), label, fill="black", font=font)
    sheet.save(path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache",type=Path)
    parser.add_argument("destination",type=Path)
    parser.add_argument("--start",type=int,default=0)
    parser.add_argument("--count",type=int,default=50)
    parser.add_argument("--tile",type=int,default=160)
    parser.add_argument("--columns",type=int,default=10)
    parser.add_argument("--include-rejected",action="store_true")
    args=parser.parse_args()
    samples=read_pool(args.cache)
    subset=[s for s in samples if args.start<=s.spec.candidate<args.start+args.count
            and (args.include_rejected or not gen.rejection(s))]
    make_sheet(subset,args.destination,args.columns,args.tile)
    print(f"{len(subset)} candidates: {[s.spec.candidate for s in subset]}")


if __name__=="__main__":
    main()
