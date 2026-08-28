import numpy as np
import random
from PIL import Image, ImageDraw
from pathlib import Path
import argparse
import colorsys

def draw_lines_on_canvas(a: int, noise: int, output_path: Path, sizex: int = 182, sizey: int = 562, lwidth: int = 2, lwidthrng: bool = False):
    canvas = Image.new("RGBA", (sizex, sizey), "white")
    overlay = Image.new("RGBA", (sizex, sizey))
    overlay_draw = ImageDraw.Draw(overlay)
    for i in range(a):
        if lwidthrng:
            linewidth = random.randint(0, 5)
        else:
            linewidth = lwidth
        color = hsv_to_rgb_tuple(random.random(), 1.0, 1.0)
        p1 = (random.randint(0, sizex), random.randint(0, sizey))
        p2 = (random.randint(0, sizex), random.randint(0, sizey))
        overlay_draw.line([p1, p2], fill=color, width=linewidth)

    Image.alpha_composite(canvas, overlay).save(output_path)

def hsv_to_rgb_tuple(h: float, s: float, v: float) -> tuple[int, int, int]:
    """Convert HSV (each in 0–1) to a Pillow-compatible RGB tuple (0–255)."""
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return (round(r * 255), round(g * 255), round(b * 255))

def parse_arguments() -> argparse.Namespace:
    project_root = Path(__file__).parent.parent
    parser = argparse.ArgumentParser(
        description="Create lines on a white canvas"
    )
    parser.add_argument("--output","--o",type=Path,default=project_root / "resources" / "simulated.png",)
    parser.add_argument("--amount","--a",type=int,default=1,)
    parser.add_argument("--noise","--n",type=int,default=0,)
    parser.add_argument("--line-width","--lw",type=int,default=2,)
    parser.add_argument("--line-width-randomize","--lwr",action="store_true",)
    parser.add_argument("--size-x","--sx",type=int, default=182)
    parser.add_argument("--size-y","--sy",type=int, default=562)
    return parser.parse_args()

if __name__ == "__main__":
    arguments = parse_arguments()
    draw_lines_on_canvas(arguments.amount, arguments.noise, arguments.output, arguments.size_x, arguments.size_y, arguments.line_width, arguments.line_width_randomize)