import numpy as np
import random
from PIL import Image, ImageDraw
from pathlib import Path
import argparse
import colorsys

def draw_lines_on_canvas(a: int, noise: int, output_path: Path, sizex: int = 182, sizey: int = 562, lwidth: int = 2, lwidthrng: bool = False, c = 0):
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
    if c != 0:
        for i in range(c):
            if lwidthrng:
                linewidth = random.randint(0, 5)
            else:
                linewidth = lwidth
            color = hsv_to_rgb_tuple(random.random(), 1.0, 1.0)
            p1 = (random.randint(0, sizex), random.randint(0, sizey))
            p2 = (random.randint(0, sizex), random.randint(0, sizey))
            p3 = (random.randint(0, sizex), random.randint(0, sizey))
            points = [p1, p2, p3]
            overlay_draw.line(points, fill=color, width=linewidth, joint="curve")

    Image.alpha_composite(canvas, overlay).save(output_path)

def add_gaussian_noise(image, std=10):
    arr = np.array(image).astype(np.float32)
    noise = np.random.normal(0, std, arr.shape)
    # Clip values to 0-255 range
    noisy_arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy_arr)

def add_salt_pepper_noise(image, prob=0.02):
    arr = np.array(image)
    # Generate random values for each pixel
    random_values = np.random.rand(*arr.shape[:2])
    
    arr[random_values < prob / 2] = 0
    arr[random_values > 1 - prob / 2] = 255
    return Image.fromarray(arr)

def add_salt_noise(image, prob=0.02):
    arr = np.array(image)
    # Generate random values for each pixel
    random_values = np.random.rand(*arr.shape[:2])
    
    arr[random_values > 1 - prob] = 255
    return Image.fromarray(arr)

def createNoise(n: int, input_path):
    image = Image.open(input_path)
    noisy_img = add_gaussian_noise(image, n)
    noisy_img.save(input_path)

def createSPNoise(p, input_path):
    img = Image.open(input_path)
    noisy_img = add_salt_pepper_noise(img, p)
    noisy_img.save(input_path)

def createSaltNoise(p, input_path):
    img = Image.open(input_path)
    noisy_img = add_salt_noise(img, p)
    noisy_img.save(input_path)

def hsv_to_rgb_tuple(h: float, s: float, v: float) -> tuple[int, int, int]:
    """Convert HSV (each in 0-1) to a Pillow-compatible RGB tuple (0-255)."""
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
    parser.add_argument("--salt-and-pepper","--sp",type=float,default=0)
    parser.add_argument("--salt-noise","--sn",type=float,default=0)
    parser.add_argument("--line-width","--lw",type=int,default=2,)
    parser.add_argument("--line-width-randomize","--lwr",action="store_true",)
    parser.add_argument("--size-x","--sx",type=int, default=182)
    parser.add_argument("--size-y","--sy",type=int, default=562)
    parser.add_argument("--curves","--c",type=int, default=0)
    return parser.parse_args()

if __name__ == "__main__":
    arguments = parse_arguments()
    draw_lines_on_canvas(arguments.amount, arguments.noise, arguments.output, arguments.size_x, arguments.size_y, arguments.line_width, arguments.line_width_randomize, arguments.curves)
    if arguments.salt_noise != 0:
        createSaltNoise(arguments.salt_noise, arguments.output)
    if arguments.noise != 0:
        createNoise(arguments.noise, arguments.output)
    if arguments.salt_and_pepper != 0:
        createSPNoise(arguments.salt_and_pepper, arguments.output)