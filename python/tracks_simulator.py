import os
import math
import numpy as np
import random
from PIL import Image, ImageDraw
from pathlib import Path
import argparse
import colorsys

def draw_lines_on_canvas(a: int, noise: int, output_path: Path, sizex: int = 182, sizey: int = 562, lwidth: int = 2, lwidthrng: bool = False, c = 0, bc = 0):
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

    if bc != 0:
        for i in range(bc):
            if lwidthrng:
                linewidth = random.randint(0, 5)
            else:
                linewidth = lwidth
            color = hsv_to_rgb_tuple(random.random(), 1.0, 1.0)
            p0 = (random.randint(0, sizex), random.randint(0, sizey))
            p1 = (random.randint(0, sizex), random.randint(0, sizey))
            p2 = (random.randint(0, sizex), random.randint(0, sizey))
            p3 = (random.randint(0, sizex), random.randint(0, sizey))
        
            points = cubic_bezier(p0, p1, p2, p3)
            overlay_draw.line(points, fill=color, width=linewidth)
    Image.alpha_composite(canvas, overlay).save(output_path)

def _track_heading(points):
    if len(points) < 2:
        return -math.pi / 2
    dx = points[-1][0] - points[-2][0]
    dy = points[-1][1] - points[-2][1]
    return math.atan2(dy, dx)


# def _arc_points(origin, start_angle: float, turn: float, length: float, sizex: int, sizey: int, steps: int = 80):
#     points = [origin]
#     step_count = max(10, steps)
#     step_length = max(1.0, length / step_count)
#     curvature = turn / max(1.0, length)

#     for i in range(1, step_count + 1):
#         s = i * step_length
#         theta = start_angle + curvature * s
#         x = origin[0] + math.cos(theta) * s
#         y = origin[1] + math.sin(theta) * s
#         if x < 0 or x > sizex or y < 0 or y > sizey:
#             break
#         points.append((int(round(x)), int(round(y))))
#     return points

def _arc_points(origin, start_angle: float, turn: float, length: float, sizex: int, sizey: int, steps: int = 80):
    """Generate points along a circular arc with constant radius."""
    points = [origin]
    step_count = max(10, steps)
    
    if turn == 0:
        # Straight line
        end_x = int(np.clip(origin[0] + math.cos(start_angle) * length, 0, sizex - 1))
        end_y = int(np.clip(origin[1] + math.sin(start_angle) * length, 0, sizey - 1))
        return [origin, (end_x, end_y)]
    
    # Calculate radius from the turn parameter
    radius = length / abs(turn) if turn != 0 else float('inf')
    
    # Find center of circle (perpendicular to start direction at distance radius)
    perpendicular_angle = start_angle + (math.pi / 2 if turn > 0 else -math.pi / 2)
    center_x = origin[0] + radius * math.cos(perpendicular_angle)
    center_y = origin[1] + radius * math.sin(perpendicular_angle)
    
    # Generate points along the arc at equal angles around the center
    angle_step = turn / step_count
    current_angle = math.atan2(origin[1] - center_y, origin[0] - center_x)
    
    for i in range(1, step_count + 1):
        current_angle += angle_step
        x = center_x + radius * math.cos(current_angle)
        y = center_y + radius * math.sin(current_angle)
        
        if x < 0 or x > sizex or y < 0 or y > sizey:
            break
        points.append((int(round(x)), int(round(y))))
    
    return points

# def _arc_points_decreasing_radius(origin, start_angle: float, turn: float, length: float, sizex: int, sizey: int, steps: int = 80):
#     """Arc with decreasing radius (tighter as it goes). Used for physics simulation."""
#     points = [origin]
#     step_count = max(10, steps)
#     step_length = max(1.0, length / step_count)
#     base_curvature = turn / max(1.0, length)

#     for i in range(1, step_count + 1):
#         s = i * step_length
#         # Progressively increase curvature as distance increases (decreasing radius)
#         progress = i / step_count  # 0 to 1
#         curvature = base_curvature * (1.0 + progress)  # Increases curvature over time
#         theta = start_angle + curvature * s
#         x = origin[0] + math.cos(theta) * s
#         y = origin[1] + math.sin(theta) * s
#         if x < 0 or x > sizex or y < 0 or y > sizey:
#             break
#         points.append((int(round(x)), int(round(y))))
#     return points

def _arc_points_decreasing_radius(origin, start_angle: float, turn: float, length: float, sizex: int, sizey: int, steps: int = 80):
    """Arc with decreasing radius (tighter as it goes). Used for physics simulation."""
    points = [origin]
    step_count = max(10, steps)
    
    if turn == 0:
        end_x = int(np.clip(origin[0] + math.cos(start_angle) * length, 0, sizex - 1))
        end_y = int(np.clip(origin[1] + math.sin(start_angle) * length, 0, sizey - 1))
        return [origin, (end_x, end_y)]
    
    # Start with radius based on length and turn
    initial_radius = length / abs(turn)
    
    # Find center of circle (perpendicular to start direction)
    perpendicular_angle = start_angle + (math.pi / 2 if turn > 0 else -math.pi / 2)
    center_x = origin[0] + initial_radius * math.cos(perpendicular_angle)
    center_y = origin[1] + initial_radius * math.sin(perpendicular_angle)
    
    # Generate points along the arc with decreasing radius
    angle_step = turn / step_count
    current_angle = math.atan2(origin[1] - center_y, origin[0] - center_x)
    
    for i in range(1, step_count + 1):
        progress = i / step_count  # 0 to 1
        # Radius decreases from initial to a smaller value
        radius = initial_radius * (1.0 - 0.5 * progress)  # Decreases to 50% of initial
        
        current_angle += angle_step
        x = center_x + radius * math.cos(current_angle)
        y = center_y + radius * math.sin(current_angle)
        
        if x < 0 or x > sizex or y < 0 or y > sizey:
            break
        points.append((int(round(x)), int(round(y))))
    
    return points

def _straight_branch(origin, heading: float, length: float, sizex: int, sizey: int):
    end_x = int(np.clip(origin[0] + math.cos(heading) * length, 0, sizex - 1))
    end_y = int(np.clip(origin[1] + math.sin(heading) * length, 0, sizey - 1))
    return [origin, (end_x, end_y)]


def _generate_branches_recursive(branch, depth_remaining, max_splits, overlay_draw, sizex, sizey, lwidth, simulate_physics=False, spirals_enabled=False, line_color=None, mask_draw=None):
    """Recursively generate child branches from a parent branch."""
    if depth_remaining <= 0 or random.random() > 0.55:
        return
    
    child_count = random.randint(1, max_splits)
    for _ in range(child_count):
        tip = branch[-1]
        # Only include spiral if spirals are explicitly enabled
        if spirals_enabled:
            child_style = random.choices(["straight", "moderate", "spiral"], weights=[0.15, 0.40, 0.45])[0]
        else:
            child_style = random.choices(["straight", "moderate"], weights=[0.25, 0.75])[0]
        
        if child_style == "straight":
            child = _straight_branch(tip, _track_heading(branch) + random.uniform(-0.7, 0.7), random.randint(30, 90), sizex, sizey)
        elif child_style == "moderate":
            if simulate_physics:
                child = _arc_points_decreasing_radius(tip, _track_heading(branch) + random.uniform(-0.9, 0.9), random.uniform(-1.0, 1.0), random.randint(35, 110), sizex, sizey, steps=110)
            else:
                child = _arc_points(tip, _track_heading(branch) + random.uniform(-0.9, 0.9), random.uniform(-1.0, 1.0), random.randint(35, 110), sizex, sizey, steps=110)
        else:  # spiral
            child = _arc_points_decreasing_radius(tip, _track_heading(branch) + random.uniform(-1.3, 1.3), random.choice([-2.8, 2.8]), random.randint(30, 80), sizex, sizey, steps=180)
        
        if len(child) > 1:
            overlay_draw.line(child, fill=line_color, width=max(1, lwidth))
            if mask_draw is not None:
                mask_draw.line(child, fill=255, width=max(1, lwidth))
            _generate_branches_recursive(child, depth_remaining - 1, max_splits, overlay_draw, sizex, sizey, lwidth, simulate_physics, spirals_enabled, line_color, mask_draw)


def draw_tracks_on_canvas(max_depth: int, output_path: Path, mask_path: Path, splits: int = 3, starty: int = 10, sizex: int = 182, sizey: int = 562, lwidth: int = 2, lwidthrng: bool = False, simulate_physics: bool = False, spirals_enabled: bool = False, save_mask: bool = False, black_and_white: bool = False, skip_incoming_line: bool = False):
    canvas = Image.new("RGBA", (sizex, sizey), "white")
    overlay = Image.new("RGBA", (sizex, sizey))
    overlay_draw = ImageDraw.Draw(overlay)
    mask = Image.new("L", (sizex, sizey), 0) if save_mask else None
    mask_draw = ImageDraw.Draw(mask) if mask is not None else None

    def line_color():
        if black_and_white:
            shade = random.randint(0, 96)
            return (shade, shade, shade)
        return hsv_to_rgb_tuple(random.random(), 1.0, 1.0)

    vertex_x = random.randint(int(sizex * 0.35), int(sizex * 0.65))
    vertex_y = random.randint(int(sizey * 0.22), int(sizey * 0.55))
    vertex = (vertex_x, vertex_y)

    incoming_start = (vertex_x + random.randint(-5, 5), sizey + random.randint(50, 200))
    incoming_end = (vertex_x + random.randint(-10, 10), vertex_y)
    if not skip_incoming_line:
        overlay_draw.line([incoming_start, incoming_end], fill=line_color(), width=max(1, lwidth))
        if mask_draw is not None:
            mask_draw.line([incoming_start, incoming_end], fill=255, width=max(1, lwidth))

    root_branches = []
    directions = [-2.7, -2.2, -1.7, -1.2, -0.7, 0.7, 1.2, 1.7, 2.2, 2.7]
    track_count = max(1, min(7, splits))

    for _ in range(track_count):
        direction = random.choice(directions)
        turn = random.uniform(-2.8, 2.8)
        length = random.randint(80, max(180, sizey // 2))
        if random.random() < 0.45:
            turn *= 1.9
            length = random.randint(140, max(260, sizey))

        origin = incoming_end
        if random.random() < 0.35:
            branch = _straight_branch(origin, direction, length, sizex, sizey)
        else:
            if simulate_physics:
                branch = _arc_points_decreasing_radius(origin, direction, turn, length, sizex, sizey, steps=260)
            else:
                branch = _arc_points(origin, direction, turn, length, sizex, sizey, steps=260)

        if len(branch) > 1:
            root_branches.append(branch)
            branch_color = line_color()
            overlay_draw.line(branch, fill=branch_color, width=max(1, lwidth))
            if mask_draw is not None:
                mask_draw.line(branch, fill=255, width=max(1, lwidth))
            _generate_branches_recursive(branch, max_depth, splits, overlay_draw, sizex, sizey, lwidth, simulate_physics, spirals_enabled, branch_color, mask_draw)

    Image.alpha_composite(canvas, overlay).save(output_path)
    if mask is not None:
        # mask_path = output_path.with_name(f"{output_path.stem}_mask{output_path.suffix}")
        mask.save(mask_path)


def Split(sizex: int, sizey: int, ps, heading: float, turn: float, spread: int = 90):
    origin = ps
    branch_paths = []
    branch_count = random.randint(1, 3)

    for _ in range(branch_count):
        local_heading = heading + random.uniform(-1.8, 1.8)
        local_turn = max(-3.2, min(3.2, turn + random.uniform(-1.8, 1.8)))
        local_length = random.randint(30, max(60, spread))
        if random.random() < 0.4:
            local_turn *= 1.9
            local_length = random.randint(110, max(180, spread * 2))

        if random.random() < 0.45:
            path = _straight_branch(origin, local_heading, local_length, sizex, sizey)
        else:
            path = _arc_points(origin, local_heading, local_turn, local_length, sizex, sizey, steps=260)
        if len(path) > 1:
            branch_paths.append(path)

    return branch_paths

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

def createGrid(sizex, sizey, g):
    gridxlen = int(sizex/g)
    gridylen = int(sizey/g)
    gridxpoints1 = []
    gridxpoints2 = []
    gridypoints1 = []
    gridypoints2 = []
    final_list = []
    x_list = []
    y_list = []
    xp1 = [random.randint(0, gridxlen), 0]
    xp2 = [random.randint(0, gridxlen), sizey]
    x1 = xp1
    x2 = xp2
    while x1[0] < sizex:
        gridxpoints1.append(x1[:])
        x1[0] += gridxlen
    while x2[0] < sizex:
        gridxpoints2.append(x2[:])
        x2[0] += gridxlen
    if len(gridxpoints1) != len(gridxpoints2):
        if len(gridxpoints1) < len(gridxpoints2):
            gridxpoints1.append([gridxpoints1[-1][0] + gridxlen, 0])
        else:
            gridxpoints2.append([gridxpoints2[-1][0] + gridxlen, sizey])
    yp1 = [0, random.randint(0, gridylen)]
    yp2 = [sizex, random.randint(0, gridylen)]
    y1 = yp1
    y2 = yp2
    while y1[1] < sizey:
        gridypoints1.append(y1[:])
        y1[1] += gridylen
    while y2[1] < sizey:
        gridypoints2.append(y2[:])
        y2[1] += gridylen
    if len(gridypoints1) != len(gridypoints2):
        if len(gridypoints1) < len(gridypoints2):
            gridypoints1.append([0, gridypoints1[-1][1] + gridylen])
        else:
            gridypoints2.append([sizex, gridypoints2[-1][1] + gridylen])
    for x in range(len(gridxpoints1)):
        x_list.append([tuple(gridxpoints1[x]), tuple(gridxpoints2[x])])
    for y in range(len(gridypoints1)):
        y_list.append([tuple(gridypoints1[y]), tuple(gridypoints2[y])])

    final_list.append(x_list)
    final_list.append(y_list)
    return final_list


def drawGrid(sizex, sizey, g, input_path, sat, bw: bool):
    lines = createGrid(sizex, sizey, g)
    image = Image.open(input_path).convert("RGBA")
    overlay = Image.new("RGBA", (sizex, sizey))
    overlay_draw = ImageDraw.Draw(overlay)
    color: tuple
    if not bw:
        color = (*hsv_to_rgb_tuple(random.random(), sat, 1.0), 128)
    else:
        color = (*hsv_to_rgb_tuple(0, 0, random.random()), 128)
    for pair in lines[0]:
        overlay_draw.line(pair, fill=color, width=1)
    for pair in lines[1]:
        overlay_draw.line(pair, fill=color, width=1)
    Image.alpha_composite(image, overlay).save(input_path)

def rotateImg(input_path, rot):
    with Image.open(input_path) as im:
        rotated = im.rotate(rot, expand=True)
        rotated.save(input_path)

def hsv_to_rgb_tuple(h: float, s: float, v: float) -> tuple[int, int, int]:
    """Convert HSV (each in 0-1) to a Pillow-compatible RGB tuple (0-255)."""
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return (round(r * 255), round(g * 255), round(b * 255))

def cubic_bezier(p0, p1, p2, p3, num_points=100):
    """Return a list of (x, y) points along a cubic bezier curve."""
    points = []
    for i in range(num_points):
        t = i / (num_points - 1)
        x = (1-t)**3 * p0[0] + 3*(1-t)**2 * t * p1[0] + 3*(1-t) * t**2 * p2[0] + t**3 * p3[0]
        y = (1-t)**3 * p0[1] + 3*(1-t)**2 * t * p1[1] + 3*(1-t) * t**2 * p2[1] + t**3 * p3[1]
        points.append((x, y))
    return points
    
def grayScaleImg(input):
    img = Image.open(input)
    grayscale_image = img.convert("L")
    grayscale_image.save(input)

def drawCircle(input, c: int, r:float, clr: tuple, w: int):
    img = Image.open(input)
    draw = ImageDraw.Draw(img)
    draw.circle(c, r, outline=clr, width=w)
    img.save(input)

def drawCircles(input, a, bw):
    if bw:
        color = color = (*hsv_to_rgb_tuple(0, 0, random.random()), 128)
    else:
        color = (*hsv_to_rgb_tuple(random.random(), 1.0, 1.0), 128)
    radius = random.randint(1, (int(math.hypot(arguments.size_x, arguments.size_y)/2)))
    centerp = (random.randint(0, arguments.size_x), random.randint(0, arguments.size_x))
    for i in range(a):
        drawCircle(input, c=centerp, r=radius, clr = color, w = arguments.line_width)
        radius += random.randint(1, (int(math.hypot(arguments.size_x, arguments.size_y)/2)))

def parse_arguments() -> argparse.Namespace:
    project_root = Path(__file__).parent.parent
    parser = argparse.ArgumentParser(
        description="Create lines on a white canvas"
    )
    parser.add_argument("--simulate-physics","--simphys","--smph",action="store_true", help="Used by the track simulator to decrease the radius of curved paths over time")
    parser.add_argument("--output","--o",type=Path,default=project_root / "resources" / "simulated.png",help="Output path for the image")
    parser.add_argument("--mask-path","--mp",type=Path,default=project_root / "resources" / "simulated_mask.png",help="Output path for the track simulator mask")
    parser.add_argument("--amount","--a",type=int,default=1,help="Amount of lines for normal simulator, recursion depth for track simulator, or the maximum recursion depth for datasets")
    parser.add_argument("--grid","--g",type=int,default=0,help="Draws an optional grid as noise")
    parser.add_argument("--grid-saturation","--grid-sat","--gs",type=float,default=0.5,help="The Saturation of the grid")
    parser.add_argument("--splits","--spl",type=int,default=3,help="Controls the amount of Splits when using the track simulator")
    parser.add_argument("--noise","--n",type=int,default=0, help="Addes optional gaussian noise to the image")
    parser.add_argument("--salt-and-pepper","--sp",type=float,default=0, help="Addes optional salt-and-pepper noise to the image")
    parser.add_argument("--salt-noise","--sn",type=float,default=0,help="Adds salt noise to the image to break apart lines")
    parser.add_argument("--circle-noise", "--cn", type=int, default=0, help="Addes circles to the image")
    parser.add_argument("--line-width","--lw",type=int,default=2,help="Controls the width of the lines")
    parser.add_argument("--line-width-randomize","--lwr",action="store_true",help="Controls whether or not the line width should be randomized")
    parser.add_argument("--size-x","--sx",type=int, default=182,help="Controls the width of the image")
    parser.add_argument("--size-y","--sy",type=int, default=562,help="Controls the height of the image")
    parser.add_argument("--curves","--c",type=int, default=0,help="Addes optional curves to the normal simulator")
    parser.add_argument("--brezier-curves", "--bc", type=int, default=0, help="Addes smoother brezier curves to the normal simulator")
    parser.add_argument("--tracks", "--t", action="store_true", help="Activates the track simulator")
    parser.add_argument("--spirals", action="store_true", help="Adds optional spirals to the track simulator")
    parser.add_argument("--save-mask", action="store_true",help="Saves the mask of the tracks produced by the track simulator")
    parser.add_argument("--skip-incoming-line", action="store_true",help="Skips the incoming line in the track simulator")
    parser.add_argument("--start-y", "--sty", type=int, default=10, help="POSSIBLY DEPRECATED Controls the starting y-position for the incoming line for the track simulator")
    parser.add_argument("--create-dataset", "--dataset", "--d", action="store_true", help="Creates a dataset")
    parser.add_argument("--dataset-size", "--ds", type=int, default = 1,help="Controls the size of the dataset")
    parser.add_argument("--dataset-name", "--dn", type=str, default="dataset", help="Assigns the name of a dataset")
    parser.add_argument("--black-and-white", "--bw", action="store_true", help="Changes the images to be in grayscale")
    return parser.parse_args()

def createDataset(s: int, dsn: str, smphy: bool, spr: bool, bw: bool):
    project_root = Path(__file__).parent.parent
    dir_path = project_root / "resources" / "datasets" / dsn
    img_path = project_root / "resources" / "datasets" / dsn / "images"
    mask_path = project_root / "resources" / "datasets" / dsn / "masks"
    os.mkdir(dir_path)
    os.mkdir(img_path)
    os.mkdir(mask_path)
    for i in range(s):
        output_path = project_root / "resources" / "datasets" / dsn / "images" / f"{dsn}_{i}.png"
        mask_output = project_root / "resources" / "datasets" / dsn / "masks" / f"{dsn}_{i}.png"
        starty = 10
        if arguments.amount < 3:
            amount = random.randint(3, 7)
        else:
            amount = random.randint(3, arguments.amount)
        if arguments.splits < 1:
            splits = random.randint(1, 4)
        else:
            splits = random.randint(1, arguments.splits)
        if arguments.size_x < 128:
            sizex = random.randint(128, 1024)
        else:
            sizex = random.randint(128, arguments.size_x)
        if arguments.size_x < 128:
            sizey = random.randint(128, 1024)
        else:
            sizey = random.randint(128, arguments.size_y)
        if arguments.line_width < 1:
            lwidth = random.randint(1, 3)
        else:
            lwidth = random.randint(1, arguments.line_width)
        lwidthrng = bool(random.getrandbits(1))
        if arguments.line_width_randomize:
            lwidthrng = arguments.line_width_randomize
        sil = bool(random.getrandbits(1))
        if arguments.skip_incoming_line:
            sil = True
        lwidthrng = bool(random.getrandbits(1))
        draw_tracks_on_canvas(amount, output_path, mask_output, splits, starty, sizex, sizey, lwidth, lwidthrng, smphy, spr, True, True, sil)
        circles = bool(random.getrandbits(1))
        if circles:
            circlesAmt = random.randint(0, arguments.circle_noise)
            drawCircles(output_path, circlesAmt, bw)
        noise = bool(random.getrandbits(1))
        if noise:
            noiseAmount = random.randint(0, 30)
            createNoise(noiseAmount, output_path)
        sp = bool(random.getrandbits(1))
        if sp:
            spAmount = random.random()
            createSPNoise(spAmount, output_path)
        sn = bool(random.getrandbits(1))
        if sn:
            snAmount = random.random()
            createSaltNoise(snAmount, output_path)
        grid = bool(random.getrandbits(1))
        if grid:
            gridDist = random.randint(2, 12)
            gridSat = random.random()
            drawGrid(sizex, sizey, gridDist, output_path, gridSat, bw)
        rot = bool(random.getrandbits(1))
        if rot:
            rotAmountMatch = random.randint(0,3)
            rotAmount = 0
            match rotAmountMatch:
                case 0:
                    rotAmount = 0
                case 1:
                    rotAmount = 90
                case 2:
                    rotAmount = 180
                case 3:
                    rotAmount = 270
            rotateImg(output_path,rotAmount)
        if bw:
            grayScaleImg(output_path)

if __name__ == "__main__":
    arguments = parse_arguments()
    if not arguments.create_dataset:
        if not arguments.tracks:
            draw_lines_on_canvas(arguments.amount, arguments.noise, arguments.output, arguments.size_x, arguments.size_y, arguments.line_width, arguments.line_width_randomize, arguments.curves, arguments.brezier_curves)
        else:
            draw_tracks_on_canvas(
                arguments.amount,
                arguments.output,
                arguments.mask_path,
                arguments.splits,
                arguments.start_y,
                arguments.size_x,
                arguments.size_y,
                arguments.line_width,
                arguments.line_width_randomize,
                arguments.simulate_physics,
                arguments.spirals,
                save_mask=arguments.save_mask,
                black_and_white=arguments.black_and_white,
                skip_incoming_line=arguments.skip_incoming_line,
            )
        if arguments.salt_noise != 0:
            createSaltNoise(arguments.salt_noise, arguments.output)
        if arguments.noise != 0:
            createNoise(arguments.noise, arguments.output)
        if arguments.salt_and_pepper != 0:
            createSPNoise(arguments.salt_and_pepper, arguments.output)
        if arguments.grid != 0:
            drawGrid(arguments.size_x, arguments.size_y, arguments.grid, arguments.output, arguments.grid_saturation)
        if arguments.black_and_white:
            grayScaleImg(arguments.output)
        if arguments.circle_noise != 0:
            drawCircles(arguments.output, arguments.circle_noise, arguments.black_and_white)
    else:
        createDataset(arguments.dataset_size, arguments.dataset_name, arguments.simulate_physics, arguments.spirals, arguments.black_and_white)