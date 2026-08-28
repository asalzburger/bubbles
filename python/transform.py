import json
from pathlib import Path
from sympy import Point, Point2D, Line, evalf, Symbol, Polygon, N, pi, Line2D, Point, oo
from collections import Counter
import itertools
import Circle
import find_seeds
from PIL import Image, ImageDraw
import numpy as np
import colorsys
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import math

Intersects = []
SeedLines = []

# simple transform function
def transform(x, y, m = 0):
    t = -m*x+y

    return t

# function to calculate the intersection point
def get_intersection_point(m0, m1, x1, y1, x2, y2):
    l1 = Line(Point(m0, transform(x1, y1, m0)), Point(m1, transform(x1, y1, m1)))
    l2 = Line(Point(m0, transform(x2, y2, m0)), Point(m1, transform(x2, y2, m1)))
    intersection_point = l1.intersection(l2)
    return intersection_point

# function to draw lines in a diagram to manually show intersection points
def draw_pixel_lines(m0, m1, input_path: Path, out_path: Path, file_name = "pixels.json"):
    script_dir = Path(__file__).parent.parent
    linesToDraw = []
    target_path = script_dir / "resources" / file_name
    with open(target_path, 'r') as file:
        pixels = json.load(file)
        for x in range(len(pixels)):
            x1 = pixels[x][0]
            y1 = pixels[x][1]
            l = ((m0, transform(x1, y1, m0)), (m1, transform(x1, y1, m1)))
            linesToDraw.append(l)
        with Image.open(input_path) as image:
            image_background = Image.new("RGBA", image.size, "white")
        lines_overlay = Image.new("RGBA", image_background.size)
        lines_draw = ImageDraw.Draw(lines_overlay)

        for i in range(len(linesToDraw)):
            hue = i / int(len(linesToDraw))
            r, g, b = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
            color = (int(r * 255), int(g * 255), int (b * 255))
            lines_draw.line([linesToDraw[i][0], linesToDraw[i][1]], fill=color,width=1)
        Image.alpha_composite(image_background, lines_overlay).save(out_path)

# function to plot a chart of transformed lines to manually inspect intersections
def draw_chart(m0, m1, window_name = "Seed m/t chart", file_name = "pixels.json"):
    script_dir = Path(__file__).parent.parent
    target_path = script_dir / "resources" / file_name
    with open(target_path, 'r') as file:
        pixels = json.load(file)
    plt.figure(window_name)
    plt.ylabel('t (intercept)')
    plt.xlabel('m (slope)')
    for i in range(len(pixels)):
        xa = pixels[i][0]
        ya = pixels[i][1]
        x = [m0, m1]
        y = [transform(xa, ya, m0), transform(xa, ya, m1)]
        plt.plot(x, y, marker="o")
    plt.show()

def plot_lines(filename, merge: bool, legend):
    script_dir = Path(__file__).parent.parent
    target_path = script_dir / "resources" / filename
    fig, ax = plt.subplots()
    fig.canvas.manager.set_window_title("Cluster Chart")
    with open(target_path, "r") as f:
        data = json.load(f)
    merged = merge_segments(data, slope_tol=0.05, dist_tol=5.0)
    if merge:
        for i, obj in enumerate(merged):
            ax.plot([obj["start"][0], obj["end"][0]],
                    [obj["start"][1], obj["end"][1]], label=f"Merged Clusters {i}")
    else:
        for obj in data:
            x1, y1 = obj["start"]
            x2, y2 = obj["end"]
            ax.plot([x1, x2], [y1, y2], label=f"Seed=s{obj['target_seed']}")
    if legend:        
        ax.legend()
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.show()

def on_key(event):
    if event.key in ('q', 'Q'):
        plt.close('all')

# histogram function for the hough transform
def hough_transform_histogram(pixels, m_min, m_max, t_min, t_max, m_step, t_step):
    m_values = np.arange(m_min, m_max, m_step)
    t_values = np.arange(t_min, t_max, t_step)
    
    # Create a 2D array to count votes
    # Index 0 is m, Index 1 is t
    accumulator = np.zeros((len(m_values), len(t_values)), dtype=int)
    
    for x, y in pixels:
        for i, m in enumerate(m_values):
            t = -m * x + y
            # Find the closest t index
            j = np.argmin(np.abs(t_values - t))
            # Increment the vote
            accumulator[i, j] += 1
            
    return m_values, t_values, accumulator

from sympy import Line2D, Point, oo

def interpolate_lines(line1: Line2D, line2: Line2D, tol=1e-9) -> Line2D | None:
    m1, m2 = line1.slope, line2.slope

    # Both vertical → "similar" means same x; interpolate x
    if m1 == oo and m2 == oo:
        x1, x2 = line1.p1.x, line2.p1.x
        return Line2D(Point((x1 + x2) / 2, 0), slope=oo)

    # One vertical, one not → not similar
    if m1 == oo or m2 == oo:
        return None

    # Finite slopes: check closeness
    if abs(m1 - m2) > tol:
        return None

    # Average slope + average y-intercept
    m = (m1 + m2) / 2
    b1 = line1.p1.y - m1 * line1.p1.x
    b2 = line2.p1.y - m2 * line2.p1.x
    b = (b1 + b2) / 2

    return Line2D(Point(0, b), slope=m)
    
# generate intersection points from a file
def intersect_available_lines_via_file(m0, m1, file_name = "pixels.json"):
    script_dir = Path(__file__).parent.parent
    target_path = script_dir / "resources" / file_name
    with open(target_path, 'r') as file:
        pixels = json.load(file)
        processed_points = []
        for p in pixels:
            x, y = p
            t0 = transform(x, y, m0)
            t1 = transform(x, y, m1)
            processed_points.append({
                'orig': p,
                't0': t0,
                't1': t1
            })
        local_intersects = []
        for p1, p2 in itertools.combinations(processed_points, 2):
            l1 = Line(Point(m0, p1['t0']), Point(m1, p1['t1']))
            l2 = Line(Point(m0, p2['t0']), Point(m1, p2['t1']))

            intersection_point = l1.intersection(l2)
            if intersection_point:
                local_intersects.extend(intersection_point)

        Intersects.extend(local_intersects)

def intersect_available_lines_vectorized(m0, m1, clear = True, file_name="pixels.json"): # this is the ai-optimised intersect_available_lines_via_file method
    script_dir = Path(__file__).parent.parent # this function runs a lot faster, but also uses more memory
    target_path = script_dir / "resources" / file_name

    if clear:
        Intersects.clear()
        #print("Clearing Intersects!")
    
    with open(target_path, 'r') as file:
        pixels = np.array(json.load(file))

    t0 = transform(pixels[:, 0], pixels[:, 1], m0) 
    t1 = transform(pixels[:, 0], pixels[:, 1], m1)

    m0_i = np.full((len(pixels), 1), m0)
    m1_i = np.full((len(pixels), 1), m1)
    t0_i = t0.reshape(-1, 1)
    t1_i = t1.reshape(-1, 1)
    
    m0_j = m0_i.T
    m1_j = m1_i.T
    t0_j = t0_i.T
    t1_j = t1_i.T

    x1, y1 = m0_i, t0_i
    x2, y2 = m1_i, t1_i
    x3, y3 = m0_j, t0_j
    x4, y4 = m1_j, t1_j

    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    
    denom_safe = np.where(denom == 0, np.nan, denom)

    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom_safe
    
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom_safe

    i_indices, j_indices = np.tril_indices(len(pixels), k=-1)
    
    valid_px = px[i_indices, j_indices]
    valid_py = py[i_indices, j_indices]
    
    mask = ~np.isnan(valid_px)
    final_intersections = np.column_stack((valid_px[mask], valid_py[mask]))

    Intersects.extend([
        Point2D(x, y, evaluate=False) 
        for x, y in final_intersections
    ])
    print(f"Created {len(Intersects)} intersection points.")

def intersect_available_lines_vectorized_list(m0, m1, seed: find_seeds.Seed, clear = True):

    global Skip
    Skip = False

    if clear:
        Intersects.clear()
        #print("Clearing Intersects!")
    
    pixels = np.array(seed.pixels)

    t0 = transform(pixels[:, 0], pixels[:, 1], m0) 
    t1 = transform(pixels[:, 0], pixels[:, 1], m1)

    m0_i = np.full((len(pixels), 1), m0)
    m1_i = np.full((len(pixels), 1), m1)
    t0_i = t0.reshape(-1, 1)
    t1_i = t1.reshape(-1, 1)
    
    m0_j = m0_i.T
    m1_j = m1_i.T
    t0_j = t0_i.T
    t1_j = t1_i.T

    x1, y1 = m0_i, t0_i
    x2, y2 = m1_i, t1_i
    x3, y3 = m0_j, t0_j
    x4, y4 = m1_j, t1_j

    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    
    denom_safe = np.where(denom == 0, np.nan, denom)

    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom_safe
    
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom_safe

    i_indices, j_indices = np.tril_indices(len(pixels), k=-1)
    
    valid_px = px[i_indices, j_indices]
    valid_py = py[i_indices, j_indices]
    
    mask = ~np.isnan(valid_px)
    final_intersections = np.column_stack((valid_px[mask], valid_py[mask]))

    Intersects.extend([
        Point2D(x, y, evaluate=False) 
        for x, y in final_intersections
    ])
    if len(Intersects) == 0:
        Skip = True
        print("No intersection points found!")
    else:
        print(f"Created {len(Intersects)} intersection points.")

# function to return the most common point
def find_most_common_point(m0 = 0, mmax = 1):
    intersect_available_lines_via_file(m0, mmax)
    counter = Counter(Intersects)
    most_common_intersect = counter.most_common(1)[0][0]
    return most_common_intersect

# simple function to return a list of the nth most common points and their frequency
def find_n_most_common_points(n, m0= 0, mmax = 1, clr = False):
    intersect_available_lines_vectorized(m0, mmax, clr)
    counter = Counter(Intersects)
    n_most_common_intersects = counter.most_common(n)
    return n_most_common_intersects

# function to return the nth most common point
def find_n_most_common_point(n, m0= 0, mmax = 1, clr = False):
    global Skip
    #intersect_available_lines_vectorized(m0, mmax, clr)
    counter = Counter(Intersects)
    if n < 0 or n >= len(counter):
        print("!WARNING! Intersect Index out of bounds")
        Skip = True
        return None
    else:
        Skip = False
    n_most_common_intersects = counter.most_common(n + 1)[n][0]
    return n_most_common_intersects

# function to create a line from a point where (x|y) is (m|t)
def createLine(trf: Point2D):
    #print("Skip in createLine():", Skip)
    if not Intersects:
        return None
    p = Point(0, trf.y.evalf())
    m = trf.x.evalf()
    return Line(p, slope = m)

# def getLines(m0, m1, seed: find_seeds.Seed,n: int, clear, length):
#     intersect_available_lines_vectorized_list(m0, m1, seed, clear)
#     Lines = []
#     common_Intersects = []
#     #("Skip in getLines()", Skip)
#     if not Skip:
#         for x in range(n):
#             common_Intersects.append(find_n_most_common_point(x,m0,m1,clear))
#         t = Symbol('t')
#         for i in range(n):
#             p1 = createLine(common_Intersects[i]).arbitrary_point(t).subs(t, 0)
#             p2 = createLine(common_Intersects[i]).arbitrary_point(t).subs(t, length)
#             start = Circle.ConvertPoint2DtoTuple(p1)
#             end = Circle.ConvertPoint2DtoTuple(p2)
#             image_line = Line(Point(*start), Point(*end))
#             Lines.append(image_line)
#         return Lines

def line_parameters(line: Line) -> tuple[float, float]:
    slope = float(line.slope)
    intercept = float(line.p1.y - line.slope * line.p1.x)
    return slope, intercept


def average_line(lines: list[Line]) -> tuple[float, float]:
    parameters = [line_parameters(line) for line in lines]
    average_slope = sum(slope for slope, _ in parameters) / len(parameters)
    average_intercept = sum(intercept for _, intercept in parameters) / len(parameters)
    return average_slope, average_intercept


def line_for_cluster(
    cluster: list[find_seeds.Seed],
    lines_by_seed: dict[find_seeds.Seed, list[Line]],
) -> Line | None:
    cluster_lines = [
        line
        for seed in cluster
        for line in lines_by_seed.get(seed, [])
    ]

    if not cluster_lines:
        return None

    reference_x = (
        min(seed.bounds[0] for seed in cluster)
        + max(seed.bounds[2] for seed in cluster)
    ) / 2

    points = [
        (reference_x, float(line.slope) * reference_x + float(
            line.p1.y - line.slope * line.p1.x
        ))
        for line in cluster_lines
    ]

    average_slope, _ = average_line(cluster_lines)
    average_y = sum(y for _, y in points) / len(points)
    average_intercept = average_y - average_slope * reference_x

    min_x = min(seed.bounds[0] for seed in cluster)
    max_x = max(seed.bounds[2] for seed in cluster)

    start = Point(min_x, average_slope * min_x + average_intercept)
    end = Point(max_x, average_slope * max_x + average_intercept)

    return Line(start, end)


def draw_line(image_path: Path, output_path: Path, line: Line, color="blue"):
    with Image.open(image_path) as image:
        annotated_image = image.convert("RGBA")

    overlay = Image.new("RGBA", annotated_image.size)
    overlay_draw = ImageDraw.Draw(overlay)
    start = Circle.ConvertPoint2DtoTuple(line.p1)
    end = Circle.ConvertPoint2DtoTuple(line.p2)
    overlay_draw.line([start, end], fill=color, width=2)
    Image.alpha_composite(annotated_image, overlay).save(output_path)

def getLines(m0, m1, seed: find_seeds.Seed, n: int, clear, length, rqr):
    intersect_available_lines_vectorized_list(m0, m1, seed, clear)
    if rqr:
        print(
            f"Intersects: {len(Intersects)}, "
            f"unique: {len(Counter(Intersects))}, "
            f"required: {n}"
        )
    counter = Counter(Intersects)
    if len(counter) < n:
        print(f"Skipping seed: only {len(counter)} unique intersections found")
        return None

    common_intersects = [
        counter.most_common(n)[index][0]
        for index in range(n)
    ]

    lines = []
    t = Symbol("t")

    min_col, _, max_col, _ = seed.bounds
    seed_center_x = (min_col + max_col) / 2

    x_start = seed_center_x - length / 2
    x_end = seed_center_x + length / 2

    for intersection in common_intersects:
        line = createLine(intersection)

        p1 = line.arbitrary_point(t).subs(t, x_start)
        p2 = line.arbitrary_point(t).subs(t, x_end)

        start = Circle.ConvertPoint2DtoTuple(p1)
        end = Circle.ConvertPoint2DtoTuple(p2)
        lines.append(Line(Point(*start), Point(*end)))

    return lines

def line_intersects_seed(line, seed: find_seeds.Seed) -> bool:
    return any(line.contains(Point(c, r)) for c, r in seed.pixels)

# function to draw lines onto the seed
# def drawLines(image_path: Path,n: int, output_path: Path, m0, mmax, clr, length: int, color="red"):
#     Lines = []
#     with Image.open(image_path) as image:
#         annotated_image = image.convert("RGBA")
#     common_Intersects = []
#     intersect_available_lines_vectorized(m0, mmax, clr)
#     for x in range(n):
#         common_Intersects.append(find_n_most_common_point(x,m0,mmax, clr))
#     if not Skip:
#         line = Image.new("RGBA", annotated_image.size)
#         line_draw = ImageDraw.Draw(line)
#         t = Symbol('t')
#         for i in range(n):
#             p1 = createLine(common_Intersects[i]).arbitrary_point(t).subs(t, 0)
#             p2 = createLine(common_Intersects[i]).arbitrary_point(t).subs(t, length)
#             start = Circle.ConvertPoint2DtoTuple(p1)
#             end = Circle.ConvertPoint2DtoTuple(p2)
#             image_line = Line(Point(*start), Point(*end))
#             line_draw.line([start, end], fill=color, width=1)
#             Lines.append(image_line)
#         SeedLines.append(Lines)
#         Image.alpha_composite(annotated_image, line).save(output_path)

def drawLines(image_path: Path, n: int, output_path: Path, m0, mmax, clr, length: int, seed: find_seeds.Seed, rqr = bool, color="red"):
    #print("Skip: ", Skip)
    l: int
    if length is None:
        l = dynamicLength(seed, 0.5)
    else:
        l = length

    Lines = getLines(m0, mmax, seed, n, clr, l, rqr)
    if not Lines:
        print("Skipping seed...")
        return

    with Image.open(image_path) as image:
        annotated_image = image.convert("RGBA")

    if not Skip:
        line = Image.new("RGBA", annotated_image.size)
        line_draw = ImageDraw.Draw(line)
        for image_line in Lines:
            start = (image_line.p1.x, image_line.p1.y)
            end   = (image_line.p2.x, image_line.p2.y)
            line_draw.line([start, end], fill=color, width=1)

        SeedLines.append(Lines)
        Image.alpha_composite(annotated_image, line).save(output_path)

def dynamicLength(seed: find_seeds.Seed, f: float):
    min_col, min_row, max_col, max_row = seed.bounds
    l = math.hypot(max_col - min_col, max_row - min_row)
    l = l + l * f
    return int(l)

import numpy as np
from itertools import combinations

def merge_segments(segments, slope_tol=0.05, dist_tol=5.0):
    """
    Merge line segments that have similar slopes and nearby endpoints.

    Args:
        segments:   list of dicts with 'start', 'end', 'slope'
        slope_tol:  max |Δslope| for two segments to be mergeable
        dist_tol:   max distance between any endpoint pair for merging

    Returns:
        list of merged segment dicts (same format)
    """
    n = len(segments)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    def endpoints_close(a, b):
        for p in (a["start"], a["end"]):
            for q in (b["start"], b["end"]):
                if np.linalg.norm(np.array(p) - np.array(q)) < dist_tol:
                    return True
        return False

    # Union compatible pairs
    for i, j in combinations(range(n), 2):
        if abs(segments[i]["slope"] - segments[j]["slope"]) < slope_tol:
            if endpoints_close(segments[i], segments[j]):
                union(i, j)

    # Group indices by root
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)

    # Build merged segments
    merged = []
    for indices in groups.values():
        if len(indices) == 1:
            merged.append(segments[indices[0]])
            continue

        all_pts = []
        for idx in indices:
            all_pts.append(np.array(segments[idx]["start"]))
            all_pts.append(np.array(segments[idx]["end"]))

        avg_slope = float(np.mean([segments[i]["slope"] for i in indices]))
        start = all_pts[np.argmin([p[0] for p in all_pts])]
        end   = all_pts[np.argmax([p[0] for p in all_pts])]

        merged.append({
            "start": start.tolist(),
            "end":   end.tolist(),
            "slope": avg_slope,
        })

    return merged   

#print(find_n_most_common_point(5))