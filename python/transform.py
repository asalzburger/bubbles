import json
from pathlib import Path
from sympy import Point, Point2D, Line, evalf, Symbol, Polygon, N, pi
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

def intersect_available_lines_vectorized(m0, m1, clear = False, file_name="pixels.json"): # this is the ai-optimised intersect_available_lines_via_file method
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
    #intersect_available_lines_vectorized(m0, mmax, clr)
    counter = Counter(Intersects)
    if n < 0 or n >= len(counter):
        print("!WARNING! Intersect Index out of bounds")
        return None
    n_most_common_intersects = counter.most_common(n + 1)[n][0]
    return n_most_common_intersects

# function to create a line from a point where (x|y) is (m|t)
def createLine(trf: Point2D):
    p = Point(0, trf.y.evalf())
    m = trf.x.evalf()
    return Line(p, slope = m)

def line_intersects_seed(line, seed: find_seeds.Seed) -> bool:
    return any(line.contains(Point(c, r)) for c, r in seed.pixels)

# function to draw lines onto the seed
def drawLines(image_path: Path,n: int, output_path: Path, m0, mmax, clr):
    Lines = []
    with Image.open(image_path) as image:
        annotated_image = image.convert("RGBA")
    common_Intersects = []
    intersect_available_lines_vectorized(m0, mmax, clr)
    for x in range(n):
        common_Intersects.append(find_n_most_common_point(x,m0,mmax, clr))
    line = Image.new("RGBA", annotated_image.size)
    line_draw = ImageDraw.Draw(line)
    t = Symbol('t')
    for i in range(n):
        p1 = createLine(common_Intersects[i]).arbitrary_point(t).subs(t, 0)
        p2 = createLine(common_Intersects[i]).arbitrary_point(t).subs(t, 100)
        start = Circle.ConvertPoint2DtoTuple(p1)
        end = Circle.ConvertPoint2DtoTuple(p2)
        image_line = Line(Point(*start), Point(*end))
        line_draw.line([start, end], fill="red", width=1)
        Lines.append(image_line)
    SeedLines.append(Lines)
    #print(Lines)
    Image.alpha_composite(annotated_image, line).save(output_path)

#print(find_n_most_common_point(5))