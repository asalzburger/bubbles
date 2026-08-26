import json
from pathlib import Path
from sympy import Point, Point2D, Line, evalf, Symbol
from collections import Counter
import itertools
import Circle
from PIL import Image, ImageDraw
import numpy as np
import colorsys

Intersects = []
def transform(x, y, m = 0):
    t = -m*x+y
    return t

def get_intersection_point(m0, m1, x1, y1, x2, y2):
    l1 = Line(Point(m0, transform(x1, y1, m0)), Point(m1, transform(x1, y1, m1)))
    l2 = Line(Point(m0, transform(x2, y2, m0)), Point(m1, transform(x2, y2, m1)))
    intersection_point = l1.intersection(l2)
    return intersection_point

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
            #l = Line(Point(m0, transform(x1, y1, m0)), Point(m1, transform(x1, y1, m1)))
            linesToDraw.append(l)
        #print(linesToDraw)
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

def intersect_available_lines_vectorized(m0, m1, file_name="pixels.json"): # this is the ai-optimised intersect_available_lines_via_file method
    script_dir = Path(__file__).parent.parent # this function runs a lot faster, but also uses more memory
    target_path = script_dir / "resources" / file_name
    
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

    
def find_most_common_point(m0 = 0, mmax = 1):
    intersect_available_lines_via_file(m0, mmax)
    counter = Counter(Intersects)
    most_common_intersect = counter.most_common(1)[0][0]
    return most_common_intersect

def find_n_most_common_points(n, m0= 0, mmax = 1):
    intersect_available_lines_vectorized(m0, mmax)
    counter = Counter(Intersects)
    n_most_common_intersects = counter.most_common(n)
    return n_most_common_intersects

def find_n_most_common_point(n, m0= 0, mmax = 1):
    intersect_available_lines_vectorized(m0, mmax)
    counter = Counter(Intersects)
    if n < 0 or n >= len(counter):
        print("!WARNING! Intersect Index out of bounds")
        return None
    n_most_common_intersects = counter.most_common(n + 1)[n][0]
    return n_most_common_intersects

def createLine(trf: Point2D):
    p = Point(0, trf.y.evalf())
    m = trf.x.evalf()
    return Line(p, slope = m)

def drawLines(image_path: Path,n: int, output_path: Path, m0, mmax):
    with Image.open(image_path) as image:
        annotated_image = image.convert("RGBA")
    common_Intersects = []
    for x in range(n):
        common_Intersects.append(find_n_most_common_point(x,m0,mmax))
    line = Image.new("RGBA", annotated_image.size)
    line_draw = ImageDraw.Draw(line)
    t = Symbol('t')
    for i in range(n):
        p1 = createLine(common_Intersects[i]).arbitrary_point(t).subs(t, 0)
        p2 = createLine(common_Intersects[i]).arbitrary_point(t).subs(t, 100)
        start = Circle.ConvertPoint2DtoTuple(p1)
        end = Circle.ConvertPoint2DtoTuple(p2)
        line_draw.line([start, end], fill="red", width=1)
    Image.alpha_composite(annotated_image, line).save(output_path)

#print(find_n_most_common_points(5))