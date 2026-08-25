import json
from pathlib import Path
from sympy import Point, Line
from collections import Counter
import itertools

Intersects = []
def transform(x, y, m = 0):
    t = -m*x+y
    return t

def get_intersection_point(m0, m1, x1, y1, x2, y2):
    l1 = Line(Point(m0, transform(x1, y1, m0)), Point(m1, transform(x1, x2, m1)))
    l2 = Line(Point(m0, transform(x2, y2, m0)), Point(m1, transform(x1, x2, m1)))
    intersection_point = l1.intersection(l2)
    return intersection_point

def intersect_available_lines(m0, m1, file_name = "pixels.json"):
    script_dir = Path(__file__).parent.parent
    target_path = script_dir / "resources" / file_name
    with open(target_path, 'r') as file:
        pixels = json.load(file)
        #pixels_length = len(pixels)
        for point1, point2 in itertools.combinations(pixels, 2):
            x1, y1 = point1
            x2, y2 = point2
            Intersects.append(get_intersection_point(m0, m1, x1, y1, x2, y2))
            print(get_intersection_point(m0, m1, x1, y1, x2, y2)) # UNFINISHED

def find_most_common_point():
    intersect_available_lines()
    counter = Counter(Intersects)
    most_common_intersect = counter.most_common(1)[0][0]
    return most_common_intersect

intersect_available_lines(0, 1)