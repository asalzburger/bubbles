from sympy import symbols, Eq, solve, Point2D, evalf
from operator import itemgetter


def calculateRadius(Ax, Ay, Bx, By, Cx, Cy):

    xm, ym, r = symbols('xm ym r')
    eq1 = Eq((Ax - xm)**2 + (Ay - ym)**2, r**2)
    eq2 = Eq((Bx - xm)**2 + (By - ym)**2, r**2)
    eq3 = Eq((Cx - xm)**2 + (Cy - ym)**2, r**2)

    solution = solve((eq1, eq2, eq3), (xm, ym, r))

    for sol in solution:
        r_val = sol[2]
        if r_val.is_real and r_val > 0:
            return round(float(r_val), 2)

    return None
    
def calculateMidPoint(Ax, Ay, Bx, By, Cx, Cy):
    xm, ym, r = symbols('xm ym r')
    eq1 = Eq((Ax - xm)**2 + (Ay - ym)**2, r**2)
    eq2 = Eq((Bx - xm)**2 + (By - ym)**2, r**2)
    eq3 = Eq((Cx - xm)**2 + (Cy - ym)**2, r**2)
    
    solution = solve((eq1, eq2, eq3), (xm, ym, r))
    
    for sol in solution:
        return Point2D(sol[0], sol[1])
    
def calculateBField(R, P): # unfinished
    B = R/P
    print(B)

def calculateRadiusfromPoints(p1: Point2D, p2: Point2D, p3: Point2D):
    x1 = p1.x.evalf()
    y1 = p1.y.evalf()
    x2 = p2.x.evalf()
    y2 = p2.y.evalf()
    x3 = p3.x.evalf()
    y3 = p3.y.evalf()
    return calculateRadius(x1, y1, x2, y2, x3, y3)

def calculateMidPointfromPoints(p1: Point2D, p2: Point2D, p3: Point2D):
    x1 = p1.x.evalf()
    y1 = p1.y.evalf()
    x2 = p2.x.evalf()
    y2 = p2.y.evalf()
    x3 = p3.x.evalf()
    y3 = p3.y.evalf()
    return calculateMidPoint(x1, y1, x2, y2, x3, y3)

def ConvertTupletoPoint2D(t: tuple):
    x = itemgetter(0)(t)
    y = itemgetter(1)(t)
    return Point2D(x, y)

def IsPointOnCircle(p1: Point2D, p2: Point2D, p3: Point2D, ptest: Point2D):
    x = ptest.x.evalf()
    y = ptest.y.evalf()
    r = calculateRadiusfromPoints(p1, p2, p3)
    h = calculateMidPointfromPoints(p1, p2, p3).x.evalf()
    k = calculateMidPointfromPoints(p1, p2, p3).y.evalf()
    if ((x-h)**2) + ((y-k)**2) == r**2:
        return True
    else:
        return False

