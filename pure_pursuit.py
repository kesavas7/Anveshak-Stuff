import csv
import math
def pure_pursuit_control(x, y, theta, path, L_d=1.0, v=1.0):
    lookahead = None
    filename='path.csv'
    with open(filename, 'r') as f:
    reader = csv.reader(f)
    next(reader) 
    for row in reader:
        px = float(row[0])
        py = float(row[1])
        dist = math.sqrt((px - x)**2 + (py - y)**2)
        if dist >= L_d:
            lookahead = (px, py)
            break
    if lookahead is None:
        return 0.0, 0.0 
    lx, ly = lookahead
    dx = lx - x
    dy = ly - y
    alpha = math.atan2(dy, dx) - theta
    alpha = normalize_angle(alpha)
    kappa = 2 * math.sin(alpha) / L_d
    omega = v * kappa
    return v, omega

