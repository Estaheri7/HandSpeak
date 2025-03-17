import numpy as np

def euclidean_distance(pt1, pt2):
    return np.linalg.norm(np.array(pt1) - np.array(pt2))

def calculate_angle(a, b, c):
    ba = np.array(a) - np.array(b)
    bc = np.array(c) - np.array(b)

    inner_product = np.dot(ba, bc)
    norms = np.linalg.norm(ba) * np.linalg.norm(bc)
    cosine_angle = inner_product / norms

    angle = np.arccos(cosine_angle)
    return np.degrees(angle)

def is_above(p1, p2):
    return p1 > p2