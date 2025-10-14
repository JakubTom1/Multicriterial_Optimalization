def get_dimensionality(points):
    if not points:
        return 0
    return len(points[0])


def naive_no_filter(points):
    dim = get_dimensionality(points)
    if dim == 0:
        return []

    pareto_front = []
    for p1 in points:
        dominated = False
        for p2 in points:
            if p1 != p2 and all(x2 <= x1 for x1, x2 in zip(p1, p2)) and any(x2 < x1 for x1, x2 in zip(p1, p2)):
                dominated = True
                break
        if not dominated:
            pareto_front.append(p1)
    return pareto_front


def naive_with_filter(points):
    dim = get_dimensionality(points)
    if dim == 0:
        return []

    filtered_points = []
    for p in points:
        if not any(all(x <= y for x, y in zip(other, p)) and any(x < y for x, y in zip(other, p))
                   for other in filtered_points):
            filtered_points = [point for point in filtered_points
                               if not (all(x <= y for x, y in zip(p, point)) and
                                       any(x < y for x, y in zip(p, point)))]
            filtered_points.append(p)
    return filtered_points

def find_ideal_point(points):
    if not points:
        return None
    dim = get_dimensionality(points)
    ideal_point = []
    for d in range(dim):
        ideal_point.append(min(point[d] for point in points))
    return tuple(ideal_point)


def calculate_distance(point, ideal_point):
    return sum((p - i) ** 2 for p, i in zip(point, ideal_point))


def ideal_point_algorithm(points):
    if not points:
        return []

    ideal_point = find_ideal_point(points)
    points_with_distances = [(point, calculate_distance(point, ideal_point)) for point in points]
    sorted_points = sorted(points_with_distances, key=lambda x: x[1])

    pareto_front = []
    for point, _ in sorted_points:
        dominated = False
        for p in pareto_front:
            if all(x <= y for x, y in zip(p, point)):
                dominated = True
                break
        if not dominated:
            pareto_front.append(point)

    return pareto_front


if __name__ == "__main__":
    X_2D = [(5, 5), (3, 6), (4, 4), (5, 3), (3, 3), (1, 8), (3, 4), (4, 5), (3, 10), (6, 6), (4, 1), (3, 5)]

    X_3D = [(1, 2, 3), (2, 2, 2), (3, 1, 2), (2, 1, 3), (1, 3, 2), (2, 3, 1), (3, 2, 1)]

    X_4D = [(1, 2, 3, 4), (2, 2, 2, 2), (1, 1, 3, 4), (1, 2, 2, 3), (3, 1, 2, 1), (2, 3, 1, 4)]

    print("Input 2D:")
    print(X_2D)
    print("\nNaive no filter 2D:")
    print(naive_no_filter(X_2D))
    print("\nNaive with filter 2D:")
    print(naive_with_filter(X_2D))
    print("\nIdeal point algorithm 2D:")
    print(ideal_point_algorithm(X_2D))
    print("Ideal point:", find_ideal_point(X_2D))

    print("\nInput 3D:")
    print(X_3D)
    print("\nNaive no filter 3D:")
    print(naive_no_filter(X_3D))
    print("\nNaive with filter 3D:")
    print(naive_with_filter(X_3D))
    print("\nIdeal point algorithm 3D:")
    print(ideal_point_algorithm(X_3D))
    print("Ideal point:", find_ideal_point(X_3D))

    print("\nInput 4D:")
    print(X_4D)
    print("\nNaive no filter 4D:")
    print(naive_no_filter(X_4D))
    print("\nNaive with filter 4D:")
    print(naive_with_filter(X_4D))
    print("\nIdeal point algorithm 4D:")
    print(ideal_point_algorithm(X_4D))
    print("Ideal point:", find_ideal_point(X_4D))
