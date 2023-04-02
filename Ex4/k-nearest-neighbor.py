import math
import random


class Point:

    def __init__(self, x: int, y: int, z: int, label: int):
        self.x = x
        self.y = y
        self.z = z
        self.label = label

    def get_x(self) -> int:
        return self.x

    def get_y(self) -> int:
        return self.y

    def get_z(self) -> int:
        return self.z

    def get_label(self) -> int:
        return self.label

    def __str__(self):
        return f"<{self.x}, {self.y}, {self.z}, {self.label}>"

    def __repr__(self):
        return f"<{self.x}, {self.y}, {self.z}, {self.label}>"


def read_file() -> [Point]:
    points_set = []
    file = open("haberman.data")
    for line in file:
        data = line.split(',')
        points_set.append(Point(int(data[0]), int(data[1]), int(data[2]), int(data[3])))
    return points_set


def split_data(points_set: [Point]) -> ([Point], [Point]):
    sample = []
    test = []
    while len(sample) < n / 2 and len(test) < n / 2:
        index = random.randint(0, len(points_set) - 1)
        p = points_set.pop(index)
        coin_flip = random.random()
        if coin_flip < 0.5:
            test.append(p)
        else:
            sample.append(p)

    while len(sample) < n / 2:
        sample.append(points_set.pop())

    while len(test) < n / 2:
        test.append(points_set.pop())

    return sample, test


def get_dist(p1: Point, p2: Point, p: int):
    if p == math.inf:
        return max(abs(p1.get_x() - p2.get_x()), abs(p1.get_y() - p2.get_y()), abs(p1.get_z() - p2.get_z()))
    if p == 1:
        return abs(p1.get_x() - p2.get_x()) + abs(p1.get_y() - p2.get_y()) + abs(p1.get_z() - p2.get_z())
    return math.pow(math.pow(p1.get_x() - p2.get_x(), 2) + math.pow(p1.get_y() - p2.get_y(), 2) + math.pow(p1.get_z() - p2.get_z(), 2), 1/2)


def get_k_nearest(p_set: [Point], point: [Point], k: int, p: int) -> [Point]:
    distances = [math.inf for _ in range(k)]
    neighbors = [None for _ in range(k)]
    for neighbor in p_set:
        current_distance = get_dist(neighbor, point, p)
        if current_distance < max(distances):
            neighbors[distances.index(max(distances))] = neighbor
            distances[distances.index(max(distances))] = current_distance
    return neighbors


def majority_vote(neighbors: [Point]) -> int:
    ones, twos = 0, 0
    for p in neighbors:
        if p.get_label() == 1:
            ones += 1
        else:
            twos += 1
    if ones > twos:
        return 1
    return 2


def nearest_neighbor(sample_set: [Point], test_set: [Point], k: int, p: int) -> (float, float):
    empirical_error = 0
    for point in sample_set:
        k_nearest = [point] if k == 1 else get_k_nearest(sample_set, point, k, p)
        label = majority_vote(k_nearest)
        if label != point.get_label():  # classified wrong
            empirical_error += 1 / len(sample_set)

    true_error = 0
    for point in test_set:
        k_nearest = get_k_nearest(sample_set, point, k, p)
        label = majority_vote(k_nearest)
        if label != point.get_label():  # classified wrong
            true_error += 1 / len(test_set)
    return empirical_error, true_error


def run():
    results = {}
    for p in ps:
        for k in range(1, 11, 2):
            results[(p, k)] = [0, 0]

    for i in range(100):
        s, t = split_data(points.copy())
        for p in ps:
            for k in range(1, 11, 2):
                empirical_error, true_error = nearest_neighbor(s, t, k, p)
                results[(p, k)] = [results[(p, k)][0] + empirical_error, results[(p, k)][1] + true_error]

    for key in results.keys():
        print(f"P = {key[0]}, K = {key[1]}: e` = {results[key][0] / 100} | e = {results[key][1] / 100} | diff = {abs((results[key][0] / 100) - (results[key][1] / 100))}")


if __name__ == '__main__':
    points = read_file()
    ps = [1, 2, math.inf]
    n = len(points)
    run()
