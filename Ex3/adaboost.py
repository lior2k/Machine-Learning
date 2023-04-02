import math
import random


class Point:

    def __init__(self, x: float, y: float, label: int):
        self.x = x
        self.y = y
        self.label = label
        self.weight = 0

    def get_x(self) -> float:
        return self.x

    def get_y(self) -> float:
        return self.y

    def get_label(self) -> int:
        return self.label

    def get_weight(self) -> float:
        return self.weight

    def set_weight(self, weight: float) -> None:
        self.weight = weight


# Represents an hypothesis
class Line:

    def __init__(self, p1: Point, p2: Point):
        self.__p1 = p1
        self.__p2 = p2
        self.weight = 0
        self.label_left_minus = True

    def get_weight(self) -> float:
        return self.weight

    def set_weight(self, weight: float) -> None:
        self.weight = weight

    # if the point is left of the line return label it -1 otherwise label it 1
    # if the line is horizontal, above points will be labeled -1 and below 1
    # if self.label_left_minus equals False do it the other way around.
    def label_point(self, p: Point):
        if ((self.__p2.get_x() - self.__p1.get_x()) * (p.get_y() - self.__p1.get_y()) -
                ((self.__p2.get_y() - self.__p1.get_y()) * (p.get_x() - self.__p1.get_x()))) > 0:
            if self.label_left_minus:
                return -1
            else:
                return 1
        else:
            if self.label_left_minus:
                return 1
            else:
                return -1

    def swap_dir(self):
        self.label_left_minus = not self.label_left_minus


def read_file() -> [Point]:
    points_set = []
    file = open('squares.txt')
    for line in file:
        data = line.split(' ')
        points_set.append(Point(float(data[0]), float(data[1]), 1 if int(data[2]) == 1 else -1))
    return points_set


def split_data(points_set: [Point]) -> ([Point], [Point]):
    sample = []
    test = []
    while len(points_set) > 0:
        if len(sample) == NUM_POINTS / 2:
            for point in points_set:
                test.append(point)
            break
        if len(test) == NUM_POINTS / 2:
            for point in points_set:
                sample.append(point)
            break
        index = random.randint(0, len(points_set) - 1)
        p = points_set.pop(index)
        coin_flip = random.random()
        if coin_flip < 0.5:
            test.append(p)
        else:
            sample.append(p)

    return sample, test


def generate_hypothesis_set(sample: [Point]) -> [Line]:
    lines = []
    for i in range(0, len(sample)):
        for j in range(i+1, len(sample)):
            lines.append(Line(sample[i], sample[j]))
    return lines


def compute_error(point_set: [Point], rule_list: [Line]):
    error = 0
    for Xi in point_set:
        Fxi = 0
        for rule in rule_list:
            Fxi += rule.get_weight() * rule.label_point(Xi)
        Hxi = -1 if Fxi < 0 else 1
        if Hxi != Xi.get_label():
            error += 1 / n
    return error


def adaboost(points_set: [Point]) -> ([Line], [Line]):
    s, t = split_data(points_set.copy())
    hypothesises = generate_hypothesis_set(s)

    for Xi in s:
        Xi.set_weight(1/n)

    # compute empirical for each hypo and if its more then 0.5 take the complementary(eg change direction) hypo
    for h in hypothesises:
        err = 0
        for Xi in s:
            err += Xi.get_weight() * 1 if h.label_point(Xi) != Xi.get_label() else 0
        if err > 0.5:
            h.swap_dir()

    best_rules = []
    # for iteration 1..k
    for i in range(0, k):
        min_err = float('inf')
        min_err_hypothesis = None
        # compute weighted error for each h in H
        for h in hypothesises:
            err = 0
            for Xi in s:
                err += Xi.get_weight() * 1 if h.label_point(Xi) != Xi.get_label() else 0
            # min_err_hypo -> h with min err
            if err < min_err:
                min_err = err
                min_err_hypothesis = h
        alpha = 0.5 * math.log((1-min_err) / min_err)
        min_err_hypothesis.set_weight(alpha)
        best_rules.append(min_err_hypothesis)

        # update points weight without 1/Zt because we don't know Zt yet. update later.
        Zt = 0
        for Xi in s:
            Dxi = Xi.get_weight() * math.exp(- alpha * min_err_hypothesis.label_point(Xi) * Xi.get_label())
            Xi.set_weight(Dxi)
            Zt += Xi.get_weight()

        # normalize weights with Zt
        for Xi in s:
            Xi.set_weight(Xi.get_weight() / Zt)

    # compute true error and empirical error for combination of 1..k rules eg the first rule alone, the first and
    # second rules combined and so on...
    empirical_errors = []
    true_errors = []
    for i in range(0, k):
        rules = [rule for rule in best_rules if best_rules.index(rule) <= i]

        # compute empirical error
        empirical_err = compute_error(s, rules)

        # compute true error
        true_error = compute_error(t, rules)

        empirical_errors.append(empirical_err)
        true_errors.append(true_error)

    return empirical_errors, true_errors


if __name__ == '__main__':
    points = read_file()
    NUM_POINTS = len(points)
    n = NUM_POINTS / 2
    k = 8
    avg_emp_errors = [0 for i in range(k)]
    avg_true_errors = [0 for i in range(k)]

    for i in range(0, int(n)):
        emp_errors, true_errors = adaboost(points)
        for j in range(k):
            avg_emp_errors[j] += emp_errors[j] / n
            avg_true_errors[j] += true_errors[j] / n

    for i in range(k):
        print(f'k = {i} empirical error: {avg_emp_errors[i]}, true error: {avg_true_errors[i]}')





