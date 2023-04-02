def read_file() -> list:
    vectors = []
    file = open('winnow_vectors.txt', 'r')
    for line in file:
        vec = []
        for i in range(0, len(line), 2):
            vec.append(int(line[i]))
        vectors.append(vec)
    return vectors


# calculate the inner product of a vector with the weights vector
def inner_prod(inner_vec: [int]):
    prod_sum = 0
    for i in range(DIM):
        prod_sum += inner_vec[i] * weightsVector[i]
    return prod_sum


# repair weights vector according to the type of mistake performed
def repair(vec: [int], val: int):
    for i in range(DIM):
        if vec[i] == 1:
            weightsVector[i] *= val


def run_winnow():
    global mistakes
    found_mistake = False
    for vec in vectors:
        mult = inner_prod(vec)
        current_guess = 0
        if mult >= DIM:
            current_guess = 1
        if current_guess != vec[DIM]:  # found mistake
            if current_guess == 1:  # guessed + but was -
                repair(vec, 0)  # set w[i] = 0
            else:  # guessed - but was +
                repair(vec, 2)  # set w[i] *= 2
            found_mistake = True
            break
    if found_mistake:
        mistakes += 1
        run_winnow()


if __name__ == '__main__':
    DIM = 20
    mistakes = 0
    weightsVector = []
    for i in range(DIM):
        weightsVector.append(1)

    vectors = read_file()

    run_winnow()
    print("Num of mistakes until no error: ", mistakes)
    print("Final weights vector: ", weightsVector)
