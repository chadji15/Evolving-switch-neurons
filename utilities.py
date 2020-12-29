import random


def clamp(x,low,high):
    if x < low:
        return low
    if x > high:
        return high
    return x

def heaviside(x):
    if x < 0:
        return 0
    return 1

def mult(w_inputs):
    product = 1
    for w_i in w_inputs:
        product *= w_i
    return product

def shuffle_lists(list1, list2):
    temp = list(zip(list1, list2))
    random.shuffle(temp)
    list1, list2 = [], []
    for a, b in temp:
        list1.append(a)
        list2.append(b)
    return list1, list2