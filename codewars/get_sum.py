def get_sum(a, b):
    return sum(range(min(a, b), max(a, b) +1))


if __name__ == '__main__':
    '''
    print(get_sum(1, 0) == 1)   // 1 + 0 = 1
    print(get_sum(1, 2) == 3)   // 1 + 2 = 3
    print(get_sum(0, 1)) == 1)   // 0 + 1 = 1
    print(get_sum(1, 1) == 1)   // 1 Since both are same
    print(get_sum(-1, 0) == -1) // -1 + 0 = -1
    print(get_sum(-1, 2) == 2) // -1 + 0 + 1 + 2 = 2
    '''
    print(get_sum(1, 0) == 1)  
    print(get_sum(1, 2) == 3)  
    print(get_sum(0, 1) == 1)
    print(get_sum(1, 1) == 1)
    print(get_sum(-1, 0) == -1)
    print(get_sum(-1, 2) == 2)
