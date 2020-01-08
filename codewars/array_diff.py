
def array_diff(ar1, ar2):
    return [x for x in ar1 if x not in set(ar1).intersection(ar2)]

if __name__ == '__main__':
    print(array_diff([1, 2, 2, 4, 3, 3, 3, 0], [3, 3, 4])) 
