def unique_in_order(arr):
    arr = list(arr)
    if arr != []
        prev_item = arr[0]
        ordered_unique_list = [prev_item]
        for current_item in arr[1:]:
            if current_item != prev_item:
                ordered_unique_list.append(current_item)
                prev_item = current_item
        return ordered_unique_list
    else:
        return []


if __name__ == '__main__':
    print(unique_in_order('AAAABBBCCDAABBB'))   #== ['A', 'B', 'C', 'D', 'A', 'B']
    print(unique_in_order('ABBCcAD'))           #== ['A', 'B', 'C', 'c', 'A', 'D']
    print(unique_in_order([1,2,2,3,3]))         #== [1,2,3]
