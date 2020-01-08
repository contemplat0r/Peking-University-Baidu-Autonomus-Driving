def create_phone_number(digits):
    #create_phone_number([1, 2, 3, 4, 5, 6, 7, 8, 9, 0]) # => returns "(123) 456-7890"
    digits = [str(i) for i in digits]
    return '({}) {}-{}'.format(''.join(digits[:3]), ''.join(digits[3:6]), ''.join(digits[6:]))

print(create_phone_number([1, 2, 3, 4, 5, 6, 7, 8, 9, 0]))
