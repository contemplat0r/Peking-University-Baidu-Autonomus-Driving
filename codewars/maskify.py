def maskify(orig_str):
    return "#" * (len(orig_str) - 4) + orig_str[-4:]


if __name__ == '__main__':
    print("4556364607935616", len("4556364607935616"))
    print(maskify("4556364607935616"))
    print(len(maskify("4556364607935616")))

    print(maskify("64607935616"))
    print(maskify("1"))
    print(maskify(""))
    print(maskify("Skippy"))
