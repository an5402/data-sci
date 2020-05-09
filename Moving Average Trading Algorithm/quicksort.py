
def quicksort(list):
    if len(list) > 1:
        left_part = []
        equals = []
        right_part = []
        pivot = list[0]
        for item in list:
            if item < pivot:
                left_part.append(item)
            elif item == pivot:
                equals.append(item)
            else:
                right_part.append(item)
        return quicksort(left_part) + equals + quicksort(right_part)
    else:
        return list


sample = [1,0,3,2,10,9,15,4]
print(quicksort(sample))


