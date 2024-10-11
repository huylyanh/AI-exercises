# Ex1: Write a program to count positive and negative numbers in a list
# data1 = [-10, -21, -4, -45, -66, 93, 11, -4, -6, 12, 11, 4]

positive = 0
negative = 0

for item in data1:
    if item > 0:
        positive += 1
    elif item < 0:
        negative +=1

print(f"Positive numbers count: {positive}")
print(f"Negative numbers count: {negative}")

# Ex2: Given a list, extract all elements whose frequency is greater than k.
data2 = [4, 6, 4, 3, 3, 4, 3, 4, 3, 8]
k = 3

dict = {}
for item in data2:
    if item in dict:
        dict[item] += 1
    else:
        dict[item] = 1
print(dict)

result = []
for item in dict:
    if dict[item] > k:
        result.append(item)
print(result)   

# Ex3: find the strongest neighbour. Given an array of N positive integers.
# The task is to find the maximum for every adjacent pair in the array.
data3 = [4, 5, 6, 7, 3, 9, 11, 2, 10]

result = []
for i in range(len(data3)-1):
    num_max = max(data3[i], data3[i+1])
    result.append(num_max)
print(result)

# Ex4: print all Possible Combinations from the three Digits
data4 = [1, 2, 3]

result = []
for i in range(len(data4)):
    for j in range(len(data4)):
        for l in range(len(data4)):
            if j!=i and l!=j and l!=i:
                result.append([data4[i], data4[j], data4[l]])
print(result)

# Ex5: Given two matrices (2 nested lists), the task is to write a Python program
# to add elements to each row from initial matrix.
# For example: Input : test_list1 = [[4, 3, 5,], [1, 2, 3], [3, 7, 4]], test_list2 = [[1], [9], [8]]
# Output : [[4, 3, 5, 1], [1, 2, 3, 9], [3, 7, 4, 8]]
data5_list1 = [[4, 3, 5, ], [1, 2, 3], [3, 7, 4]]
data5_list2 = [[1, 3], [9, 3, 5, 7], [8]]

for i in range(len(data5_list1)):
    data5_list1[i].extend(data5_list2[i]) 
print(data5_list1)

# Ex6:  Write a program which will find all such numbers which are divisible by 7
# but are not a multiple of 5, between 2000 and 3200 (both included).
# The numbers obtained should be printed in a comma-separated sequence on a single line.

result = []
for num in range(2000, 3201):
    if num % 7 == 0 and num % 5 != 0:
        result.append(num)
print(result)

# Ex7: Write a program, which will find all such numbers between 1000 and 3000 (both included) such that each digit of the number is an even number.
# The numbers obtained should be printed in a comma-separated sequence on a single line.

result = []
for num in range(1000, 3001):
    num_to_str = str(num)
    isEven = True

    for i in num_to_str:
        if int(i) % 2 != 0:
            isEven = False
    
    if isEven == True:
        result.append(num_to_str)

print(",".join(result))

        


