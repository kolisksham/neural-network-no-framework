import numpy as np

dat = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]

a = np.array(dat)

# case 1 (wrong)

res1 = a - np.max(a, axis = 1) # [[3 6 9], [3 6 9], [3 6 9]]
print(res1)

# treating array as 1-D vector and broadcasting column-wise
print("="*100)
#case 2 (correct)

res2 = a - np.max(a, axis = 1, keepdims= True) # [[3 3 3], [6 6 6], [9 9 9]]
print(res2)

# treating array as 2-D array and vroadcasting row-wise
