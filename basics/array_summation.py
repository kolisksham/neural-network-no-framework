import numpy as np

data = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
] 
a = np.array(data)

print(f"np.sum(a) = {np.sum(a)}")
print("="*100)
print(f"np.sum(a, axis = 0) = {np.sum(a, axis = 0)}") # [1+4+7  2+5+8  3+6+9] dim = 3 {1-D Array}
print("="*100)
print(f"np.sum(a, axis = 1) = {np.sum(a, axis = 1)}") # [1+2+3  4+5+6  7+8+9]
print("="*100)

# Keep Dims ( for 2-D Arrays)

print(f"np.sum(a, axis = 0, keepdims = True) = {np.sum(a, axis = 0, keepdims = True)}") 
print("="*100)
print(f"np.sum(a, axis = 1, keepdims = True) =\n{np.sum(a, axis = 1, keepdims = True)}") 
print("="*100)