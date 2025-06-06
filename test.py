
from mylib import sum_vector, rbf_distance, RBFN

print(sum_vector([1.0, 2.0, 3.0]))  # attend 6.0

# Demonstrate distance function
print(rbf_distance([0.0, 0.0], [3.0, 4.0]))

# Simple RBFN usage
rbf = RBFN()
X = [[0.0], [1.0], [2.0], [3.0]]
y = [0.0, 1.0, 2.0, 3.0]
rbf.fit(X, y, n_centers=2)
print(rbf.predict([1.5]))
