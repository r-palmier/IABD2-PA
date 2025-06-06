from mylib import Perceptron, RBFN, rbf_distance

print("==== Test Perceptron ====")
# Problème XOR simplifié en -1 / 1
x_train = [[0, 0], [0, 1], [1, 0], [1, 1]]
y_train = [-1.0, 1.0, 1.0, -1.0]  # On encode XOR comme -1/+1 pour le perceptron

perceptron = Perceptron()
perceptron.fit(x_train, y_train, 10)

for x in x_train:
    y_pred = perceptron.predict(x)
    print(f"Perceptron({x}) = {y_pred}")

print("\n==== Test RBFN ====")
# Données en cloche autour de 1
x_train = [[0.0], [1.0], [2.0]]
y_train = [0.0, 1.0, 0.0]

rbf = RBFN()
rbf.fit(x_train, y_train, n_centers=2)

for x in [[0.0], [1.0], [2.0]]:
    y_pred = rbf.predict(x)
    print(f"RBFN({x}) = {y_pred}")

print("\n==== Distance ====")
print("Distance entre [1, 2] et [4, 6] :", rbf_distance([1, 2], [4, 6]))
