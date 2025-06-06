from mylib import Perceptron

# Table de vérité du AND logique
X = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
Y = [-1.0, -1.0, -1.0, 1.0]  # cible: -1 sauf pour [1.0, 1.0]

model = Perceptron()
model.fit(X, Y, iterations=10)

for x in X:
    print(f"{x} → {model.predict(x)}")
