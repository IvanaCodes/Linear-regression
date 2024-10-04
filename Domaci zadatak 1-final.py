# Model sadrzi i uvodjenje i neuvodjenje polinom.karakteristika
# Ne daje dobre rezultate

# Importovanje biblioteka
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json


# Ucitavanje podataka
data = pd.read_csv('data.csv', header=None)

# Ucitavanje obelezja i labele
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Dodavanje kolone za bias matrici odlika
X.insert(0, "Bias", 1)

# Hiperparametri
alpha = 0.1
learning_rate = 0.01
iterations = 50000
k = 5

# Inicijalizujemo promenljive da bismo sačuvali najbolje rezultate
best_theta_without_poly = None
best_cost_without_poly = float('inf')
best_theta_with_poly = None
best_cost_with_poly = float('inf')

# Podela podataka u skupove za train i testiranje
split_index = int(0.7 * len(data))
X_train = X.iloc[:split_index]
y_train = y.iloc[:split_index]
X_test = X.iloc[split_index:]
y_test = y.iloc[split_index:]

# Inicijalizacija lista za čuvanje rezultata iteracije
iteration_results_without_poly = []
iteration_results_with_poly = []

# Definisemo lasso_gradient_descent funkciju
def lasso_gradient_descent(X, y, alpha, learning_rate, iterations):
    m = len(y)
    theta = np.zeros(X.shape[1])
    for iteration in range(iterations):
        predictions = X @ theta
        errors = predictions - y
        gradient = (1 / m) * X.T.dot(errors) + alpha * np.sign(theta)
        theta = theta - learning_rate * gradient
        cost = lasso_cost_function(X, y, theta, alpha)
        if iteration % 1000 == 0:
            print(f"Iteration {iteration}, Cost: {cost}")
    return theta, cost

# Definisemo lasso_cost_function
def lasso_cost_function(X, y, theta, alpha):
    m = len(y)
    predictions = X @ theta
    cost = (1 / (2 * m)) * np.sum((predictions - y) ** 2) + alpha * np.sum(np.abs(theta))
    return cost

# Definisemo add_polynomial_features function za dodavanje polinomijalnih karakteristika
def add_polynomial_features(X, degree=2):
    X_poly = X.copy()  # Copy original features
    for col in X.columns:
        for d in range(2, degree + 1):
            new_col = f"{col}^{d}"  # New column name
            X_poly[new_col] = X[col] ** d  # Add polynomial features
    return X_poly

# Petlja kroz k iteracija
for i in range(k):
    # Treniranje modela bez polinomialnih odlika
    theta_without_poly, cost_without_poly = lasso_gradient_descent(X_train, y_train, alpha, learning_rate, iterations)
    iteration_results_without_poly.append({'theta': theta_without_poly.tolist(), 'cost': cost_without_poly})

    if cost_without_poly < best_cost_without_poly:
        best_cost_without_poly = cost_without_poly
        best_theta_without_poly = theta_without_poly

    # Treniranje modela sa polinomialnim odlikama
    X_train_poly = add_polynomial_features(X_train)
    X_test_poly = add_polynomial_features(X_test)
    theta_with_poly, cost_with_poly = lasso_gradient_descent(X_train_poly, y_train, alpha, learning_rate, iterations)
    iteration_results_with_poly.append({'theta': theta_with_poly.tolist(), 'cost': cost_with_poly})

    if cost_with_poly < best_cost_with_poly:
        best_cost_with_poly = cost_with_poly
        best_theta_with_poly = theta_with_poly

# Sačuvamo rezultate ponavljanja u JSON datoteke
with open('iteration_results_without_poly.json', 'w') as json_file:
    json.dump(iteration_results_without_poly, json_file)

with open('iteration_results_with_poly.json', 'w') as json_file:
    json.dump(iteration_results_with_poly, json_file)

# Izdvojimo bias
bias_theta_without_poly = best_theta_without_poly[0]
bias_theta_with_poly = best_theta_with_poly[0]

# Predikcije
predictions_train_without_poly = X_train @ np.array(best_theta_without_poly)
predictions_test_without_poly = X_test @ np.array(best_theta_without_poly)

predictions_train_with_poly = X_train_poly @ np.array(best_theta_with_poly)
predictions_test_with_poly = X_test_poly @ np.array(best_theta_with_poly)

# Plotovanje rezultata
plt.figure(figsize=(8, 6))
plt.scatter(y_train, predictions_train_without_poly)
plt.plot([min(y_train), max(y_train)], [min(y_train), max(y_train)], '--', color='red')
plt.xlabel('Actual Train')
plt.ylabel('Predicted Train')
plt.title('Actual vs Predicted Values (Without Polynomial Features)')
plt.grid(True)
plt.show()

plt.figure(figsize=(8, 6))
plt.scatter(y_test, predictions_test_without_poly)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], '--', color='red')
plt.xlabel('Actual Test')
plt.ylabel('Predicted Test')
plt.title('Actual vs Predicted Values (Without Polynomial Features)')
plt.grid(True)
plt.show()

plt.figure(figsize=(8, 6))
plt.scatter(y_train, predictions_train_with_poly)
plt.plot([min(y_train), max(y_train)], [min(y_train), max(y_train)], '--', color='blue')
plt.xlabel('Actual Train')
plt.ylabel('Predicted Train')
plt.title('Actual vs Predicted Values (With Polynomial Features)')
plt.grid(True)
plt.show()

plt.figure(figsize=(8, 6))
plt.scatter(y_test, predictions_test_with_poly)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], '--', color='blue')
plt.xlabel('Actual Test')
plt.ylabel('Predicted Test')
plt.title('Actual vs Predicted Values (With Polynomial Features)')
plt.grid(True)
plt.show()

# Tacnost bez polinomialnih odlika
accuracy_train_without_poly = np.mean((np.round(predictions_train_without_poly) == y_train).astype(int)) * 100
accuracy_test_without_poly = np.mean((np.round(predictions_test_without_poly) == y_test).astype(int)) * 100

print("Accuracy on training set (without polynomial features): {:.2f}%".format(accuracy_train_without_poly))
print("Accuracy on test set (without polynomial features): {:.2f}%".format(accuracy_test_without_poly))

# Tacnost sa polynomialnim odlikama
accuracy_train_with_poly = np.mean((np.round(predictions_train_with_poly) == y_train).astype(int)) * 100
accuracy_test_with_poly = np.mean((np.round(predictions_test_with_poly) == y_test).astype(int)) * 100

print("Accuracy on training set (with polynomial features): {:.2f}%".format(accuracy_train_with_poly))
print("Accuracy on test set (with polynomial features): {:.2f}%".format(accuracy_test_with_poly))


# Funkcije metrike evaluacije
def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def r_squared(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)

mse = mean_squared_error(y_train, predictions_train_without_poly)
mse_p= mean_squared_error(y_train, predictions_train_with_poly)
rmse = root_mean_squared_error(y_train, predictions_train_without_poly)
rmse_p = root_mean_squared_error(y_train, predictions_train_with_poly)
mae = mean_absolute_error(y_train, predictions_train_without_poly)
mae_p = mean_absolute_error(y_train, predictions_train_with_poly)
r2 = r_squared(y_train, predictions_train_without_poly)
r2_p = r_squared(y_train, predictions_train_with_poly)

print("Mean Squared Error:", mse, mse_p)
print("Root Mean Squared Error:", rmse,rmse_p)
print("Mean Absolute Error:", mae,mae_p)
print("R-squared:", r2,r2_p)

# Tacnost ne deluje dobro, vrv se ne racuna onako
# Hipoteze su postavljene drugacije X @ theta ( nije kao sto sam mislila da treba theta_0 + theta_1*X + theta_2*X**2
# Hiperparametri nisu menjani


