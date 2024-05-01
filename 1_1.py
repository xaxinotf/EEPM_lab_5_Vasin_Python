import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Визначення констант та початкових умов
A11 = np.array([[0.2, 0.7], [0.4, 0.1]])
A12 = np.array([[0.4, 0.1], [0.3, 0.4]])
A21 = np.array([0.3, 0.4])
A22 = 0.4
initial_x1 = np.array([550, 300])

# Функція для c1(t)
def c1(t):
    return np.array([350, 150]) * np.exp(0.01 * t)

# Система диференціальних рівнянь
def model(t, y):
    x1, x2 = y[:2], y[2]
    dx1dt = A11 @ x1 + A12 @ np.array([x2, x2]) + c1(t)
    dx2dt = A21 @ x1 + A22 * x2 + 40
    return np.concatenate((dx1dt, [dx2dt]))

# Часовий інтервал інтеграції
t_span = (0, 5)
t_eval = np.linspace(*t_span, 500)

# Розв'язання системи рівнянь
sol = solve_ivp(model, t_span, np.concatenate((initial_x1, [40])), t_eval=t_eval)

# Графіки динаміки векторів x1(t) та x2(t)
plt.figure(figsize=(12, 8))
plt.plot(t_eval, sol.y[0], label='$x_{1,1}(t)$')
plt.plot(t_eval, sol.y[1], label='$x_{1,2}(t)$')
plt.plot(t_eval, sol.y[2], label='$x_2(t)$')
plt.title('Динаміка векторів $x_1(t)$ та $x_2(t)$')
plt.xlabel('Час, t')
plt.ylabel('Значення')
plt.legend()
plt.show()

# Обчислення технологічного темпу зростання
eigenvalues, eigenvectors = np.linalg.eig(A11)  # Використовуємо матрицю A11
technological_growth_rate = np.max(np.real(eigenvalues))
max_eigen_index = np.argmax(np.real(eigenvalues))
corresponding_eigenvector = eigenvectors[:, max_eigen_index]

print("Технологічний темп зростання (λ):", technological_growth_rate)
print("Власний вектор, що відповідає технологічному темпу зростання:", corresponding_eigenvector)
