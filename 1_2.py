import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Припустимо, що ви маєте вже визначені матриці B, C1, A1
B = np.array([[1, 0], [0, 1]])  # Змініть на вашу матрицю
C1 = np.array([[1, 2], [3, 4]])  # Змініть на вашу матрицю
A1 = np.array([[0.5, 1.5], [2, 3.5]])  # Змініть на вашу матрицю

# Обчислення матриці G
G = np.linalg.inv(B) @ (C1 - A1)

# Функція для розв'язку системи диференційних рівнянь замкненої системи
def closed_system(t, x):
    return G @ x

# Початкові умови для x1
initial_x1 = np.array([1, 0])  # Змініть на ваші початкові умови

# Часовий інтервал інтеграції
t_span = (0, 10)
t_eval = np.linspace(*t_span, 100)

# Розв'язання системи рівнянь для замкненої системи
sol = solve_ivp(closed_system, t_span, initial_x1, t_eval=t_eval)

# Розв'язання системи рівнянь для траєкторії з технологічним темпом
scaled_initial_conditions = initial_x1 * 2  # Масштабування для відображення зміни умов
sol_technological = solve_ivp(closed_system, t_span, scaled_initial_conditions, t_eval=t_eval)

# Підготовка даних для векторного поля
X, Y = np.meshgrid(np.linspace(0, 10000, 20), np.linspace(0, 10000, 20))
U, V = np.zeros(X.shape), np.zeros(Y.shape)
NI, NJ = X.shape

for i in range(NI):
    for j in range(NJ):
        x_point = np.array([X[i, j], Y[i, j]])
        velocity = G @ x_point
        U[i, j] = velocity[0]
        V[i, j] = velocity[1]

# Візуалізація векторного поля та траєкторій
plt.quiver(X, Y, U, V, color='red', alpha=0.5)
plt.plot(sol.y[0], sol.y[1], 'b-', label='Замкнена система')
plt.plot(sol_technological.y[0], sol_technological.y[1], 'g-', label='Траєкторія з технологічним темпом')
plt.title('Фазовий портрет $x_1(t)$ з векторним полем')
plt.xlabel('$x_{1,1}$')
plt.ylabel('$x_{1,2}$')
plt.legend()
plt.grid(True)
plt.show()
