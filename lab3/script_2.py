import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
import os


def solve_analyze_plot():
    print("--- НАЧАЛО РЕШЕНИЯ ---\n")

    # Создаем папку для изображений, если её нет
    os.makedirs('images', exist_ok=True)

    # 1. Определение символьных переменных
    x, y, lam1, lam2 = sp.symbols('x y lambda_1 lambda_2')

    # 2. Ввод условий задачи
    # Целевая функция
    f = 4 * x ** 2 + 2 * x * y + 5 * y ** 2 - 4 * x + 6 * y
    # Ограничения
    g1 = x - y - 2
    g2 = 3 * x + y - 1

    # 3. Метод множителей Лагранжа
    L = f - lam1 * g1 - lam2 * g2  # Исправлено: должно быть минус для минимизации

    # Градиент Лагранжиана (система уравнений)
    equations = [
        sp.diff(L, x),
        sp.diff(L, y),
        sp.diff(L, lam1),
        sp.diff(L, lam2)
    ]

    # Решение системы
    sol = sp.solve(equations, (x, y, lam1, lam2))

    # Конвертация результатов в числа float для удобства
    sol_x = float(sol[x])
    sol_y = float(sol[y])
    sol_l1 = float(sol[lam1])
    sol_l2 = float(sol[lam2])
    f_val = float(f.subs({x: sol_x, y: sol_y}))

    print(f"1. Найденная точка экстремума:")
    print(f"   x* = {sol_x}")
    print(f"   y* = {sol_y}")
    print(f"   f(x*, y*) = {f_val}")
    print(f"   Множители: λ₁ = {sol_l1}, λ₂ = {sol_l2}\n")

    # 4. Проверка критерия Сильвестра (окаймленная матрица Гессе)
    print("2. Проверка критерия Сильвестра (окаймленная матрица Гессе):")

    # Матрица Гессе целевой функции
    H_f = sp.Matrix([
        [sp.diff(f, x, x), sp.diff(f, x, y)],
        [sp.diff(f, y, x), sp.diff(f, y, y)]
    ])

    # Матрица Якоби ограничений
    J = sp.Matrix([
        [sp.diff(g1, x), sp.diff(g1, y)],
        [sp.diff(g2, x), sp.diff(g2, y)]
    ])

    # Окаймленная матрица Гессе
    B = sp.Matrix([
        [0, 0, J[0, 0], J[1, 0]],
        [0, 0, J[0, 1], J[1, 1]],
        [J[0, 0], J[0, 1], H_f[0, 0], H_f[0, 1]],
        [J[1, 0], J[1, 1], H_f[1, 0], H_f[1, 1]]
    ])

    det_B = float(B.det())

    print(f"   Матрица Гессе H(f):")
    print(f"   {H_f}")
    print(f"\n   Матрица Якоби ограничений J:")
    print(f"   {J}")
    print(f"\n   Окаймленная матрица Гессе B:")
    print(f"   {B}")
    print(f"\n   Определитель det(B) = {det_B}")

    if det_B > 0:
        character = "ЛОКАЛЬНЫЙ МИНИМУМ"
        print(f"   >> ВЫВОД: det(B) > 0 ⇒ {character}")
    elif det_B < 0:
        character = "ЛОКАЛЬНЫЙ МАКСИМУМ"
        print(f"   >> ВЫВОД: det(B) < 0 ⇒ {character}")
    else:
        character = "ТРЕБУЕТСЯ ДОПОЛНИТЕЛЬНЫЙ АНАЛИЗ"
        print(f"   >> ВЫВОД: det(B) = 0 ⇒ {character}")

    # 5. Визуализация 1: Контурный график с ограничениями
    plt.figure(figsize=(12, 10))

    # Сетка координат
    x_vis = np.linspace(sol_x - 2, sol_x + 2, 400)
    y_vis = np.linspace(sol_y - 2, sol_y + 2, 400)
    X, Y = np.meshgrid(x_vis, y_vis)
    Z = 4 * X ** 2 + 2 * X * Y + 5 * Y ** 2 - 4 * X + 6 * Y

    # Линии уровня (контуры целевой функции)
    levels = np.linspace(f_val - 5, f_val + 50, 25)
    cp = plt.contour(X, Y, Z, levels=levels, cmap='viridis', alpha=0.6)
    plt.clabel(cp, inline=True, fontsize=8)

    # Ограничения
    # g1: y = x - 2
    plt.plot(x_vis, x_vis - 2, 'r--', label='g₁: x - y - 2 = 0', linewidth=2.5)
    # g2: y = 1 - 3x
    plt.plot(x_vis, 1 - 3 * x_vis, 'b--', label='g₂: 3x + y - 1 = 0', linewidth=2.5)

    # Найденная точка
    plt.plot(sol_x, sol_y, 'ro', markersize=12, label=f'Точка экстремума ({sol_x:.3f}, {sol_y:.3f})', zorder=10)
    plt.text(sol_x + 0.05, sol_y + 0.05, f'M*', fontsize=12, fontweight='bold', 
             bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

    plt.title(f'Условная оптимизация методом множителей Лагранжа\nf(x*, y*) = {f_val:.4f}, Характер: {character}', 
              fontsize=14, fontweight='bold')
    plt.xlabel('x', fontsize=12)
    plt.ylabel('y', fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(fontsize=10, loc='upper right')
    plt.axis('equal')

    # Центрируем график вокруг решения
    plt.xlim(sol_x - 1.5, sol_x + 1.5)
    plt.ylim(sol_y - 1.5, sol_y + 1.5)

    plt.tight_layout()
    plt.savefig('images/contour_plot.png', dpi=300, bbox_inches='tight')
    print(f"\n3. График сохранен: images/contour_plot.png")
    plt.close()

    # 6. Визуализация 2: 3D поверхность
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Сетка для 3D
    x_3d = np.linspace(sol_x - 2, sol_x + 2, 100)
    y_3d = np.linspace(sol_y - 2, sol_y + 2, 100)
    X_3d, Y_3d = np.meshgrid(x_3d, y_3d)
    Z_3d = 4 * X_3d ** 2 + 2 * X_3d * Y_3d + 5 * Y_3d ** 2 - 4 * X_3d + 6 * Y_3d

    # Поверхность
    surf = ax.plot_surface(X_3d, Y_3d, Z_3d, cmap='viridis', alpha=0.7, 
                          linewidth=0, antialiased=True)

    # Точка экстремума
    ax.scatter([sol_x], [sol_y], [f_val], color='red', s=200, 
              label=f'Экстремум: ({sol_x:.3f}, {sol_y:.3f}, {f_val:.3f})')

    # Линии ограничений на поверхности
    # g1: y = x - 2
    x_g1 = np.linspace(sol_x - 1.5, sol_x + 1.5, 100)
    y_g1 = x_g1 - 2
    z_g1 = 4 * x_g1 ** 2 + 2 * x_g1 * y_g1 + 5 * y_g1 ** 2 - 4 * x_g1 + 6 * y_g1
    ax.plot(x_g1, y_g1, z_g1, 'r--', linewidth=3, label='g₁: x - y - 2 = 0')

    # g2: y = 1 - 3x
    x_g2 = np.linspace(sol_x - 1.5, sol_x + 1.5, 100)
    y_g2 = 1 - 3 * x_g2
    z_g2 = 4 * x_g2 ** 2 + 2 * x_g2 * y_g2 + 5 * y_g2 ** 2 - 4 * x_g2 + 6 * y_g2
    ax.plot(x_g2, y_g2, z_g2, 'b--', linewidth=3, label='g₂: 3x + y - 1 = 0')

    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_zlabel('f(x, y)', fontsize=12)
    ax.set_title(f'3D визуализация целевой функции\nf(x*, y*) = {f_val:.4f}', 
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig('images/3d_surface.png', dpi=300, bbox_inches='tight')
    print(f"4. 3D график сохранен: images/3d_surface.png")
    plt.close()

    # 7. Визуализация 3: Пересечение ограничений
    plt.figure(figsize=(10, 8))

    # Ограничения
    x_line = np.linspace(sol_x - 2, sol_x + 2, 400)
    plt.plot(x_line, x_line - 2, 'r-', label='g₁: x - y - 2 = 0', linewidth=2.5)
    plt.plot(x_line, 1 - 3 * x_line, 'b-', label='g₂: 3x + y - 1 = 0', linewidth=2.5)

    # Точка пересечения (решение системы ограничений)
    plt.plot(sol_x, sol_y, 'go', markersize=15, label=f'Решение: ({sol_x:.3f}, {sol_y:.3f})', zorder=10)
    plt.text(sol_x + 0.05, sol_y + 0.05, f'M*', fontsize=14, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

    plt.title('Пересечение ограничений', fontsize=14, fontweight='bold')
    plt.xlabel('x', fontsize=12)
    plt.ylabel('y', fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(fontsize=11)
    plt.xlim(sol_x - 2, sol_x + 2)
    plt.ylim(sol_y - 2, sol_y + 2)
    plt.axis('equal')

    plt.tight_layout()
    plt.savefig('images/constraints_intersection.png', dpi=300, bbox_inches='tight')
    print(f"5. График пересечения ограничений сохранен: images/constraints_intersection.png")
    plt.close()

    print("\n--- РЕШЕНИЕ ЗАВЕРШЕНО ---")
    
    return {
        'x': sol_x, 'y': sol_y, 'f_val': f_val,
        'lambda1': sol_l1, 'lambda2': sol_l2,
        'det_B': det_B, 'character': character
    }


if __name__ == "__main__":
    solve_analyze_plot()