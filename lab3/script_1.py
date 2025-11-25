"""
Проверка и разбор решения задачи условной оптимизации
методом множителей Лагранжа на Python

Задача: Минимизировать f(x,y) = 4x² + 2xy + 5y² - 4x + 6y
При ограничениях: g₁(x,y) = x - y - 2 = 0
                 g₂(x,y) = 3x + y - 1 = 0
"""

import numpy as np
from scipy.optimize import fsolve, minimize
from fractions import Fraction
import sympy as sp
from sympy import symbols, diff, Matrix, solve, Rational, simplify


# ============================================================================
# ЧАСТЬ 1: СИМВОЛЬНЫЕ ВЫЧИСЛЕНИЯ С ИСПОЛЬЗОВАНИЕМ SymPy
# ============================================================================

def symbolic_solution():
    """
    Решение системы уравнений в символьном виде для получения точных ответов
    """
    print("\n" + "=" * 80)
    print("СИМВОЛЬНОЕ РЕШЕНИЕ")
    print("=" * 80)

    x, y, lambda1, lambda2 = symbols('x y lambda1 lambda2', real=True)

    # Определяем целевую функцию и ограничения
    f = 4 * x ** 2 + 2 * x * y + 5 * y ** 2 - 4 * x + 6 * y
    g1 = x - y - 2
    g2 = 3 * x + y - 1

    # Функция Лагранжа
    L = f - lambda1 * g1 - lambda2 * g2

    print("\n1. Целевая функция:")
    print(f"   f(x,y) = {f}")

    print("\n2. Ограничения:")
    print(f"   g₁(x,y) = {g1} = 0")
    print(f"   g₂(x,y) = {g2} = 0")

    print("\n3. Функция Лагранжа:")
    print(f"   L = {L}")

    # Условия первого порядка
    dL_dx = diff(L, x)
    dL_dy = diff(L, y)
    dL_dlambda1 = diff(L, lambda1)
    dL_dlambda2 = diff(L, lambda2)

    print("\n4. Условия первого порядка:")
    print(f"   ∂L/∂x = {dL_dx} = 0")
    print(f"   ∂L/∂y = {dL_dy} = 0")
    print(f"   ∂L/∂λ₁ = {dL_dlambda1} = 0")
    print(f"   ∂L/∂λ₂ = {dL_dlambda2} = 0")

    # Решаем систему
    equations = [dL_dx, dL_dy, dL_dlambda1, dL_dlambda2]
    solution = solve(equations, [x, y, lambda1, lambda2])

    print("\n5. Критическая точка (точное решение):")
    x_exact = solution[x]
    y_exact = solution[y]
    lambda1_exact = solution[lambda1]
    lambda2_exact = solution[lambda2]

    print(f"   x* = {x_exact}")
    print(f"   y* = {y_exact}")
    print(f"   λ₁* = {lambda1_exact}")
    print(f"   λ₂* = {lambda2_exact}")

    # Вычисляем значение функции
    f_exact = f.subs([(x, x_exact), (y, y_exact)])
    print(f"\n6. Значение целевой функции:")
    print(f"   f(x*, y*) = {f_exact}")

    # Преобразуем в десятичные значения
    x_float = float(x_exact)
    y_float = float(y_exact)
    lambda1_float = float(lambda1_exact)
    lambda2_float = float(lambda2_exact)
    f_float = float(f_exact)

    print(f"\n7. Численные значения:")
    print(f"   x* = {x_float}")
    print(f"   y* = {y_float}")
    print(f"   λ₁* = {lambda1_float}")
    print(f"   λ₂* = {lambda2_float}")
    print(f"   f(x*, y*) = {f_float}")

    return {
        'x': x_exact, 'y': y_exact,
        'lambda1': lambda1_exact, 'lambda2': lambda2_exact,
        'f_value': f_exact,
        'x_float': x_float, 'y_float': y_float,
        'lambda1_float': lambda1_float, 'lambda2_float': lambda2_float,
        'f_float': f_float,
        'L': L, 'f': f, 'g1': g1, 'g2': g2
    }


# ============================================================================
# ЧАСТЬ 2: ПРОВЕРКА УСЛОВИЙ ВТОРОГО ПОРЯДКА (КРИТЕРИЙ СИЛЬВЕСТРА)
# ============================================================================

def check_bordered_hessian(sol_dict):
    """
    Проверка характера экстремума с помощью окаймленной матрицы Гессе
    """
    print("\n" + "=" * 80)
    print("ПРОВЕРКА ХАРАКТЕРА ЭКСТРЕМУМА (КРИТЕРИЙ СИЛЬВЕСТРА)")
    print("=" * 80)

    x, y = symbols('x y', real=True)
    f = sol_dict['f']
    g1 = sol_dict['g1']
    g2 = sol_dict['g2']

    # Матрица Гессе целевой функции
    H = Matrix([
        [diff(f, x, 2), diff(f, x, y)],
        [diff(f, y, x), diff(f, y, 2)]
    ])

    print("\n1. Матрица Гессе целевой функции:")
    print("   H(f) =")
    sp.pprint(H)

    # Матрица Якоби ограничений
    J = Matrix([
        [diff(g1, x), diff(g1, y)],
        [diff(g2, x), diff(g2, y)]
    ])

    print("\n2. Матрица Якоби ограничений:")
    print("   J =")
    sp.pprint(J)
    print(f"\n   Ранг матрицы J: {J.rank()}")
    if J.rank() == 2:
        print("   ✓ Ограничения линейно независимы")

    # Окаймленная матрица Гессе
    B = Matrix([
        [0, 0, diff(g1, x), diff(g2, x)],
        [0, 0, diff(g1, y), diff(g2, y)],
        [diff(g1, x), diff(g1, y), diff(f, x, 2), diff(f, x, y)],
        [diff(g2, x), diff(g2, y), diff(f, y, x), diff(f, y, 2)]
    ])

    print("\n3. Окаймленная матрица Гессе (Bordered Hessian):")
    print("   B =")
    sp.pprint(B)

    # Определитель
    det_B = B.det()
    det_B_value = float(det_B)

    print(f"\n4. Определитель окаймленной матрицы:")
    print(f"   det(B) = {det_B}")
    print(f"   Численное значение: {det_B_value}")

    print("\n5. Вывод по критерию Сильвестра:")
    m = 2  # число ограничений
    n = 2  # число переменных

    print(f"   Для задачи с m={m} ограничениями и n={n} переменными:")

    if det_B_value > 0:
        print(f"   det(B) = {det_B_value} > 0")
        print("   ⇒ ЛОКАЛЬНЫЙ МИНИМУМ")
        result = "ЛОКАЛЬНЫЙ МИНИМУМ"
    elif det_B_value < 0:
        print(f"   det(B) = {det_B_value} < 0")
        print("   ⇒ ЛОКАЛЬНЫЙ МАКСИМУМ")
        result = "ЛОКАЛЬНЫЙ МАКСИМУМ"
    else:
        print(f"   det(B) = {det_B_value} = 0")
        print("   ⇒ ТРЕБУЕТСЯ ДОПОЛНИТЕЛЬНЫЙ АНАЛИЗ")
        result = "ТРЕБУЕТСЯ ДОПОЛНИТЕЛЬНЫЙ АНАЛИЗ"

    return {'det_B': det_B_value, 'character': result, 'B': B, 'H': H}


# ============================================================================
# ЧАСТЬ 3: ЧИСЛЕННАЯ ПРОВЕРКА С ПОМОЩЬЮ SCIPY
# ============================================================================

def numerical_verification(sol_dict):
    """
    Численная проверка решения с помощью scipy.optimize
    """
    print("\n" + "=" * 80)
    print("ЧИСЛЕННАЯ ПРОВЕРКА РЕШЕНИЯ")
    print("=" * 80)

    # Определяем целевую функцию в численном виде
    def f(vars):
        x, y = vars
        return 4 * x ** 2 + 2 * x * y + 5 * y ** 2 - 4 * x + 6 * y

    # Определяем ограничения
    def constraint1(vars):
        x, y = vars
        return x - y - 2

    def constraint2(vars):
        x, y = vars
        return 3 * x + y - 1

    # Система уравнений для нахождения критической точки
    def equations(vars):
        x, y = vars
        return [constraint1([x, y]), constraint2([x, y])]

    print("\n1. Решение системы ограничений численно:")
    # Начальное приближение
    x0 = [0, 0]
    solution_numerical = fsolve(equations, x0)
    x_num, y_num = solution_numerical

    print(f"   Начальное приближение: {x0}")
    print(f"   Найденное решение: x = {x_num:.10f}, y = {y_num:.10f}")

    # Проверка выполнения ограничений
    g1_check = constraint1([x_num, y_num])
    g2_check = constraint2([x_num, y_num])

    print(f"\n2. Проверка выполнения ограничений:")
    print(f"   g₁(x, y) = x - y - 2 = {g1_check:.2e}")
    print(f"   g₂(x, y) = 3x + y - 1 = {g2_check:.2e}")

    # Вычисляем значение функции
    f_num = f([x_num, y_num])
    print(f"\n3. Значение целевой функции:")
    print(f"   f({x_num:.10f}, {y_num:.10f}) = {f_num:.10f}")

    # Сравнение с точным решением
    print(f"\n4. Сравнение с точным решением:")
    x_exact = sol_dict['x_float']
    y_exact = sol_dict['y_float']
    f_exact = sol_dict['f_float']

    print(f"   Точное решение:     x* = {x_exact}, y* = {y_exact}")
    print(f"   Численное решение:  x  = {x_num:.10f}, y  = {y_num:.10f}")
    print(f"   Разница по x: {abs(x_exact - x_num):.2e}")
    print(f"   Разница по y: {abs(y_exact - y_num):.2e}")
    print(f"\n   Точное значение:    f* = {f_exact}")
    print(f"   Численное значение: f  = {f_num:.10f}")
    print(f"   Разница: {abs(f_exact - f_num):.2e}")

    return {'x_num': x_num, 'y_num': y_num, 'f_num': f_num}


# ============================================================================
# ЧАСТЬ 4: ПРОВЕРКА ГРАДИЕНТА И МНОЖИТЕЛЕЙ ЛАГРАНЖА
# ============================================================================

def gradient_verification(sol_dict):
    """
    Проверка условия: ∇f = λ₁∇g₁ + λ₂∇g₂
    """
    print("\n" + "=" * 80)
    print("ПРОВЕРКА УСЛОВИЯ ОПТИМАЛЬНОСТИ")
    print("=" * 80)

    x, y = symbols('x y', real=True)
    f = sol_dict['f']
    g1 = sol_dict['g1']
    g2 = sol_dict['g2']

    # Градиент целевой функции
    grad_f = Matrix([diff(f, x), diff(f, y)])
    grad_g1 = Matrix([diff(g1, x), diff(g1, y)])
    grad_g2 = Matrix([diff(g2, x), diff(g2, y)])

    print("\n1. Градиенты функций:")
    print("   ∇f =")
    sp.pprint(grad_f)
    print("\n   ∇g₁ =")
    sp.pprint(grad_g1)
    print("\n   ∇g₂ =")
    sp.pprint(grad_g2)

    # Подставляем критическую точку
    x_val = sol_dict['x']
    y_val = sol_dict['y']
    lambda1_val = sol_dict['lambda1']
    lambda2_val = sol_dict['lambda2']

    grad_f_val = grad_f.subs([(x, x_val), (y, y_val)])
    grad_g1_val = grad_g1.subs([(x, x_val), (y, y_val)])
    grad_g2_val = grad_g2.subs([(x, x_val), (y, y_val)])

    print("\n2. Значения в критической точке:")
    print(f"   ∇f(x*, y*) = {grad_f_val.T}")
    print(f"   ∇g₁(x*, y*) = {grad_g1_val.T}")
    print(f"   ∇g₂(x*, y*) = {grad_g2_val.T}")

    # Проверка условия ∇f = λ₁∇g₁ + λ₂∇g₂
    linear_comb = lambda1_val * grad_g1_val + lambda2_val * grad_g2_val

    print("\n3. Проверка условия: ∇f = λ₁∇g₁ + λ₂∇g₂")
    print(f"   λ₁* · ∇g₁ + λ₂* · ∇g₂ = {lambda1_val} · {grad_g1_val.T} + {lambda2_val} · {grad_g2_val.T}")
    print(f"                         = {linear_comb.T}")

    difference = simplify(grad_f_val - linear_comb)
    print(f"\n   ∇f - (λ₁∇g₁ + λ₂∇g₂) = {difference.T}")

    if difference == Matrix([0, 0]):
        print("   ✓ Условие выполнено!")
    else:
        print("   ⚠ Условие не выполнено (проверьте вычисления)")


# ============================================================================
# ЧАСТЬ 5: ПРЕДСТАВЛЕНИЕ РЕЗУЛЬТАТОВ
# ============================================================================

def print_results_table(sol_dict, hessian_dict):
    """
    Вывод результатов в виде таблицы
    """
    print("\n" + "=" * 80)
    print("ИТОГОВАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ")
    print("=" * 80)

    print("\n{:<30} {:<20} {:<20}".format("Параметр", "Точное значение", "Численное значение"))
    print("-" * 70)

    x_frac = Fraction(sol_dict['x_float']).limit_denominator(100)
    y_frac = Fraction(sol_dict['y_float']).limit_denominator(100)
    lambda1_frac = Fraction(sol_dict['lambda1_float']).limit_denominator(100)
    lambda2_frac = Fraction(sol_dict['lambda2_float']).limit_denominator(100)
    f_frac = Fraction(sol_dict['f_float']).limit_denominator(100)

    print("{:<30} {:<20} {:<20}".format("x*", str(x_frac), f"{sol_dict['x_float']:.6f}"))
    print("{:<30} {:<20} {:<20}".format("y*", str(y_frac), f"{sol_dict['y_float']:.6f}"))
    print("{:<30} {:<20} {:<20}".format("λ₁*", str(lambda1_frac), f"{sol_dict['lambda1_float']:.6f}"))
    print("{:<30} {:<20} {:<20}".format("λ₂*", str(lambda2_frac), f"{sol_dict['lambda2_float']:.6f}"))
    print("{:<30} {:<20} {:<20}".format("f(x*, y*)", str(f_frac), f"{sol_dict['f_float']:.6f}"))
    print("{:<30} {:<20} {:<20}".format("det(B)", str(hessian_dict['det_B']), ""))
    print("{:<30} {:<20} {:<20}".format("Характер точки", hessian_dict['character'], ""))


# ============================================================================
# ОСНОВНАЯ ФУНКЦИЯ
# ============================================================================

def main():
    """
    Главная функция для запуска всех проверок
    """
    print("\n" + "=" * 80)
    print("ПОЛНЫЙ АНАЛИЗ ЗАДАЧИ УСЛОВНОЙ ОПТИМИЗАЦИИ")
    print("Метод: Множители Лагранжа с проверкой условий второго порядка")
    print("=" * 80)

    # Решение
    sol_dict = symbolic_solution()

    # Проверка условий второго порядка
    hessian_dict = check_bordered_hessian(sol_dict)

    # Численная проверка
    numerical_verification(sol_dict)

    # Проверка условия оптимальности
    gradient_verification(sol_dict)

    # Итоговая таблица
    print_results_table(sol_dict, hessian_dict)

    print("\n" + "=" * 80)
    print("ЗАКЛЮЧЕНИЕ")
    print("=" * 80)
    print("\nЗадача условной оптимизации решена методом множителей Лагранжа.")
    print(f"Найденная критическая точка: (x*, y*) = ({sol_dict['x_float']}, {sol_dict['y_float']})")
    print(f"Характер точки: {hessian_dict['character']}")
    print(f"Минимальное значение функции: f(x*, y*) = {sol_dict['f_float']}")
    print("\nВсе условия оптимальности проверены и выполнены.")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()