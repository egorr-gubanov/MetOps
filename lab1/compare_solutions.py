#!/usr/bin/env python3
"""
Скрипт для сравнения решений задачи линейного программирования
между собственной реализацией симплекс-метода и scipy.optimize.linprog
"""

import sys
import os
import numpy as np
from simplex_solver import SimplexSolver
from simplex_solver_lib import LinprogSolver


def compare_solutions(file_path, verbose=False):
    """
    Сравнивает решения от обоих методов.
    
    Args:
        file_path: путь к файлу с задачей
        verbose: выводить подробную информацию
    """
    print("\n" + "="*80)
    print(f"СРАВНЕНИЕ РЕШЕНИЙ ДЛЯ ФАЙЛА: {file_path}")
    print("="*80)
    
    # Решение с помощью scipy
    print("\n[1] Решение с помощью scipy.optimize.linprog:")
    print("-" * 80)
    scipy_solver = LinprogSolver(file_path)
    scipy_result = scipy_solver.solve()
    scipy_solver.print_solution(scipy_result)
    
    # Решение с помощью собственной реализации
    print("\n[2] Решение с помощью собственной реализации симплекс-метода:")
    print("-" * 80)
    
    # Временно отключаем verbose для собственной реализации
    solver = SimplexSolver(file_path)
    solver.verbose = verbose
    solver.solve()
    
    # Извлекаем результаты из собственной реализации
    if solver.is_infeasible or solver.is_unbounded:
        own_success = False
        own_x = None
        own_z = None
    else:
        own_success = True
        own_x = {}
        for i in range(solver.num_original_vars):
            own_x[f'x{i+1}'] = 0.0
        
        rhs_col = solver.tableau.shape[1] - 1
        for i, basis_col_idx in enumerate(solver.basis):
            if basis_col_idx < solver.num_original_vars:
                var_name = solver.all_var_names[basis_col_idx]
                own_x[var_name] = solver.tableau[i, rhs_col]
        
        final_z_value = solver.tableau[-1, -1]
        # Для minimize в таблице хранится -Z (поскольку мы минимизируем -Z = maximize Z)
        # Для maximize значение уже корректное
        own_z = -final_z_value if solver.objective_type == 'minimize' else final_z_value
    
    # Сравнение результатов
    print("\n" + "="*80)
    print("СРАВНЕНИЕ РЕЗУЛЬТАТОВ")
    print("="*80)
    
    if scipy_result.success and own_success:
        # Сравниваем значения переменных
        print("\nОптимальная точка:")
        print(f"{'Переменная':<10} {'Scipy':<15} {'Собственная':<15} {'Разница':<15}")
        print("-" * 55)
        
        max_diff = 0
        for i in range(len(scipy_result.x)):
            var_name = f'x{i+1}'
            scipy_val = scipy_result.x[i]
            own_val = own_x.get(var_name, 0.0)
            diff = abs(scipy_val - own_val)
            max_diff = max(max_diff, diff)
            print(f"{var_name:<10} {scipy_val:15.6f} {own_val:15.6f} {diff:15.6e}")
        
        # Сравниваем значения целевой функции
        scipy_z = np.dot(scipy_solver.objective_coeffs, scipy_result.x)
        # own_z уже содержит корректное значение после исправления в simplex_solver.py
        own_z_formatted = own_z
        
        print(f"\nЗначение целевой функции:")
        print(f"{'Scipy:':<15} {scipy_z:.6f}")
        print(f"{'Собственная:':<15} {own_z_formatted:.6f}")
        print(f"{'Разница:':<15} {abs(scipy_z - own_z_formatted):.6e}")
        
        # Вывод о совпадении
        tolerance = 1e-5
        if max_diff < tolerance and abs(scipy_z - own_z_formatted) < tolerance:
            print(f"\n✓ РЕЗУЛЬТАТЫ СОВПАДАЮТ (погрешность < {tolerance})")
        else:
            print(f"\n⚠ РЕЗУЛЬТАТЫ РАСХОДЯТСЯ (погрешность ≥ {tolerance})")
    
    elif not scipy_result.success and not own_success:
        print("\n✓ ОБА МЕТОДА СОГЛАСНЫ: задача не имеет решения")
        print(f"  Scipy сообщение: {scipy_result.message}")
        if solver.is_infeasible:
            print("  Собственная реализация: задача несовместна")
        if solver.is_unbounded:
            print("  Собственная реализация: целевая функция не ограничена")
    
    else:
        print("\n⚠ РЕЗУЛЬТАТЫ РАСХОДЯТСЯ:")
        if scipy_result.success:
            print("  Scipy: задача имеет решение")
        else:
            print(f"  Scipy: {scipy_result.message}")
        
        if own_success:
            print("  Собственная реализация: задача имеет решение")
        else:
            if solver.is_infeasible:
                print("  Собственная реализация: задача несовместна")
            if solver.is_unbounded:
                print("  Собственная реализация: целевая функция не ограничена")
    
    print("="*80 + "\n")


def main():
    if len(sys.argv) < 2:
        print("Использование: python compare_solutions.py <файл1> [файл2] ... [--verbose]")
        print("Пример: python compare_solutions.py problem_variant.txt")
        print("Пример: python compare_solutions.py problem_variant.txt problem2.txt --verbose")
        sys.exit(1)
    
    verbose = '--verbose' in sys.argv
    files = [f for f in sys.argv[1:] if f != '--verbose']
    
    for file_path in files:
        if not os.path.exists(file_path):
            print(f"Ошибка: файл {file_path} не найден")
            continue
        
        try:
            compare_solutions(file_path, verbose)
        except Exception as e:
            print(f"Ошибка при обработке {file_path}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
