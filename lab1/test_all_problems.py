#!/usr/bin/env python3
"""
Автоматическое тестирование всех 20 задач.
Сравнивает результаты собственной реализации симплекс-метода с scipy.
"""

import os
import sys
import numpy as np
from simplex_solver import SimplexSolver
from simplex_solver_lib import LinprogSolver


class Colors:
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    RESET = '\033[0m'
    BOLD = '\033[1m'


def test_problem(file_path, problem_num):
    """Тестирует одну задачу."""
    print(f"\n{Colors.CYAN}{'='*80}")
    print(f"ТЕСТ #{problem_num}: {os.path.basename(file_path)}")
    print(f"{'='*80}{Colors.RESET}")
    
    try:
        # Решение с помощью scipy
        scipy_solver = LinprogSolver(file_path)
        scipy_result = scipy_solver.solve()
        
        # Решение с помощью собственной реализации
        solver = SimplexSolver(file_path)
        solver.verbose = False  # Отключаем подробный вывод
        solver.solve()
        
        # Извлекаем результаты
        scipy_success = scipy_result.success
        own_success = not (solver.is_infeasible or solver.is_unbounded)
        
        if scipy_success and own_success:
            # Сравниваем значения переменных
            own_x = {}
            for i in range(solver.num_original_vars):
                own_x[f'x{i+1}'] = 0.0
            
            rhs_col = solver.tableau.shape[1] - 1
            for i, basis_col_idx in enumerate(solver.basis):
                if basis_col_idx < solver.num_original_vars:
                    var_name = solver.all_var_names[basis_col_idx]
                    own_x[var_name] = solver.tableau[i, rhs_col]
            
            # Сравниваем
            max_diff = 0
            for i in range(len(scipy_result.x)):
                var_name = f'x{i+1}'
                scipy_val = scipy_result.x[i]
                own_val = own_x.get(var_name, 0.0)
                diff = abs(scipy_val - own_val)
                max_diff = max(max_diff, diff)
            
            # Сравниваем значения ЦФ
            scipy_z = np.dot(scipy_solver.objective_coeffs, scipy_result.x)
            final_z_value = solver.tableau[-1, -1]
            
            if solver.objective_type == 'minimize':
                own_z = -final_z_value
            else:
                own_z = final_z_value
            
            z_diff = abs(scipy_z - own_z)
            
            # Вывод результата
            print(f"\n{Colors.BLUE}Решение:{Colors.RESET}")
            print(f"  Оптимальная точка: x = ({', '.join([f'{scipy_result.x[i]:.4f}' for i in range(len(scipy_result.x))])})")
            print(f"  Z = {scipy_z:.6f}")
            
            print(f"\n{Colors.BLUE}Сравнение:{Colors.RESET}")
            print(f"  Макс. разница в переменных: {max_diff:.6e}")
            print(f"  Разница в Z: {z_diff:.6e}")
            
            tolerance = 1e-5
            if max_diff < tolerance and z_diff < tolerance:
                print(f"\n{Colors.GREEN}✓ ТЕСТ ПРОЙДЕН{Colors.RESET} (погрешность < {tolerance})")
                return True
            else:
                print(f"\n{Colors.RED}✗ ТЕСТ НЕ ПРОЙДЕН{Colors.RESET} (погрешность ≥ {tolerance})")
                return False
                
        elif not scipy_success and not own_success:
            print(f"\n{Colors.YELLOW}⚠ ОБА МЕТОДА: нет решения{Colors.RESET}")
            return True
        else:
            print(f"\n{Colors.RED}✗ РЕЗУЛЬТАТЫ РАСХОДЯТСЯ{Colors.RESET}")
            if scipy_success:
                print(f"  Scipy: есть решение")
            else:
                print(f"  Scipy: {scipy_result.message}")
            
            if own_success:
                print(f"  Собственная: есть решение")
            else:
                if solver.is_infeasible:
                    print(f"  Собственная: несовместна")
                if solver.is_unbounded:
                    print(f"  Собственная: не ограничена")
            return False
            
    except Exception as e:
        print(f"\n{Colors.RED}✗ ОШИБКА: {e}{Colors.RESET}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Главная функция тестирования."""
    print(f"{Colors.BOLD}{Colors.CYAN}")
    print("="*80)
    print("АВТОМАТИЧЕСКОЕ ТЕСТИРОВАНИЕ ВСЕХ ЗАДАЧ")
    print("="*80)
    print(f"{Colors.RESET}")
    
    # Ищем все файлы задач
    test_dir = "test_problems"
    if not os.path.exists(test_dir):
        print(f"{Colors.RED}Ошибка: директория '{test_dir}/' не найдена!{Colors.RESET}")
        print(f"\nСначала запустите: python generate_test_problems.py")
        sys.exit(1)
    
    problem_files = sorted([f for f in os.listdir(test_dir) if f.endswith('.txt')])
    
    if not problem_files:
        print(f"{Colors.RED}Ошибка: не найдено файлов задач в '{test_dir}/'!{Colors.RESET}")
        sys.exit(1)
    
    print(f"Найдено задач: {len(problem_files)}\n")
    
    # Тестируем все задачи
    results = {}
    for i, filename in enumerate(problem_files, start=1):
        file_path = os.path.join(test_dir, filename)
        success = test_problem(file_path, i)
        results[filename] = success
    
    # Итоговый отчет
    print(f"\n{Colors.BOLD}{Colors.CYAN}")
    print("="*80)
    print("ИТОГОВЫЙ ОТЧЕТ")
    print("="*80)
    print(f"{Colors.RESET}\n")
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for filename, success in results.items():
        if success:
            status = f"{Colors.GREEN}✓ PASS{Colors.RESET}"
        else:
            status = f"{Colors.RED}✗ FAIL{Colors.RESET}"
        print(f"{status}: {filename}")
    
    print(f"\n{Colors.BOLD}Пройдено: {passed}/{total}{Colors.RESET}")
    
    if passed == total:
        print(f"\n{Colors.GREEN}{Colors.BOLD}🎉 ВСЕ ТЕСТЫ ПРОЙДЕНЫ!{Colors.RESET}")
        return 0
    else:
        print(f"\n{Colors.YELLOW}⚠ Некоторые тесты не прошли{Colors.RESET}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

