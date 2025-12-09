"""
Модуль для решения задач целочисленного линейного программирования (ЦЛП).

Используется метод ветвей и границ (Branch and Bound):
1. Решаем задачу как обычную ЗЛП (релаксация)
2. Если решение целочисленное - готово
3. Если нет - выбираем переменную с дробной частью
4. Создаем две подзадачи: одна с ограничением x <= floor(x*), другая с x >= ceil(x*)
5. Рекурсивно решаем подзадачи
6. Отсекаем подзадачи, которые не могут дать лучшее решение (граница)
"""

import numpy as np
import sys
import copy
from simplex_solver import SimplexSolver
import tempfile
import os


class IntegerSolver:
    """
    Решает задачи целочисленного линейного программирования методом ветвей и границ.
    """
    
    def __init__(self, file_path, integer_vars=None):
        """
        Инициализация решателя целочисленных задач.
        
        Args:
            file_path: путь к файлу с задачей
            integer_vars: множество индексов переменных, которые должны быть целыми
                         (0-based). Если None, все переменные целочисленные.
        """
        self.file_path = file_path
        self.integer_vars = integer_vars  # индексы переменных, которые должны быть целыми
        self.best_solution = None
        self.best_value = None
        self.best_is_maximize = None
        self.nodes_explored = 0
        self.verbose = True
        
        # Парсим исходную задачу
        self.base_solver = SimplexSolver(file_path)
        self.base_solver.verbose = False  # Отключаем вывод для подзадач
        
        # Если integer_vars не указан, все переменные целочисленные
        if self.integer_vars is None:
            self.integer_vars = set(range(self.base_solver.num_original_vars))
        
        self.best_is_maximize = (self.base_solver.objective_type == 'maximize')
        
        # Инициализируем best_value в зависимости от типа задачи
        if self.best_is_maximize:
            self.best_value = float('-inf')  # Для максимизации начинаем с -∞
        else:
            self.best_value = float('inf')   # Для минимизации начинаем с +∞
    
    def is_integer_solution(self, solution, tolerance=1e-6):
        """
        Проверяет, является ли решение целочисленным.
        
        Args:
            solution: словарь {var_name: value}
            tolerance: допустимая погрешность для проверки целочисленности
        
        Returns:
            bool: True если решение целочисленное
        """
        for var_name, value in solution.items():
            # Проверяем только переменные, которые должны быть целыми
            # Извлекаем индекс переменной из имени (x1 -> 0, x2 -> 1, ...)
            try:
                var_idx = int(var_name[1:]) - 1  # x1 -> 0, x2 -> 1, ...
                if var_idx in self.integer_vars:
                    if abs(value - round(value)) > tolerance:
                        return False
            except:
                # Если не удалось извлечь индекс, пропускаем
                pass
        return True
    
    def get_fractional_variable(self, solution):
        """
        Находит переменную с наибольшей дробной частью.
        
        Args:
            solution: словарь {var_name: value}
        
        Returns:
            tuple: (var_name, value) или (None, None) если все целочисленные
        """
        max_fractional = 0
        fractional_var = None
        fractional_value = None
        
        for var_name, value in solution.items():
            try:
                var_idx = int(var_name[1:]) - 1
                if var_idx in self.integer_vars:
                    fractional = abs(value - round(value))
                    if fractional > max_fractional:
                        max_fractional = fractional
                        fractional_var = var_name
                        fractional_value = value
            except:
                pass
        
        return fractional_var, fractional_value
    
    def solve_relaxation(self, additional_constraints=None):
        """
        Решает релаксацию задачи (без требования целочисленности).
        
        Args:
            additional_constraints: список дополнительных ограничений в формате
                                   [(coeffs, type, rhs), ...]
        
        Returns:
            tuple: (solver, solution_dict, z_value) или None если неразрешима
        """
        # Создаем временный файл с задачей
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8')
        temp_file_path = temp_file.name
        
        try:
            # Читаем исходный файл
            with open(self.file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Записываем в временный файл
            with open(temp_file_path, 'w', encoding='utf-8') as f:
                # Копируем исходные строки
                for line in lines:
                    f.write(line)
                
                # Добавляем дополнительные ограничения
                if additional_constraints:
                    for coeffs, const_type, rhs in additional_constraints:
                        coeffs_str = " ".join([f"{c:.6f}" for c in coeffs])
                        f.write(f"{coeffs_str} {const_type} {rhs:.6f}\n")
            
            # Решаем
            solver = SimplexSolver(temp_file_path)
            solver.verbose = False
            solver.solve()
            
            if solver.is_infeasible or solver.is_unbounded:
                return None
            
            # Извлекаем решение
            solution = {}
            rhs_col = solver.tableau.shape[1] - 1
            
            for i, basis_col_idx in enumerate(solver.basis):
                if basis_col_idx < solver.num_original_vars:
                    var_name = solver.all_var_names[basis_col_idx]
                    solution[var_name] = solver.tableau[i, rhs_col]
            
            # Значение целевой функции
            final_z_value = solver.tableau[-1, -1]
            if solver.objective_type == 'minimize':
                z_value = -final_z_value
            else:
                z_value = final_z_value
            
            return solver, solution, z_value
            
        finally:
            # Удаляем временный файл
            try:
                os.unlink(temp_file_path)
            except:
                pass
    
    def branch_and_bound(self, additional_constraints=None, depth=0):
        """
        Рекурсивный метод ветвей и границ.
        
        Args:
            additional_constraints: список дополнительных ограничений
            depth: глубина рекурсии (для отладки)
        
        Returns:
            bool: True если найдено решение
        """
        self.nodes_explored += 1
        
        if self.verbose and depth == 0:
            print(f"\n{'='*80}")
            print("РЕШЕНИЕ ЗАДАЧИ ЦЕЛОЧИСЛЕННОГО ЛИНЕЙНОГО ПРОГРАММИРОВАНИЯ")
            print(f"{'='*80}")
            print("Метод: Ветвей и границ (Branch and Bound)")
            print(f"Целочисленные переменные: {sorted([i+1 for i in self.integer_vars])}")
            print(f"{'='*80}\n")
        
        # Решаем релаксацию
        result = self.solve_relaxation(additional_constraints)
        
        if result is None:
            # Задача неразрешима
            return False
        
        solver, solution, z_value = result
        
        # Проверяем границу (bound)
        if self.best_is_maximize:
            # Для максимизации: если текущее значение меньше лучшего, отсекаем
            if z_value < self.best_value - 1e-6:
                if self.verbose and depth == 0:
                    print(f"Отсечение по границе: z = {z_value:.6f} < best = {self.best_value:.6f}")
                return False
        else:
            # Для минимизации: если текущее значение больше лучшего, отсекаем
            if z_value > self.best_value + 1e-6:
                if self.verbose and depth == 0:
                    print(f"Отсечение по границе: z = {z_value:.6f} > best = {self.best_value:.6f}")
                return False
        
        # Проверяем, целочисленное ли решение
        if self.is_integer_solution(solution):
            # Найдено целочисленное решение
            # Проверяем, лучше ли это решение
            is_better = False
            if self.best_is_maximize:
                is_better = z_value > self.best_value + 1e-6
            else:
                is_better = z_value < self.best_value - 1e-6
            
            if is_better or (self.best_solution is None):
                self.best_solution = solution.copy()
                self.best_value = z_value
                if self.verbose:
                    print(f"\n✓ Найдено целочисленное решение (узел {self.nodes_explored}):")
                    print(f"  Z = {z_value:.6f}")
            return True
        
        # Решение не целочисленное - нужно ветвление
        fractional_var, fractional_value = self.get_fractional_variable(solution)
        
        if fractional_var is None:
            # Не должно произойти, но на всякий случай
            return False
        
        var_idx = int(fractional_var[1:]) - 1
        floor_val = int(np.floor(fractional_value))
        ceil_val = int(np.ceil(fractional_value))
        
        if self.verbose and depth < 2:  # Выводим только для первых уровней
            print(f"  Ветвление по переменной {fractional_var} = {fractional_value:.6f}")
            print(f"    Создаем подзадачи: {fractional_var} <= {floor_val} и {fractional_var} >= {ceil_val}")
        
        # Создаем ограничения для ветвления
        new_constraints_left = (additional_constraints or []) + [
            (self._create_constraint_coeffs(var_idx, self.base_solver.num_original_vars), '<=', float(floor_val))
        ]
        new_constraints_right = (additional_constraints or []) + [
            (self._create_constraint_coeffs(var_idx, self.base_solver.num_original_vars), '>=', float(ceil_val))
        ]
        
        # Рекурсивно решаем подзадачи
        found_left = self.branch_and_bound(new_constraints_left, depth + 1)
        found_right = self.branch_and_bound(new_constraints_right, depth + 1)
        
        return found_left or found_right
    
    def _create_constraint_coeffs(self, var_idx, num_vars):
        """
        Создает вектор коэффициентов для ограничения на одну переменную.
        
        Args:
            var_idx: индекс переменной (0-based)
            num_vars: общее количество переменных
        
        Returns:
            np.array: вектор коэффициентов
        """
        coeffs = np.zeros(num_vars)
        coeffs[var_idx] = 1.0
        return coeffs
    
    def solve(self):
        """
        Главный метод решения задачи целочисленного программирования.
        """
        # Запускаем метод ветвей и границ
        found = self.branch_and_bound()
        
        if self.best_solution is None:
            print("\n" + "="*80)
            print("РЕЗУЛЬТАТ")
            print("="*80)
            print("Целочисленное решение не найдено.")
            print("Возможные причины:")
            print("  - Задача не имеет допустимых целочисленных решений")
            print("  - Область допустимых решений пуста")
            print("="*80)
            return
        
        # Выводим результат
        print("\n" + "="*80)
        print("ОПТИМАЛЬНОЕ ЦЕЛОЧИСЛЕННОЕ РЕШЕНИЕ")
        print("="*80)
        
        print("\nОптимальная точка:")
        for var_name in sorted(self.best_solution.keys(), 
                              key=lambda x: int(x[1:]) if x[1:].isdigit() else x):
            value = self.best_solution[var_name]
            print(f"  {var_name} = {int(round(value))}")
        
        print(f"\nЗначение целевой функции:")
        print(f"  Z = {self.best_value:.6f}")
        
        print(f"\nСтатистика:")
        print(f"  Исследовано узлов: {self.nodes_explored}")
        print("="*80)
    
    def get_solution(self):
        """
        Возвращает найденное решение.
        
        Returns:
            tuple: (solution_dict, z_value) или (None, None)
        """
        # Проверяем, было ли найдено валидное решение
        if self.best_solution is None:
            return None, None
        # Проверяем, что best_value не остался в начальном состоянии
        if (self.best_is_maximize and self.best_value == float('-inf')) or \
           (not self.best_is_maximize and self.best_value == float('inf')):
            return None, None
        return self.best_solution.copy(), self.best_value


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Использование: python integer_solver.py <путь_к_файлу_задачи> [integer_vars]")
        print("  integer_vars: номера переменных, которые должны быть целыми (1-based, через пробел)")
        print("                Если не указано, все переменные целочисленные")
        print("\nПример:")
        print("  python integer_solver.py problem.txt")
        print("  python integer_solver.py problem.txt 1 2 3")
        sys.exit(1)
    
    file_path = sys.argv[1]
    
    integer_vars = None
    if len(sys.argv) > 2:
        integer_vars = set([int(v) - 1 for v in sys.argv[2:]])  # Переводим в 0-based
    
    solver = IntegerSolver(file_path, integer_vars)
    solver.solve()
