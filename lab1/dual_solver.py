"""
Модуль для решения двойственной задачи линейного программирования.

Двойственная задача строится из исходной задачи по правилам:
- Если исходная задача минимизирует, двойственная максимизирует (и наоборот)
- Коэффициенты целевой функции двойственной = правые части ограничений исходной
- Правые части ограничений двойственной = коэффициенты целевой функции исходной
- Матрица ограничений двойственной = транспонированная матрица исходной
- Типы ограничений инвертируются по правилам двойственности
"""

import numpy as np
import sys
from simplex_solver import SimplexSolver


class DualProblemBuilder:
    """
    Класс для построения двойственной задачи из исходной.
    """
    
    def __init__(self, primal_solver):
        """
        Инициализация на основе решенной исходной задачи.
        
        Args:
            primal_solver: Экземпляр SimplexSolver с решенной исходной задачей
        """
        self.primal = primal_solver
        self.dual_objective_type = None
        self.dual_objective_coeffs = None
        self.dual_constraint_coeffs = []
        self.dual_constraint_types = []
        self.dual_constraint_rhs = []
        self.dual_var_names = []
        
    def build_dual(self):
        """
        Строит двойственную задачу из исходной.
        
        Правила построения двойственной задачи:
        
        Исходная задача (примальная):
        min c^T x
        при Ax >= b, x >= 0
        
        Двойственная задача:
        max b^T y
        при A^T y <= c, y >= 0
        
        Для смешанных ограничений:
        - Ограничение "=" в примальной -> переменная y неограниченная в двойственной
        - Ограничение ">=" в примальной -> переменная y >= 0 в двойственной
        - Ограничение "<=" в примальной -> переменная y <= 0 в двойственной (заменяем на y' = -y, y' >= 0)
        - Переменная x неограниченная в примальной -> ограничение "=" в двойственной
        - Переменная x >= 0 в примальной -> ограничение "<=" в двойственной (для minimize)
        """
        # 1. Инвертируем тип целевой функции
        if self.primal.objective_type == 'minimize':
            self.dual_objective_type = 'maximize'
        else:
            self.dual_objective_type = 'minimize'
        
        # 2. Коэффициенты целевой функции двойственной = правые части ограничений исходной
        self.dual_objective_coeffs = np.array(self.primal.constraint_rhs)
        
        # 3. Правые части ограничений двойственной = коэффициенты целевой функции исходной
        self.dual_constraint_rhs = self.primal.objective_coeffs.copy()
        
        # 4. Матрица ограничений двойственной = транспонированная матрица исходной
        A_primal = np.array(self.primal.constraint_coeffs)
        A_dual = A_primal.T
        
        # 5. Формируем ограничения двойственной задачи
        num_dual_vars = len(self.dual_objective_coeffs)  # количество переменных двойственной = количеству ограничений исходной
        num_dual_constraints = len(self.dual_constraint_rhs)  # количество ограничений двойственной = количеству переменных исходной
        
        # Имена переменных двойственной задачи (y1, y2, ...)
        self.dual_var_names = [f'y{i+1}' for i in range(num_dual_vars)]
        
        # Для каждой переменной исходной задачи создаем ограничение в двойственной
        for j in range(num_dual_constraints):
            # Коэффициенты ограничения = j-й столбец матрицы A_dual (j-я строка транспонированной)
            constraint_coeffs = A_dual[j, :].copy()
            
            # Определяем тип ограничения на основе типа переменной в исходной задаче
            if j in self.primal.unrestricted_vars:
                # Неограниченная переменная в исходной -> ограничение "=" в двойственной
                constraint_type = '='
            else:
                # Переменная x_j >= 0 в исходной -> ограничение "<=" в двойственной (для minimize)
                # или ">=" (для maximize)
                if self.primal.objective_type == 'minimize':
                    constraint_type = '<='
                else:
                    constraint_type = '>='
            
            self.dual_constraint_coeffs.append(constraint_coeffs)
            self.dual_constraint_types.append(constraint_type)
        
        # 6. Определяем неограниченные переменные двойственной задачи
        # Переменная y_i неограниченная, если соответствующее ограничение в исходной было "="
        self.dual_unrestricted_vars = set()
        for i, const_type in enumerate(self.primal.constraint_types):
            if const_type == '=':
                self.dual_unrestricted_vars.add(i)
            # Для ограничений "<=" и ">=" переменные двойственной задачи имеют знак,
            # но мы будем обрабатывать это через преобразование задачи
    
    def write_dual_to_file(self, file_path):
        """
        Записывает двойственную задачу в файл в том же формате, что и исходная.
        """
        with open(file_path, 'w', encoding='utf-8') as f:
            # Тип задачи
            f.write(f"{self.dual_objective_type}\n")
            
            # Коэффициенты целевой функции
            f.write(" ".join([f"{coeff:.6f}" for coeff in self.dual_objective_coeffs]) + "\n")
            
            # Ограничения
            for i in range(len(self.dual_constraint_coeffs)):
                coeffs_str = " ".join([f"{coeff:.6f}" for coeff in self.dual_constraint_coeffs[i]])
                f.write(f"{coeffs_str} {self.dual_constraint_types[i]} {self.dual_constraint_rhs[i]:.6f}\n")
            
            # Неограниченные переменные
            if self.dual_unrestricted_vars:
                unrestricted_list = [str(i+1) for i in sorted(self.dual_unrestricted_vars)]
                f.write(f"unrestricted {' '.join(unrestricted_list)}\n")
    
    def print_dual_problem(self):
        """
        Выводит двойственную задачу в читаемом формате.
        """
        print("\n" + "="*80)
        print("ДВОЙСТВЕННАЯ ЗАДАЧА")
        print("="*80)
        
        # Целевая функция
        obj_str = " + ".join([f"{self.dual_objective_coeffs[i]:.2f}·{self.dual_var_names[i]}" 
                              for i in range(len(self.dual_objective_coeffs))])
        print(f"\n{self.dual_objective_type.capitalize()}: W = {obj_str}")
        
        # Ограничения
        print("\nПри ограничениях:")
        for i in range(len(self.dual_constraint_coeffs)):
            constr_str = " + ".join([f"{self.dual_constraint_coeffs[i][j]:.2f}·{self.dual_var_names[j]}" 
                                     for j in range(len(self.dual_constraint_coeffs[i]))])
            print(f"  ({i+1}) {constr_str} {self.dual_constraint_types[i]} {self.dual_constraint_rhs[i]:.2f}")
        
        # Ограничения на переменные
        if self.dual_unrestricted_vars:
            restricted_vars = [self.dual_var_names[i] for i in range(len(self.dual_var_names)) 
                              if i not in self.dual_unrestricted_vars]
            if restricted_vars:
                print(f"  {', '.join(restricted_vars)} ≥ 0")
            unrestricted_vars = [self.dual_var_names[i] for i in self.dual_unrestricted_vars]
            print(f"  {', '.join(unrestricted_vars)} - неограниченные")
        else:
            print(f"  {', '.join(self.dual_var_names)} ≥ 0")
        
        print("="*80)


def solve_dual_problem(primal_file_path, verbose=True):
    """
    Решает исходную задачу, строит двойственную и решает её.
    
    Args:
        primal_file_path: путь к файлу с исходной задачей
        verbose: выводить ли подробную информацию
    
    Returns:
        tuple: (решение исходной задачи, решение двойственной задачи)
    """
    print("="*80)
    print("РЕШЕНИЕ ИСХОДНОЙ (ПРИМАЛЬНОЙ) ЗАДАЧИ")
    print("="*80)
    
    # Решаем исходную задачу
    primal_solver = SimplexSolver(primal_file_path)
    primal_solver.verbose = verbose
    primal_solver.solve()
    
    if primal_solver.is_infeasible or primal_solver.is_unbounded:
        print("\nИсходная задача не имеет решения. Двойственная задача не может быть построена.")
        return None, None
    
    # Строим двойственную задачу
    dual_builder = DualProblemBuilder(primal_solver)
    dual_builder.build_dual()
    dual_builder.print_dual_problem()
    
    # Сохраняем двойственную задачу во временный файл
    import tempfile
    import os
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8')
    temp_file_path = temp_file.name
    dual_builder.write_dual_to_file(temp_file_path)
    temp_file.close()
    
    print("\n" + "="*80)
    print("РЕШЕНИЕ ДВОЙСТВЕННОЙ ЗАДАЧИ")
    print("="*80)
    
    # Решаем двойственную задачу
    dual_solver = SimplexSolver(temp_file_path)
    dual_solver.verbose = verbose
    dual_solver.solve()
    
    # Выводим сравнение результатов
    print("\n" + "="*80)
    print("СРАВНЕНИЕ РЕЗУЛЬТАТОВ")
    print("="*80)
    
    if not dual_solver.is_infeasible and not dual_solver.is_unbounded:
        primal_z = -primal_solver.tableau[-1, -1] if primal_solver.objective_type == 'minimize' else primal_solver.tableau[-1, -1]
        dual_w = -dual_solver.tableau[-1, -1] if dual_solver.objective_type == 'minimize' else dual_solver.tableau[-1, -1]
        
        print(f"\nЗначение целевой функции исходной задачи: Z* = {primal_z:.6f}")
        print(f"Значение целевой функции двойственной задачи: W* = {dual_w:.6f}")
        print(f"Разница: |Z* - W*| = {abs(primal_z - dual_w):.6f}")
        
        if abs(primal_z - dual_w) < 1e-6:
            print("✓ Теорема двойственности подтверждена: Z* = W*")
        else:
            print("⚠ Внимание: значения не совпадают (возможна ошибка округления)")
    
    # Удаляем временный файл
    try:
        os.unlink(temp_file_path)
    except:
        pass
    
    return primal_solver, dual_solver


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Использование: python dual_solver.py <путь_к_файлу_задачи>")
        sys.exit(1)
    
    solve_dual_problem(sys.argv[1], verbose=True)
