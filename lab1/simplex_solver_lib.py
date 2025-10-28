import numpy as np
import re
from scipy.optimize import linprog


class LinprogSolver:
    """
    Решает задачу линейного программирования с помощью scipy.optimize.linprog
    и сравнивает результаты с собственной реализацией.
    """

    def __init__(self, file_path):
        self.objective_type = None
        self.objective_coeffs = None
        self.constraint_coeffs = []
        self.constraint_types = []
        self.constraint_rhs = []
        self.unrestricted_vars = set()
        
        self.parse_file(file_path)
        
    def parse_file(self, file_path):
        """Считывает файл с постановкой ЗЛП."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

                self.objective_type = lines[0].strip().lower()
                if self.objective_type not in ['maximize', 'minimize']:
                    raise ValueError("Неверный тип задачи (ожидается 'maximize' или 'minimize')")

                self.objective_coeffs = np.array([float(c) for c in lines[1].strip().split()])
                num_original_vars = len(self.objective_coeffs)

                for line in lines[2:]:
                    line = line.strip()
                    if not line:
                        continue
                    
                    # Проверка на строку unrestricted
                    if line.lower().startswith('unrestricted'):
                        parts = line.split()[1:]
                        if parts and parts[0].lower() == 'all':
                            self.unrestricted_vars = set(range(num_original_vars))
                        else:
                            for var_num_str in parts:
                                var_idx = int(var_num_str) - 1
                                self.unrestricted_vars.add(var_idx)
                        continue

                    parts = re.split(r'\s*([<=>]=?)\s*', line)
                    if len(parts) != 3:
                        raise ValueError(f"Неверный формат ограничения: {line}")

                    self.constraint_coeffs.append(np.array([float(c) for c in parts[0].split()]))
                    self.constraint_types.append(parts[1])
                    self.constraint_rhs.append(float(parts[2]))

        except Exception as e:
            print(f"Ошибка при чтении файла: {e}")
            raise

    def solve(self):
        """
        Решает ЗЛП с помощью scipy.optimize.linprog.
        """
        num_vars = len(self.objective_coeffs)
        
        # Преобразование целевой функции
        c = self.objective_coeffs
        if self.objective_type == 'maximize':
            c = -c  # linprog минимизирует
        
        # Формирование матрицы ограничений
        A_ub = []
        b_ub = []
        A_eq = []
        b_eq = []
        
        for i in range(len(self.constraint_coeffs)):
            coeffs = self.constraint_coeffs[i]
            rhs = self.constraint_rhs[i]
            const_type = self.constraint_types[i]
            
            # Обработка отрицательных правых частей
            if rhs < 0:
                rhs *= -1
                coeffs = -coeffs
                if const_type == '<=':
                    const_type = '>='
                elif const_type == '>=':
                    const_type = '<='
            
            if const_type == '<=':
                A_ub.append(coeffs)
                b_ub.append(rhs)
            elif const_type == '>=':
                A_ub.append(-coeffs)
                b_ub.append(-rhs)
            elif const_type == '=':
                A_eq.append(coeffs)
                b_eq.append(rhs)
        
        # Преобразуем в numpy массивы
        if A_ub:
            A_ub = np.array(A_ub)
            b_ub = np.array(b_ub)
        else:
            A_ub = None
            b_ub = None
            
        if A_eq:
            A_eq = np.array(A_eq)
            b_eq = np.array(b_eq)
        else:
            A_eq = None
            b_eq = None
        
        # Решение задачи
        # Установка границ: unrestricted переменные могут быть любого знака
        bounds = []
        for i in range(num_vars):
            if i in self.unrestricted_vars:
                bounds.append((None, None))  # Без ограничений
            else:
                bounds.append((0, None))  # x_i >= 0
        
        result = linprog(
            c=c,
            A_ub=A_ub,
            b_ub=b_ub,
            A_eq=A_eq,
            b_eq=b_eq,
            bounds=bounds,
            method='highs'  # Используем современный симплекс-метод
        )
        
        return result
    
    def print_solution(self, result):
        """Выводит решение в читаемом формате."""
        print("\n" + "="*60)
        print("РЕШЕНИЕ С ПОМОЩЬЮ SCIPY.OPTIMIZE.LINPROG")
        print("="*60)
        
        if result.success:
            x = result.x
            
            print("\nОптимальная точка:")
            for i in range(len(x)):
                print(f"x{i+1} = {x[i]:.6f}")
            
            z_value = np.dot(self.objective_coeffs, x)
            print(f"\nЗначение целевой функции:")
            print(f"Z = {z_value:.6f}")
            
        else:
            print("\nЗадача не имеет решения:")
            print(f"Причина: {result.message}")
            if result.status == 2:  # Infeasible
                print("Ограничения несовместны (нет допустимых решений)")
            elif result.status == 3:  # Unbounded
                print("Целевая функция не ограничена")
        
        print("\nСтатус решения:", result.message)
        print("Количество итераций:", result.nit if hasattr(result, 'nit') else 'N/A')
        print("="*60)
        
        return result.success


def solve_with_scipy(file_path):
    """
    Обертка для решения задачи с помощью scipy.
    """
    solver = LinprogSolver(file_path)
    result = solver.solve()
    solver.print_solution(result)
    return result


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("Использование: python simplex_solver_lib.py <путь_к_файлу_задачи>")
        sys.exit(1)
    
    solve_with_scipy(sys.argv[1])

