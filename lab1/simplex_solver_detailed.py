import numpy as np
import sys
import re

# Цветные ANSI-коды для вывода
class Colors:
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    RESET = '\033[0m'
    RESET_ALL = '\033[0m'
    BOLD = '\033[1m'

Fore = Colors
Style = Colors


class DetailedSimplexSolver:
    """
    Улучшенная реализация двухфазного симплекс-метода с детальным
    пошаговым выводом всех преобразований и математических операций.
    """

    def __init__(self, file_path):
        self.verbose = True
        self.objective_type = None
        self.objective_coeffs = None
        self.constraint_coeffs = []
        self.constraint_types = []
        self.constraint_rhs = []

        self.var_names = []
        self.all_var_names = []
        self.num_original_vars = 0
        self.basis = []
        
        # Неограниченные переменные (могут быть < 0)
        self.unrestricted_vars = set()
        self.var_mapping = {}

        self.tableau = None

        self.is_unbounded = False
        self.is_infeasible = False

        # Шаг 1: Считывание
        self.parse_file(file_path)
        self.print_initial_problem()

    def print_header(self, text, level=1):
        """
        ЧТО ДЕЛАЕТ: Выводит красивые заголовки разных уровней.
        ПРОСТЫМИ СЛОВАМИ: Делает вывод программы красивым и структурированным.
        """
        if level == 1:
            print(f"\n{Fore.CYAN}{'='*80}")
            print(f"{Fore.CYAN}{text.center(80)}")
            print(f"{Fore.CYAN}{'='*80}{Style.RESET_ALL}\n")
        elif level == 2:
            print(f"\n{Fore.YELLOW}{'─'*80}")
            print(f"{Fore.YELLOW}{text}")
            print(f"{Fore.YELLOW}{'─'*80}{Style.RESET_ALL}\n")
        else:
            print(f"\n{Fore.GREEN}▸ {text}{Style.RESET_ALL}")

    def print_success(self, text):
        """ЧТО ДЕЛАЕТ: Выводит сообщения об успехе зелёным цветом с галочкой ✓"""
        print(f"{Fore.GREEN}✓ {text}{Style.RESET_ALL}")

    def print_warning(self, text):
        """ЧТО ДЕЛАЕТ: Выводит предупреждения жёлтым цветом с треугольником ⚠"""
        print(f"{Fore.YELLOW}⚠ {text}{Style.RESET_ALL}")

    def print_error(self, text):
        """ЧТО ДЕЛАЕТ: Выводит ошибки красным цветом с крестиком ✗"""
        print(f"{Fore.RED}✗ {text}{Style.RESET_ALL}")

    def print_info(self, text):
        """ЧТО ДЕЛАЕТ: Выводит информационные сообщения синим цветом с символом ℹ"""
        print(f"{Fore.BLUE}ℹ {text}{Style.RESET_ALL}")

    def parse_file(self, file_path):
        """
        ЧТО ДЕЛАЕТ: Читает файл с задачей и извлекает все данные.
        ПРОСТЫМИ СЛОВАМИ: Превращает текст из файла в числа для вычислений.
        (Точно такая же функция, как в simplex_solver.py, но с цветным выводом)
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

                self.objective_type = lines[0].strip().lower()
                if self.objective_type not in ['maximize', 'minimize']:
                    raise ValueError("Неверный тип задачи")

                self.objective_coeffs = np.array([float(c) for c in lines[1].strip().split()])
                self.num_original_vars = len(self.objective_coeffs)
                self.var_names = [f'x{i + 1}' for i in range(self.num_original_vars)]

                for line in lines[2:]:
                    line = line.strip()
                    if not line:
                        continue
                    
                    # Проверка на строку unrestricted
                    if line.lower().startswith('unrestricted'):
                        parts = line.split()[1:]
                        if parts and parts[0].lower() == 'all':
                            self.unrestricted_vars = set(range(self.num_original_vars))
                            self.print_info(f"ВСЕ переменные могут быть отрицательными")
                        else:
                            for var_num_str in parts:
                                var_idx = int(var_num_str) - 1
                                self.unrestricted_vars.add(var_idx)
                            var_list = ', '.join([f'x{i+1}' for i in sorted(self.unrestricted_vars)])
                            self.print_info(f"Неограниченные переменные: {var_list}")
                        continue

                    parts = re.split(r'\s*([<=>]=?)\s*', line)
                    if len(parts) != 3:
                        raise ValueError(f"Неверный формат: {line}")

                    self.constraint_coeffs.append(np.array([float(c) for c in parts[0].split()]))
                    self.constraint_types.append(parts[1])
                    self.constraint_rhs.append(float(parts[2]))

            # Обработка отрицательных RHS
            for i in range(len(self.constraint_rhs)):
                if self.constraint_rhs[i] < 0:
                    self.print_warning(f"Ограничение {i+1}: правая часть отрицательна, инвертируем знак")
                    self.constraint_rhs[i] *= -1
                    self.constraint_coeffs[i] *= -1
                    if self.constraint_types[i] == '<=':
                        self.constraint_types[i] = '>='
                    elif self.constraint_types[i] == '>=':
                        self.constraint_types[i] = '<='

        except Exception as e:
            self.print_error(f"Ошибка при чтении файла: {e}")
            sys.exit(1)

    def print_initial_problem(self):
        """Выводит исходную постановку задачи."""
        self.print_header("ЭТАП 0: ИСХОДНАЯ ПОСТАНОВКА ЗАДАЧИ", 1)
        
        # Целевая функция
        if self.objective_type == 'maximize':
            print(f"{Fore.CYAN}Максимизировать:{Style.RESET_ALL} Z = ", end="")
        else:
            print(f"{Fore.CYAN}Минимизировать:{Style.RESET_ALL} Z = ", end="")
        
        terms = []
        for i, c in enumerate(self.objective_coeffs):
            if c != 0:
                if c > 0 and terms:
                    terms.append(f"+ {c}·x{i+1}")
                elif c < 0:
                    terms.append(f"- {abs(c)}·x{i+1}")
                else:
                    terms.append(f"{c}·x{i+1}")
        print(" ".join(terms))
        
        # Ограничения
        print(f"\n{Fore.CYAN}При ограничениях:{Style.RESET_ALL}")
        for i in range(len(self.constraint_coeffs)):
            terms = []
            for j, c in enumerate(self.constraint_coeffs[i]):
                if c != 0:
                    if c > 0 and terms:
                        terms.append(f"+ {c}·x{j+1}")
                    elif c < 0:
                        terms.append(f"- {abs(c)}·x{j+1}")
                    else:
                        terms.append(f"{c}·x{j+1}")
            constraint_str = " ".join(terms)
            print(f"  ({i+1}) {constraint_str} {self.constraint_types[i]} {self.constraint_rhs[i]}")
        
        # Условие неотрицательности
        print(f"  {', '.join(self.var_names)} ≥ 0")

    def print_canonical_form(self):
        """Выводит приведение к каноническому виду."""
        self.print_header("ЭТАП 1: ПРИВЕДЕНИЕ К КАНОНИЧЕСКОМУ ВИДУ", 1)
        
        # 1.1 Преобразование целевой функции
        self.print_header("1.1. Преобразование целевой функции к минимизации", 2)
        
        if self.objective_type == 'maximize':
            self.print_info("Задача на МАКСИМИЗАЦИЮ")
            print(f"Преобразуем: max(Z) → min(-Z)")
            print(f"\nИсходная ЦФ: Z = ", end="")
            self._print_objective(self.objective_coeffs)
            print(f"\nПосле замены: -Z = ", end="")
            self._print_objective(-self.objective_coeffs)
            self.print_success("Теперь минимизируем -Z")
        else:
            self.print_info("Задача уже на МИНИМИЗАЦИЮ")
            print(f"ЦФ: Z = ", end="")
            self._print_objective(self.objective_coeffs)
        
        # 1.2 Преобразование ограничений
        self.print_header("1.2. Добавление дополнительных переменных", 2)
        
        s_count, e_count, a_count = 0, 0, 0
        
        for i in range(len(self.constraint_coeffs)):
            const_type = self.constraint_types[i]
            print(f"\n{Fore.CYAN}Ограничение {i+1}:{Style.RESET_ALL} ", end="")
            self._print_constraint(self.constraint_coeffs[i], const_type, self.constraint_rhs[i])
            
            if const_type == '<=':
                s_count += 1
                print(f"  {Fore.GREEN}→ Добавляем остаточную переменную s{s_count} ≥ 0")
                print(f"  {Fore.GREEN}  Новое ограничение: ", end="")
                self._print_constraint(self.constraint_coeffs[i], '=', self.constraint_rhs[i], extra_var=f"+ s{s_count}")
                print(f"  {Fore.GREEN}  s{s_count} войдёт в начальный базис{Style.RESET_ALL}")
                
            elif const_type == '>=':
                e_count += 1
                a_count += 1
                print(f"  {Fore.YELLOW}→ Добавляем избыточную переменную e{e_count} ≥ 0")
                print(f"  {Fore.YELLOW}→ Добавляем искусственную переменную a{a_count} ≥ 0")
                print(f"  {Fore.YELLOW}  Новое ограничение: ", end="")
                self._print_constraint(self.constraint_coeffs[i], '=', self.constraint_rhs[i], 
                                     extra_var=f"- e{e_count} + a{a_count}")
                print(f"  {Fore.YELLOW}  a{a_count} войдёт в начальный базис{Style.RESET_ALL}")
                
            elif const_type == '=':
                a_count += 1
                print(f"  {Fore.MAGENTA}→ Добавляем искусственную переменную a{a_count} ≥ 0")
                print(f"  {Fore.MAGENTA}  Новое ограничение: ", end="")
                self._print_constraint(self.constraint_coeffs[i], '=', self.constraint_rhs[i], 
                                     extra_var=f"+ a{a_count}")
                print(f"  {Fore.MAGENTA}  a{a_count} войдёт в начальный базис{Style.RESET_ALL}")
        
        print(f"\n{Fore.CYAN}Итого добавлено переменных:{Style.RESET_ALL}")
        print(f"  • Остаточных (s): {s_count}")
        print(f"  • Избыточных (e): {e_count}")
        print(f"  • Искусственных (a): {a_count}")
        
        if a_count > 0:
            self.print_warning(f"Есть искусственные переменные → требуется двухфазный метод")
        else:
            self.print_success("Искусственные переменные не нужны → одна фаза")

    def _print_objective(self, coeffs):
        """Вывод целевой функции."""
        terms = []
        for i, c in enumerate(coeffs):
            if c != 0:
                if c > 0 and terms:
                    terms.append(f"+ {c}·x{i+1}")
                elif c < 0:
                    terms.append(f"- {abs(c)}·x{i+1}")
                else:
                    terms.append(f"{c}·x{i+1}")
        print(" ".join(terms))

    def _print_constraint(self, coeffs, sign, rhs, extra_var=""):
        """Вывод ограничения."""
        terms = []
        for i, c in enumerate(coeffs):
            if c != 0:
                if c > 0 and terms:
                    terms.append(f"+ {c}·x{i+1}")
                elif c < 0:
                    terms.append(f"- {abs(c)}·x{i+1}")
                else:
                    terms.append(f"{c}·x{i+1}")
        constraint_str = " ".join(terms)
        if extra_var:
            constraint_str += f" {extra_var}"
        print(f"{constraint_str} {sign} {rhs}")

    def prepare_tableau(self):
        """Формирование начальной симплекс-таблицы."""
        self.print_canonical_form()
        
        self.print_header("ЭТАП 2: ФОРМИРОВАНИЕ НАЧАЛЬНОЙ СИМПЛЕКС-ТАБЛИЦЫ", 1)

        # Шаг 0: Замена неограниченных переменных
        if self.unrestricted_vars:
            self.print_header(f"Замена неограниченных переменных", 3)
            self.print_info(f"Обрабатываем {len(self.unrestricted_vars)} неограниченных переменных")
            
            new_obj_coeffs = []
            new_constraint_coeffs = [[] for _ in range(len(self.constraint_coeffs))]
            new_var_names = []
            
            for var_idx in range(self.num_original_vars):
                if var_idx in self.unrestricted_vars:
                    old_coeff = self.objective_coeffs[var_idx]
                    new_obj_coeffs.append(old_coeff)
                    new_obj_coeffs.append(-old_coeff)
                    
                    plus_idx = len(new_var_names)
                    new_var_names.append(f'{self.var_names[var_idx]}+')
                    new_var_names.append(f'{self.var_names[var_idx]}-')
                    self.var_mapping[var_idx] = (plus_idx, plus_idx + 1)
                    
                    for constr_idx in range(len(self.constraint_coeffs)):
                        old_constr_coeff = self.constraint_coeffs[constr_idx][var_idx]
                        new_constraint_coeffs[constr_idx].append(old_constr_coeff)
                        new_constraint_coeffs[constr_idx].append(-old_constr_coeff)
                    
                    print(f"  {Fore.CYAN}{self.var_names[var_idx]} = {self.var_names[var_idx]}+ - {self.var_names[var_idx]}-{Style.RESET_ALL}")
                else:
                    new_obj_coeffs.append(self.objective_coeffs[var_idx])
                    self.var_mapping[var_idx] = (len(new_var_names), None)
                    new_var_names.append(self.var_names[var_idx])
                    
                    for constr_idx in range(len(self.constraint_coeffs)):
                        new_constraint_coeffs[constr_idx].append(self.constraint_coeffs[constr_idx][var_idx])
            
            self.objective_coeffs = np.array(new_obj_coeffs)
            self.constraint_coeffs = [np.array(row) for row in new_constraint_coeffs]
            self.var_names = new_var_names
            self.num_original_vars = len(self.var_names)

        num_constraints = len(self.constraint_coeffs)
        self.all_var_names = list(self.var_names)
        A_ext = np.array(self.constraint_coeffs)

        s_count, e_count, a_count = 0, 0, 0
        self.basis = []

        # Добавление дополнительных переменных
        for i in range(num_constraints):
            const_type = self.constraint_types[i]

            if const_type == '<=':
                s_count += 1
                var_name = f's{s_count}'
                self.all_var_names.append(var_name)
                s_col = np.zeros((num_constraints, 1))
                s_col[i, 0] = 1.0
                A_ext = np.hstack((A_ext, s_col))
                self.basis.append(len(self.all_var_names) - 1)

            elif const_type == '>=':
                e_count += 1
                var_name_e = f'e{e_count}'
                self.all_var_names.append(var_name_e)
                e_col = np.zeros((num_constraints, 1))
                e_col[i, 0] = -1.0
                A_ext = np.hstack((A_ext, e_col))

                a_count += 1
                var_name_a = f'a{a_count}'
                self.all_var_names.append(var_name_a)
                a_col = np.zeros((num_constraints, 1))
                a_col[i, 0] = 1.0
                A_ext = np.hstack((A_ext, a_col))
                self.basis.append(len(self.all_var_names) - 1)

            elif const_type == '=':
                a_count += 1
                var_name_a = f'a{a_count}'
                self.all_var_names.append(var_name_a)
                a_col = np.zeros((num_constraints, 1))
                a_col[i, 0] = 1.0
                A_ext = np.hstack((A_ext, a_col))
                self.basis.append(len(self.all_var_names) - 1)

        num_vars_total = A_ext.shape[1]

        # Строка Z
        z_row = np.zeros(num_vars_total + 1)
        z_coeffs = self.objective_coeffs.copy()
        if self.objective_type == 'maximize':
            z_coeffs = -z_coeffs
        z_row[:self.num_original_vars] = z_coeffs

        self.print_info("Формирование строки целевой функции Z")
        print(f"Коэффициенты при переменных: {z_row[:-1]}")

        # Строка W
        w_row = np.zeros(num_vars_total + 1)
        if a_count > 0:
            self.print_info(f"Формирование вспомогательной функции W (сумма {a_count} искусственных переменных)")
            a_indices = [i for i, name in enumerate(self.all_var_names) if name.startswith('a')]
            w_row[a_indices] = 1.0
            print(f"W = {' + '.join([self.all_var_names[i] for i in a_indices])}")

        self.tableau = np.vstack((
            np.hstack((A_ext, np.array(self.constraint_rhs).reshape(-1, 1))),
            z_row,
            w_row
        ))

        print(f"\n{Fore.CYAN}Начальный базис:{Style.RESET_ALL}")
        for i, b_idx in enumerate(self.basis):
            print(f"  Строка {i+1}: {self.all_var_names[b_idx]}")

        # Приведение к базисному виду
        self.print_header("2.1. Приведение строк Z и W к базисному виду", 3)
        self.tableau = self._update_obj_rows_for_basis(self.tableau, self.basis, 
                                                        [num_constraints, num_constraints + 1])

        w_row_idx = self.tableau.shape[0] - 1
        self.print_tableau("Начальная симплекс-таблица (после приведения к базисному виду)", 
                          w_row_idx, [num_constraints, num_constraints + 1])

    def _update_obj_rows_for_basis(self, tableau, basis, obj_row_indices):
        """Приводит строки ЦФ к базисному виду."""
        for row_idx in obj_row_indices:
            obj_row = tableau[row_idx, :]
            for i, basis_col_idx in enumerate(basis):
                coeff = obj_row[basis_col_idx]
                if np.abs(coeff) > 1e-6:
                    print(f"  Строка {row_idx}: вычитаем {coeff:.2f} × (строка {i+1})")
                    obj_row -= coeff * tableau[i, :]
        return tableau

    def _find_pivot_row(self, tableau, pivot_col, num_constraint_rows):
        """Находит разрешающую строку."""
        rhs_col = tableau.shape[1] - 1
        min_ratio = float('inf')
        pivot_row = -1

        print(f"\n  {Fore.CYAN}Поиск разрешающей строки (min-ratio test):{Style.RESET_ALL}")
        for i in range(num_constraint_rows):
            val = tableau[i, pivot_col]
            if val > 1e-6:
                rhs = tableau[i, rhs_col]
                ratio = rhs / val
                print(f"    Строка {i+1}: {rhs:.4f} / {val:.4f} = {ratio:.4f}")
                # Правило Бланда: при равных отношениях выбираем строку с меньшим индексом
                if ratio < min_ratio - 1e-9:
                    min_ratio = ratio
                    pivot_row = i
                elif abs(ratio - min_ratio) < 1e-9 and (pivot_row == -1 or i < pivot_row):
                    pivot_row = i

        if pivot_row != -1:
            self.print_success(f"Выбрана строка {pivot_row+1} (минимальное отношение = {min_ratio:.4f})")
        else:
            self.print_error("Нет подходящей строки → задача неограничена")
        
        return pivot_row

    def _solve(self, tableau, basis, obj_row_idx, num_constraint_rows, phase_name):
        """Один цикл симплекс-метода."""
        self.print_header(f"ИТЕРАЦИИ СИМПЛЕКС-МЕТОДА ({phase_name})", 2)
        
        iteration = 1
        while True:
            print(f"\n{Fore.YELLOW}{'─'*80}")
            print(f"{Fore.YELLOW}Итерация {iteration}{Style.RESET_ALL}")
            print(f"{Fore.YELLOW}{'─'*80}{Style.RESET_ALL}")

            obj_row = tableau[obj_row_idx, :-1]

            # 1. Выбор разрешающего столбца (правило Бланда для детерминированности)
            # Выбираем столбец с наименьшим индексом среди тех, где коэффициент отрицательный
            pivot_col = -1
            for j in range(len(obj_row)):
                if obj_row[j] < -1e-6:
                    pivot_col = j
                    break

            if pivot_col == -1:
                self.print_success("Оптимум достигнут! Все коэффициенты в строке ЦФ ≥ 0")
                return "optimal", tableau, basis

            print(f"\n  {Fore.CYAN}1. Выбор разрешающего столбца:{Style.RESET_ALL}")
            print(f"     {Fore.GREEN}→ Столбец: {self.all_var_names[pivot_col]} (индекс {pivot_col})")
            print(f"     {Fore.GREEN}→ Значение в строке ЦФ: {obj_row[pivot_col]:.4f} (первый отрицательный){Style.RESET_ALL}")

            # 2. Выбор разрешающей строки
            print(f"\n  {Fore.CYAN}2. Выбор разрешающей строки:{Style.RESET_ALL}")
            pivot_row = self._find_pivot_row(tableau, pivot_col, num_constraint_rows)

            if pivot_row == -1:
                self.print_error("Задача не ограничена!")
                return "unbounded", tableau, basis

            pivot_element = tableau[pivot_row, pivot_col]
            print(f"\n  {Fore.CYAN}3. Разрешающий элемент:{Style.RESET_ALL}")
            print(f"     {Fore.MAGENTA}→ Позиция: ({pivot_row+1}, {pivot_col+1})")
            print(f"     {Fore.MAGENTA}→ Значение: {pivot_element:.4f}{Style.RESET_ALL}")

            # 3. Обновление базиса
            old_basis_var = self.all_var_names[basis[pivot_row]]
            new_basis_var = self.all_var_names[pivot_col]
            
            print(f"\n  {Fore.CYAN}4. Изменение базиса:{Style.RESET_ALL}")
            print(f"     {Fore.RED}← Выходит из базиса: {old_basis_var}")
            print(f"     {Fore.GREEN}→ Входит в базис: {new_basis_var}{Style.RESET_ALL}")
            
            basis[pivot_row] = pivot_col

            # 4. Преобразование таблицы (метод Гаусса-Жордана)
            print(f"\n  {Fore.CYAN}5. Преобразование таблицы (метод Гаусса-Жордана):{Style.RESET_ALL}")
            print(f"     а) Делим разрешающую строку на {pivot_element:.4f}")
            tableau[pivot_row, :] /= pivot_element

            print(f"     б) Обнуляем остальные элементы в столбце {self.all_var_names[pivot_col]}")
            for i in range(tableau.shape[0]):
                if i != pivot_row:
                    factor = tableau[i, pivot_col]
                    if np.abs(factor) > 1e-10:
                        print(f"        Строка {i+1}: вычитаем {factor:.4f} × (разрешающая строка)")
                        tableau[i, :] -= factor * tableau[pivot_row, :]

            obj_rows = [r for r in [num_constraint_rows, num_constraint_rows + 1] if r < tableau.shape[0]]
            self.print_tableau(f"Таблица после итерации {iteration}", obj_row_idx, obj_rows)

            iteration += 1

    def solve(self):
        """Главный метод решения."""
        self.prepare_tableau()

        num_constraints = len(self.constraint_coeffs)
        w_row_idx = self.tableau.shape[0] - 1
        z_row_idx = w_row_idx - 1

        a_indices = [i for i, name in enumerate(self.all_var_names) if name.startswith('a')]

        # ФАЗА 1
        if not a_indices:
            self.print_header("ФАЗА 1 НЕ ТРЕБУЕТСЯ", 1)
            self.print_success("Искусственные переменные отсутствуют")
            self.tableau = np.delete(self.tableau, w_row_idx, axis=0)
        else:
            self.print_header("ФАЗА 1: ПОИСК ДОПУСТИМОГО БАЗИСНОГО РЕШЕНИЯ", 1)
            self.print_info(f"Минимизируем W = сумму {len(a_indices)} искусственных переменных")
            
            status, self.tableau, self.basis = self._solve(
                self.tableau, self.basis, w_row_idx, num_constraints, "ФАЗА 1"
            )

            if status == "unbounded":
                self.print_error("Вспомогательная задача не ограничена")
                self.is_infeasible = True
                return

            min_w = self.tableau[w_row_idx, -1]
            
            print(f"\n{Fore.CYAN}Результат Фазы 1:{Style.RESET_ALL}")
            print(f"  min(W) = {min_w:.6f}")

            if min_w > 1e-6:
                self.print_error(f"min(W) = {min_w:.6f} > 0")
                self.print_error("Задача НЕ ИМЕЕТ ДОПУСТИМЫХ РЕШЕНИЙ (ограничения несовместны)")
                self.is_infeasible = True
                return
            else:
                self.print_success(f"min(W) = {min_w:.6f} ≈ 0")
                self.print_success("Найдено допустимое базисное решение!")

            # Переход к Фазе 2
            self.print_header("ПЕРЕХОД К ФАЗЕ 2", 2)
            self.print_info("Удаляем строку W и столбцы искусственных переменных")
            
            self.tableau = np.delete(self.tableau, w_row_idx, axis=0)
            
            a_indices.sort(reverse=True)
            for a_idx in a_indices:
                print(f"  Удаляем столбец: {self.all_var_names[a_idx]}")
                self.tableau = np.delete(self.tableau, a_idx, axis=1)
                self.all_var_names.pop(a_idx)
                self.basis = [b if b < a_idx else b - 1 for b in self.basis]

        # ФАЗА 2
        self.print_header("ФАЗА 2: РЕШЕНИЕ ОСНОВНОЙ ЗАДАЧИ", 1)
        
        z_row_idx = self.tableau.shape[0] - 1
        self.print_tableau("Начальная таблица Фазы 2", z_row_idx, [z_row_idx])

        status, self.tableau, self.basis = self._solve(
            self.tableau, self.basis, z_row_idx, num_constraints, "ФАЗА 2"
        )

        if status == "unbounded":
            self.print_error("Целевая функция не ограничена")
            self.is_unbounded = True
            return

        self.print_tableau("Финальная оптимальная таблица", z_row_idx, [z_row_idx])

        # РЕЗУЛЬТАТ
        self.print_header("ИТОГОВЫЙ РЕЗУЛЬТАТ", 1)
        self.print_results()

    def print_results(self):
        """Вывод финального результата."""
        if self.is_infeasible:
            self.print_error("Задача НЕ ИМЕЕТ РЕШЕНИЯ (ограничения несовместны)")
            return
        
        if self.is_unbounded:
            self.print_error("Задача НЕ ИМЕЕТ РЕШЕНИЯ (целевая функция не ограничена)")
            return

        self.print_success("ОПТИМАЛЬНОЕ РЕШЕНИЕ НАЙДЕНО!")
        
        print(f"\n{Fore.CYAN}Оптимальная точка:{Style.RESET_ALL}")
        solution = {}
        for var_name in self.var_names:
            solution[var_name] = 0.0

        rhs_col = self.tableau.shape[1] - 1
        for i, basis_col_idx in enumerate(self.basis):
            if basis_col_idx < self.num_original_vars:
                var_name = self.all_var_names[basis_col_idx]
                solution[var_name] = self.tableau[i, rhs_col]

        for var_name, value in solution.items():
            print(f"  {var_name} = {Fore.GREEN}{value:.6f}{Style.RESET_ALL}")

        final_z_value = self.tableau[-1, -1]
        
        # Правильный вывод значения ЦФ
        # В таблице храни��ся значение минимизированной функции
        # Для maximize мы минимизировали -Z, поэтому значение в таблице = -Z_opt
        # Поэтому Z_opt = -final_z_value (для maximize)
        # Для minimize значение в таблице уже корректное
        if self.objective_type == 'minimize':
            z_display = -final_z_value
        else:
            z_display = final_z_value
        print(f"\n{Fore.CYAN}Значение целевой функции:{Style.RESET_ALL}")
        print(f"  Z = {Fore.GREEN}{z_display:.6f}{Style.RESET_ALL}")

    def print_tableau(self, title, main_obj_row_idx, all_obj_row_indices):
        """Красивая печать симплекс-таблицы."""
        print(f"\n{Fore.YELLOW}{'═'*80}")
        print(f"{Fore.YELLOW}{title}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}{'═'*80}{Style.RESET_ALL}\n")

        header = self.all_var_names + ['RHS']
        col_width = 10
        
        # Заголовок
        print(f"{Fore.CYAN}{'Базис':<{col_width}}", end="")
        for h in header:
            print(f"{h:>{col_width}}", end="")
        print(f"{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'-'*(col_width * (len(header) + 1))}{Style.RESET_ALL}")

        num_constraints = len(self.constraint_coeffs)

        # Строки ограничений
        for i in range(num_constraints):
            basis_var_name = self.all_var_names[self.basis[i]]
            print(f"{Fore.GREEN}{basis_var_name:<{col_width}}{Style.RESET_ALL}", end="")
            for val in self.tableau[i, :]:
                print(f"{val:{col_width}.4f}", end="")
            print()

        print(f"{Fore.CYAN}{'-'*(col_width * (len(header) + 1))}{Style.RESET_ALL}")

        # Строки ЦФ
        for row_idx in all_obj_row_indices:
            if row_idx >= self.tableau.shape[0]:
                continue

            is_main = (row_idx == main_obj_row_idx)
            obj_name = "W" if row_idx > main_obj_row_idx else "Z"
            
            if is_main:
                print(f"{Fore.MAGENTA}{'→ ' + obj_name:<{col_width}}{Style.RESET_ALL}", end="")
            else:
                print(f"{Fore.YELLOW}{obj_name:<{col_width}}{Style.RESET_ALL}", end="")
            
            for val in self.tableau[row_idx, :]:
                if is_main and val < -1e-6:
                    print(f"{Fore.RED}{val:{col_width}.4f}{Style.RESET_ALL}", end="")
                else:
                    print(f"{val:{col_width}.4f}", end="")
            print()
        
        print(f"{Fore.CYAN}{'═'*(col_width * (len(header) + 1))}{Style.RESET_ALL}\n")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Использование: python simplex_solver_detailed.py <путь_к_файлу_задачи>")
        sys.exit(1)

    solver = DetailedSimplexSolver(sys.argv[1])
    solver.solve()

