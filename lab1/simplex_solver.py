import numpy as np
import sys
import re


class SimplexSolver:
    """
    Реализует двухфазный симплекс-метод с подробным
    пошаговым выводом в консоль.
    """

    def __init__(self, file_path):
        # --- ФЛАГ ДЛЯ ПОДРОБНОГО ВЫВОДА ---
        # Установите True для пошагового вывода, False - для тихого режима
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
        self.unrestricted_vars = set()  # индексы переменных (0-based)
        self.var_mapping = {}  # отображение: старый индекс -> (индекс x+, индекс x-)

        self.tableau = None

        self.is_unbounded = False
        self.is_infeasible = False

        # Шаг 1: Считывание
        self.parse_file(file_path)

        if self.verbose:
            print("[ЭТАП 1: Считывание и парсинг файла ЗАВЕРШЕН]")
            print(f"  Тип задачи: {self.objective_type}")
            print(f"  Коэффициенты Z: {self.objective_coeffs}")
            print(f"  Ограничения ({len(self.constraint_coeffs)}):")
            for i in range(len(self.constraint_coeffs)):
                print(f"    {self.constraint_coeffs[i]} {self.constraint_types[i]} {self.constraint_rhs[i]}")
            print("-" * 40)

    def parse_file(self, file_path):
        """
        ЧТО ДЕЛАЕТ: Читает текстовый файл с задачей и вытаскивает из него:
        - Что мы делаем: максимизируем или минимизируем?
        - Коэффициенты целевой функции (числа, которые мы умножаем на переменные)
        - Все ограничения (условия, которые должны выполняться)
        - Какие переменные могут быть отрицательными (unrestricted)
        
        ПРОСТЫМИ СЛОВАМИ: Берёт файл и превращает текст в числа, с которыми можно работать.
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

                self.objective_type = lines[0].strip().lower()
                if self.objective_type not in ['maximize', 'minimize']:
                    raise ValueError("Неверный тип задачи (ожидается 'maximize' или 'minimize')")

                self.objective_coeffs = np.array([float(c) for c in lines[1].strip().split()])
                self.num_original_vars = len(self.objective_coeffs)
                self.var_names = [f'x{i + 1}' for i in range(self.num_original_vars)]

                for line in lines[2:]:
                    line = line.strip()
                    if not line:
                        continue
                    
                    # Проверка на строку unrestricted
                    if line.lower().startswith('unrestricted'):
                        parts = line.split()[1:]  # убираем слово "unrestricted"
                        if parts and parts[0].lower() == 'all':
                            # Все переменные неограниченные
                            self.unrestricted_vars = set(range(self.num_original_vars))
                            if self.verbose:
                                print(f"  [Парсинг]: ВСЕ переменные могут быть отрицательными")
                        else:
                            # Конкретные переменные (номера 1-based)
                            for var_num_str in parts:
                                var_idx = int(var_num_str) - 1  # переводим в 0-based
                                self.unrestricted_vars.add(var_idx)
                            if self.verbose:
                                var_list = ', '.join([f'x{i+1}' for i in sorted(self.unrestricted_vars)])
                                print(f"  [Парсинг]: Неограниченные переменные: {var_list}")
                        continue

                    parts = re.split(r'\s*([<=>]=?)\s*', line)
                    if len(parts) != 3:
                        raise ValueError(f"Неверный формат ограничения: {line}")

                    self.constraint_coeffs.append(np.array([float(c) for c in parts[0].split()]))
                    self.constraint_types.append(parts[1])
                    self.constraint_rhs.append(float(parts[2]))

            for i in range(len(self.constraint_rhs)):
                if self.constraint_rhs[i] < 0:
                    if self.verbose:
                        print(f"  [Парсинг]: Обнаружено отрицательное RHS в строке {i}, инвертирую знак ограничения.")
                    self.constraint_rhs[i] *= -1
                    self.constraint_coeffs[i] *= -1
                    if self.constraint_types[i] == '<=':
                        self.constraint_types[i] = '>='
                    elif self.constraint_types[i] == '>=':
                        self.constraint_types[i] = '<='

        except Exception as e:
            print(f"Ошибка при чтении файла: {e}")
            sys.exit(1)

    def prepare_tableau(self):
        """
        ЧТО ДЕЛАЕТ: Готовит большую таблицу для решения задачи. Это самая важная подготовка!
        
        Шаги:
        1. Заменяем неограниченные переменные на разность двух неотрицательных: x = x+ - x-
        2. Если была задача "максимизировать", превращаем её в "минимизировать" (умножаем на -1)
        3. Добавляем дополнительные переменные к ограничениям:
           - Для "<=" добавляем "slack" переменные (s1, s2, ...) - это как "запас"
           - Для ">=" добавляем "surplus" (e1, e2, ...) и "artificial" (a1, a2, ...) переменные
           - Для "=" добавляем только "artificial" переменные
        4. Строим большую таблицу со всеми числами
        5. Добавляем строку W (вспомогательная функция) если нужно
        
        ПРОСТЫМИ СЛОВАМИ: Превращаем задачу в специальную таблицу, с которой умеет работать алгоритм.
        """
        if self.verbose:
            print("[ЭТАП 2 и 3: Приведение к каноническому виду и Формирование вспомогательной задачи (Фаза 1)]")

        # Шаг 0: Замена неограниченных переменных
        if self.unrestricted_vars:
            if self.verbose:
                print(f"  [Замена переменных]: Обработка {len(self.unrestricted_vars)} неограниченных переменных")
            
            # Новые коэффициенты для целевой функции
            new_obj_coeffs = []
            # Новые коэффициенты для ограничений
            new_constraint_coeffs = [[] for _ in range(len(self.constraint_coeffs))]
            # Новые имена переменных
            new_var_names = []
            
            for var_idx in range(self.num_original_vars):
                if var_idx in self.unrestricted_vars:
                    # x_i = x_i+ - x_i-
                    old_coeff = self.objective_coeffs[var_idx]
                    new_obj_coeffs.append(old_coeff)   # коэффициент для x+
                    new_obj_coeffs.append(-old_coeff)  # коэффициент для x-
                    
                    plus_idx = len(new_var_names)
                    new_var_names.append(f'{self.var_names[var_idx]}+')
                    new_var_names.append(f'{self.var_names[var_idx]}-')
                    self.var_mapping[var_idx] = (plus_idx, plus_idx + 1)
                    
                    # Обновляем ограничения
                    for constr_idx in range(len(self.constraint_coeffs)):
                        old_constr_coeff = self.constraint_coeffs[constr_idx][var_idx]
                        new_constraint_coeffs[constr_idx].append(old_constr_coeff)   # для x+
                        new_constraint_coeffs[constr_idx].append(-old_constr_coeff)  # для x-
                    
                    if self.verbose:
                        print(f"    {self.var_names[var_idx]} = {self.var_names[var_idx]}+ - {self.var_names[var_idx]}-")
                else:
                    # Обычная переменная (x_i >= 0)
                    new_obj_coeffs.append(self.objective_coeffs[var_idx])
                    self.var_mapping[var_idx] = (len(new_var_names), None)
                    new_var_names.append(self.var_names[var_idx])
                    
                    for constr_idx in range(len(self.constraint_coeffs)):
                        new_constraint_coeffs[constr_idx].append(self.constraint_coeffs[constr_idx][var_idx])
            
            # Обновляем данные
            self.objective_coeffs = np.array(new_obj_coeffs)
            self.constraint_coeffs = [np.array(row) for row in new_constraint_coeffs]
            self.var_names = new_var_names
            self.num_original_vars = len(self.var_names)

        num_constraints = len(self.constraint_coeffs)
        self.all_var_names = list(self.var_names)
        A_ext = np.array(self.constraint_coeffs)

        s_count, e_count, a_count = 0, 0, 0
        self.basis = []

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
                if self.verbose:
                    print(f"  Ограничение {i} (<=): Добавлена остаточная переменная {var_name}, включена в базис.")

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
                if self.verbose:
                    print(
                        f"  Ограничение {i} (>=): Добавлены избыточная {var_name_e} и искусственная {var_name_a}. {var_name_a} включена в базис.")

            elif const_type == '=':
                a_count += 1
                var_name_a = f'a{a_count}'
                self.all_var_names.append(var_name_a)
                a_col = np.zeros((num_constraints, 1))
                a_col[i, 0] = 1.0
                A_ext = np.hstack((A_ext, a_col))
                self.basis.append(len(self.all_var_names) - 1)
                if self.verbose:
                    print(f"  Ограничение {i} (=): Добавлена искусственная переменная {var_name_a}, включена в базис.")

        num_vars_total = A_ext.shape[1]

        # Cтрока Z
        z_row = np.zeros(num_vars_total + 1)
        z_coeffs = self.objective_coeffs
        if self.objective_type == 'maximize':
            z_coeffs = -z_coeffs
            if self.verbose:
                print("  [ЦФ]: Задача 'maximize', будем минимизировать -Z.")
        z_row[:self.num_original_vars] = z_coeffs

        # Строка W
        w_row = np.zeros(num_vars_total + 1)
        if a_count > 0:
            if self.verbose:
                print("  [ЦФ]: Сформирована вспомогательная функция W (сумма 'a' переменных).")
            a_indices = [i for i, name in enumerate(self.all_var_names) if name.startswith('a')]
            w_row[a_indices] = 1.0

        self.tableau = np.vstack((
            np.hstack((A_ext, np.array(self.constraint_rhs).reshape(-1, 1))),
            z_row,
            w_row
        ))

        if self.verbose:
            print("  [Таблица]: Z и W приводятся к базисному виду...")
        self.tableau = self._update_obj_rows_for_basis(self.tableau, self.basis, [num_constraints, num_constraints + 1])

        if self.verbose:
            w_row_idx = self.tableau.shape[0] - 1
            self.print_tableau("КОНТРОЛЬНАЯ ТОЧКА: Начальная таблица Фазы 1 (до итераций)", w_row_idx,
                               [num_constraints, num_constraints + 1])

    def _update_obj_rows_for_basis(self, tableau, basis, obj_row_indices):
        """
        ЧТО ДЕЛАЕТ: Приводит в порядок строки с целевыми функциями (Z и W).
        
        ПРОСТЫМИ СЛОВАМИ: В начале таблицы некоторые столбцы должны быть равны нулю 
        (те, где стоят базисные переменные). Эта функция вычитает строки друг из друга,
        чтобы в нужных местах получились нули.
        """
        for row_idx in obj_row_indices:
            obj_row = tableau[row_idx, :]
            for i, basis_col_idx in enumerate(basis):
                coeff = obj_row[basis_col_idx]
                if np.abs(coeff) > 1e-6:
                    obj_row -= coeff * tableau[i, :]
        return tableau

    def _find_pivot_row(self, tableau, pivot_col, num_constraint_rows):
        """
        ЧТО ДЕЛАЕТ: Выбирает строку, с которой будем работать на этом шаге.
        
        КАК: Делит правую часть на элемент в выбранном столбце для каждой строки.
        Выбирает строку, где это отношение минимальное (но положительное!).
        Если отношения одинаковые - берём строку с меньшим номером (правило Бланда).
        
        ПРОСТЫМИ СЛОВАМИ: Ищем, какую переменную нужно вывести из решения,
        чтобы не нарушить условия задачи.
        """
        rhs_col = tableau.shape[1] - 1
        min_ratio = float('inf')
        pivot_row = -1

        if self.verbose:
            print(
                f"    [Поиск строки]: Ищем мин. отношение для столбца {self.all_var_names[pivot_col]} (индекс {pivot_col}):")

        for i in range(num_constraint_rows):
            val = tableau[i, pivot_col]
            if val > 1e-6:  # Знаменатель > 0
                rhs = tableau[i, rhs_col]
                ratio = rhs / val
                if self.verbose:
                    print(f"      Строка {i}: {rhs:.2f} / {val:.2f} = {ratio:.2f}")
                # Правило Бланда: при равных отношениях выбираем строку с меньшим индексом
                if ratio < min_ratio - 1e-9:
                    min_ratio = ratio
                    pivot_row = i
                elif abs(ratio - min_ratio) < 1e-9 and (pivot_row == -1 or i < pivot_row):
                    pivot_row = i
            # else:
            #     if self.verbose:
            #         print(f"      Строка {i}: Значение в столбце ({val:.2f}) <= 0, пропускаем.")

        if self.verbose:
            if pivot_row != -1:
                print(f"    [Поиск строки]: Выбрана строка {pivot_row} (мин. отношение = {min_ratio:.2f}).")
            else:
                print(f"    [Поиск строки]: Не найдено подходящей строки (все <= 0).")
        return pivot_row

    def _solve(self, tableau, basis, obj_row_idx, num_constraint_rows):
        """
        ЧТО ДЕЛАЕТ: Это главный цикл решения! Повторяется, пока не найдём ответ.
        
        На каждом шаге:
        1. Смотрим в строку целевой функции - есть ли отрицательные числа?
        2. Если нет - мы нашли оптимум! Ура!
        3. Если есть - выбираем ПЕРВЫЙ столбец с отрицательным числом (правило Бланда)
        4. Выбираем строку по минимальному отношению
        5. Пересчитываем всю таблицу через этот "опорный элемент"
        6. Повторяем, пока не найдём оптимум
        
        ПРОСТЫМИ СЛОВАМИ: Постепенно улучшаем решение, делая шаги от одной вершины 
        многоугольника допустимых решений к другой, пока не найдём самую лучшую точку.
        """
        iteration = 1
        while True:
            if self.verbose:
                print(f"\n  --- Итерация {iteration} ---")

            obj_row = tableau[obj_row_idx, :-1]

            # 1. Выбор разрешающего столбца (правило Бланда для детерминированности)
            # Выбираем столбец с наименьшим индексом среди тех, где коэффициент отрицательный
            pivot_col = -1
            for j in range(len(obj_row)):
                if obj_row[j] < -1e-6:
                    pivot_col = j
                    break

            if pivot_col == -1:
                if self.verbose:
                    print("  [Поиск столбца]: Оптимум найден (все Cj-Zj >= 0). Завершение итераций.")
                return "optimal", tableau, basis

            if self.verbose:
                print(
                    f"  [Поиск столбца]: Выбран столбец {self.all_var_names[pivot_col]} (индекс {pivot_col}) со значением {obj_row[pivot_col]:.2f}.")

            # 2. Выбор разрешающей строки
            pivot_row = self._find_pivot_row(tableau, pivot_col, num_constraint_rows)

            if pivot_row == -1:
                if self.verbose:
                    print("  [Поиск строки]: Ошибка! Решение не ограничено.")
                return "unbounded", tableau, basis

            if self.verbose:
                print(
                    f"  [Пивот]: Разрешающий элемент в ({pivot_row}, {pivot_col}) = {tableau[pivot_row, pivot_col]:.2f}")
                print(
                    f"  [Базис]: Переменная {self.all_var_names[basis[pivot_row]]} выходит из базиса, {self.all_var_names[pivot_col]} входит.")

            # 3. Обновление базиса
            basis[pivot_row] = pivot_col

            # 4. Пересчет таблицы (Гаусс-Жордан)
            pivot_element = tableau[pivot_row, pivot_col]
            tableau[pivot_row, :] /= pivot_element

            for i in range(tableau.shape[0]):
                if i != pivot_row:
                    factor = tableau[i, pivot_col]
                    tableau[i, :] -= factor * tableau[pivot_row, :]

            if self.verbose:
                obj_rows = [r for r in [num_constraint_rows, num_constraint_rows + 1] if r < tableau.shape[0]]
                self.print_tableau(f"Таблица после итерации {iteration}", obj_row_idx, obj_rows)

            iteration += 1

    def solve(self):
        """
        ЧТО ДЕЛАЕТ: Это главная функция, которая запускает весь процесс решения задачи!
        
        Шаги:
        1. ФАЗА 1 (если нужна): Проверяем, можно ли вообще решить задачу.
           Ищем хотя бы какое-то допустимое решение, даже не оптимальное.
           Если не находим - задача несовместна (нет решений).
        
        2. ФАЗА 2: Улучшаем найденное допустимое решение до оптимального.
           Делаем его максимально хорошим.
        
        3. Выводим результат: либо оптимальное решение, либо сообщение об ошибке.
        
        ПРОСТЫМИ СЛОВАМИ: Сначала ищем "хоть какое-то" решение, потом делаем его "самым лучшим".
        """

        # Этапы 1, 2, 3 происходят в __init__ и prepare_tableau
        self.prepare_tableau()

        num_constraints = len(self.constraint_coeffs)

        w_row_idx = self.tableau.shape[0] - 1
        z_row_idx = w_row_idx - 1

        a_indices = [i for i, name in enumerate(self.all_var_names) if name.startswith('a')]

        # --- ФАЗА 1 ---
        if self.verbose:
            print("\n[ЭТАП 4: Решение вспомогательной задачи (Фаза 1)]")

        if not a_indices:
            if self.verbose:
                print("  Искусственные переменные не требуются. Пропуск Фазы 1.")
            self.tableau = np.delete(self.tableau, w_row_idx, axis=0)  # Удаляем W-строку
        else:
            status, self.tableau, self.basis = self._solve(
                self.tableau, self.basis, w_row_idx, num_constraints
            )

            if status == "unbounded":
                print("Ошибка: Вспомогательная задача не ограничена.")
                self.is_infeasible = True
                return

            min_w = self.tableau[w_row_idx, -1]
            if self.verbose:
                self.print_tableau(f"КОНТРОЛЬНАЯ ТОЧКА: Итоговая таблица Фазы 1", w_row_idx, [z_row_idx, w_row_idx])

            if min_w > 1e-6:
                print(f"Фаза 1 завершена. min(W) = {min_w:.2f} > 0.")
                print("Задача не имеет допустимых решений (ограничения несовместны).")
                self.is_infeasible = True
                return
            else:
                if self.verbose:
                    print(f"Фаза 1 завершена. min(W) = {min_w:.2f}. Найдено допустимое решение.")

            # --- ФАЗА 5 ---
            if self.verbose:
                print("\n[ЭТАП 5: Переход к основной задаче (Фаза 2)]")

            # Удаляем столбцы искусственных переменных и строку W
            self.tableau = np.delete(self.tableau, w_row_idx, axis=0)  # Удаляем W-строку
            if self.verbose:
                print("  Строка W удалена.")

            a_indices.sort(reverse=True)
            deleted_names = []
            for a_idx in a_indices:
                self.tableau = np.delete(self.tableau, a_idx, axis=1)
                deleted_names.append(self.all_var_names.pop(a_idx))
                self.basis = [b if b < a_idx else b - 1 for b in self.basis]

            if self.verbose:
                print(f"  Столбцы искусственных переменных ({', '.join(reversed(deleted_names))}) удалены.")

        # --- ФАЗА 6 ---
        if self.verbose:
            print("\n[ЭТАП 6: Решение основной задачи (Фаза 2)]")

        z_row_idx = self.tableau.shape[0] - 1  # Теперь это последняя строка

        if self.verbose:
            self.print_tableau("КОНТРОЛЬНАЯ ТОЧКА: Начальная таблица Фазы 2", z_row_idx, [z_row_idx])

        status, self.tableau, self.basis = self._solve(
            self.tableau, self.basis, z_row_idx, num_constraints
        )

        if status == "unbounded":
            print("Задача не имеет оптимального решения (целевая функция не ограничена).")
            self.is_unbounded = True
            return

        if self.verbose:
            self.print_tableau("КОНТРОЛЬНАЯ ТОЧКА: Финальная таблица (Оптимум Фазы 2)", z_row_idx, [z_row_idx])

        # --- ФАЗА 7 ---
        print("\n" + "-" * 40)
        print("[ЭТАП 7: Формирование ответа]")
        if not self.is_unbounded and not self.is_infeasible:
            print("Оптимальное решение найдено!")

        self.print_results()

    def print_results(self):
        """
        ЧТО ДЕЛАЕТ: Красиво выводит на экран найденное решение.
        
        Показывает:
        - Значения всех переменных (x1, x2, x3, ...)
        - Значение целевой функции Z
        - Или сообщение об ошибке, если решения нет
        
        ПРОСТЫМИ СЛОВАМИ: Берёт числа из таблицы и показывает их в понятном виде.
        """
        if self.is_infeasible or self.is_unbounded:
            print("-" * 40)
            return

        print("\nОптимальная точка:")

        solution = {}
        for var_name in self.var_names:
            solution[var_name] = 0.0

        rhs_col = self.tableau.shape[1] - 1

        for i, basis_col_idx in enumerate(self.basis):
            if basis_col_idx < self.num_original_vars:
                var_name = self.all_var_names[basis_col_idx]
                solution[var_name] = self.tableau[i, rhs_col]

        # Если были неограниченные переменные, восстанавливаем их исходные значения
        if self.unrestricted_vars:
            original_solution = {}
            # Читаем из self.__init__ сохраненные оригинальные имена
            original_num_vars = len(self.var_mapping)
            
            for orig_idx in range(original_num_vars):
                mapping = self.var_mapping.get(orig_idx)
                if mapping:
                    plus_idx, minus_idx = mapping
                    if minus_idx is not None:
                        # Неограниченная переменная: x = x+ - x-
                        plus_name = self.var_names[plus_idx]
                        minus_name = self.var_names[minus_idx]
                        x_plus = solution.get(plus_name, 0.0)
                        x_minus = solution.get(minus_name, 0.0)
                        # Восстанавливаем оригинальное имя (удаляем '+' из имени)
                        orig_name = plus_name[:-1]  # убираем '+'
                        original_solution[orig_name] = x_plus - x_minus
                    else:
                        # Обычная переменная
                        var_name = self.var_names[plus_idx]
                        original_solution[var_name] = solution.get(var_name, 0.0)
            
            # Выводим оригинальные значения
            for var_name in sorted(original_solution.keys(), key=lambda x: int(x[1:]) if x[1:].isdigit() else x):
                value = original_solution[var_name]
                print(f"{var_name} = {value:.2f}")
        else:
            # Обычный вывод
            for var_name, value in solution.items():
                print(f"{var_name} = {value:.2f}")

        final_z_value = self.tableau[-1, -1]

        # Для minimize в таблице хранится -Z (поскольку мы минимизируем -Z = maximize Z)
        # Для maximize значение уже корректное
        if self.objective_type == 'minimize':
            print(f"\nЗначение целевой функции:\nZ = {-final_z_value:.2f}")
        else:
            print(f"\nЗначение целевой функции:\nZ = {final_z_value:.2f}")

        print("-" * 40)

    def print_tableau(self, title, main_obj_row_idx, all_obj_row_indices):
        """
        ЧТО ДЕЛАЕТ: Выводит таблицу симплекс-метода на экран в красивом виде.
        
        Показывает:
        - Все переменные (столбцы)
        - Все ограничения (строки)
        - Базисные переменные (какие переменные сейчас в решении)
        - Строки целевых функций (Z и W)
        
        ПРОСТЫМИ СЛОВАМИ: Превращает большую таблицу с числами в читаемый формат,
        чтобы можно было понять, что происходит на каждом шаге.
        """
        print(f"\n--- {title} ---")

        # Заголовки столбцов
        header = self.all_var_names + ['RHS']
        col_width = 9
        print("Basis".ljust(col_width), end="")
        for h in header:
            print(h.ljust(col_width), end="")
        print()

        num_constraints = len(self.constraint_coeffs)

        # Строки ограничений
        for i in range(num_constraints):
            basis_var_name = self.all_var_names[self.basis[i]]
            print(basis_var_name.ljust(col_width), end="")
            for val in self.tableau[i, :]:
                print(f"{val: {col_width}.2f}", end="")
            print()

        # Строки ЦФ
        for row_idx in all_obj_row_indices:
            if row_idx >= self.tableau.shape[0]:
                continue  # Строка уже могла быть удалена (например, W)

            is_main = (row_idx == main_obj_row_idx)
            # Определяем имя строки ЦФ (Z или W)
            obj_name = "W" if row_idx > main_obj_row_idx else "Z"
            if is_main:
                obj_name = f"-> {obj_name}"  # Указываем, по какой строке оптимизируем

            print(obj_name.ljust(col_width), end="")
            for val in self.tableau[row_idx, :]:
                print(f"{val: {col_width}.2f}", end="")
            print()
        print("-" * (col_width * (len(header) + 1)))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Использование: python simplex_solver_verbose.py <путь_к_файлу_задачи>")
        sys.exit(1)

    # verbose=True включен по умолчанию в __init__
    solver = SimplexSolver(sys.argv[1])
    solver.solve()
