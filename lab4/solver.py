"""
Решатель задачи динамического программирования для оптимизации портфеля
"""
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from models import Portfolio, Scenario, Decision
from constants import PACKAGES, MIN_CB1, MIN_CB2, MIN_DEP, NUM_STAGES, USE_COMMISSIONS, USE_MIN_CONSTRAINTS


class DynamicProgrammingSolver:
    """
    Решатель задачи оптимального управления портфелем методом динамического программирования
    """
    
    def __init__(self, 
                 initial_portfolio: Portfolio,
                 scenarios: Dict[int, List[Scenario]],
                 commissions: Dict[str, float],
                 criterion: str = 'bayesian',
                 use_commissions: bool = USE_COMMISSIONS,
                 use_min_constraints: bool = USE_MIN_CONSTRAINTS):
        """
        Инициализация решателя
        
        Args:
            initial_portfolio: Начальный портфель
            scenarios: Словарь сценариев по этапам {stage: [Scenario, ...]}
            commissions: Словарь комиссий {'cb1': 0.04, 'cb2': 0.07, 'dep': 0.05}
            criterion: Критерий оптимальности ('bayesian' - максимизация ожидаемого дохода)
            use_commissions: Флаг использования комиссий
            use_min_constraints: Флаг использования ограничений на минимум
        """
        self.initial_portfolio = initial_portfolio
        self.scenarios = scenarios
        self.commissions = commissions
        self.criterion = criterion
        self.use_commissions = use_commissions
        self.use_min_constraints = use_min_constraints
        
        # Таблицы функции ценности и оптимальных действий
        # value_function[t][portfolio] = оптимальная ценность состояния
        # best_actions[t][portfolio] = оптимальное действие
        self.value_function: Dict[int, Dict[Portfolio, float]] = defaultdict(dict)
        self.best_actions: Dict[int, Dict[Portfolio, Tuple[float, float, float]]] = defaultdict(dict)
        
        # Кэш для достижимых состояний
        self._reachable_states_cache: Dict[int, List[Portfolio]] = {}
    
    def generate_all_actions(self, portfolio: Portfolio) -> List[Tuple[float, float, float]]:
        """
        Генерирует все возможные действия для портфеля
        
        Учитывает:
        - Ограничения на минимум (нельзя продать ниже минимума)
        - Кассовое ограничение (с учетом комиссий)
        - Дискретизация через пакеты
        
        Args:
            portfolio: Текущий портфель
            
        Returns:
            Список возможных действий (delta_cb1, delta_cb2, delta_dep)
        """
        actions = []
        
        # Определяем диапазоны для каждого актива
        # Максимальная продажа: current - MIN (только если ограничения включены)
        if self.use_min_constraints:
            max_sell_cb1 = portfolio.cb1 - MIN_CB1
            max_sell_cb2 = portfolio.cb2 - MIN_CB2
            max_sell_dep = portfolio.dep - MIN_DEP
        else:
            max_sell_cb1 = portfolio.cb1
            max_sell_cb2 = portfolio.cb2
            max_sell_dep = portfolio.dep
        
        # Максимальная покупка ограничена кассой (с учетом комиссий)
        # Для упрощения используем пакеты
        package_cb1 = PACKAGES['cb1']
        package_cb2 = PACKAGES['cb2']
        package_dep = PACKAGES['dep']
        
        # Генерируем действия в пакетах
        # Для ЦБ1: от -3 до +3 пакетов (или до максимума продажи)
        max_sell_packages_cb1 = int(max_sell_cb1 / package_cb1) if max_sell_cb1 > 0 else 0
        max_buy_packages_cb1 = self._estimate_max_buy_packages(
            portfolio, 'cb1', package_cb1
        )
        
        # Для ЦБ2: от -2 до +2 пакетов
        max_sell_packages_cb2 = int(max_sell_cb2 / package_cb2) if max_sell_cb2 > 0 else 0
        max_buy_packages_cb2 = self._estimate_max_buy_packages(
            portfolio, 'cb2', package_cb2
        )
        
        # Для Деп: от -2 до +2 пакетов
        max_sell_packages_dep = int(max_sell_dep / package_dep) if max_sell_dep > 0 else 0
        max_buy_packages_dep = self._estimate_max_buy_packages(
            portfolio, 'dep', package_dep
        )
        
        # Генерируем все комбинации
        for delta_cb1_packages in range(-min(3, max_sell_packages_cb1), 
                                       min(4, max_buy_packages_cb1 + 1)):
            for delta_cb2_packages in range(-min(2, max_sell_packages_cb2),
                                           min(3, max_buy_packages_cb2 + 1)):
                for delta_dep_packages in range(-min(2, max_sell_packages_dep),
                                               min(3, max_buy_packages_dep + 1)):
                    # Пропускаем действие "ничего не делать" если уже есть
                    if delta_cb1_packages == 0 and delta_cb2_packages == 0 and delta_dep_packages == 0:
                        # Добавляем только один раз
                        if not actions:
                            actions.append((0.0, 0.0, 0.0))
                        continue
                    
                    # Конвертируем пакеты в денежные единицы
                    action = (
                        delta_cb1_packages * package_cb1,
                        delta_cb2_packages * package_cb2,
                        delta_dep_packages * package_dep
                    )
                    
                    # Проверяем допустимость
                    if portfolio.can_apply_action(action, self.commissions, 
                                                  self.use_commissions, 
                                                  self.use_min_constraints):
                        actions.append(action)
        
        # Добавляем действие "ничего не делать" если его еще нет
        if (0.0, 0.0, 0.0) not in actions:
            actions.append((0.0, 0.0, 0.0))
        
        return actions
    
    def _estimate_max_buy_packages(self, portfolio: Portfolio, asset: str, 
                                   package_size: float) -> int:
        """
        Оценивает максимальное количество пакетов, которые можно купить
        
        Args:
            portfolio: Текущий портфель
            asset: Название актива ('cb1', 'cb2', 'dep')
            package_size: Размер пакета в д.е.
            
        Returns:
            Максимальное количество пакетов для покупки
        """
        if self.use_commissions:
            commission = self.commissions[asset]
            # Стоимость одного пакета с комиссией
            cost_per_package = package_size * (1 + commission)
        else:
            # Без комиссий: стоимость = размер пакета
            cost_per_package = package_size
        
        # Максимальное количество пакетов, которые можно купить
        max_packages = int(portfolio.cash / cost_per_package) if cost_per_package > 0 else 0
        
        # Ограничиваем разумным максимумом
        return min(max_packages, 5)
    
    def generate_reachable_states(self, stage: int) -> List[Portfolio]:
        """
        Генерирует все достижимые состояния на этапе stage
        
        Args:
            stage: Номер этапа (1, 2, 3)
            
        Returns:
            Список достижимых состояний портфеля
        """
        if stage in self._reachable_states_cache:
            return self._reachable_states_cache[stage]
        
        reachable = set()
        
        if stage == 1:
            # На этапе 1 начинаем с начального портфеля
            prev_states = [self.initial_portfolio]
        else:
            # На этапе t > 1 берем состояния из предыдущего этапа
            # после применения всех действий и сценариев
            prev_states = list(self.value_function[stage - 1].keys())
        
        # Для каждого предыдущего состояния
        for portfolio in prev_states:
            # Генерируем все возможные действия
            actions = self.generate_all_actions(portfolio)
            
            for action in actions:
                    if not portfolio.can_apply_action(action, self.commissions,
                                                     self.use_commissions,
                                                     self.use_min_constraints):
                        continue
                    
                    # Применяем действие
                    portfolio_after_action = portfolio.apply_action(action, self.commissions,
                                                                   self.use_commissions)
                    
                    if stage == NUM_STAGES:
                        # На последнем этапе просто добавляем состояние после действия
                        reachable.add(portfolio_after_action)
                    else:
                        # Применяем все возможные сценарии
                        for scenario in self.scenarios[stage]:
                            portfolio_after_scenario = portfolio_after_action.apply_scenario(scenario)
                            # Проверяем ограничения
                            if portfolio_after_scenario.check_constraints(self.use_min_constraints):
                                reachable.add(portfolio_after_scenario)
        
        result = list(reachable)
        self._reachable_states_cache[stage] = result
        return result
    
    def solve_backward(self):
        """
        Обратное прохождение динамического программирования
        
        Алгоритм:
        1. Инициализация этапа 3: J_3(s) = total_value(s)
        2. Для t = 2, 1:
           - Для каждого состояния s_t:
             - Перебрать все допустимые действия
             - Вычислить ожидаемую ценность
             - Выбрать оптимальное действие
        """
        print("=" * 70)
        print("ЗАПУСК ДИНАМИЧЕСКОГО ПРОГРАММИРОВАНИЯ")
        print(f"Комиссии: ЦБ1={self.commissions['cb1']:.2%}, "
              f"ЦБ2={self.commissions['cb2']:.2%}, "
              f"Деп={self.commissions['dep']:.2%}")
        print("=" * 70)
        
        # ИНИЦИАЛИЗАЦИЯ: Этап 3 (финальный)
        print(f"\n[ЭТАП {NUM_STAGES}] Инициализация конечных состояний...")
        reachable_states_3 = self.generate_reachable_states(NUM_STAGES)
        print(f"  Найдено {len(reachable_states_3)} достижимых состояний")
        
        for portfolio in reachable_states_3:
            # На последнем этапе ценность = полная стоимость портфеля
            self.value_function[NUM_STAGES][portfolio] = portfolio.total_value()
        
        print(f"  ✓ Инициализировано {len(self.value_function[NUM_STAGES])} состояний")
        
        # ОБРАТНОЕ ПРОХОЖДЕНИЕ
        for t in range(NUM_STAGES - 1, 0, -1):  # t = 2, 1
            print(f"\n[ЭТАП {t}] Обратное прохождение...")
            
            # Получаем все достижимые состояния на этапе t
            reachable_states = self.generate_reachable_states(t)
            print(f"  Найдено {len(reachable_states)} достижимых состояний")
            
            state_count = 0
            for portfolio_t in reachable_states:
                state_count += 1
                
                if state_count % 50 == 0:
                    print(f"    Обработано {state_count} состояний...")
                
                best_value = -float('inf')
                best_action = None
                
                # Перебираем все возможные действия
                actions = self.generate_all_actions(portfolio_t)
                
                for action in actions:
                    # Проверяем допустимость
                    if not portfolio_t.can_apply_action(action, self.commissions,
                                                        self.use_commissions,
                                                        self.use_min_constraints):
                        continue
                    
                    # Применяем действие
                    portfolio_after_action = portfolio_t.apply_action(action, self.commissions,
                                                                      self.use_commissions)
                    
                    # Вычисляем ожидаемую ценность через все сценарии
                    expected_value = 0.0
                    
                    for scenario in self.scenarios[t]:
                        # Применяем сценарий
                        portfolio_after_scenario = portfolio_after_action.apply_scenario(scenario)
                        
                        # Проверяем ограничения
                        if not portfolio_after_scenario.check_constraints(self.use_min_constraints):
                            # Если состояние недопустимо, используем очень низкую ценность
                            expected_value = -float('inf')
                            break
                        
                        # Ищем ближайшее состояние в таблице функции ценности
                        future_value = self._find_closest_value(
                            portfolio_after_scenario, t + 1
                        )
                        
                        if future_value is None:
                            # Если состояние не найдено, используем его собственную ценность
                            future_value = portfolio_after_scenario.total_value()
                        
                        expected_value += scenario.probability * future_value
                    
                    # Обновляем лучший результат
                    if expected_value > best_value:
                        best_value = expected_value
                        best_action = action
                
                # Сохраняем результат
                if best_action is not None:
                    self.value_function[t][portfolio_t] = best_value
                    self.best_actions[t][portfolio_t] = best_action
            
            print(f"  ✓ Обработано {state_count} состояний на этапе {t}")
    
    def _find_closest_value(self, portfolio: Portfolio, stage: int) -> Optional[float]:
        """
        Находит значение функции ценности для ближайшего состояния
        
        Args:
            portfolio: Портфель для поиска
            stage: Номер этапа
            
        Returns:
            Значение функции ценности или None
        """
        if portfolio in self.value_function[stage]:
            return self.value_function[stage][portfolio]
        
        # Если точного совпадения нет, ищем ближайшее состояние
        # (используем евклидово расстояние)
        min_distance = float('inf')
        closest_value = None
        
        for state, value in self.value_function[stage].items():
            distance = (
                (state.cb1 - portfolio.cb1) ** 2 +
                (state.cb2 - portfolio.cb2) ** 2 +
                (state.dep - portfolio.dep) ** 2 +
                (state.cash - portfolio.cash) ** 2
            ) ** 0.5
            
            if distance < min_distance:
                min_distance = distance
                closest_value = value
        
        return closest_value
    
    def get_optimal_path(self) -> Tuple[List[Portfolio], List[Tuple[float, float, float]], float]:
        """
        Восстанавливает оптимальную траекторию
        
        Returns:
            Кортеж (path, actions, total_value)
            - path: список портфелей по этапам
            - actions: список оптимальных действий
            - total_value: максимальная ожидаемая ценность
        """
        path = [self.initial_portfolio]
        actions = []
        current_portfolio = self.initial_portfolio
        
        for t in range(1, NUM_STAGES + 1):
            # Находим ближайшее состояние в таблице
            closest_state = self._find_closest_state(current_portfolio, t)
            
            if closest_state is None:
                # Если не найдено, используем текущий портфель
                action = (0.0, 0.0, 0.0)
            else:
                action = self.best_actions[t].get(closest_state, (0.0, 0.0, 0.0))
            
            actions.append(action)
            
            # Применяем действие
            current_portfolio = current_portfolio.apply_action(action, self.commissions,
                                                              self.use_commissions)
            path.append(current_portfolio)
            
            # Если не последний этап, применяем ожидаемый сценарий
            if t < NUM_STAGES:
                # Используем взвешенное среднее сценариев
                expected_multiplier_cb1 = sum(
                    s.probability * s.cb1_multiplier for s in self.scenarios[t]
                )
                expected_multiplier_cb2 = sum(
                    s.probability * s.cb2_multiplier for s in self.scenarios[t]
                )
                expected_multiplier_dep = sum(
                    s.probability * s.dep_multiplier for s in self.scenarios[t]
                )
                
                current_portfolio = Portfolio(
                    cb1=current_portfolio.cb1 * expected_multiplier_cb1,
                    cb2=current_portfolio.cb2 * expected_multiplier_cb2,
                    dep=current_portfolio.dep * expected_multiplier_dep,
                    cash=current_portfolio.cash
                )
                path.append(current_portfolio)
        
        # Получаем максимальную ожидаемую ценность
        closest_initial = self._find_closest_state(self.initial_portfolio, 1)
        if closest_initial:
            total_value = self.value_function[1].get(closest_initial, 0.0)
        else:
            total_value = path[-1].total_value()
        
        return path, actions, total_value
    
    def _find_closest_state(self, portfolio: Portfolio, stage: int) -> Optional[Portfolio]:
        """Находит ближайшее состояние в таблице"""
        if portfolio in self.value_function[stage]:
            return portfolio
        
        min_distance = float('inf')
        closest_state = None
        
        for state in self.value_function[stage].keys():
            distance = (
                (state.cb1 - portfolio.cb1) ** 2 +
                (state.cb2 - portfolio.cb2) ** 2 +
                (state.dep - portfolio.dep) ** 2 +
                (state.cash - portfolio.cash) ** 2
            ) ** 0.5
            
            if distance < min_distance:
                min_distance = distance
                closest_state = state
        
        return closest_state
