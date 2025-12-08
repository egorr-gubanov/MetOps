# Псевдокод алгоритма динамического программирования

## Общая структура алгоритма

```
АЛГОРИТМ: Оптимальное управление портфелем методом DP

ВХОД:
  - initial_portfolio: начальный портфель (cb1, cb2, dep, cash)
  - scenarios: словарь сценариев по этапам {stage: [Scenario, ...]}
  - commissions: словарь комиссий {'cb1': rate, 'cb2': rate, 'dep': rate}
  - use_commissions: флаг использования комиссий
  - use_min_constraints: флаг использования ограничений на минимум

ВЫХОД:
  - value_function: таблица функции ценности J_t(s_t)
  - best_actions: таблица оптимальных действий u_t*(s_t)
  - optimal_path: оптимальная траектория портфеля
```

## Псевдокод обратного прохождения DP

```
ФУНКЦИЯ solve_backward():
  // ИНИЦИАЛИЗАЦИЯ: Этап T (финальный)
  reachable_states_T = generate_reachable_states(T)
  
  ДЛЯ КАЖДОГО portfolio В reachable_states_T:
    value_function[T][portfolio] = portfolio.total_value()
  
  // ОБРАТНОЕ ПРОХОЖДЕНИЕ
  ДЛЯ t = T-1 ДО 1 (в обратном порядке):
    reachable_states_t = generate_reachable_states(t)
    
    ДЛЯ КАЖДОГО portfolio_t В reachable_states_t:
      best_value = -бесконечность
      best_action = NULL
      
      actions = generate_all_actions(portfolio_t)
      
      ДЛЯ КАЖДОГО action В actions:
        ЕСЛИ НЕ can_apply_action(portfolio_t, action):
          ПРОДОЛЖИТЬ
        
        portfolio_after_action = apply_action(portfolio_t, action)
        
        expected_value = 0.0
        
        ДЛЯ КАЖДОГО scenario В scenarios[t]:
          portfolio_after_scenario = apply_scenario(portfolio_after_action, scenario)
          
          ЕСЛИ НЕ check_constraints(portfolio_after_scenario):
            expected_value = -бесконечность
            ПРЕРВАТЬ
          
          future_value = find_closest_value(portfolio_after_scenario, t+1)
          expected_value += scenario.probability * future_value
        
        ЕСЛИ expected_value > best_value:
          best_value = expected_value
          best_action = action
      
      value_function[t][portfolio_t] = best_value
      best_actions[t][portfolio_t] = best_action

КОНЕЦ ФУНКЦИИ
```

## Псевдокод генерации действий

```
ФУНКЦИЯ generate_all_actions(portfolio):
  actions = []
  
  // Определяем диапазоны для каждого актива
  ЕСЛИ use_min_constraints:
    max_sell_cb1 = portfolio.cb1 - MIN_CB1
    max_sell_cb2 = portfolio.cb2 - MIN_CB2
    max_sell_dep = portfolio.dep - MIN_DEP
  ИНАЧЕ:
    max_sell_cb1 = portfolio.cb1
    max_sell_cb2 = portfolio.cb2
    max_sell_dep = portfolio.dep
  
  // Максимальная покупка ограничена кассой
  max_buy_packages_cb1 = estimate_max_buy_packages(portfolio, 'cb1')
  max_buy_packages_cb2 = estimate_max_buy_packages(portfolio, 'cb2')
  max_buy_packages_dep = estimate_max_buy_packages(portfolio, 'dep')
  
  // Генерируем все комбинации пакетов
  ДЛЯ delta_cb1_packages = -max_sell_packages_cb1 ДО max_buy_packages_cb1:
    ДЛЯ delta_cb2_packages = -max_sell_packages_cb2 ДО max_buy_packages_cb2:
      ДЛЯ delta_dep_packages = -max_sell_packages_dep ДО max_buy_packages_dep:
        action = (
          delta_cb1_packages * PACKAGE_CB1,
          delta_cb2_packages * PACKAGE_CB2,
          delta_dep_packages * PACKAGE_DEP
        )
        
        ЕСЛИ can_apply_action(portfolio, action):
          actions.ДОБАВИТЬ(action)
  
  ВЕРНУТЬ actions

КОНЕЦ ФУНКЦИИ
```

## Псевдокод проверки допустимости действия

```
ФУНКЦИЯ can_apply_action(portfolio, action, commissions, use_commissions, use_min_constraints):
  delta_cb1, delta_cb2, delta_dep = action
  
  // Вычисляем эффективную стоимость
  ЕСЛИ use_commissions:
    effective_cost = calculate_cost_with_commissions(action, commissions)
  ИНАЧЕ:
    effective_cost = delta_cb1 + delta_cb2 + delta_dep
  
  // Проверяем кассовое ограничение
  ЕСЛИ effective_cost > portfolio.cash:
    ВЕРНУТЬ ЛОЖЬ
  
  // Вычисляем новые позиции
  new_cb1 = portfolio.cb1 + delta_cb1
  new_cb2 = portfolio.cb2 + delta_cb2
  new_dep = portfolio.dep + delta_dep
  
  // Проверяем ограничения на минимум
  ЕСЛИ use_min_constraints:
    ЕСЛИ new_cb1 < MIN_CB1 ИЛИ new_cb2 < MIN_CB2 ИЛИ new_dep < MIN_DEP:
      ВЕРНУТЬ ЛОЖЬ
  
  // Проверяем неотрицательность
  ЕСЛИ new_cb1 < 0 ИЛИ new_cb2 < 0 ИЛИ new_dep < 0:
    ВЕРНУТЬ ЛОЖЬ
  
  // Проверяем кассу после действия
  new_cash = portfolio.cash - effective_cost
  ЕСЛИ new_cash < 0:
    ВЕРНУТЬ ЛОЖЬ
  
  ВЕРНУТЬ ИСТИНА

КОНЕЦ ФУНКЦИИ
```

## Псевдокод расчета эффективной стоимости

```
ФУНКЦИЯ calculate_effective_cost(action, commissions, use_commissions):
  delta_cb1, delta_cb2, delta_dep = action
  cost = 0.0
  
  ЕСЛИ use_commissions:
    // С комиссиями
    ЕСЛИ delta_cb1 > 0:
      cost += delta_cb1 * (1 + commissions['cb1'])  // Покупка
    ИНАЧЕ ЕСЛИ delta_cb1 < 0:
      cost += delta_cb1 * (1 - commissions['cb1'])  // Продажа
    
    // Аналогично для cb2 и dep
    ЕСЛИ delta_cb2 > 0:
      cost += delta_cb2 * (1 + commissions['cb2'])
    ИНАЧЕ ЕСЛИ delta_cb2 < 0:
      cost += delta_cb2 * (1 - commissions['cb2'])
    
    ЕСЛИ delta_dep > 0:
      cost += delta_dep * (1 + commissions['dep'])
    ИНАЧЕ ЕСЛИ delta_dep < 0:
      cost += delta_dep * (1 - commissions['dep'])
  ИНАЧЕ:
    // Без комиссий
    cost = delta_cb1 + delta_cb2 + delta_dep
  
  ВЕРНУТЬ cost

КОНЕЦ ФУНКЦИИ
```

## Псевдокод восстановления оптимального пути

```
ФУНКЦИЯ get_optimal_path():
  path = [initial_portfolio]
  actions = []
  current_portfolio = initial_portfolio
  
  ДЛЯ t = 1 ДО T:
    // Находим ближайшее состояние в таблице
    closest_state = find_closest_state(current_portfolio, t)
    
    ЕСЛИ closest_state НЕ NULL:
      action = best_actions[t][closest_state]
    ИНАЧЕ:
      action = (0.0, 0.0, 0.0)  // Без изменений
    
    actions.ДОБАВИТЬ(action)
    
    // Применяем действие
    current_portfolio = apply_action(current_portfolio, action)
    path.ДОБАВИТЬ(current_portfolio)
    
    // Если не последний этап, применяем ожидаемый сценарий
    ЕСЛИ t < T:
      expected_multiplier_cb1 = Σ (scenario.probability * scenario.cb1_multiplier)
      expected_multiplier_cb2 = Σ (scenario.probability * scenario.cb2_multiplier)
      expected_multiplier_dep = Σ (scenario.probability * scenario.dep_multiplier)
      
      current_portfolio = Portfolio(
        cb1 = current_portfolio.cb1 * expected_multiplier_cb1,
        cb2 = current_portfolio.cb2 * expected_multiplier_cb2,
        dep = current_portfolio.dep * expected_multiplier_dep,
        cash = current_portfolio.cash
      )
      path.ДОБАВИТЬ(current_portfolio)
  
  // Получаем максимальную ожидаемую ценность
  closest_initial = find_closest_state(initial_portfolio, 1)
  ЕСЛИ closest_initial:
    total_value = value_function[1][closest_initial]
  ИНАЧЕ:
    total_value = path[-1].total_value()
  
  ВЕРНУТЬ (path, actions, total_value)

КОНЕЦ ФУНКЦИИ
```

## Псевдокод генерации достижимых состояний

```
ФУНКЦИЯ generate_reachable_states(stage):
  ЕСЛИ stage В кэше:
    ВЕРНУТЬ кэш[stage]
  
  reachable = множество()
  
  ЕСЛИ stage == 1:
    prev_states = [initial_portfolio]
  ИНАЧЕ:
    prev_states = все состояния из value_function[stage-1]
  
  ДЛЯ КАЖДОГО portfolio В prev_states:
    actions = generate_all_actions(portfolio)
    
    ДЛЯ КАЖДОГО action В actions:
      ЕСЛИ НЕ can_apply_action(portfolio, action):
        ПРОДОЛЖИТЬ
      
      portfolio_after_action = apply_action(portfolio, action)
      
      ЕСЛИ stage == T:
        reachable.ДОБАВИТЬ(portfolio_after_action)
      ИНАЧЕ:
        ДЛЯ КАЖДОГО scenario В scenarios[stage]:
          portfolio_after_scenario = apply_scenario(portfolio_after_action, scenario)
          
          ЕСЛИ check_constraints(portfolio_after_scenario):
            reachable.ДОБАВИТЬ(portfolio_after_scenario)
  
  кэш[stage] = список(reachable)
  ВЕРНУТЬ кэш[stage]

КОНЕЦ ФУНКЦИИ
```

## Описание основных операций

### Применение действия

```
ФУНКЦИЯ apply_action(portfolio, action, commissions, use_commissions):
  delta_cb1, delta_cb2, delta_dep = action
  
  effective_cost = calculate_effective_cost(action, commissions, use_commissions)
  
  ВЕРНУТЬ Portfolio(
    cb1 = portfolio.cb1 + delta_cb1,
    cb2 = portfolio.cb2 + delta_cb2,
    dep = portfolio.dep + delta_dep,
    cash = portfolio.cash - effective_cost
  )

КОНЕЦ ФУНКЦИИ
```

### Применение сценария

```
ФУНКЦИЯ apply_scenario(portfolio, scenario):
  ВЕРНУТЬ Portfolio(
    cb1 = portfolio.cb1 * scenario.cb1_multiplier,
    cb2 = portfolio.cb2 * scenario.cb2_multiplier,
    dep = portfolio.dep * scenario.dep_multiplier,
    cash = portfolio.cash  // Касса не меняется сценарием
  )

КОНЕЦ ФУНКЦИИ
```

### Проверка ограничений

```
ФУНКЦИЯ check_constraints(portfolio, use_min_constraints):
  result = (portfolio.cash >= 0)
  
  ЕСЛИ use_min_constraints:
    result = result И (portfolio.cb1 >= MIN_CB1) И 
                    (portfolio.cb2 >= MIN_CB2) И 
                    (portfolio.dep >= MIN_DEP)
  
  ВЕРНУТЬ result

КОНЕЦ ФУНКЦИИ
```

## Сложность алгоритма

- **Временная сложность:** O(S × A × Sc × T)
  - S - количество состояний на этапе
  - A - количество действий для состояния
  - Sc - количество сценариев на этапе
  - T - количество этапов

- **Пространственная сложность:** O(S × T)
  - Хранение таблиц функции ценности и оптимальных действий

## Особенности реализации

1. **Дискретизация:** Использование пакетов (25% от начального объема) для ограничения пространства состояний
2. **Кэширование:** Кэш достижимых состояний для ускорения вычислений
3. **Поиск ближайшего состояния:** При отсутствии точного совпадения используется евклидово расстояние
4. **Опциональные функции:** Комиссии и ограничения на минимум можно включать/выключать
