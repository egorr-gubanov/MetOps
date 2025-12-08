# Диаграмма классов программы

## Текстовая диаграмма классов

```
┌─────────────────────────────────────────────────────────────────┐
│                         Portfolio                               │
├─────────────────────────────────────────────────────────────────┤
│ - cb1: float                                                    │
│ - cb2: float                                                    │
│ - dep: float                                                    │
│ - cash: float                                                   │
├─────────────────────────────────────────────────────────────────┤
│ + total_value() -> float                                        │
│ + check_constraints(use_min_constraints) -> bool               │
│ + can_apply_action(action, commissions, use_commissions,        │
│                    use_min_constraints) -> bool                │
│ + apply_action(action, commissions, use_commissions)            │
│   -> Portfolio                                                  │
│ + apply_scenario(scenario) -> Portfolio                        │
│ - _calculate_effective_cost(action, commissions,                │
│                             use_commissions) -> float          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ использует
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                         Scenario                                │
├─────────────────────────────────────────────────────────────────┤
│ - situation: str                                               │
│ - probability: float                                           │
│ - cb1_multiplier: float                                        │
│ - cb2_multiplier: float                                        │
│ - dep_multiplier: float                                        │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                         Decision                                │
├─────────────────────────────────────────────────────────────────┤
│ - portfolio: Portfolio                                         │
│ - stage: int                                                    │
│ - action: Tuple[float, float, float]                           │
│ - expected_value: float                                        │
│ - details: Dict                                                 │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                  DynamicProgrammingSolver                       │
├─────────────────────────────────────────────────────────────────┤
│ - initial_portfolio: Portfolio                                 │
│ - scenarios: Dict[int, List[Scenario]]                         │
│ - commissions: Dict[str, float]                                │
│ - criterion: str                                                │
│ - use_commissions: bool                                         │
│ - use_min_constraints: bool                                     │
│ - value_function: Dict[int, Dict[Portfolio, float]]            │
│ - best_actions: Dict[int, Dict[Portfolio, Tuple]]              │
│ - _reachable_states_cache: Dict[int, List[Portfolio]]          │
├─────────────────────────────────────────────────────────────────┤
│ + __init__(initial_portfolio, scenarios, commissions,           │
│            criterion, use_commissions, use_min_constraints)    │
│ + generate_all_actions(portfolio) -> List[Tuple]                │
│ + generate_reachable_states(stage) -> List[Portfolio]          │
│ + solve_backward()                                              │
│ + get_optimal_path() -> Tuple[List, List, float]                │
│ - _estimate_max_buy_packages(portfolio, asset, package_size)   │
│   -> int                                                        │
│ - _find_closest_value(portfolio, stage) -> Optional[float]       │
│ - _find_closest_state(portfolio, stage) -> Optional[Portfolio] │
└─────────────────────────────────────────────────────────────────┘
         │                    │
         │ использует         │ создает
         ▼                    ▼
┌──────────────────┐  ┌──────────────────┐
│   Portfolio      │  │   Scenario       │
└──────────────────┘  └──────────────────┘
```

## Описание классов

### 1. Portfolio (dataclass, frozen=True)

**Назначение:** Представляет состояние портфеля инвестора в денежном выражении.

**Атрибуты:**
- `cb1: float` - стоимость ЦБ1 в д.е.
- `cb2: float` - стоимость ЦБ2 в д.е.
- `dep: float` - стоимость Депозитов в д.е.
- `cash: float` - свободные средства в д.е.

**Методы:**
- `total_value() -> float` - вычисляет полную стоимость портфеля
- `check_constraints(use_min_constraints, min_cb1, min_cb2, min_dep) -> bool` - проверяет ограничения
- `can_apply_action(action, commissions, use_commissions, use_min_constraints, ...) -> bool` - проверяет допустимость действия
- `apply_action(action, commissions, use_commissions) -> Portfolio` - применяет действие к портфелю
- `apply_scenario(scenario) -> Portfolio` - применяет сценарий к портфелю
- `_calculate_effective_cost(action, commissions, use_commissions) -> float` - вычисляет эффективную стоимость действия

**Особенности:**
- Класс является frozen dataclass для использования в качестве ключа словаря
- Все значения в денежном выражении (д.е.)

### 2. Scenario (dataclass)

**Назначение:** Представляет один сценарий развития на этапе.

**Атрибуты:**
- `situation: str` - название ситуации (благоприятная, нейтральная, негативная)
- `probability: float` - вероятность наступления сценария
- `cb1_multiplier: float` - мультипликатор для ЦБ1
- `cb2_multiplier: float` - мультипликатор для ЦБ2
- `dep_multiplier: float` - мультипликатор для Депозитов

**Методы:**
- Нет методов (только данные)

**Особенности:**
- Используется для описания стохастических изменений стоимости активов

### 3. Decision (dataclass)

**Назначение:** Хранит оптимальное решение для состояния портфеля.

**Атрибуты:**
- `portfolio: Portfolio` - состояние портфеля
- `stage: int` - номер этапа
- `action: Tuple[float, float, float]` - оптимальное действие (delta_cb1, delta_cb2, delta_dep)
- `expected_value: float` - ожидаемая ценность
- `details: Dict` - дополнительные детали решения

**Методы:**
- Нет методов (только данные)

**Особенности:**
- Используется для хранения результатов решения

### 4. DynamicProgrammingSolver (class)

**Назначение:** Реализует алгоритм динамического программирования для оптимизации портфеля.

**Атрибуты:**
- `initial_portfolio: Portfolio` - начальный портфель
- `scenarios: Dict[int, List[Scenario]]` - сценарии по этапам
- `commissions: Dict[str, float]` - комиссии брокеров
- `criterion: str` - критерий оптимальности ('bayesian')
- `use_commissions: bool` - флаг использования комиссий
- `use_min_constraints: bool` - флаг использования ограничений на минимум
- `value_function: Dict[int, Dict[Portfolio, float]]` - таблица функции ценности
- `best_actions: Dict[int, Dict[Portfolio, Tuple]]` - таблица оптимальных действий
- `_reachable_states_cache: Dict[int, List[Portfolio]]` - кэш достижимых состояний

**Методы:**
- `__init__(...)` - инициализация решателя
- `generate_all_actions(portfolio) -> List[Tuple]` - генерирует все возможные действия
- `generate_reachable_states(stage) -> List[Portfolio]` - генерирует достижимые состояния
- `solve_backward()` - обратное прохождение DP
- `get_optimal_path() -> Tuple[List, List, float]` - восстанавливает оптимальный путь
- `_estimate_max_buy_packages(...) -> int` - оценивает максимальное количество пакетов для покупки
- `_find_closest_value(...) -> Optional[float]` - находит значение функции ценности для ближайшего состояния
- `_find_closest_state(...) -> Optional[Portfolio]` - находит ближайшее состояние в таблице

**Особенности:**
- Реализует алгоритм обратного прохождения DP
- Использует кэширование для оптимизации
- Поддерживает опциональные комиссии и ограничения

## Связи между классами

1. **Portfolio использует Scenario:**
   - Метод `apply_scenario()` применяет сценарий к портфелю

2. **DynamicProgrammingSolver использует Portfolio:**
   - Хранит портфели в таблицах функции ценности и оптимальных действий
   - Генерирует действия для портфелей
   - Проверяет допустимость действий

3. **DynamicProgrammingSolver использует Scenario:**
   - Использует сценарии для вычисления ожидаемой ценности

4. **Decision использует Portfolio:**
   - Хранит состояние портфеля в решении

## Вспомогательные модули

### data_loader.py
- Функции для загрузки данных из Excel
- Валидация сценариев
- Инициализация портфеля

### path_recovery.py
- Функции восстановления оптимальной траектории
- Monte Carlo симуляция

### visualization.py
- Функции построения графиков
- Визуализация результатов

### constants.py
- Константы проекта (начальные значения, комиссии, ограничения, флаги)

## Диаграмма зависимостей модулей

```
main.py
  │
  ├──> data_loader.py
  │      └──> constants.py
  │      └──> models.py
  │
  ├──> solver.py
  │      └──> constants.py
  │      └──> models.py
  │
  ├──> path_recovery.py
  │      └──> solver.py
  │      └──> models.py
  │      └──> constants.py
  │
  └──> visualization.py
         └──> models.py
         └──> solver.py
         └──> constants.py
```

## Принципы проектирования

1. **Разделение ответственности:**
   - `models.py` - модели данных
   - `solver.py` - алгоритм решения
   - `data_loader.py` - загрузка данных
   - `visualization.py` - визуализация

2. **Инкапсуляция:**
   - Приватные методы начинаются с `_`
   - Данные инкапсулированы в классах

3. **Расширяемость:**
   - Опциональные функции (комиссии, ограничения) через флаги
   - Легко добавить новые критерии оптимальности

4. **Иммутабельность:**
   - Portfolio - frozen dataclass для использования в словарях
   - Применение действий создает новые объекты
