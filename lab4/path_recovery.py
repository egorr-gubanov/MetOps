"""
Восстановление оптимальной траектории и симуляция пути
"""
import random
from typing import List, Tuple, Dict, Optional
from models import Portfolio, Scenario
from solver import DynamicProgrammingSolver
from constants import NUM_STAGES


def recover_path(solver: DynamicProgrammingSolver, 
                initial_portfolio: Portfolio,
                scenarios: Dict[int, List[Scenario]]) -> Tuple[List[Portfolio], List[Tuple[float, float, float]], float]:
    """
    Восстанавливает оптимальную траекторию из решений DP
    
    Args:
        solver: Решатель с уже решенной задачей
        initial_portfolio: Начальный портфель
        scenarios: Словарь сценариев по этапам
        
    Returns:
        Кортеж (path, actions, total_value)
    """
    return solver.get_optimal_path()


def simulate_path(portfolio: Portfolio,
                 actions: List[Tuple[float, float, float]],
                 scenarios: Dict[int, List[Scenario]],
                 commissions: Dict[str, float],
                 use_commissions: bool = False,
                 random_seed: Optional[int] = None) -> List[Portfolio]:
    """
    Симулирует путь портфеля с конкретными сценариями
    
    Args:
        portfolio: Начальный портфель
        actions: Список действий для каждого этапа
        scenarios: Словарь сценариев по этапам
        commissions: Словарь комиссий
        use_commissions: Флаг использования комиссий
        random_seed: Семя для генератора случайных чисел (для воспроизводимости)
        
    Returns:
        Список портфелей по этапам
    """
    if random_seed is not None:
        random.seed(random_seed)
    
    path = [portfolio]
    current_portfolio = portfolio
    
    for t in range(1, NUM_STAGES + 1):
        if t - 1 < len(actions):
            action = actions[t - 1]
        else:
            action = (0.0, 0.0, 0.0)
        
        # Применяем действие
        current_portfolio = current_portfolio.apply_action(action, commissions, use_commissions)
        path.append(current_portfolio)
        
        # Если не последний этап, применяем случайный сценарий
        if t < NUM_STAGES and t in scenarios:
            # Выбираем сценарий согласно вероятностям
            scenario = _select_scenario_by_probability(scenarios[t])
            current_portfolio = current_portfolio.apply_scenario(scenario)
            path.append(current_portfolio)
    
    return path


def _select_scenario_by_probability(scenarios: List[Scenario]) -> Scenario:
    """
    Выбирает сценарий согласно вероятностям
    
    Args:
        scenarios: Список сценариев
        
    Returns:
        Выбранный сценарий
    """
    r = random.random()
    cumulative = 0.0
    
    for scenario in scenarios:
        cumulative += scenario.probability
        if r <= cumulative:
            return scenario
    
    # Если что-то пошло не так, возвращаем последний
    return scenarios[-1]


def monte_carlo_simulation(solver: DynamicProgrammingSolver,
                          initial_portfolio: Portfolio,
                          scenarios: Dict[int, List[Scenario]],
                          n_simulations: int = 1000,
                          random_seed: Optional[int] = None) -> Dict:
    """
    Выполняет Monte Carlo симуляцию для валидации решения
    
    Args:
        solver: Решатель с уже решенной задачей
        initial_portfolio: Начальный портфель
        scenarios: Словарь сценариев по этапам
        n_simulations: Количество симуляций
        random_seed: Семя для генератора случайных чисел
        
    Returns:
        Словарь с результатами: {'mean': float, 'std': float, 'min': float, 'max': float}
    """
    if random_seed is not None:
        random.seed(random_seed)
    
    # Получаем оптимальные действия
    _, actions, _ = solver.get_optimal_path()
    
    results = []
    
    for _ in range(n_simulations):
        path = simulate_path(initial_portfolio, actions, scenarios, 
                           solver.commissions, solver.use_commissions)
        final_value = path[-1].total_value()
        results.append(final_value)
    
    import statistics
    
    return {
        'mean': statistics.mean(results),
        'std': statistics.stdev(results) if len(results) > 1 else 0.0,
        'min': min(results),
        'max': max(results),
        'results': results
    }


def print_path_details(path: List[Portfolio], 
                      actions: List[Tuple[float, float, float]]):
    """
    Выводит детальную информацию о пути
    
    Args:
        path: Список портфелей по этапам
        actions: Список действий
    """
    print("\n" + "=" * 70)
    print("ДЕТАЛЬНАЯ ТРАЕКТОРИЯ ПОРТФЕЛЯ")
    print("=" * 70)
    
    stage = 0
    for i, portfolio in enumerate(path):
        if i % 2 == 0:
            # Состояние до действия
            print(f"\nЭтап {stage}, состояние ДО действия:")
            print(f"  {portfolio}")
            
            if stage < len(actions):
                action = actions[stage]
                delta_cb1, delta_cb2, delta_dep = action
                print(f"  Действие: ЦБ1={delta_cb1:+.2f}, ЦБ2={delta_cb2:+.2f}, Деп={delta_dep:+.2f}")
            stage += 1
        else:
            # Состояние после сценария
            print(f"  После сценария:")
            print(f"  {portfolio}")
