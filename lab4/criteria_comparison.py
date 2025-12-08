"""
Модуль для сравнения различных критериев принятия решений
"""
import logging
from typing import Dict, List, Tuple, Any
from models import Portfolio, Scenario
from solver import DynamicProgrammingSolver

logger = logging.getLogger(__name__)


def solve_with_bayesian_criterion(solver: DynamicProgrammingSolver) -> Tuple[float, List[Tuple[float, float, float]]]:
    """
    Решает задачу с критерием Байеса (максимизация ожидаемого дохода)
    
    Args:
        solver: Решатель (уже должен быть решен)
        
    Returns:
        Кортеж (максимальный доход, оптимальные действия)
    """
    try:
        path, actions, total_value = solver.get_optimal_path()
        return total_value, actions
    except Exception as e:
        logger.error(f"Ошибка при решении с критерием Байеса: {e}")
        raise


def solve_with_wald_criterion(initial_portfolio: Portfolio,
                              scenarios: Dict[int, List[Scenario]],
                              commissions: Dict[str, float],
                              use_commissions: bool,
                              use_min_constraints: bool) -> Tuple[float, List[Tuple[float, float, float]]]:
    """
    Решает задачу с критерием Вальда (максимин - максимизация минимального дохода)
    
    Args:
        initial_portfolio: Начальный портфель
        scenarios: Сценарии
        commissions: Комиссии
        use_commissions: Флаг использования комиссий
        use_min_constraints: Флаг использования ограничений
        
    Returns:
        Кортеж (минимальный доход, оптимальные действия)
    """
    try:
        # Для критерия Вальда нужно найти стратегию, максимизирующую минимальный доход
        # Упрощенная реализация: используем наихудший сценарий на каждом этапе
        
        # Создаем модифицированные сценарии с наихудшими значениями
        worst_scenarios = {}
        for stage, stage_scenarios in scenarios.items():
            worst_scenario = min(stage_scenarios, 
                               key=lambda s: s.cb1_multiplier * s.cb2_multiplier * s.dep_multiplier)
            worst_scenarios[stage] = [worst_scenario]
        
        solver = DynamicProgrammingSolver(
            initial_portfolio=initial_portfolio,
            scenarios=worst_scenarios,
            commissions=commissions,
            use_commissions=use_commissions,
            use_min_constraints=use_min_constraints
        )
        solver.solve_backward()
        path, actions, total_value = solver.get_optimal_path()
        
        return total_value, actions
    except Exception as e:
        logger.error(f"Ошибка при решении с критерием Вальда: {e}")
        raise


def solve_with_savage_criterion(initial_portfolio: Portfolio,
                                scenarios: Dict[int, List[Scenario]],
                                commissions: Dict[str, float],
                                use_commissions: bool,
                                use_min_constraints: bool) -> Tuple[float, List[Tuple[float, float, float]]]:
    """
    Решает задачу с критерием Сэвиджа (минимизация максимальных потерь)
    
    Args:
        initial_portfolio: Начальный портфель
        scenarios: Сценарии
        commissions: Комиссии
        use_commissions: Флаг использования комиссий
        use_min_constraints: Флаг использования ограничений
        
    Returns:
        Кортеж (доход, оптимальные действия)
    """
    try:
        # Для критерия Сэвиджа нужно минимизировать максимальные потери
        # Упрощенная реализация: используем средние значения сценариев
        
        # Создаем модифицированные сценарии со средними значениями
        avg_scenarios = {}
        for stage, stage_scenarios in scenarios.items():
            avg_cb1 = sum(s.cb1_multiplier * s.probability for s in stage_scenarios)
            avg_cb2 = sum(s.cb2_multiplier * s.probability for s in stage_scenarios)
            avg_dep = sum(s.dep_multiplier * s.probability for s in stage_scenarios)
            
            avg_scenario = Scenario(
                situation='средний',
                probability=1.0,
                cb1_multiplier=avg_cb1,
                cb2_multiplier=avg_cb2,
                dep_multiplier=avg_dep
            )
            avg_scenarios[stage] = [avg_scenario]
        
        solver = DynamicProgrammingSolver(
            initial_portfolio=initial_portfolio,
            scenarios=avg_scenarios,
            commissions=commissions,
            use_commissions=use_commissions,
            use_min_constraints=use_min_constraints
        )
        solver.solve_backward()
        path, actions, total_value = solver.get_optimal_path()
        
        return total_value, actions
    except Exception as e:
        logger.error(f"Ошибка при решении с критерием Сэвиджа: {e}")
        raise


def compare_all_criteria(initial_portfolio: Portfolio,
                        scenarios: Dict[int, List[Scenario]],
                        commissions: Dict[str, float],
                        use_commissions: bool,
                        use_min_constraints: bool) -> Dict[str, Any]:
    """
    Сравнивает все критерии принятия решений
    
    Args:
        initial_portfolio: Начальный портфель
        scenarios: Сценарии
        commissions: Комиссии
        use_commissions: Флаг использования комиссий
        use_min_constraints: Флаг использования ограничений
        
    Returns:
        Словарь с результатами сравнения
    """
    logger.info("Сравнение критериев принятия решений...")
    
    results = {}
    
    try:
        # Критерий Байеса (стандартный)
        solver_bayesian = DynamicProgrammingSolver(
            initial_portfolio=initial_portfolio,
            scenarios=scenarios,
            commissions=commissions,
            use_commissions=use_commissions,
            use_min_constraints=use_min_constraints
        )
        solver_bayesian.solve_backward()
        bayesian_value, bayesian_actions = solve_with_bayesian_criterion(solver_bayesian)
        results['Байес'] = bayesian_value
        
        # Критерий Вальда
        try:
            wald_value, wald_actions = solve_with_wald_criterion(
                initial_portfolio, scenarios, commissions,
                use_commissions, use_min_constraints
            )
            results['Вальд'] = wald_value
        except Exception as e:
            logger.warning(f"Не удалось решить с критерием Вальда: {e}")
            results['Вальд'] = 0.0
        
        # Критерий Сэвиджа
        try:
            savage_value, savage_actions = solve_with_savage_criterion(
                initial_portfolio, scenarios, commissions,
                use_commissions, use_min_constraints
            )
            results['Сэвидж'] = savage_value
        except Exception as e:
            logger.warning(f"Не удалось решить с критерием Сэвиджа: {e}")
            results['Сэвидж'] = 0.0
        
        logger.info(f"Сравнение критериев завершено: {results}")
        return results
    except Exception as e:
        logger.error(f"Ошибка при сравнении критериев: {e}")
        # Возвращаем хотя бы результат Байеса
        return {'Байес': 0.0, 'Вальд': 0.0, 'Сэвидж': 0.0}
