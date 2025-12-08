"""
Модуль для анализа чувствительности решения к параметрам
"""
import copy
import logging
from typing import Dict, List, Tuple, Any
from models import Portfolio, Scenario
from solver import DynamicProgrammingSolver
from data_loader import initialize_portfolio

logger = logging.getLogger(__name__)


def analyze_sensitivity_probabilities(initial_portfolio: Portfolio,
                                    base_scenarios: Dict[int, List[Scenario]],
                                    commissions: Dict[str, float],
                                    use_commissions: bool,
                                    use_min_constraints: bool,
                                    changes: List[float] = [-10, -5, 0, 5, 10]) -> List[Dict[str, float]]:
    """
    Анализирует чувствительность к изменению вероятностей сценариев
    
    Args:
        initial_portfolio: Начальный портфель
        base_scenarios: Базовые сценарии
        commissions: Словарь комиссий
        use_commissions: Флаг использования комиссий
        use_min_constraints: Флаг использования ограничений
        changes: Список изменений в процентах
        
    Returns:
        Список результатов: [{'parameter': str, 'change_percent': float, 'max_income': float}, ...]
    """
    results = []
    
    try:
        # Базовое решение
        base_solver = DynamicProgrammingSolver(
            initial_portfolio=initial_portfolio,
            scenarios=base_scenarios,
            commissions=commissions,
            use_commissions=use_commissions,
            use_min_constraints=use_min_constraints
        )
        base_solver.solve_backward()
        _, _, base_value = base_solver.get_optimal_path()
        
        # Анализируем каждый этап
        for stage in sorted(base_scenarios.keys()):
            for change_percent in changes:
                if change_percent == 0:
                    # Базовое значение уже вычислено
                    results.append({
                        'parameter': f'stage_{stage}_base',
                        'change_percent': 0.0,
                        'max_income': base_value
                    })
                    continue
                
                # Создаем модифицированные сценарии
                modified_scenarios = copy.deepcopy(base_scenarios)
                stage_scenarios = modified_scenarios[stage]
                
                # Изменяем вероятности пропорционально
                total_prob = sum(s.probability for s in stage_scenarios)
                if total_prob > 1e-6:
                    # Нормализуем изменения
                    change_factor = 1 + change_percent / 100.0
                    new_scenarios = []
                    
                    for scenario in stage_scenarios:
                        new_prob = scenario.probability * change_factor
                        new_scenarios.append(Scenario(
                            situation=scenario.situation,
                            probability=new_prob,
                            cb1_multiplier=scenario.cb1_multiplier,
                            cb2_multiplier=scenario.cb2_multiplier,
                            dep_multiplier=scenario.dep_multiplier
                        ))
                    
                    # Нормализуем вероятности
                    total_new_prob = sum(s.probability for s in new_scenarios)
                    if total_new_prob > 1e-6:
                        for scenario in new_scenarios:
                            scenario.probability /= total_new_prob
                    
                    modified_scenarios[stage] = new_scenarios
                    
                    # Решаем с модифицированными сценариями
                    try:
                        solver = DynamicProgrammingSolver(
                            initial_portfolio=initial_portfolio,
                            scenarios=modified_scenarios,
                            commissions=commissions,
                            use_commissions=use_commissions,
                            use_min_constraints=use_min_constraints
                        )
                        solver.solve_backward()
                        _, _, max_income = solver.get_optimal_path()
                        
                        results.append({
                            'parameter': f'stage_{stage}',
                            'change_percent': change_percent,
                            'max_income': max_income
                        })
                    except Exception as e:
                        logger.warning(f"Ошибка при анализе чувствительности для stage {stage}, change {change_percent}%: {e}")
                        continue
        
        logger.info(f"Анализ чувствительности к вероятностям завершен: {len(results)} результатов")
        return results
    except Exception as e:
        logger.error(f"Ошибка при анализе чувствительности к вероятностям: {e}")
        raise


def analyze_sensitivity_commissions(initial_portfolio: Portfolio,
                                   scenarios: Dict[int, List[Scenario]],
                                   base_commissions: Dict[str, float],
                                   use_commissions: bool,
                                   use_min_constraints: bool,
                                   changes: List[float] = [-10, -5, 0, 5, 10]) -> List[Dict[str, float]]:
    """
    Анализирует чувствительность к изменению комиссий
    
    Args:
        initial_portfolio: Начальный портфель
        scenarios: Сценарии
        base_commissions: Базовые комиссии
        use_commissions: Флаг использования комиссий
        use_min_constraints: Флаг использования ограничений
        changes: Список изменений в процентах
        
    Returns:
        Список результатов: [{'parameter': str, 'change_percent': float, 'max_income': float}, ...]
    """
    results = []
    
    if not use_commissions:
        logger.info("Комиссии выключены, анализ чувствительности к комиссиям пропущен")
        return results
    
    try:
        # Базовое решение
        base_solver = DynamicProgrammingSolver(
            initial_portfolio=initial_portfolio,
            scenarios=scenarios,
            commissions=base_commissions,
            use_commissions=True,
            use_min_constraints=use_min_constraints
        )
        base_solver.solve_backward()
        _, _, base_value = base_solver.get_optimal_path()
        
        # Анализируем каждый тип актива
        for asset in ['cb1', 'cb2', 'dep']:
            for change_percent in changes:
                if change_percent == 0:
                    # Базовое значение
                    results.append({
                        'parameter': f'{asset}_base',
                        'change_percent': 0.0,
                        'max_income': base_value
                    })
                    continue
                
                # Создаем модифицированные комиссии
                modified_commissions = base_commissions.copy()
                base_comm = base_commissions[asset]
                new_comm = max(0.0, base_comm * (1 + change_percent / 100.0))
                modified_commissions[asset] = new_comm
                
                # Решаем с модифицированными комиссиями
                try:
                    solver = DynamicProgrammingSolver(
                        initial_portfolio=initial_portfolio,
                        scenarios=scenarios,
                        commissions=modified_commissions,
                        use_commissions=True,
                        use_min_constraints=use_min_constraints
                    )
                    solver.solve_backward()
                    _, _, max_income = solver.get_optimal_path()
                    
                    results.append({
                        'parameter': asset,
                        'change_percent': change_percent,
                        'max_income': max_income
                    })
                except Exception as e:
                    logger.warning(f"Ошибка при анализе чувствительности для {asset}, change {change_percent}%: {e}")
                    continue
        
        logger.info(f"Анализ чувствительности к комиссиям завершен: {len(results)} результатов")
        return results
    except Exception as e:
        logger.error(f"Ошибка при анализе чувствительности к комиссиям: {e}")
        raise


def run_full_sensitivity_analysis(initial_portfolio: Portfolio,
                                 scenarios: Dict[int, List[Scenario]],
                                 commissions: Dict[str, float],
                                 use_commissions: bool,
                                 use_min_constraints: bool) -> Dict[str, List[Dict[str, float]]]:
    """
    Выполняет полный анализ чувствительности
    
    Args:
        initial_portfolio: Начальный портфель
        scenarios: Сценарии
        commissions: Комиссии
        use_commissions: Флаг использования комиссий
        use_min_constraints: Флаг использования ограничений
        
    Returns:
        Словарь с результатами: {'probabilities': [...], 'commissions': [...]}
    """
    logger.info("Запуск полного анализа чувствительности...")
    
    try:
        prob_results = analyze_sensitivity_probabilities(
            initial_portfolio, scenarios, commissions,
            use_commissions, use_min_constraints
        )
        
        comm_results = []
        if use_commissions:
            comm_results = analyze_sensitivity_commissions(
                initial_portfolio, scenarios, commissions,
                use_commissions, use_min_constraints
            )
        
        return {
            'probabilities': prob_results,
            'commissions': comm_results
        }
    except Exception as e:
        logger.error(f"Ошибка при полном анализе чувствительности: {e}")
        return {
            'probabilities': [],
            'commissions': []
        }
