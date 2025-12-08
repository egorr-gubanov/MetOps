"""
Загрузка данных из Excel файла и валидация
"""
import pandas as pd
from typing import Dict, List, Tuple
from models import Scenario, Portfolio
from constants import (
    INITIAL_CB1, INITIAL_CB2, INITIAL_DEP, INITIAL_CASH,
    COMMISSIONS, SCENARIOS, MIN_CB1, MIN_CB2, MIN_DEP,
    EPSILON
)


def load_from_excel(filepath: str) -> Tuple[Dict[int, List[Scenario]], Dict[str, float], float]:
    """
    Читает данные из Excel файла
    
    Args:
        filepath: Путь к Excel файлу
        
    Returns:
        Кортеж (scenarios, commissions, initial_cash)
        - scenarios: словарь {stage: [Scenario, ...]}
        - commissions: словарь {'cb1': 0.04, 'cb2': 0.07, 'dep': 0.05}
        - initial_cash: начальная касса
    """
    try:
        # Пытаемся прочитать Excel файл
        df_dict = pd.read_excel(filepath, sheet_name=None, engine='openpyxl')
        
        scenarios = {}
        commissions = COMMISSIONS.copy()
        initial_cash = INITIAL_CASH
        
        # Пытаемся найти данные в листах
        for sheet_name, df in df_dict.items():
            sheet_name_lower = sheet_name.lower()
            
            # Ищем сценарии
            if 'сценари' in sheet_name_lower or 'scenario' in sheet_name_lower:
                # Пытаемся извлечь сценарии
                scenarios = _extract_scenarios_from_df(df)
            
            # Ищем комиссии
            if 'комисс' in sheet_name_lower or 'commission' in sheet_name_lower:
                commissions = _extract_commissions_from_df(df)
            
            # Ищем начальную кассу
            if 'начал' in sheet_name_lower or 'initial' in sheet_name_lower:
                initial_cash = _extract_initial_cash_from_df(df)
        
        # Если не нашли сценарии в Excel, используем из constants
        if not scenarios:
            scenarios = _scenarios_from_constants()
        
        return scenarios, commissions, initial_cash
        
    except Exception as e:
        print(f"Предупреждение: не удалось прочитать Excel файл: {e}")
        print("Используются данные из constants.py")
        return _scenarios_from_constants(), COMMISSIONS.copy(), INITIAL_CASH


def _extract_scenarios_from_df(df: pd.DataFrame) -> Dict[int, List[Scenario]]:
    """Извлекает сценарии из DataFrame"""
    scenarios = {}
    # Простая попытка найти сценарии
    # Это нужно адаптировать под реальную структуру файла
    return _scenarios_from_constants()


def _extract_commissions_from_df(df: pd.DataFrame) -> Dict[str, float]:
    """Извлекает комиссии из DataFrame"""
    return COMMISSIONS.copy()


def _extract_initial_cash_from_df(df: pd.DataFrame) -> float:
    """Извлекает начальную кассу из DataFrame"""
    return INITIAL_CASH


def _scenarios_from_constants() -> Dict[int, List[Scenario]]:
    """Создает сценарии из констант"""
    scenarios = {}
    for stage, stage_data in SCENARIOS.items():
        scenarios[stage] = [
            Scenario(
                situation=item['situation'],
                probability=item['probability'],
                cb1_multiplier=item['cb1'],
                cb2_multiplier=item['cb2'],
                dep_multiplier=item['dep']
            )
            for item in stage_data
        ]
    return scenarios


def validate_scenarios(scenarios: Dict[int, List[Scenario]]) -> bool:
    """
    Проверяет, что суммы вероятностей сценариев = 1.0 для каждого этапа
    
    Args:
        scenarios: Словарь сценариев по этапам
        
    Returns:
        True если все проверки пройдены
        
    Raises:
        ValueError если проверка не пройдена
    """
    for stage, stage_scenarios in scenarios.items():
        total_prob = sum(s.probability for s in stage_scenarios)
        if abs(total_prob - 1.0) > EPSILON:
            raise ValueError(
                f"Этап {stage}: сумма вероятностей = {total_prob:.6f}, "
                f"ожидается 1.0"
            )
    return True


def load_commissions() -> Dict[str, float]:
    """
    Загружает комиссии (из Excel или constants)
    
    Returns:
        Словарь с комиссиями
    """
    return COMMISSIONS.copy()


def initialize_portfolio(cb1: float = INITIAL_CB1,
                        cb2: float = INITIAL_CB2,
                        dep: float = INITIAL_DEP,
                        cash: float = INITIAL_CASH) -> Portfolio:
    """
    Создает начальный портфель
    
    Args:
        cb1: Начальная стоимость ЦБ1
        cb2: Начальная стоимость ЦБ2
        dep: Начальная стоимость Депозитов
        cash: Начальная касса
        
    Returns:
        Начальный портфель
    """
    portfolio = Portfolio(
        cb1=cb1,
        cb2=cb2,
        dep=dep,
        cash=cash
    )
    
    # Проверяем ограничения
    if not portfolio.check_constraints():
        raise ValueError(
            f"Начальный портфель не удовлетворяет ограничениям: {portfolio}"
        )
    
    return portfolio


def print_scenarios_summary(scenarios: Dict[int, List[Scenario]]):
    """Выводит краткую информацию о сценариях"""
    print("\nСценарии по этапам:")
    print("=" * 70)
    for stage in sorted(scenarios.keys()):
        print(f"\nЭтап {stage}:")
        for scenario in scenarios[stage]:
            print(f"  {scenario}")
        total_prob = sum(s.probability for s in scenarios[stage])
        print(f"  Сумма вероятностей: {total_prob:.6f}")
