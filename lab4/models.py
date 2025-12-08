"""
Модели данных для задачи динамического программирования портфеля
"""
from dataclasses import dataclass
from typing import Dict, Tuple
from constants import MIN_CB1, MIN_CB2, MIN_DEP


@dataclass(frozen=True)
class Portfolio:
    """
    Портфель инвестора - в ДЕНЕЖНОМ ВЫРАЖЕНИИ
    
    frozen=True позволяет использовать Portfolio как ключ в словаре
    """
    cb1: float   # Стоимость ЦБ1 в д.е.
    cb2: float   # Стоимость ЦБ2 в д.е.
    dep: float   # Стоимость Депозитов в д.е.
    cash: float  # Свободные средства в д.е.
    
    def total_value(self) -> float:
        """Полная стоимость портфеля"""
        return self.cb1 + self.cb2 + self.dep + self.cash
    
    def check_constraints(self, 
                         use_min_constraints: bool = False,
                         min_cb1: float = MIN_CB1, 
                         min_cb2: float = MIN_CB2, 
                         min_dep: float = MIN_DEP) -> bool:
        """
        Проверяет ограничения на минимум и неотрицательность кассы
        
        Args:
            use_min_constraints: Флаг использования ограничений на минимум
            min_cb1: Минимальная стоимость ЦБ1
            min_cb2: Минимальная стоимость ЦБ2
            min_dep: Минимальная стоимость Депозитов
            
        Returns:
            True если все ограничения выполнены
        """
        result = self.cash >= 0
        
        if use_min_constraints:
            result = result and (self.cb1 >= min_cb1 and 
                                self.cb2 >= min_cb2 and 
                                self.dep >= min_dep)
        
        return result
    
    def _calculate_effective_cost(self, action: Tuple[float, float, float], 
                                 commissions: Dict[str, float],
                                 use_commissions: bool = False) -> float:
        """
        Вычисляет эффективную стоимость действия с учетом комиссий
        
        При покупке (delta > 0): cost = delta * (1 + commission) если use_commissions=True, иначе cost = delta
        При продаже (delta < 0): received = |delta| * (1 - commission) если use_commissions=True, иначе received = |delta|
        
        Args:
            action: Кортеж (delta_cb1, delta_cb2, delta_dep)
            commissions: Словарь с комиссиями {'cb1': 0.04, 'cb2': 0.07, 'dep': 0.05}
            use_commissions: Флаг использования комиссий
            
        Returns:
            Эффективная стоимость (положительная для покупки, отрицательная для продажи)
        """
        delta_cb1, delta_cb2, delta_dep = action
        cost = 0.0
        
        if use_commissions:
            # ЦБ1
            if delta_cb1 > 0:  # Покупка
                cost += delta_cb1 * (1 + commissions['cb1'])
            elif delta_cb1 < 0:  # Продажа
                cost += delta_cb1 * (1 - commissions['cb1'])
            
            # ЦБ2
            if delta_cb2 > 0:
                cost += delta_cb2 * (1 + commissions['cb2'])
            elif delta_cb2 < 0:
                cost += delta_cb2 * (1 - commissions['cb2'])
            
            # Депозиты
            if delta_dep > 0:
                cost += delta_dep * (1 + commissions['dep'])
            elif delta_dep < 0:
                cost += delta_dep * (1 - commissions['dep'])
        else:
            # Без комиссий: эффективная стоимость = просто сумма дельт
            cost = delta_cb1 + delta_cb2 + delta_dep
        
        return cost
    
    def can_apply_action(self, action: Tuple[float, float, float], 
                        commissions: Dict[str, float],
                        use_commissions: bool = False,
                        use_min_constraints: bool = False,
                        min_cb1: float = MIN_CB1,
                        min_cb2: float = MIN_CB2,
                        min_dep: float = MIN_DEP) -> bool:
        """
        Проверяет допустимость действия с учетом комиссий и ограничений
        
        Args:
            action: Кортеж (delta_cb1, delta_cb2, delta_dep)
            commissions: Словарь с комиссиями
            use_commissions: Флаг использования комиссий
            use_min_constraints: Флаг использования ограничений на минимум
            min_cb1: Минимальная стоимость ЦБ1
            min_cb2: Минимальная стоимость ЦБ2
            min_dep: Минимальная стоимость Депозитов
            
        Returns:
            True если действие допустимо
        """
        delta_cb1, delta_cb2, delta_dep = action
        
        # Вычисляем эффективную стоимость (с комиссиями или без)
        effective_cost = self._calculate_effective_cost(action, commissions, use_commissions)
        
        # Проверяем кассовое ограничение
        # Если cost > 0, нужны деньги на покупку
        # Если cost < 0, получаем деньги от продажи
        if effective_cost > self.cash:
            return False
        
        # Проверяем новые позиции после действия
        new_cb1 = self.cb1 + delta_cb1
        new_cb2 = self.cb2 + delta_cb2
        new_dep = self.dep + delta_dep
        
        # Проверяем минимумы (только если включены)
        if use_min_constraints:
            if new_cb1 < min_cb1 or new_cb2 < min_cb2 or new_dep < min_dep:
                return False
        
        # Проверяем неотрицательность
        if new_cb1 < 0 or new_cb2 < 0 or new_dep < 0:
            return False
        
        # Проверяем, что касса после действия не отрицательна
        new_cash = self.cash - effective_cost
        if new_cash < 0:
            return False
        
        return True
    
    def apply_action(self, action: Tuple[float, float, float], 
                    commissions: Dict[str, float],
                    use_commissions: bool = False) -> 'Portfolio':
        """
        Применяет действие с учетом комиссий (или без них)
        
        Args:
            action: Кортеж (delta_cb1, delta_cb2, delta_dep)
            commissions: Словарь с комиссиями
            use_commissions: Флаг использования комиссий
            
        Returns:
            Новый портфель после применения действия
        """
        delta_cb1, delta_cb2, delta_dep = action
        
        # Вычисляем эффективную стоимость (с комиссиями или без)
        effective_cost = self._calculate_effective_cost(action, commissions, use_commissions)
        
        # Создаем новый портфель
        return Portfolio(
            cb1=self.cb1 + delta_cb1,
            cb2=self.cb2 + delta_cb2,
            dep=self.dep + delta_dep,
            cash=self.cash - effective_cost
        )
    
    def apply_scenario(self, scenario: 'Scenario') -> 'Portfolio':
        """
        Применяет сценарий к портфелю (изменяет стоимость активов)
        
        Args:
            scenario: Сценарий с мультипликаторами
            
        Returns:
            Новый портфель после применения сценария
        """
        return Portfolio(
            cb1=self.cb1 * scenario.cb1_multiplier,
            cb2=self.cb2 * scenario.cb2_multiplier,
            dep=self.dep * scenario.dep_multiplier,
            cash=self.cash  # Касса не меняется сценарием
        )
    
    def __str__(self) -> str:
        return (f"Portfolio(cb1={self.cb1:.2f}, cb2={self.cb2:.2f}, "
                f"dep={self.dep:.2f}, cash={self.cash:.2f}, "
                f"total={self.total_value():.2f})")


@dataclass
class Scenario:
    """Один сценарий развития на этапе"""
    situation: str
    probability: float
    cb1_multiplier: float
    cb2_multiplier: float
    dep_multiplier: float
    
    def __str__(self) -> str:
        return (f"Scenario({self.situation}, p={self.probability:.2f}, "
                f"cb1×{self.cb1_multiplier:.2f}, cb2×{self.cb2_multiplier:.2f}, "
                f"dep×{self.dep_multiplier:.2f})")


@dataclass
class Decision:
    """Оптимальное решение для состояния"""
    portfolio: Portfolio
    stage: int
    action: Tuple[float, float, float]
    expected_value: float
    details: Dict = None
    
    def __post_init__(self):
        if self.details is None:
            self.details = {}
    
    def __str__(self) -> str:
        delta_cb1, delta_cb2, delta_dep = self.action
        return (f"Decision(stage={self.stage}, "
                f"action=({delta_cb1:.2f}, {delta_cb2:.2f}, {delta_dep:.2f}), "
                f"value={self.expected_value:.2f})")
