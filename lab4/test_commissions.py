"""
Тесты для проверки расчета комиссий и ограничений
"""
from models import Portfolio
from constants import COMMISSIONS, MIN_CB1, MIN_CB2, MIN_DEP


def test_commission_on_buy():
    """Проверяет комиссию при покупке"""
    commission = COMMISSIONS['cb1']
    buy_amount = 50.0
    
    # При покупке платим больше из-за комиссии
    effective_cost = buy_amount * (1 + commission)
    expected = 50.0 * 1.04  # 52.0
    
    assert abs(effective_cost - expected) < 1e-6, \
        f"Ожидалось {expected}, получено {effective_cost}"
    print("✓ Комиссия при покупке рассчитана правильно")


def test_commission_on_sell():
    """Проверяет комиссию при продаже"""
    commission = COMMISSIONS['cb1']
    sell_amount = 50.0
    
    # При продаже получаем меньше из-за комиссии
    received = sell_amount * (1 - commission)
    expected = 50.0 * 0.96  # 48.0
    
    assert abs(received - expected) < 1e-6, \
        f"Ожидалось {expected}, получено {received}"
    print("✓ Комиссия при продаже рассчитана правильно")


def test_portfolio_constraints():
    """Проверяет ограничения на минимум"""
    # Портфель в пределах ограничений
    portfolio = Portfolio(cb1=100.0, cb2=800.0, dep=400.0, cash=50.0)
    assert portfolio.check_constraints(), \
        "Портфель должен удовлетворять ограничениям"
    
    # Портфель с нарушением ограничений
    portfolio_bad = Portfolio(cb1=20.0, cb2=800.0, dep=400.0, cash=50.0)
    assert not portfolio_bad.check_constraints(), \
        "Портфель не должен удовлетворять ограничениям (ЦБ1 < MIN)"
    
    print("✓ Ограничения на минимум проверяются правильно")


def test_action_with_commissions():
    """Проверяет применение действия с комиссиями"""
    portfolio = Portfolio(cb1=100.0, cb2=800.0, dep=400.0, cash=100.0)
    action = (25.0, 0.0, 0.0)  # Покупаем ЦБ1 на 25 д.е.
    
    new_portfolio = portfolio.apply_action(action, COMMISSIONS)
    
    # Проверяем новую стоимость ЦБ1
    assert abs(new_portfolio.cb1 - 125.0) < 1e-6, \
        f"Ожидалось cb1=125.0, получено {new_portfolio.cb1}"
    
    # Проверяем новую кассу (100 - 25*1.04 = 100 - 26 = 74)
    expected_cash = 100.0 - 25.0 * 1.04
    assert abs(new_portfolio.cash - expected_cash) < 1e-6, \
        f"Ожидалось cash={expected_cash}, получено {new_portfolio.cash}"
    
    print("✓ Действие с комиссиями применяется правильно")


def test_sell_action_with_commissions():
    """Проверяет продажу с комиссиями"""
    portfolio = Portfolio(cb1=100.0, cb2=800.0, dep=400.0, cash=0.0)
    action = (-25.0, 0.0, 0.0)  # Продаем ЦБ1 на 25 д.е.
    
    new_portfolio = portfolio.apply_action(action, COMMISSIONS)
    
    # Проверяем новую стоимость ЦБ1
    assert abs(new_portfolio.cb1 - 75.0) < 1e-6, \
        f"Ожидалось cb1=75.0, получено {new_portfolio.cb1}"
    
    # При продаже получаем меньше из-за комиссии
    # received = 25 * (1 - 0.04) = 24
    # cost = -25 * (1 - 0.04) = -24
    # new_cash = 0 - (-24) = 24
    expected_cash = 25.0 * (1 - COMMISSIONS['cb1'])
    assert abs(new_portfolio.cash - expected_cash) < 1e-6, \
        f"Ожидалось cash={expected_cash}, получено {new_portfolio.cash}"
    
    print("✓ Продажа с комиссиями работает правильно")


def test_can_apply_action():
    """Проверяет метод can_apply_action"""
    portfolio = Portfolio(cb1=100.0, cb2=800.0, dep=400.0, cash=100.0)
    
    # Действие допустимо
    action1 = (25.0, 0.0, 0.0)  # Покупка на 25 д.е. (стоит 26 с комиссией)
    assert portfolio.can_apply_action(action1, COMMISSIONS), \
        "Действие должно быть допустимо"
    
    # Действие недопустимо (недостаточно кассы)
    action2 = (100.0, 0.0, 0.0)  # Покупка на 100 д.е. (стоит 104, но касса только 100)
    assert not portfolio.can_apply_action(action2, COMMISSIONS), \
        "Действие не должно быть допустимо (недостаточно кассы)"
    
    # Действие недопустимо (нарушение минимума)
    portfolio_low = Portfolio(cb1=35.0, cb2=800.0, dep=400.0, cash=100.0)
    action3 = (-10.0, 0.0, 0.0)  # Продажа 10 д.е. приведет к cb1=25 < MIN_CB1=30
    assert not portfolio_low.can_apply_action(action3, COMMISSIONS), \
        "Действие не должно быть допустимо (нарушение минимума)"
    
    print("✓ Метод can_apply_action работает правильно")


def test_multiple_actions():
    """Проверяет действие с несколькими активами"""
    portfolio = Portfolio(cb1=100.0, cb2=800.0, dep=400.0, cash=200.0)
    
    # Покупаем ЦБ1 на 25 и ЦБ2 на 200
    action = (25.0, 200.0, 0.0)
    
    # Стоимость: 25*1.04 + 200*1.07 = 26 + 214 = 240
    # Касса: 200, недостаточно!
    assert not portfolio.can_apply_action(action, COMMISSIONS), \
        "Действие не должно быть допустимо (недостаточно кассы для всех покупок)"
    
    # Уменьшаем покупку ЦБ2
    action2 = (25.0, 100.0, 0.0)
    # Стоимость: 25*1.04 + 100*1.07 = 26 + 107 = 133
    # Касса: 200, достаточно
    assert portfolio.can_apply_action(action2, COMMISSIONS), \
        "Действие должно быть допустимо"
    
    new_portfolio = portfolio.apply_action(action2, COMMISSIONS)
    expected_cash = 200.0 - (25.0 * 1.04 + 100.0 * 1.07)
    assert abs(new_portfolio.cash - expected_cash) < 1e-6, \
        f"Ожидалось cash={expected_cash}, получено {new_portfolio.cash}"
    
    print("✓ Действия с несколькими активами работают правильно")


def run_all_tests():
    """Запускает все тесты"""
    print("=" * 70)
    print("ЗАПУСК ТЕСТОВ")
    print("=" * 70)
    
    try:
        test_commission_on_buy()
        test_commission_on_sell()
        test_portfolio_constraints()
        test_action_with_commissions()
        test_sell_action_with_commissions()
        test_can_apply_action()
        test_multiple_actions()
        
        print("\n" + "=" * 70)
        print("✓ ВСЕ ТЕСТЫ ПРОЙДЕНЫ УСПЕШНО")
        print("=" * 70)
        return True
    except AssertionError as e:
        print(f"\n✗ ТЕСТ НЕ ПРОЙДЕН: {e}")
        return False
    except Exception as e:
        print(f"\n✗ ОШИБКА В ТЕСТЕ: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    run_all_tests()
