"""
Главный скрипт для решения задачи динамического программирования портфеля
"""
import time
import os
from data_loader import (
    load_from_excel, validate_scenarios, load_commissions, 
    initialize_portfolio, print_scenarios_summary
)
from solver import DynamicProgrammingSolver
from path_recovery import recover_path, monte_carlo_simulation, print_path_details
from visualization import (
    plot_portfolio_evolution, plot_expected_values, plot_actions,
    plot_portfolio_composition, create_images_directory
)
from constants import NUM_STAGES, USE_COMMISSIONS, USE_MIN_CONSTRAINTS, INITIAL_CASH


def main():
    """Главная функция"""
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 68 + "║")
    print("║" + "   ДИНАМИЧЕСКОЕ ПРОГРАММИРОВАНИЕ - ОПТИМИЗАЦИЯ ПОРТФЕЛЯ".center(68) + "║")
    print("║" + " " * 68 + "║")
    print("╚" + "=" * 68 + "╝")
    
    # 1. Загрузка данных
    print("\n[1/8] Загрузка данных...")
    excel_file = "Данные для постановки задачи.xlsx"
    
    if os.path.exists(excel_file):
        scenarios, commissions, initial_cash = load_from_excel(excel_file)
        print(f"  ✓ Данные загружены из {excel_file}")
    else:
        print(f"  ⚠ Файл {excel_file} не найден, используются данные из constants.py")
        from constants import SCENARIOS, COMMISSIONS, INITIAL_CASH
        scenarios = {}
        for stage, stage_data in SCENARIOS.items():
            from models import Scenario
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
        commissions = COMMISSIONS.copy()
        initial_cash = INITIAL_CASH
    
    # Валидация сценариев
    try:
        validate_scenarios(scenarios)
        print(f"  ✓ Загружено сценариев для {len(scenarios)} этапов")
        if USE_COMMISSIONS:
            print(f"  ✓ Комиссии ВКЛЮЧЕНЫ: ЦБ1={commissions['cb1']:.2%}, "
                  f"ЦБ2={commissions['cb2']:.2%}, Деп={commissions['dep']:.2%}")
        else:
            print(f"  ✓ Комиссии ВЫКЛЮЧЕНЫ")
    except ValueError as e:
        print(f"  ✗ Ошибка валидации: {e}")
        return
    
    # 2. Инициализация портфеля
    print("\n[2/8] Инициализация портфеля...")
    try:
        initial_portfolio = initialize_portfolio(cash=initial_cash)
        print(f"  ✓ Начальный портфель: {initial_portfolio}")
        print(f"  ✓ Полная стоимость: {initial_portfolio.total_value():.2f} д.е.")
        print(f"  ✓ Проверка ограничений: {initial_portfolio.check_constraints(USE_MIN_CONSTRAINTS)}")
    except ValueError as e:
        print(f"  ✗ Ошибка инициализации: {e}")
        return
    
    # 3. Создание решателя
    print("\n[3/8] Инициализация решателя DP...")
    print(f"  Настройки:")
    print(f"    - Комиссии: {'ВКЛЮЧЕНЫ' if USE_COMMISSIONS else 'ВЫКЛЮЧЕНЫ'}")
    print(f"    - Ограничения на минимум: {'ВКЛЮЧЕНЫ' if USE_MIN_CONSTRAINTS else 'ВЫКЛЮЧЕНЫ'}")
    solver = DynamicProgrammingSolver(
        initial_portfolio=initial_portfolio,
        scenarios=scenarios,
        commissions=commissions,
        criterion='bayesian',
        use_commissions=USE_COMMISSIONS,
        use_min_constraints=USE_MIN_CONSTRAINTS
    )
    print(f"  ✓ Решатель готов")
    
    # 4. Решение задачи
    print("\n[4/8] Решение задачи (обратное прохождение)...")
    start_time = time.time()
    try:
        solver.solve_backward()
        elapsed = time.time() - start_time
        print(f"  ✓ Решение найдено за {elapsed:.2f} сек")
    except Exception as e:
        print(f"  ✗ Ошибка при решении: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 5. Восстановление оптимального пути
    print("\n[5/8] Восстановление оптимальной траектории...")
    try:
        path, actions, total_value = recover_path(solver, initial_portfolio, scenarios)
        print(f"  ✓ Оптимальный путь найден")
        print(f"  ✓ Максимальный ожидаемый доход: {total_value:.2f} д.е.")
    except Exception as e:
        print(f"  ✗ Ошибка при восстановлении пути: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 6. Вывод результатов
    print("\n[6/8] Вывод результатов...")
    print("\n" + "=" * 70)
    print("ОПТИМАЛЬНЫЕ РЕШЕНИЯ ПО ЭТАПАМ")
    print("=" * 70)
    
    for t, action in enumerate(actions, 1):
        delta_cb1, delta_cb2, delta_dep = action
        print(f"\nЭтап {t}:")
        if action == (0.0, 0.0, 0.0):
            print("  Действие: Без изменений")
        else:
            print(f"  ЦБ1: {'+' if delta_cb1 > 0 else ''}{delta_cb1:.2f} д.е.")
            print(f"  ЦБ2: {'+' if delta_cb2 > 0 else ''}{delta_cb2:.2f} д.е.")
            print(f"  Деп: {'+' if delta_dep > 0 else ''}{delta_dep:.2f} д.е.")
    
    print("\n" + "=" * 70)
    print(f"МАКСИМАЛЬНЫЙ ОЖИДАЕМЫЙ ДОХОД: {total_value:.2f} д.е.")
    print("=" * 70)
    
    # Детальная информация о пути
    print_path_details(path, actions)
    
    # 7. Monte Carlo валидация
    print("\n[7/8] Monte Carlo валидация (1000 симуляций)...")
    try:
        mc_results = monte_carlo_simulation(
            solver, initial_portfolio, scenarios, 
            n_simulations=1000, random_seed=42
        )
        print(f"  ✓ Ожидание: {mc_results['mean']:.2f} д.е.")
        print(f"  ✓ Стд.откл: {mc_results['std']:.2f} д.е.")
        print(f"  ✓ Min: {mc_results['min']:.2f} д.е., Max: {mc_results['max']:.2f} д.е.")
    except Exception as e:
        print(f"  ⚠ Ошибка Monte Carlo: {e}")
    
    # 8. Визуализация
    print("\n[8/8] Визуализация результатов...")
    try:
        create_images_directory()
        plot_portfolio_evolution(path, "images/portfolio_evolution.png")
        plot_expected_values(solver, "images/expected_values.png")
        plot_actions(actions, "images/actions.png")
        plot_portfolio_composition(path, "images/portfolio_composition.png")
        print(f"  ✓ Графики сохранены в директории images/")
    except Exception as e:
        print(f"  ⚠ Ошибка визуализации: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 70)
    print("✅ ГОТОВО! Решение найдено и результаты сохранены")
    print("=" * 70)
    
    # Краткая сводка
    print("\nКРАТКАЯ СВОДКА:")
    print(f"  Начальная стоимость: {initial_portfolio.total_value():.2f} д.е.")
    print(f"  Ожидаемая конечная стоимость: {total_value:.2f} д.е.")
    print(f"  Ожидаемый доход: {total_value - initial_portfolio.total_value():.2f} д.е.")
    print(f"  Относительный доход: {(total_value / initial_portfolio.total_value() - 1) * 100:.2f}%")
    
    # Предложение генерации полного отчета
    print("\n" + "=" * 70)
    print("💡 Для генерации полного отчета с расширенной визуализацией выполните:")
    print("   python generate_report.py")
    print("=" * 70)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '--full-report':
        # Запуск генерации полного отчета
        from generate_report import main as generate_full_report
        generate_full_report()
    else:
        main()
