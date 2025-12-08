"""
Визуализация результатов решения задачи динамического программирования
"""
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict
from models import Portfolio
from solver import DynamicProgrammingSolver
from constants import NUM_STAGES
import os


def plot_portfolio_evolution(path: List[Portfolio], 
                           save_path: str = "portfolio_evolution.png"):
    """
    Строит график эволюции портфеля по этапам
    
    Args:
        path: Список портфелей по этапам
        save_path: Путь для сохранения графика
    """
    # Извлекаем данные
    stages = []
    cb1_values = []
    cb2_values = []
    dep_values = []
    cash_values = []
    total_values = []
    
    for i, portfolio in enumerate(path):
        stages.append(i)
        cb1_values.append(portfolio.cb1)
        cb2_values.append(portfolio.cb2)
        dep_values.append(portfolio.dep)
        cash_values.append(portfolio.cash)
        total_values.append(portfolio.total_value())
    
    # Создаем график
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # График 1: Компоненты портфеля
    ax1.plot(stages, cb1_values, 'o-', label='ЦБ1', linewidth=2, markersize=6)
    ax1.plot(stages, cb2_values, 's-', label='ЦБ2', linewidth=2, markersize=6)
    ax1.plot(stages, dep_values, '^-', label='Депозиты', linewidth=2, markersize=6)
    ax1.plot(stages, cash_values, 'd-', label='Касса', linewidth=2, markersize=6)
    
    ax1.set_xlabel('Этап', fontsize=12)
    ax1.set_ylabel('Стоимость (д.е.)', fontsize=12)
    ax1.set_title('Эволюция компонентов портфеля', fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(stages)
    
    # График 2: Общая стоимость
    ax2.plot(stages, total_values, 'o-', color='red', linewidth=3, markersize=8)
    ax2.set_xlabel('Этап', fontsize=12)
    ax2.set_ylabel('Общая стоимость (д.е.)', fontsize=12)
    ax2.set_title('Эволюция общей стоимости портфеля', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(stages)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  График сохранен: {save_path}")
    plt.close()


def plot_expected_values(solver: DynamicProgrammingSolver,
                        save_path: str = "expected_values.png"):
    """
    Строит график функции ценности по этапам
    
    Args:
        solver: Решатель с решенной задачей
        save_path: Путь для сохранения графика
    """
    fig, axes = plt.subplots(1, NUM_STAGES, figsize=(15, 5))
    if NUM_STAGES == 1:
        axes = [axes]
    
    for stage in range(1, NUM_STAGES + 1):
        ax = axes[stage - 1]
        
        if stage in solver.value_function:
            values = list(solver.value_function[stage].values())
            states_count = len(values)
            
            # Сортируем значения для лучшей визуализации
            sorted_values = sorted(values)
            
            ax.plot(range(len(sorted_values)), sorted_values, 'o-', 
                   linewidth=2, markersize=4)
            ax.set_xlabel('Состояние (индекс)', fontsize=10)
            ax.set_ylabel('Ожидаемая ценность (д.е.)', fontsize=10)
            ax.set_title(f'Функция ценности J_{stage}(s)', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.text(0.05, 0.95, f'Состояний: {states_count}', 
                   transform=ax.transAxes, fontsize=9,
                   verticalalignment='top', bbox=dict(boxstyle='round', 
                   facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  График сохранен: {save_path}")
    plt.close()


def plot_actions(actions: List[tuple], 
                save_path: str = "actions.png"):
    """
    Визуализирует действия по этапам
    
    Args:
        actions: Список действий (delta_cb1, delta_cb2, delta_dep)
        save_path: Путь для сохранения графика
    """
    stages = list(range(1, len(actions) + 1))
    delta_cb1 = [a[0] for a in actions]
    delta_cb2 = [a[1] for a in actions]
    delta_dep = [a[2] for a in actions]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(stages))
    width = 0.25
    
    ax.bar(x - width, delta_cb1, width, label='Δ ЦБ1', alpha=0.8)
    ax.bar(x, delta_cb2, width, label='Δ ЦБ2', alpha=0.8)
    ax.bar(x + width, delta_dep, width, label='Δ Депозиты', alpha=0.8)
    
    ax.set_xlabel('Этап', fontsize=12)
    ax.set_ylabel('Изменение стоимости (д.е.)', fontsize=12)
    ax.set_title('Оптимальные действия по этапам', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(stages)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  График сохранен: {save_path}")
    plt.close()


def plot_portfolio_composition(path: List[Portfolio],
                              save_path: str = "portfolio_composition.png"):
    """
    Строит круговые диаграммы состава портфеля по этапам
    
    Args:
        path: Список портфелей по этапам
        save_path: Путь для сохранения графика
    """
    # Выбираем ключевые этапы для визуализации
    key_stages = [0, len(path) // 2, len(path) - 1]
    key_stages = [i for i in key_stages if i < len(path)]
    
    fig, axes = plt.subplots(1, len(key_stages), figsize=(5 * len(key_stages), 5))
    if len(key_stages) == 1:
        axes = [axes]
    
    for idx, stage_idx in enumerate(key_stages):
        portfolio = path[stage_idx]
        ax = axes[idx]
        
        # Исключаем кассу из круговой диаграммы (или включаем, если нужно)
        values = [portfolio.cb1, portfolio.cb2, portfolio.dep, portfolio.cash]
        labels = ['ЦБ1', 'ЦБ2', 'Депозиты', 'Касса']
        colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
        
        # Фильтруем нулевые значения
        filtered_data = [(v, l, c) for v, l, c in zip(values, labels, colors) if v > 1e-6]
        if filtered_data:
            values, labels, colors = zip(*filtered_data)
            
            ax.pie(values, labels=labels, colors=colors, autopct='%1.1f%%',
                  startangle=90, textprops={'fontsize': 10})
            ax.set_title(f'Этап {stage_idx}\nОбщая стоимость: {portfolio.total_value():.2f} д.е.',
                        fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  График сохранен: {save_path}")
    plt.close()


def create_images_directory():
    """Создает директорию для изображений, если её нет"""
    if not os.path.exists('images'):
        os.makedirs('images')
        print("  Создана директория images/")
