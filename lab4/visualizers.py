"""
Расширенный модуль визуализации для задачи динамического программирования портфеля
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path
import logging
from models import Portfolio, Scenario
from solver import DynamicProgrammingSolver
from constants import (
    INITIAL_CB1, INITIAL_CB2, INITIAL_DEP, INITIAL_CASH,
    COMMISSIONS, MIN_CB1, MIN_CB2, MIN_DEP,
    NUM_STAGES, USE_COMMISSIONS, USE_MIN_CONSTRAINTS
)

# Опциональный импорт scipy
try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

logger = logging.getLogger(__name__)

# Настройки matplotlib для русского языка
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['figure.titlesize'] = 16

# Цветовая палитра (дружественная к цветовой слепоте)
COLORS = {
    'cb1': '#1f77b4',      # синий
    'cb2': '#ff7f0e',      # оранжевый
    'dep': '#2ca02c',      # зеленый
    'cash': '#9467bd',     # фиолетовый
    'positive': '#2ca02c', # зеленый (покупка)
    'negative': '#d62728', # красный (продажа)
    'neutral': '#7f7f7f',  # серый (без изменений)
    'total': '#e377c2'     # розовый
}


def visualize_initial_portfolio(initial_portfolio: Portfolio,
                               commissions: Dict[str, float],
                               output_dir: str = "output/diagrams") -> str:
    """
    Создает круговую диаграмму начального портфеля
    
    Args:
        initial_portfolio: Начальный портфель
        commissions: Словарь комиссий
        output_dir: Директория для сохранения
        
    Returns:
        Путь к сохраненному файлу
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    filepath = Path(output_dir) / "01_initial_portfolio.png"
    
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # График 1: Круговая диаграмма
        values = [initial_portfolio.cb1, initial_portfolio.cb2, 
                 initial_portfolio.dep, initial_portfolio.cash]
        labels = ['ЦБ1', 'ЦБ2', 'Депозиты', 'Касса']
        colors_list = [COLORS['cb1'], COLORS['cb2'], COLORS['dep'], COLORS['cash']]
        
        # Фильтруем нулевые значения
        filtered_data = [(v, l, c) for v, l, c in zip(values, labels, colors_list) if v > 1e-6]
        if filtered_data:
            values, labels, colors_list = zip(*filtered_data)
            
            wedges, texts, autotexts = ax1.pie(
                values, labels=labels, colors=colors_list,
                autopct='%1.1f%%', startangle=90,
                textprops={'fontsize': 12, 'fontweight': 'bold'}
            )
            
            # Улучшаем читаемость процентов
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
        
        ax1.set_title('Начальное состояние портфеля', fontsize=16, fontweight='bold', pad=20)
        
        # График 2: Таблица параметров
        ax2.axis('off')
        table_data = [
            ['Параметр', 'Значение'],
            ['Начальный капитал', f'{initial_portfolio.total_value():.2f} д.е.'],
            ['ЦБ1', f'{initial_portfolio.cb1:.2f} д.е.'],
            ['ЦБ2', f'{initial_portfolio.cb2:.2f} д.е.'],
            ['Депозиты', f'{initial_portfolio.dep:.2f} д.е.'],
            ['Касса', f'{initial_portfolio.cash:.2f} д.е.'],
            ['', ''],
            ['Комиссия ЦБ1', f'{commissions["cb1"]:.1%}' if USE_COMMISSIONS else 'Выключена'],
            ['Комиссия ЦБ2', f'{commissions["cb2"]:.1%}' if USE_COMMISSIONS else 'Выключена'],
            ['Комиссия Деп', f'{commissions["dep"]:.1%}' if USE_COMMISSIONS else 'Выключена'],
            ['', ''],
            ['Минимум ЦБ1', f'{MIN_CB1:.2f} д.е.' if USE_MIN_CONSTRAINTS else 'Выключен'],
            ['Минимум ЦБ2', f'{MIN_CB2:.2f} д.е.' if USE_MIN_CONSTRAINTS else 'Выключен'],
            ['Минимум Деп', f'{MIN_DEP:.2f} д.е.' if USE_MIN_CONSTRAINTS else 'Выключен'],
        ]
        
        table = ax2.table(cellText=table_data, cellLoc='left', loc='center',
                          colWidths=[0.4, 0.6])
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 2)
        
        # Стилизация таблицы
        for i in range(len(table_data)):
            if i == 0:  # Заголовок
                for j in range(2):
                    table[(i, j)].set_facecolor('#4CAF50')
                    table[(i, j)].set_text_props(weight='bold', color='white')
            else:
                for j in range(2):
                    table[(i, j)].set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')
        
        ax2.set_title('Параметры задачи', fontsize=16, fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"График начального портфеля сохранен: {filepath}")
        return str(filepath)
    except Exception as e:
        logger.error(f"Ошибка при создании графика начального портфеля: {e}")
        raise


def visualize_scenarios(scenarios: Dict[int, List[Scenario]],
                       output_dir: str = "output/diagrams") -> Tuple[str, str]:
    """
    Создает визуализацию сценариев: тепловую карту и дерево сценариев
    
    Args:
        scenarios: Словарь сценариев по этапам
        output_dir: Директория для сохранения
        
    Returns:
        Кортеж путей к сохраненным файлам (heatmap, tree)
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    filepath_heatmap = Path(output_dir) / "02_scenarios_matrix.png"
    filepath_tree = Path(output_dir) / "02_scenario_tree.png"
    
    try:
        # График 1: Тепловая карта вероятностей
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Подготовка данных для heatmap
        stages = sorted(scenarios.keys())
        situation_names = ['Благоприятная', 'Нейтральная', 'Негативная']
        
        prob_matrix = []
        for stage in stages:
            stage_probs = []
            for situation in situation_names:
                # Находим вероятность для данной ситуации
                prob = 0.0
                for scenario in scenarios[stage]:
                    if scenario.situation.lower() in situation.lower() or \
                       situation.lower() in scenario.situation.lower():
                        prob = scenario.probability
                        break
                stage_probs.append(prob)
            prob_matrix.append(stage_probs)
        
        prob_matrix = np.array(prob_matrix)
        
        # Создаем heatmap
        im = ax.imshow(prob_matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
        
        # Устанавливаем метки
        ax.set_xticks(np.arange(len(situation_names)))
        ax.set_yticks(np.arange(len(stages)))
        ax.set_xticklabels(situation_names)
        ax.set_yticklabels([f'Этап {s}' for s in stages])
        
        # Добавляем значения в ячейки
        for i in range(len(stages)):
            for j in range(len(situation_names)):
                text = ax.text(j, i, f'{prob_matrix[i, j]:.2f}',
                             ha="center", va="center", color="black", fontweight='bold')
        
        ax.set_title('Матрица вероятностей сценариев', fontsize=16, fontweight='bold', pad=20)
        plt.colorbar(im, ax=ax, label='Вероятность')
        plt.tight_layout()
        plt.savefig(filepath_heatmap, dpi=150, bbox_inches='tight')
        plt.close()
        
        # График 2: Дерево сценариев (упрощенное представление)
        fig, ax = plt.subplots(figsize=(14, 10))
        ax.axis('off')
        
        # Позиционирование узлов
        y_positions = {1: 0.8, 2: 0.5, 3: 0.2}
        x_base = 0.1
        x_spacing = 0.25
        
        # Рисуем дерево
        for stage in stages:
            y = y_positions[stage]
            x_start = x_base
            
            # Узел этапа
            circle = plt.Circle((x_start, y), 0.03, color=COLORS['cb1'], zorder=3)
            ax.add_patch(circle)
            ax.text(x_start, y + 0.05, f'Этап {stage}', ha='center', fontsize=12, fontweight='bold')
            
            # Сценарии
            for i, scenario in enumerate(scenarios[stage]):
                x_scenario = x_base + (i + 1) * x_spacing
                
                # Линия от этапа к сценарию
                ax.plot([x_start + 0.03, x_scenario - 0.02], [y, y], 
                       'k-', linewidth=2, alpha=0.5)
                
                # Узел сценария
                circle = plt.Circle((x_scenario, y), 0.02, color=COLORS['dep'], zorder=3)
                ax.add_patch(circle)
                
                # Подпись сценария
                label = f"{scenario.situation[:4]}\np={scenario.probability:.2f}"
                ax.text(x_scenario, y - 0.05, label, ha='center', fontsize=9,
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title('Дерево сценариев', fontsize=16, fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.savefig(filepath_tree, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Графики сценариев сохранены: {filepath_heatmap}, {filepath_tree}")
        return str(filepath_heatmap), str(filepath_tree)
    except Exception as e:
        logger.error(f"Ошибка при создании графиков сценариев: {e}")
        raise


def visualize_dp_process(solver: DynamicProgrammingSolver,
                        timing_data: Dict[int, float] = None,
                        output_dir: str = "output/diagrams") -> str:
    """
    Создает визуализацию процесса динамического программирования
    
    Args:
        solver: Решатель с решенной задачей
        timing_data: Словарь времени выполнения по этапам {stage: time_sec}
        output_dir: Директория для сохранения
        
    Returns:
        Путь к сохраненному файлу
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    filepath = Path(output_dir) / "03_dp_process.png"
    
    try:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # График 1: Количество состояний по этапам
        stages = sorted(solver.value_function.keys())
        num_states = [len(solver.value_function[stage]) for stage in stages]
        
        ax1.bar(stages, num_states, color=COLORS['cb1'], alpha=0.7, edgecolor='black', linewidth=1.5)
        ax1.set_xlabel('Этап', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Количество состояний', fontsize=12, fontweight='bold')
        ax1.set_title('Количество состояний на каждом этапе', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.set_xticks(stages)
        
        # Добавляем значения на столбцы
        for stage, count in zip(stages, num_states):
            ax1.text(stage, count, str(count), ha='center', va='bottom', fontweight='bold')
        
        # График 2: Лучшие значения функции ценности по этапам
        best_values = []
        for stage in stages:
            if solver.value_function[stage]:
                best_value = max(solver.value_function[stage].values())
                best_values.append(best_value)
            else:
                best_values.append(0)
        
        ax2.plot(stages, best_values, 'o-', color=COLORS['positive'], 
                linewidth=3, markersize=10, markerfacecolor='white', markeredgewidth=2)
        ax2.set_xlabel('Этап', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Лучшее значение функции ценности (д.е.)', fontsize=12, fontweight='bold')
        ax2.set_title('Эволюция оптимального решения', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.set_xticks(stages)
        
        # График 3: Время выполнения (если доступно)
        if timing_data:
            times = [timing_data.get(stage, 0) for stage in stages]
            ax3.bar(stages, times, color=COLORS['cb2'], alpha=0.7, edgecolor='black', linewidth=1.5)
            ax3.set_xlabel('Этап', fontsize=12, fontweight='bold')
            ax3.set_ylabel('Время выполнения (сек)', fontsize=12, fontweight='bold')
            ax3.set_title('Время выполнения по этапам', fontsize=14, fontweight='bold')
            ax3.grid(True, alpha=0.3, axis='y')
            ax3.set_xticks(stages)
        else:
            ax3.text(0.5, 0.5, 'Данные о времени\nнедоступны', 
                    ha='center', va='center', fontsize=14, transform=ax3.transAxes)
            ax3.axis('off')
        
        # График 4: Таблица статистики
        ax4.axis('off')
        table_data = [['Этап', 'Состояний', 'Время (сек)', 'Лучшее значение']]
        for stage in stages:
            num = num_states[stages.index(stage)]
            time_val = timing_data.get(stage, 0) if timing_data else 0
            best_val = best_values[stages.index(stage)]
            table_data.append([
                str(stage),
                str(num),
                f'{time_val:.2f}' if time_val > 0 else 'N/A',
                f'{best_val:.2f}'
            ])
        
        table = ax4.table(cellText=table_data, cellLoc='center', loc='center',
                         colWidths=[0.2, 0.25, 0.25, 0.3])
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 2)
        
        # Стилизация таблицы
        for i in range(len(table_data)):
            for j in range(4):
                if i == 0:  # Заголовок
                    table[(i, j)].set_facecolor('#4CAF50')
                    table[(i, j)].set_text_props(weight='bold', color='white')
                else:
                    table[(i, j)].set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')
        
        ax4.set_title('Статистика процесса DP', fontsize=14, fontweight='bold', pad=20)
        
        plt.suptitle('Процесс динамического программирования - Обратное прохождение', 
                    fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"График процесса DP сохранен: {filepath}")
        return str(filepath)
    except Exception as e:
        logger.error(f"Ошибка при создании графика процесса DP: {e}")
        raise


def visualize_optimal_actions(actions: List[Tuple[float, float, float]],
                              expected_values: List[float] = None,
                              output_dir: str = "output/diagrams") -> str:
    """
    Создает визуализацию оптимальных действий на каждом этапе
    
    Args:
        actions: Список действий по этапам
        expected_values: Ожидаемые значения для каждого этапа (опционально)
        output_dir: Директория для сохранения
        
    Returns:
        Путь к сохраненному файлу
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    filepath = Path(output_dir) / "04_optimal_actions.png"
    
    try:
        num_stages = len(actions)
        fig, axes = plt.subplots(1, num_stages, figsize=(6 * num_stages, 8))
        if num_stages == 1:
            axes = [axes]
        
        stages = list(range(1, num_stages + 1))
        
        for idx, (stage, action) in enumerate(zip(stages, actions)):
            ax = axes[idx]
            delta_cb1, delta_cb2, delta_dep = action
            
            # Подготовка данных для столбчатой диаграммы
            assets = ['ЦБ1', 'ЦБ2', 'Деп']
            deltas = [delta_cb1, delta_cb2, delta_dep]
            colors = []
            
            for delta in deltas:
                if delta > 1e-6:
                    colors.append(COLORS['positive'])
                elif delta < -1e-6:
                    colors.append(COLORS['negative'])
                else:
                    colors.append(COLORS['neutral'])
            
            # Создаем столбчатую диаграмму
            bars = ax.bar(assets, deltas, color=colors, alpha=0.7, 
                         edgecolor='black', linewidth=1.5)
            
            # Добавляем значения на столбцы
            for bar, delta in zip(bars, deltas):
                height = bar.get_height()
                label = f'{delta:+.1f}'
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       label, ha='center', va='bottom' if height >= 0 else 'top',
                       fontweight='bold', fontsize=11)
            
            ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
            ax.set_ylabel('Изменение стоимости (д.е.)', fontsize=11, fontweight='bold')
            ax.set_title(f'Этап {stage}', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            
            # Добавляем ожидаемое значение если доступно
            if expected_values and idx < len(expected_values):
                ax.text(0.5, 0.95, f'Ожидаемый доход:\n{expected_values[idx]:.2f} д.е.',
                       transform=ax.transAxes, ha='center', va='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                       fontsize=10, fontweight='bold')
        
        # Легенда
        legend_elements = [
            mpatches.Patch(facecolor=COLORS['positive'], label='Покупка'),
            mpatches.Patch(facecolor=COLORS['negative'], label='Продажа'),
            mpatches.Patch(facecolor=COLORS['neutral'], label='Без изменений')
        ]
        fig.legend(handles=legend_elements, loc='upper center', ncol=3, 
                  fontsize=12, framealpha=0.9)
        
        plt.suptitle('Оптимальные действия на каждом этапе', 
                    fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"График оптимальных действий сохранен: {filepath}")
        return str(filepath)
    except Exception as e:
        logger.error(f"Ошибка при создании графика оптимальных действий: {e}")
        raise


def visualize_portfolio_evolution(path: List[Portfolio],
                                  output_dir: str = "output/diagrams") -> str:
    """
    Создает визуализацию эволюции портфеля (stacked bar chart)
    
    Args:
        path: Список портфелей по этапам
        output_dir: Директория для сохранения
        
    Returns:
        Путь к сохраненному файлу
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    filepath = Path(output_dir) / "05_portfolio_evolution.png"
    
    try:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        # Извлекаем данные
        stages = list(range(len(path)))
        cb1_values = [p.cb1 for p in path]
        cb2_values = [p.cb2 for p in path]
        dep_values = [p.dep for p in path]
        cash_values = [p.cash for p in path]
        total_values = [p.total_value() for p in path]
        
        # График 1: Stacked bar chart
        width = 0.6
        x = np.arange(len(stages))
        
        ax1.bar(x, cb1_values, width, label='ЦБ1', color=COLORS['cb1'], alpha=0.8)
        ax1.bar(x, cb2_values, width, bottom=cb1_values, label='ЦБ2', 
               color=COLORS['cb2'], alpha=0.8)
        ax1.bar(x, dep_values, width, bottom=np.array(cb1_values) + np.array(cb2_values),
               label='Депозиты', color=COLORS['dep'], alpha=0.8)
        ax1.bar(x, cash_values, width, 
               bottom=np.array(cb1_values) + np.array(cb2_values) + np.array(dep_values),
               label='Касса', color=COLORS['cash'], alpha=0.8)
        
        ax1.set_xlabel('Этап', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Стоимость (д.е.)', fontsize=12, fontweight='bold')
        ax1.set_title('Эволюция компонентов портфеля', fontsize=14, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels([f'Этап {i}' for i in stages])
        ax1.legend(loc='upper left', fontsize=11)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # График 2: Общая стоимость
        ax2.plot(stages, total_values, 'o-', color=COLORS['total'], 
                linewidth=3, markersize=10, markerfacecolor='white', markeredgewidth=2)
        ax2.fill_between(stages, total_values, alpha=0.3, color=COLORS['total'])
        ax2.set_xlabel('Этап', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Общая стоимость (д.е.)', fontsize=12, fontweight='bold')
        ax2.set_title('Эволюция общей стоимости портфеля', fontsize=14, fontweight='bold')
        ax2.set_xticks(stages)
        ax2.grid(True, alpha=0.3)
        
        # Добавляем значения на точки
        for stage, value in zip(stages, total_values):
            ax2.text(stage, value, f'{value:.0f}', ha='center', va='bottom',
                    fontweight='bold', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"График эволюции портфеля сохранен: {filepath}")
        return str(filepath)
    except Exception as e:
        logger.error(f"Ошибка при создании графика эволюции портфеля: {e}")
        raise


def visualize_monte_carlo(mc_results: Dict[str, Any],
                          output_dir: str = "output/diagrams") -> Tuple[str, str]:
    """
    Создает визуализацию результатов Monte Carlo: гистограмму и CDF
    
    Args:
        mc_results: Словарь с результатами MC (должен содержать 'results')
        output_dir: Директория для сохранения
        
    Returns:
        Кортеж путей к сохраненным файлам (distribution, cdf)
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    filepath_dist = Path(output_dir) / "06_monte_carlo_distribution.png"
    filepath_cdf = Path(output_dir) / "06_monte_carlo_cdf.png"
    
    try:
        results = mc_results.get('results', [])
        if not results:
            raise ValueError("Результаты MC пусты")
        
        mean_val = mc_results.get('mean', np.mean(results))
        std_val = mc_results.get('std', np.std(results))
        min_val = mc_results.get('min', np.min(results))
        max_val = mc_results.get('max', np.max(results))
        
        # График 1: Гистограмма распределения
        fig, ax = plt.subplots(figsize=(14, 8))
        
        n_bins = min(30, len(results) // 10) if len(results) > 100 else 20
        n_bins = max(10, n_bins)
        
        n, bins, patches = ax.hist(results, bins=n_bins, color=COLORS['cb1'], 
                                  alpha=0.7, edgecolor='black', linewidth=1.5)
        
        # Накладываем нормальное распределение
        if HAS_SCIPY and std_val > 0:
            x = np.linspace(min_val, max_val, 100)
            try:
                y = stats.norm.pdf(x, mean_val, std_val) * len(results) * (bins[1] - bins[0])
                ax.plot(x, y, 'r-', linewidth=2, label=f'Нормальное распределение\n(μ={mean_val:.2f}, σ={std_val:.2f})')
            except:
                pass
        
        # Вертикальная линия со средним
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, 
                  label=f'Среднее: {mean_val:.2f} д.е.')
        ax.axvline(mean_val - std_val, color='orange', linestyle=':', linewidth=1.5, alpha=0.7)
        ax.axvline(mean_val + std_val, color='orange', linestyle=':', linewidth=1.5, alpha=0.7,
                  label=f'±1σ: [{mean_val - std_val:.2f}, {mean_val + std_val:.2f}]')
        
        ax.set_xlabel('Финальная стоимость портфеля (д.е.)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Количество симуляций', fontsize=12, fontweight='bold')
        ax.set_title('Распределение доходов при Monte Carlo симуляции (5000 симуляций)', 
                    fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Добавляем статистику в тексте
        stats_text = f'Мин: {min_val:.2f} д.е.\nМакс: {max_val:.2f} д.е.'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
               fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(filepath_dist, dpi=150, bbox_inches='tight')
        plt.close()
        
        # График 2: CDF (кумулятивное распределение)
        fig, ax = plt.subplots(figsize=(14, 8))
        
        sorted_results = np.sort(results)
        cumulative = np.arange(1, len(sorted_results) + 1) / len(sorted_results)
        
        ax.plot(sorted_results, cumulative, linewidth=2.5, color=COLORS['positive'])
        ax.fill_between(sorted_results, cumulative, alpha=0.3, color=COLORS['positive'])
        
        # Вертикальные линии для перцентилей
        p5 = np.percentile(results, 5)
        p95 = np.percentile(results, 95)
        ax.axvline(p5, color='red', linestyle='--', linewidth=2, alpha=0.7, label=f'P5: {p5:.2f}')
        ax.axvline(p95, color='red', linestyle='--', linewidth=2, alpha=0.7, label=f'P95: {p95:.2f}')
        ax.axvline(mean_val, color='orange', linestyle=':', linewidth=2, alpha=0.7, label=f'Среднее: {mean_val:.2f}')
        
        ax.set_xlabel('Финальная стоимость портфеля (д.е.)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Вероятность (кумулятивная)', fontsize=12, fontweight='bold')
        ax.set_title('Кумулятивное распределение доходов (CDF)', fontsize=14, fontweight='bold')
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(filepath_cdf, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Графики Monte Carlo сохранены: {filepath_dist}, {filepath_cdf}")
        return str(filepath_dist), str(filepath_cdf)
    except Exception as e:
        logger.error(f"Ошибка при создании графиков Monte Carlo: {e}")
        # Если scipy недоступен, создаем упрощенную версию
        try:
            return _visualize_monte_carlo_simple(mc_results, output_dir)
        except:
            raise


def _visualize_monte_carlo_simple(mc_results: Dict[str, Any],
                                  output_dir: str = "output/diagrams") -> Tuple[str, str]:
    """Упрощенная версия без scipy"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    filepath_dist = Path(output_dir) / "06_monte_carlo_distribution.png"
    filepath_cdf = Path(output_dir) / "06_monte_carlo_cdf.png"
    
    results = mc_results.get('results', [])
    mean_val = mc_results.get('mean', np.mean(results))
    std_val = mc_results.get('std', np.std(results))
    min_val = mc_results.get('min', np.min(results))
    max_val = mc_results.get('max', np.max(results))
    
    # Гистограмма
    fig, ax = plt.subplots(figsize=(14, 8))
    n_bins = min(30, len(results) // 10) if len(results) > 100 else 20
    ax.hist(results, bins=n_bins, color=COLORS['cb1'], alpha=0.7, edgecolor='black')
    ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Среднее: {mean_val:.2f}')
    ax.set_xlabel('Финальная стоимость (д.е.)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Количество симуляций', fontsize=12, fontweight='bold')
    ax.set_title('Распределение доходов (Monte Carlo)', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filepath_dist, dpi=150, bbox_inches='tight')
    plt.close()
    
    # CDF
    fig, ax = plt.subplots(figsize=(14, 8))
    sorted_results = np.sort(results)
    cumulative = np.arange(1, len(sorted_results) + 1) / len(sorted_results)
    ax.plot(sorted_results, cumulative, linewidth=2.5, color=COLORS['positive'])
    ax.set_xlabel('Финальная стоимость (д.е.)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Вероятность (кумулятивная)', fontsize=12, fontweight='bold')
    ax.set_title('Кумулятивное распределение (CDF)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filepath_cdf, dpi=150, bbox_inches='tight')
    plt.close()
    
    return str(filepath_dist), str(filepath_cdf)


def visualize_sensitivity(sensitivity_results: Dict[str, List[Dict[str, float]]],
                          output_dir: str = "output/diagrams") -> Tuple[str, str]:
    """
    Создает визуализацию анализа чувствительности
    
    Args:
        sensitivity_results: Словарь с результатами {'probabilities': [...], 'commissions': [...]}
        output_dir: Директория для сохранения
        
    Returns:
        Кортеж путей к сохраненным файлам (probabilities, commissions)
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    filepath_prob = Path(output_dir) / "07_sensitivity_probabilities.png"
    filepath_comm = Path(output_dir) / "07_sensitivity_commissions.png"
    
    try:
        # График 1: Чувствительность к вероятностям
        prob_results = sensitivity_results.get('probabilities', [])
        if prob_results:
            fig, ax = plt.subplots(figsize=(14, 8))
            
            # Группируем по этапам
            stages_data = {}
            for result in prob_results:
                param = result.get('parameter', '')
                stage = param.split('_')[0] if '_' in param else '1'
                if stage not in stages_data:
                    stages_data[stage] = {'changes': [], 'incomes': []}
                stages_data[stage]['changes'].append(result.get('change_percent', 0))
                stages_data[stage]['incomes'].append(result.get('max_income', 0))
            
            # Рисуем линии для каждого этапа
            for stage, data in stages_data.items():
                sorted_data = sorted(zip(data['changes'], data['incomes']))
                changes, incomes = zip(*sorted_data)
                ax.plot(changes, incomes, 'o-', linewidth=2.5, markersize=8,
                       label=f'Этап {stage}')
            
            ax.set_xlabel('Изменение вероятности (%)', fontsize=12, fontweight='bold')
            ax.set_ylabel('Максимальный доход (д.е.)', fontsize=12, fontweight='bold')
            ax.set_title('Чувствительность к изменению вероятностей сценариев', 
                        fontsize=14, fontweight='bold')
            ax.legend(loc='best', fontsize=11)
            ax.grid(True, alpha=0.3)
            ax.axvline(0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
            plt.tight_layout()
            plt.savefig(filepath_prob, dpi=150, bbox_inches='tight')
            plt.close()
        
        # График 2: Чувствительность к комиссиям
        comm_results = sensitivity_results.get('commissions', [])
        if comm_results:
            fig, ax = plt.subplots(figsize=(14, 8))
            
            # Группируем по типам активов
            assets_data = {}
            for result in comm_results:
                param = result.get('parameter', '')
                asset = param.split('_')[0] if '_' in param else 'cb1'
                if asset not in assets_data:
                    assets_data[asset] = {'changes': [], 'incomes': []}
                assets_data[asset]['changes'].append(result.get('change_percent', 0))
                assets_data[asset]['incomes'].append(result.get('max_income', 0))
            
            # Рисуем линии для каждого актива
            asset_names = {'cb1': 'ЦБ1', 'cb2': 'ЦБ2', 'dep': 'Депозиты'}
            for asset, data in assets_data.items():
                sorted_data = sorted(zip(data['changes'], data['incomes']))
                changes, incomes = zip(*sorted_data)
                label = asset_names.get(asset, asset)
                ax.plot(changes, incomes, 'o-', linewidth=2.5, markersize=8, label=label)
            
            ax.set_xlabel('Изменение комиссии (%)', fontsize=12, fontweight='bold')
            ax.set_ylabel('Максимальный доход (д.е.)', fontsize=12, fontweight='bold')
            ax.set_title('Чувствительность к изменению комиссий', 
                        fontsize=14, fontweight='bold')
            ax.legend(loc='best', fontsize=11)
            ax.grid(True, alpha=0.3)
            ax.axvline(0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
            plt.tight_layout()
            plt.savefig(filepath_comm, dpi=150, bbox_inches='tight')
            plt.close()
        
        logger.info(f"Графики чувствительности сохранены: {filepath_prob}, {filepath_comm}")
        return str(filepath_prob), str(filepath_comm)
    except Exception as e:
        logger.error(f"Ошибка при создании графиков чувствительности: {e}")
        raise


def visualize_criteria_comparison(criteria_results: Dict[str, float],
                                  output_dir: str = "output/diagrams") -> str:
    """
    Создает визуализацию сравнения критериев принятия решений
    
    Args:
        criteria_results: Словарь {критерий: максимальный_доход}
        output_dir: Директория для сохранения
        
    Returns:
        Путь к сохраненному файлу
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    filepath = Path(output_dir) / "08_criteria_comparison.png"
    
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # График 1: Столбчатая диаграмма
        criteria_names = list(criteria_results.keys())
        values = list(criteria_results.values())
        colors_list = [COLORS['cb1'], COLORS['cb2'], COLORS['dep']][:len(criteria_names)]
        
        bars = ax1.bar(criteria_names, values, color=colors_list, alpha=0.7,
                      edgecolor='black', linewidth=1.5)
        
        # Добавляем значения на столбцы
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.2f}', ha='center', va='bottom',
                   fontweight='bold', fontsize=12)
        
        ax1.set_ylabel('Максимальный доход (д.е.)', fontsize=12, fontweight='bold')
        ax1.set_title('Сравнение критериев принятия решений', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # График 2: Таблица сравнения
        ax2.axis('off')
        table_data = [['Критерий', 'Доход (д.е.)', 'Относительно Байеса']]
        bayesian_value = criteria_results.get('Байес', values[0] if values else 0)
        
        for name, value in criteria_results.items():
            relative = ((value - bayesian_value) / bayesian_value * 100) if bayesian_value > 0 else 0
            table_data.append([
                name,
                f'{value:.2f}',
                f'{relative:+.2f}%'
            ])
        
        table = ax2.table(cellText=table_data, cellLoc='center', loc='center',
                         colWidths=[0.4, 0.3, 0.3])
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 2)
        
        # Стилизация
        for i in range(len(table_data)):
            for j in range(3):
                if i == 0:
                    table[(i, j)].set_facecolor('#4CAF50')
                    table[(i, j)].set_text_props(weight='bold', color='white')
                else:
                    table[(i, j)].set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')
        
        ax2.set_title('Детальное сравнение', fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"График сравнения критериев сохранен: {filepath}")
        return str(filepath)
    except Exception as e:
        logger.error(f"Ошибка при создании графика сравнения критериев: {e}")
        raise


def visualize_3d_value_function(solver: DynamicProgrammingSolver,
                               output_dir: str = "output/diagrams") -> str:
    """
    Создает 3D визуализацию функции ценности
    
    Args:
        solver: Решатель с решенной задачей
        output_dir: Директория для сохранения
        
    Returns:
        Путь к сохраненному файлу
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    filepath = Path(output_dir) / "09_3d_value_function.png"
    
    try:
        # Берем данные из этапа 1 (начальный этап принятия решений)
        if 1 not in solver.value_function or not solver.value_function[1]:
            logger.warning("Нет данных для 3D визуализации")
            return ""
        
        # Подготавливаем данные для 3D графика
        portfolios = list(solver.value_function[1].keys())
        values = list(solver.value_function[1].values())
        
        # Извлекаем доли активов
        cb1_ratios = []
        cb2_ratios = []
        value_list = []
        
        for portfolio, value in zip(portfolios, values):
            total = portfolio.total_value()
            if total > 1e-6:
                cb1_ratio = portfolio.cb1 / total
                cb2_ratio = portfolio.cb2 / total
                cb1_ratios.append(cb1_ratio)
                cb2_ratios.append(cb2_ratio)
                value_list.append(value)
        
        if not cb1_ratios:
            logger.warning("Недостаточно данных для 3D визуализации")
            return ""
        
        # Создаем сетку для интерполяции
        fig = plt.figure(figsize=(16, 12))
        ax = fig.add_subplot(111, projection='3d')
        
        # Создаем сетку
        cb1_range = np.linspace(min(cb1_ratios), max(cb1_ratios), 20)
        cb2_range = np.linspace(min(cb2_ratios), max(cb2_ratios), 20)
        CB1, CB2 = np.meshgrid(cb1_range, cb2_range)
        
        # Интерполируем значения
        try:
            if HAS_SCIPY:
                try:
                    from scipy.interpolate import griddata
                    Z = griddata((cb1_ratios, cb2_ratios), value_list, (CB1, CB2), method='cubic')
                except ImportError:
                    raise ImportError("scipy.interpolate not available")
            else:
                raise ImportError("scipy not available")
            
            # Рисуем поверхность
            surf = ax.plot_surface(CB1, CB2, Z, cmap='viridis', alpha=0.8, 
                                  linewidth=0, antialiased=True)
            
            ax.set_xlabel('Доля ЦБ1 в портфеле', fontsize=11, fontweight='bold')
            ax.set_ylabel('Доля ЦБ2 в портфеле', fontsize=11, fontweight='bold')
            ax.set_zlabel('Максимальный доход (д.е.)', fontsize=11, fontweight='bold')
            ax.set_title('3D визуализация функции ценности', fontsize=14, fontweight='bold', pad=20)
            
            fig.colorbar(surf, ax=ax, shrink=0.5, aspect=20, label='Доход (д.е.)')
        except:
            # Если интерполяция не удалась, используем scatter plot
            scatter = ax.scatter(cb1_ratios, cb2_ratios, value_list, 
                               c=value_list, cmap='viridis', s=50, alpha=0.6)
            ax.set_xlabel('Доля ЦБ1', fontsize=11, fontweight='bold')
            ax.set_ylabel('Доля ЦБ2', fontsize=11, fontweight='bold')
            ax.set_zlabel('Доход (д.е.)', fontsize=11, fontweight='bold')
            ax.set_title('3D визуализация функции ценности (scatter)', 
                        fontsize=14, fontweight='bold', pad=20)
            fig.colorbar(scatter, ax=ax, shrink=0.5, aspect=20, label='Доход (д.е.)')
        
        plt.tight_layout()
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"3D график функции ценности сохранен: {filepath}")
        return str(filepath)
    except Exception as e:
        logger.error(f"Ошибка при создании 3D графика: {e}")
        # Создаем упрощенную версию без 3D
        try:
            return _visualize_3d_simple(solver, output_dir)
        except:
            return ""


def _visualize_3d_simple(solver: DynamicProgrammingSolver,
                         output_dir: str = "output/diagrams") -> str:
    """Упрощенная версия 3D визуализации (контурный график)"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    filepath = Path(output_dir) / "09_3d_value_function.png"
    
    if 1 not in solver.value_function:
        return ""
    
    portfolios = list(solver.value_function[1].keys())
    values = list(solver.value_function[1].values())
    
    cb1_ratios = []
    cb2_ratios = []
    value_list = []
    
    for portfolio, value in zip(portfolios, values):
        total = portfolio.total_value()
        if total > 1e-6:
            cb1_ratios.append(portfolio.cb1 / total)
            cb2_ratios.append(portfolio.cb2 / total)
            value_list.append(value)
    
    if not cb1_ratios:
        return ""
    
    fig, ax = plt.subplots(figsize=(14, 10))
    scatter = ax.scatter(cb1_ratios, cb2_ratios, c=value_list, cmap='viridis', 
                        s=100, alpha=0.6, edgecolors='black', linewidth=0.5)
    ax.set_xlabel('Доля ЦБ1 в портфеле', fontsize=12, fontweight='bold')
    ax.set_ylabel('Доля ЦБ2 в портфеле', fontsize=12, fontweight='bold')
    ax.set_title('Функция ценности (контурный график)', fontsize=14, fontweight='bold')
    plt.colorbar(scatter, ax=ax, label='Максимальный доход (д.е.)')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    
    return str(filepath)


def visualize_decision_tree(solver: DynamicProgrammingSolver,
                           max_nodes: int = 50,
                           output_dir: str = "output/diagrams") -> str:
    """
    Создает визуализацию дерева решений (упрощенное представление)
    
    Args:
        solver: Решатель с решенной задачей
        max_nodes: Максимальное количество узлов для отображения
        output_dir: Директория для сохранения
        
    Returns:
        Путь к сохраненному файлу
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    filepath = Path(output_dir) / "10_decision_tree.png"
    
    try:
        # Упрощенное представление дерева решений
        fig, ax = plt.subplots(figsize=(16, 12))
        ax.axis('off')
        
        # Позиционирование по уровням (этапам)
        y_levels = {0: 0.9, 1: 0.6, 2: 0.3, 3: 0.05}
        
        # Рисуем дерево для каждого этапа
        for stage in sorted(solver.value_function.keys()):
            if stage > NUM_STAGES:
                continue
            
            y = y_levels.get(stage - 1, 0.5)
            portfolios = list(solver.value_function[stage].keys())[:max_nodes]
            
            # Распределяем узлы по горизонтали
            num_nodes = len(portfolios)
            if num_nodes > 0:
                x_spacing = 0.8 / max(num_nodes, 1)
                x_start = 0.1
                
                for i, portfolio in enumerate(portfolios):
                    x = x_start + i * x_spacing
                    value = solver.value_function[stage][portfolio]
                    
                    # Рисуем узел
                    circle = plt.Circle((x, y), 0.015, color=COLORS['cb1'], zorder=3)
                    ax.add_patch(circle)
                    
                    # Подпись с ценностью
                    ax.text(x, y - 0.03, f'{value:.0f}', ha='center', fontsize=8,
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.7))
                    
                    # Линии к следующему этапу
                    if stage < NUM_STAGES and stage + 1 in solver.value_function:
                        next_portfolios = list(solver.value_function[stage + 1].keys())[:max_nodes]
                        if next_portfolios:
                            next_y = y_levels.get(stage, 0.5)
                            next_x_start = 0.1
                            next_x_spacing = 0.8 / max(len(next_portfolios), 1)
                            
                            # Рисуем несколько линий (упрощенно)
                            for j in range(min(3, len(next_portfolios))):
                                next_x = next_x_start + j * next_x_spacing
                                ax.plot([x, next_x], [y - 0.015, next_y + 0.015],
                                       'k-', linewidth=0.5, alpha=0.3, zorder=1)
                
                # Подпись этапа
                ax.text(0.05, y, f'Этап {stage}', ha='right', va='center',
                       fontsize=12, fontweight='bold',
                       bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title('Дерево решений (упрощенное представление)', 
                    fontsize=16, fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"График дерева решений сохранен: {filepath}")
        return str(filepath)
    except Exception as e:
        logger.error(f"Ошибка при создании графика дерева решений: {e}")
        # Создаем упрощенную версию
        try:
            fig, ax = plt.subplots(figsize=(14, 10))
            ax.text(0.5, 0.5, 'Дерево решений\n(упрощенное представление)', 
                   ha='center', va='center', fontsize=16, transform=ax.transAxes)
            ax.axis('off')
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            plt.close()
            return str(filepath)
        except:
            return ""
