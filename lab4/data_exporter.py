"""
Модуль для экспорта данных в CSV файлы
"""
import csv
from typing import List, Dict, Tuple, Any
from pathlib import Path
from models import Portfolio
import logging

logger = logging.getLogger(__name__)


def export_optimal_actions(actions: List[Tuple[float, float, float]], 
                          expected_values: List[float] = None,
                          output_dir: str = "output/data") -> str:
    """
    Экспортирует оптимальные действия в CSV файл
    
    Args:
        actions: Список действий по этапам [(delta_cb1, delta_cb2, delta_dep), ...]
        expected_values: Список ожидаемых значений для каждого этапа (опционально)
        output_dir: Директория для сохранения файла
        
    Returns:
        Путь к созданному CSV файлу
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    filepath = Path(output_dir) / "optimal_actions.csv"
    
    try:
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['stage', 'delta_cb1', 'delta_cb2', 'delta_dep', 'expected_value'])
            
            for i, action in enumerate(actions, 1):
                delta_cb1, delta_cb2, delta_dep = action
                expected_value = expected_values[i-1] if expected_values and i-1 < len(expected_values) else 0.0
                writer.writerow([i, delta_cb1, delta_cb2, delta_dep, expected_value])
        
        logger.info(f"Экспортировано {len(actions)} действий в {filepath}")
        return str(filepath)
    except Exception as e:
        logger.error(f"Ошибка при экспорте оптимальных действий: {e}")
        raise


def export_portfolio_evolution(path: List[Portfolio],
                              output_dir: str = "output/data") -> str:
    """
    Экспортирует эволюцию портфеля в CSV файл
    
    Args:
        path: Список портфелей по этапам
        output_dir: Директория для сохранения файла
        
    Returns:
        Путь к созданному CSV файлу
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    filepath = Path(output_dir) / "portfolio_evolution.csv"
    
    try:
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['stage', 'cb1', 'cb2', 'dep', 'cash', 'total_value'])
            
            for i, portfolio in enumerate(path):
                writer.writerow([
                    i,
                    portfolio.cb1,
                    portfolio.cb2,
                    portfolio.dep,
                    portfolio.cash,
                    portfolio.total_value()
                ])
        
        logger.info(f"Экспортировано {len(path)} состояний портфеля в {filepath}")
        return str(filepath)
    except Exception as e:
        logger.error(f"Ошибка при экспорте эволюции портфеля: {e}")
        raise


def export_monte_carlo_results(mc_results: Dict[str, Any],
                               output_dir: str = "output/data") -> str:
    """
    Экспортирует результаты Monte Carlo симуляции в CSV файл
    
    Args:
        mc_results: Словарь с результатами MC, должен содержать 'results' (список значений)
        output_dir: Директория для сохранения файла
        
    Returns:
        Путь к созданному CSV файлу
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    filepath = Path(output_dir) / "monte_carlo_results.csv"
    
    try:
        results = mc_results.get('results', [])
        
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['simulation_id', 'final_value'])
            
            for i, value in enumerate(results, 1):
                writer.writerow([i, value])
        
        # Также сохраняем статистику в отдельную строку
        stats_filepath = Path(output_dir) / "monte_carlo_stats.csv"
        with open(stats_filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['metric', 'value'])
            writer.writerow(['mean', mc_results.get('mean', 0)])
            writer.writerow(['std', mc_results.get('std', 0)])
            writer.writerow(['min', mc_results.get('min', 0)])
            writer.writerow(['max', mc_results.get('max', 0)])
        
        logger.info(f"Экспортировано {len(results)} результатов MC в {filepath}")
        return str(filepath)
    except Exception as e:
        logger.error(f"Ошибка при экспорте результатов MC: {e}")
        raise


def export_sensitivity_analysis(sensitivity_results: Dict[str, List[Dict[str, float]]],
                                output_dir: str = "output/data") -> str:
    """
    Экспортирует анализ чувствительности в CSV файл
    
    Args:
        sensitivity_results: Словарь с результатами анализа чувствительности
            Формат: {'probabilities': [...], 'commissions': [...]}
            Каждый элемент: {'parameter': str, 'change_percent': float, 'max_income': float}
        output_dir: Директория для сохранения файла
        
    Returns:
        Путь к созданному CSV файлу
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    filepath = Path(output_dir) / "sensitivity_analysis.csv"
    
    try:
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['parameter_type', 'parameter', 'change_percent', 'max_income'])
            
            # Экспортируем результаты по вероятностям
            prob_results = sensitivity_results.get('probabilities', [])
            for result in prob_results:
                writer.writerow([
                    'probability',
                    result.get('parameter', ''),
                    result.get('change_percent', 0),
                    result.get('max_income', 0)
                ])
            
            # Экспортируем результаты по комиссиям
            comm_results = sensitivity_results.get('commissions', [])
            for result in comm_results:
                writer.writerow([
                    'commission',
                    result.get('parameter', ''),
                    result.get('change_percent', 0),
                    result.get('max_income', 0)
                ])
        
        total_results = len(prob_results) + len(comm_results)
        logger.info(f"Экспортировано {total_results} результатов анализа чувствительности в {filepath}")
        return str(filepath)
    except Exception as e:
        logger.error(f"Ошибка при экспорте анализа чувствительности: {e}")
        raise


def export_all_data(actions: List[Tuple[float, float, float]],
                   path: List[Portfolio],
                   mc_results: Dict[str, Any],
                   sensitivity_results: Dict[str, List[Dict[str, float]]] = None,
                   expected_values: List[float] = None,
                   output_dir: str = "output/data") -> Dict[str, str]:
    """
    Экспортирует все данные в CSV файлы
    
    Args:
        actions: Список оптимальных действий
        path: Список портфелей по этапам
        mc_results: Результаты Monte Carlo
        sensitivity_results: Результаты анализа чувствительности (опционально)
        expected_values: Ожидаемые значения для действий (опционально)
        output_dir: Директория для сохранения файлов
        
    Returns:
        Словарь с путями к созданным файлам
    """
    exported_files = {}
    
    try:
        exported_files['optimal_actions'] = export_optimal_actions(actions, expected_values, output_dir)
        exported_files['portfolio_evolution'] = export_portfolio_evolution(path, output_dir)
        exported_files['monte_carlo'] = export_monte_carlo_results(mc_results, output_dir)
        
        if sensitivity_results:
            exported_files['sensitivity'] = export_sensitivity_analysis(sensitivity_results, output_dir)
        
        logger.info(f"Все данные экспортированы в {output_dir}")
        return exported_files
    except Exception as e:
        logger.error(f"Ошибка при экспорте всех данных: {e}")
        raise
