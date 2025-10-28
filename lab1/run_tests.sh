#!/bin/bash

# Скрипт для тестирования симплекс-метода
# Автор: Губанов Егор Романович

# Цвета для вывода
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Функция для красивого заголовка
print_header() {
    echo ""
    echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║${NC}  $1"
    echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"
    echo ""
}

# Функция для вывода меню
show_menu() {
    clear
    echo -e "${CYAN}╔═══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║${NC}           ТЕСТИРОВАНИЕ СИМПЛЕКС-МЕТОДА                      ${CYAN}║${NC}"
    echo -e "${CYAN}║${NC}           Губанов Егор Романович, МетОпт 1.1                ${CYAN}║${NC}"
    echo -e "${CYAN}╚═══════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "${GREEN}Выберите действие:${NC}"
    echo ""
    echo "  1) Проверить окружение (venv, зависимости)"
    echo "  2) Запустить ОДНУ задачу (детальный вывод)"
    echo "  3) Сравнить решения для ОДНОЙ задачи"
    echo "  4) Запустить ВСЕ 20 тестов (автоматически)"
    echo "  5) Сгенерировать заново тестовые файлы"
    echo "  6) Показать структуру проекта"
    echo ""
    echo "  0) Выход"
    echo ""
    echo -n -e "${YELLOW}Ваш выбор: ${NC}"
}

# Функция проверки окружения
check_environment() {
    print_header "ПРОВЕРКА ОКРУЖЕНИЯ"
    
    # Проверка venv
    if [ -d "venv" ]; then
        echo -e "${GREEN}✓${NC} Виртуальное окружение найдено"
    else
        echo -e "${RED}✗${NC} Виртуальное окружение не найдено"
        echo -e "${YELLOW}Создание venv...${NC}"
        python3 -m venv venv
    fi
    
    # Активация venv
    echo -e "${YELLOW}Активация venv...${NC}"
    source venv/bin/activate
    
    # Проверка зависимостей
    echo -e "${YELLOW}Проверка зависимостей...${NC}"
    pip list | grep -E "numpy|scipy"
    
    if [ $? -ne 0 ]; then
        echo -e "${RED}✗${NC} Зависимости не установлены"
        echo -e "${YELLOW}Установка зависимостей...${NC}"
        pip install -r requirements.txt
    else
        echo -e "${GREEN}✓${NC} Все зависимости установлены"
    fi
    
    echo ""
    echo -e "${GREEN}Окружение готово к работе!${NC}"
    echo ""
    read -p "Нажмите Enter для продолжения..."
}

# Функция запуска одной задачи с детальным выводом
run_single_detailed() {
    print_header "ЗАПУСК ОДНОЙ ЗАДАЧИ (ДЕТАЛЬНО)"
    
    echo -e "${YELLOW}Доступные задачи:${NC} problem01.txt - problem20.txt"
    echo -e "${YELLOW}Ваш вариант:${NC} problem10.txt"
    echo ""
    echo -n "Введите номер задачи (01-20) или Enter для варианта 10: "
    read task_num
    
    if [ -z "$task_num" ]; then
        task_num="10"
    fi
    
    # Форматирование номера
    task_file=$(printf "test_problems/problem%02d.txt" $task_num)
    
    if [ ! -f "$task_file" ]; then
        echo -e "${RED}✗${NC} Файл $task_file не найден!"
        read -p "Нажмите Enter для продолжения..."
        return
    fi
    
    echo ""
    print_header "ДЕТАЛЬНОЕ РЕШЕНИЕ: $task_file"
    
    source venv/bin/activate
    python3 simplex_solver_detailed.py "$task_file"
    
    echo ""
    read -p "Нажмите Enter для продолжения..."
}

# Функция сравнения для одной задачи
compare_single() {
    print_header "СРАВНЕНИЕ РЕШЕНИЙ ДЛЯ ОДНОЙ ЗАДАЧИ"
    
    echo -e "${YELLOW}Доступные задачи:${NC} problem01.txt - problem20.txt"
    echo ""
    echo -n "Введите номер задачи (01-20) или Enter для варианта 10: "
    read task_num
    
    if [ -z "$task_num" ]; then
        task_num="10"
    fi
    
    # Форматирование номера
    task_file=$(printf "test_problems/problem%02d.txt" $task_num)
    
    if [ ! -f "$task_file" ]; then
        echo -e "${RED}✗${NC} Файл $task_file не найден!"
        read -p "Нажмите Enter для продолжения..."
        return
    fi
    
    echo ""
    print_header "СРАВНЕНИЕ: $task_file"
    
    source venv/bin/activate
    python3 compare_solutions.py "$task_file"
    
    echo ""
    read -p "Нажмите Enter для продолжения..."
}

# Функция запуска всех тестов
run_all_tests() {
    print_header "ЗАПУСК ВСЕХ 20 ТЕСТОВ"
    
    source venv/bin/activate
    python3 test_all_problems.py
    
    echo ""
    read -p "Нажмите Enter для продолжения..."
}

# Функция генерации тестовых файлов
generate_tests() {
    print_header "ГЕНЕРАЦИЯ ТЕСТОВЫХ ФАЙЛОВ"
    
    echo -e "${YELLOW}Генерация 20 тестовых задач...${NC}"
    
    source venv/bin/activate
    python3 generate_test_problems.py
    
    echo ""
    echo -e "${GREEN}✓${NC} Тестовые файлы сгенерированы в test_problems/"
    echo ""
    read -p "Нажмите Enter для продолжения..."
}

# Функция показа структуры проекта
show_structure() {
    print_header "СТРУКТУРА ПРОЕКТА"
    
    echo -e "${CYAN}Основные файлы:${NC}"
    echo ""
    ls -lh | grep -v "^d" | grep -v "total" | awk '{print "  " $9, "(" $5 ")"}'
    
    echo ""
    echo -e "${CYAN}Директории:${NC}"
    echo ""
    echo "  test_problems/ ($(ls test_problems | wc -l | xargs) файлов)"
    echo "  __pycache__/ (кэш Python)"
    echo "  venv/ (виртуальное окружение)"
    
    echo ""
    echo -e "${CYAN}Тестовые задачи:${NC}"
    echo ""
    ls test_problems/*.txt 2>/dev/null | head -5 | awk '{print "  " $0}'
    echo "  ..."
    
    echo ""
    read -p "Нажмите Enter для продолжения..."
}

# Основной цикл
while true; do
    show_menu
    read choice
    
    case $choice in
        1) check_environment ;;
        2) run_single_detailed ;;
        3) compare_single ;;
        4) run_all_tests ;;
        5) generate_tests ;;
        6) show_structure ;;
        0) 
            echo ""
            echo -e "${GREEN}До свидания!${NC}"
            echo ""
            exit 0
            ;;
        *)
            echo -e "${RED}Неверный выбор. Попробуйте снова.${NC}"
            sleep 1
            ;;
    esac
done





