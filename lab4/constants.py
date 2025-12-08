"""
Константы для задачи динамического программирования портфеля
"""

# Начальные позиции портфеля (в д.е.)
INITIAL_CB1 = 100.0
INITIAL_CB2 = 800.0
INITIAL_DEP = 400.0
INITIAL_CASH = 600.0  # Свободные средства

# Флаги опциональных функций (можно включить/выключить)
USE_COMMISSIONS = True  # Комиссии включены
USE_MIN_CONSTRAINTS = True  # Ограничения на минимум включены

# Минимальные ограничения (в д.е.) - используются только если USE_MIN_CONSTRAINTS = True
MIN_CB1 = 30.0
MIN_CB2 = 150.0
MIN_DEP = 100.0

# Комиссии брокеров (в долях) - используются только если USE_COMMISSIONS = True
COMMISSION_CB1 = 0.04  # 4%
COMMISSION_CB2 = 0.07  # 7%
COMMISSION_DEP = 0.05  # 5%

COMMISSIONS = {
    'cb1': COMMISSION_CB1,
    'cb2': COMMISSION_CB2,
    'dep': COMMISSION_DEP
}

# Пакеты (25% от начального объема)
PACKAGE_CB1 = INITIAL_CB1 * 0.25   # 25 д.е.
PACKAGE_CB2 = INITIAL_CB2 * 0.25   # 200 д.е.
PACKAGE_DEP = INITIAL_DEP * 0.25   # 100 д.е.

PACKAGES = {
    'cb1': PACKAGE_CB1,
    'cb2': PACKAGE_CB2,
    'dep': PACKAGE_DEP
}

# Сценарии для каждого этапа (если не читаются из Excel)
SCENARIOS = {
    1: [  # Этап 1
        {'situation': 'благоприятная', 'probability': 0.60, 'cb1': 1.20, 'cb2': 1.10, 'dep': 1.07},
        {'situation': 'нейтральная',   'probability': 0.30, 'cb1': 1.05, 'cb2': 1.02, 'dep': 1.03},
        {'situation': 'негативная',    'probability': 0.10, 'cb1': 0.80, 'cb2': 0.95, 'dep': 1.00},
    ],
    2: [  # Этап 2
        {'situation': 'благоприятная', 'probability': 0.30, 'cb1': 1.40, 'cb2': 1.15, 'dep': 1.01},
        {'situation': 'нейтральная',   'probability': 0.20, 'cb1': 1.05, 'cb2': 1.00, 'dep': 1.00},
        {'situation': 'негативная',    'probability': 0.50, 'cb1': 0.60, 'cb2': 0.90, 'dep': 1.00},
    ],
    3: [  # Этап 3
        {'situation': 'благоприятная', 'probability': 0.40, 'cb1': 1.15, 'cb2': 1.12, 'dep': 1.05},
        {'situation': 'нейтральная',   'probability': 0.40, 'cb1': 1.05, 'cb2': 1.01, 'dep': 1.01},
        {'situation': 'негативная',    'probability': 0.20, 'cb1': 0.70, 'cb2': 0.94, 'dep': 1.00},
    ]
}

# Количество этапов
NUM_STAGES = 3

# Точность для сравнения чисел с плавающей точкой
EPSILON = 1e-6
