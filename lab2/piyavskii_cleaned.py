import numpy as np
import matplotlib.pyplot as plt
import math
import time
from typing import Callable, List, Tuple, Optional
from matplotlib.backends.backend_pdf import PdfPages


# ==========================
# Безопасный парсер строки -> функция
# ==========================
def parse_function(func_str: str) -> Callable[[float], float]:
    """
    Преобразует строку в функцию f(x).
    Поддерживаются стандартные функции из math и numpy.
    """
    if "=" in func_str:
        func_str = func_str.split("=", 1)[1].strip()

    safe_globals = {
        "__builtins__": None,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "asin": math.asin,
        "acos": math.acos,
        "atan": math.atan,
        "exp": math.exp,
        "log": math.log,
        "log10": math.log10,
        "sqrt": math.sqrt,
        "pi": math.pi,
        "e": math.e,
        "abs": abs,
        "np": np,
    }

    expr = func_str

    try:
        def f(x: float) -> float:
            return float(eval(expr, safe_globals, {"x": x}))

        _ = f(0.5)
        return f
    except Exception as e:
        raise ValueError(f"Невозможно распарсить функцию: {e}")


# ==========================
# Приближённая оценка константы Липшица (если L не задано)
# ==========================
def estimate_lipschitz(f: Callable[[float], float], a: float, b: float, samples: int = 2000) -> float:
    """
    Оценивает константу Липшица L как максимум модуля производной на отрезке [a,b].
    """
    xs = np.linspace(a, b, samples)
    ys = np.array([f(x) for x in xs])
    dx = xs[1] - xs[0]
    dy = np.gradient(ys, dx)
    L_est = float(np.max(np.abs(dy)))
    return max(L_est, 1e-6)


# ==========================
# Метод Пиявского
# ==========================

def piyavskii_method(
    f: Callable[[float], float],
    a: float,
    b: float,
    L: Optional[float] = None,
    eps: float = 1e-2,
    max_iter: int = 5000,
    timeout: Optional[float] = None,
    store_full_history: bool = False,
) -> Tuple[float, float, int, float, List[Tuple[List[float], List[float]]], float]:
    """
    Возвращает (x_min, y_min, iterations, total_time, history, L_used)
    """

    if a >= b:
        raise ValueError("Неверный отрезок: a должен быть < b")
    if eps <= 0:
        raise ValueError("eps должно быть > 0")

    start_time = time.time()

    if L is None:
        L = estimate_lipschitz(f, a, b)

    if L <= 0:
        raise ValueError("Константа Липшица L должна быть положительной")

    X: List[float] = [a, b]
    Y: List[float] = [f(a), f(b)]

    history: List[Tuple[List[float], List[float]]] = [(X.copy(), Y.copy())]

    iterations = 0

    while True:
        iterations += 1

        if timeout is not None and (time.time() - start_time) > timeout:
            print("Прервано по таймауту")
            break

        y_min = min(Y)

        R = []
        for i in range(len(X) - 1):
            xi, xip1 = X[i], X[i + 1]
            yi, yip1 = Y[i], Y[i + 1]
            ri = 0.5 * (yi + yip1) - 0.5 * L * (xip1 - xi)
            R.append(ri)

        j = int(np.argmin(R))
        R_min = R[j]

        if y_min - R_min <= eps:
            break

        xj, xjp1 = X[j], X[j + 1]
        yj, yjp1 = Y[j], Y[j + 1]

        x_new = 0.5 * (xj + xjp1) + (yj - yjp1) / (2.0 * L)
        x_new = max(min(x_new, xjp1 - 1e-14), xj + 1e-14)

        y_new = f(x_new)

        X.insert(j + 1, x_new)
        Y.insert(j + 1, y_new)

        if store_full_history:
            history.append((X.copy(), Y.copy()))
        else:
            history = [(X.copy(), Y.copy())]

        if iterations >= max_iter:
            print(f"Достигнуто max_iter = {max_iter}")
            break

    total_time = time.time() - start_time

    y_min_final = min(Y)
    min_indices = [i for i, y in enumerate(Y) if abs(y - y_min_final) < 1e-10]
    x_min_final = X[min_indices[0]]

    return x_min_final, y_min_final, iterations, total_time, history, L


# ==========================
# Визуализация и отчёт
# ==========================

def plot_results(
    f: Callable[[float], float],
    a: float,
    b: float,
    L: float,
    history: List[Tuple[List[float], List[float]]],
    x_min: float,
    y_min: float,
    title: str = "Результаты метода Пиявского",
    save_path: Optional[str] = None,
):
    x_plot = np.linspace(a, b, 2000)
    y_plot = np.array([f(x) for x in x_plot])

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(x_plot, y_plot, label="f(x)")

    final_X, final_Y = history[-1]
    final_X_np = np.array(final_X)
    final_Y_np = np.array(final_Y)

    all_v = final_Y_np - L * np.abs(x_plot[:, None] - final_X_np[None, :])
    g_plot = np.max(all_v, axis=1)
    ax.plot(x_plot, g_plot, label="Итоговая ломаная (нижняя оценка)", linestyle="--")

    ax.scatter(final_X, final_Y, label="Точки вычислений f(x)")
    ax.scatter([x_min], [y_min], label=f"Найденный минимум: x≈{x_min:.6f}, f≈{y_min:.6f}", marker="*", s=200)

    ax.set_title(title)
    ax.set_xlabel('x')
    ax.set_ylabel('f(x)')
    ax.grid(True, linestyle=':')
    ax.legend()

    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
    plt.show()
    return fig


def save_report(pdf_path: str, fig_list: List[plt.Figure], summary_text: str):
    with PdfPages(pdf_path) as pdf:
        for fig in fig_list:
            pdf.savefig(fig)
        fig_text = plt.figure(figsize=(8.27, 11.69))
        fig_text.clf()
        fig_text.text(0.01, 0.99, "Результаты расчёта", fontsize=14, weight='bold')
        fig_text.text(0.01, 0.95, summary_text, fontsize=10)
        pdf.savefig(fig_text)
        plt.close('all')


# ==========================
# Демонстрации (примерные функции с несколькими локальными минимумами)
# ==========================

def demo_examples():
    demos = []

    func_str = "(x-2)**2 + 5*sin(pi*x)"
    f = parse_function(func_str)
    a, b = 0.0, 6.0
    x_min, y_min, iters, ttime, history, L_used = piyavskii_method(f, a, b, L=None, eps=0.01)
    fig1 = plot_results(f, a, b, L_used, history, x_min, y_min, title=f"Метод Пиявского: {func_str}")
    summary1 = f"Функция: {func_str}\nОтрезок: [{a}, {b}]\nL (оценка): {L_used:.6g}\n eps=0.01\n Итераций: {iters}\n Время: {ttime:.6f} сек\n x_min≈{x_min:.6f}, f(x_min)≈{y_min:.6f}"
    demos.append((fig1, summary1))

    A = 10
    func_str2 = f"{A} + (x**2 - {A}*cos(2*pi*x))"
    f2 = parse_function(func_str2)
    a2, b2 = -5.12, 5.12
    x_min2, y_min2, iters2, ttime2, history2, L2 = piyavskii_method(f2, a2, b2, L=None, eps=0.01)
    fig2 = plot_results(f2, a2, b2, L2, history2, x_min2, y_min2, title=f"Rastrigin (1D): {func_str2}")
    summary2 = f"Функция: Rastrigin 1D\nОтрезок: [{a2}, {b2}]\nL (оценка): {L2:.6g}\n eps=0.01\n Итераций: {iters2}\n Время: {ttime2:.6f} сек\n x_min≈{x_min2:.6f}, f(x_min)≈{y_min2:.6f}"
    demos.append((fig2, summary2))

    func_str3 = "-20*exp(-0.2*sqrt(x**2)) - exp(0.5*(cos(2*pi*x) + 1)) + 20 + e"
    f3 = parse_function(func_str3)
    a3, b3 = -5.0, 5.0
    x_min3, y_min3, iters3, ttime3, history3, L3 = piyavskii_method(f3, a3, b3, L=None, eps=0.01)
    fig3 = plot_results(f3, a3, b3, L3, history3, x_min3, y_min3, title="Ackley (1D)")
    summary3 = f"Функция: Ackley 1D\nОтрезок: [{a3}, {b3}]\nL (оценка): {L3:.6g}\n eps=0.01\n Итераций: {iters3}\n Время: {ttime3:.6f} сек\n x_min≈{x_min3:.6f}, f(x_min)≈{y_min3:.6f}"
    demos.append((fig3, summary3))

    pdf_path = "piyavskii_report.pdf"
    figs = [d[0] for d in demos]
    summaries = "\n\n".join(d[1] for d in demos)
    save_report(pdf_path, figs, summaries)
    print(f"Отчёт сохранён в {pdf_path}")


if __name__ == '__main__':
    demo_examples()
