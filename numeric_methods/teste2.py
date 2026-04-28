"""
================================================================================
TEMPLATE PADRÃO — MÉTODOS NUMÉRICOS
================================================================================
Disciplina : Métodos Numéricos
Autor      : Seu Nome
Matrícula  : 000000
Data       : 2026-04-28
================================================================================

DESCRIÇÃO DO PROBLEMA:
    Breve descrição do que este script resolve.
    Ex.: Encontrar a raiz de f(x) = x³ - 2x - 5 no intervalo [2, 3]
         usando o Método da Bissecção.

MÉTODO UTILIZADO:
    Nome do método + referência teórica se quiser.

ENTRADAS:
    a, b    → extremos do intervalo
    tol     → tolerância (critério de parada)
    max_it  → número máximo de iterações

SAÍDAS:
    raiz aproximada, número de iterações, tabela de convergência
================================================================================
"""

# ==============================================================================
# 1. IMPORTS
# ==============================================================================
import numpy as np
import matplotlib.pyplot as plt


# ==============================================================================
# 2. DEFINIÇÃO DA FUNÇÃO (f(x) que queremos trabalhar)
# ==============================================================================

def f(x):
    """
    Função alvo do problema.

    Parâmetros
    ----------
    x : float
        Ponto de avaliação.

    Retorna
    -------
    float
        Valor de f(x).
    """
    return x**3 - 2*x - 5


def df(x):
    """
    Derivada de f(x) — necessária em métodos como Newton-Raphson.

    Parâmetros
    ----------
    x : float
        Ponto de avaliação.

    Retorna
    -------
    float
        Valor de f'(x).
    """
    return 3*x**2 - 2


# ==============================================================================
# 3. IMPLEMENTAÇÃO DO MÉTODO NUMÉRICO
# ==============================================================================

def bisseccao(f, a, b, tol=1e-6, max_it=100):
    """
    Método da Bissecção para encontrar raízes de f(x) = 0.

    O método exige que f(a) * f(b) < 0 (Teorema de Bolzano).

    Parâmetros
    ----------
    f      : callable  → função contínua
    a      : float     → extremo esquerdo do intervalo
    b      : float     → extremo direito do intervalo
    tol    : float     → tolerância de parada (padrão: 1e-6)
    max_it : int       → máximo de iterações permitidas (padrão: 100)

    Retorna
    -------
    raiz     : float → aproximação da raiz
    historico: list  → lista com dicionários de cada iteração

    Levanta
    -------
    ValueError
        Se f(a) e f(b) tiverem o mesmo sinal (condição inválida).
    """

    # --- Verificação das condições iniciais ---
    if f(a) * f(b) >= 0:
        raise ValueError(
            f"f(a) e f(b) devem ter sinais opostos.\n"
            f"f({a}) = {f(a):.4f}, f({b}) = {f(b):.4f}"
        )

    historico = []   # guarda dados de cada iteração para análise

    for i in range(1, max_it + 1):

        # --- Passo principal do método ---
        c = (a + b) / 2.0      # ponto médio
        fc = f(c)
        erro = abs(b - a) / 2  # estimativa do erro

        # --- Salvar dados da iteração ---
        historico.append({
            "iteracao": i,
            "a": a,
            "b": b,
            "c": c,
            "f(c)": fc,
            "erro": erro
        })

        # --- Critério de parada ---
        if erro < tol or fc == 0.0:
            break

        # --- Atualizar intervalo ---
        if f(a) * fc < 0:
            b = c   # raiz está em [a, c]
        else:
            a = c   # raiz está em [c, b]

    return c, historico


def newton_raphson(f, df, x0, tol=1e-6, max_it=100):
    """
    Método de Newton-Raphson para encontrar raízes de f(x) = 0.

    Parâmetros
    ----------
    f      : callable → função
    df     : callable → derivada de f
    x0     : float    → chute inicial
    tol    : float    → tolerância de parada
    max_it : int      → máximo de iterações

    Retorna
    -------
    x        : float → aproximação da raiz
    historico: list  → dados de cada iteração
    """
    x = x0
    historico = []

    for i in range(1, max_it + 1):

        fx  = f(x)
        dfx = df(x)

        if dfx == 0:
            raise ZeroDivisionError(f"Derivada nula em x = {x}. Escolha outro x0.")

        x_novo = x - fx / dfx
        erro   = abs(x_novo - x)

        historico.append({
            "iteracao": i,
            "x":        x,
            "f(x)":     fx,
            "f'(x)":    dfx,
            "x_novo":   x_novo,
            "erro":     erro
        })

        x = x_novo

        if erro < tol:
            break

    return x, historico


# ==============================================================================
# 4. SAÍDA / RELATÓRIO
# ==============================================================================

def imprimir_tabela(historico, colunas=None):
    """
    Imprime uma tabela formatada com o histórico de iterações.

    Parâmetros
    ----------
    historico : list of dict → retorno dos métodos
    colunas   : list         → quais colunas mostrar (None = todas)
    """
    if not historico:
        print("Nenhuma iteração registrada.")
        return

    if colunas is None:
        colunas = list(historico[0].keys())

    # Cabeçalho
    header = " | ".join(f"{c:>12}" for c in colunas)
    print("-" * len(header))
    print(header)
    print("-" * len(header))

    # Linhas
    for linha in historico:
        valores = []
        for c in colunas:
            v = linha[c]
            if isinstance(v, int):
                valores.append(f"{v:>12d}")
            else:
                valores.append(f"{v:>12.6f}")
        print(" | ".join(valores))

    print("-" * len(header))


# ==============================================================================
# 5. VISUALIZAÇÃO
# ==============================================================================

def plotar_funcao(f, a, b, raiz=None, titulo="Gráfico da Função"):
    """
    Plota f(x) no intervalo [a, b] e marca a raiz encontrada.

    Parâmetros
    ----------
    f     : callable → função a plotar
    a, b  : float    → intervalo do gráfico
    raiz  : float    → posição da raiz (opcional)
    titulo: str      → título do gráfico
    """
    x_vals = np.linspace(a, b, 500)
    y_vals = f(x_vals)

    plt.figure(figsize=(8, 5))
    plt.axhline(0, color="black", linewidth=0.8, linestyle="--")
    plt.plot(x_vals, y_vals, color="royalblue", linewidth=2, label="f(x)")

    if raiz is not None:
        plt.scatter([raiz], [f(raiz)], color="crimson", zorder=5,
                    label=f"Raiz ≈ {raiz:.6f}")

    plt.title(titulo, fontsize=14, fontweight="bold")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("grafico_metodo.png", dpi=150)
    plt.show()


# ==============================================================================
# 6. PROGRAMA PRINCIPAL
# ==============================================================================

def main():
    """
    Ponto de entrada do programa.
    Chama os métodos, imprime resultados e gera visualização.
    """

    print("=" * 60)
    print("  MÉTODOS NUMÉRICOS — Busca de Raízes")
    print("  f(x) = x³ - 2x - 5")
    print("=" * 60)

    # --- Parâmetros ---
    a, b   = 2.0, 3.0
    tol    = 1e-6
    max_it = 50

    # ---- BISSECÇÃO ----
    print("\n>>> MÉTODO DA BISSECÇÃO")
    raiz_bis, hist_bis = bisseccao(f, a, b, tol, max_it)
    imprimir_tabela(hist_bis, colunas=["iteracao", "a", "b", "c", "f(c)", "erro"])
    print(f"\n  Raiz encontrada : {raiz_bis:.8f}")
    print(f"  Iterações       : {len(hist_bis)}")
    print(f"  f(raiz)         : {f(raiz_bis):.2e}")

    # ---- NEWTON-RAPHSON ----
    print("\n>>> MÉTODO DE NEWTON-RAPHSON")
    x0 = 2.5
    raiz_nr, hist_nr = newton_raphson(f, df, x0, tol, max_it)
    imprimir_tabela(hist_nr, colunas=["iteracao", "x", "f(x)", "f'(x)", "x_novo", "erro"])
    print(f"\n  Raiz encontrada : {raiz_nr:.8f}")
    print(f"  Iterações       : {len(hist_nr)}")
    print(f"  f(raiz)         : {f(raiz_nr):.2e}")

    # ---- GRÁFICO ----
    plotar_funcao(f, a, b, raiz=raiz_nr, titulo="f(x) = x³ - 2x - 5")


# ==============================================================================
# Ponto de entrada padrão Python
# ==============================================================================
if __name__ == "__main__":
    main()
