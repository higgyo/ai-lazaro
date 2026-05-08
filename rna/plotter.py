import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots


def plot_eqm_dois_treinamentos(eqm_t1: list, eqm_t2: list, filename: str = "grafico_eqm_t1_t2.html") -> None:
    """
    Plota o EQM em função das épocas para os dois primeiros treinamentos
    numa mesma figura, conforme exigido pelo item 3 do documento.

    Parâmetros
    ----------
    eqm_t1 : list
        Lista de valores de EQM por época do Treinamento 1
    eqm_t2 : list
        Lista de valores de EQM por época do Treinamento 2
    filename : str
        Caminho do arquivo HTML gerado
    """
    fig = go.Figure()

    # Treinamento 1
    fig.add_trace(go.Scatter(
        x=list(range(1, len(eqm_t1) + 1)),
        y=eqm_t1,
        mode="lines+markers",
        name="Treinamento 1 (T1)",
        line=dict(color="#2563EB", width=2),
        marker=dict(size=4, symbol="circle"),
    ))

    # Treinamento 2
    fig.add_trace(go.Scatter(
        x=list(range(1, len(eqm_t2) + 1)),
        y=eqm_t2,
        mode="lines+markers",
        name="Treinamento 2 (T2)",
        line=dict(color="#DC2626", width=2, dash="dash"),
        marker=dict(size=4, symbol="square"),
    ))

    fig.update_layout(
        title=dict(
            text="EQM por Época — Treinamentos T1 e T2 (ADALINE — Regra Delta)",
            font=dict(size=16),
        ),
        xaxis=dict(
            title="Época",
            showgrid=True,
            gridcolor="#e5e7eb",
        ),
        yaxis=dict(
            title="Erro Quadrático Médio (EQM)",
            type="log",
            showgrid=True,
            gridcolor="#e5e7eb",
            tickformat=".2e",
        ),
        legend=dict(x=0.75, y=0.95),
        template="plotly_white",
        width=900,
        height=550,
        annotations=[
            dict(
                text="η = 0.0025 | ε = 10⁻⁶",
                xref="paper", yref="paper",
                x=0.01, y=0.01,
                showarrow=False,
                font=dict(size=11, color="#6b7280"),
            )
        ],
    )

    fig.write_html(filename)
    print(f"Gráfico EQM (T1 e T2) salvo em: {filename}")


def plot_eqm_individual(eqm: list, training_index: int, filename: str = None) -> None:
    """
    Gera gráfico individual do EQM por época para um treinamento específico.

    Parâmetros
    ----------
    eqm : list
        Lista de EQM por época
    training_index : int
        Índice do treinamento (1 a 5)
    filename : str
        Caminho do arquivo HTML (auto-gerado se None)
    """
    if filename is None:
        filename = f"grafico_eqm_treinamento_{training_index}.html"

    cores = ["#2563EB", "#DC2626", "#16A34A", "#D97706", "#7C3AED"]
    cor = cores[(training_index - 1) % len(cores)]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(1, len(eqm) + 1)),
        y=eqm,
        mode="lines+markers",
        name=f"T{training_index}",
        line=dict(color=cor, width=2),
        marker=dict(size=4),
    ))

    fig.update_layout(
        title=f"EQM por Época — Treinamento {training_index} (ADALINE)",
        xaxis=dict(title="Época", showgrid=True, gridcolor="#e5e7eb"),
        yaxis=dict(
            title="EQM",
            type="log",
            showgrid=True,
            gridcolor="#e5e7eb",
            tickformat=".2e",
        ),
        template="plotly_white",
        width=800,
        height=450,
    )

    fig.write_html(filename)
    print(f"Gráfico EQM salvo em: {filename}")
