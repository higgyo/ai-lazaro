import plotly.graph_objects as go
import numpy as np

def plot_training_interactive(X, d, weights_history, training_index):
    """
    Gera um arquivo HTML com o gráfico 3D iterativo do processo de treinamento
    usando Plotly.
    X: coordenadas [x1, x2, x3]
    d: classes reais
    weights_history: lista de pesos ao final de cada época
    training_index: número do treinamento (para o nome do arquivo)
    """
    fig = go.Figure()
    
    # Separar os pontos por classe
    X = np.array(X)
    d = np.array(d)
    
    class_1 = X[d == -1] # Classe C1
    class_2 = X[d == 1]  # Classe C2
    
    # Encontrar os limites para a malha do plano
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    
    x1_grid, x2_grid = np.meshgrid(np.linspace(x1_min, x1_max, 10),
                                   np.linspace(x2_min, x2_max, 10))
    
    # Configurar frames para animação (cada frame é uma época)
    frames = []
    
    # Estado inicial: apenas os pontos scatter
    fig.add_trace(go.Scatter3d(
        x=class_1[:, 0], y=class_1[:, 1], z=class_1[:, 2],
        mode='markers',
        marker=dict(size=5, color='blue', symbol='circle'),
        name='Classe C1 (-1)'
    ))
    
    fig.add_trace(go.Scatter3d(
        x=class_2[:, 0], y=class_2[:, 1], z=class_2[:, 2],
        mode='markers',
        marker=dict(size=5, color='red', symbol='cross'),
        name='Classe C2 (+1)'
    ))

    # Vamos construir o plano para a primeira época como trace principal do plano
    # A equação do plano separador é: w1*x1 + w2*x2 + w3*x3 - w0 = 0
    # Logo: x3 = (w0 - w1*x1 - w2*x2) / w3
    w = weights_history[0]
    
    def get_plane_z(w_t):
        if abs(w_t[2]) < 1e-6: # Evitar divisão por zero
            z_grid = np.zeros_like(x1_grid)
        else:
            z_grid = (w_t[3] - w_t[0]*x1_grid - w_t[1]*x2_grid) / w_t[2]
        return z_grid
    
    z_init = get_plane_z(weights_history[0])
    fig.add_trace(go.Surface(
        x=x1_grid, y=x2_grid, z=z_init,
        colorscale='Viridis', opacity=0.5,
        name='Plano Separador'
    ))
    
    for epoch, w in enumerate(weights_history):
        z_grid = get_plane_z(w)
        
        frames.append(go.Frame(
            data=[
                go.Scatter3d(x=class_1[:, 0], y=class_1[:, 1], z=class_1[:, 2], mode='markers'),
                go.Scatter3d(x=class_2[:, 0], y=class_2[:, 1], z=class_2[:, 2], mode='markers'),
                go.Surface(x=x1_grid, y=x2_grid, z=z_grid, colorscale='Viridis', opacity=0.5)
            ],
            name=f'Época {epoch}'
        ))
        
    fig.frames = frames
    
    # Configurar o slider e botões
    sliders = [{
        "pad": {"b": 10, "t": 60},
        "len": 0.9,
        "x": 0.1,
        "y": 0,
        "steps": [
            {
                "args": [[f.name], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}],
                "label": str(k),
                "method": "animate"
            } for k, f in enumerate(frames)
        ]
    }]
    
    fig.update_layout(
        title=f"Evolução do Plano Separador - Treinamento {training_index}",
        updatemenus=[{
            "type": "buttons",
            "buttons": [
                {
                    "args": [None, {"frame": {"duration": 500, "redraw": True}, "fromcurrent": True, "transition": {"duration": 300, "easing": "quadratic-in-out"}}],
                    "label": "Play",
                    "method": "animate"
                },
                {
                    "args": [[None], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate", "transition": {"duration": 0}}],
                    "label": "Pause",
                    "method": "animate"
                }
            ],
            "direction": "left",
            "pad": {"r": 10, "t": 87},
            "showactive": False,
            "x": 0.1,
            "y": 0,
            "xanchor": "right",
            "yanchor": "top"
        }],
        sliders=sliders,
        scene=dict(
            xaxis_title='x1',
            yaxis_title='x2',
            zaxis_title='x3'
        )
    )
    
    file_name = f'grafico_treinamento_{training_index}.html'
    fig.write_html(file_name)
    print(f"Gráfico interativo salvo em: {file_name}")
