import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import warnings
warnings.filterwarnings('ignore')

STYLE = {
    'data_color': '#2563eb',
    'line_color': '#e05000',
    'bg':         '#fafafa',
}

df = pd.read_csv('caso2_notas.csv')

FEATURES = ['horas_estudio', 'horas_suenio', 'asistencia_pct', 'ejercicios_resueltos']
TARGET = 'nota_final ( Y )'
Y_DATA = df[TARGET].values

def calcular_mse(y_real, y_pred):
    return np.mean((y_real - y_pred)**2) / 2

def sgd_crudo(x, y, epocas, lr):
    """Ejecuta SGD directamente sobre los datos crudos."""
    x_array = np.array(x)
    n = len(x_array)
    
    # Inicialización
    w, b = 0.0, 0.0 
    
    if epocas == 0:
        mse = calcular_mse(y, (0.0 * x_array) + b)
        return 0.0, b, mse
    
    np.random.seed(42) 
    for _ in range(epocas):
        indices = np.random.permutation(n)
        for i in indices:
            xi = x_array[i]
            yi = y[i]
            
            # Cálculo del error y actualización pura
            error = (w * xi + b) - yi
            w = w - lr * error * xi
            b = b - lr * error
            
            # Evitar overflow (explosión del gradiente)
            if np.isnan(w) or np.isinf(w):
                return np.nan, np.nan, np.nan
            
    y_pred = w * x_array + b
    mse = calcular_mse(y, y_pred)
    
    return w, b, mse

def main():
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.patch.set_facecolor(STYLE['bg'])
    fig.suptitle('Entrenamiento SGD CRUDO (Sin Estandarizar)', 
                 fontsize=14, fontweight='bold', y=0.98)
    axes = axes.flatten()
    
    lineas_grafico = []
    x_lines = []
    
    for i, feature in enumerate(FEATURES):
        x = df[feature].values
        ax = axes[i]
        ax.scatter(x, Y_DATA, color=STYLE['data_color'], alpha=0.6, s=30)
        
        x_rango = np.linspace(x.min(), x.max(), 100)
        w0, b0, mse0 = sgd_crudo(x, Y_DATA, epocas=0, lr=0.0001)
        y_rango = w0 * x_rango + b0
        
        linea, = ax.plot(x_rango, y_rango, color=STYLE['line_color'], lw=2.5)
        ax.set_title(f'{feature}\nMSE: {mse0:.2f}  |  y = {w0:.2f}x + {b0:.2f}', fontsize=10)
        ax.set_xlabel(feature, fontsize=9)
        ax.set_ylabel('Nota', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        lineas_grafico.append(linea)
        x_lines.append(x_rango)

    plt.subplots_adjust(bottom=0.20, hspace=0.4, wspace=0.2)
    
    ax_epocas = plt.axes([0.2, 0.08, 0.6, 0.03], facecolor='#f0f0f0')
    ax_lr     = plt.axes([0.2, 0.03, 0.6, 0.03], facecolor='#f0f0f0')
    
    # El learning rate se ajusta a una escala logarítmica para mejor control
    slider_epocas = Slider(ax_epocas, 'Épocas', 0, 100, valinit=0, valstep=1, color='#2563eb')
    slider_lr     = Slider(ax_lr, 'Learning Rate (lr)', 0.00001, 0.01, valinit=0.0001, valstep=0.00001, color='#c05000')

    def update(val):
        ep = int(slider_epocas.val)
        lr = slider_lr.val
        
        for i, feature in enumerate(FEATURES):
            x = df[feature].values
            w, b, mse = sgd_crudo(x, Y_DATA, ep, lr)
            
            if np.isnan(w):
                # Si explota, se muestra un error
                lineas_grafico[i].set_ydata(np.nan * x_lines[i])
                axes[i].set_title(f'{feature}\nEXPLOSIÓN DE GRADIENTE (Usa un lr menor)', color='red', fontsize=10)
            else:
                y_nueva = w * x_lines[i] + b
                lineas_grafico[i].set_ydata(y_nueva)
                axes[i].set_title(f'{feature}\nMSE: {mse:.2f}  |  y = {w:.2f}x + {b:.2f}', color='black', fontsize=10)
            
        fig.canvas.draw_idle()

    slider_epocas.on_changed(update)
    slider_lr.on_changed(update)
    plt.show()

if __name__ == '__main__':
    main()