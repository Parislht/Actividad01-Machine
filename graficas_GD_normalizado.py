import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def gradiente_descendente(X, Y, lr, epochs):
    m = len(Y)
    w, b = 0.0, 0.0
    
    for _ in range(epochs):
        # f(x) = wx + b
        y_pred = (w * X) + b
        # Derivadas (gradientes)
        w -= lr * ((1 / m) * np.sum((y_pred - Y) * X))
        b -= lr * ((1 / m) * np.sum(y_pred - Y))
        
    return w, b

def main():
    # Hiperparámetros matemáticamente estables para Z-score
    iteraciones = 1000
    tasa_paso = 0.01

    # Carga y normalización directa 
    df = pd.read_csv('caso2_notas.csv')
    df_norm = (df - df.mean()) / df.std()
    
    Y = df_norm.iloc[:, -1].values
    columnas_X = df_norm.columns[:-1]
    
    # 4 gráficos detallados
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    print(f"--- Entrenamiento iniciado | LR: {tasa_paso} | Epochs: {iteraciones} ---")
    
    for i, col in enumerate(columnas_X):
        X = df_norm[col].values
        
        # Ejecutar Gradiente
        w, b = gradiente_descendente(X, Y, lr=tasa_paso, epochs=iteraciones)
        print(f"{col:20} -> w (peso): {w:.4f} | b (sesgo): {b:.4f}")
        
        # Visualización
        axes[i].scatter(X, Y, alpha=0.6, color='#2980b9', edgecolor='black', label='Datos Reales')
        x_line = np.linspace(X.min(), X.max(), 100)
        y_line = (w * x_line) + b
        axes[i].plot(x_line, y_line, color='#c0392b', linewidth=3, label=f'Regresión: {w:.2f}x + {b:.2f}')
        
        axes[i].set_title(f'Análisis Univariado: {col.replace("_", " ").title()}', fontsize=12, fontweight='bold')
        axes[i].set_xlabel(f'{col} (Z-score)')
        axes[i].set_ylabel('Nota Final (Z-score)')
        axes[i].legend()
        axes[i].grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()