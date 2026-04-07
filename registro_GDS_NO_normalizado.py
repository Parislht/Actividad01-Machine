def predecir(x, w, b):
    return w * x + b

def calcular_mse(datos, w, b):
    n = len(datos)
    total = sum((y - predecir(x, w, b))**2 for x, y in datos)
    return total / n

def gradiente_una_muestra(x, y, w, b):
    error = y - predecir(x, w, b)
    grad_w = -2 * x * error
    grad_b = -2 * error
    return grad_w, grad_b

def solucion_analitica(datos):
    n = len(datos)
    sx  = sum(x for x, y in datos)
    sy  = sum(y for x, y in datos)
    sxy = sum(x*y for x, y in datos)
    sx2 = sum(x**2 for x, y in datos)

    m = (n * sxy - sx * sy) / (n * sx2 - sx**2)
    b = (sy - m * sx) / n
    return m, b

def separador(titulo):
    linea = "=" * 60
    print(f"\n{linea}")
    print(f"  {titulo}")
    print(linea)

# ===========================================================================
# STOCHASTIC GRADIENT DESCENT
# ===========================================================================

def stochastic_gradient_descent(datos, lr=0.05, epocas=10, verbose=True, mostrar_cada=1):
    w, b = 0.0, 0.0
    historial = []
    random.seed(42)

    if verbose:
        print(f"  lr={lr}  |  N={len(datos)}  |  epocas={epocas}")
        print(f"  {'Epoca':>6}  {'w':>10}  {'b':>10}  {'MSE':>12}")
        print(f"  {'-'*6}  {'-'*10}  {'-'*10}  {'-'*12}")

    for epoca in range(epocas):
        mse = calcular_mse(datos, w, b)
        historial.append((epoca, w, b, mse))

        if verbose and (epoca % mostrar_cada == 0 or epoca < 3):
            print(f"  {epoca:>6}  {w:>10.6f}  {b:>10.6f}  {mse:>12.4f}")

        indices = list(range(len(datos)))
        random.shuffle(indices)

        for idx in indices:
            x_i, y_i = datos[idx]
            gw, gb = gradiente_una_muestra(x_i, y_i, w, b)
            w = w - lr * gw
            b = b - lr * gb

    mse_final = calcular_mse(datos, w, b)
    historial.append((epocas, w, b, mse_final))

    if verbose:
        print(f"  {epocas:>6}  {w:>10.6f}  {b:>10.6f}  {mse_final:>12.4f}  <-- fin")
        print(f"\n  Resultado SGD: w = {w:.6f}  |  b = {b:.6f}")
        print(f"  MSE final    : {mse_final:.4f}")

    return w, b, historial

# ===========================================================================
# EJECUCIÓN CON EL CSV
# ===========================================================================

if __name__ == "__main__":
    print("  APLICANDO SGD A CASO2_NOTAS.CSV (SIN ESTANDARIZAR)")

    # 1. Cargar el dataset
    df = pd.read_csv('caso2_notas.csv')

    features = ['horas_estudio', 'horas_suenio', 'asistencia_pct', 'ejercicios_resueltos']
    target = 'nota_final ( Y )'

    # 2. Iterar por cada variable
    for feature in features:
        separador(f"ANALIZANDO: {feature.upper()} vs NOTA FINAL")
        
        # Preparar los datos para esta característica
        datos = list(zip(df[feature], df[target]))
        
        # Asignar un Learning Rate seguro según la escala de la variable
        if feature == 'ejercicios_resueltos':
            lr_seguro = 0.00001
            epocas = 10
        elif feature == 'asistencia_pct':
            lr_seguro = 0.0001
            epocas = 10
        else: 
            lr_seguro = 0.005
            epocas = 10

        # 4. Solución Analítica
        w_ideal, b_ideal = solucion_analitica(datos)
        mse_ideal = calcular_mse(datos, w_ideal, b_ideal)
        print(f"Meta Ideal (Analítica) -> w: {w_ideal:.6f} | b: {b_ideal:.6f} | MSE: {mse_ideal:.4f}\n")

        # 5. Entrenar con SGD
        stochastic_gradient_descent(datos, lr=lr_seguro, epocas=epocas, mostrar_cada=epocas//5)