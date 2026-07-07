import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# MÉTODOS DE INTERPOLACIÓN (Chapra Fig. 18.7 y 18.11)
# =============================================================================

def interpolacion_newton(x, y, xi):
    """
    Polinomio de interpolación de Newton con diferencias divididas.
    (Basado en la Figura 18.7 de Chapra)
    x, y: Arreglos de datos conocidos.
    xi: Valor o arreglo de valores donde se quiere interpolar.
    """
    n = len(x)
    # Crear matriz para la tabla de diferencias divididas
    fdd = np.zeros((n, n))
    fdd[:, 0] = y
    
    # Calcular las diferencias divididas hacia adelante
    for j in range(1, n):
        for i in range(n - j):
            fdd[i, j] = (fdd[i + 1, j - 1] - fdd[i, j - 1]) / (x[i + j] - x[i])
            
    # Función interna para evaluar un solo punto
    def evaluar_punto(x_val):
        xterm = 1.0
        yint_val = fdd[0, 0]
        for order in range(1, n):
            xterm *= (x_val - x[order - 1])
            yint_val += fdd[0, order] * xterm
        return yint_val

    # Si xi es un solo número, retorna un número. Si es un arreglo (para graficar), retorna arreglo.
    if np.isscalar(xi):
        return evaluar_punto(xi)
    else:
        return np.array([evaluar_punto(val) for val in xi])


def interpolacion_lagrange(x, y, xi):
    """
    Polinomio de interpolación de Lagrange.
    (Basado en la Figura 18.11 de Chapra)
    x, y: Arreglos de datos conocidos.
    xi: Valor o arreglo de valores donde se quiere interpolar.
    """
    n = len(x)
    
    def evaluar_punto(x_val):
        sum_val = 0.0
        for i in range(n):
            producto = y[i]
            for j in range(n):
                if i != j:
                    producto *= (x_val - x[j]) / (x[i] - x[j])
            sum_val += producto
        return sum_val

    if np.isscalar(xi):
        return evaluar_punto(xi)
    else:
        return np.array([evaluar_punto(val) for val in xi])

# =============================================================================
# GENERADOR DE GRÁFICAS DE INTERPOLACIÓN
# =============================================================================

def graficar_interpolaciones(x_datos, y_datos, func_original=None, grado_maximo=None):
    """
    Genera las gráficas de los datos, la función original (opcional) y 
    las interpolaciones de Newton y Lagrange.
    """
    plt.figure(figsize=(10, 6))
    
    # Rango suave para graficar las curvas (100 puntos entre el min y max de x)
    x_curva = np.linspace(min(x_datos), max(x_datos), 200)
    
    # 1. Graficar función original si se conoce
    if func_original is not None:
        y_original = func_original(x_curva)
        plt.plot(x_curva, y_original, 'k--', label="Función Original", linewidth=2, alpha=0.7)
        
    # 2. Graficar los puntos de datos reales
    plt.scatter(x_datos, y_datos, color='red', s=80, zorder=5, label="Datos Conocidos")
    
    # Seleccionar cuántos puntos usar según el grado deseado (Grado N requiere N+1 puntos)
    if grado_maximo is None or grado_maximo >= len(x_datos):
        puntos_usar = len(x_datos)
        etiqueta_grado = len(x_datos) - 1
    else:
        puntos_usar = grado_maximo + 1
        etiqueta_grado = grado_maximo
        
    x_subset = x_datos[:puntos_usar]
    y_subset = y_datos[:puntos_usar]
    
    # 3. Calcular y graficar Interpolación de Newton
    y_newton = interpolacion_newton(x_subset, y_subset, x_curva)
    plt.plot(x_curva, y_newton, 'b-', label=f"Newton (Grado {etiqueta_grado})", linewidth=1.5)
    
    # 4. Calcular y graficar Interpolación de Lagrange
    y_lagrange = interpolacion_lagrange(x_subset, y_subset, x_curva)
    plt.plot(x_curva, y_lagrange, 'g:', label=f"Lagrange (Grado {etiqueta_grado})", linewidth=2.5)

    # Configuraciones de la gráfica
    plt.title(f"Comparación de Métodos de Interpolación (Grado {etiqueta_grado})")
    plt.xlabel("Eje X")
    plt.ylabel("Eje Y")
    plt.grid(True, linestyle=":", alpha=0.7)
    plt.legend() # Leyenda debidamente identificada
    plt.show()

    # =============================================================================
# MÉTODOS DE INTEGRACIÓN NUMÉRICA 
# =============================================================================

# -----------------------------------------------------------------------------
# Figura 21.9 de Chapra: Regla del Trapecio
# -----------------------------------------------------------------------------

def trapecio_ap_multiple(h, n, f):
    """
    Fig. 21.9a: Regla del trapecio de aplicación múltiple.
    h: Tamaño del paso.
    n: Número de segmentos (número de puntos es n+1).
    f: Arreglo de valores de la función evaluada en los puntos.
    """
    suma = f[0]
    for i in range(1, n):
        suma += 2 * f[i]
    suma += f[n]
    return h * suma / 2

def trapecio_datos_desiguales(x, y):
    """
    Fig. 21.9b: Regla del trapecio para datos desigualmente espaciados.
    x, y: Arreglos de datos.
    """
    n = len(x)
    suma = 0.0
    for i in range(1, n):
        suma += (x[i] - x[i-1]) * (y[i-1] + y[i]) / 2
    return suma

# -----------------------------------------------------------------------------
# Figura 21.13 de Chapra: Reglas de Simpson
# -----------------------------------------------------------------------------

def simpson_13(h, f0, f1, f2):
    """
    Fig. 21.13a: Regla de Simpson 1/3 para un solo segmento (3 puntos).
    """
    return 2 * h * (f0 + 4 * f1 + f2) / 6

def simpson_38(h, f0, f1, f2, f3):
    """
    Fig. 21.13b: Regla de Simpson 3/8 para un solo segmento (4 puntos).
    """
    return 3 * h * (f0 + 3 * f1 + 3 * f2 + f3) / 8

def simpson_13_multiple(h, n, f):
    """
    Fig. 21.13c: Regla de Simpson 1/3 de aplicación múltiple.
    Requiere que n (número de segmentos) sea par.
    """
    suma = f[0]
    # Suma de los términos impares (multiplicados por 4)
    for i in range(1, n, 2):
        suma += 4 * f[i]
    # Suma de los términos pares (multiplicados por 2)
    for i in range(2, n - 1, 2):
        suma += 2 * f[i]
        
    suma += f[n]
    return h * suma / 3

def simpson_int(x, f):
    """
    Fig. 21.13d: Integración general (Evaluador maestro).
    Determina automáticamente qué método usar dependiendo de la 
    cantidad de segmentos (n) disponibles.
    """
    n = len(x) - 1 # Número de segmentos
    h = x[1] - x[0] # Asume espaciamiento uniforme para Simpson
    
    suma = 0.0
    
    if n == 1:
        # Solo hay 1 segmento, usar Trapecio
        suma = trapecio_ap_multiple(h, n, f)
    else:
        m = n
        odd = (n % 2 != 0) # True si la cantidad de segmentos es impar
        
        if odd and n > 1:
            # Si n es impar, el último tramo de 3 segmentos usa Simpson 3/8
            suma += simpson_38(h, f[n-3], f[n-2], f[n-1], f[n])
            m = n - 3 # Los segmentos restantes para Simpson 1/3
            
        if m > 1:
            # Los segmentos pares restantes usan Simpson 1/3 múltiple
            suma += simpson_13_multiple(h, m, f)
            
    return suma