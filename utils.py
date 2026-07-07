import os

# =============================================================================
# FUNCIONES DE INTERFAZ Y LIMPIEZA
# =============================================================================

def limpiar_pantalla():
    """Limpia la consola dependiendo del sistema operativo (Windows o Linux/Mac)."""
    os.system('cls' if os.name == 'nt' else 'clear')

def pausar():
    """Pausa la ejecución para que el usuario pueda leer los resultados."""
    input("\nPresiona ENTER para continuar...")

# =============================================================================
# FUNCIONES DE VALIDACIÓN DE ENTRADAS (Anti-Errores)
# =============================================================================

def leer_flotante(mensaje):
    """Obliga al usuario a ingresar un número real válido."""
    while True:
        try:
            valor = float(input(mensaje))
            return valor
        except ValueError:
            print("[Error] Por favor, ingresa un número válido (ej. 3.14 o -5).")

def leer_entero(mensaje, minimo=None):
    """Obliga al usuario a ingresar un número entero válido."""
    while True:
        try:
            valor = int(input(mensaje))
            if minimo is not None and valor < minimo:
                print(f"[Error] El valor debe ser mayor o igual a {minimo}.")
                continue
            return valor
        except ValueError:
            print("[Error] Por favor, ingresa un número entero válido (sin decimales).")

# =============================================================================
# FUNCIONES PARA LEER ESTRUCTURAS DE DATOS (Vectores y Matrices)
# =============================================================================

def leer_vector(n, nombre_vector=""):
    """Lee un vector de tamaño n ingresado por el usuario."""
    vector = []
    print(f"\n--- Ingreso de datos para {nombre_vector} ({n} elementos) ---")
    for i in range(n):
        val = leer_flotante(f"Ingrese el valor [{i+1}]: ")
        vector.append(val)
    return vector

def leer_matriz_y_vector(n):
    """
    Lee una matriz cuadrada A de tamaño n x n y un vector b de tamaño n.
    Ideal para los métodos de Gauss y Gauss-Seidel.
    """
    A = []
    b = []
    print(f"\n--- Ingreso de Matriz de Coeficientes A ({n}x{n}) ---")
    for i in range(n):
        fila = []
        for j in range(n):
            val = leer_flotante(f"Ingrese A[{i+1}][{j+1}]: ")
            fila.append(val)
        A.append(fila)
        
    print(f"\n--- Ingreso de Vector de Términos Independientes b ({n} elementos) ---")
    for i in range(n):
        val = leer_flotante(f"Ingrese b[{i+1}]: ")
        b.append(val)
        
    return A, b

def leer_datos_xy(n):
    """
    Lee pares de datos (x, y) para métodos de interpolación e integración.
    """
    x = []
    y = []
    print(f"\n--- Ingreso de pares de datos (X, Y) para {n} puntos ---")
    for i in range(n):
        print(f"Punto {i+1}:")
        val_x = leer_flotante("  Valor de X: ")
        val_y = leer_flotante("  Valor de Y (o f(x)): ")
        x.append(val_x)
        y.append(val_y)
    return x, y