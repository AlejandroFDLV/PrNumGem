import numpy as np
from tabulate import tabulate

# =============================================================================
# MÉTODOS PARA SISTEMAS DE ECUACIONES LINEALES
# =============================================================================

def gauss_pivoteo_parcial(A, b):
    """
    Resuelve el sistema Ax = b utilizando Eliminación de Gauss con 
    pivoteo parcial por máximo de columna.
    """
    # Convertimos a float64 para evitar truncamiento y errores de redondeo de enteros
    A = np.array(A, dtype=np.float64)
    b = np.array(b, dtype=np.float64)
    n = len(b)
    
    # Crear la matriz aumentada [A|b]
    Ab = np.concatenate((A, b.reshape(n, 1)), axis=1)
    
    print("\n--- MATRIZ AUMENTADA INICIAL ---")
    print(np.round(Ab, 4))
    
    # FASE DE ELIMINACIÓN HACIA ADELANTE
    for i in range(n):
        # 1. Pivoteo Parcial: Buscar el máximo en la columna actual (desde la fila i hacia abajo)
        max_row = np.argmax(np.abs(Ab[i:n, i])) + i
        
        # Intercambiar la fila actual con la fila del valor máximo
        if i != max_row:
            Ab[[i, max_row]] = Ab[[max_row, i]]
            print(f"\n[Pivoteo] Se intercambió la fila {i+1} con la fila {max_row+1}:")
            print(np.round(Ab, 4))
            
        # Validación de matriz singular (división por cero)
        if abs(Ab[i, i]) < 1e-12:
            print("\n[Error] El sistema no tiene solución única (la matriz es singular o casi singular).")
            return None
            
        # 2. Eliminación
        for j in range(i + 1, n):
            factor = Ab[j, i] / Ab[i, i]
            Ab[j, i:] = Ab[j, i:] - factor * Ab[i, i:]
            
        if i < n - 1:
            print(f"\n--- Eliminación debajo del pivote en columna {i+1} ---")
            print(np.round(Ab, 4))
            
    # FASE DE SUSTITUCIÓN HACIA ATRÁS
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        suma_conocidos = np.dot(Ab[i, i+1:n], x[i+1:n])
        x[i] = (Ab[i, -1] - suma_conocidos) / Ab[i, i]
        
    print("\n[Éxito] Eliminación de Gauss finalizada.")
    print("Vector solución x:")
    # Formateo de salida elegante
    solucion = [[f"x{k+1}", val] for k, val in enumerate(x)]
    print(tabulate(solucion, headers=["Variable", "Valor"], floatfmt=".6f", tablefmt="fancy_grid"))
    
    return x


def gauss_seidel(A, b, tol, x0=None):
    """
    Resuelve el sistema Ax = b utilizando el método iterativo de Gauss-Seidel.
    Incluye un máximo de 150 iteraciones como exigen las instrucciones.
    """
    A = np.array(A, dtype=np.float64)
    b = np.array(b, dtype=np.float64)
    n = len(b)
    
    # Si el usuario no da un vector inicial, asumimos ceros
    if x0 is None:
        x = np.zeros(n)
    else:
        x = np.array(x0, dtype=np.float64)
        
    iteraciones = []
    
    # Verificación opcional de convergencia (Dominancia diagonal)
    diagonal = np.diag(np.abs(A))
    suma_resto = np.sum(np.abs(A), axis=1) - diagonal
    if np.any(diagonal < suma_resto):
        print("\n[Advertencia] La matriz no es estrictamente diagonalmente dominante.")
        print("El método de Gauss-Seidel podría no converger.\n")
        
    # Bucle principal (Máximo 150 iteraciones según el punto 1.a)
    for k in range(1, 151):
        x_ant = x.copy()
        
        # Calcular los nuevos valores de x
        for i in range(n):
            suma = 0
            for j in range(n):
                if j != i:
                    suma += A[i, j] * x[j] # Usa los valores más actualizados de x
            x[i] = (b[i] - suma) / A[i, i]
            
        # Calcular error relativo para cada variable y tomar el máximo
        errores = np.zeros(n)
        for i in range(n):
            if x[i] != 0:
                errores[i] = abs((x[i] - x_ant[i]) / x[i])
            else:
                errores[i] = 0
                
        error_max = np.max(errores)
        
        # Guardamos la data para la tabla (Iteración, variables x1..xn, error)
        fila_tabla = [k] + list(x) + [error_max]
        iteraciones.append(fila_tabla)
        
        # Criterio de parada
        if error_max <= tol:
            print(f"\n[Éxito] Convergencia por error aceptable en la iteración {k}.")
            break
    else:
        # Se ejecuta si se agotan las 150 iteraciones sin el 'break'
        print(f"\n[Aviso] Se alcanzó el número máximo de iteraciones (150).")
        
    # Preparar el encabezado dinámico para la tabla
    headers = ["Iteración"] + [f"x{i+1}" for i in range(n)] + ["Error Relativo Max"]
    print("\n" + tabulate(iteraciones, headers=headers, floatfmt=".8f", tablefmt="fancy_grid"))
    
    print("\nVector solución convergente:")
    solucion = [[f"x{i+1}", val] for i, val in enumerate(x)]
    print(tabulate(solucion, headers=["Variable", "Valor"], floatfmt=".6f", tablefmt="fancy_grid"))
    
    return x