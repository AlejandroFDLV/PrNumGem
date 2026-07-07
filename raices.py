import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from tabulate import tabulate

# =============================================================================
# FUNCIONES AUXILIARES PARA GRÁFICOS
# =============================================================================

def configurar_grafica(f_num, x_min, x_max, titulo):
    """
    Configura la gráfica base (la curva de la función y el eje X) antes de 
    empezar a graficar los iterados.
    """
    x_vals = np.linspace(x_min, x_max, 400)
    y_vals = f_num(x_vals)
    
    plt.ion() # Activa el modo interactivo para animar los iterados
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(x_vals, y_vals, label="f(x)", color="blue", zorder=1)
    ax.axhline(0, color="black", linewidth=1, linestyle="--", zorder=2) # Eje X
    
    ax.set_title(titulo)
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.grid(True, linestyle=":", alpha=0.7)
    
    return fig, ax

def finalizar_grafica(ax, x_raiz, f_raiz):
    """
    Coloca el punto final (la raíz encontrada), actualiza la leyenda y 
    pausa la gráfica para que el usuario pueda ver el resultado.
    """
    ax.scatter(x_raiz, f_raiz, color="green", s=100, label="Raíz obtenida", zorder=5)
    ax.legend()
    plt.ioff() # Apaga el modo interactivo
    plt.show() # Muestra la gráfica final estática

# =============================================================================
# MÉTODOS NUMÉRICOS
# =============================================================================

def metodo_biseccion(func_str, xl, xu, tol):
    # 1. Parsear la función con SymPy
    x = sp.Symbol('x')
    f_sym = sp.sympify(func_str)
    f = sp.lambdify(x, f_sym, 'numpy') # Convierte a función evaluable por NumPy

    # 2. Validar cambio de signo
    if f(xl) * f(xu) >= 0:
        print("Error: f(xl) y f(xu) deben tener signos opuestos en el intervalo dado.")
        return

    # 3. Configurar gráfica (El dominio es directamente [xl, xu] como pide el proyecto)
    fig, ax = configurar_grafica(f, xl, xu, f"Método de Bisección: {func_str}")
    
    iteraciones = []
    x_ant = xl
    
    # 4. Bucle principal (Máximo 150 iteraciones)
    for i in range(1, 151):
        xr = (xl + xu) / 2
        fxr = f(xr)
        
        # Cálculo del error relativo aproximado
        error = abs((xr - x_ant) / xr) if xr != 0 else 0
        
        iteraciones.append([i, xr, fxr, error])
        
        # Graficar iterado (Color rojo)
        label_iter = "Iterados" if i == 1 else "" # Para no repetir en la leyenda
        ax.scatter(xr, fxr, color="red", label=label_iter, zorder=4)
        plt.pause(0.5) # Pausa para visualizar la "evolución gráfica"
        
        # 5. Criterio de parada: Error <= tol o f(x) <= tol
        if error <= tol or abs(fxr) <= tol:
            print(f"\n[Éxito] Convergencia por error aceptable en la iteración {i}.")
            print(f"Valor convergente: x = {xr} | Evaluación f(x) = {fxr}")
            break
            
        # Reasignación de límites
        if f(xl) * fxr < 0:
            xu = xr
        else:
            xl = xr
            
        x_ant = xr
    else:
        # Se ejecuta si el bucle termina sin el 'break' (Llegó a 150)
        print("\n[Aviso] Se alcanzó el número máximo de iteraciones (150).")
        print(f"Último valor obtenido: x = {xr} | Evaluación f(x) = {fxr}")
        
    # Imprimir tabla y finalizar gráfico
    print("\n" + tabulate(iteraciones, headers=["Iteración", "x", "f(x)", "Error Relativo"], floatfmt=".8f", tablefmt="fancy_grid"))
    finalizar_grafica(ax, xr, fxr)


def metodo_newton(func_str, x0, tol, dom_min, dom_max):
    # 1. Parsear función y calcular la derivada automáticamente
    x = sp.Symbol('x')
    f_sym = sp.sympify(func_str)
    df_sym = sp.diff(f_sym, x) 
    
    f = sp.lambdify(x, f_sym, 'numpy')
    df = sp.lambdify(x, df_sym, 'numpy')
    
    # 2. Configurar gráfica (Dominio dado por el usuario)
    fig, ax = configurar_grafica(f, dom_min, dom_max, f"Método de Newton: {func_str}")
    
    iteraciones = []
    xi = x0
    
    for i in range(1, 151):
        fxi = f(xi)
        dfxi = df(xi)
        
        # Evitar división por cero
        if dfxi == 0:
            print("Error matemático: La derivada es cero, el método diverge.")
            break
            
        xi_sig = xi - (fxi / dfxi)
        error = abs((xi_sig - xi) / xi_sig) if xi_sig != 0 else 0
        
        iteraciones.append([i, xi_sig, f(xi_sig), error])
        
        label_iter = "Iterados" if i == 1 else ""
        ax.scatter(xi_sig, f(xi_sig), color="red", label=label_iter, zorder=4)
        plt.pause(0.5)
        
        if error <= tol or abs(f(xi_sig)) <= tol:
            print(f"\n[Éxito] Convergencia por error aceptable en la iteración {i}.")
            print(f"Valor convergente: x = {xi_sig} | Evaluación f(x) = {f(xi_sig)}")
            break
            
        xi = xi_sig
    else:
        print("\n[Aviso] Se alcanzó el número máximo de iteraciones (150).")
        print(f"Último valor obtenido: x = {xi_sig} | Evaluación f(x) = {f(xi_sig)}")
        
    print("\n" + tabulate(iteraciones, headers=["Iteración", "x", "f(x)", "Error Relativo"], floatfmt=".8f", tablefmt="fancy_grid"))
    finalizar_grafica(ax, xi_sig, f(xi_sig))


def metodo_secante(func_str, x0, x1, tol, dom_min, dom_max):
    # 1. Parsear función
    x = sp.Symbol('x')
    f_sym = sp.sympify(func_str)
    f = sp.lambdify(x, f_sym, 'numpy')
    
    # 2. Configurar gráfica (Dominio dado por el usuario)
    fig, ax = configurar_grafica(f, dom_min, dom_max, f"Método de la Secante: {func_str}")
    
    iteraciones = []
    x_ant = x0
    x_act = x1
    
    for i in range(1, 151):
        fx_ant = f(x_ant)
        fx_act = f(x_act)
        
        # Evitar división por cero
        if (fx_act - fx_ant) == 0:
            print("Error matemático: División por cero generada por f(xi) = f(xi-1).")
            break
            
        x_sig = x_act - (fx_act * (x_ant - x_act)) / (fx_ant - fx_act)
        error = abs((x_sig - x_act) / x_sig) if x_sig != 0 else 0
        
        iteraciones.append([i, x_sig, f(x_sig), error])
        
        label_iter = "Iterados" if i == 1 else ""
        ax.scatter(x_sig, f(x_sig), color="red", label=label_iter, zorder=4)
        plt.pause(0.5)
        
        if error <= tol or abs(f(x_sig)) <= tol:
            print(f"\n[Éxito] Convergencia por error aceptable en la iteración {i}.")
            print(f"Valor convergente: x = {x_sig} | Evaluación f(x) = {f(x_sig)}")
            break
            
        x_ant = x_act
        x_act = x_sig
    else:
        print("\n[Aviso] Se alcanzó el número máximo de iteraciones (150).")
        print(f"Último valor obtenido: x = {x_sig} | Evaluación f(x) = {f(x_sig)}")
        
    print("\n" + tabulate(iteraciones, headers=["Iteración", "x", "f(x)", "Error Relativo"], floatfmt=".8f", tablefmt="fancy_grid"))
    finalizar_grafica(ax, x_sig, f(x_sig))