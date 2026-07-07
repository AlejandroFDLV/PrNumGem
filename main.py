import utils
from raices import metodo_biseccion, metodo_newton, metodo_secante
from sistemas_ecuaciones import gauss_pivoteo_parcial, gauss_seidel
from interpolacion_integracion import graficar_interpolaciones, simpson_int, trapecio_datos_desiguales

def menu_raices():
    utils.limpiar_pantalla()
    print("=== 1. CÁLCULO DE RAÍCES ===")
    print("1. Método de Bisección")
    print("2. Método de Newton")
    print("3. Método de la Secante")
    print("4. Volver al Menú Principal")
    
    opc = utils.leer_entero("\nSeleccione un método (1-4): ")
    if opc == 4: return

    print("\n--- INGRESO DE DATOS ---")
    print("Ejemplo de función: x**3 - x - 2  |  sin(x) - x/2  |  exp(-x) - x")
    func_str = input("Ingrese la función f(x): ")
    tol = utils.leer_flotante("Ingrese la tolerancia (ej. 0.0001): ")

    if opc == 1:
        xl = utils.leer_flotante("Ingrese el límite inferior (xl): ")
        xu = utils.leer_flotante("Ingrese el límite superior (xu): ")
        print("\nCalculando...")
        metodo_biseccion(func_str, xl, xu, tol)

    elif opc == 2:
        x0 = utils.leer_flotante("Ingrese el valor inicial (x0): ")
        dom_min = utils.leer_flotante("Ingrese límite inferior del dominio para la gráfica: ")
        dom_max = utils.leer_flotante("Ingrese límite superior del dominio para la gráfica: ")
        print("\nCalculando...")
        metodo_newton(func_str, x0, tol, dom_min, dom_max)

    elif opc == 3:
        x0 = utils.leer_flotante("Ingrese el primer valor inicial (x0): ")
        x1 = utils.leer_flotante("Ingrese el segundo valor inicial (x1): ")
        dom_min = utils.leer_flotante("Ingrese límite inferior del dominio para la gráfica: ")
        dom_max = utils.leer_flotante("Ingrese límite superior del dominio para la gráfica: ")
        print("\nCalculando...")
        metodo_secante(func_str, x0, x1, tol, dom_min, dom_max)
        
    utils.pausar()

def menu_sistemas():
    utils.limpiar_pantalla()
    print("=== 2. SISTEMAS DE ECUACIONES LINEALES ===")
    print("1. Gauss con Pivoteo Parcial")
    print("2. Gauss-Seidel")
    print("3. Volver al Menú Principal")
    
    opc = utils.leer_entero("\nSeleccione un método (1-3): ")
    if opc == 3: return

    n = utils.leer_entero("\nIngrese el tamaño del sistema (n): ", minimo=2)
    A, b = utils.leer_matriz_y_vector(n)

    if opc == 1:
        print("\nCalculando Eliminación de Gauss...")
        gauss_pivoteo_parcial(A, b)
    elif opc == 2:
        tol = utils.leer_flotante("Ingrese la tolerancia para Gauss-Seidel (ej. 0.001): ")
        print("\nCalculando Gauss-Seidel...")
        gauss_seidel(A, b, tol)
        
    utils.pausar()

def menu_interpolacion_integracion():
    utils.limpiar_pantalla()
    print("=== 3. INTERPOLACIÓN E INTEGRACIÓN ===")
    print("1. Gráficas de Interpolación (Newton y Lagrange)")
    print("2. Integración Numérica (Trapecio / Simpson)")
    print("3. Volver al Menú Principal")
    
    opc = utils.leer_entero("\nSeleccione un método (1-3): ")
    if opc == 3: return

    n = utils.leer_entero("\nIngrese la cantidad de puntos de datos conocidos: ", minimo=2)
    x_datos, y_datos = utils.leer_datos_xy(n)

    if opc == 1:
        grado = utils.leer_entero(f"Ingrese el grado de interpolación máximo deseado (1 a {n-1}): ", minimo=1)
        
        # AJUSTE EXIGIDO: Preguntar por la función original si se conoce
        conoce_func = input("\n¿Conoce la función original f(x)? (s/n): ").strip().lower()
        func_original = None
        
        if conoce_func == 's':
            import sympy as sp
            print("Ejemplo de función original: cos(x) | x**2 | exp(-x)")
            func_str = input("Ingrese la función original f(x): ")
            try:
                x = sp.Symbol('x')
                f_sym = sp.sympify(func_str)
                func_original = sp.lambdify(x, f_sym, 'numpy')
            except Exception as e:
                print(f"[Advertencia] Error al interpretar la función: {e}")
                print("Se procederá a graficar únicamente los puntos y las interpolaciones.")
                func_original = None
        
        print("\nGenerando gráficas...")
        # Pasamos el parámetro func_original (puede ser None si no se conoce)
        graficar_interpolaciones(x_datos, y_datos, func_original=func_original, grado_maximo=grado)
    
    elif opc == 2:
        print("\nCalculando Integrales...")
        integral_simpson = simpson_int(x_datos, y_datos)
        integral_trapecio = trapecio_datos_desiguales(x_datos, y_datos)
        
        print(f"\nResultado usando el evaluador de Simpson: {integral_simpson:.6f}")
        print(f"Resultado usando Trapecio (datos desiguales): {integral_trapecio:.6f}")

    utils.pausar()

def main():
    while True:
        utils.limpiar_pantalla()
        print("=================================================")
        print("     PROYECTO DE PROGRAMACIÓN NUMÉRICA 2026      ")
        print("=================================================")
        print("1. Cálculo de Raíces de Funciones")
        print("2. Resolución de Sistemas de Ecuaciones Lineales")
        print("3. Interpolación e Integración Numérica")
        print("4. Salir")
        
        opcion = utils.leer_entero("\nSeleccione una opción (1-4): ")
        
        if opcion == 1:
            menu_raices()
        elif opcion == 2:
            menu_sistemas()
        elif opcion == 3:
            menu_interpolacion_integracion()
        elif opcion == 4:
            print("\n¡Programa finalizado exitosamente!")
            break
        else:
            print("[Error] Opción no válida.")
            utils.pausar()

if __name__ == "__main__":
    main()