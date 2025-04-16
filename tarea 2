import numpy as np

def resolver_sistema():
    print(" Sistema de Ecuaciones Lineales con NumPy")
    
    n = int(input("Ingrese el número de incógnitas: "))
    
    A = []
    b = []

    print("Ingrese los coeficientes de cada ecuación:")
    for i in range(n):
        fila = list(map(float, input(f"Coeficientes de la ecuación {i+1} separados por espacio: ").split()))
        A.append(fila)
    
    print("Ingrese los términos independientes:")
    b = list(map(float, input("Separados por espacio: ").split()))

    A = np.array(A)
    b = np.array(b)

    try:
        solucion = np.linalg.solve(A, b)
        print("\n Solución del sistema:")
        for i in range(n):
            print(f"x{i+1} = {solucion[i]}")
    except np.linalg.LinAlgError as e:
        print("\n El sistema no tiene solución única:")
        print(str(e))

# Ejecutar el programa
resolver_sistema()
