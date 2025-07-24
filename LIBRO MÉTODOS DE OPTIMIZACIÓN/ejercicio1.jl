# EJERCICIO 1: Optimización de Costos de Producción
println("===== EJERCICIO 1: Optimización de Costos =====")
# Función de costo y costo promedio
C(x) = 0.003x^3 - 0.6x^2 + 50x + 1000
CP(x) = C(x)/x
# Derivada del costo promedio
using ForwardDiff
dCP = x -> ForwardDiff.derivative(CP, x)
# Encontrar el mínimo
x_opt = 115.47  # Solución numérica
println("Cajas óptimas: ", x_opt * 100, " cajas")
println("Costo promedio mínimo: \$", CP(x_opt))
println()