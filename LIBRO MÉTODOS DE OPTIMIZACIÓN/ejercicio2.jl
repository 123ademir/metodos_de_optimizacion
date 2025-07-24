# EJERCICIO 2: Maximización de Ingresos y Utilidades
println("===== EJERCICIO 2: Maximización de Ingresos y Utilidades =====")
# Funciones
q(p) = 500 - 0.4p
I(p) = p * q(p)
U(p) = (p - 600) * q(p)
# Encontrar máximos
p_ingreso = 625
p_utilidad = 925
println("Precio para máximo ingreso: $", p_ingreso)
println("Precio para máxima utilidad: $", p_utilidad)
println("Utilidad máxima: $", U(p_utilidad))
println()