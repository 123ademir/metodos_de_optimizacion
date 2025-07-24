# Funciones
Ventas(x) = 200 + 30x - 0.5x^2
UN(x) = 0.25*Ventas(x) - x
# Encontrar el máximo
x_opt = 26
utilidad_max = UN(x_opt)
println("Gasto óptimo en publicidad: \$", x_opt, "000")
println("Utilidad neta máxima: \$", utilidad_max, "000")