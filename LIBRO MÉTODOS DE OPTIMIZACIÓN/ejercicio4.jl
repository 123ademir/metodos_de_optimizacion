# Función de costo
C(y) = 2 * (sqrt(100 + y^2) + sqrt(225 + (30-y)^2))
# Encontrar el mínimo
y_opt = 12  # Solución analítica
costo_min = C(y_opt)
println("Ubicación óptima de B: y = ", y_opt, " km")
println("Costo mínimo: \$", round(costo_min, digits=2))