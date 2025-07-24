# Función de costo
C(r) = 0.04π * r^2 + 7.1/r
# Encontrar el mínimo
r_opt = (7.1/(0.08π))^(1/3)
h_opt = 355/(π * r_opt^2)
println("Radio óptimo: ", round(r_opt, digits=2), " cm")
println("Altura óptima: ", round(h_opt, digits=2), " cm")
println("Costo mínimo: \$", round(C(r_opt), digits=4))