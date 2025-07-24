# Análisis de opciones
opciones = [
    ("A(4)+B(1)", 1100),
    ("A(4)+C(1)", 1050),
    ("B(3)+C(2)", 1400),
    ("A(1)+C(4)", 1200)
]
# Encontrar el mínimo
min_costo = minimum(x[2] for x in opciones)
mejor_opcion = [x for x in opciones if x[2] == min_costo][1]
println("Mejor opción: ", mejor_opcion[1])
println("Costo de aceleración: \$", mejor_opcion[2])
println("Ahorro vs penalización: \$", 2500 - mejor_opcion[2])