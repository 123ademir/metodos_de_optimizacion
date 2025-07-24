# Análisis de casos
x1_prod, y1_prod = 40, 20
U1_prod = 40*x1_prod + 30*y1_prod
x2_prod, y2_prod = 0, 100
U2_prod = 40*x2_prod + 30*y2_prod
println("Caso 1 (x=40, y=20): U = \$", U1_prod)
println("Caso 2 (x=0, y=100): U = \$", U2_prod)
println("Óptimo: producir ", x2_prod, " de A y ", y2_prod, " de B")