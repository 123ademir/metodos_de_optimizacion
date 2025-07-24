# Parámetros
D = 10000  # demanda anual
S = 50     # costo por pedido
H = 4      # costo de mantener
# EOQ
EOQ = sqrt(2*D*S/H)
N = D/EOQ
CT_min = sqrt(2*D*S*H)
println("EOQ: ", EOQ, " unidades")
println("Número de pedidos: ", N, " por año")
println("Costo total mínimo: \$", CT_min)