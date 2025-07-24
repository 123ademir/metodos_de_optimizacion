# Parámetros
w = 0.5
σ_A, σ_B = 0.12, 0.08
ρ = 0.3
# Riesgo del portafolio
σ_p = sqrt(w^2*σ_A^2 + (1-w)^2*σ_B^2 + 2*w*(1-w)*ρ*σ_A*σ_B)
println("Proporción en A: ", w*100, "%")
println("Proporción en B: ", (1-w)*100, "%")
println("Riesgo del portafolio: ", round(σ_p*100, digits=1), "%")