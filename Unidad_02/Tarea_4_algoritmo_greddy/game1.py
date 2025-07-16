import pandas as pd
import numpy as np
import cvxpy as cp
import os
from typing import Tuple, Dict

# ------------------------------
# 1. Configuración inicial
# ------------------------------
def check_file_exists(filename: str) -> bool:
    """Verifica si el archivo existe en el directorio actual"""
    current_dir = os.getcwd()
    file_path = os.path.join(current_dir, filename)
    
    print(f"Directorio actual: {current_dir}")
    print(f"Buscando archivo: {file_path}")
    
    if os.path.exists(file_path):
        print("✅ Archivo encontrado")
        return True
    else:
        print("❌ Archivo no encontrado")
        print("Archivos en el directorio actual:")
        for file in os.listdir(current_dir):
            print(f"  - {file}")
        return False

# ------------------------------
# 2. Carga y validación de datos
# ------------------------------
def load_data(filename: str) -> pd.DataFrame:
    """Carga y prepara los datos"""
    data = pd.read_excel(filename)
    
    # Validar columnas requeridas
    required_columns = [
        'activo_id', 'retorno_esperado', 'volatilidad', 'beta', 
        'liquidez_score', 'sector', 'precio_accion', 'min_inversion'
    ]
    
    if not all(col in data.columns for col in required_columns):
        raise ValueError("Faltan columnas requeridas en el dataset")
    
    # Convertir retornos y volatilidades a decimales
    data['retorno_esperado'] = data['retorno_esperado'].astype(float) / 100
    data['volatilidad'] = data['volatilidad'].astype(float) / 100
    
    # Validar rangos
    if (data['retorno_esperado'].min() < 0.05 or data['retorno_esperado'].max() > 0.18 or
        data['volatilidad'].min() < 0.07 or data['volatilidad'].max() > 0.30):
        raise ValueError("Datos fuera de los rangos esperados")
    
    return data

# ------------------------------
# 3. Optimización del portafolio
# ------------------------------
def optimize_portfolio(data: pd.DataFrame, 
                      budget: float = 1_000_000, 
                      lambda_risk: float = 0.5, 
                      max_beta: float = 1.2, 
                      sector_limit: float = 0.3,
                      min_assets: int = 5) -> Tuple[pd.DataFrame, float, float, float]:
    """Optimiza el portafolio usando programación convexa con variables enteras"""
    
    n_assets = len(data)
    sectors = data['sector'].unique()
    
    # Extraer parámetros
    returns = data['retorno_esperado'].values
    volatilities = data['volatilidad'].values
    betas = data['beta'].values
    min_inv = data['min_inversion'].values
    prices = data['precio_accion'].values
    sectors_list = data['sector'].values
    
    # Crear máscaras de sector
    sector_masks = {s: (sectors_list == s) for s in sectors}
    
    # Variables de optimización
    weights = cp.Variable(n_assets)  # Porcentajes de asignación
    y = cp.Variable(n_assets, boolean=True)  # Variables binarias para activos seleccionados
    
    # Función objetivo: Maximizar U = sum(r_i*w_i) - λ*sum(σ_i^2*w_i^2)
    port_return = returns @ weights
    port_risk = cp.sum_squares(cp.multiply(volatilities, weights))
    utility = port_return - lambda_risk * port_risk
    
    # Restricciones
    constraints = [
        cp.sum(weights) == 1,  # Suma de pesos = 100%
        weights >= 0,  # No ventas en corto
        cp.sum(y) >= min_assets,  # Mínimo 5 activos
        weights <= y,  # Si y_i=0 => w_i=0
        betas @ weights <= max_beta,  # Límite de beta
    ]
    
    # Restricciones por sector
    for s in sectors:
        sector_weight = cp.sum(weights[sector_masks[s]])
        constraints.append(sector_weight <= sector_limit)
    
    # Restricción de inversión mínima
    for i in range(n_assets):
        constraints.append(weights[i] * budget >= min_inv[i] * y[i])
    
    # Formulación del problema
    problem = cp.Problem(cp.Maximize(utility), constraints)
    
    # Resolver
    try:
        problem.solve(solver=cp.ECOS_BB, verbose=False)
    except Exception as e:
        raise ValueError(f"Error en la optimización: {str(e)}")
    
    if problem.status != 'optimal':
        raise ValueError("No se encontró solución óptima")
    
    # Procesar resultados
    optimal_weights = weights.value
    optimal_weights[optimal_weights < 1e-6] = 0  # Eliminar pesos negligible
    
    # Calcular asignación de capital
    capital_allocated = optimal_weights * budget
    n_shares = np.floor(capital_allocated / prices)
    
    # Ajustar capital asignado para números enteros de acciones
    capital_allocated = n_shares * prices
    optimal_weights = capital_allocated / budget
    
    # Filtrar solo activos con peso > 0
    selected = optimal_weights > 0
    portfolio = data[selected].copy()
    portfolio['peso'] = optimal_weights[selected]
    portfolio['capital_asignado'] = capital_allocated[selected]
    portfolio['n_acciones'] = n_shares[selected]
    
    # Calcular métricas finales
    final_weights = optimal_weights[selected]
    port_return = returns[selected] @ final_weights
    port_risk = np.sqrt(np.sum((volatilities[selected]*2) * (final_weights*2)))
    port_beta = betas[selected] @ final_weights
    
    return portfolio, port_return, port_risk, port_beta

# ------------------------------
# 4. Cálculo del puntaje
# ------------------------------
def calculate_score(port_return: float, 
                   port_risk: float, 
                   constraints_violated: int = 0, 
                   time_factor: float = 1.2) -> float:
    """Calcula el puntaje según la fórmula del torneo"""
    # Factor de restricciones
    if constraints_violated == 0:
        fr = 1.0
    elif constraints_violated == 1:
        fr = 0.8
    else:
        fr = 0.6
    
    # Puntaje base (corregida la fórmula de volatilidad)
    score = 1000 * (port_return - 0.5 * port_risk) * fr * time_factor
    return score

# ------------------------------
# 5. Visualización de resultados
# ------------------------------
def print_results(portfolio: pd.DataFrame, 
                 port_return: float, 
                 port_risk: float, 
                 port_beta: float, 
                 score: float) -> None:
    """Muestra los resultados de la optimización"""
    print("\n" + "="*60)
    print("PORTFOLIO OPTIMIZADO (SOLUCIÓN CORREGIDA)")
    print("="*60)
    
    print("\nActivos seleccionados:")
    print(portfolio[['activo_id', 'sector', 'peso', 'capital_asignado', 'n_acciones', 
                     'retorno_esperado', 'volatilidad', 'beta']].to_string(float_format="%.4f"))
    
    print("\nResumen del portafolio:")
    print(f"- Retorno esperado: {port_return:.2%}")
    print(f"- Volatilidad: {port_risk:.2%}")
    print(f"- Beta: {port_beta:.2f}")
    print(f"- Puntaje estimado: {score:.2f}")
    
    print("\nDistribución por sector:")
    sector_dist = portfolio.groupby('sector')['peso'].sum()
    for sector, weight in sector_dist.items():
        print(f"Sector {sector}: {weight:.1%}")
    
    print(f"\nNúmero de activos seleccionados: {len(portfolio)}")

# ------------------------------
# 6. Función principal
# ------------------------------
def main():
    print("OPTIMIZADOR DE PORTAFOLIO - TORNEO OPTIMABATTLE (VERSIÓN CORREGIDA)")
    print("="*60)
    
    filename = 'Ronda1.xlsx'
    
    # Paso 1: Verificar archivo
    if not check_file_exists(filename):
        return
    
    # Paso 2: Cargar datos
    try:
        data = load_data(filename)
        print("\n✅ Datos cargados correctamente")
        print(f"Número de activos disponibles: {len(data)}")
    except Exception as e:
        print(f"❌ Error al cargar datos: {str(e)}")
        return
    
    # Paso 3: Optimización
    try:
        portfolio, port_return, port_risk, port_beta = optimize_portfolio(data)
        score = calculate_score(port_return, port_risk)
        print_results(portfolio, port_return, port_risk, port_beta, score)
    except Exception as e:
        print(f"❌ Error en la optimización: {str(e)}")

if __name__ == "__main__":
    main()