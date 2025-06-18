import numpy as np
import matplotlib.pyplot as plt

# Configuración básica
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

def plot_convergencia_basica():
    """Gráfica básica de convergencia del SA"""
    # Datos simulados
    iteraciones = list(range(1000))
    temp_inicial = 1000
    temp_final = 0.1
    
    # Temperatura exponencial
    temperaturas = [temp_inicial * (temp_final/temp_inicial)**(i/1000) for i in iteraciones]
    
    # Costos simulados
    costos = []
    costo_actual = 2500
    for i in iteraciones:
        progreso = i / 1000
        costo_base = 2500 - (2500 - 1200) * (progreso**0.7)
        ruido = np.random.normal(0, 50 * (1-progreso))
        costo_actual = max(costo_base + ruido, 1200)
        costos.append(costo_actual)
    
    # Crear gráfica
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Temperatura
    ax1.plot(iteraciones, temperaturas, 'r-', linewidth=2)
    ax1.set_ylabel('Temperatura')
    ax1.set_title('Convergencia del Simulated Annealing', fontsize=16, fontweight='bold')
    ax1.set_yscale('log')
    ax1.grid(True)
    
    # Costos
    ax2.plot(iteraciones, costos, 'b-', linewidth=2)
    ax2.axhline(y=1200, color='red', linestyle='--', label='Óptimo')
    ax2.set_xlabel('Iteraciones')
    ax2.set_ylabel('Costo (km)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

def plot_comparacion_basica():
    """Comparación básica de métodos"""
    metodos = ['Greedy', 'Genetic\nAlgorithm', 'Ant\nColony', 'Simulated\nAnnealing', 'Random\nSearch']
    costos = [1850, 1420, 1380, 1250, 2100]
    
    plt.figure(figsize=(12, 6))
    
    # Crear barras con colores diferentes
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    bars = plt.bar(metodos, costos, color=colors, alpha=0.7)
    
    # Destacar SA
    bars[3].set_color('gold')
    bars[3].set_edgecolor('black')
    bars[3].set_linewidth(2)
    
    plt.title('Comparación de Algoritmos - Costo de Rutas', fontsize=16, fontweight='bold')
    plt.ylabel('Costo Promedio (km)')
    plt.grid(True, alpha=0.3)
    
    # Añadir valores
    for i, v in enumerate(costos):
        plt.text(i, v + 30, str(v), ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.show()

def plot_eficiencia_basica():
    """Métricas de eficiencia básicas"""
    metricas = ['Distancia\n(km)', 'Tiempo\n(horas)', 'Combustible\n(litros)', 'Costo\n($/día)']
    antes = [2100, 8.5, 85, 320]
    despues = [1250, 5.2, 52, 195]
    
    x = np.arange(len(metricas))
    width = 0.35
    
    plt.figure(figsize=(12, 6))
    
    plt.bar(x - width/2, antes, width, label='Antes SA', color='red', alpha=0.7)
    plt.bar(x + width/2, despues, width, label='Después SA', color='green', alpha=0.7)
    
    plt.xlabel('Métricas')
    plt.ylabel('Valores')
    plt.title('Eficiencia: Antes vs Después del SA', fontsize=16, fontweight='bold')
    plt.xticks(x, metricas)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def ejecutar_graficas_basicas():
    """Ejecutar todas las gráficas básicas"""
    print("Generando gráficas básicas...")
    
    print("1. Convergencia del SA...")
    plot_convergencia_basica()
    
    print("2. Comparación de métodos...")
    plot_comparacion_basica()
    
    print("3. Métricas de eficiencia...")
    plot_eficiencia_basica()
    
    print("¡Gráficas generadas exitosamente!")

if __name__ == "__main__":
    ejecutar_graficas_basicas()