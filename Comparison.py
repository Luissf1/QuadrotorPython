import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

# ==========================================================
# Resultados obtenidos
# ==========================================================

# --- Resultados SIN disturbio ---
resultados_no_dist = [
    [1.101101101, 0.981613847, 0.08759185 , 0.625743666],
    [0.830830831, 0.127173193, 0.112271798, 0.642020181],
    [0.950950951, 0.060452183, 0.235049988, 0.911622821],
    [1.011011011, 1.473953159, 0.078539115, 0.591643162],
    [0.810810811, 0.0        , 0.011456358, 0.197389013]
]

# --- Resultados CON disturbio ---
resultados_dist = [
    [1.051051051, 1.897292264, 0.075040146, 0.548239485],
    [0.870870871, 0.032821126, 0.003134391, 0.11741169 ],
    [0.860860861, 0.189644937, 0.201623918, 0.822642547],
    [1.021021021, 1.834098678, 0.071787703, 0.524398508],
    [0.800800801, 0.0        , 0.011703052, 0.211794205]
]

# ==========================================================
# Crear DataFrames
# ==========================================================
df_no_dist = pd.DataFrame(resultados_no_dist, columns=["SettlingTime","Overshoot","ITSE","IAE"])
df_disturbio = pd.DataFrame(resultados_dist, columns=["SettlingTime","Overshoot","ITSE","IAE"])

# Add identifiers
df_no_dist['Flight'] = ['NoDist_Flight_' + str(i+1) for i in range(len(df_no_dist))]
df_disturbio['Flight'] = ['Dist_Flight_' + str(i+1) for i in range(len(df_disturbio))]
df_no_dist['Type'] = 'Without Disturbance'
df_disturbio['Type'] = 'With Disturbance'

# Combine both datasets
df_combined = pd.concat([df_no_dist, df_disturbio], ignore_index=True)

# ==========================================================
# Definir pesos de la función de costo
# ==========================================================
w_settle, w_overshoot, w_itse, w_iae = 0.3, 0.3, 0.2, 0.2

def cost_function(settle, overshoot, itse, iae):
    return (w_settle*settle +
            w_overshoot*overshoot +
            w_itse*itse +
            w_iae*iae)

# Agregar columna de costo
df_no_dist["Costo"] = df_no_dist.apply(lambda row: cost_function(row["SettlingTime"], row["Overshoot"], row["ITSE"], row["IAE"]), axis=1)
df_disturbio["Costo"] = df_disturbio.apply(lambda row: cost_function(row["SettlingTime"], row["Overshoot"], row["ITSE"], row["IAE"]), axis=1)
df_combined["Costo"] = df_combined.apply(lambda row: cost_function(row["SettlingTime"], row["Overshoot"], row["ITSE"], row["IAE"]), axis=1)

# ==========================================================
# Calcular promedios y mejoras
# ==========================================================
resumen = pd.DataFrame({
    "Controller": ["PSO-PID (No Disturbance)", "PSO-PID (With Disturbance)"],
    "SettlingTime": [df_no_dist["SettlingTime"].mean(), df_disturbio["SettlingTime"].mean()],
    "Overshoot": [df_no_dist["Overshoot"].mean(), df_disturbio["Overshoot"].mean()],
    "ITSE": [df_no_dist["ITSE"].mean(), df_disturbio["ITSE"].mean()],
    "IAE": [df_no_dist["IAE"].mean(), df_disturbio["IAE"].mean()],
    "Costo": [df_no_dist["Costo"].mean(), df_disturbio["Costo"].mean()]
})

# Calculate percentage improvements
improvements = []
for i in range(1, len(resumen.columns)):
    no_dist = resumen.iloc[0, i]
    with_dist = resumen.iloc[1, i]
    improvement = ((no_dist - with_dist) / no_dist) * 100
    improvements.append(improvement)

resumen_improvement = pd.DataFrame({
    "Metric": resumen.columns[1:],
    "No Disturbance": resumen.iloc[0, 1:].values,
    "With Disturbance": resumen.iloc[1, 1:].values,
    "Improvement (%)": improvements
})

print("\n=== Tabla comparativa de promedios ===\n")
print(resumen)

print("\n=== Mejora porcentual ===\n")
print(resumen_improvement)

# ==========================================================
# Visualización mejorada para presentación
# ==========================================================
plt.style.use('default')
fig = plt.figure(figsize=(20, 12))

# 1. Comparación de costos individuales
ax1 = plt.subplot(2, 3, 1)
colors = ['skyblue']*5 + ['salmon']*5
bars = ax1.bar(df_combined['Flight'], df_combined['Costo'], color=colors)
ax1.set_title("Costo por Vuelo Individual", fontsize=14, fontweight='bold')
ax1.set_ylabel("Costo")
ax1.tick_params(axis='x', rotation=45)
ax1.grid(True, linestyle="--", alpha=0.6)

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.3f}', ha='center', va='bottom', fontsize=9)

# Add legend
legend_elements = [Patch(facecolor='skyblue', label='Sin Disturbio'),
                  Patch(facecolor='salmon', label='Con Disturbio')]
ax1.legend(handles=legend_elements, loc='upper right')

# 2. Comparación de métricas promedio
ax2 = plt.subplot(2, 3, 2)
metrics = ['SettlingTime', 'Overshoot', 'ITSE', 'IAE', 'Costo']
x_pos = np.arange(len(metrics))
width = 0.35

no_dist_vals = [resumen.iloc[0][m] for m in metrics]
with_dist_vals = [resumen.iloc[1][m] for m in metrics]

bars1 = ax2.bar(x_pos - width/2, no_dist_vals, width, label='Sin Disturbio', alpha=0.8, color='skyblue')
bars2 = ax2.bar(x_pos + width/2, with_dist_vals, width, label='Con Disturbio', alpha=0.8, color='salmon')

ax2.set_xlabel('Métricas')
ax2.set_ylabel('Valores')
ax2.set_title('Comparación de Métricas Promedio', fontsize=14, fontweight='bold')
ax2.set_xticks(x_pos)
ax2.set_xticklabels(['Settling\nTime', 'Overshoot', 'ITSE', 'IAE', 'Costo'])
ax2.legend()
ax2.grid(True, alpha=0.3)

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)

# 3. Mejora porcentual
ax3 = plt.subplot(2, 3, 3)
colors = ['green' if imp > 0 else 'red' for imp in resumen_improvement['Improvement (%)']]
bars = ax3.bar(resumen_improvement['Metric'], resumen_improvement['Improvement (%)'], color=colors, alpha=0.7)
ax3.set_xlabel('Métricas')
ax3.set_ylabel('Mejora (%)')
ax3.set_title('Mejora Porcentual con Controlador Optimizado', fontsize=14, fontweight='bold')
ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
ax3.tick_params(axis='x', rotation=45)
ax3.grid(True, alpha=0.3)

# Add value labels
for bar in bars:
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.1f}%', ha='center', va='bottom' if height > 0 else 'top', fontsize=10)

# 4. Scatter plot de SettlingTime vs Overshoot
ax4 = plt.subplot(2, 3, 4)
scatter1 = ax4.scatter(df_no_dist['SettlingTime'], df_no_dist['Overshoot'], color='blue', 
            label='Sin Disturbio', s=100, alpha=0.7)
scatter2 = ax4.scatter(df_disturbio['SettlingTime'], df_disturbio['Overshoot'], color='red', 
            label='Con Disturbio', s=100, alpha=0.7)

# Add flight labels
for i, row in df_no_dist.iterrows():
    ax4.annotate(f'N{i+1}', (row['SettlingTime'], row['Overshoot']), 
                xytext=(5, 5), textcoords='offset points', fontsize=8)
for i, row in df_disturbio.iterrows():
    ax4.annotate(f'D{i+1}', (row['SettlingTime'], row['Overshoot']), 
                xytext=(5, 5), textcoords='offset points', fontsize=8)

ax4.set_title("SettlingTime vs Overshoot", fontsize=14, fontweight='bold')
ax4.set_xlabel("SettlingTime")
ax4.set_ylabel("Overshoot")
ax4.legend()
ax4.grid(True, linestyle="--", alpha=0.6)

# 5. Radar chart
ax5 = plt.subplot(2, 3, 5, polar=True)
labels = ["SettlingTime", "Overshoot", "ITSE", "IAE", "Costo"]
num_vars = len(labels)

# Valores promedio
no_dist_values = resumen.iloc[0,1:].values.tolist()
dist_values = resumen.iloc[1,1:].values.tolist()

# Cerrar el radar (último punto = primer punto)
angles = np.linspace(0, 2*np.pi, num_vars, endpoint=False).tolist()
no_dist_values += no_dist_values[:1]
dist_values += dist_values[:1]
angles += angles[:1]

ax5.plot(angles, no_dist_values, "o-", linewidth=2, label="PSO-PID (No Disturbance)")
ax5.fill(angles, no_dist_values, alpha=0.25)

ax5.plot(angles, dist_values, "o-", linewidth=2, label="PSO-PID (With Disturbance)")
ax5.fill(angles, dist_values, alpha=0.25)

ax5.set_thetagrids(np.degrees(angles[:-1]), labels)
ax5.set_title("Radar Chart - Comparative Metrics", fontsize=14, fontweight='bold')
ax5.legend(loc="upper right", bbox_to_anchor=(1.5, 1.1))

# 6. Costo promedio
ax6 = plt.subplot(2, 3, 6)
bars = ax6.bar(resumen["Controller"], resumen["Costo"], color=["skyblue","salmon"])
ax6.set_title("Comparación de Costo Promedio", fontsize=14, fontweight='bold')
ax6.set_ylabel("Costo")
ax6.grid(True, linestyle="--", alpha=0.6)

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    ax6.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.3f}', ha='center', va='bottom', fontsize=12)

plt.tight_layout()
plt.savefig('comprehensive_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# ==========================================================
# Tabla para presentación
# ==========================================================
print("\n" + "="*80)
print("TABLA PARA PRESENTACIÓN - RESUMEN DE RESULTADOS")
print("="*80)

presentation_table = pd.DataFrame({
    'Métrica': ['Tiempo de Estabilización (s)', 'Sobreimpulso (%)', 'ITSE', 'IAE', 'Costo Total'],
    'Sin Disturbio': [f"{resumen.iloc[0,1]:.3f}", f"{resumen.iloc[0,2]:.3f}", 
                     f"{resumen.iloc[0,3]:.3f}", f"{resumen.iloc[0,4]:.3f}", f"{resumen.iloc[0,5]:.3f}"],
    'Con Disturbio': [f"{resumen.iloc[1,1]:.3f}", f"{resumen.iloc[1,2]:.3f}", 
                     f"{resumen.iloc[1,3]:.3f}", f"{resumen.iloc[1,4]:.3f}", f"{resumen.iloc[1,5]:.3f}"],
    'Mejora (%)': [f"{improvements[0]:.1f}%", f"{improvements[1]:.1f}%", 
                  f"{improvements[2]:.1f}%", f"{improvements[3]:.1f}%", f"{improvements[4]:.1f}%"]
})

print(presentation_table.to_string(index=False))