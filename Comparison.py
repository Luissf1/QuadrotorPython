import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

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

# ==========================================================
# Calcular promedios
# ==========================================================
resumen = pd.DataFrame({
    "Controller": ["PSO-PID (No Disturbance)", "PSO-PID (With Disturbance)"],
    "SettlingTime": [df_no_dist["SettlingTime"].mean(), df_disturbio["SettlingTime"].mean()],
    "Overshoot": [df_no_dist["Overshoot"].mean(), df_disturbio["Overshoot"].mean()],
    "ITSE": [df_no_dist["ITSE"].mean(), df_disturbio["ITSE"].mean()],
    "IAE": [df_no_dist["IAE"].mean(), df_disturbio["IAE"].mean()],
    "Costo": [df_no_dist["Costo"].mean(), df_disturbio["Costo"].mean()]
})

print("\n=== Tabla comparativa de promedios ===\n")
print(resumen)

# ==========================================================
# Gráfico de barras (Costo)
# ==========================================================
plt.figure(figsize=(8,5))
plt.bar(resumen["Controller"], resumen["Costo"], color=["skyblue","salmon"])
plt.title("Comparación de Costo Promedio")
plt.ylabel("Costo")
plt.grid(True, linestyle="--", alpha=0.6)
plt.show()

# ==========================================================
# Gráfico radar (spider chart)
# ==========================================================
# Métricas
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

# Crear figura
fig, ax = plt.subplots(figsize=(6,6), subplot_kw=dict(polar=True))

ax.plot(angles, no_dist_values, "o-", linewidth=2, label="PSO-PID (No Disturbance)")
ax.fill(angles, no_dist_values, alpha=0.25)

ax.plot(angles, dist_values, "o-", linewidth=2, label="PSO-PID (With Disturbance)")
ax.fill(angles, dist_values, alpha=0.25)

ax.set_thetagrids(np.degrees(angles[:-1]), labels)
ax.set_title("Radar Chart - Comparative Metrics", size=14)
ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
plt.show()
