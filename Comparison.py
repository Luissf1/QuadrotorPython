import pandas as pd
import matplotlib.pyplot as plt

# ==========================================================
# Resultados obtenidos (ejemplo: llena con tus datos)
# ==========================================================

# --- Resultados SIN disturbio ---
# Cada fila: [SettlingTime, Overshoot, ITSE, IAE]
resultados_no_dist = [
    [0.82, 0.05, 0.010, 0.200],
    [0.90, 0.01, 0.008, 0.220],
    [0.87, 0.04, 0.012, 0.210],
    [0.88, 0.03, 0.009, 0.190],
    [0.84, 0.06, 0.011, 0.205]
]

# --- Resultados CON disturbio (los que ya me diste) ---
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
# Graficar comparación (ejemplo: barras de costo)
# ==========================================================
plt.figure(figsize=(8,5))
plt.bar(resumen["Controller"], resumen["Costo"], color=["skyblue","salmon"])
plt.title("Comparación de Costo Promedio")
plt.ylabel("Costo")
plt.grid(True, linestyle="--", alpha=0.6)
plt.show()
