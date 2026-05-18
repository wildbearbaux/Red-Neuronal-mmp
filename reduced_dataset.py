import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.neural_network import MLPClassifier


def zscore_df(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        out[col] = (out[col] - out[col].mean()) / out[col].std(ddof=0)
    return out


# Cargar datos y separar target
data = pd.read_excel("DataSet_completo.xlsx")
data = data.drop(columns=["Longitud"])

feature_cols = [c for c in data.columns if c != "Clase"]
y = data["Clase"]

# 1) Estandarización inicial para análisis de correlación
x_std = zscore_df(data[feature_cols], feature_cols)

# 2) Matriz de correlación entre características
corr_matrix_abs = x_std.corr().abs()

# 3) Identificar pares con correlación alta |r| > 0.8 y eliminar redundancia
threshold = 0.80
upper = corr_matrix_abs.where(np.triu(np.ones(corr_matrix_abs.shape), k=1).astype(bool))
to_drop = [col for col in upper.columns if any(upper[col] > threshold)]

x_reduced = x_std.drop(columns=to_drop)

# 4) Re-estandarizar después de eliminar características (por limpieza numérica)
x_reduced = zscore_df(x_reduced, list(x_reduced.columns))

# Hiperparámetros de Fase 2 (mismos que en main.py)
def build_phase2_mlp() -> MLPClassifier:
    return MLPClassifier(
        hidden_layer_sizes=(22,),
        activation="relu",
        solver="adam",
        learning_rate_init=0.1,
        max_iter=50_000,
        random_state=42,
    )


cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Baseline Fase 2 (todas las características)
mlp_full = build_phase2_mlp()
y_pred_full = cross_val_predict(mlp_full, x_std, y, cv=cv)
acc_full = accuracy_score(y, y_pred_full)
cm_full = confusion_matrix(y, y_pred_full)

# MLP con dataset reducido (mismos hiperparámetros)
mlp_reduced = build_phase2_mlp()
y_pred_reduced = cross_val_predict(mlp_reduced, x_reduced, y, cv=cv)
acc_reduced = accuracy_score(y, y_pred_reduced)
cm_reduced = confusion_matrix(y, y_pred_reduced)

# Ajuste único para curva de aprendizaje por iteración (loss_curve_)
mlp_curve = build_phase2_mlp()
mlp_curve.fit(x_reduced, y)

print("=== Reducción de características ===")
print("Características originales:", len(feature_cols))
print("Características eliminadas (|r| > 0.8):", len(to_drop))
print("Características finales:", x_reduced.shape[1])
print("Variables eliminadas:", to_drop)

print("\n=== Comparación Fase 2 vs Reducido ===")
print(f"Accuracy Fase 2 (todas): {acc_full * 100:.2f}%")
print(f"Accuracy Reducido:       {acc_reduced * 100:.2f}%")
print(f"Diferencia:              {(acc_reduced - acc_full) * 100:.2f} puntos")

# Visualizaciones
sns.heatmap(corr_matrix_abs, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Matriz de correlación absoluta (antes de reducir)")
plt.tight_layout()
plt.show()

sns.heatmap(x_reduced.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Matriz de correlación (dataset reducido)")
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
sns.heatmap(cm_full, annot=True, fmt="d", ax=axes[0])
axes[0].set_title("Confusion matrix - Fase 2")
sns.heatmap(cm_reduced, annot=True, fmt="d", ax=axes[1])
axes[1].set_title("Confusion matrix - Reducido")
plt.tight_layout()
plt.show()

plt.figure(figsize=(7, 4))
plt.plot(range(1, len(mlp_curve.loss_curve_) + 1), mlp_curve.loss_curve_)
plt.xlabel("Iteración")
plt.ylabel("Loss")
plt.title("Curva de aprendizaje MLP (dataset reducido)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
