import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.neural_network import MLPClassifier


data = pd.read_excel("DataSet_completo.xlsx")
data = data.drop(columns=["Longitud"])


def calculate_z_score(df: pd.DataFrame, col_name: str) -> pd.Series:
    return (df[col_name] - df[col_name].mean()) / df[col_name].std(ddof=0)


# Normalizar características numéricas (excepto clase)
normalized_data = data.copy()
feature_cols = [c for c in data.columns if c != "Clase"]
for col in feature_cols:
    normalized_data[col] = calculate_z_score(normalized_data, col)

# Convertir Clase a etiquetas numéricas para poder usar Pearson con el target
if normalized_data["Clase"].dtype == "object":
    y = pd.factorize(normalized_data["Clase"])[0]
else:
    y = normalized_data["Clase"].to_numpy()

x_all = normalized_data[feature_cols]

# 1) Filtro supervisado: dejar solo variables con correlación útil vs target
corr_vs_target = x_all.apply(lambda s: s.corr(pd.Series(y))).abs()
min_target_corr = 0.20
selected_by_target = corr_vs_target[corr_vs_target >= min_target_corr].index.tolist()

# Si el filtro quedó muy agresivo, mantener al menos las 4 mejores
if len(selected_by_target) < 4:
    selected_by_target = corr_vs_target.sort_values(ascending=False).head(4).index.tolist()

x_target_filtered = x_all[selected_by_target]

# 2) Eliminar redundancia entre predictores con umbral más agresivo
pred_corr = x_target_filtered.corr().abs()
upper = pred_corr.where(np.triu(np.ones(pred_corr.shape), k=1).astype(bool))

# Reducido más agresivo que antes (0.75 -> 0.60)
threshold = 0.60
to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

x_reduced = x_target_filtered.drop(columns=to_drop)

# Evaluación robusta para elegir arquitectura
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

candidates = [
    (4,),
    (6,),
    (8,),
    (6, 4),
    (8, 4),
]

best_score = -1
best_arch = None
for arch in candidates:
    model = MLPClassifier(
        hidden_layer_sizes=arch,
        activation="relu",
        solver="adam",
        learning_rate_init=0.01,
        alpha=0.001,
        max_iter=50_000,
        random_state=42,
    )
    scores = cross_val_score(model, x_reduced, y, cv=cv, scoring="accuracy")
    score = scores.mean()
    if score > best_score:
        best_score = score
        best_arch = arch

print("Variables originales:", len(feature_cols))
print("Después de correlación con target:", len(selected_by_target))
print("Variables eliminadas por redundancia:", len(to_drop))
print("Variables finales:", x_reduced.shape[1])
print("Arquitectura elegida:", best_arch)
print("Accuracy CV promedio:", round(best_score * 100, 2), "%")

# Entrenar/predicción CV final con la mejor arquitectura
best_model = MLPClassifier(
    hidden_layer_sizes=best_arch,
    activation="relu",
    solver="adam",
    learning_rate_init=0.01,
    alpha=0.001,
    max_iter=50_000,
    random_state=42,
)

y_pred = cross_val_predict(best_model, x_reduced, y, cv=cv)
conf_matrix = confusion_matrix(y, y_pred)
accuracy = accuracy_score(y, y_pred)

# Visualizaciones
sns.heatmap(x_reduced.corr(), annot=True, cmap="coolwarm", fmt=".2f", xticklabels=True, yticklabels=True)
plt.title("Correlation matrix (reduced dataset)")
plt.show()

x_reduced.hist(bins=20, figsize=(12, 8))
plt.tight_layout()
plt.show()

sns.heatmap(conf_matrix, annot=True, annot_kws={"size": 12})
plt.title("Confusion matrix")
plt.show()

print("Accuracy final:", accuracy * 100, "%")
