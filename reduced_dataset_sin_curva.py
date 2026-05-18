import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.neural_network import MLPClassifier


def zscore_df(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        out[col] = (out[col] - out[col].mean()) / out[col].std(ddof=0)
    return out


def build_phase2_mlp() -> MLPClassifier:
    return MLPClassifier(
        hidden_layer_sizes=(22,),
        activation="relu",
        solver="adam",
        learning_rate_init=0.1,
        max_iter=50_000,
        random_state=42,
    )


def main() -> None:
    data = pd.read_excel("DataSet_completo.xlsx")
    data = data.drop(columns=["Longitud"])

    feature_cols = [c for c in data.columns if c != "Clase"]
    y = data["Clase"]

    x_std = zscore_df(data[feature_cols], feature_cols)

    corr_matrix_abs = x_std.corr().abs()
    upper = corr_matrix_abs.where(np.triu(np.ones(corr_matrix_abs.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > 0.80)]

    x_reduced = x_std.drop(columns=to_drop)
    x_reduced = zscore_df(x_reduced, list(x_reduced.columns))

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    y_pred_full = cross_val_predict(build_phase2_mlp(), x_std, y, cv=cv)
    y_pred_reduced = cross_val_predict(build_phase2_mlp(), x_reduced, y, cv=cv)

    acc_full = accuracy_score(y, y_pred_full)
    acc_reduced = accuracy_score(y, y_pred_reduced)

    cm_full = confusion_matrix(y, y_pred_full)
    cm_reduced = confusion_matrix(y, y_pred_reduced)

    print("=== Script sin curva de aprendizaje ===")
    print("Características originales:", len(feature_cols))
    print("Características eliminadas (|r| > 0.8):", len(to_drop))
    print("Características finales:", x_reduced.shape[1])
    print("Variables eliminadas:", to_drop)
    print()
    print(f"Accuracy Fase 2 (todas): {acc_full * 100:.2f}%")
    print(f"Accuracy Reducido:       {acc_reduced * 100:.2f}%")
    print(f"Diferencia:              {(acc_reduced - acc_full) * 100:.2f} puntos")
    print("\nMatriz de confusión Fase 2:")
    print(cm_full)
    print("\nMatriz de confusión Reducido:")
    print(cm_reduced)


if __name__ == "__main__":
    main()
