import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier


def zscore_df(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        out[col] = (out[col] - out[col].mean()) / out[col].std(ddof=0)
    return out


def main() -> None:
    data = pd.read_excel("DataSet_completo.xlsx")
    data = data.drop(columns=["Longitud"])

    feature_cols = [c for c in data.columns if c != "Clase"]
    x = zscore_df(data[feature_cols], feature_cols)
    y = data["Clase"]

    iterations = range(1, 500, 10)
    scores = []

    for it in iterations:
        neural_net = MLPClassifier(
            hidden_layer_sizes=(22,),
            activation="relu",
            solver="adam",
            learning_rate_init=0.1,
            max_iter=it,
            random_state=42,
        )

        cv_scores = cross_val_score(neural_net, x, y, cv=5)
        scores.append(cv_scores.mean())

    plt.figure(figsize=(8, 5))
    plt.plot(iterations, scores, marker="o")
    plt.xlabel("Número de iteraciones (max_iter)")
    plt.ylabel("Accuracy promedio CV=5")
    plt.title("Curva de aprendizaje usando validación cruzada")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
