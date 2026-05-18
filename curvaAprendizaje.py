import matplotlib.pyplot as plt
import pandas as pd
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
    x = zscore_df(data[feature_cols], feature_cols)
    y = data["Clase"]

    model = build_phase2_mlp()
    model.fit(x, y)

    plt.figure(figsize=(8, 4))
    plt.plot(range(1, len(model.loss_curve_) + 1), model.loss_curve_, color="blue")
    plt.title("Curva de aprendizaje MLP")
    plt.xlabel("Iteración")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
