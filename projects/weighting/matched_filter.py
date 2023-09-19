import numpy as np
import xarray as xr
import os, tqdm

import matplotlib.pyplot as plt

plt.style.use(
    "/mnt/c/Users/johan/OneDrive/Skrivebord/Master_Thesis/setup/matplotlib_style/standard_plot_style.mplstyle"
)

from matplotlib.colors import LinearSegmentedColormap

cmap = LinearSegmentedColormap.from_list("standard_colors", ["C0", "white", "C1"])


path = "/mnt/d/Data/master_thesis/trajectories/readout test_161128/demodulated.nc"
data = xr.open_dataset(
    path,
)


def demod(data, start, stop):
    return data.sel(adc_timestamp=slice(start * 1e-9, stop * 1e-9))


def convert_to_train_set(data, start, stop):
    data = demod(data, start, stop)
    X = np.concatenate(
        [
            np.dstack([data.I_ground, data.Q_ground]),
            np.dstack([data.I_excited, data.Q_excited]),
        ],
        axis=0,
    )
    y = np.concatenate(
        (np.zeros(data.I_ground.sample.size), np.ones(data.I_excited.sample.size))
    )
    return X, y, data.adc_timestamp * 1e9


def scatter_with_classification(X, y_pred, y_true, title=None):
    fig, ax = plt.subplots(
        2,
        2,
        figsize=(12, 8),
        gridspec_kw={"height_ratios": [2, 1], "width_ratios": [2, 1]},
    )

    # Scatter pred and "true"
    ax[0, 0].scatter(X[:, 0], X[:, 1], c=y_pred, cmap=cmap, alpha=0.5)
    ax[1, 1].scatter(X[:, 0], X[:, 1], c=y_true, cmap=cmap, alpha=0.5)

    ax[1, 0].hist(X[y_pred == 0, 0], bins=30, color="C0", alpha=0.5)
    ax[1, 0].hist(X[y_pred == 1, 0], bins=30, color="C1", alpha=0.5)
    ax[1, 0].invert_yaxis()

    ax[0, 1].hist(
        X[y_true == 0, 1], bins=30, color="C0", alpha=0.5, orientation="horizontal"
    )
    ax[0, 1].hist(
        X[y_true == 1, 1], bins=30, color="C1", alpha=0.5, orientation="horizontal"
    )

    ax[0, 0].set(
        xlabel="I",
        ylabel="Q",
        title="Predicted",
    )
    ax[1, 0].set(
        xlabel="I",
        ylabel="Counts",
    )
    ax[0, 1].set(
        ylabel="Q",
        xlabel="Counts",
    )
    ax[1, 1].set(
        xlabel="I",
        ylabel="Q",
        title="True",
    )
    title = __file__.split("/")[-1].split(".")[0] if title is None else title
    fig.suptitle(title, fontsize=24)

    fig.tight_layout()

    return fig, ax


# Evaluation
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from seaborn import heatmap


def evaluate(y_pred, y_pred_score, y_true, title=None):
    confusion = confusion_matrix(y_true, y_pred, normalize="true")
    auc = roc_auc_score(y_true, y_pred_score)
    fpr, tpr, _ = roc_curve(y_true, y_pred_score)

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ticklabels = ["Ground", "Excited"]
    heatmap(
        confusion,
        annot=True,
        ax=ax[0],
        cmap=cmap,
        xticklabels=ticklabels,
        yticklabels=ticklabels,
        vmin=0,
        vmax=1,
    )

    ax[1].plot(
        fpr,
        tpr,
        color="C0",
        label=f"AUC = {auc:.2f}",
    )
    ax[1].text(0.5, 0.5, f"AUC = {auc:.3f}", fontsize=24)

    ax[0].set(title="Confusion Matrix")

    ax[1].set(
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        title="ROC Curve",
    )

    title = __file__.split("/")[-1].split(".")[0] if title is None else title
    fig.suptitle(title, fontsize=24)

    fig.tight_layout()

    string_output = f"""
    Confusion Matrix:
    {confusion[0, 0]:.3f}\t{confusion[0, 1]:.3f}
    {confusion[1, 0]:.3f}\t{confusion[1, 1]:.3f} \n
    AUC: {auc:.4f}
    """

    print(string_output)

    return fig, ax


X, y, times = convert_to_train_set(data, 0, 25000)

from sklearn.model_selection import train_test_split

# PREPROCESS!

##
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)
##

# INSERT METHOD HERE!

ground_train = X_train[y_train == 0]
excited_train = X_train[y_train == 1]

weights = (ground_train.mean(axis=0) - excited_train.mean(axis=0)) / (
    ground_train.std(axis=0) ** 2 + excited_train.std(axis=0) ** 2
)

# Plot filter weight for I and Q
fig, ax = plt.subplots(nrows=2)
ax[0].plot(times, weights[:, 0])
ax[0].set(xlabel="Time (ns)", ylabel="weight", title="I weights")
ax[1].plot(times, weights[:, 1])
ax[1].set(xlabel="Time (ns)", ylabel="weight", title="Q weights")
fig.tight_layout()

input_train, input_test = weights * X_train, weights * X_test
input_train, input_test = input_train.mean(axis=1), input_test.mean(axis=1)


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

lda = LDA()
lda.fit(input_train, y_train)

y_pred = lda.predict(input_test)
y_score = lda.predict_proba(input_test)


# Evaluation
plot_test = X_test.mean(axis=1)
fig, ax = scatter_with_classification(plot_test, y_pred, y_test)
fig_eval, ax_eval = evaluate(y_pred, y_score[:, 1], y_test)
