import json
import pandas as pd
import numpy as np
import hdbscan

from pathlib import Path

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KernelDensity

import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

# Daten laden
def load_tourism_data(path: Path) -> pd.DataFrame:
    with path.open("r", encoding="utf8") as f:
        data = json.load(f)

    df = pd.json_normalize(data, sep=".")

    print("Form des DataFrames:", df.shape)
    print(df[["Id", "Latitude", "Longitude", "Altitude"]].dropna())

    print("DataFrame shape:", df.shape)
    print(df.head())
    return df




#data_path = Path("/Users/florianbohrer/Documents/0uni/WiSe_2526/DataVis/lab2/TourismAssociation_Dataset.json")
#with open(data_path, "r", encoding="utf8") as f:
 #   data = json.load(f)


def cluster_geodata(
        df: pd.DataFrame,
        n_clusters: int = 4,
        min_cluster_size: int = 3
) -> tuple [pd.DataFrame, np.ndarray]:
    feature_cols = ["Latitude", "Longitude", "Altitude"]
    X3d = df[feature_cols].values
    print("Anzahl verwendeter Eintraege:", X3d.shape[0])

    # 3D Clustering
    scaler3d = StandardScaler()
    X3d_scaled = scaler3d.fit_transform(X3d)

    kmeans = KMeans(n_clusters= n_clusters, random_state=42, n_init="auto")
    df["cluster_kmeans"] = kmeans.fit_predict(X3d_scaled)

    hdb = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric="euclidean")
    df["cluster_hdbscan"] = hdb.fit_predict(X3d_scaled)
    return df, X3d_scaled



#this function trains the 2d Kernel density on the already scaled data
def compute_kde_grid(
    X2_scaled: np.ndarray,
    bandwidth: float = 0.25,
    grid_size: int = 150,
    margin: float = 0.5
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit KDE on 2D scaled data and return grid and density surface."""
    kde = KernelDensity(bandwidth=bandwidth)
    kde.fit(X2_scaled)

    x_min, x_max = X2_scaled[:, 0].min() - margin, X2_scaled[:, 0].max() + margin
    y_min, y_max = X2_scaled[:, 1].min() - margin, X2_scaled[:, 1].max() + margin

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, grid_size),
        np.linspace(y_min, y_max, grid_size)
    )

    grid = np.vstack([xx.ravel(), yy.ravel()]).T
    density = np.exp(kde.score_samples(grid)).reshape(xx.shape)
    return xx, yy, density



def plot_kde_with_clusters(
        X2_scaled: np.ndarray,
        labels: np.ndarray,
        xx: np.ndarray,
        yy: np.ndarray,
        density: np.ndarray,
        title: str = "KDE Demsity Surface with HBDSCAN Cluster Overlay"
) ->None:
    """Plot 3D KDE surface with cluster points slightly lifted above the surface."""
    fig = plt.figure(figsize=(20,15))
    ax = fig.add_subplot(111, projection="3d")

    surf = ax.plot_surface(
        xx, yy, density,
        cmap = cm.viridis,
        linewidth = 0,
        antialiased = True,
        alpha=0.75
    )

    fig.colorbar(surf, ax=ax, shrink=0.6, label="Density Level")

    scatter_z = np.zeros_like(labels, dtype=float)
    for i in range(len(scatter_z)):
        ix = np.argmin(np.abs(xx[0] - X2_scaled[i, 0]))
        iy = np.argmin(np.abs(yy[:, 0] - X2_scaled[i, 1]))
        scatter_z[i] = density[iy, ix] + 0.02

    ax.scatter(
        X2_scaled[:, 0],
        X2_scaled[:, 1],
        scatter_z,
        c=labels,
        cmap="tab10",
        s=40,
        depthshade=True
    )

    ax.set_xlabel("Latitude (scaled)")
    ax.set_ylabel("Longitude (scaled)")
    ax.set_zlabel("Density")
    ax.view_init(elev=30, azim=235)

    plt.show()


def main() -> None:
    data_path = Path("/Users/florianbohrer/Documents/0uni/WiSe_2526/DataVis/lab2/TourismAssociation_Dataset.json")
    df = load_tourism_data(data_path)
    df, X3d_scaled = cluster_geodata(df)

    X2 = df[["Latitude","Longitude"]].values
    scaler2 = StandardScaler()
    X2_scaled = scaler2.fit_transform(X2)
    xx, yy, density = compute_kde_grid(X2_scaled, bandwidth=0.25, grid_size=150)

    labels_hdb = df["cluster_hdbscan"].values
    plot_kde_with_clusters(X2_scaled, labels_hdb, xx, yy, density)


if __name__ == "__main__":
    main()