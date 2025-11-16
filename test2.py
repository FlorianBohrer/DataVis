import json
import pandas as pd

import numpy as np
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
from matplotlib import cm
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import StandardScaler





# Daten laden
data_path = Path("/Users/florianbohrer/Documents/0uni/WiSe_2526/DataVis/lab2/TourismAssociation_Dataset.json")

with open(data_path, "r", encoding="utf8") as f:
    data = json.load(f)

df = pd.json_normalize(data, sep=".")

print("Form des DataFrames:", df.shape)
print(df[["Id", "Latitude", "Longitude", "Altitude"]].head())


# Feature Auswahl
feature_cols = ["Latitude", "Longitude", "Altitude"]
df_geo = df.dropna(subset=feature_cols).copy()
X = df_geo[feature_cols].values

print("Anzahl verwendeter Eintraege:", X.shape[0])



# Daten vorbereiten
X = df_geo[["Latitude", "Longitude"]].values

# Clusterlabels waehlen
# Du kannst auch df_geo["cluster_kmeans"] nehmen
labels = df_geo["cluster_hdbscan"].values

# Skalierung
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# KDE Modell
kde = KernelDensity(bandwidth=0.25)
kde.fit(X_scaled)

# Gitter definieren
x_min, x_max = X_scaled[:, 0].min() - 0.5, X_scaled[:, 0].max() + 0.5
y_min, y_max = X_scaled[:, 1].min() - 0.5, X_scaled[:, 1].max() + 0.5

xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 150),
    np.linspace(y_min, y_max, 150)
)

grid = np.vstack([xx.ravel(), yy.ravel()]).T
density = np.exp(kde.score_samples(grid)).reshape(xx.shape)

# 3D Plot
fig = plt.figure(figsize=(20, 15))
ax = fig.add_subplot(111, projection="3d")

# KDE Surface
cmap = cm.viridis
norm = plt.Normalize(vmin=density.min(), vmax=density.max())

surf = ax.plot_surface(
    xx, yy, density,
    cmap=cmap,
    norm=norm,
    rstride=1, cstride=1,
    linewidth=0,
    antialiased=True,
    alpha=0.75
)

fig.colorbar(surf, ax=ax, shrink=0.6, label="Density Level")

# Clusterpunkte auf der Surface darstellen
# wir heben die Punkte leicht ueber die Flaeche an
scatter_z = np.zeros_like(labels, dtype=float)
for i in range(len(scatter_z)):
    # naechster Gitterpunkt finden
    ix = np.argmin(np.abs(xx[0] - X_scaled[i, 0]))
    iy = np.argmin(np.abs(yy[:, 0] - X_scaled[i, 1]))
    scatter_z[i] = density[iy, ix] + 0.02

ax.scatter(
    X_scaled[:, 0],
    X_scaled[:, 1],
    scatter_z,
    c=labels,
    cmap="tab10",
    s=40,
    depthshade=True
)

ax.set_xlabel("Latitude (scaled)")
ax.set_ylabel("Longitude (scaled)")
ax.set_zlabel("Density")
plt.title("KDE Density Surface mit Cluster Overlay")

ax.view_init(elev=30, azim=235)

plt.show()
