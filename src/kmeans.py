"""Модуль реализации алгоритма кластеризации k-means на Python."""

import math
import random
from typing import List, Optional


class KMeans:
    """Класс для кластеризации данных методом k-means."""

    def __init__(
        self,
        n_clusters: int = 3,
        max_iter: int = 100,
        seed: Optional[int] = None,
    ) -> None:
        """Инициализирует параметры алгоритма K-Means.

        Args:
            n_clusters: Количество кластеров.
            max_iter: Максимальное количество итераций.
            seed: Зерно генератора случайных чисел.
        """
        self.n_clusters: int = n_clusters
        self.max_iter: int = max_iter
        self.seed: Optional[int] = seed
        self.centroids: List[List[float]] = []
        self.inertia_: float = 0.0

    def _init_centroids(self, X: List[List[float]]) -> List[List[float]]:
        """Инициализирует начальные центры кластеров случайным образом из данных.

        Args:
            X: Исходный набор точек.

        Returns:
            Список координат начальных центроидов.
        """
        if self.seed is not None:
            random.seed(self.seed)
        return random.sample(X, self.n_clusters)

    def _euclidean_distance(self, p1: List[float], p2: List[float]) -> float:
        """Вычисляет евклидово расстояние между двумя точками.

        Args:
            p1: Первая точка.
            p2: Вторая точка.

        Returns:
            Евклидово расстояние.
        """
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))

    def _assign(self, X: List[List[float]]) -> List[int]:
        """Назначает каждую точку к ближайшему центроиду.

        Args:
            X: Исходный набор точек.

        Returns:
            Список индексов ближайших центроидов для каждой точки.
        """
        labels: List[int] = []
        for point in X:
            distances: List[float] = [
                self._euclidean_distance(point, center) for center in self.centroids
            ]
            labels.append(distances.index(min(distances)))
        return labels

    def _update(self, X: List[List[float]], labels: List[int]) -> List[List[float]]:
        """Пересчитывает координаты центроидов как среднее арифметическое точек.

        Args:
            X: Исходный набор точек.
            labels: Текущие метки кластеров для каждой точки.

        Returns:
            Обновленный список координат центроидов.
        """
        if not X:
            return [[0.0] for _ in range(self.n_clusters)]

        dim: int = len(X[0])
        new_centroids: List[List[float]] = [[0.0] * dim for _ in range(self.n_clusters)]
        counts: List[int] = [0] * self.n_clusters

        for point, label in zip(X, labels):
            for i in range(dim):
                new_centroids[label][i] += point[i]
            counts[label] += 1

        for i in range(self.n_clusters):
            if counts[i] > 0:
                new_centroids[i] = [val / counts[i] for val in new_centroids[i]]
            else:
                new_centroids[i] = self.centroids[i]
        return new_centroids

    def fit(self, X: List[List[float]]) -> List[int]:
        """Обучает модель на входных данных X.

        Args:
            X: Исходный набор точек.

        Returns:
            Итоговые метки кластеров для точек.
        """
        self.centroids = self._init_centroids(X)

        for _ in range(self.max_iter):
            labels: List[int] = self._assign(X)
            new_centroids: List[List[float]] = self._update(X, labels)

            if new_centroids == self.centroids:
                break
            self.centroids = new_centroids

        final_labels: List[int] = self._assign(X)
        self.inertia_ = self.inertia(X, final_labels)
        return final_labels

    def predict(self, X: List[List[float]]) -> List[int]:
        """Возвращает метки ближайших кластеров для новых точек.

        Args:
            X: Набор новых точек.

        Returns:
            Список меток кластеров.
        """
        return self._assign(X)

    def inertia(self, X: List[List[float]], labels: List[int]) -> float:
        """Вычисляет сумму квадратов расстояний от точек до центров кластеров.

        Args:
            X: Исходный набор точек.
            labels: Метки кластеров для точек.

        Returns:
            Значение критерия inertia (СКО).
        """
        total_sq_dist: float = 0.0
        for point, label in zip(X, labels):
            center: List[float] = self.centroids[label]
            total_sq_dist += sum((a - b) ** 2 for a, b in zip(point, center))
        return total_sq_dist


def generate_mock_data(
    samples: int,
    seed: int,
    features: int = 2,
    n_clusters: int = 3,
) -> tuple[list[list[float]], list[int]]:
    """Генерирует синтетические данные из 3 кластеров с помощью NumPy.

    Args:
        samples: Общее количество точек.
        seed: Зерно генератора случайных чисел.
        features: Размерность пространства.
        n_clusters: Количество генерируемых кластеров.

    Returns:
        Кортеж из списка координат точек X и списка истинных меток.
    """
    import numpy as np

    np.random.seed(seed)

    centers = np.random.uniform(-10.0, 10.0, size=(n_clusters, features)).tolist()

    samples_per_center: int = samples // n_clusters

    x_data: list[list[float]] = []
    true_labels: list[int] = []

    for i, center in enumerate(centers):
        points = np.random.normal(
            loc=center, scale=0.8, size=(samples_per_center, features)
        ).tolist()
        x_data.extend(points)
        true_labels.extend([i] * samples_per_center)

    return x_data, true_labels


def plot_clusters(
    x_data: list[list[float]],
    labels: list[int],
    centroids: list[list[float]],
    n_clusters: int,
    filename: str = "docs/clusters_plot.png",
) -> None:
    """Строит и сохраняет график кластеров с центроидами.

    Args:
        x_data: Список координат точек.
        labels: Присвоенные метки кластеров.
        centroids: Координаты центроидов.
        n_clusters: Количество кластеров.
        filename: Путь для сохранения изображения.
    """
    from pathlib import Path
    import matplotlib.pyplot as plt

    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    x_coords: list[float] = [p[0] for p in x_data]
    y_coords: list[float] = [p[1] for p in x_data]

    plt.figure(figsize=(16, 9))
    plt.scatter(
        x_coords, y_coords, c=labels, cmap="viridis", alpha=0.6, edgecolors="black"
    )
    c_x: list[float] = [c[0] for c in centroids]
    c_y: list[float] = [c[1] for c in centroids]
    plt.scatter(c_x, c_y, c="red", s=200, marker="X", label="Центроиды")

    plt.title(f"Результат кластеризации k-means (k={n_clusters})")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.close()


def plot_elbow(
    x_data: list[list[float]],
    max_iter: int,
    seed: int,
    filename: str = "docs/elbow_plot.png",
) -> None:
    """Строит график метода локтя для выбора оптимального k.

    Args:
        x_data: Список координат точек.
        max_iter: Максимальное число итераций.
        seed: Зерно генератора случайных чисел.
        filename: Путь для сохранения изображения.
    """
    from pathlib import Path
    import matplotlib.pyplot as plt

    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    k_values: list[int] = list(range(1, 7))
    inertias: list[float] = []

    for k in k_values:
        model = KMeans(n_clusters=k, max_iter=max_iter, seed=seed)
        model.fit(x_data)
        inertias.append(model.inertia_)

    plt.figure(figsize=(8, 5))
    plt.plot(k_values, inertias, marker="o", linestyle="--", color="blue")
    plt.xlabel("Количество кластеров (k)")
    plt.ylabel("СКО внутри кластеров (Inertia)")
    plt.title("Метод локтя для определения оптимального k")
    plt.grid(True)
    plt.savefig(filename)
    plt.close()


def main() -> None:
    """Точка входа CLI: разбирает аргументы и запускает кластеризацию."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Кластеризация методом k-means на Python."
    )
    parser.add_argument("--samples", type=int, default=300, help="Количество точек")
    parser.add_argument("--k", type=int, default=3, help="Количество кластеров")
    parser.add_argument("--iter", type=int, default=100, help="Максимум итераций")
    parser.add_argument("--seed", type=int, default=42, help="Зерно случайности")
    parser.add_argument(
        "--elbow", action="store_true", help="Построить график метода локтя"
    )
    args = parser.parse_args()

    if args.samples <= 0 or args.k <= 0 or args.iter <= 0:
        print("Ошибка: параметры --samples, --k и --iter должны быть больше 0.")
        return

    x_data, _ = generate_mock_data(
        samples=args.samples, seed=args.seed, n_clusters=args.k
    )

    if args.elbow:
        plot_elbow(x_data, args.iter, args.seed)
        print("График метода локтя сохранён в docs/elbow_plot.png")
        return

    if args.k > len(x_data):
        print("Ошибка: количество кластеров не может превышать число точек.")
        return

    model = KMeans(n_clusters=args.k, max_iter=args.iter, seed=args.seed)
    labels = model.fit(x_data)
    counts: dict[int, int] = {i: labels.count(i) for i in range(args.k)}

    print(f"Точек в кластерах: {counts}")
    print(f"Inertia (СКО): {model.inertia_:.4f}")
    plot_clusters(x_data, labels, model.centroids, args.k)
    print("График кластеров сохранён в docs/clusters_plot.png")


if __name__ == "__main__":
    main()
