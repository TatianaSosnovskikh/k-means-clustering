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
