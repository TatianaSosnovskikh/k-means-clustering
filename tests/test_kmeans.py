"""Модуль тестирования алгоритма k-means с помощью pytest."""

from typing import List, Tuple
import pytest
from src.kmeans import KMeans


@pytest.fixture
def sample_data() -> Tuple[List[List[float]], List[int]]:
    """Фикстура для генерации простых, явно разделенных тестовых данных.

    Генерирует 3 явных кластера (по 3 точки в каждом) вокруг далеких центров.

    Returns:
        Кортеж из списка точек X и списка их истинных меток.
    """
    centers = [[0.0, 0.0], [1000.0, 1000.0], [-1000.0, -1000.0]]

    X: List[List[float]] = []
    true_labels: List[int] = []

    for cluster_id, center in enumerate(centers):
        X.append([center[0], center[1]])
        X.append([center[0] + 0.01, center[1] + 0.01])
        X.append([center[0] - 0.01, center[1] - 0.01])

        true_labels.extend([cluster_id] * 3)
    return X, true_labels


def test_reproducibility(sample_data: Tuple[List[List[float]], List[int]]) -> None:
    """Проверяет воспроизводимость результатов при одинаковом seed."""
    X, _ = sample_data

    model1 = KMeans(n_clusters=3, max_iter=10, seed=42)
    labels1 = model1.fit(X)

    model2 = KMeans(n_clusters=3, max_iter=10, seed=42)
    labels2 = model2.fit(X)

    assert labels1 == labels2, (
        "Ошибка: Одинаковый seed привел к разным меткам кластеров!"
    )
    assert model1.centroids == model2.centroids, (
        "Ошибка: Одинаковый seed привел к разным координатам центроидов!"
    )


def test_inertia_monotonicity(
    sample_data: Tuple[List[List[float]], List[int]],
) -> None:
    """Проверяет монотонность: при росте итераций inertia падает или не растет."""
    X, _ = sample_data

    # Модель с 1 итерацией (почти не обучалась)
    model_low = KMeans(n_clusters=3, max_iter=1, seed=42)
    model_low.fit(X)

    # Модель с 50 итерациями (успела сойтись к оптимуму)
    model_high = KMeans(n_clusters=3, max_iter=50, seed=42)
    model_high.fit(X)

    assert model_high.inertia_ <= model_low.inertia_, (
        f"Ошибка: СКО увеличилось при росте итераций! "
        f"Inertia(1 iter)={model_low.inertia_}, Inertia(50 iter)={model_high.inertia_}"
    )


def test_different_k(sample_data: Tuple[List[List[float]], List[int]]) -> None:
    """Проверяет, что при разных k генерируется разное число кластеров и центроидов."""
    X, _ = sample_data

    for k in range(2, 5):
        model = KMeans(n_clusters=k, seed=42)
        labels = model.fit(X)
        assert len(model.centroids) == k, (
            f"Ошибка: Для k={k} создано неверное число центроидов "
            f"({len(model.centroids)})!"
        )
        assert len(set(labels)) <= k, (
            f"Ошибка: Количество уникальных меток ({len(set(labels))}) "
            f"превысило заданное k={k}!"
        )


def test_cluster_accuracy(
    sample_data: Tuple[List[List[float]], List[int]],
) -> None:
    """Проверяет точность: точки из одного истинного кластера группируются вместе."""
    X, _ = sample_data
    model = KMeans(n_clusters=3, seed=42)
    predicted_labels = model.fit(X)

    # Разбиваем предсказанные метки на исходные группы по 3 точки
    group1 = predicted_labels[0:3]
    group2 = predicted_labels[3:6]
    group3 = predicted_labels[6:9]

    # set() оставляет только уникальные значения.
    # Если в группе все точки принадлежат одному кластеру, длина сета будет равна 1.
    assert len(set(group1)) == 1, (
        f"Ошибка: Точки из первого кластера разбились на разные группы: {group1}!"
    )
    assert len(set(group2)) == 1, (
        f"Ошибка: Точки из второго кластера разбились на разные группы: {group2}!"
    )
    assert len(set(group3)) == 1, (
        f"Ошибка: Точки из третьего кластера разбились на разные группы: {group3}!"
    )


def test_generate_mock_data() -> None:
    """Проверяет корректность работы генератора синтетических данных."""
    from src.kmeans import generate_mock_data

    samples: int = 300
    seed: int = 42
    features: int = 2
    n_clusters: int = 3

    X, labels = generate_mock_data(
        samples=samples,
        seed=seed,
        features=features,
        n_clusters=n_clusters,
    )

    # Проверяем типы возвращаемых данных
    assert isinstance(X, list), "Ошибка: Матрица X должна быть списком!"
    assert isinstance(labels, list), "Ошибка: Список меток должен быть списком!"

    # Проверяем размерность данных
    assert len(X) == samples, (
        f"Ошибка: Количество сгенерированных точек ({len(X)}) не равно {samples}!"
    )
    assert len(labels) == samples, (
        f"Ошибка: Количество сгенерированных меток ({len(labels)}) не равно {samples}!"
    )

    # Проверяем, что каждая точка двумерна (имеет X и Y координаты)
    assert all(len(point) == features for point in X), (
        f"Ошибка: Не все точки имеют заданную размерность пространства ({features})!"
    )

    # Проверяем распределение меток
    for i in range(n_clusters):
        assert labels.count(i) == samples // n_clusters, (
            f"Ошибка: В кластере {i} находится {labels.count(i)} точек "
            f"вместо {samples // n_clusters}!"
        )
