"""KNN y escalado estándar implementados desde cero.

Reglas:
- Los cálculos se hacen solo con NumPy. No se puede usar sklearn.neighbors, sklearn.preprocessing
  ni scipy.spatial.
- KNNClasificador hereda de ClassifierMixin y BaseEstimator solo para poder usarse dentro de un
  Pipeline y de GridSearchCV de scikit-learn.
- Nada de bucles sobre filas para calcular distancias: se calculan con operaciones vectorizadas,
  por lotes de `tamano_lote` filas para no llenar la memoria.

Verificá tu trabajo con:  python -m pytest -q tests/test_knn.py
"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class EscaladorEstandar:
    """Estandariza cada columna: (x - media) / desvío.

    Atributos que crea fit: media_ y desvio_ (arrays de forma (n_columnas,)).
    - El desvío es el poblacional (np.std con ddof=0), igual que StandardScaler.
    - Si una columna es constante (desvío 0), se usa desvío 1 para no dividir por cero.
    """

    def fit(self, X):
        X_arr = np.asarray(X, dtype=float)
        self.media_ = np.mean(X_arr, axis=0)
        desvio = np.std(X_arr, axis=0, ddof=0)
        self.desvio_ = np.where(desvio == 0, 1.0, desvio)
        return self

    def transform(self, X):
        X_arr = np.asarray(X, dtype=float)
        return (X_arr - self.media_) / self.desvio_

    def fit_transform(self, X):
        return self.fit(X).transform(X)


class KNNClasificador(ClassifierMixin, BaseEstimator):
    """Clasificador de k vecinos más cercanos (búsqueda por fuerza bruta).

    Parámetros
    ----------
    k : int, por defecto 5
        Cantidad de vecinos.
    pesos : "uniforme" o "distancia"
        "uniforme": cada vecino vota con peso 1.
        "distancia": cada vecino vota con peso 1 / distancia. Si la consulta coincide exactamente con
        uno o más ejemplos (distancia 0), votan solo esos ejemplos, con peso 1 cada uno.
    metrica : "euclidea" o "manhattan"
    tamano_lote : int, por defecto 1000
        Cantidad de filas de consulta que se procesan juntas al calcular distancias.

    Atributos que crea fit
    ----------------------
    X_ : array (n_entrenamiento, n_variables) con los datos de entrenamiento, en float.
    classes_ : array (n_clases,) con las clases ordenadas (np.unique).
    y_codificada_ : array (n_entrenamiento,) con el índice de la clase de cada ejemplo en classes_.
    """

    def __init__(self, k=5, pesos="uniforme", metrica="euclidea", tamano_lote=1000):
        self.k = k
        self.pesos = pesos
        self.metrica = metrica
        self.tamano_lote = tamano_lote

    def fit(self, X, y):
        """KNN no estima parámetros: guarda los datos. Devuelve self."""
        self.X_ = np.asarray(X, dtype=float)
        self.classes_, self.y_codificada_ = np.unique(y, return_inverse=True)

        metrica = (
            self.metrica.lower().replace("í", "i")
            if isinstance(self.metrica, str)
            else self.metrica
        )
        if metrica == "euclidea":
            self.norm_X_train_ = np.sum(self.X_**2, axis=1, keepdims=True).T
        return self

    def _distancias(self, X):
        """Matriz (n_consultas, n_entrenamiento) con la distancia entre cada consulta y cada ejemplo.

        Pista para la euclídea: ||a - b||² = ||a||² + ||b||² - 2 a·b. Por redondeo puede dar valores
        levemente negativos: recortalos a 0 antes de la raíz cuadrada.
        """
        X_query = np.asarray(X, dtype=float)
        if X_query.ndim == 1:
            X_query = np.atleast_2d(X_query)

        n_queries = X_query.shape[0]
        n_train = self.X_.shape[0]
        dists = np.empty((n_queries, n_train), dtype=float)

        tamano_lote = self.tamano_lote if self.tamano_lote > 0 else n_queries
        metrica = (
            self.metrica.lower().replace("í", "i")
            if isinstance(self.metrica, str)
            else self.metrica
        )

        for start in range(0, n_queries, tamano_lote):
            end = min(start + tamano_lote, n_queries)
            X_batch = X_query[start:end]

            if metrica == "euclidea":
                norm_batch = np.sum(X_batch**2, axis=1, keepdims=True)
                dot_prod = np.dot(X_batch, self.X_.T)
                dist_sq = norm_batch + self.norm_X_train_ - 2 * dot_prod
                dists[start:end] = np.sqrt(np.maximum(0.0, dist_sq))
            elif metrica == "manhattan":
                dists[start:end] = np.sum(
                    np.abs(
                        X_batch[:, np.newaxis, :] - self.X_[np.newaxis, :, :]
                    ),
                    axis=2,
                )
            else:
                raise ValueError(f"Métrica no soportada: {self.metrica}")

        return dists

    def kneighbors(self, X):
        """Devuelve (distancias, indices), ambos de forma (n_consultas, k), ordenados de menor a mayor distancia.

        Pista: np.argpartition encuentra los k menores sin ordenar toda la fila; después ordená solo esos k.
        """
        dists = self._distancias(X)
        n_queries, n_train = dists.shape
        k = min(self.k, n_train)

        if n_train <= k:
            indices = np.argsort(dists, axis=1)
            row_idx = np.arange(n_queries)[:, np.newaxis]
            sorted_dists = dists[row_idx, indices]
            return sorted_dists, indices

        part_idx = np.argpartition(dists, k - 1, axis=1)[:, :k]
        row_idx = np.arange(n_queries)[:, np.newaxis]
        part_dists = dists[row_idx, part_idx]

        sort_in_k = np.argsort(part_dists, axis=1)
        top_k_indices = part_idx[row_idx, sort_in_k]
        top_k_dists = part_dists[row_idx, sort_in_k]

        return top_k_dists, top_k_indices

    def predict_proba(self, X):
        """Proporción (ponderada según `pesos`) de vecinos de cada clase. Columnas en el orden de classes_."""
        top_k_dists, top_k_indices = self.kneighbors(X)
        n_queries = top_k_dists.shape[0]
        n_classes = len(self.classes_)

        top_k_y = self.y_codificada_[top_k_indices]

        pesos = (
            self.pesos.lower().replace("í", "i")
            if isinstance(self.pesos, str)
            else self.pesos
        )

        if pesos == "uniforme":
            weights = np.ones_like(top_k_dists, dtype=float)
        elif pesos == "distancia":
            is_zero = top_k_dists == 0
            has_zero = np.any(is_zero, axis=1, keepdims=True)

            with np.errstate(divide="ignore"):
                inv_dists = np.where(is_zero, 0.0, 1.0 / top_k_dists)

            weights = np.where(has_zero, np.where(is_zero, 1.0, 0.0), inv_dists)
        else:
            raise ValueError(f"Pesos no soportados: {self.pesos}")

        probas = np.zeros((n_queries, n_classes), dtype=float)
        row_indices_flat = np.repeat(
            np.arange(n_queries), weights.shape[1]
        )
        col_indices_flat = top_k_y.ravel()
        weights_flat = weights.ravel()

        np.add.at(probas, (row_indices_flat, col_indices_flat), weights_flat)

        # Normalizar luego de sumar los votos evita probabilidades mayores a 1
        # por el redondeo al acumular pesos previamente normalizados.
        probas /= probas.sum(axis=1, keepdims=True)
        return probas

    def predict(self, X):
        """Clase con mayor probabilidad. En caso de empate, la primera en classes_ (como np.argmax)."""
        probas = self.predict_proba(X)
        max_idx = np.argmax(probas, axis=1)
        return self.classes_[max_idx]
