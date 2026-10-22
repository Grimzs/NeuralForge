"""
Нейросеть-граф произвольной структуры.

Отличие от MLP: нейроны и связи задаются вручную пользователем.
Каждая связь — отдельный вес. Удаление связи удаляет её вес.
Вставка нейрона в разрыв связи может сохранить сигнал
(w1 = w, w2 = 1). Backprop идёт по топологическому порядку DAG.
"""
import numpy as np

ACTS = {
    "linear":  (lambda z: z,                              lambda z, a: np.ones_like(z)),
    "relu":    (lambda z: np.maximum(0.0, z),             lambda z, a: (z > 0).astype(float)),
    "tanh":    (lambda z: np.tanh(z),                     lambda z, a: 1 - a ** 2),
    "sigmoid": (lambda z: 1 / (1 + np.exp(-np.clip(z, -500, 500))),
                lambda z, a: a * (1 - a)),
}


class GraphNetwork:
    """Граф нейронов с ручными связями и обучением backprop."""

    def __init__(self, graph: dict, seed: int = 42):
        self.loss_name = graph.get("loss", "bce")
        self.neurons: dict[int, dict] = {}
        for n in graph.get("neurons", []):
            self.neurons[int(n["id"])] = {
                "kind": n.get("kind", "hidden"),
                "activation": n.get("activation"),
                "comment": n.get("comment", ""),
                "b": float(n.get("b", 0.0)),  # смещение сохраняется при загрузке
                "frozen": not bool(n.get("trainable", True)),  # ❄ заморозка весов
            }
        if not self.neurons:
            raise ValueError("В графе нет нейронов")

        ids = list(self.neurons)
        self.input_ids = [i for i in ids if self.neurons[i]["kind"] == "input"]
        self.output_ids = [i for i in ids if self.neurons[i]["kind"] == "output"]
        if not self.input_ids:
            raise ValueError("Нет входных нейронов")
        if not self.output_ids:
            raise ValueError("Нет выходных нейронов")

        # Активации по умолчанию: скрытые tanh, выход — sigmoid/mse-линия
        for i in ids:
            if self.neurons[i]["kind"] == "hidden":
                if self.neurons[i]["activation"] not in ACTS:
                    self.neurons[i]["activation"] = "tanh"
            elif self.neurons[i]["kind"] == "output":
                if self.neurons[i]["activation"] not in ACTS:
                    self.neurons[i]["activation"] = "sigmoid" if self.loss_name == "bce" else "linear"

        # Связи: кортежи (from, to), опциональный стартовый вес из конфига
        self.edges: list[tuple[int, int]] = []
        self.W: dict[tuple[int, int], float] = {}
        rng = np.random.default_rng(seed)  # один генератор — разные стартовые веса
        for e in graph.get("edges", []):
            key = (int(e["from"]), int(e["to"]))
            if key[0] == key[1]:
                raise ValueError("Связь нейрона с самим собой запрещена")
            if key in self.W:
                continue  # дубликат
            self.edges.append(key)
            self.W[key] = float(e.get("w", rng.normal(0, 0.5)))

        # Смещение (bias) у каждого вычисляемого нейрона — без него
        # многие задачи (например XOR) нерешаемы в принципе
        self.B: dict[int, float] = {i: n["b"] for i, n in self.neurons.items()
                                    if n["kind"] != "input"}

        self._validate_dag()

        self.incoming: dict[int, list[int]] = {i: [] for i in ids}
        self.outgoing: dict[int, list[int]] = {i: [] for i in ids}
        for p, q in self.edges:
            self.incoming[q].append(p)
            self.outgoing[p].append(q)
        self.order = self._topo()

        self.loss_history: list[float] = []
        self._gw: dict[tuple[int, int], float] = {}
        self._gb: dict[int, float] = {}

    # ── Проверки структуры ──
    def _validate_dag(self):
        for p, q in self.edges:
            if p not in self.neurons or q not in self.neurons:
                raise ValueError(f"Связь ({p}→{q}) ссылается на несуществующий нейрон")
            if self.neurons[q]["kind"] == "input":
                raise ValueError("Связь не может вести во входной нейрон")
            if self.neurons[p]["kind"] == "output":
                raise ValueError("Связь не может исходить из выходного нейрона")
        self._topo()

    def _topo(self) -> list[int]:
        """Топологический порядок не-входных нейронов (Kahn)."""
        ids = list(self.neurons)
        indeg = {i: 0 for i in ids}
        for _, q in self.edges:
            indeg[q] += 1
        queue = sorted(i for i in ids if indeg[i] == 0)
        order = []
        while queue:
            n = queue.pop(0)
            order.append(n)
            for p, q in self.edges:
                if p == n:
                    indeg[q] -= 1
                    if indeg[q] == 0:
                        queue.append(q)
        if len(order) != len(ids):
            raise ValueError("Граф содержит цикл — обучение невозможно")
        # входные в конец не попадут в order? Попадут (indeg 0), но они не вычисляются.
        return [i for i in order if self.neurons[i]["kind"] != "input"]

    # ── Прямой проход ──
    def forward(self, X) -> tuple[dict, dict]:
        X = np.asarray(X, dtype=float)
        m = X.shape[0]
        A: dict[int, np.ndarray] = {}
        for k, i in enumerate(self.input_ids):
            if k >= X.shape[1]:
                raise ValueError(f"Данных не хватает: вход {i} ждёт столбец {k}")
            A[i] = X[:, k]
        Z: dict[int, np.ndarray] = {}
        for i in self.order:
            z = np.zeros(m) + self.B[i]
            for p in self.incoming[i]:
                z = z + self.W[(p, i)] * A[p]
            Z[i] = z
            A[i] = ACTS[self.neurons[i]["activation"]][0](z)
        return A, Z

    def predict(self, X) -> np.ndarray:
        A, _ = self.forward(X)
        return np.stack([A[o] for o in self.output_ids], axis=1)

    # ── Один шаг обучения (full batch) ──
    def _step(self, X, y, lr) -> float:
        A, Z = self.forward(X)
        out = np.stack([A[o] for o in self.output_ids], axis=1)
        m = X.shape[0]

        # dL/da на выходе (усреднение по батчу делает .mean() в градиентах,
        # здесь учитываем только число выходов)
        n_out = max(1, len(self.output_ids))
        if self.loss_name == "mse":
            loss = float(np.mean((out - y) ** 2))
            d_out = 2.0 * (out - y) / n_out
            combined = False
        else:  # bce (выходы обязаны быть sigmoid)
            eps = 1e-9
            oc = np.clip(out, eps, 1 - eps)
            loss = float(-np.mean(y * np.log(oc) + (1 - y) * np.log(1 - oc)))
            d_out = (out - y) / n_out
            combined = True

        # dL/dz для выходов
        DZ: dict[int, np.ndarray] = {}
        G_da: dict[int, np.ndarray] = {}
        for k, o in enumerate(self.output_ids):
            if combined:
                DZ[o] = d_out[:, k]  # sigmoid+bce: уже объединённый градиент
            else:
                DZ[o] = d_out[:, k] * ACTS[self.neurons[o]["activation"]][1](Z[o], A[o])

        # Обратный проход по топологии
        for nid in reversed(self.order):
            dz = DZ.get(nid)
            if dz is None:
                g = G_da.get(nid, np.zeros(m))
                dz = g * ACTS[self.neurons[nid]["activation"]][1](Z[nid], A[nid])
                DZ[nid] = dz
            self._gb[nid] = float(dz.mean())  # градиент смещения
            for p in self.incoming[nid]:
                self._gw[(p, nid)] = float((A[p] * dz).mean())
                G_da[p] = G_da.get(p, np.zeros(m)) + self.W[(p, nid)] * dz

        for e, gw in self._gw.items():
            if not self.neurons[e[1]]["frozen"]:   # ❄ замороженный нейрон не учит входные веса
                self.W[e] -= lr * gw
        for i, gb in self._gb.items():
            if not self.neurons[i]["frozen"]:      # …и своё смещение
                self.B[i] -= lr * gb
        self._gw.clear()
        self._gb.clear()
        return loss

    def fit(self, X, y, epochs=100, lr=0.5, batch_size=None):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        m = len(X)
        for _ in range(max(1, epochs)):
            if batch_size and batch_size < m:
                idx = np.random.permutation(m)
                losses = []
                for s in range(0, m, batch_size):
                    b = idx[s:s + batch_size]
                    losses.append(self._step(X[b], y[b], lr))
                self.loss_history.append(float(np.mean(losses)))
            else:
                self.loss_history.append(self._step(X, y, lr))
        return self

    # ── Совместимость структур / перенос весов ──
    def same_structure(self, graph: dict) -> bool:
        try:
            other = GraphNetwork(graph)
        except ValueError:
            return False
        return (set(other.W) == set(self.W)
                and other.input_ids == self.input_ids
                and other.output_ids == self.output_ids
                and other.loss_name == self.loss_name
                and all(other.neurons[i]["activation"] == self.neurons[i]["activation"]
                        for i in self.neurons)
                and all(other.neurons[i]["frozen"] == self.neurons[i]["frozen"]
                        for i in self.neurons))

    def carry_weights(self, old: "GraphNetwork"):
        """Переносит веса и смещения из старой сети (после правки графа)."""
        carried = 0
        for e in self.W:
            if e in old.W:
                self.W[e] = old.W[e]
                carried += 1
        for i in self.B:
            if i in old.B:
                self.B[i] = old.B[i]
        self.loss_history = list(old.loss_history)
        return carried

    # ── Экспорт ──
    def weights_payload(self) -> list[dict]:
        return [{"from": p, "to": q, "w": round(w, 5)} for (p, q), w in self.W.items()]

    def biases_payload(self) -> dict:
        return {str(i): round(b, 5) for i, b in self.B.items()}

    def summary(self) -> str:
        total_w = len(self.edges)
        n_frozen = sum(1 for n in self.neurons.values() if n["frozen"])
        return (f"GraphNetwork: {len(self.neurons)} нейронов "
                f"(вх {len(self.input_ids)} / вых {len(self.output_ids)}), "
                f"{total_w} связей-весов, loss={self.loss_name}"
                + (f", заморожено {n_frozen}" if n_frozen else ""))
