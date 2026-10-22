"""
Сборка нейросети из пользовательской конфигурации конструктора.

Конфигурация (JSON):
{
  "layers": [
    {"kind": "input",  "size": 2, "comment": "вход: x1,x2"},
    {"kind": "hidden", "size": 4, "activation": "relu", "comment": "скрытый слой"},
    {"kind": "output", "size": 1, "activation": "sigmoid", "comment": "выход"}
  ],
  "hidden_activation": "relu",
  "output_activation": "sigmoid",
  "loss": "bce"
}
"""
from .network import NeuralNetwork

KINDS = ("input", "hidden", "output")


def validate_config(config: dict) -> list[dict]:
    """Проверяет конфиг и возвращает нормализованный список слоёв."""
    layers = config.get("layers")
    if not isinstance(layers, list) or len(layers) < 3:
        raise ValueError("Нужно минимум 3 слоя: входной, скрытый, выходной")

    norm = []
    for i, l in enumerate(layers):
        kind = l.get("kind")
        if kind not in KINDS:
            raise ValueError(f"Слой {i}: неизвестный тип '{kind}'")
        size = int(l.get("size", 1))
        if size < 1:
            raise ValueError(f"Слой {i}: размер должен быть >= 1")
        if size > 64:
            raise ValueError(f"Слой {i}: максимум 64 нейрона на слой")
        norm.append({
            "kind": kind,
            "size": size,
            "activation": l.get("activation"),
            "comment": l.get("comment", ""),
        })

    if norm[0]["kind"] != "input":
        raise ValueError("Первый слой должен быть входным")
    if norm[-1]["kind"] != "output":
        raise ValueError("Последний слой должен быть выходным")
    if not any(l["kind"] == "hidden" for l in norm):
        raise ValueError("Нужен хотя бы один скрытый слой")
    return norm


def build_network(config: dict, seed: int = 42) -> NeuralNetwork:
    """Строит NeuralNetwork по конфигурации конструктора.

    Активация берётся из поля слоя (для hidden/output), иначе — глобальная.
    """
    norm = validate_config(config)
    sizes = [l["size"] for l in norm]
    hidden_act = config.get("hidden_activation", "relu")
    out_act = config.get("output_activation", "sigmoid")
    loss = config.get("loss", "bce")
    net = NeuralNetwork(sizes, hidden_act, out_act, loss, seed=seed)

    # Переопределение активации на уровне отдельного слоя
    for i, l in enumerate(norm):
        if i == 0 or not l.get("activation"):
            continue
        net.layers[i - 1].activation = l["activation"]
    return net


def same_architecture(net: NeuralNetwork, config: dict) -> bool:
    """Совпадает ли архитектура (для продолжения обучения без сброса весов)."""
    try:
        norm = validate_config(config)
    except ValueError:
        return False
    if net.layer_sizes != [l["size"] for l in norm]:
        return False
    for i, l in enumerate(norm):
        if i == 0:
            continue
        if net.layers[i - 1].activation != l.get("activation"):
            return False
    return net.loss_name == config.get("loss", "bce")


def weights_blob(net: NeuralNetwork) -> dict:
    """Веса сети в JSON-совместимом виде (для сохранения)."""
    return {
        "layer_sizes": net.layer_sizes,
        "hidden_activation": net.hidden_activation,
        "output_activation": net.output_activation,
        "loss": net.loss_name,
        "layers": [
            {"W": l.W.tolist(), "b": l.b.tolist(), "activation": l.activation}
            for l in net.layers
        ],
        "loss_history": net.loss_history,
    }


def restore_weights(net: NeuralNetwork, blob: dict) -> None:
    """Восстанавливает веса из weights_blob()."""
    import numpy as np
    for layer, ld in zip(net.layers, blob["layers"]):
        layer.W = np.array(ld["W"])
        layer.b = np.array(ld["b"])
        layer.activation = ld["activation"]
    net.loss_history = blob.get("loss_history", [])


def network_from_file(path: str) -> tuple[NeuralNetwork, dict]:
    """Загружает сохранённую сеть (веса + конфиг) из файла."""
    import json
    with open(path, "r", encoding="utf-8") as f:
        blob = json.load(f)
    net = build_network(blob["config"])
    restore_weights(net, blob["network"])
    return net, blob["config"]
