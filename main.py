import json
import numpy as np
from pathlib import Path
from typing import List

from fastapi import FastAPI, Request, HTTPException, UploadFile, File
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from openpyxl import load_workbook, Workbook
from openpyxl.utils import range_boundaries, get_column_letter

from neural import datasets as ds
from neural.graphnet import GraphNetwork

# ── Пути ──
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
NETS_DIR = DATA_DIR / "networks"
NETS_DIR.mkdir(parents=True, exist_ok=True)
NEURONS_DIR = DATA_DIR / "neurons"   # библиотека готовых нейронов (конструктор 1)
NEURONS_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR = DATA_DIR / "tables"     # xlsx-таблицы данных (страница «Таблицы»)
TABLES_DIR.mkdir(parents=True, exist_ok=True)
MODULES_DIR = DATA_DIR / "modules"   # библиотека модулей-блоков (выделенные фрагменты графа)
MODULES_DIR.mkdir(parents=True, exist_ok=True)

# ── Загрузка JSON ──
def load_json(filename: str) -> dict:
    with open(DATA_DIR / filename, "r", encoding="utf-8") as f:
        return json.load(f)

SITE_INFO = load_json("site_info.json")

# ── Приложение ──
app = FastAPI(title=SITE_INFO["name"], version=SITE_INFO["version"])
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "templates")
templates.env.globals["site"] = SITE_INFO

# Текущая сеть-граф (живёт в памяти процесса)
CURRENT: dict = {"net": None, "graph": None}


# ════════════════════════════════════════
# Страницы
# ════════════════════════════════════════
@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse(request=request, name="index.html", context={"active": "home"})


@app.get("/neurons")
async def neurons_page(request: Request):
    return templates.TemplateResponse(request=request, name="neurons.html", context={"active": "neurons"})


@app.get("/build")
async def build_page(request: Request):
    return templates.TemplateResponse(request=request, name="build.html", context={"active": "build"})


@app.get("/tables")
async def tables_page(request: Request):
    return templates.TemplateResponse(request=request, name="tables.html", context={"active": "tables"})


@app.get("/settings")
async def settings(request: Request):
    saved = sorted(p.stem for p in NETS_DIR.glob("*.json"))
    return templates.TemplateResponse(
        request=request, name="settings.html",
        context={"active": "settings", "saved_nets": saved},
    )


@app.get("/help")
async def help_page(request: Request):
    return templates.TemplateResponse(request=request, name="help.html", context={"active": "help"})


@app.get("/api/site")
async def api_site():
    return SITE_INFO


# ════════════════════════════════════════
# Датасеты-пресеты
# ════════════════════════════════════════
@app.get("/api/datasets")
async def api_datasets():
    """Готовые датасеты; клиент по ним строит стартовый граф."""
    np.random.seed(7)
    pairs = {
        "xor": ds.xor_dataset(),
        "circle": ds.circle_dataset(200),
        "spiral": ds.spiral_dataset(200),
        "linear": ds.linear_dataset(100),
    }
    meta = {
        "xor":    {"desc": "XOR", "loss": "bce", "inputs": 2, "outputs": 1},
        "circle": {"desc": "Внутри/вне круга", "loss": "bce", "inputs": 2, "outputs": 1},
        "spiral": {"desc": "Две спирали", "loss": "bce", "inputs": 2, "outputs": 1},
        "linear": {"desc": "y = 2x+1 (выход linear, mse)", "loss": "mse", "inputs": 1, "outputs": 1},
    }
    out = {}
    for name, (X, y) in pairs.items():
        out[name] = {**meta[name],
                     "X": np.round(X, 4).tolist(),
                     "y": np.round(y, 4).tolist()}
    return out


# ════════════════════════════════════════
# API сети-графа
# ════════════════════════════════════════
class TrainStepRequest(BaseModel):
    graph: dict            # {neurons:[{id,kind,activation,comment}], edges:[{from,to,w?}], loss}
    X: list[list[float]]
    y: list[list[float]]
    epochs: int = 100
    lr: float = 0.5
    batch_size: int | None = None
    reset: bool = False


class PredictRequest(BaseModel):
    X: list[list[float]]


class SaveRequest(BaseModel):
    name: str
    graph: dict


def _parse_matrix(rows, name: str) -> np.ndarray:
    try:
        m = np.asarray(rows, dtype=float)
    except (TypeError, ValueError):
        raise HTTPException(400, f"Поле '{name}': все значения должны быть числами")
    if m.ndim == 1:
        m = m.reshape(-1, 1)
    if m.shape[0] == 0:
        raise HTTPException(400, f"Поле '{name}': пусто")
    return m


def _build_or_carry(graph: dict, reset: bool) -> GraphNetwork:
    """Строит сеть; при изменении структуры переносит веса живых связей."""
    old = CURRENT.get("net")
    try:
        net = GraphNetwork(graph)
    except ValueError as e:
        raise HTTPException(400, str(e))
    if not reset and old is not None and not old.same_structure(graph):
        net.carry_weights(old)  # связи, что выжили, продолжают жить со своими весами
    if not reset and old is not None and old.same_structure(graph):
        net = old              # та же структура — обучаем ту же сеть
    return net


@app.post("/api/gnet/train_step")
async def api_gnet_train_step(req: TrainStepRequest):
    """Шаг обучения сети-графа (продолжение без сброса весов)."""
    X = _parse_matrix(req.X, "X")
    y = _parse_matrix(req.y, "y")
    if X.shape[0] != y.shape[0]:
        raise HTTPException(400, "Число строк X и y не совпадает")

    net = _build_or_carry(req.graph, req.reset)

    n_in = len(net.input_ids)
    n_out = len(net.output_ids)
    if X.shape[1] != n_in:
        raise HTTPException(400, f"В графе {n_in} входных нейронов, в данных столбцов: {X.shape[1]}")
    if y.shape[1] != n_out:
        raise HTTPException(400, f"В графе {n_out} выходных нейронов, в целях столбцов: {y.shape[1]}")

    CURRENT["net"] = net
    CURRENT["graph"] = req.graph
    net.fit(X, y, epochs=max(1, min(req.epochs, 5000)),
            lr=req.lr, batch_size=req.batch_size)
    return {
        "epochs_done": len(net.loss_history),
        "final_loss": net.loss_history[-1] if net.loss_history else None,
        "loss_history": net.loss_history[-500:],
        "summary": net.summary(),
    }


@app.post("/api/gnet/reset")
async def api_gnet_reset():
    CURRENT["net"] = None
    CURRENT["graph"] = None
    return {"ok": True}


@app.post("/api/gnet/predict")
async def api_gnet_predict(req: PredictRequest):
    """Прогон через граф. Активации всех нейронов и все веса — для подсветки."""
    net = CURRENT.get("net")
    if net is None:
        raise HTTPException(409, "Сеть ещё не обучена — сначала обучите её")
    X = _parse_matrix(req.X, "X")
    if X.shape[1] != len(net.input_ids):
        raise HTTPException(400, f"Сеть ждёт {len(net.input_ids)} вход(ов)")
    A, _ = net.forward(X)
    out = np.stack([A[o] for o in net.output_ids], axis=1)
    return {
        "output": np.round(out, 4).tolist(),
        "inputs": np.round(X, 4).tolist(),
        "activations": {str(i): np.round(A[i], 4).tolist() for i in A},
        "weights": net.weights_payload(),
        "biases": net.biases_payload(),
        "input_ids": net.input_ids,
        "output_ids": net.output_ids,
    }


@app.post("/api/gnet/save")
async def api_gnet_save(req: SaveRequest):
    """Сохраняет граф (+веса, если он обучен) в data/networks/."""
    safe = "".join(c for c in req.name if c.isalnum() or c in "-_ ").strip()
    if not safe:
        raise HTTPException(400, "Пустое имя")
    blob = {"graph": req.graph}
    net = CURRENT.get("net")
    if net is not None and CURRENT.get("graph") == req.graph:
        blob["weights"] = net.weights_payload()
        blob["biases"] = net.biases_payload()
    path = NETS_DIR / f"{safe}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(blob, f, ensure_ascii=False, indent=2)
    return {"saved": path.name}


@app.get("/api/gnet/list")
async def api_gnet_list():
    return sorted(p.stem for p in NETS_DIR.glob("*.json"))


@app.get("/api/gnet/load/{name}")
async def api_gnet_load(name: str):
    path = NETS_DIR / f"{name}.json"
    if not path.is_file():
        raise HTTPException(404, f"«{name}» не найден")
    with open(path, "r", encoding="utf-8") as f:
        blob = json.load(f)
    graph = blob["graph"]
    if "weights" in blob:
        # Веса и смещения вклеиваем в граф, чтобы восстановить обучение
        wmap = {(w["from"], w["to"]): w["w"] for w in blob["weights"]}
        for e in graph.get("edges", []):
            if (e["from"], e["to"]) in wmap:
                e["w"] = wmap[(e["from"], e["to"])]
        bmap = blob.get("biases", {})
        for n in graph.get("neurons", []):
            if str(n["id"]) in bmap:
                n["b"] = bmap[str(n["id"])]
        try:
            net = GraphNetwork(graph)
            CURRENT["net"] = net
            CURRENT["graph"] = graph
        except ValueError:
            pass  # конфиг битый — отдаём как есть, клиент покажет структуру
    return blob


# ════════════════════════════════════════
# API библиотеки модулей (конструктор 2)
# ════════════════════════════════════════
@app.post("/api/module/save")
async def api_module_save(req: SaveRequest):
    """Сохраняет выделенный фрагмент графа (модуль) в data/modules/."""
    safe = "".join(c for c in req.name if c.isalnum() or c in "-_ ").strip()
    if not safe:
        raise HTTPException(400, "Пустое имя")
    path = MODULES_DIR / f"{safe}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(req.graph, f, ensure_ascii=False, indent=2)
    return {"saved": path.name}


@app.get("/api/module/list")
async def api_module_list():
    return sorted(p.stem for p in MODULES_DIR.glob("*.json"))


@app.get("/api/module/load/{name}")
async def api_module_load(name: str):
    path = MODULES_DIR / f"{name}.json"
    if not path.is_file():
        raise HTTPException(404, "Модуль не найден")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ════════════════════════════════════════
# API библиотеки нейронов (конструктор 1)
# ════════════════════════════════════════
class NeuronSaveRequest(BaseModel):
    name: str
    neuron: dict          # {kind, activation?, b?, comment?}


class NameRequest(BaseModel):
    name: str


def _validate_neuron(n: dict) -> dict:
    kind = n.get("kind", "hidden")
    if kind not in ("input", "hidden", "output"):
        raise HTTPException(400, f"Неизвестный тип нейрона: «{kind}»")
    act = n.get("activation")
    if act is not None and act not in ("linear", "relu", "tanh", "sigmoid"):
        raise HTTPException(400, f"Неизвестная активация: «{act}»")
    try:
        b = float(n.get("b", 0.0))
    except (TypeError, ValueError):
        raise HTTPException(400, "Смещение (b) должно быть числом")
    return {"kind": kind, "activation": act, "b": b, "comment": str(n.get("comment", ""))}


@app.get("/api/neuron/list")
async def api_neuron_list():
    """Библиотека готовых нейронов: полные карточки для выбора в конструкторе."""
    out = []
    for p in sorted(NEURONS_DIR.glob("*.json")):
        try:
            with open(p, "r", encoding="utf-8") as f:
                card = json.load(f)
            out.append({"name": p.stem, **card})
        except (json.JSONDecodeError, OSError):
            continue  # битая карточка — не ломаем список
    return out


@app.post("/api/neuron/save")
async def api_neuron_save(req: NeuronSaveRequest):
    safe = "".join(c for c in req.name if c.isalnum() or c in "-_ ").strip()
    if not safe:
        raise HTTPException(400, "Пустое имя")
    card = _validate_neuron(req.neuron)
    path = NEURONS_DIR / f"{safe}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(card, f, ensure_ascii=False, indent=2)
    return {"saved": path.name}


@app.post("/api/neuron/delete")
async def api_neuron_delete(req: NameRequest):
    path = NEURONS_DIR / f"{req.name}.json"
    if not path.is_file():
        raise HTTPException(404, f"«{req.name}» не найден")
    path.unlink()
    return {"deleted": req.name}


# ════════════════════════════════════════
# API xlsx-таблиц данных (страница «Таблицы»)
# Структура: колонки x* — входы, y* — выходы,
# pX*/pY* — пояснения к соответствующим x*/y*,
# колонка p/описание — пояснение самой таблицы.
# ════════════════════════════════════════
class TableSaveRequest(BaseModel):
    file: str
    sheet: str
    range: str                    # исходный диапазон в A1-нотации (для перезаписи)
    description: str = ""
    columns: list[dict]           # [{header, values}]
    mode: str = "overwrite"       # overwrite | copy
    copy_name: str = ""


DESC_HEADERS = ("p", "опис", "поясн", "примеч", "desc", "description", "note", "comment")


def _table_role(header) -> str:
    """Роль колонки по заголовку: x | y | px | py | desc | other."""
    h = str(header or "").strip().lower()
    if not h:
        return "ignore"
    if h.startswith("px") or h.startswith("pх"):
        return "px"
    if h.startswith("py") or h.startswith("pу"):
        return "py"
    if any(h == d or h.startswith(d) for d in DESC_HEADERS):
        return "desc"
    if h.startswith("x") or h.startswith("х"):
        return "x"
    if h.startswith("y") or h.startswith("у"):
        return "y"
    return "other"


def _json_val(v):
    if v is None:
        return None
    if isinstance(v, float) and v != v:   # NaN
        return None
    return v


def _cell_val(v):
    """Значение из клиента → значение ячейки xlsx."""
    if v is None:
        return None
    if isinstance(v, str):
        s = v.strip()
        if not s:
            return None
        try:
            return float(s)
        except ValueError:
            return s
    return v


def _a1(c1, r1, c2, r2) -> str:
    return f"{get_column_letter(c1)}{r1}:{get_column_letter(c2)}{r2}"


@app.get("/api/table/list")
async def api_table_list():
    return [{"name": p.stem, "file": p.name} for p in sorted(TABLES_DIR.glob("*.xlsx"))]


@app.post("/api/table/upload")
async def api_table_upload(files: List[UploadFile] = File(...)):
    """Принимает сразу несколько xlsx-файлов."""
    saved, skipped = [], []
    for f in files:
        if not (f.filename or "").lower().endswith(".xlsx"):
            skipped.append(f.filename)
            continue
        stem = "".join(c for c in Path(f.filename).stem if c.isalnum() or c in "-_ ").strip() or "table"
        target, i = TABLES_DIR / f"{stem}.xlsx", 1
        while target.exists():   # не затираем: книга (2), книга (3)…
            target = TABLES_DIR / f"{stem} ({i}).xlsx"
            i += 1
        with open(target, "wb") as out:
            out.write(await f.read())
        saved.append(target.name)
    return {"saved": saved, "skipped": skipped}


@app.get("/api/table/open")
async def api_table_open(file: str, sheet: str = "", rng: str = ""):
    """Разбор листа: колонки по ролям + данные из диапазона (или весь лист)."""
    path = TABLES_DIR / file
    if not path.is_file():
        raise HTTPException(404, f"«{file}» не найден")
    try:
        wb = load_workbook(path, data_only=True)
    except Exception:
        raise HTTPException(400, "Не удалось открыть xlsx-файл")
    sheets = wb.sheetnames
    ws = wb[sheet] if sheet and sheet in sheets else wb.active
    if ws is None:
        raise HTTPException(400, "В книге нет листов")
    try:
        c1, r1, c2, r2 = range_boundaries(rng.upper()) if rng else range_boundaries(ws.dimensions)
    except Exception:
        raise HTTPException(400, f"Неверный диапазон: «{rng}» (пример: B2:E40)")
    c2, r2 = min(c2, c1 + 63), min(r2, r1 + 501)   # разумный потолок
    headers = [ws.cell(r1, c).value for c in range(c1, c2 + 1)]
    columns = []
    for i, h in enumerate(headers):
        vals = [_json_val(ws.cell(r, c1 + i).value) for r in range(r1 + 1, r2 + 1)]
        columns.append({
            "header": str(h) if h is not None else "",
            "role": _table_role(h),
            "values": vals,
            "numeric": all(v is None or isinstance(v, (int, float)) for v in vals),
        })
    description = ""
    for col in columns:
        if col["role"] == "desc":
            description = next((str(v) for v in col["values"] if v not in (None, "")), "")
            break
    wb.close()
    return {
        "file": file, "sheet": ws.title, "sheets": sheets,
        "range": _a1(c1, r1, c2, r2),
        "description": description,
        "columns": columns,
        "n_rows": r2 - r1,
    }


@app.post("/api/table/save")
async def api_table_save(req: TableSaveRequest):
    """Сохранение правок: «overwrite» — в тот же файл и диапазон, «copy» — новым файлом."""
    cols = [c for c in req.columns if str(c.get("header", "")).strip() or any(v not in (None, "") for v in c.get("values", []))]
    if not cols:
        raise HTTPException(400, "Таблица пуста")
    if req.description.strip():
        # пояснение таблицы: колонка p, если её ещё нет
        if not any(_table_role(c.get("header")) == "desc" for c in cols):
            vals = [req.description.strip()] + [None] * max(0, len(cols[0].get("values", [])) - 1)
            cols.append({"header": "p", "values": vals})
        else:
            for c in cols:
                if _table_role(c.get("header")) == "desc":
                    v = list(c.get("values", []))
                    v[0] = req.description.strip()
                    c["values"] = v
                    break

    if req.mode == "copy":
        stem = "".join(ch for ch in (req.copy_name or Path(req.file).stem + " копия")
                       if ch.isalnum() or ch in "-_ ").strip() or "table"
        target, i = TABLES_DIR / f"{stem}.xlsx", 1
        while target.exists():
            target = TABLES_DIR / f"{stem} ({i}).xlsx"
            i += 1
        wb = Workbook()
        ws = wb.active
        ws.title = (req.sheet or "Лист1")[:31]
        r1, c1 = 1, 1
    else:
        path = TABLES_DIR / req.file
        if not path.is_file():
            raise HTTPException(404, f"«{req.file}» не найден")
        wb = load_workbook(path)   # без data_only: чужие формулы не трогаем
        if req.sheet not in wb.sheetnames:
            raise HTTPException(400, f"Лист «{req.sheet}» не найден")
        ws = wb[req.sheet]
        try:
            c1, r1, _, _ = range_boundaries(req.range.upper())
        except Exception:
            c1, r1 = 1, 1
        # очистить прежний диапазон (правки могли укоротить таблицу)
        try:
            oc1, or1, oc2, or2 = range_boundaries(req.range.upper())
            for r in range(or1, or2 + 1):
                for c in range(oc1, oc2 + 1):
                    ws.cell(r, c).value = None
        except Exception:
            pass
        target = path

    n_rows = max(len(c.get("values", [])) for c in cols)
    for j, c in enumerate(cols):
        ws.cell(r1, c1 + j).value = str(c.get("header", ""))
        vals = list(c.get("values", [])) + [None] * (n_rows - len(c.get("values", [])))
        for i, v in enumerate(vals):
            ws.cell(r1 + 1 + i, c1 + j).value = _cell_val(v)
    wb.save(target)
    return {"saved": target.name, "range": _a1(c1, r1, c1 + len(cols) - 1, r1 + n_rows)}


@app.post("/api/table/delete")
async def api_table_delete(req: NameRequest):
    path = TABLES_DIR / f"{req.name}.xlsx"
    if not path.is_file():
        raise HTTPException(404, f"«{req.name}» не найден")
    path.unlink()
    return {"deleted": req.name}
