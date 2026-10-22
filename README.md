# NeuralForge 🧠

**NeuralForge Neuro-Constructor** — визуальный конструктор графовых нейронных сетей:
проектируйте, обучайте и анализируйте сети с произвольной архитектурой графа прямо в браузере.

**NeuralForge Neuro-Constructor** — a visual constructor for graph-based neural networks:
design, train and analyse networks with an arbitrary graph architecture right in the browser.

---

## 🇷🇺 О проекте

Инструмент для проектирования, обучения и анализа нейросетей со свободным графом:
точки-нейроны и связи-веса вы строите сами, вручную. Поддержка модульности
(сохранение блоков сети в библиотеку), работа с XLSX-датасетами
и интерактивный анализ весов в реальном времени.

### Возможности

- **1 · Нейроны** — настройка отдельного нейрона (тип, активация, смещение b) и библиотека карточек
- **2 · Сборка** — конструктор сети: точки, связи, папки-контейнеры, заморозка обучения (❄)
- **3 · Всё вместе** — то же плюс тестирование: по таблице и вручную в реальном времени
- **📦 Модули** — сохранение выделенного блока сети и вставка его как интегрируемого блока
- **📋 Таблицы** — загрузка и редактирование XLSX, объединение нескольких файлов в датасет
- 🌐 **Двуязычный интерфейс** (русский / English) и светлая/тёмная тема
- Адаптивная вёрстка: компьютер, планшет, телефон (тач + щипок-зум)

### Быстрый старт

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
uvicorn main:app --reload
```

Откройте http://127.0.0.1:8000

### Стек

FastAPI · Jinja2 · NumPy · OpenPyXL · Python 3.11 · ванильный JS на фронтенде

---

## 🇬🇧 About

A tool for designing, training and analysing neural networks with a free graph:
you build the neuron-points and weight-connections yourself, by hand. Modular
design (saving network blocks to a library), XLSX dataset support and
interactive real-time weight analysis.

### Features

- **1 · Neurons** — configure a single neuron (type, activation, bias b) and a card library
- **2 · Assembly** — network constructor: points, connections, folder containers, training freeze (❄)
- **3 · All together** — the same plus testing: by table and manually in real time
- **📦 Modules** — save a selected network block and insert it as an integrable block
- **📋 Tables** — upload and edit XLSX, combine several files into a dataset
- 🌐 **Bilingual UI** (Russian / English) and light/dark theme
- Responsive layout: desktop, tablet, phone (touch + pinch zoom)

### Quick start

```bash
python -m venv .venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload
```

Open http://127.0.0.1:8000

### Stack

FastAPI · Jinja2 · NumPy · OpenPyXL · Python 3.11 · vanilla JS frontend

---

## 💜 Поддержка / Support

Понравился проект? Поддержите автора на Boosty — это поможет обзавестись новым
оборудованием и делать новые проекты. Спасибо за внимание!

Enjoying the project? Support the author on Boosty — it helps get new equipment
and build new projects. Thank you for your attention!

**[💜 Boosty — boosty.to/nacro](https://boosty.to/nacro)**

## 📄 Лицензия / License

MIT © [Grimzs](https://github.com/Grimzs)
