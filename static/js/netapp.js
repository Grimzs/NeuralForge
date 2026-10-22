/* ================================================================
   НЕЙРО-КОНСТРУКТОР: общий движок (страницы «2 · Сборка» и «3 · Всё вместе»)
   • Точки-нейроны ставятся по одной (⊙) или столбиком-слоем (⧉)
   • Связи проводятся ВРУЧНУЮ (режим 🔗): тяни от точки к точке
   • Каждая связь — отдельный вес; удаление связи теряет вес
   • Клик по связи — вставить скрытый нейрон прямо в разрыв
    • Библиотека готовых нейронов (/api/neuron/*, создаётся в «1 · Нейроны»)
    • Библиотека модулей-блоков (/api/module/*): 📋/✂️ папки → 💾, 📥 — вставка
        (в режиме «интегрируемый блок» входы модуля становятся hidden-портами)
   • ❄ trainable=false — заморозка: обучение веса нейрона не трогает
   • Обучение: пошаговые запросы, веса живут на сервере
   ================================================================ */

// Инъекция стилей для информационного окна
const style = document.createElement('style');
style.textContent = `
    .node-info-tooltip {
        position: absolute;
        display: none;
        background: #2a2a32;
        color: #fff;
        border: 1px solid #6f7ff0;
        padding: 10px;
        border-radius: 8px;
        font-size: 12px;
        pointer-events: none;
        z-index: 10000;
        box-shadow: 0 4px 15px rgba(0,0,0,0.5);
        line-height: 1.5;
        font-family: system-ui, sans-serif;
    }
    .node-info-tooltip b { color: #6f7ff0; }
    .node-info-tooltip .info-row { margin-bottom: 4px; }
`;
document.head.appendChild(style);

window.NetApp = (() => {
    const $ = id => document.getElementById(id);
    const SVG_NS = 'http://www.w3.org/2000/svg';

    // ── Состояние графа ──
    let graph = { neurons: [], edges: [], loss: 'bce' };
    let folders = [];           // окна-папки: {id, name, x, y, w, h}
    let nextId = 1;
    let nextFolderId = 1;
    let clipboard = null;       // буфер для копировать/вырезать: {folder, neurons, edges}
    let folderDrag = null;      // перетаскивание папки (с точками внутри)
    let folderResize = null;    // изменение размеров папки за уголок
    let selected = null;        // id выбранной точки
    let selectedRow = 0;        // строка данных для подсветки
    let lastResult = null;      // ответ predict
    let lossHist = [];
    let training = false;
    let datasets = null;
    let library = [];           // библиотека готовых нейронов
    let liveTimer = null;       // debounce ручного теста в реальном времени
    let tooltip = null;         // элемент информационного окна

    // ── Режимы UI ──
    let mode = 'move';          // 'move' | 'add' | 'link'
    let addKind = 'hidden';     // что ставим
    let addHow = 'single';      // 'single' | 'multi'
    const R = 15;               // радиус точки

    const T = (k, p) => I18N.t(k, p);
    const KIND = { input: 'k-input', hidden: 'k-hidden', output: 'k-output' };

    // ── Геометрия сцены ──
    let view = { scale: 1, tx: 0, ty: 0 };

    const byId = id => graph.neurons.find(n => n.id === id);
    const edgeKey = e => e.from + '→' + e.to;
    function elemGn(el) { return el && el.closest ? el.closest('.gn') : null; }
    function elemFol(el) { return el && el.closest ? el.closest('.folder') : null; }

    function labelOf(n) {
        const same = graph.neurons.filter(x => x.kind === n.kind);
        const i = same.indexOf(n);
        const p = { input: 'x', hidden: 'h', output: 'y' }[n.kind];
        return p + (i + 1);
    }

    // ══════════════════════════
    // Инфо-панель (Тултип)
    // ══════════════════════════
    function showNodeInfo(ev, n) {
        if (!tooltip) return;

        const sources = graph.edges
            .filter(e => e.to === n.id)
            .map(e => `${labelOf(byId(e.from))} (w: ${e.w?.toFixed(2) || '?'})`);

        const targets = graph.edges
            .filter(e => e.from === n.id)
            .map(e => `${labelOf(byId(e.to))} (w: ${e.w?.toFixed(2) || '?'})`);

        let currentVal = '—';
        if (lastResult && lastResult.activations[String(n.id)]) {
            const val = lastResult.activations[String(n.id)][selectedRow];
            currentVal = val !== undefined ? val.toFixed(4) : '—';
        }

        tooltip.innerHTML = `
            <div class="info-row"><b>${T('net.tip.node')}</b> ${labelOf(n)} (ID: ${n.id})</div>
            <div class="info-row"><b>${T('net.tip.kind')}</b> ${T('k.' + n.kind)}</div>
            <div class="info-row"><b>${T('net.tip.act')}</b> ${n.activation || 'linear'}</div>
            <div class="info-row"><b>${T('net.tip.val')}</b> ${currentVal}</div>
            <div class="info-row"><b>${T('net.tip.inputs')}</b> ${sources.length ? sources.join(', ') : T('net.tip.noIn')}</div>
            <div class="info-row"><b>${T('net.tip.outputs')}</b> ${targets.length ? targets.join(', ') : T('net.tip.noOut')}</div>
            ${n.comment ? `<div class="info-row"><b>${T('net.tip.note')}</b> ${n.comment}</div>` : ''}
        `;

        tooltip.style.display = 'block';
        tooltip.style.left = (ev.pageX + 15) + 'px';
        tooltip.style.top = (ev.pageY + 15) + 'px';
    }

    // ══════════════════════════
    // Отрисовка
    // ══════════════════════════
    function render() {
        const world = $('world'), svg = $('edgeSvg');
        world.querySelectorAll('.gn,.folder').forEach(e => e.remove());
        svg.innerHTML = '';
        $('stageEmpty').style.display = graph.neurons.length ? 'none' : 'flex';

        folders.forEach(f => world.appendChild(folderEl(f)));

        graph.neurons.forEach(n => {
            const el = document.createElement('div');
            el.className = 'gn ' + KIND[n.kind] + (selected === n.id ? ' selected' : '')
                + (n.trainable === false ? ' frozen' : '');
            el.dataset.id = n.id;
            el.style.left = n.x + 'px';
            el.style.top = n.y + 'px';
            el.textContent = labelOf(n);

            // Интеграция тултипа вместо el.title
            el.addEventListener('mouseenter', (e) => showNodeInfo(e, n));
            el.addEventListener('mousemove', (e) => {
                if (tooltip) {
                    tooltip.style.left = (e.pageX + 15) + 'px';
                    tooltip.style.top = (e.pageY + 15) + 'px';
                }
            });
            el.addEventListener('mouseleave', () => {
                if (tooltip) tooltip.style.display = 'none';
            });

            if (n.comment) {
                const c = document.createElement('span');
                c.className = 'gn-comment';
                c.textContent = n.comment;
                el.appendChild(c);
            }
            if (n.trainable === false) {
                const fr = document.createElement('span');
                fr.className = 'frost';
                fr.textContent = '❄';
                el.appendChild(fr);
            }
            world.appendChild(el);
        });

        const wmap = new Map();
        if (lastResult) lastResult.weights.forEach(w =>
            wmap.set(w.from + '→' + w.to, w.w));
        const acts = actOfRow();
        let maxS = 1e-9;
        if (acts) graph.edges.forEach(e => {
            const w = wmap.get(edgeKey(e));
            if (w !== undefined && acts[e.from] !== undefined)
                maxS = Math.max(maxS, Math.abs(w) * Math.abs(acts[e.from]));
        });

        graph.edges.forEach(e => {
            const p = document.createElementNS(SVG_NS, 'path');
            p.setAttribute('class', 'gedge');
            p.dataset.from = e.from; p.dataset.to = e.to;
            const w = wmap.get(edgeKey(e));
            if (w !== undefined && acts && acts[e.from] !== undefined) {
                const s = Math.abs(w) * Math.abs(acts[e.from]) / maxS;
                p.style.strokeOpacity = 0.15 + 0.85 * s;
                p.style.strokeWidth = 0.7 + 3 * s;
                p.classList.add(s > 0.55 ? 'hot' : s > 0.2 ? 'warm' : 'cold');
            }
            p.title = T('net.edge.title', { a: labelOf(byId(e.from)), b: labelOf(byId(e.to)) }) +
                (w !== undefined ? T('net.edge.weight', { w }) : '') +
                T('net.edge.hint');
            svg.appendChild(p);
        });
        updateEdges();
        updateStats();
        rebuildLiveInputs();
    }

    function updateEdges() {
        const svg = $('edgeSvg');
        svg.querySelectorAll('.gedge').forEach(p => {
            const a = byId(+p.dataset.from), b = byId(+p.dataset.to);
            if (!a || !b) return;
            const mx = (a.x + b.x) / 2, my = (a.y + b.y) / 2;
            const dx = b.x - a.x, dy = b.y - a.y;
            const len = Math.hypot(dx, dy) || 1;
            const off = Math.min(24, len * 0.12);
            const cx = mx - dy / len * off, cy = my + dx / len * off;
            p.setAttribute('d', `M ${a.x} ${a.y} Q ${cx} ${cy} ${b.x} ${b.y}`);
        });
        let maxX = 400, maxY = 300;
        graph.neurons.forEach(n => { maxX = Math.max(maxX, n.x + 80); maxY = Math.max(maxY, n.y + 80); });
        $('edgeSvg').setAttribute('width', maxX);
        $('edgeSvg').setAttribute('height', maxY);
        $('world').style.width = maxX + 'px';
        $('world').style.height = maxY + 'px';
    }

    function actOfRow() {
        if (!lastResult) return null;
        const out = {};
        Object.entries(lastResult.activations).forEach(([id, arr]) => {
            out[+id] = arr[selectedRow] !== undefined ? arr[selectedRow] : arr[0];
        });
        lastResult.input_ids.forEach((id, k) => {
            const row = lastResult.inputs[selectedRow];
            if (row) out[id] = row[k];
        });
        return out;
    }

    function updateStats() {
        const nH = graph.neurons.filter(n => n.kind === 'hidden').length;
        const nF = graph.neurons.filter(n => n.trainable === false).length;
        $('netStats').textContent =
            T('net.stats.main', { n: graph.neurons.length, m: graph.edges.length }) +
            (nH ? T('net.stats.hidden', { n: nH }) : '') +
            (nF ? T('net.stats.frozen', { n: nF }) : '') +
            (folders.length ? T('net.stats.folders', { n: folders.length }) : '');
    }

    function pointsInFolder(f) {
        return graph.neurons.filter(n =>
            n.x >= f.x && n.x <= f.x + f.w && n.y >= f.y && n.y <= f.y + f.h);
    }

    function folderEl(f) {
        const el = document.createElement('div');
        el.className = 'folder';
        el.dataset.fid = f.id;
        el.style.cssText = `left:${f.x}px;top:${f.y}px;width:${f.w}px;height:${f.h}px`;
        const inside = pointsInFolder(f);
        const anyTrainable = inside.some(p => p.trainable !== false);
        el.innerHTML = `
            <div class="folder-head">
                <span class="fname" title="${T('net_folder.rename.title')}">${f.name}</span>
                <span class="hint">${T('net_folder.pts', { n: inside.length })}</span>
                <span class="folder-btns">
                    <button data-a="freeze" title="${anyTrainable ? T('net_folder.freeze.title') : T('net_folder.thaw.title')}">${anyTrainable ? '❄' : '🔥'}</button>
                    <button data-a="copy"  title="${T('net_folder.copy.title')}">📋</button>
                    <button data-a="cut"   title="${T('net_folder.cut.title')}">✂️</button>
                    <button data-a="del"   title="${T('net_folder.del.title')}">🗑</button>
                </span>
            </div>
            <div class="folder-resize" title="${T('net_folder.resize.title')}"></div>`;
        el.querySelector('[data-a="freeze"]').onclick = ev => {
            ev.stopPropagation();
            const pts = pointsInFolder(f);
            if (!pts.length) return flash(T('net_folder.noPts'));
            const thaw = pts.some(p => p.trainable !== false);
            pts.forEach(p => { p.trainable = !thaw; });
            render(); renderInspector();
            flash(thaw ? T('net_folder.froze', { n: pts.length }) : T('net_folder.thawed', { n: pts.length }));
        };
        el.querySelector('[data-a="copy"]').onclick = ev => { ev.stopPropagation(); copyFolder(f, false); };
        el.querySelector('[data-a="cut"]').onclick = ev => { ev.stopPropagation(); copyFolder(f, true); };
        el.querySelector('[data-a="del"]').onclick = ev => {
            ev.stopPropagation();
            const pts = pointsInFolder(f);
            if (!confirm(T('net_folder.delConfirm', { name: f.name, with: pts.length ? T('net_folder.delWith', { n: pts.length }) : '' }))) return;
            removeFolderWithPoints(f);
        };
        el.querySelector('.fname').ondblclick = ev => {
            ev.stopPropagation();
            const name = prompt(T('net_folder.namePrompt'), f.name);
            if (name) { f.name = name; render(); }
        };
        el.querySelector('.folder-resize').addEventListener('mousedown', ev => {
            ev.stopPropagation(); ev.preventDefault();
            folderResize = { f };
        });
        return el;
    }

    function createFolder() {
        const c = viewportCenterWorld();
        const f = { id: nextFolderId++, name: T('net_folder.nameDefault', { n: nextFolderId }), x: c.x - 130, y: c.y - 100, w: 260, h: 200 };
        folders.push(f);
        render();
        flash(T('net_folder.created', { name: f.name }));
    }

    function copyFolder(f, cut) {
        const pts = pointsInFolder(f);
        const ids = new Set(pts.map(p => p.id));
        const internal = graph.edges.filter(e => ids.has(e.from) && ids.has(e.to));
        clipboard = {
            folder: { ...f },
            neurons: pts.map(p => ({ ...p })),
            edges: internal.map(e => ({ ...e })),
        };
        if (cut) {
            removeFolderWithPoints(f, false);
            flash(T('net_folder.cutDone', { name: f.name }));
        } else {
            flash(T('net_folder.copied', { name: f.name }));
        }
    }

    function removeFolderWithPoints(f, confirmAsk = true) {
        const pts = pointsInFolder(f);
        if (confirmAsk && !confirm(T('net_folder.delConfirm2', { name: f.name }))) return;
        const ids = new Set(pts.map(p => p.id));
        graph.neurons = graph.neurons.filter(n => !ids.has(n.id));
        graph.edges = graph.edges.filter(e => !ids.has(e.from) && !ids.has(e.to));
        folders = folders.filter(x => x !== f);
        render(); renderInspector();
    }

    function pasteClipboard() {
        if (!clipboard) return flash(T('net_clipboard.empty'));
        const idMap = new Map();
        const pts = clipboard.neurons.map(p => {
            const nid = nextId++;
            idMap.set(p.id, nid);
            return { ...p, id: nid };
        });
        const edges = clipboard.edges
            .filter(e => idMap.has(e.from) && idMap.has(e.to))
            .map(e => ({ from: idMap.get(e.from), to: idMap.get(e.to), w: e.w }));
        const c = viewportCenterWorld();
        const f0 = clipboard.folder;
        const dx = c.x - (f0.x + f0.w / 2), dy = c.y - (f0.y + f0.h / 2);
        pts.forEach(p => { p.x += dx; p.y += dy; });
        const f = { ...f0, id: nextFolderId++, name: f0.name + T('net_clipboard.copySuffix'), x: f0.x + dx, y: f0.y + dy };
        folders.push(f);
        graph.neurons.push(...pts);
        graph.edges.push(...edges);
        render();
    }

    function viewportCenterWorld() {
        const wrap = $('canvasWrap');
        return {
            x: (wrap.clientWidth / 2 - view.tx) / view.scale,
            y: (wrap.clientHeight / 2 - view.ty) / view.scale,
        };
    }

    function addNeuronAt(x, y, kind, activation) {
        const n = { id: nextId++, kind, activation: activation || null, comment: '', x, y };
        graph.neurons.push(n);
        return n;
    }

    function placeAt(x, y) {
        const act = addKind === 'input' ? null : $('addAct').value;
        if (addHow === 'single') {
            addNeuronAt(x, y, addKind, act);
        } else {
            const count = Math.max(1, Math.min(64, +$('addCount').value || 1));
            for (let i = 0; i < count; i++) {
                const yy = y + (i - (count - 1) / 2) * 56;
                addNeuronAt(x, yy, addKind, act);
            }
        }
        selected = null;
        render(); renderInspector();
    }

    function removeNeuron(id) {
        const n = byId(id); if (!n) return;
        graph.neurons = graph.neurons.filter(n2 => n2.id !== id);
        graph.edges = graph.edges.filter(e => e.from !== id && e.to !== id);
        if (selected === id) selected = null;
        render(); renderInspector();
    }

    function wouldCycle(from, to) {
        const seen = new Set();
        const stack = [to];
        while (stack.length) {
            const cur = stack.pop();
            if (cur === from) return true;
            if (seen.has(cur)) continue;
            seen.add(cur);
            graph.edges.filter(e => e.from === cur).forEach(e => stack.push(e.to));
        }
        return false;
    }

    function connect(from, to) {
        const a = byId(from), b = byId(to);
        if (!a || !b || from === to) return;
        if (b.kind === 'input') return flash(T('net_edge.noIntoInput'));
        if (a.kind === 'output') return flash(T('net_edge.noFromOutput'));
        if (graph.edges.some(e => e.from === from && e.to === to)) return;
        if (wouldCycle(from, to)) return flash(T('net_edge.noCycle'));
        graph.edges.push({ from, to });
        render();
    }

    function removeEdge(from, to) {
        graph.edges = graph.edges.filter(e2 => !(e2.from === from && e2.to === to));
        render();
    }

    function insertOnEdge(from, to) {
        const e = graph.edges.find(e2 => e2.from === from && e2.to === to);
        if (!e) return;
        const a = byId(from), b = byId(to);
        const n = addNeuronAt((a.x + b.x) / 2, (a.y + b.y) / 2, 'hidden', $('addAct').value);
        const w = e.w !== undefined ? e.w : undefined;
        graph.edges = graph.edges.filter(e2 => e2 !== e);
        graph.edges.push({ from, to: n.id, w });
        graph.edges.push({ from: n.id, to, w: 1 });
        selected = n.id;
        render(); renderInspector();
    }

    function renderInspector() {
        const box = $('inspector');
        if (!box) return;
        const n = selected !== null ? byId(selected) : null;
        if (!n) { box.innerHTML = T('net.inspector.empty'); return; }
        const frozen = n.trainable === false;
        box.innerHTML = `
            <b>${T('k.' + n.kind)} ${labelOf(n)}</b> <span class="hint">#${n.id}</span>
            ${n.kind !== 'input' ? `<div class="row-btns" style="display:grid;grid-template-columns:1fr 1fr">
                <div><span class="hint">${T('neurons.act.hint')}</span>
                    <select id="insAct">${['tanh','relu','sigmoid','linear']
                        .map(a2 => `<option ${n.activation === a2 ? 'selected' : ''}>${a2}</option>`).join('')}</select></div>
                <div><span class="hint">&nbsp;</span><button class="btn btn-sm btn-danger" id="insDel" style="width:100%">${T('net.ins.del')}</button></div>
            </div>` : `<div class="row-btns"><button class="btn btn-sm btn-danger" id="insDel">${T('net.ins.delPoint')}</button></div>`}
            ${n.kind !== 'input' ? `<div class="row-btns"><button class="btn btn-sm ${frozen ? '' : 'btn-primary'}" id="insTrain" style="width:100%">${frozen ? T('net.ins.trainOff') : T('net.ins.trainOn')}</button></div>` : ''}
            <div style="margin-top:6px"><span class="hint">${T('neurons.cmt.hint')}</span>
                <input id="insCmt" value="${(n.comment || '').replace(/"/g, '&quot;')}" placeholder="${T('net.ins.cmtPh')}"></div>`;
        const sel = $('insAct');
        if (sel) sel.onchange = e2 => { n.activation = e2.target.value; render(); };
        $('insDel').onclick = () => removeNeuron(n.id);
        const tr = $('insTrain');
        if (tr) tr.onclick = () => {
            n.trainable = n.trainable === false ? true : false;
            render(); renderInspector();
        };
        $('insCmt').onchange = e2 => { n.comment = e2.target.value; render(); };
    }

    function applyView() {
        $('world').style.transform = `translate(${view.tx}px, ${view.ty}px) scale(${view.scale})`;
        $('zoomVal').textContent = Math.round(view.scale * 100) + '%';
    }
    function toWorld(clientX, clientY) {
        const r = $('canvasWrap').getBoundingClientRect();
        return { x: (clientX - r.left - view.tx) / view.scale, y: (clientY - r.top - view.ty) / view.scale };
    }
    function zoomAt(factor, cx, cy) {
        const ns = Math.min(3, Math.max(0.2, view.scale * factor));
        view.tx = cx - (cx - view.tx) * (ns / view.scale);
        view.ty = cy - (cy - view.ty) * (ns / view.scale);
        view.scale = ns; applyView();
    }
    function fit() {
        let x1 = 1e9, y1 = 1e9, x2 = -1e9, y2 = -1e9;
        if (!graph.neurons.length && !folders.length) { view = { scale: 1, tx: 20, ty: 20 }; return applyView(); }
        graph.neurons.forEach(n => { x1 = Math.min(x1, n.x - 60); y1 = Math.min(y1, n.y - 60); x2 = Math.max(x2, n.x + 60); y2 = Math.max(y2, n.y + 60); });
        folders.forEach(f => { x1 = Math.min(x1, f.x - 20); y1 = Math.min(y1, f.y - 20); x2 = Math.max(x2, f.x + f.w + 20); y2 = Math.max(y2, f.y + f.h + 20); });
        const wrap = $('canvasWrap');
        const w = wrap.clientWidth, h = wrap.clientHeight;
        view.scale = Math.min(1.4, Math.min(w / (x2 - x1), h / (y2 - y1)) * 0.95);
        view.tx = (w - (x2 - x1) * view.scale) / 2 - x1 * view.scale;
        view.ty = (h - (y2 - y1) * view.scale) / 2 - y1 * view.scale;
        applyView();
    }

    function setMode(m) {
        mode = m;
        const st = $('stage');
        if (st) st.className = 'stage mode-' + m;
        const seg = $('segMode');
        if (seg) seg.querySelectorAll('button').forEach(b => b.classList.toggle('active', b.dataset.v === m));
    }

    function bindStage() {
        const wrap = $('canvasWrap');
        $('zoomIn').onclick = () => zoomAt(1.25, wrap.clientWidth / 2, wrap.clientHeight / 2);
        $('zoomOut').onclick = () => zoomAt(0.8, wrap.clientWidth / 2, wrap.clientHeight / 2);
        $('zoomFit').onclick = fit;
        $('newFolderBtn').onclick = createFolder;
        $('pasteBtn').onclick = pasteClipboard;
        $('clearBtn').onclick = () => {
            if (!confirm(T('net_clear.confirm'))) return;
            graph.neurons = []; graph.edges = []; folders = [];
            nextId = 1; nextFolderId = 1; selected = null; lastResult = null;
            render(); renderInspector();
        };

        wrap.addEventListener('wheel', ev => {
            ev.preventDefault();
            const r = wrap.getBoundingClientRect();
            zoomAt(ev.deltaY < 0 ? 1.12 : 0.89, ev.clientX - r.left, ev.clientY - r.top);
        }, { passive: false });

        let dragN = null, pan = null, downBg = null, linkDrag = null;

        wrap.addEventListener('mousedown', ev => {
            const gn = ev.target.closest('.gn');
            const ge = ev.target.closest('.gedge');
            if (gn) {
                const n = byId(+gn.dataset.id);
                if (mode === 'link') {
                    linkDrag = { from: n.id, temp: tempPath() };
                    ev.preventDefault();
                } else {
                    dragN = { n, el: gn, dx: n.x - toWorld(ev.clientX, ev.clientY).x, dy: n.y - toWorld(ev.clientX, ev.clientY).y };
                    gn.classList.add('dragging');
                    ev.preventDefault();
                }
                return;
            }
            if (ge) return;
            const fol = ev.target.closest('.folder');
            if (fol && mode === 'move') {
                const f = folders.find(x => x.id === +fol.dataset.fid);
                if (f) {
                    const w = toWorld(ev.clientX, ev.clientY);
                    folderDrag = { f, sx: w.x, sy: w.y, f0x: f.x, f0y: f.y,
                        pts: pointsInFolder(f).map(p => ({ n: p, x0: p.x, y0: p.y })) };
                    ev.preventDefault();
                    return;
                }
            }
            downBg = { x: ev.clientX, y: ev.clientY, tx: view.tx, ty: view.ty };
            pan = downBg;
        });

        window.addEventListener('mousemove', ev => {
            if (dragN) {
                const w = toWorld(ev.clientX, ev.clientY);
                dragN.n.x = w.x + dragN.dx; dragN.n.y = w.y + dragN.dy;
                dragN.el.style.left = dragN.n.x + 'px';
                dragN.el.style.top = dragN.n.y + 'px';
                updateEdges();
            } else if (linkDrag) {
                const w = toWorld(ev.clientX, ev.clientY);
                const a = byId(linkDrag.from);
                linkDrag.temp.setAttribute('d', `M ${a.x} ${a.y} L ${w.x} ${w.y}`);
            } else if (folderDrag) {
                const w = toWorld(ev.clientX, ev.clientY);
                const d = folderDrag;
                d.f.x = d.f0x + w.x - d.sx; d.f.y = d.f0y + w.y - d.sy;
                const el = document.querySelector(`.folder[data-fid="${d.f.id}"]`);
                if (el) { el.style.left = d.f.x + 'px'; el.style.top = d.f.y + 'px'; }
                d.pts.forEach(p2 => {
                    p2.n.x = p2.x0 + w.x - d.sx; p2.n.y = p2.y0 + w.y - d.sy;
                    const pe = document.querySelector(`.gn[data-id="${p2.n.id}"]`);
                    if (pe) { pe.style.left = p2.n.x + 'px'; pe.style.top = p2.n.y + 'px'; }
                });
                updateEdges();
            } else if (folderResize) {
                const w = toWorld(ev.clientX, ev.clientY);
                const f = folderResize.f;
                f.w = Math.max(140, w.x - f.x); f.h = Math.max(100, w.y - f.y);
                const el = document.querySelector(`.folder[data-fid="${f.id}"]`);
                if (el) { el.style.width = f.w + 'px'; el.style.height = f.h + 'px'; }
            } else if (pan) {
                view.tx = pan.tx + ev.clientX - pan.x;
                view.ty = pan.ty + ev.clientY - pan.y;
                applyView();
            }
        });

        window.addEventListener('mouseup', ev => {
            if (dragN) { dragN.el.classList.remove('dragging'); dragN = null; render(); }
            if (folderDrag || folderResize) { folderDrag = null; folderResize = null; render(); }
            if (linkDrag) {
                const gn = elemGn(document.elementFromPoint(ev.clientX, ev.clientY));
                if (gn) connect(linkDrag.from, +gn.dataset.id);
                linkDrag.temp.remove(); linkDrag = null;
            }
            if (pan && downBg) {
                const dist = Math.hypot(ev.clientX - downBg.x, ev.clientY - downBg.y);
                if (dist < 5 && mode === 'add') {
                    const w = toWorld(ev.clientX, ev.clientY);
                    placeAt(w.x, w.y);
                }
                if (dist < 5 && mode === 'move') {
                    if (!ev.target.closest('.gn')) { selected = null; render(); renderInspector(); }
                }
            }
            pan = null; downBg = null;
        });

        // Touch events simplified...
        wrap.addEventListener('touchstart', ev => {
            if (ev.touches.length === 2) {
                const [a, b] = ev.touches;
                const r = wrap.getBoundingClientRect();
                pinch = { d: Math.hypot(a.clientX - b.clientX, a.clientY - b.clientY),
                          cx: (a.clientX + b.clientX) / 2 - r.left, cy: (a.clientY + b.clientY) / 2 - r.top,
                          s: view.scale };
                return;
            }
            const t = ev.touches[0];
            const gn = elemGn(document.elementFromPoint(t.clientX, t.clientY));
            if (gn) {
                const n = byId(+gn.dataset.id);
                if (mode === 'link') {
                    tLink = { from: n.id, temp: tempPath() };
                } else {
                    tDrag = { n, el: gn, dx: n.x - toWorld(t.clientX, t.clientY).x, dy: n.y - toWorld(t.clientX, t.clientY).y, moved: false };
                }
                ev.preventDefault();
            }
        }, { passive: false });

        wrap.addEventListener('touchmove', ev => {
            ev.preventDefault();
            if (ev.touches.length === 2 && pinch) {
                const [a, b] = ev.touches;
                const d = Math.hypot(a.clientX - b.clientX, a.clientY - b.clientY);
                const ns = Math.min(3, Math.max(0.2, pinch.s * d / pinch.d));
                zoomAt(ns / view.scale, pinch.cx, pinch.cy);
                return;
            }
            const t = ev.touches[0];
            if (tDrag) {
                tDrag.moved = true;
                const w = toWorld(t.clientX, t.clientY);
                tDrag.n.x = w.x + tDrag.dx; tDrag.n.y = w.y + tDrag.dy;
                tDrag.el.style.left = tDrag.n.x + 'px';
                tDrag.el.style.top = tDrag.n.y + 'px';
                updateEdges();
            } else if (tLink) {
                const w = toWorld(t.clientX, t.clientY);
                const a = byId(tLink.from);
                tLink.temp.setAttribute('d', `M ${a.x} ${a.y} L ${w.x} ${w.y}`);
            }
        }, { passive: false });

        wrap.addEventListener('touchend', ev => {
            const t = ev.changedTouches[0];
            if (tDrag && !tDrag.moved) { selected = tDrag.n.id; render(); renderInspector(); }
            if (tDrag) { tDrag.el.classList.remove('dragging'); render(); }
            if (tLink) {
                const gn = elemGn(document.elementFromPoint(t.clientX, t.clientY));
                if (gn) connect(tLink.from, +gn.dataset.id);
                tLink.temp.remove();
            }
            tDrag = tLink = pinch = null;
        });

        function tempPath() {
            const p = document.createElementNS(SVG_NS, 'path');
            p.setAttribute('class', 'gtemp');
            $('edgeSvg').appendChild(p);
            return p;
        }

        $('edgeSvg').addEventListener('click', ev => {
            const p = ev.target.closest('.gedge');
            if (p) insertOnEdge(+p.dataset.from, +p.dataset.to);
        });
        $('edgeSvg').addEventListener('contextmenu', ev => {
            const p = ev.target.closest('.gedge');
            if (p) { ev.preventDefault(); removeEdge(+p.dataset.from, +p.dataset.to); }
        });
        $('world').addEventListener('contextmenu', ev => {
            const gn = ev.target.closest('.gn');
            if (gn) { ev.preventDefault(); removeNeuron(+gn.dataset.id); }
        });
        $('world').addEventListener('dblclick', ev => {
            const gn = ev.target.closest('.gn');
            if (!gn) return;
            const n = byId(+gn.dataset.id);
            const c = prompt(T('net.cmtPrompt'), n.comment || '');
            if (c !== null) { n.comment = c; render(); renderInspector(); }
        });
        $('world').addEventListener('click', ev => {
            const gn = ev.target.closest('.gn');
            if (gn && mode === 'move') { selected = +gn.dataset.id; render(); renderInspector(); }
        });
    }

    async function loadDatasets() {
        const r = await fetch('/api/datasets');
        datasets = await r.json();
    }

    function applyDataset(name) {
        if (name === 'manual' || !datasets || !datasets[name]) return;
        const d = datasets[name];
        $('dataX').value = d.X.map(r2 => r2.join(',')).join('\n');
        $('dataY').value = d.y.map(r2 => r2.join(',')).join('\n');
        graph = { neurons: [], edges: [], loss: d.loss };
        folders = []; nextId = 1; nextFolderId = 1;
        selected = null; lastResult = null;
        const inIds = [], outIds = [];
        for (let i = 0; i < d.inputs; i++)
            inIds.push(addNeuronAt(120, 200 + i * 70 - (d.inputs - 1) * 35, 'input', null).id);
        for (let k = 0; k < d.outputs; k++)
            outIds.push(addNeuronAt(700, 200 + k * 70, 'output',
                d.loss === 'mse' ? 'linear' : 'sigmoid').id);
        inIds.forEach(a => outIds.forEach(b => graph.edges.push({ from: a, to: b })));
        lossHist = []; drawChart();
        render(); renderInspector(); fit();
    }

    function parseData(text, ctx) {
        const rows = text.trim().split(/\r?\n/).filter(r2 => r2.trim())
            .map((r2, i) => r2.split(/[,;\t]+/).map(v => {
                const n = parseFloat(v);
                if (isNaN(n)) throw new Error(T('net.parse.notNum', { ctx, n: i + 1, v }));
                return n;
            }));
        if (!rows.length) throw new Error(T('net.parse.empty', { ctx }));
        return rows;
    }
    const getData = () => ({ X: parseData($('dataX').value, 'X'), y: parseData($('dataY').value, 'y') });

    async function refreshLibrary() {
        const sel = $('libSelect');
        if (!sel) return;
        try {
            const r = await fetch('/api/neuron/list');
            library = await r.json();
        } catch (e) { library = []; }
        sel.innerHTML = `<option value="">${T('net.lib.chooseNeuron')}</option>` +
            library.map(c => `<option value="${c.name}">${c.name} · ${T('k.' + c.kind)}${c.activation ? ' · ' + c.activation : ''}</option>`).join('');
    }

    function addFromLibrary() {
        const sel = $('libSelect');
        if (!sel) return;
        const card = library.find(c => c.name === sel.value);
        if (!card) return flash(T('net.lib.pickFirst'));
        const trainable = $('libTrainable') ? $('libTrainable').checked : true;
        const c = viewportCenterWorld();
        const n = addNeuronAt(c.x + (Math.random() * 80 - 40), c.y + (Math.random() * 80 - 40),
            card.kind, card.activation || null);
        n.comment = card.comment || '';
        if (typeof card.b === 'number' && card.b !== 0) n.b = card.b;
        if (!trainable) n.trainable = false;
        selected = n.id;
        render(); renderInspector();
    }

    function graphPayload() {
        return {
            neurons: graph.neurons.map(n => {
                const o = { id: n.id, kind: n.kind, activation: n.activation, comment: n.comment,
                            trainable: n.trainable !== false };
                if (n.b !== undefined) o.b = n.b;
                return o;
            }),
            edges: graph.edges.map(e => ({ from: e.from, to: e.to, w: e.w })),
            folders: folders.map(f => ({ ...f })),
            loss: graph.loss,
        };
    }

    function syncWeights(d) {
        const wmap = new Map(d.weights.map(w => [w.from + '→' + w.to, w.w]));
        graph.edges.forEach(e => { const w = wmap.get(edgeKey(e)); if (w !== undefined) e.w = w; });
    }

    async function trainStep(reset = false) {
        const { X, y } = getData();
        const r = await fetch('/api/gnet/train_step', {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                graph: graphPayload(), X, y,
                epochs: Math.max(1, +$('epStep').value || 100),
                lr: +$('lr').value || 0.5,
                batch_size: +$('batch').value || null,
                reset,
            }),
        });
        const d = await r.json();
        if (!r.ok) throw new Error(d.detail || T('net.train.err'));
        lossHist = d.loss_history;
        const me = $('mEpochs'); if (me) me.textContent = d.epochs_done;
        const ml = $('mLoss'); if (ml) ml.textContent = d.final_loss != null ? d.final_loss.toPrecision(3) : '—';
        drawChart();
        return d;
    }

    function setTrainingUI(on) {
        training = on;
        const b = $('trainBtn');
        if (b) b.textContent = on ? T('net.train.pause') : T('net.train.run');
        $('trainProg').style.width = on ? '100%' : '0%';
    }

    async function trainingLoop() {
        if (!training) return;
        try {
            const d = await trainStep(false);
            flash(T('net.train.loop', { n: d.epochs_done, l: d.final_loss.toPrecision(3) }));
            await predictAll(false);
            if ($('testIn')) await runTest();
            if (d.final_loss < 1e-4) { setTrainingUI(false); flash(T('net.train.done')); return; }
            setTimeout(trainingLoop, 30);
        } catch (e) {
            setTrainingUI(false);
            flash(T('err.prefix') + e.message);
        }
    }

    async function runTest() {
        const v = $('testIn').value.trim();
        if (!v) return;
        const row = v.split(/[,;\t]+/).map(Number);
        try {
            const r = await fetch('/api/gnet/predict', {
                method: 'POST', headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ X: [row] }),
            });
            const d = await r.json();
            const out = d.output[0];
            $('testOut').textContent = T('net.test.result', { in: row.join(', '), out: out.map(x => x.toFixed(4)).join(', ') });
        } catch (e) {
            $('testOut').textContent = T('net.test.errPrefix') + e.message;
        }
    }

    async function runTestTable() {
        if (!$('testX') || !$('testTable')) return;
        const out = $('testOut2');
        try {
            const X = parseData($('testX').value, T('net.parse.ctx.testX'));
            const yText = $('testY').value.trim();
            const y = yText ? parseData(yText, T('net.parse.ctx.testY')) : null;
            const r = await fetch('/api/gnet/predict', {
                method: 'POST', headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ X }),
            });
            const d = await r.json();
            let html = `<tr><th>✓</th><th>${T('net.tbl.in')}</th><th>${T('net.tbl.pred')}</th><th>${T('net.tbl.target')}</th><th>${T('net.tbl.diff')}</th></tr>`;
            d.output.forEach((row, i) => {
                const tRow = y ? y[i] : null;
                html += `<tr><td>${tRow ? '?' : ''}</td><td>${X[i].join(', ')}</td><td>${row.map(v => (+v.toFixed(3))).join(', ')}</td><td>${tRow ? tRow.join(', ') : ''}</td><td>—</td></tr>`;
            });
            $('testTable').innerHTML = html;
        } catch (e) { if (out) out.textContent = T('net.test.errPrefix') + e.message; }
    }

    function rebuildLiveInputs() {
        const box = $('liveInputs');
        if (!box) return;
        const ins = graph.neurons.filter(n => n.kind === 'input');
        if (box.children.length === ins.length) return;
        box.innerHTML = ins.map((n, k) =>
            `<input type="number" step="0.05" value="0" title="${labelOf(n)}">`).join('');
        box.querySelectorAll('input').forEach(i => i.addEventListener('input', () => {
            clearTimeout(liveTimer);
            liveTimer = setTimeout(livePredict, 180);
        }));
    }

    async function livePredict() {
        const box = $('liveInputs');
        if (!box) return;
        const row = Array.prototype.map.call(box.querySelectorAll('input'), i => parseFloat(i.value) || 0);
        try {
            const r = await fetch('/api/gnet/predict', {
                method: 'POST', headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ X: [row] }),
            });
            const d = await r.json();
            lastResult = d; selectedRow = 0;
            syncWeights(d);
            const lo = $('liveOut');
            if (lo) lo.textContent = d.output[0].map(x => x.toFixed(4)).join(', ');
            render();
        } catch (e) {
            const lo = $('liveOut');
            if (lo) lo.textContent = e.message;
        }
    }

    async function predictAll(ui = true) {
        try {
            const { X } = getData();
            const r = await fetch('/api/gnet/predict', {
                method: 'POST', headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ X }),
            });
            const d = await r.json();
            lastResult = d;
            syncWeights(d);
            if ($('predTable')) renderTable(X);
            render();
            if (ui && $('predOut')) $('predOut').textContent = d.output.map(r2 => r2.join(', ')).join('\n');
        } catch (e) { if (ui && $('predOut')) $('predOut').textContent = T('err.prefix') + e.message; }
    }

    function renderTable(X) {
        let y = null; try { y = getData().y; } catch (e) {}
        const out = lastResult.output;
        let html = `<tr><th>✓</th><th>${T('net.tbl.in')}</th><th>${T('net.tbl.pred')}</th><th>${T('net.tbl.target')}</th><th>${T('net.tbl.diff')}</th></tr>`;
        out.forEach((row, i) => {
            const tRow = y ? y[i] : null;
            html += `<tr data-i="${i}" class="${i === selectedRow ? 'sel' : ''}"><td>?</td><td>${X[i].join(', ')}</td><td>${row.map(v => (+v.toFixed(3))).join(', ')}</td><td>${tRow ? tRow.join(', ') : ''}</td><td>—</td></tr>`;
        });
        $('predTable').innerHTML = html;
        $('predTable').querySelectorAll('tr[data-i]').forEach(tr =>
            tr.onclick = () => { selectedRow = +tr.dataset.i; renderTable(X); render(); });
    }

    function drawChart() {
        const c = $('lossChart');
        if (!c) return;
        const dpr = window.devicePixelRatio || 1;
        const w = c.clientWidth || 260, h = 110;
        c.width = w * dpr; c.height = h * dpr;
        const ctx = c.getContext('2d');
        ctx.scale(dpr, dpr);
        ctx.clearRect(0, 0, w, h);
        if (!lossHist || lossHist.length < 2) return;
        const cs = getComputedStyle(document.documentElement);
        const accent = cs.getPropertyValue('--accent').trim() || '#6f7ff0';
        const dim = 'rgba(128,128,140,.5)';
        const max = Math.max(...lossHist), min = Math.min(...lossHist);
        const n = lossHist.length;
        ctx.strokeStyle = dim; ctx.lineWidth = 1;
        ctx.strokeRect(0.5, 0.5, w - 1, h - 1);
        ctx.beginPath();
        lossHist.forEach((v, i) => {
            const x = 4 + (w - 8) * i / (n - 1);
            const yv = max === min ? h / 2 : h - 6 - (h - 12) * (v - min) / (max - min);
            i ? ctx.lineTo(x, yv) : ctx.moveTo(x, yv);
        });
        ctx.strokeStyle = accent; ctx.lineWidth = 2; ctx.stroke();
    }

    async function refreshCfgList() {
        const sel = $('loadCfgSelect');
        if (!sel) return;
        const r = await fetch('/api/gnet/list');
        const names = await r.json();
        sel.innerHTML = `<option value="">${T('net.cfg.load')}</option>` +
            names.map(n => `<option>${n}</option>`).join('');
    }

    async function refreshModuleList() {
        const sel = $('loadModSelect');
        if (!sel) return;
        const r = await fetch('/api/module/list');
        const names = await r.json();
        sel.innerHTML = `<option value="">${T('net.cfg.load')}</option>` +
            names.map(n => `<option>${n}</option>`).join('');
    }

    async function saveCurrentModule() {
        if (!clipboard || !clipboard.neurons.length)
            return alert(T('net.mod.needCopy'));
        const name = $('modName').value.trim();
        if (!name) return alert(T('net.mod.needName'));
        const payload = {
            neurons: clipboard.neurons.map(p => ({ ...p })),
            edges: clipboard.edges.map(e => ({ ...e })),
        };
        const r = await fetch('/api/module/save', {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name, graph: payload }),
        });
        const d = await r.json();
        if (!r.ok) return alert(T('err.prefix') + (d.detail || r.status));
        flash(T('net.mod.saved', { name: d.saved, n: payload.neurons.length, m: payload.edges.length }));
        await refreshModuleList();
    }

    async function loadModule() {
        const name = $('loadModSelect') && $('loadModSelect').value;
        if (!name) return;
        const replaceInputs = $('modReplaceInputs') && $('modReplaceInputs').checked;
        const r = await fetch('/api/module/load/' + encodeURIComponent(name));
        if (!r.ok) {
            const d = await r.json().catch(() => ({}));
            return alert(T('err.prefix') + (d.detail || r.status));
        }
        const mod = await r.json();
        if (!mod.neurons || !mod.neurons.length) return alert(T('net.mod.empty'));
        const idMap = new Map();
        const pts = mod.neurons.map(p => {
            const nid = nextId++;
            idMap.set(p.id, nid);
            const kind = replaceInputs && p.kind === 'input' ? 'hidden' : p.kind;
            return { ...p, id: nid, kind };
        });
        const edges = (mod.edges || [])
            .filter(e => idMap.has(e.from) && idMap.has(e.to))
            .map(e => ({ from: idMap.get(e.from), to: idMap.get(e.to), w: e.w }));
        const c = viewportCenterWorld();
        const dx = c.x - (Math.min(...pts.map(p => p.x)) + Math.max(...pts.map(p => p.x))) / 2;
        const dy = c.y - (Math.min(...pts.map(p => p.y)) + Math.max(...pts.map(p => p.y))) / 2;
        pts.forEach(p => { p.x += dx; p.y += dy; });
        graph.neurons.push(...pts);
        graph.edges.push(...edges);
        render(); renderInspector();
        flash(T('net.mod.loaded', { name, n: pts.length, m: edges.length }) +
            (replaceInputs ? T('net.mod.asPorts') : ''));
    }

    function applyTablesHandoff(kind) {
        let applied = false;
        const use = (key, xId, yId) => {
            const raw = localStorage.getItem('neuro_tables_' + key);
            const xEl = $(xId);
            if (!raw || !xEl) return;
            try {
                const d = JSON.parse(raw);
                xEl.value = d.X;
                const yEl = $(yId);
                if (yEl && d.y) yEl.value = d.y;
                applied = true;
            } catch (e) {}
        };
        if (kind === 'train' || kind === 'both') use('train', 'dataX', 'dataY');
        if (kind === 'test' || kind === 'both') use('test', 'testX', 'testY');
        return applied;
    }

    function bindUI() {
        const on = (id, ev, fn) => { const el = $(id); if (el) el.addEventListener(ev, fn); };
        const segMode = $('segMode');
        if (segMode) segMode.querySelectorAll('button').forEach(b => b.onclick = () => setMode(b.dataset.v));
        const segKind = $('segKind');
        if (segKind) segKind.querySelectorAll('button').forEach(b => b.onclick = () => {
            addKind = b.dataset.v;
            segKind.querySelectorAll('button').forEach(x => x.classList.toggle('active', x === b));
            const aw = $('actWrap');
            if (aw) aw.style.visibility = addKind === 'input' ? 'hidden' : 'visible';
        });
        const segHow = $('segHow');
        if (segHow) segHow.querySelectorAll('button').forEach(b => b.onclick = () => {
            addHow = b.dataset.v;
            segHow.querySelectorAll('button').forEach(x => x.classList.toggle('active', x === b));
            const cw = $('cntWrap');
            if (cw) cw.hidden = addHow !== 'multi';
        });

        on('dsSelect', 'change', e2 => applyDataset(e2.target.value));
        const trainBtn = $('trainBtn');
        if (trainBtn) trainBtn.onclick = () => {
            if (training) { setTrainingUI(false); return; }
            setTrainingUI(true); trainingLoop();
        };
        on('stepBtn', 'click', async () => {
            await trainStep(false);
            await predictAll();
        });
        on('resetBtn', 'click', async () => {
            setTrainingUI(false);
            await fetch('/api/gnet/reset', { method: 'POST' });
            lossHist = []; lastResult = null;
            graph.edges.forEach(e2 => delete e2.w);
            drawChart(); render();
        });
        on('repredBtn', 'click', () => { predictAll(); });
        on('testBtn', 'click', () => runTest());
        on('saveCfgBtn', 'click', async () => {
            const name = $('cfgName').value.trim();
            if (!name) return alert(T('net.cfg.needName'));
            const g = JSON.parse(JSON.stringify(graphPayload()));
            graph.neurons.forEach((n, i) => { g.neurons[i].x = n.x; g.neurons[i].y = n.y; });
            await fetch('/api/gnet/save', {
                method: 'POST', headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ name, graph: g }),
            });
            refreshCfgList();
        });
        on('loadCfgBtn', 'click', async () => {
            const name = $('loadCfgSelect').value;
            const r = await fetch('/api/gnet/load/' + encodeURIComponent(name));
            const d = await r.json();
            const g = d.graph;
            graph = { neurons: g.neurons.map(n => ({ ...n })), edges: (g.edges || []).map(e => ({ ...e })), loss: g.loss || 'bce' };
            folders = (g.folders || []).map(f => ({ ...f }));
            nextId = Math.max(0, ...graph.neurons.map(n => n.id)) + 1;
            selected = null; lastResult = null;
            render(); renderInspector(); fit();
        });
        on('saveModBtn', 'click', saveCurrentModule);
        on('loadModBtn', 'click', loadModule);
        on('libAddBtn', 'click', addFromLibrary);
        on('libRefreshBtn', 'click', refreshLibrary);
        on('runTestTableBtn', 'click', () => runTestTable());
        on('takeTablesTrainBtn', 'click', () => applyTablesHandoff('train'));
        on('takeTablesTestBtn', 'click', () => applyTablesHandoff('test'));

        const bindFile = (id, target) => on(id, 'change', ev => {
            const f = ev.target.files[0];
            if (f) f.text().then(t => { $(target).value = t; });
        });
        bindFile('fileX', 'dataX');
        bindFile('fileY', 'dataY');

        on('openWinBtn', 'click', () => window.open(location.pathname, 'neuroCtor', 'width=1280,height=860'));
    }

    function flash(msg) { const el = $('trainStatus'); if (el) el.textContent = msg; }

    async function init() {
        // Создание элемента тултипа
        tooltip = document.createElement('div');
        tooltip.className = 'node-info-tooltip';
        document.body.appendChild(tooltip);

        bindStage(); bindUI();
        window.addEventListener('langchange', () => {
            setTrainingUI(training);
            render(); renderInspector();
            refreshLibrary(); refreshCfgList(); refreshModuleList();
        });
        await loadDatasets();
        await refreshCfgList();
        await refreshModuleList();
        await refreshLibrary();

        if (sessionStorage.getItem('neuro_tables_apply')) {
            sessionStorage.removeItem('neuro_tables_apply');
            applyTablesHandoff('both');
        }
        const x1 = addNeuronAt(120, 170, 'input', null);
        const x2 = addNeuronAt(120, 250, 'input', null);
        const y1 = addNeuronAt(700, 210, 'output', 'sigmoid');
        graph.edges.push({ from: x1.id, to: y1.id }, { from: x2.id, to: y1.id });
        render(); renderInspector();
        setTimeout(fit, 80);
    }
    document.addEventListener('DOMContentLoaded', init);

    return { graph: () => graph, addFromLibrary, refreshLibrary, runTestTable, livePredict };
})();
