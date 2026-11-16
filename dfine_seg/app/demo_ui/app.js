// D-FINE-seg demo client. No build step, no dependencies: talks to dfine_seg/app/demo.py.
'use strict';

const $ = (id) => document.getElementById(id);
// task -> [dropdown group, short name in the closed dropdown]
const TASK_GROUPS = {
  detect: ['Detection · COCO', 'detection'],
  segment: ['Instance segmentation · COCO', 'instance segmentation'],
  sem_seg: ['Semantic segmentation · Cityscapes', 'semantic segmentation'],
  kpts: ['Keypoints · COCO person', 'keypoints'],
};
const CUSTOM = '__custom';

const S = {
  backend: 'dfine',
  mode: 'file',
  model: null, // info of the loaded model
  sam3: false,
  samUsed: false,
  file: null, // the open image or video
  kind: null, // 'image' | 'video'
  imgBusy: false,
  imgPending: false,
  cam: null, // MediaStream while the webcam runs
  job: null, // running video job
  lastJob: null, // finished job whose uploaded source "Process again" reuses
};

// ─── helpers ────────────────────────────────────────────────────────────
async function api(path, { json, body, query } = {}) {
  const url = query ? `${path}?${new URLSearchParams(query)}` : path;
  const headers = { 'X-DFine': '1' };
  if (json) headers['Content-Type'] = 'application/json';
  const res = await fetch(url, { method: 'POST', headers, body: json ? JSON.stringify(json) : body });
  if (!res.ok) {
    let msg = `${res.status} ${res.statusText}`;
    try { msg = (await res.json()).error || msg; } catch (_) { /* not JSON */ }
    throw new Error(msg);
  }
  return res;
}

let toastTimer;
function toast(msg) {
  const t = $('toast');
  t.textContent = msg;
  t.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { t.hidden = true; }, 7000);
}

function setPill(kind, text) {
  const p = $('pill');
  p.className = `pill mono ${kind}`;
  p.lastElementChild.textContent = text;
  p.title = text;
}

function swapUrl(el, attr, blob) {
  const old = el.dataset.url;
  el.dataset.url = URL.createObjectURL(blob);
  el[attr] = el.dataset.url;
  if (old) URL.revokeObjectURL(old);
  return el.dataset.url;
}

function settings(extra = {}) {
  return {
    backend: S.backend,
    prompt: $('prompt').value,
    conf: $('conf').value,
    minimize: $('minimize').checked ? '1' : '0',
    ...extra,
  };
}

const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

// ─── model ──────────────────────────────────────────────────────────────
function describe(m) {
  if (!m || !m.backend) return '';
  const classes = m.classes ? `${m.classes} class${m.classes === 1 ? '' : 'es'}` : 'classes unknown, name them in Advanced';
  return [m.backend, m.task || 'task from graph', classes, m.device, m.input && `input ${m.input}`]
    .filter(Boolean).join(' · ');
}

function showModel(m) {
  S.model = m;
  $('model-info').className = 'model-info small';
  $('model-info').innerHTML = `<b>${esc(m.name)}</b><br>${esc(describe(m))}`;
  if (S.backend === 'dfine') setPill('ok', m.preset ? m.spec : `${m.name} · ${m.task || 'graph'}`);
}

function syncSelect(spec) {
  const sel = $('model');
  const preset = [...sel.options].some((o) => o.value === spec);
  sel.value = preset ? spec : CUSTOM;
  $('custom-row').hidden = preset;
  if (!preset) $('path').value = spec;
}

async function loadModel(spec) {
  setPill('busy', `loading ${spec}…`);
  $('model-info').textContent = 'loading…';
  try {
    const res = await api('/api/load', { json: { model: spec, names: $('names').value, input_size: $('size').value } });
    showModel((await res.json()).model);
    syncSelect(S.model.spec);
    runImage();
  } catch (e) {
    toast(`Could not load ${spec}: ${e.message}`);
    if (S.model) { showModel(S.model); syncSelect(S.model.spec); } else setPill('bad', 'no model');
  }
}

async function uploadWeights(file) {
  setPill('busy', `uploading ${file.name}…`);
  try {
    const res = await api('/api/upload', { body: file, query: { name: file.name } });
    await loadModel((await res.json()).path);
  } catch (e) {
    toast(`Upload failed: ${e.message}`);
    if (S.model) showModel(S.model);
  }
}

function setBackend(b) {
  S.backend = b;
  for (const t of $('backend').children) t.setAttribute('aria-selected', String(t.dataset.backend === b));
  $('dfine-controls').hidden = b !== 'dfine';
  $('sam-controls').hidden = b !== 'sam3';
  if (b === 'sam3') setPill('ok', `SAM3 · ${$('prompt').value}`);
  else if (S.model) showModel(S.model);
  runImage();
}

// ─── image or video file ────────────────────────────────────────────────
let busyTimer;
function busy(on) {
  clearTimeout(busyTimer);
  if (!on) { $('busy').hidden = true; return; }
  $('busy-text').textContent = S.backend === 'sam3' && !S.samUsed
    ? 'Loading SAM3 - the first run can take a while…' : 'Running…';
  busyTimer = setTimeout(() => { $('busy').hidden = false; }, 250); // fast runs never flash it
}

function show(ids) {
  const all = ['drop', 'compare', 'vid-in', 'vid-preview', 'vid-out', 'vid-progress', 'file-new', 'vid-cancel', 'file-dl', 'vid-run'];
  for (const id of all) $(id).hidden = !ids.includes(id);
}

// Before a run "Process video" is the main action; once there is a result, Download is.
function setRunButton(text, primary) {
  $('vid-run').textContent = text;
  $('vid-run').classList.toggle('btn-primary', primary);
  $('vid-run').classList.toggle('btn-ghost', !primary);
}

function openFile(file) {
  if (!file) return;
  const kind = file.type.startsWith('image/') ? 'image' : file.type.startsWith('video/') ? 'video' : null;
  if (!kind) return toast(`${file.name}: not an image or a video.`);
  clearFile();
  S.file = file;
  S.kind = kind;
  setMode('file');
  $('file-dl').download = kind === 'image' ? 'dfine-seg.jpg' : 'dfine-seg.mp4';
  if (kind === 'image') {
    swapUrl($('img-in'), 'src', file);
    runImage();
  } else {
    swapUrl($('vid-in'), 'src', file);
    setRunButton('Process video', true);
    show(['vid-in', 'vid-run', 'file-new']);
    $('file-stats').textContent = `${file.name} · ${(file.size / 1e6).toFixed(1)} MB`;
  }
}

function clearFile() {
  if (S.job) api(`/api/video/${S.job}/cancel`).catch(() => {});
  Object.assign(S, { file: null, kind: null, job: null, lastJob: null });
  $('vid-in').pause();
  $('vid-out').pause();
  $('vid-out').removeAttribute('src');
  show(['drop']);
  $('file-stats').textContent = '';
  $('file').value = '';
  $('stride-ctrl').hidden = true;
}

// Slider drags fire faster than the model: one request in flight, the latest settings queued.
async function runImage() {
  if (S.kind !== 'image' || S.mode !== 'file') return;
  if (S.imgBusy) { S.imgPending = true; return; }
  S.imgBusy = true;
  S.imgPending = false;
  busy(true);
  const file = S.file;
  try {
    const res = await api('/api/image', { body: file, query: settings() });
    const blob = await res.blob();
    if (S.file !== file) return; // another file was opened meanwhile
    const out = $('img-out');
    $('file-dl').href = swapUrl(out, 'src', blob);
    await out.decode();
    if (S.backend === 'sam3') S.samUsed = true;
    show(['compare', 'file-new', 'file-dl']);
    setWipe(Number($('compare').dataset.p || 0)); // the knob clamp needs the shown width
    const ms = res.headers.get('X-Infer-Ms');
    $('file-stats').innerHTML = `<b>${ms} ms</b> · ${esc(res.headers.get('X-Summary'))} · ${out.naturalWidth}×${out.naturalHeight}`;
  } catch (e) {
    toast(e.message);
  } finally {
    S.imgBusy = false;
    busy(false);
    if (S.imgPending) runImage();
  }
}

// wipe: left of the line shows the input, right of it the result
function setWipe(p) {
  p = Math.min(1, Math.max(0, p));
  const c = $('compare');
  const w = c.clientWidth;
  c.style.setProperty('--wipe', `${p * 100}%`);
  // keep the knob inside the frame at the edges
  c.style.setProperty('--knob', `${Math.min(Math.max(p * w, 22), w - 22) - p * w}px`);
  c.style.setProperty('--tag-l', p > 0.12 ? 1 : 0);
  $('wipe').setAttribute('aria-valuenow', Math.round(p * 100));
  c.dataset.p = p;
}

function bindWipe() {
  const c = $('compare');
  const move = (e) => {
    const r = c.getBoundingClientRect();
    setWipe((e.clientX - r.left) / r.width);
  };
  c.addEventListener('pointerdown', (e) => {
    c.setPointerCapture(e.pointerId);
    move(e);
    c.onpointermove = move;
  });
  c.addEventListener('pointerup', () => { c.onpointermove = null; });
  $('wipe').addEventListener('keydown', (e) => {
    const step = { ArrowLeft: -0.05, ArrowRight: 0.05 }[e.key];
    if (step) { e.preventDefault(); setWipe(Number(c.dataset.p || 0) + step); }
  });
  window.addEventListener('resize', () => setWipe(Number(c.dataset.p || 0)));
  setWipe(0);
}

// ─── webcam ─────────────────────────────────────────────────────────────
async function startCam() {
  if (!navigator.mediaDevices?.getUserMedia) {
    $('cam-note').textContent = 'The browser only allows the camera on localhost or https. Open this page via 127.0.0.1, or use an SSH tunnel.';
    return;
  }
  try {
    S.cam = await navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false });
  } catch (e) {
    return toast(`Camera: ${e.message}`);
  }
  const v = $('cam-src');
  v.srcObject = S.cam;
  await v.play();
  $('cam-idle').hidden = true;
  $('cam-out').hidden = false;
  $('cam-stop').hidden = false;
  camLoop(S.cam);
}

async function camLoop(stream) {
  const v = $('cam-src');
  const grab = document.createElement('canvas');
  const out = $('cam-out');
  const ctx = out.getContext('2d');
  const times = [];
  while (S.cam === stream) {
    const scale = Math.min(1, 1280 / v.videoWidth);
    grab.width = Math.round(v.videoWidth * scale);
    grab.height = Math.round(v.videoHeight * scale);
    grab.getContext('2d').drawImage(v, 0, 0, grab.width, grab.height);
    const frame = await new Promise((r) => grab.toBlob(r, 'image/jpeg', 0.85));
    try {
      const res = await api('/api/image', { body: frame, query: settings({ quality: 80 }) });
      const bmp = await createImageBitmap(await res.blob());
      if (S.cam !== stream) break;
      if (out.width !== bmp.width) { out.width = bmp.width; out.height = bmp.height; }
      ctx.drawImage(bmp, 0, 0);
      const size = `${bmp.width}×${bmp.height}`;
      bmp.close();
      if (S.backend === 'sam3') S.samUsed = true;
      const now = performance.now();
      times.push(now);
      while (times[0] < now - 1000) times.shift();
      $('cam-stats').innerHTML = `<b>${times.length} fps</b> · model ${res.headers.get('X-Infer-Ms')} ms · ${esc(res.headers.get('X-Summary'))} · ${size}`;
    } catch (e) {
      toast(e.message);
      stopCam();
    }
  }
}

function stopCam() {
  if (!S.cam) return;
  S.cam.getTracks().forEach((t) => t.stop());
  S.cam = null;
  $('cam-src').srcObject = null;
  $('cam-idle').hidden = false;
  $('cam-out').hidden = true;
  $('cam-stop').hidden = true;
  $('cam-stats').textContent = '';
}

// ─── video jobs ─────────────────────────────────────────────────────────
// The first run uploads the file; "Process again" reruns the source the server kept.
async function startJob() {
  const query = settings({ stride: $('stride').value });
  if (S.lastJob) {
    try {
      return (await (await api(`/api/video/${S.lastJob}/rerun`, { query })).json()).job;
    } catch (_) { /* source gone (server restarted?): upload again */ }
  }
  $('file-stats').textContent = 'uploading…';
  return (await (await api('/api/video', { body: S.file, query })).json()).job;
}

async function runVideo() {
  $('vid-in').pause();
  $('vid-out').removeAttribute('src'); // the server drops the previous result on a new job
  $('vid-out').load();
  show(['vid-in', 'vid-progress']);
  $('vid-progress').firstElementChild.style.width = '0';
  const file = S.file;
  let id;
  try {
    id = await startJob();
  } catch (e) {
    toast(e.message);
    show(['vid-in', 'vid-run', 'file-new']);
    return;
  }
  if (S.file !== file) return;
  S.job = id;
  $('vid-cancel').hidden = false;
  let shown = -1;
  while (S.job === id) {
    await new Promise((r) => setTimeout(r, 400));
    let res, st;
    try {
      res = await fetch(`/api/video/${id}`);
      st = await res.json();
    } catch (_) { continue; } // network blip: keep polling
    if (S.job !== id) return;
    if (!res.ok) { // server restarted, or a newer job cleared this finished one
      return finishVideo(id, { status: 'error', error: 'This video job is gone from the server. Process it again.' });
    }
    const pct = st.total ? Math.min(100, (100 * st.frames) / st.total) : 0;
    $('vid-progress').firstElementChild.style.width = `${pct}%`;
    const fps = st.fps ? ` · <b>${st.fps.toFixed(1)} fps</b>` : '';
    $('file-stats').innerHTML = `${st.status} · ${st.frames}${st.total ? ` / ${st.total}` : ''} frames${fps}`;
    if (st.frames > 0 && st.frames > shown && st.status === 'running') {
      shown = st.frames;
      const pv = await fetch(`/api/video/${id}/preview`);
      if (pv.ok && S.job === id) {
        swapUrl($('vid-preview'), 'src', await pv.blob());
        $('vid-in').hidden = true;
        $('vid-preview').hidden = false;
      }
    }
    if (['done', 'cancelled', 'error'].includes(st.status)) return finishVideo(id, st);
  }
}

function finishVideo(id, st) {
  S.job = null;
  S.lastJob = id;
  setRunButton('Process again', st.status === 'error');
  if (st.status === 'error') {
    toast(st.error);
    show(['vid-in', 'vid-run', 'file-new']);
    return;
  }
  if (S.backend === 'sam3') S.samUsed = true;
  const url = `/api/video/${id}/result`;
  $('file-dl').href = url;
  if (st.playable) {
    $('vid-out').src = url;
    show(['vid-out', 'vid-run', 'file-new', 'file-dl']);
  } else {
    show(['vid-in', 'vid-run', 'file-new', 'file-dl']);
    toast('ffmpeg is not installed, so the result is mp4v, which browsers cannot play. Download it instead.');
  }
  const settingsUsed = `conf ${Number($('conf').value).toFixed(2)}, stride ${$('stride').value}`;
  $('file-stats').innerHTML = `${st.status} · ${st.frames} frames · <b>${st.fps.toFixed(1)} fps</b> · ${settingsUsed}`;
}

// ─── modes ──────────────────────────────────────────────────────────────
function setMode(mode) {
  if (S.mode === 'webcam' && mode !== 'webcam') stopCam(); // release the camera
  S.mode = mode;
  for (const t of $('modes').children) t.setAttribute('aria-selected', String(t.dataset.mode === mode));
  for (const v of document.querySelectorAll('.view')) v.hidden = v.dataset.view !== mode;
  $('stride-ctrl').hidden = !(mode === 'file' && S.kind === 'video');
}

function dropZone(el, onFile) {
  el.addEventListener('dragover', (e) => { e.preventDefault(); el.classList.add('over'); });
  el.addEventListener('dragleave', () => el.classList.remove('over'));
  el.addEventListener('drop', (e) => {
    e.preventDefault();
    el.classList.remove('over');
    if (e.dataTransfer.files[0]) onFile(e.dataTransfer.files[0]);
  });
}

// ─── init ───────────────────────────────────────────────────────────────
async function init() {
  const st = await (await fetch('/api/state')).json();
  S.sam3 = st.sam3;
  const sel = $('model');
  for (const [task, [group, short]] of Object.entries(TASK_GROUPS)) {
    const g = document.createElement('optgroup');
    g.label = group;
    for (const p of st.presets.filter((x) => x.task === task)) g.append(new Option(`${p.size.toUpperCase()} · ${short}`, p.key));
    sel.append(g);
  }
  sel.append(new Option('Custom path…', CUSTOM));
  if (st.model?.spec) { showModel(st.model); syncSelect(st.model.spec); }

  const samTab = $('backend').children[1];
  if (!st.sam3) {
    samTab.disabled = true;
    samTab.title = "SAM3 needs transformers: pip install 'dfine-seg[label]'";
  }

  sel.addEventListener('change', () => {
    $('custom-row').hidden = sel.value !== CUSTOM;
    if (sel.value === CUSTOM) $('path').focus();
    else loadModel(sel.value);
  });
  $('load-path').addEventListener('click', () => $('path').value.trim() && loadModel($('path').value.trim()));
  $('path').addEventListener('keydown', (e) => { if (e.key === 'Enter') $('load-path').click(); });
  $('apply').addEventListener('click', () => S.model && loadModel(S.model.spec));
  $('weights').addEventListener('change', (e) => { if (e.target.files[0]) uploadWeights(e.target.files[0]); e.target.value = ''; });

  for (const t of $('backend').children) t.addEventListener('click', () => !t.disabled && setBackend(t.dataset.backend));
  $('prompt').addEventListener('change', () => { if (S.backend === 'sam3') setPill('ok', `SAM3 · ${$('prompt').value}`); runImage(); });
  $('conf').addEventListener('input', () => { $('conf-val').textContent = Number($('conf').value).toFixed(2); runImage(); });
  $('minimize').addEventListener('change', runImage);
  $('stride').addEventListener('input', () => { $('stride-val').textContent = $('stride').value; });

  for (const t of $('modes').children) t.addEventListener('click', () => { setMode(t.dataset.mode); runImage(); });

  $('file').addEventListener('change', (e) => openFile(e.target.files[0]));
  dropZone($('file-area'), openFile);
  $('file-new').addEventListener('click', clearFile);
  document.addEventListener('paste', (e) => {
    const f = [...(e.clipboardData?.files || [])].find((x) => x.type.startsWith('image/'));
    if (f) openFile(f);
  });
  bindWipe();

  $('cam-start').addEventListener('click', startCam);
  $('cam-stop').addEventListener('click', stopCam);

  $('vid-run').addEventListener('click', runVideo);
  $('vid-cancel').addEventListener('click', () => S.job && api(`/api/video/${S.job}/cancel`).catch((e) => toast(e.message)));

  // focus rings for keyboard users only, as on the showcase page
  const root = document.documentElement;
  addEventListener('pointerdown', () => { root.dataset.input = 'mouse'; });
  addEventListener('keydown', () => { root.dataset.input = 'keyboard'; });
}

init().catch((e) => { setPill('bad', 'server unreachable'); toast(e.message); });
