// Self-contained subset of KJNodes' utility.js — only the helpers
// the BAT Points Editor needs. Forked from
// ComfyUI-KJNodes/web/js/utility.js so the BAT node pack works
// without KJNodes installed.

const { app } = window.comfyAPI.app;

export function makeUUID() {
  let dt = new Date().getTime()
  const uuid = 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, (c) => {
    const r = ((dt + Math.random() * 16) % 16) | 0
    dt = Math.floor(dt / 16)
    return (c === 'x' ? r : (r & 0x3) | 0x8).toString(16)
  })
  return uuid
}

export function chainCallback(object, property, callback) {
  if (object == undefined) {
    console.error("Tried to add callback to non-existant object")
    return;
  }
  if (property in object) {
    const callback_orig = object[property]
    object[property] = function () {
      const r = callback_orig.apply(this, arguments);
      callback.apply(this, arguments);
      return r
    };
  } else {
    object[property] = callback;
  }
}

export function addMiddleClickPan(element) {
  const onMouseDown = (e) => {
    if (e.button !== 1) return;
    e.preventDefault();
    const ds = app.canvas?.ds;
    if (!ds) return;
    const startX = e.clientX, startY = e.clientY;
    const startOffsetX = ds.offset[0], startOffsetY = ds.offset[1];
    const onMove = (me) => {
      ds.offset[0] = startOffsetX + (me.clientX - startX);
      ds.offset[1] = startOffsetY + (me.clientY - startY);
      app.canvas.setDirty(true, true);
    };
    const onUp = () => {
      document.removeEventListener('mousemove', onMove);
      document.removeEventListener('mouseup', onUp);
    };
    document.addEventListener('mousemove', onMove);
    document.addEventListener('mouseup', onUp);
  };
  element.addEventListener('mousedown', onMouseDown);
  return () => element.removeEventListener('mousedown', onMouseDown);
}

export function resolveSourcePreview(node, inputSlot) {
  if (!node.graph) return null;
  const input = node.inputs?.[inputSlot];
  if (!input || input.link == null) return null;
  const link = node.graph.links?.get(input.link);
  if (!link) return null;
  const srcNode = node.graph.getNodeById(link.origin_id);
  if (!srcNode) return null;

  const vpWidget = srcNode.widgets?.find((w) => w.name === "videopreview");
  if (vpWidget?.videoEl?.src) {
    return { isVideo: true, videoEl: vpWidget.videoEl };
  }

  const w = srcNode.widgets?.find((w) => w.name === "image" || w.name === "video");
  if (w?.value) {
    let subfolder = "", fname = w.value;
    const lastSlash = fname.lastIndexOf("/");
    if (lastSlash >= 0) { subfolder = fname.substring(0, lastSlash); fname = fname.substring(lastSlash + 1); }
    const isVideo = w.name === "video";
    const url = `/view?filename=${encodeURIComponent(fname)}&type=input&subfolder=${encodeURIComponent(subfolder)}`;
    return { url, isVideo };
  }

  if (srcNode.imgs?.length > 0 && srcNode.imgs[0].src) {
    return { url: srcNode.imgs[0].src, isVideo: false };
  }
  return null;
}

export function watchImageInputs(node, inputName, onChange) {
  let watchedWidgets = [];

  function unwatchWidgets() {
    for (const { widget, origCb } of watchedWidgets) widget.callback = origCb;
    watchedWidgets = [];
  }

  function resolve() {
    if (!node.inputs) return [];
    const slots = inputName
      ? node.inputs.map((inp, i) => inp.name === inputName ? i : -1).filter(i => i >= 0)
      : node.inputs.map((inp, i) => inp.type === "IMAGE" ? i : -1).filter(i => i >= 0);
    return slots.map(i => resolveSourcePreview(node, i)).filter(s => s !== null);
  }

  function watch() {
    unwatchWidgets();
    if (!node.inputs || !node.graph) return;
    const slots = inputName
      ? node.inputs.map((inp, i) => inp.name === inputName ? i : -1).filter(i => i >= 0)
      : node.inputs.map((inp, i) => inp.type === "IMAGE" ? i : -1).filter(i => i >= 0);
    for (const slotIdx of slots) {
      const input = node.inputs[slotIdx];
      if (!input || input.link == null) continue;
      const link = node.graph.links?.get(input.link);
      if (!link) continue;
      const srcNode = node.graph.getNodeById(link.origin_id);
      if (!srcNode) continue;
      const w = srcNode.widgets?.find(w => w.name === "image" || w.name === "video");
      if (!w) continue;
      const origCb = w.callback;
      w.callback = function (...args) {
        if (origCb) origCb.apply(this, args);
        setTimeout(() => onChange(resolve()), 100);
      };
      watchedWidgets.push({ widget: w, origCb });
    }
  }

  chainCallback(node, "onConnectionsChange", function (type) {
    if (type != null && type !== 1) return;
    setTimeout(() => {
      watch();
      onChange(resolve());
    }, 100);
  });

  chainCallback(node, "onRemoved", unwatchWidgets);

  return unwatchWidgets;
}

export function captureVideoFrame(videoEl, callback) {
  const capture = () => {
    if (!videoEl.videoWidth || !videoEl.videoHeight) return;
    const c = document.createElement("canvas");
    c.width = videoEl.videoWidth;
    c.height = videoEl.videoHeight;
    c.getContext("2d").drawImage(videoEl, 0, 0);
    callback(c);
  };
  if (videoEl.readyState >= 2) {
    capture();
  } else {
    const onReady = () => { videoEl.removeEventListener("loadeddata", onReady); capture(); };
    videoEl.addEventListener("loadeddata", onReady);
  }
}

export function rectHitTest(mx, my, x1, y1, x2, y2, radius) {
  const hit = (cx, cy) => Math.abs(mx - cx) < radius && Math.abs(my - cy) < radius;
  if (hit(x1, y1)) return "resize-tl";
  if (hit(x2, y1)) return "resize-tr";
  if (hit(x1, y2)) return "resize-bl";
  if (hit(x2, y2)) return "resize-br";
  if (mx >= x1 && mx <= x2 && Math.abs(my - y1) < radius) return "resize-t";
  if (mx >= x1 && mx <= x2 && Math.abs(my - y2) < radius) return "resize-b";
  if (my >= y1 && my <= y2 && Math.abs(mx - x1) < radius) return "resize-l";
  if (my >= y1 && my <= y2 && Math.abs(mx - x2) < radius) return "resize-r";
  if (mx >= x1 && mx <= x2 && my >= y1 && my <= y2) return "move";
  return null;
}

export function cursorForBboxMode(mode) {
  if (mode === "move") return "move";
  if (mode === "resize-tl" || mode === "resize-br") return "nwse-resize";
  if (mode === "resize-tr" || mode === "resize-bl") return "nesw-resize";
  if (mode === "resize-t" || mode === "resize-b") return "ns-resize";
  if (mode === "resize-l" || mode === "resize-r") return "ew-resize";
  return null;
}
