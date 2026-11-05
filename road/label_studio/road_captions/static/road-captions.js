/* road-captions.js: the live sentence for the road review screen.
 *
 * nginx adds this file to every Label Studio page. It does nothing unless the labeling config
 * has the controls "decision" and "caption" and the view idAttr "road-sentence" (see config.xml).
 *
 * It reads the editor through window.Htx (the store Label Studio puts on window).
 * When the intern presses a key or types words, it asks /road-api/sentence for the sentence,
 * shows it under the keys, and writes it into the hidden "caption" field that is saved.
 * While a sentence is not ready, "caption" stays empty, and Label Studio does not allow submit.
 */
(function () {
  "use strict";
  if (window.__roadCaptions) return;
  window.__roadCaptions = { version: 1 };

  var API = "/road-api/sentence";
  var TICK_MS = 200;
  var TYPING_PAUSE_MS = 700;
  var PREFETCH = true; // ask for the Detector's sentence when a task opens, so key 1 is instant
  var NO_SENTENCE = { bad_box: 1, unsure: 1 };
  var SOURCE_TEXT = {
    bedrock: "Written by the model",
    template: "Template sentence",
    intern: "Your words, as typed",
    lookalike: "Fixed sentence for this look-alike",
    saved: "Saved sentence",
  };

  var seen = null; // { annKey, key } of the selected annotation
  var settle = null; // { annKey, until } while a saved annotation loads
  var pre = null; // { annKey, at } prefetch time for a new annotation
  var timer = null;
  var seq = 0; // only the answer to the newest request is used
  var view = { kind: "idle" };
  var prefetched = {};

  function ctl(ann, name) {
    try { return (ann.names && ann.names.get(name)) || null; } catch (e) { return null; }
  }
  function choice(ann, name) {
    var c = ctl(ann, name);
    if (!c || typeof c.selectedValues !== "function") return null;
    var v = c.selectedValues();
    return v && v.length ? String(v[0]) : null;
  }
  function texts(ann, name) {
    var c = ctl(ann, name);
    return c && c.regions ? c.regions.map(function (r) { return r._value; }).filter(Boolean) : [];
  }
  function wordsOf(ann) {
    var c = ctl(ann, "words");
    if (!c) return "";
    var parts = (c.regions || []).map(function (r) { return r._value; });
    if (c._value) parts.push(c._value);
    return parts.join(", ").replace(/\s+/g, " ").trim();
  }
  function isReadOnly(ann) {
    try { return typeof ann.isReadOnly === "function" ? ann.isReadOnly() : !ann.editable; } catch (e) { return true; }
  }
  function isSaved(ann) { // submitted before: never rewrite it just because someone opened it
    try { return !!(ann.exists || ann.pk); } catch (e) { return false; }
  }
  function setText(ann, name, text) {
    var c = ctl(ann, name);
    if (!c) return;
    var regions = c.regions || [];
    if (text && regions.length === 1) {
      if (regions[0]._value !== text) regions[0].setValue(text);
      return;
    }
    if (regions.length) {
      // Clear the text and delete its result area, like the editor does for an empty result.
      // (TextArea.remove() and onChange() make Label Studio 1.23 touch the deleted area: console error.)
      var area = c.result && c.result.area;
      c.updateFromResult(null);
      if (area) ann.deleteArea(area);
    }
    if (text) c.addText(text);
  }

  function readState(store) {
    var ann = store.annotationStore && store.annotationStore.selected;
    if (!ann || !ctl(ann, "decision") || !ctl(ann, "caption")) return null;
    var taskId = store.task && store.task.id;
    return {
      store: store, ann: ann, taskId: taskId, annKey: taskId + ":" + ann.id,
      decision: choice(ann, "decision"), lookalike: choice(ann, "lookalike"), words: wordsOf(ann),
      readOnly: isReadOnly(ann), saved: isSaved(ann),
    };
  }
  function keyOf(s) { return [s.decision || "", s.lookalike || "", s.words].join("|"); }
  function stillSelected(s) {
    var store = window.Htx;
    return store === s.store && store.annotationStore && store.annotationStore.selected === s.ann;
  }

  function post(body) {
    return fetch(API, {
      method: "POST", credentials: "same-origin",
      headers: { "Content-Type": "application/json", "X-Road-Captions": "1" },
      body: JSON.stringify(body),
    });
  }

  function prefetch(taskId) {
    post({ task_id: taskId, decision: "Correct", lookalike: null, words: "", prefetch: true }).catch(function () {});
  }

  function changed(s, typing) {
    clearTimeout(timer);
    timer = null;
    var my = ++seq;
    setText(s.ann, "caption", "");
    setText(s.ann, "caption_source", "");
    if (!s.decision) return show({ kind: "idle" });
    if (NO_SENTENCE[s.decision]) return show({ kind: "none" });
    if (s.decision === "not_a_defect" && !s.lookalike) {
      return show({ kind: "hint", text: "Press q-u to say what it is." });
    }
    var body = { task_id: s.taskId, decision: s.decision, lookalike: s.lookalike, words: s.words };
    if (typing && s.words) {
      show({ kind: "typing" });
      timer = setTimeout(function () { ask(s, body, my); }, TYPING_PAUSE_MS);
    } else {
      ask(s, body, my);
    }
  }

  function ask(s, body, my) {
    show({ kind: "busy" });
    post(body)
      .then(function (r) {
        return r.json().catch(function () { return {}; }).then(function (j) {
          if (r.ok) return j;
          return { ok: false, message: j.detail || ("The sentence service answered HTTP " + r.status + "."), retry: true };
        });
      })
      .catch(function () {
        return { ok: false, message: "The sentence service did not answer.", retry: true };
      })
      .then(function (res) {
        if (my !== seq || !stillSelected(s)) return; // the intern moved on
        if (res.ok && res.sentence) {
          setText(s.ann, "caption", res.sentence);
          setText(s.ann, "caption_source", res.source || "");
          show({ kind: "ok", sentence: res.sentence, source: res.source, note: res.message });
        } else if (res.ok) {
          show({ kind: "none" });
        } else {
          show({ kind: "error", text: res.message || "No sentence.", retry: !!res.retry });
        }
      });
  }

  function retry() {
    var store = window.Htx;
    var s = store && readState(store);
    if (s && !s.readOnly) changed(s, false);
  }

  // ---- panel -----------------------------------------------------------------------------
  function show(v) { view = v; render(); }

  function addStyle() {
    if (document.getElementById("rc-style")) return;
    var css =
      ".rc-panel{margin:8px 0 6px;padding:10px 12px;border:1px solid #c9ced6;border-radius:6px;" +
      "background:#f7f8fa;color:#1f2329;font:14px/1.45 system-ui,sans-serif;max-width:720px}" +
      ".rc-panel .rc-head{font-size:12px;color:#5b6472;margin-bottom:2px}" +
      ".rc-panel .rc-text{font-size:16px}" +
      ".rc-panel.rc-ok{border-color:#2e9e62;background:#eef8f2}" +
      ".rc-panel.rc-error{border-color:#d64545;background:#fdf1f1}" +
      ".rc-panel.rc-busy .rc-text,.rc-panel.rc-idle .rc-text{color:#5b6472}" +
      ".rc-panel .rc-note{font-size:12px;color:#5b6472;margin-top:4px}" +
      ".rc-panel button{margin-top:6px;font:inherit;font-size:13px;padding:2px 10px;cursor:pointer}";
    var el = document.createElement("style");
    el.id = "rc-style";
    el.textContent = css;
    document.head.appendChild(el);
  }

  function render() {
    var host = document.getElementById("road-sentence");
    if (!host) return;
    addStyle();
    var box = host.querySelector(".rc-panel");
    if (!box) {
      box = document.createElement("div");
      box.setAttribute("aria-live", "polite");
      host.appendChild(box);
    }
    var head = "Sentence", text = "", note = "", cls = "rc-idle", retryBtn = false;
    if (view.kind === "idle") text = "Press a key: 1 correct, 2-4 label, 5 not a defect, 6 bad box, 7 unsure.";
    else if (view.kind === "none") text = "No sentence is needed for this key.";
    else if (view.kind === "hint") text = view.text;
    else if (view.kind === "typing") { cls = "rc-busy"; text = "Waiting until you stop typing..."; }
    else if (view.kind === "busy") { cls = "rc-busy"; text = "Writing the sentence..."; }
    else if (view.kind === "ok") {
      cls = "rc-ok"; text = view.sentence;
      head = "Sentence - " + (SOURCE_TEXT[view.source] || view.source || "saved");
      note = view.note || "";
    } else if (view.kind === "error") {
      cls = "rc-error"; head = "No sentence yet"; text = view.text;
      note = "Submit stays blocked until a sentence is ready. Press 7 (unsure) to skip the sentence.";
      retryBtn = !!view.retry;
    }
    var stamp = [cls, head, text, note, retryBtn].join("\u0001");
    if (box.getAttribute("data-stamp") === stamp) return;
    box.setAttribute("data-stamp", stamp);
    box.className = "rc-panel " + cls;
    box.textContent = "";
    var h = document.createElement("div"); h.className = "rc-head"; h.textContent = head; box.appendChild(h);
    var t = document.createElement("div"); t.className = "rc-text"; t.textContent = text; box.appendChild(t);
    if (note) { var n = document.createElement("div"); n.className = "rc-note"; n.textContent = note; box.appendChild(n); }
    if (retryBtn) {
      var b = document.createElement("button");
      b.type = "button"; b.textContent = "Try again";
      b.addEventListener("click", retry);
      box.appendChild(b);
    }
  }

  // ---- main loop -------------------------------------------------------------------------
  function tick() {
    var store = window.Htx;
    if (!store) return;
    var s = readState(store);
    if (!s) return;
    if (PREFETCH && s.taskId && !prefetched[s.taskId] && !s.readOnly && !s.saved && !s.decision) {
      // Wait 600 ms: when a submitted task opens, the editor shows a blank annotation for a moment.
      if (!pre || pre.annKey !== s.annKey) pre = { annKey: s.annKey, at: Date.now() + 600 };
      else if (Date.now() >= pre.at) { prefetched[s.taskId] = 1; prefetch(s.taskId); }
    }
    var key = keyOf(s);
    if (!seen || seen.annKey !== s.annKey) { // first look at this annotation: keep what it has
      if (s.saved && !s.decision) { // its saved answers may still be loading: wait up to 1.5 s
        if (!settle || settle.annKey !== s.annKey) settle = { annKey: s.annKey, until: Date.now() + 1500 };
        if (Date.now() < settle.until) return;
      }
      seen = { annKey: s.annKey, key: key };
      clearTimeout(timer);
      seq++;
      var saved = texts(s.ann, "caption")[0];
      if (saved) show({ kind: "ok", sentence: saved, source: texts(s.ann, "caption_source")[0] || "saved" });
      else if (!s.decision) show({ kind: "idle" });
      else if (NO_SENTENCE[s.decision]) show({ kind: "none" });
      else if (s.saved || s.readOnly) show({ kind: "hint", text: "No saved sentence. Press the key again to write one." });
      else changed(s, false); // a draft with a key already pressed
      return;
    }
    render(); // the editor may have drawn a new #road-sentence
    if (seen.key === key) {
      // Undo can remove the saved sentence while the panel still shows it: write it again.
      if (view.kind === "ok" && !s.readOnly && texts(s.ann, "caption")[0] !== view.sentence) {
        setText(s.ann, "caption", view.sentence);
        setText(s.ann, "caption_source", view.source || "");
      }
      return;
    }
    var onlyWords = seen.key.split("|").slice(0, 2).join("|") === key.split("|").slice(0, 2).join("|");
    seen.key = key;
    if (!s.readOnly) changed(s, onlyWords);
  }

  setInterval(function () {
    try { tick(); } catch (e) { if (window.console) console.warn("road-captions:", e); }
  }, TICK_MS);
})();
