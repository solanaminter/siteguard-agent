/* SiteGuard demo page: tabs + data-driven explorer panels. */
(function () {
  "use strict";

  /* ---------- tabs ---------- */
  var tabs = document.querySelectorAll(".tab");
  tabs.forEach(function (t) {
    t.addEventListener("click", function () {
      tabs.forEach(function (x) { x.classList.remove("active"); });
      t.classList.add("active");
      document.querySelectorAll(".panel").forEach(function (p) {
        p.classList.toggle("active", p.id === "panel-" + t.dataset.tab);
      });
    });
  });

  /* ---------- detections gallery (real pipeline frames) ---------- */
  var frames = [
    { src: "assets/frames/escalation.jpg", title: "Compliant crew",
      cap: "CAM 01: two workers with yellow hardhats correctly classified (0.95 / 0.94, green boxes). The restricted yard-corner zone is drawn top-right. No incident — the quiet majority of frames." },
    { src: "assets/frames/tracking.jpg", title: "Awaiting human review",
      cap: "INC-0001 held for approval: motion + no hardhat on track #2 (0.95). Medium risk goes to a person — the agent asks instead of acting." },
    { src: "assets/frames/zone.jpg", title: "Vision clears the worker",
      cap: "Same track #2 now reads HARDHAT. This later evidence is what withdraws INC-0001's pending request — the decision is reopened, not frozen." },
    { src: "assets/frames/approval.jpg", title: "Escalated: zone + no hardhat",
      cap: "CAM 03: track #4 inside the restricted excavation zone without a hardhat (0.94). Combined risk hits 90 — the safety alert is dispatched automatically." },
    { src: "assets/frames/withdrawn.jpg", title: "Persistent violation",
      cap: "CAM 04: track #5 flagged NO HARDHAT (0.95) while an escalated alert stands. The banner stays until the violation clears or a human resolves it." }
  ];
  var grid = document.getElementById("frame-grid");
  frames.forEach(function (f) {
    var card = document.createElement("div");
    card.className = "frame-card";
    var img = document.createElement("img");
    img.src = f.src; img.alt = f.title; img.loading = "lazy";
    var cap = document.createElement("div");
    cap.className = "cap";
    var b = document.createElement("b"); b.textContent = f.title;
    var s = document.createElement("span"); s.textContent = f.cap;
    cap.appendChild(b); cap.appendChild(s);
    card.appendChild(img); card.appendChild(cap);
    grid.appendChild(card);
  });

  /* ---------- data helpers ---------- */
  function get(path) {
    return fetch(path).then(function (r) {
      if (!r.ok) throw new Error(path + " " + r.status);
      return r.json();
    });
  }
  function setStat(name, val) {
    var el = document.querySelector('[data-stat="' + name + '"]');
    if (el) el.textContent = val;
  }
  function metricRow(dl, label, val) {
    var dt = document.createElement("dt"); dt.textContent = label;
    var dd = document.createElement("dd"); dd.textContent = val;
    dl.appendChild(dt); dl.appendChild(dd);
  }

  /* ---------- agent trace: incident cards ---------- */
  function compressHistory(hist) {
    // Collapse long runs of identical decisions: keep first, last, and changes.
    var out = [];
    hist.forEach(function (h, i) {
      var prev = hist[i - 1], next = hist[i + 1];
      var change = !prev || prev.decision !== h.decision ||
                   !next || next.decision !== h.decision;
      if (change) out.push(h);
    });
    return out;
  }
  get("data/demo_summary.json").then(function (s) {
    setStat("incidents", s.incidents.length);
    setStat("trace", s.trace_events);
    var wrap = document.getElementById("incident-grid");
    s.incidents.forEach(function (inc) {
      var card = document.createElement("div");
      card.className = "incident";
      var badge = document.createElement("span");
      badge.className = "badge " + inc.status;
      badge.textContent = inc.status;
      var h = document.createElement("h3");
      h.textContent = inc.id + " · risk " + inc.severity;
      var kinds = document.createElement("div");
      kinds.className = "kinds";
      kinds.textContent = "signals: " + inc.kinds.join(", ") +
        " · frames " + inc.created_frame + " → " +
        (inc.resolved_frame === null || inc.resolved_frame === undefined ? "open" : inc.resolved_frame) +
        (inc.reviewer ? " · reviewed by " + inc.reviewer : "");
      var ul = document.createElement("ul");
      ul.className = "history";
      compressHistory(inc.decision_history).forEach(function (d) {
        var li = document.createElement("li");
        var b = document.createElement("b");
        b.textContent = "f" + d.frame + " · " + d.decision.replace(/_/g, " ");
        li.appendChild(b);
        li.appendChild(document.createTextNode(" — " + d.note));
        ul.appendChild(li);
      });
      card.appendChild(badge); card.appendChild(h);
      card.appendChild(kinds); card.appendChild(ul);
      wrap.appendChild(card);
    });
  }).catch(function (e) { console.warn(e); });

  /* ---------- evaluation metrics ---------- */
  get("data/eval_results.json").then(function (d) {
    var s = d.summary || d;
    var dl = document.getElementById("metrics-curated");
    metricRow(dl, "Detection F1", s.detection.f1);
    metricRow(dl, "Detection P / R",
      s.detection.precision + " / " + s.detection.recall);
    metricRow(dl, "Hardhat F1", s.hardhat_classification.f1);
    metricRow(dl, "Hardhat accuracy", s.hardhat_classification.accuracy);
    metricRow(dl, "Zone accuracy", s.zone_accuracy);
    metricRow(dl, "Latency mean / p95",
      s.latency_ms.mean + " / " + s.latency_ms.p95 + " ms");
    setStat("detf1", s.detection.f1);
  }).catch(function (e) { console.warn(e); });

  get("data/eval_hf_results.json").then(function (d) {
    var s = d.summary || d;
    var dl = document.getElementById("metrics-hf");
    metricRow(dl, "Hardhat F1", s.hardhat_classification.f1);
    metricRow(dl, "Hardhat P / R",
      s.hardhat_classification.precision + " / " + s.hardhat_classification.recall);
    metricRow(dl, "Hardhat accuracy", s.hardhat_classification.accuracy);
    metricRow(dl, "Head coverage", s.head_coverage_recall +
      " (" + s.heads_covered + "/" + s.heads_total + ")");
    metricRow(dl, "Images", s.n_images);
    metricRow(dl, "Latency mean / p95",
      s.latency_ms.mean + " / " + s.latency_ms.p95 + " ms");
    setStat("hatacc", s.hardhat_classification.accuracy);
  }).catch(function () {
    var dl = document.getElementById("metrics-hf");
    dl.innerHTML = "<dt>status</dt><dd style='font-size:1rem'>benchmark running — refresh after the GPU pass</dd>";
    setStat("hatacc", "…");
  });
})();
