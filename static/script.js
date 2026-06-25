// ── State ──────────────────────────────────────────────────
let debounceTimer = null;
let lastTranslated = "";

// ── View switching ──────────────────────────────────────────
function showView(name, navEl) {
  document.querySelectorAll(".view").forEach(v => v.classList.remove("active"));
  document.querySelectorAll(".nav-item").forEach(n => n.classList.remove("active"));
  document.getElementById("view-" + name).classList.add("active");
  if (navEl) navEl.classList.add("active");
  if (name === "history") loadHistory();
}

// ── Keyboard shortcuts ──────────────────────────────────────
document.addEventListener("keydown", function(e) {
  // Enter or Ctrl+Enter → translate (only when textarea is focused)
  if (document.activeElement === document.getElementById("input-text")) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      runTranslate();
      return;
    }
    if (e.key === "Enter" && e.ctrlKey) {
      e.preventDefault();
      runTranslate();
      return;
    }
  }
  // Escape → clear
  if (e.key === "Escape") {
    clearAll();
    return;
  }
  // Ctrl+C when output is visible → copy translation
  if ((e.ctrlKey || e.metaKey) && e.key === "c") {
    const copyBtn = document.getElementById("copy-btn");
    if (copyBtn.style.display !== "none") {
      const selection = window.getSelection().toString();
      if (!selection) {
        e.preventDefault();
        copyResult();
      }
    }
  }
});

// ── Char counter + debounce auto-translate ──────────────────
document.getElementById("input-text").addEventListener("input", function() {
  const size = this.value.length;
  const counter = document.getElementById("char-count");
  counter.textContent = size + " / 200";
  counter.className = "char-count" + (size > 200 ? " over" : "");

  // hide warning on new input
  document.getElementById("lang-warning").style.display = "none";

  // debounce: auto-translate after 1s of no typing
  clearTimeout(debounceTimer);
  if (this.value.trim().length > 3) {
    debounceTimer = setTimeout(() => {
      runTranslate(true); // true = silent (don't show loading for debounce)
    }, 1000);
  }
});

// ── Input helpers ───────────────────────────────────────────
function cleanInput(text) {
  text = text.toLowerCase().trim();
  text = text.replace(/([^\s])([?.!,])/g, "$1 $2");
  text = text.replace(/  +/g, " ");
  if (text.length > 0 && ![".", "?", "!"].includes(text[text.length - 1])) {
    text = text + " .";
  }
  return text;
}

function isFrench(text) {
  const frenchWords = ["je", "tu", "il", "elle", "nous", "vous",
    "ils", "les", "des", "une", "est", "sont",
    "avec", "pour", "dans", "que", "qui"];
  const words = text.toLowerCase().split(" ");
  const matches = words.filter(w => frenchWords.includes(w));
  return matches.length >= 2;
}

async function pasteText() {
  try {
    const text = await navigator.clipboard.readText();
    document.getElementById("input-text").value = text;
    document.getElementById("input-text").dispatchEvent(new Event("input"));
  } catch (e) {
    document.getElementById("input-text").focus();
  }
}

// ── Main translate ──────────────────────────────────────────
async function runTranslate(silent = false) {
  const raw = document.getElementById("input-text").value;
  if (!raw.trim()) return;

  const text = cleanInput(raw);

  // French detection
  if (isFrench(text)) {
    document.getElementById("lang-warning").style.display = "block";
    return;
  }

  // Don't re-translate identical input
  if (text === lastTranslated) return;

  const output   = document.getElementById("output");
  const loading  = document.getElementById("loading");
  const copyBtn  = document.getElementById("copy-btn");
  const dot      = document.getElementById("divider-dot");
  const candBar  = document.getElementById("candidates-bar");

  if (!silent) {
    loading.style.display = "flex";
    output.style.display = "none";
    copyBtn.style.display = "none";
    candBar.style.display = "none";
    dot.classList.add("translating");
  }

  try {
    // Fetch best translation + confidence
    const [res1, res2] = await Promise.all([
      fetch("/translate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text })
      }),
      fetch("/translate/candidates", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text })
      })
    ]);

    const data      = await res1.json();
    const candData  = await res2.json();

    loading.style.display = "none";
    output.style.display = "block";
    dot.classList.remove("translating");

    output.textContent = data.translation;
    copyBtn.style.display = "flex";
    document.getElementById("hint").textContent = "";
    lastTranslated = text;

    // Confidence badge
    const badge = document.getElementById("conf-badge");
    const conf  = data.confidence;
    badge.textContent = conf + "% confidence";
    badge.className   = "conf-badge " + (conf >= 80 ? "high" : conf >= 50 ? "mid" : "low");
    badge.style.display = "block";

    // Candidates
    const candList = document.getElementById("candidate-list");
    candList.innerHTML = "";
    candData.candidates.forEach((c, i) => {
      const item = document.createElement("div");
      item.className = "candidate-item";
      item.innerHTML = `<span>${c}</span><span class="candidate-rank">#${i + 1}</span>`;
      item.title = "Double-click to copy";
      item.onclick = () => selectCandidate(c);
      item.ondblclick = () => {
        navigator.clipboard.writeText(c);
        item.style.background = "#dcfce7";
        setTimeout(() => item.style.background = "", 800);
      };
      candList.appendChild(item);
    });
    candBar.style.display = "block";

  } catch (e) {
    loading.style.display = "none";
    dot.classList.remove("translating");
    output.style.display = "block";
    output.innerHTML = '<span style="color:#ef4444">Could not reach the server. Is the Flask app running?</span>';
  }
}

// ── Output actions ──────────────────────────────────────────
function selectCandidate(text) {
  const output = document.getElementById("output");
  output.textContent = text;
  output.style.background = "#f0fdf4";
  setTimeout(() => output.style.background = "", 600);
}

function copyResult() {
  const text = document.getElementById("output").textContent;
  if (!text || text === "Translation appears here") return;
  navigator.clipboard.writeText(text);
  const btn = document.getElementById("copy-btn");
  btn.innerHTML = '<i class="ti ti-check" style="color:var(--accent)"></i>';
  setTimeout(() => { btn.innerHTML = '<i class="ti ti-copy"></i>'; }, 1500);
}

function clearAll() {
  document.getElementById("input-text").value = "";
  document.getElementById("output").innerHTML = '<span class="output-placeholder">Translation appears here</span>';
  document.getElementById("char-count").textContent = "0 / 200";
  document.getElementById("char-count").className = "char-count";
  document.getElementById("copy-btn").style.display = "none";
  document.getElementById("conf-badge").style.display = "none";
  document.getElementById("candidates-bar").style.display = "none";
  document.getElementById("lang-warning").style.display = "none";
  document.getElementById("hint").innerHTML = "Tip: punctuate your input — <em>hello , how are you ?</em>";
  document.getElementById("divider-dot").classList.remove("translating");
  lastTranslated = "";
  clearTimeout(debounceTimer);
  document.getElementById("input-text").focus();
}

// ── History ─────────────────────────────────────────────────
async function loadHistory() {
  const container = document.getElementById("history-content");
  container.innerHTML = '<div style="color:#9ca3af;padding:20px;font-size:14px">Loading…</div>';

  try {
    const res  = await fetch("/history");
    const data = await res.json();

    if (!data.history || data.history.length === 0) {
      container.innerHTML = `
        <div class="empty-state">
          <i class="ti ti-history"></i>
          No translations yet. Start translating to see your history here.
        </div>`;
      return;
    }

    let html = `<table class="history-table">
      <thead>
        <tr>
          <th>English</th>
          <th>French</th>
          <th>Confidence</th>
          <th>Time</th>
        </tr>
      </thead>
      <tbody>`;

    data.history.forEach(r => {
      const cls = r.confidence >= 80 ? "high" : r.confidence >= 50 ? "mid" : "low";
      html += `<tr>
        <td>${r.english}</td>
        <td>${r.french}</td>
        <td><span class="conf-pill ${cls}">${r.confidence}%</span></td>
        <td>${r.timestamp}</td>
      </tr>`;
    });

    html += "</tbody></table>";
    container.innerHTML = html;

  } catch (e) {
    container.innerHTML = '<div style="color:#ef4444;padding:20px;font-size:14px">Failed to load history.</div>';
  }
}