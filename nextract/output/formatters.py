from __future__ import annotations

import csv
import html
import json
from io import StringIO
from typing import Any

from nextract.core import BaseFormatter, ExtractionResult


def _json_default(obj: Any) -> Any:
    """Default JSON serializer for non-standard types."""
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "__float__"):
        return float(obj)
    return str(obj)


class JsonFormatter(BaseFormatter):
    """Format extraction results as JSON."""

    def format(self, result: ExtractionResult, **kwargs: Any) -> str:
        indent = kwargs.get("indent", 2)
        return json.dumps(result.data, ensure_ascii=False, indent=indent, default=_json_default)


class MarkdownFormatter(BaseFormatter):
    """Format extraction results as Markdown."""

    def format(self, result: ExtractionResult, **kwargs: Any) -> str:
        payload = json.dumps(result.data, ensure_ascii=False, indent=2, default=_json_default)
        return "\n".join(
            [
                "# Extraction Result",
                "",
                "```json",
                payload,
                "```",
            ]
        )


class HtmlFormatter(BaseFormatter):
    """Format extraction results as HTML."""

    def format(self, result: ExtractionResult, **kwargs: Any) -> str:
        theme = str(kwargs.get("theme") or "system").lower()
        if theme not in {"light", "dark", "system"}:
            theme = "system"

        payload = json.dumps(result.data, ensure_ascii=False, indent=2, default=_json_default)
        payload_html = html.escape(payload)
        payload_script = (
            json.dumps(result.data, ensure_ascii=False, indent=2, default=_json_default)
            .replace("&", "\\u0026")
            .replace("<", "\\u003c")
            .replace(">", "\\u003e")
            .replace("\u2028", "\\u2028")
            .replace("\u2029", "\\u2029")
        )

        template = """<!doctype html>
<html lang="en" data-theme="__NEXTRACT_THEME__">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="color-scheme" content="light dark">
  <title>Nextract Extraction</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,300..700&family=IBM+Plex+Sans:wght@300;400;500;600&family=IBM+Plex+Mono:wght@300;400;500&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg: #f4efe6;
      --bg-2: #fbf6ee;
      --ink: #1b1916;
      --muted: #5d564f;
      --accent: #c64f2a;
      --accent-2: #0f6f6a;
      --panel: #fffaf2;
      --panel-border: #ded5c7;
      --code-bg: #f8f1e6;
      --shadow: rgba(21, 18, 14, 0.16);
      --grid: rgba(0, 0, 0, 0.06);
      --glow: rgba(198, 79, 42, 0.18);
      color-scheme: light;
    }

    html[data-theme="dark"] {
      --bg: #0c1112;
      --bg-2: #121819;
      --ink: #f1e9dd;
      --muted: #b3a89a;
      --accent: #f2a25c;
      --accent-2: #6cc7c2;
      --panel: #141c1d;
      --panel-border: #283335;
      --code-bg: #0f1516;
      --shadow: rgba(0, 0, 0, 0.5);
      --grid: rgba(255, 255, 255, 0.06);
      --glow: rgba(242, 162, 92, 0.2);
      color-scheme: dark;
    }

    html[data-theme="system"] {
      color-scheme: light dark;
    }

    @media (prefers-color-scheme: dark) {
      html[data-theme="system"] {
        --bg: #0c1112;
        --bg-2: #121819;
        --ink: #f1e9dd;
        --muted: #b3a89a;
        --accent: #f2a25c;
        --accent-2: #6cc7c2;
        --panel: #141c1d;
        --panel-border: #283335;
        --code-bg: #0f1516;
        --shadow: rgba(0, 0, 0, 0.5);
        --grid: rgba(255, 255, 255, 0.06);
        --glow: rgba(242, 162, 92, 0.2);
      }
    }

    * {
      box-sizing: border-box;
    }

    body {
      margin: 0;
      min-height: 100vh;
      background: var(--bg);
      color: var(--ink);
      font-family: "IBM Plex Sans", "Gill Sans", "Trebuchet MS", sans-serif;
      letter-spacing: 0.01em;
    }

    body::before {
      content: "";
      position: fixed;
      inset: 0;
      background:
        radial-gradient(1200px 700px at 10% -10%, var(--glow), transparent 60%),
        radial-gradient(900px 600px at 90% 10%, rgba(15, 111, 106, 0.12), transparent 60%),
        repeating-linear-gradient(90deg, transparent 0 7px, var(--grid) 7px 8px),
        repeating-linear-gradient(180deg, transparent 0 7px, var(--grid) 7px 8px);
      pointer-events: none;
      z-index: -1;
    }

    .page {
      max-width: 1200px;
      margin: 0 auto;
      padding: 48px 24px 96px;
      display: grid;
      gap: 32px;
    }

    .hero {
      display: grid;
      gap: 24px;
      padding: 28px;
      border: 1px solid var(--panel-border);
      background: var(--panel);
      box-shadow: 0 28px 60px var(--shadow);
      border-radius: 20px;
      animation: rise 600ms ease-out both;
    }

    .hero-top {
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
    }

    .brand {
      display: grid;
      gap: 6px;
    }

    .brand-title {
      font-family: "Fraunces", "Bookman Old Style", "Palatino", serif;
      font-size: clamp(2rem, 4vw, 3.5rem);
      font-weight: 600;
      letter-spacing: 0.02em;
      margin: 0;
    }

    .brand-subtitle {
      font-size: 0.95rem;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.2em;
    }

    .theme-toggle {
      display: inline-flex;
      border: 1px solid var(--panel-border);
      border-radius: 999px;
      overflow: hidden;
      background: var(--bg-2);
    }

    .theme-toggle button {
      border: 0;
      background: transparent;
      padding: 8px 14px;
      font-size: 0.8rem;
      text-transform: uppercase;
      letter-spacing: 0.16em;
      color: var(--muted);
      cursor: pointer;
    }

    .theme-toggle button[aria-pressed="true"] {
      background: var(--ink);
      color: var(--bg-2);
    }

    .meta-grid {
      display: grid;
      gap: 16px;
      grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
    }

    .stat {
      padding: 16px;
      border: 1px solid var(--panel-border);
      border-radius: 14px;
      background: var(--bg-2);
    }

    .stat-label {
      font-size: 0.7rem;
      text-transform: uppercase;
      letter-spacing: 0.18em;
      color: var(--muted);
    }

    .stat-value {
      font-size: 1.4rem;
      font-weight: 600;
      margin-top: 8px;
      font-family: "Fraunces", "Bookman Old Style", serif;
    }

    .layout {
      display: grid;
      gap: 24px;
      grid-template-columns: minmax(0, 2fr) minmax(0, 1fr);
    }

    .panel {
      border: 1px solid var(--panel-border);
      background: var(--panel);
      border-radius: 18px;
      overflow: hidden;
      box-shadow: 0 24px 50px var(--shadow);
      animation: rise 700ms ease-out both;
    }

    .panel-head {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      padding: 16px 20px;
      border-bottom: 1px solid var(--panel-border);
      background: var(--bg-2);
      font-size: 0.85rem;
      text-transform: uppercase;
      letter-spacing: 0.16em;
    }

    .panel-actions {
      display: inline-flex;
      gap: 8px;
    }

    .panel-actions button {
      border: 1px solid var(--panel-border);
      background: var(--panel);
      color: var(--ink);
      padding: 6px 10px;
      border-radius: 999px;
      font-size: 0.7rem;
      text-transform: uppercase;
      letter-spacing: 0.12em;
      cursor: pointer;
    }

    .panel-actions button:hover {
      border-color: var(--accent);
      color: var(--accent);
    }

    .code {
      margin: 0;
      padding: 24px;
      background: var(--code-bg);
      font-family: "IBM Plex Mono", "Inconsolata", "Menlo", monospace;
      font-size: 0.9rem;
      line-height: 1.6;
      overflow: auto;
      max-height: 70vh;
      white-space: pre;
    }

    .code.wrap {
      white-space: pre-wrap;
      word-break: break-word;
    }

    .side {
      display: grid;
      gap: 16px;
      padding: 20px;
    }

    .side h2 {
      font-family: "Fraunces", "Bookman Old Style", serif;
      margin: 0 0 8px;
      font-size: 1.4rem;
    }

    .side p {
      margin: 0;
      color: var(--muted);
      line-height: 1.6;
    }

    .badge-row {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
    }

    .badge {
      padding: 6px 12px;
      border-radius: 999px;
      border: 1px solid var(--panel-border);
      background: var(--bg-2);
      font-size: 0.7rem;
      text-transform: uppercase;
      letter-spacing: 0.18em;
    }

    .list {
      display: grid;
      gap: 10px;
      margin-top: 10px;
    }

    .list-item {
      padding: 12px;
      border-radius: 12px;
      border: 1px dashed var(--panel-border);
      font-size: 0.85rem;
      color: var(--muted);
      background: rgba(255, 255, 255, 0.06);
    }

    @keyframes rise {
      from {
        opacity: 0;
        transform: translateY(10px);
      }
      to {
        opacity: 1;
        transform: translateY(0);
      }
    }

    @media (max-width: 960px) {
      .layout {
        grid-template-columns: 1fr;
      }
    }

    @media (prefers-reduced-motion: reduce) {
      * {
        animation: none !important;
        transition: none !important;
      }
    }
  </style>
</head>
<body>
  <div class="page">
    <header class="hero">
      <div class="hero-top">
        <div class="brand">
          <h1 class="brand-title">Nextract</h1>
          <div class="brand-subtitle">Extraction dossier</div>
        </div>
        <div class="theme-toggle" role="group" aria-label="Theme">
          <button type="button" data-theme-value="light" aria-pressed="false">Light</button>
          <button type="button" data-theme-value="dark" aria-pressed="false">Dark</button>
          <button type="button" data-theme-value="system" aria-pressed="true">System</button>
        </div>
      </div>
      <div class="meta-grid">
        <div class="stat">
          <div class="stat-label">Type</div>
          <div class="stat-value" data-stat="type">-</div>
        </div>
        <div class="stat">
          <div class="stat-label">Top Level</div>
          <div class="stat-value" data-stat="top">-</div>
        </div>
        <div class="stat">
          <div class="stat-label">Lines</div>
          <div class="stat-value" data-stat="lines">-</div>
        </div>
        <div class="stat">
          <div class="stat-label">Size</div>
          <div class="stat-value" data-stat="bytes">-</div>
        </div>
      </div>
    </header>

    <main class="layout">
      <section class="panel">
        <div class="panel-head">
          <div>JSON Payload</div>
          <div class="panel-actions">
            <button type="button" id="copy-btn">Copy JSON</button>
            <button type="button" id="wrap-btn">Wrap</button>
          </div>
        </div>
        <pre id="json" class="code">__NEXTRACT_PAYLOAD_HTML__</pre>
      </section>

      <aside class="panel">
        <div class="side">
          <h2>Snapshot</h2>
          <p>A compact view of the extracted payload with layout tuned for quick verification and handoff.</p>
          <div class="badge-row">
            <div class="badge">format: json</div>
            <div class="badge" data-stat="generated">generated: -</div>
          </div>
          <div class="list">
            <div class="list-item">Use the theme switcher to match your environment or review on light paper.</div>
            <div class="list-item">Copy or wrap the payload before pasting into downstream systems.</div>
            <div class="list-item">Large outputs scroll in the left panel with preserved indentation.</div>
          </div>
        </div>
      </aside>
    </main>
  </div>

  <script type="application/json" id="payload">__NEXTRACT_PAYLOAD_JSON__</script>
  <script>
    (() => {
      const storageKey = "nextract-theme";
      const root = document.documentElement;
      const buttons = document.querySelectorAll("[data-theme-value]");
      const payloadEl = document.getElementById("payload");
      const codeEl = document.getElementById("json");
      const copyBtn = document.getElementById("copy-btn");
      const wrapBtn = document.getElementById("wrap-btn");

      const setPressed = (value) => {
        buttons.forEach((btn) => {
          const active = btn.getAttribute("data-theme-value") === value;
          btn.setAttribute("aria-pressed", String(active));
        });
      };

      const applyTheme = (value) => {
        root.dataset.theme = value;
        setPressed(value);
      };

      const saved = window.localStorage.getItem(storageKey);
      const initial = root.dataset.theme || "system";
      applyTheme(saved || initial);

      buttons.forEach((btn) => {
        btn.addEventListener("click", () => {
          const value = btn.getAttribute("data-theme-value") || "system";
          window.localStorage.setItem(storageKey, value);
          applyTheme(value);
        });
      });

      const payloadText = payloadEl ? payloadEl.textContent || "" : "";
      const lines = payloadText ? payloadText.split("\n").length : 0;
      const bytes = payloadText ? new TextEncoder().encode(payloadText).length : 0;

      let dataType = "unknown";
      let topCount = 0;
      try {
        const parsed = JSON.parse(payloadText);
        if (Array.isArray(parsed)) {
          dataType = "array";
          topCount = parsed.length;
        } else if (parsed === null) {
          dataType = "null";
        } else if (typeof parsed === "object") {
          dataType = "object";
          topCount = Object.keys(parsed).length;
        } else {
          dataType = typeof parsed;
        }
      } catch (err) {
        dataType = "invalid";
      }

      const setStat = (name, value) => {
        const el = document.querySelector(`[data-stat="${name}"]`);
        if (el) {
          el.textContent = value;
        }
      };

      const formatBytes = (value) => {
        if (value < 1024) {
          return `${value} B`;
        }
        if (value < 1024 * 1024) {
          return `${(value / 1024).toFixed(1)} KB`;
        }
        return `${(value / 1024 / 1024).toFixed(1)} MB`;
      };

      setStat("lines", lines.toString());
      setStat("bytes", formatBytes(bytes));
      setStat("type", dataType);
      setStat("top", topCount ? topCount.toString() : "-");

      const stamp = document.querySelector("[data-stat='generated']");
      if (stamp) {
        stamp.textContent = `generated: ${new Date().toLocaleString()}`;
      }

      if (copyBtn && codeEl) {
        copyBtn.addEventListener("click", async () => {
          const text = payloadText;
          const reset = () => {
            copyBtn.textContent = "Copy JSON";
          };
          try {
            await navigator.clipboard.writeText(text);
            copyBtn.textContent = "Copied";
          } catch (err) {
            const range = document.createRange();
            range.selectNodeContents(codeEl);
            const selection = window.getSelection();
            if (selection) {
              selection.removeAllRanges();
              selection.addRange(range);
            }
            try {
              document.execCommand("copy");
              copyBtn.textContent = "Copied";
            } catch (err2) {
              copyBtn.textContent = "Copy failed";
            }
            if (selection) {
              selection.removeAllRanges();
            }
          }
          window.setTimeout(reset, 1400);
        });
      }

      if (wrapBtn && codeEl) {
        wrapBtn.addEventListener("click", () => {
          codeEl.classList.toggle("wrap");
          wrapBtn.textContent = codeEl.classList.contains("wrap") ? "Unwrap" : "Wrap";
        });
      }
    })();
  </script>
</body>
</html>
"""

        return (
            template
            .replace("__NEXTRACT_THEME__", theme)
            .replace("__NEXTRACT_PAYLOAD_HTML__", payload_html)
            .replace("__NEXTRACT_PAYLOAD_JSON__", payload_script)
        )


class CsvFormatter(BaseFormatter):
    """Format extraction results as CSV."""

    @staticmethod
    def _safe_csv_cell(value: object) -> object:
        """Sanitize cell value to prevent CSV formula injection."""
        s = str(value) if value is not None else ""
        if s and s[0] in ("=", "+", "-", "@", "\t", "\r"):
            return "'" + s
        return s

    def format(self, result: ExtractionResult, **kwargs: Any) -> str:
        data = result.data
        output = StringIO()
        writer = csv.writer(output)

        if isinstance(data, list):
            rows = []
            for row in data:
                if isinstance(row, dict):
                    rows.append(row)
                else:
                    rows.append({"value": row})
            if not rows:
                return ""
            headers = sorted({key for row in rows for key in row.keys()})
            writer.writerow([self._safe_csv_cell(h) for h in headers])
            for row in rows:
                writer.writerow([self._safe_csv_cell(row.get(header, "")) for header in headers])
            return output.getvalue()

        if isinstance(data, dict):
            writer.writerow(["field", "value"])
            for key, value in data.items():
                writer.writerow([self._safe_csv_cell(key), self._safe_csv_cell(json.dumps(value, ensure_ascii=False, default=_json_default))])
            return output.getvalue()

        writer.writerow(["value"])
        writer.writerow([self._safe_csv_cell(data)])
        return output.getvalue()
