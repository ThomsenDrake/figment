"""Presentation tokens for the Figment Gradio Server interface."""

FIGMENT_CSS = """
@import url("https://fonts.googleapis.com/css2?family=Inter:opsz,wght@14..32,400..850&display=swap");

:root {
  --figment-ink: #18181b;
  --figment-muted: #60606a;
  --figment-subtle: #8a8a95;
  --figment-line: rgba(24, 24, 27, 0.12);
  --figment-line-strong: rgba(24, 24, 27, 0.18);
  --figment-canvas: #f7f7f4;
  --figment-panel: #ffffff;
  --figment-panel-soft: #f2f6f5;
  --figment-control: #fbfbfa;
  --figment-blue: #155eef;
  --figment-blue-soft: #e8f0ff;
  --figment-green: #087443;
  --figment-green-soft: #e5f6ed;
  --figment-red: #c81e1e;
  --figment-red-soft: #fff0ee;
  --figment-amber: #b45f06;
  --figment-amber-soft: #fff5df;
  --figment-radius: 8px;
  --figment-shadow: 0 18px 50px rgba(24, 24, 27, 0.08);
  color-scheme: light;
  font-family: "Inter", ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  font-feature-settings: "cv02", "cv03", "cv04", "cv11", "ss01", "ss03";
}

* {
  box-sizing: border-box;
}

html {
  background: var(--figment-canvas);
}

body {
  margin: 0;
  min-width: 320px;
  color: var(--figment-ink);
  background:
    linear-gradient(180deg, rgba(242, 246, 245, 0.85), rgba(247, 247, 244, 0.98) 420px),
    var(--figment-canvas);
  -webkit-font-smoothing: antialiased;
  text-rendering: optimizeLegibility;
}

button,
input,
select,
textarea {
  font: inherit;
}

button {
  cursor: pointer;
}

button:disabled,
input:disabled,
textarea:disabled {
  cursor: not-allowed;
  opacity: 0.58;
}

a {
  color: inherit;
}

.figment-app-shell {
  isolate: isolate;
  min-height: 100vh;
  padding: 16px;
  background: var(--figment-canvas);
}

.figment-mission-rail,
.figment-operation-board {
  display: contents;
}

.figment-topbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 20px;
  max-width: 1480px;
  margin: 0 auto;
  padding: 18px 20px;
  border: 1px solid rgba(255, 255, 255, 0.14);
  border-radius: var(--figment-radius);
  background:
    linear-gradient(135deg, rgba(8, 116, 67, 0.24), transparent 38%),
    linear-gradient(90deg, #18181b, #202022);
  box-shadow: var(--figment-shadow);
  color: #ffffff;
}

.figment-brand {
  display: flex;
  align-items: baseline;
  gap: 14px;
  min-width: 0;
}

.figment-logo {
  flex: 0 0 auto;
  color: #ffffff;
  font-size: 28px;
  font-weight: 760;
  letter-spacing: 0;
  line-height: 1;
  text-decoration: none;
}

.figment-positioning {
  min-width: 0;
  color: rgba(255, 255, 255, 0.82);
  font-size: 15px;
  line-height: 1.4;
}

.figment-safety {
  display: flex;
  align-items: center;
  gap: 9px;
  flex: 0 1 auto;
  min-width: 260px;
  justify-content: flex-end;
  color: rgba(255, 255, 255, 0.9);
  font-size: 14px;
  line-height: 1.4;
}

.figment-safety-mark {
  display: grid;
  place-items: center;
  flex: 0 0 auto;
  width: 22px;
  height: 22px;
  border: 1px solid rgba(255, 255, 255, 0.7);
  border-radius: 999px;
  font-size: 14px;
  font-weight: 800;
}

.figment-statusline,
.figment-tabs,
.figment-live-status,
.figment-view,
.figment-footer-rail {
  max-width: 1480px;
  margin-right: auto;
  margin-left: auto;
}

.figment-statusline {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 9px;
  margin-top: 12px;
  padding: 11px 12px;
  border: 1px solid var(--figment-line);
  border-radius: var(--figment-radius);
  background: rgba(255, 255, 255, 0.84);
}

.figment-statusline strong,
.figment-footer-cluster strong {
  color: var(--figment-muted);
  font-size: 13px;
  font-weight: 720;
}

.figment-chip-row,
.figment-footer-cluster {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 8px;
}

.figment-chip {
  display: inline-flex;
  align-items: center;
  max-width: 100%;
  min-height: 26px;
  padding: 4px 9px;
  border: 1px solid var(--figment-line);
  border-radius: 999px;
  background: #ffffff;
  color: var(--figment-ink);
  font-size: 13px;
  font-weight: 650;
  line-height: 1.25;
  white-space: nowrap;
}

.figment-chip.blue {
  border-color: rgba(21, 94, 239, 0.22);
  background: var(--figment-blue-soft);
  color: #1149b8;
}

.figment-chip.green {
  border-color: rgba(8, 116, 67, 0.2);
  background: var(--figment-green-soft);
  color: var(--figment-green);
}

.figment-chip.red {
  border-color: rgba(200, 30, 30, 0.2);
  background: var(--figment-red-soft);
  color: var(--figment-red);
}

.figment-chip.amber {
  border-color: rgba(180, 95, 6, 0.24);
  background: var(--figment-amber-soft);
  color: var(--figment-amber);
}

.figment-tabs {
  display: flex;
  gap: 6px;
  margin-top: 12px;
  padding: 6px;
  border: 1px solid var(--figment-line);
  border-radius: var(--figment-radius);
  background: rgba(255, 255, 255, 0.82);
  overflow-x: auto;
}

.figment-tab-button {
  flex: 0 0 auto;
  min-height: 38px;
  padding: 8px 12px;
  border: 0;
  border-radius: 6px;
  background: transparent;
  color: var(--figment-muted);
  font-size: 14px;
  font-weight: 650;
  white-space: nowrap;
}

.figment-tab-button.is-active {
  background: #f0f3f1;
  color: var(--figment-ink);
}

.figment-tab-button:focus-visible,
.figment-button:focus-visible,
input:focus-visible,
select:focus-visible,
textarea:focus-visible,
summary:focus-visible {
  outline: 2px solid var(--figment-blue);
  outline-offset: 2px;
}

.figment-live-status {
  margin-top: 10px;
  margin-bottom: 0;
  color: var(--figment-muted);
  font-size: 14px;
  line-height: 1.5;
}

.figment-view {
  margin-top: 12px;
}

.figment-workspace {
  display: grid;
  grid-template-columns: minmax(0, 9fr) minmax(360px, 7fr);
  gap: 12px;
  align-items: start;
}

.figment-workspace-intake {
  grid-template-columns: minmax(0, 10fr) minmax(340px, 6fr);
}

.figment-panel {
  min-width: 0;
  padding: 16px;
  border: 1px solid var(--figment-line);
  border-radius: var(--figment-radius);
  background: var(--figment-panel);
  box-shadow: 0 1px 0 rgba(24, 24, 27, 0.03);
  overflow-x: auto;
}

.figment-sticky-panel {
  position: sticky;
  top: 12px;
}

.figment-panel-heading {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 12px;
  margin-bottom: 12px;
}

.figment-panel-heading h2,
.figment-section-title {
  margin: 0;
  color: var(--figment-ink);
  font-size: 16px;
  font-weight: 720;
  letter-spacing: 0;
  line-height: 1.35;
}

.figment-kicker {
  display: block;
  margin-bottom: 6px;
  color: var(--figment-green);
  font-size: 13px;
  font-weight: 720;
  line-height: 1.25;
}

.figment-panel-heading p,
.figment-section-subtitle {
  margin: 4px 0 0;
  max-width: 72ch;
  color: var(--figment-muted);
  font-size: 14px;
  line-height: 1.5;
}

.figment-panel-heading-action {
  align-items: center;
}

.figment-section-divider {
  height: 1px;
  margin: 16px 0;
  background: var(--figment-line);
}

.figment-inline-controls,
.figment-audio-grid {
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto;
  gap: 10px;
  align-items: end;
}

.figment-audio-grid {
  grid-template-columns: minmax(0, 3fr) minmax(260px, 2fr);
  align-items: stretch;
}

.figment-audio-actions {
  display: flex;
  flex-direction: column;
  align-items: stretch;
  justify-content: end;
  gap: 8px;
  min-width: 0;
}

.figment-audio-draft {
  display: grid;
  gap: 12px;
  min-width: 0;
}

.figment-review-panel {
  display: grid;
  align-content: start;
}

.figment-review-heading {
  align-items: start;
}

.figment-field-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 10px;
}

.figment-control {
  display: grid;
  gap: 6px;
  min-width: 0;
}

.figment-control-wide {
  grid-column: 1 / -1;
}

.figment-control span {
  color: var(--figment-muted);
  font-size: 13px;
  font-weight: 680;
  line-height: 1.35;
}

.figment-control input,
.figment-control select,
.figment-control textarea {
  width: 100%;
  min-width: 0;
  border: 1px solid var(--figment-line-strong);
  border-radius: 6px;
  background: var(--figment-control);
  color: var(--figment-ink);
  font-size: 15px;
  line-height: 1.45;
}

.figment-control input,
.figment-control select {
  min-height: 40px;
  padding: 8px 10px;
}

.figment-control textarea {
  min-height: 88px;
  padding: 9px 10px;
  resize: vertical;
}

.figment-button {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  min-height: 38px;
  padding: 8px 12px;
  border: 1px solid transparent;
  border-radius: 6px;
  font-size: 14px;
  font-weight: 720;
  line-height: 1.25;
  white-space: nowrap;
}

.figment-button-primary {
  border-color: var(--figment-blue);
  background: var(--figment-blue);
  color: #ffffff;
}

.figment-button-primary:disabled {
  border-color: var(--figment-line-strong);
  background: #eef0f3;
  color: var(--figment-muted);
}

.figment-button-secondary {
  border-color: var(--figment-line-strong);
  background: #ffffff;
  color: var(--figment-ink);
}

.figment-button-loading {
  color: rgba(255, 255, 255, 0.78);
}

.figment-full-button {
  width: 100%;
  margin-top: 12px;
}

.figment-panel-soft {
  border: 1px solid var(--figment-line);
  border-radius: var(--figment-radius);
  background: var(--figment-panel-soft);
  padding: 12px;
}

.figment-card-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 10px;
}

.figment-mini-card {
  min-width: 0;
  min-height: 112px;
  border: 1px solid var(--figment-line);
  border-radius: var(--figment-radius);
  background: #ffffff;
  padding: 12px;
  overflow-x: auto;
}

.figment-mini-card h4 {
  margin: 0 0 8px;
  color: var(--figment-ink);
  font-size: 14px;
  font-weight: 720;
  line-height: 1.35;
}

.figment-mini-card p {
  margin: 0;
  color: var(--figment-muted);
  font-size: 14px;
  line-height: 1.5;
}

.figment-mini-card ul {
  margin: 0;
  padding-left: 18px;
}

.figment-checklist {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 8px 16px;
  margin: 0;
  padding: 0;
  list-style: none;
}

.figment-checklist li {
  position: relative;
  min-width: 0;
  padding-left: 22px;
  color: var(--figment-ink);
  font-size: 14px;
  line-height: 1.45;
}

.figment-checklist li::before {
  content: "";
  position: absolute;
  left: 0;
  top: 4px;
  width: 13px;
  height: 13px;
  border: 1px solid rgba(24, 24, 27, 0.34);
  border-radius: 3px;
  background: #ffffff;
}

.figment-checklist.checked li::before {
  border-color: var(--figment-blue);
  background: var(--figment-blue);
  box-shadow: inset 0 0 0 3px #ffffff;
}

.figment-table {
  width: 100%;
  min-width: 560px;
  border-collapse: collapse;
  background: transparent;
  color: var(--figment-ink);
  font-size: 13px;
  line-height: 1.45;
}

.figment-table th,
.figment-table td {
  padding: 10px 8px;
  border-bottom: 1px solid var(--figment-line);
  text-align: left;
  vertical-align: top;
}

.figment-table th {
  color: var(--figment-muted);
  font-size: 12px;
  font-weight: 760;
  white-space: nowrap;
}

.figment-table tr:last-child td {
  border-bottom: 0;
}

.figment-urgency-banner {
  display: grid;
  grid-template-columns: minmax(0, 1fr) minmax(240px, auto);
  gap: 12px;
  align-items: center;
  padding: 14px;
  border: 1px solid var(--figment-line);
  border-radius: var(--figment-radius);
  background: var(--figment-panel-soft);
}

.figment-urgency-word {
  display: inline-flex;
  align-items: center;
  min-height: 42px;
  padding: 0 16px;
  border-radius: 6px;
  font-size: 24px;
  font-weight: 820;
  letter-spacing: 0;
}

.figment-urgency-word.routine {
  background: var(--figment-blue-soft);
  color: #1149b8;
}

.figment-urgency-word.monitor {
  background: var(--figment-green-soft);
  color: var(--figment-green);
}

.figment-urgency-word.urgent {
  background: var(--figment-amber-soft);
  color: var(--figment-amber);
}

.figment-urgency-word.emergency {
  background: var(--figment-red);
  color: #ffffff;
}

.figment-lockout {
  border: 1px solid rgba(200, 30, 30, 0.22);
  border-radius: 6px;
  padding: 11px;
  background: var(--figment-red-soft);
  color: #8f1616;
  font-size: 13px;
  line-height: 1.45;
}

.figment-json {
  display: block;
  min-height: 210px;
  max-height: 420px;
  margin: 0;
  padding: 12px;
  overflow: auto;
  border: 1px solid var(--figment-line);
  border-radius: 6px;
  background: #111114;
  color: #edfdf5;
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
  font-size: 12px;
  line-height: 1.55;
  white-space: pre-wrap;
  overflow-wrap: anywhere;
}

.figment-json-tall {
  min-height: 520px;
  max-height: 720px;
}

.figment-disclosure {
  margin-top: 12px;
  border-top: 1px solid var(--figment-line);
  padding-top: 10px;
}

.figment-disclosure summary {
  color: var(--figment-muted);
  font-size: 14px;
  font-weight: 680;
  cursor: pointer;
}

.figment-disclosure .figment-json {
  margin-top: 10px;
}

.figment-footer-rail {
  display: flex;
  align-items: center;
  justify-content: space-between;
  flex-wrap: wrap;
  gap: 12px;
  margin-top: 12px;
  padding: 12px;
  border: 1px solid var(--figment-line);
  border-radius: var(--figment-radius);
  background: rgba(255, 255, 255, 0.84);
}

.figment-muted {
  color: var(--figment-muted);
}
.figment-app-shell {
  --figment-ink: #13201b;
  --figment-muted: #5f6d68;
  --figment-subtle: #87938f;
  --figment-line: rgba(19, 32, 27, 0.12);
  --figment-line-strong: rgba(19, 32, 27, 0.2);
  --figment-canvas: #edf3ee;
  --figment-panel: #fffefa;
  --figment-panel-soft: #eef8f3;
  --figment-control: #ffffff;
  --figment-blue: #1065d8;
  --figment-blue-soft: #e7f0ff;
  --figment-green: #08734f;
  --figment-green-soft: #dff5eb;
  --figment-red: #c63a2b;
  --figment-red-soft: #fff0ec;
  --figment-amber: #a85f00;
  --figment-amber-soft: #fff2d2;
  --figment-radius: 8px;
  --figment-shadow: none;
  display: grid;
  grid-template-columns: minmax(260px, 300px) minmax(0, 1fr);
  grid-template-areas:
    "rail board";
  align-items: start;
  gap: 16px;
  max-width: 1620px;
  margin: 0 auto;
  padding: 18px;
}

.figment-app-shell .figment-mission-rail {
  grid-area: rail;
  display: grid;
  gap: 16px;
  align-content: start;
  min-width: 0;
  max-height: none;
  overflow: visible;
  position: static;
}

.figment-app-shell .figment-operation-board {
  grid-area: board;
  display: block;
  min-width: 0;
}

.figment-app-shell .figment-topbar {
  grid-area: auto;
  display: grid;
  gap: 18px;
  align-content: start;
  margin: 0;
  min-height: 236px;
  padding: 22px;
  color: #f8fff9;
  background:
    linear-gradient(180deg, rgba(255, 255, 255, 0.08), transparent 42%),
    #15241e;
  border: 1px solid rgba(255, 255, 255, 0.14);
  box-shadow: none;
}

.figment-app-shell .figment-brand {
  display: grid;
  gap: 12px;
}

.figment-app-shell .figment-logo {
  color: #ffffff;
  font-size: 34px;
}

.figment-app-shell .figment-positioning {
  max-width: 23ch;
  color: rgba(248, 255, 249, 0.82);
  font-size: 17px;
  line-height: 1.45;
}

.figment-app-shell .figment-safety {
  align-self: end;
  min-width: 0;
  justify-content: flex-start;
  padding-top: 16px;
  border-top: 1px solid rgba(255, 255, 255, 0.16);
  color: rgba(248, 255, 249, 0.9);
  font-size: 14px;
}

.figment-app-shell .figment-statusline {
  grid-area: auto;
  align-self: stretch;
  display: grid;
  margin: 0;
  padding: 14px;
  background: #fffefa;
  border-color: rgba(19, 32, 27, 0.12);
}

.figment-app-shell .figment-statusline strong {
  flex-basis: 100%;
  color: #08734f;
  font-size: 14px;
}

.figment-app-shell .figment-tabs {
  grid-area: auto;
  counter-reset: figment-step;
  display: grid;
  gap: 8px;
  align-content: start;
  margin: 0;
  padding: 12px;
  background: #fffefa;
  border-color: rgba(19, 32, 27, 0.12);
  overflow: visible;
}

.figment-app-shell .figment-tab-button {
  counter-increment: figment-step;
  justify-content: flex-start;
  gap: 10px;
  width: 100%;
  min-height: 48px;
  padding: 10px 11px;
  border: 1px solid transparent;
  background: transparent;
  color: var(--figment-muted);
  text-align: left;
}

.figment-app-shell .figment-tab-button::before {
  content: "0" counter(figment-step);
  display: inline-grid;
  place-items: center;
  flex: 0 0 auto;
  width: 30px;
  height: 30px;
  border: 1px solid var(--figment-line);
  border-radius: 999px;
  color: #08734f;
  font-size: 12px;
  font-weight: 760;
}

.figment-app-shell .figment-tab-button.is-active {
  border-color: rgba(8, 115, 79, 0.18);
  background: var(--figment-green-soft);
  color: var(--figment-ink);
}

.figment-app-shell .figment-live-status {
  grid-area: auto;
  margin: 0;
  padding: 12px;
  background: #fffefa;
  border: 1px solid rgba(19, 32, 27, 0.12);
  border-radius: var(--figment-radius);
  color: #50635b;
  font-size: 15px;
}

.figment-app-shell .figment-view {
  margin: 0;
}

.figment-app-shell .figment-footer-rail {
  grid-area: auto;
  display: grid;
  margin: 0;
  padding: 12px;
  background: #fffefa;
  border-color: rgba(19, 32, 27, 0.12);
}

.figment-app-shell .figment-footer-cluster {
  align-items: flex-start;
}

.figment-app-shell .figment-workspace {
  grid-template-columns: minmax(0, 1.15fr) minmax(320px, 0.85fr);
  gap: 18px;
}

.figment-app-shell .figment-workspace-intake {
  grid-template-columns: minmax(0, 1.25fr) minmax(340px, 0.75fr);
}

.figment-app-shell .figment-panel {
  position: relative;
  padding: 22px;
  background: #fffefa;
  border-color: rgba(19, 32, 27, 0.12);
  overflow: hidden;
}

.figment-app-shell .figment-panel::before {
  content: "";
  position: absolute;
  top: 0;
  right: 0;
  left: 0;
  height: 5px;
  background: linear-gradient(90deg, #08734f, #1065d8 54%, #c63a2b);
}

.figment-app-shell .figment-panel-heading h2 {
  font-size: 18px;
}

.figment-app-shell .figment-panel-heading p,
.figment-app-shell .figment-section-subtitle {
  font-size: 15px;
}

.figment-app-shell .figment-section-divider {
  margin: 22px 0;
  background: rgba(19, 32, 27, 0.1);
}

.figment-app-shell .figment-intake-panel {
  padding: 24px;
}

.figment-app-shell .figment-audio-draft {
  padding-top: 2px;
}

.figment-app-shell .figment-audio-draft .figment-panel-heading {
  margin-bottom: 0;
}

.figment-app-shell .figment-review-panel {
  gap: 0;
}

.figment-app-shell .figment-review-heading {
  display: grid;
  gap: 12px;
}

.figment-app-shell .figment-review-heading .figment-button {
  width: 100%;
}

.figment-app-shell .figment-field-grid {
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 14px;
}

.figment-app-shell .figment-inline-controls {
  grid-template-columns: minmax(0, 1fr) minmax(120px, auto);
}

.figment-app-shell .figment-control input,
.figment-app-shell .figment-control select,
.figment-app-shell .figment-control textarea {
  border-color: rgba(19, 32, 27, 0.16);
  background: #ffffff;
  font-size: 16px;
}

.figment-app-shell .figment-control input,
.figment-app-shell .figment-control select {
  min-height: 46px;
}

.figment-app-shell .figment-button {
  min-height: 42px;
}

.figment-app-shell .figment-audio-draft .figment-button {
  min-height: 38px;
}

.figment-app-shell .figment-button-primary {
  border-color: #08734f;
  background: #08734f;
}

.figment-app-shell .figment-button-primary:disabled {
  border-color: rgba(19, 32, 27, 0.14);
  background: #e9eee9;
}

.figment-app-shell .figment-sticky-panel {
  top: 18px;
}

.figment-app-shell .figment-json {
  background: #16342b;
  border-color: rgba(8, 115, 79, 0.2);
  color: #f0fff7;
}

.figment-app-shell #intake-json {
  min-height: 320px;
}

.figment-app-shell #audio-json {
  min-height: 180px;
}

.figment-app-shell .figment-mini-card,
.figment-app-shell .figment-panel-soft {
  background: #f8fbf7;
  border-color: rgba(19, 32, 27, 0.1);
}

.figment-app-shell .figment-urgency-banner {
  background: #f7faf6;
  border-color: rgba(19, 32, 27, 0.12);
}

@media (max-width: 1100px) {
  .figment-workspace,
  .figment-workspace-intake,
  .figment-audio-grid {
    grid-template-columns: 1fr;
  }

  .figment-sticky-panel {
    position: static;
  }
}

@media (max-width: 760px) {
  .figment-app-shell {
    padding: 10px;
  }

  .figment-topbar,
  .figment-brand,
  .figment-safety,
  .figment-panel-heading,
  .figment-panel-heading-action,
  .figment-footer-rail {
    display: grid;
    justify-content: stretch;
  }

  .figment-safety {
    min-width: 0;
    justify-content: start;
  }

  .figment-logo {
    font-size: 26px;
  }

  .figment-positioning,
  .figment-safety,
  .figment-control input,
  .figment-control select,
  .figment-control textarea,
  .figment-button,
  .figment-live-status {
    font-size: 16px;
  }

  .figment-statusline,
  .figment-tabs,
  .figment-panel,
  .figment-footer-rail {
    padding: 10px;
  }

  .figment-inline-controls,
  .figment-field-grid,
  .figment-card-grid,
  .figment-checklist,
  .figment-urgency-banner {
    grid-template-columns: 1fr;
  }

  .figment-button {
    min-height: 44px;
    width: 100%;
  }

  .figment-json {
    min-height: 180px;
  }
}

@media (max-width: 1100px) {
  .figment-app-shell {
    display: flex;
    flex-direction: column;
    padding: 10px;
  }

  .figment-app-shell .figment-mission-rail,
  .figment-app-shell .figment-operation-board {
    max-height: none;
    overflow: visible;
    position: static;
  }

  .figment-app-shell .figment-mission-rail {
    display: contents;
  }

  .figment-app-shell .figment-operation-board {
    display: block;
    order: 3;
  }

  .figment-app-shell .figment-topbar,
  .figment-app-shell .figment-statusline,
  .figment-app-shell .figment-tabs,
  .figment-app-shell .figment-live-status,
  .figment-app-shell .figment-view,
  .figment-app-shell .figment-footer-rail {
    margin-top: 12px;
    min-width: 0;
    width: 100%;
  }

  .figment-app-shell .figment-topbar {
    order: 1;
    min-height: 0;
    padding: 18px;
  }

  .figment-app-shell .figment-positioning {
    max-width: 34ch;
  }

  .figment-app-shell .figment-safety {
    padding-top: 12px;
  }

  .figment-app-shell .figment-tabs {
    order: 2;
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 8px;
    overflow: visible;
  }

  .figment-app-shell .figment-tab-button {
    width: 100%;
    min-height: 52px;
    white-space: normal;
  }

  .figment-app-shell .figment-live-status {
    order: 4;
  }

  .figment-app-shell .figment-statusline {
    order: 5;
  }

  .figment-app-shell .figment-footer-rail {
    order: 6;
  }

  .figment-app-shell .figment-workspace,
  .figment-app-shell .figment-workspace-intake {
    grid-template-columns: 1fr;
  }
}
"""
