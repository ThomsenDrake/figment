"""Presentation tokens for the Figment Gradio interface."""

FIGMENT_CSS = """
:root {
  --figment-ink: #0f172a;
  --figment-muted: #526174;
  --figment-line: #d8e0ea;
  --figment-panel: #ffffff;
  --figment-panel-soft: #f8fafc;
  --figment-blue: #0b57d0;
  --figment-blue-soft: #eaf2ff;
  --figment-green: #17803d;
  --figment-green-soft: #eaf8ef;
  --figment-red: #c91515;
  --figment-red-soft: #fff0f0;
  --figment-amber: #b15b00;
  --figment-amber-soft: #fff7e8;
}

.gradio-container {
  max-width: none !important;
  color: var(--figment-ink);
  background: #f5f7fb !important;
  padding: 0 !important;
}

.figment-shell {
  min-height: 100vh;
}

.figment-topbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 24px;
  padding: 16px 22px;
  color: #fff;
  background: linear-gradient(90deg, #061425 0%, #0b1d32 56%, #07111f 100%);
  border-radius: 0;
}

.figment-topbar,
.figment-topbar * {
  color: #fff !important;
}

.figment-brand {
  display: flex;
  align-items: baseline;
  gap: 18px;
  min-width: 0;
}

.figment-logo {
  font-size: 30px;
  line-height: 1;
  font-weight: 760;
}

.figment-positioning {
  font-size: 15px;
  opacity: 0.92;
  white-space: nowrap;
}

.figment-safety {
  display: flex;
  align-items: center;
  gap: 9px;
  font-size: 14px;
  opacity: 0.95;
  white-space: nowrap;
}

.figment-safety-mark {
  display: grid;
  place-items: center;
  width: 22px;
  height: 22px;
  border: 1px solid rgba(255, 255, 255, 0.7);
  border-radius: 999px;
  font-weight: 800;
}

.figment-statusline {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 10px 14px;
  padding: 12px 16px;
  margin: 14px 18px 0;
  border: 1px solid var(--figment-line);
  border-radius: 8px;
  background: #fff;
}

.figment-statusline strong {
  font-size: 14px;
}

.figment-chip {
  display: inline-flex;
  align-items: center;
  min-height: 26px;
  padding: 4px 10px;
  border: 1px solid var(--figment-line);
  border-radius: 6px;
  background: #fff;
  color: var(--figment-ink);
  font-size: 13px;
  font-weight: 650;
}

.figment-chip.blue {
  border-color: #93b7ff;
  background: var(--figment-blue-soft);
  color: #0749b2;
}

.figment-chip.green {
  border-color: #9ed7ad;
  background: var(--figment-green-soft);
  color: #106a30;
}

.figment-chip.red {
  border-color: #ffabab;
  background: var(--figment-red-soft);
  color: var(--figment-red);
}

.figment-chip.amber {
  border-color: #ffd58d;
  background: var(--figment-amber-soft);
  color: var(--figment-amber);
}

.figment-tabs {
  margin: 14px 18px 0;
}

.figment-tabs > .tab-nav,
.figment-tabs [role="tablist"] {
  background: #fff;
  border: 1px solid var(--figment-line);
  border-radius: 8px 8px 0 0;
}

.figment-tabs button[role="tab"] {
  min-height: 54px;
  font-weight: 680;
}

.figment-tabs button[aria-selected="true"] {
  color: var(--figment-blue) !important;
  border-bottom: 3px solid var(--figment-blue) !important;
}

.figment-tab-body {
  padding: 14px 0 0;
}

.figment-panel {
  border: 1px solid var(--figment-line) !important;
  border-radius: 8px !important;
  background: var(--figment-panel) !important;
  box-shadow: 0 1px 0 rgba(15, 23, 42, 0.03);
}

.figment-panel-soft {
  border: 1px solid var(--figment-line);
  border-radius: 8px;
  background: var(--figment-panel-soft);
  padding: 12px;
}

.figment-section-title {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  margin: 0 0 10px;
  color: var(--figment-ink);
  font-size: 16px;
  font-weight: 760;
}

.figment-section-subtitle {
  margin: -4px 0 12px;
  color: var(--figment-muted);
  font-size: 13px;
}

.figment-quick-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 8px;
}

.figment-demo-pill {
  padding: 8px 10px;
  border: 1px solid var(--figment-line);
  border-radius: 6px;
  background: #fff;
  font-size: 12px;
  font-weight: 650;
  text-align: center;
}

.figment-output-grid {
  display: grid;
  grid-template-columns: minmax(0, 1.05fr) minmax(0, 1fr);
  gap: 14px;
}

.figment-card-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 10px;
}

.figment-mini-card {
  min-height: 120px;
  border: 1px solid var(--figment-line);
  border-radius: 8px;
  background: #fff;
  padding: 12px;
}

.figment-mini-card h4 {
  margin: 0 0 8px;
  font-size: 14px;
}

.figment-mini-card ul {
  margin: 0;
  padding-left: 18px;
}

.figment-table {
  width: 100%;
  border-collapse: collapse;
  overflow: hidden;
  border: 1px solid var(--figment-line);
  border-radius: 8px;
  background: #fff;
  font-size: 13px;
}

.figment-table th,
.figment-table td {
  padding: 10px 12px;
  border-bottom: 1px solid var(--figment-line);
  text-align: left;
  vertical-align: top;
}

.figment-table th {
  background: #f7f9fc;
  color: #243044;
  font-size: 12px;
  font-weight: 760;
}

.figment-table tr:last-child td {
  border-bottom: 0;
}

.figment-urgency-banner {
  display: grid;
  grid-template-columns: 1fr auto;
  gap: 14px;
  align-items: center;
  padding: 14px 16px;
  border: 1px solid var(--figment-line);
  border-radius: 8px;
  background: #fff;
}

.figment-urgency-word {
  display: inline-flex;
  align-items: center;
  min-height: 44px;
  padding: 0 18px;
  border-radius: 6px;
  font-size: 26px;
  font-weight: 850;
  letter-spacing: 0;
}

.figment-urgency-word.routine {
  background: #eef6ff;
  color: #0b57d0;
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
  background: #d41111;
  color: #fff;
}

.figment-lockout {
  border: 1px solid #ef7b7b;
  border-radius: 8px;
  padding: 12px;
  background: var(--figment-red-soft);
  color: #7c1010;
  font-size: 13px;
}

.figment-checklist {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 8px 18px;
  margin: 0;
  padding: 0;
  list-style: none;
}

.figment-checklist li {
  position: relative;
  padding-left: 22px;
  color: #172033;
  font-size: 13px;
}

.figment-checklist li::before {
  content: "";
  position: absolute;
  left: 0;
  top: 2px;
  width: 13px;
  height: 13px;
  border: 1px solid #9aa9bb;
  border-radius: 3px;
  background: #fff;
}

.figment-checklist.checked li::before {
  border-color: var(--figment-blue);
  background: var(--figment-blue);
  box-shadow: inset 0 0 0 3px #fff;
}

.figment-code-panel textarea,
.figment-code-panel .json-viewer,
.figment-code-panel pre {
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace !important;
}

.figment-action-row {
  display: grid;
  grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
  gap: 10px;
}

.figment-footer-rail {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  margin: 14px 18px 18px;
  padding: 12px 16px;
  border: 1px solid var(--figment-line);
  border-radius: 8px;
  background: #fff;
}

.figment-footer-cluster {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 10px;
}

.figment-json-compact {
  min-height: 220px;
}

.figment-json-tall {
  min-height: 520px;
}

.figment-muted {
  color: var(--figment-muted);
}

.figment-hide-label > label {
  display: none !important;
}

button.primary,
.primary button {
  background: var(--figment-blue) !important;
  border-color: var(--figment-blue) !important;
  color: #fff !important;
  min-height: 42px !important;
  height: 42px !important;
  flex: 0 0 auto !important;
}

.figment-panel button {
  min-height: 38px !important;
  height: auto !important;
  flex: 0 0 auto !important;
}

.figment-panel textarea {
  min-height: 38px !important;
}

.figment-panel .form {
  gap: 8px !important;
}

@media (max-width: 1000px) {
  .figment-topbar,
  .figment-output-grid,
  .figment-card-grid,
  .figment-action-row {
    grid-template-columns: 1fr;
  }

  .figment-topbar {
    display: grid;
  }

  .figment-positioning,
  .figment-safety {
    white-space: normal;
  }

  .figment-quick-grid {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
}
"""
