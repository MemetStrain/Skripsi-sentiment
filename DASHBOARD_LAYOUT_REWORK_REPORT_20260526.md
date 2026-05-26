# Dashboard Layout Rework Report — 2026-05-26

Single file edited: `website/web/templates/dashboard.html`
No change to `views.py` or any other file.

---

## §2 — Configuration Comparison table removed

Deleted in one edit block:
- HTML: `<h3>Configuration comparison</h3>`, its description, `<select id="metric-select">`, and `<div id="metrics-table">` (the entire Row C `lg:col-span-2` section).
- JS: `fmtMetric()` helper, `isLowerBetter()` helper, `renderMetricsTable()` function, the `metric-select` `change` event listener, and the init/guard block that called `renderMetricsTable()` or wrote the "No comparison data yet…" fallback.

Post-deletion grep of the whole file for `metrics-table`, `metric-select`, `renderMetricsTable`:
**Zero remaining references** (the orphaned `.metric-select` CSS class remains — harmless, no element targets it).

**Hero tiles unaffected:** `#ki-hero-comparison` and `#ki-hero-metrics` are rendered by `renderKeyInsightsHero(forecasts)` which reads `forecasts.metrics` and `winnersData` directly — they do not depend on the deleted table. Both tiles continue to populate normally.

---

## §3a — Left info rail rebuilt (7 cards)

The `<aside class="lg:col-span-1 space-y-4">` now contains, top to bottom:

| # | Card | Source |
|---|------|--------|
| 1 | **Price stats · 90d** | Existing — unchanged |
| 2 | **Market regime · price** (`#ki-hero-price`) | Moved from former Row A right column |
| 3 | **Feature set** | Existing — unchanged |
| 4 | **Best model · h=1** | Existing — unchanged |
| 5 | **News sentiment** (`#ki-hero-sentiment`) | Moved from former Row B right column |
| 6 | **Feature comparison** (`#ki-hero-comparison`) | Moved from former Row C right column |
| 7 | **Forecast reliability** (`#ki-hero-metrics`) | Moved from former Row C right column |

Each hero tile (cards 2, 5, 6, 7) uses the standard sidebar card chrome:
`bg-white rounded-xl shadow-card border border-slate-100 p-5` with a `text-[10px] uppercase tracking-wide text-slate-400` label header — matching the existing sidebar card style. Each tile is full rail width (no sub-grid cramping).

The **Forecast Models** card that previously sat at position 2 in the sidebar was removed from the rail and relocated to the combined block (§3c).

`getElementById('key-insights')` was grepped in the `<script>` block — zero references. The `#key-insights` wrapper had already been removed in a prior session; no empty placeholder was needed.

---

## §3b — Main column restructured (charts dominate)

`<div class="lg:col-span-3 space-y-5">` now contains three direct children:

1. **Price chart** `<section>` — full `lg:col-span-3` width. All internals unchanged: `#priceChart`, horizon legend, `#fan-mode-toggle`, `#price-insight-strip`, state legend.
2. **Sentiment chart** `<section>` — full width. All internals unchanged: `#sentimentChart`, `#sentiment-insight-strip`.
3. **Combined "Model details & breakdown"** `<section>` — new (§3c).

The former inner `grid grid-cols-1 lg:grid-cols-3 gap-5` wrappers for Rows A, B, and C are gone. No hero tiles remain in the main column.

---

## §3c — Combined "Model details & breakdown" block

One card (`bg-white rounded-xl shadow-card border border-slate-100 p-6`) titled **"Model details & breakdown"** with subtitle **"Per-horizon winning configs and the full per-category insight breakdown."**

Contains, in order:

**Forecast models section** (relocated from sidebar):
- `<h4 class="eyebrow">Forecast models</h4>`
- "Auto-picked per horizon…" description
- `<div id="winners-summary">` — populated by `renderWinnersSummary()` from `winnersData`; no JS change needed.

Separated by a `border-b border-slate-100` divider.

**Four collapsible detail lists** in `grid grid-cols-1 md:grid-cols-2 gap-5`:

| Summary label | `<ul>` id | Filled by |
|---------------|-----------|-----------|
| Price & regime | `ki-price` | `renderKeyInsights()` |
| News sentiment | `ki-sentiment` | `renderKeyInsights()` |
| Configuration comparison | `ki-comparison` | `renderKeyInsights()` |
| Forecast reliability | `ki-metrics` | `renderKeyInsights()` |

All `<ul>` ids are unchanged — `renderKeyInsights()` fills them by id with no JS modification. Each `<summary>` uses `class="details-summary"` (existing CSS: `text-xs font-medium text-slate-500 cursor-pointer select-none` + `focus-visible` ring matching the `horizon-toggle` pattern).

Trade-off acknowledged: the hero tiles (Feature Comparison, Forecast Reliability) live in the left rail while their detail lists are in this bottom block — this is the intentional single consolidated details section design.

---

## §3d — Column balance

The 7-card rail will be taller than the 2 charts + combined block at typical `lg`/`xl` laptop widths. The `space-y-4` rail spacing keeps cards compact. The combined block at the bottom of the main column adds meaningful height (winners rows + 4 collapsible lists) to close the gap.

The optional `xl:grid-cols-2` sub-wrapper mitigation (for tiles 5+6 or 6+7 side-by-side at xl) was not implemented — the combined block provides sufficient main-column height that the gap is acceptable. If after visual review the rail still overshoots substantially, wrapping cards 6+7 in `<div class="grid grid-cols-1 xl:grid-cols-2 gap-4">` will close it without changing the `lg` single-column stacking order.

---

## All required ids — post-edit verification

Grep confirmed all 16 JS-referenced ids survive in the new structure:
`ki-hero-price`, `ki-hero-sentiment`, `ki-hero-comparison`, `ki-hero-metrics`,
`ki-price`, `ki-sentiment`, `ki-comparison`, `ki-metrics`,
`priceChart`, `sentimentChart`, `price-insight-strip`, `sentiment-insight-strip`,
`horizon-legend`, `fan-mode-toggle`, `winners-summary`, `forecasts-status`.
`.tip` and `.horizon-toggle` class selectors: 11 occurrences (unchanged).

---

## Verification checklist

- [ ] `cd website && python manage.py runserver` — console clean (zero JS errors)
- [ ] Grep: zero references to `metrics-table`, `metric-select`, `renderMetricsTable`
- [ ] Both charts render; horizon toggles, `fan-mode-toggle`, `.tip` tooltips, price tooltip + caption strips intact
- [ ] Left rail: 7 cards in order (Price stats → Market Regime → Feature set → Best model → News sentiment → Feature comparison → Forecast reliability), each full rail width
- [ ] Hero tiles populate after forecasts load (no "pending…" stuck)
- [ ] Combined block: `winners-summary` rows populated; 4 `<details>` expand to their populated lists
- [ ] Columns roughly balanced — no large blank gap at `lg`/`xl`
- [ ] Mobile (<768px): single column, no overflow; `<summary>` shows focus ring
