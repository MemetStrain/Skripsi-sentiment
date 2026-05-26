# Dashboard UX Fixes Report — 2026-05-26

Single file edited: `website/web/templates/dashboard.html`
No change to `views.py` or any other file.

---

## `#key-insights` JS reference audit

Grepped the `<script>` block for `getElementById('key-insights')` and `'key-insights'`.
**Result: not referenced in JS.** The id appeared only as the section's own HTML declaration.
The outer `<section id="key-insights">` wrapper was therefore removed entirely; all inner ids
(`ki-hero-price`, `ki-hero-sentiment`, `ki-hero-comparison`, `ki-hero-metrics`,
`ki-price`, `ki-sentiment`, `ki-comparison`, `ki-metrics`) are preserved and still filled
by the existing JS functions.

---

## Fix P1 — Tooltip reframe (price chart)

**Location:** `options.plugins.tooltip.callbacks` inside the `new Chart(ctx, …)` call.

Changes:
1. Added `title` callback: `(items) => items.length ? \`All forecasts for ${items[0].label}\` : ''`
2. Changed `label` callback to build a `let line` instead of a direct return, then
   appends `(±x% vs close)` for all model lines where `c.datasetIndex !== 0` and a
   valid close price exists at that index. The close-price line itself (dataset 0) is
   skipped so it never shows a self-comparison.
3. Added `footer` callback: static caption explaining the same-day framing
   (`'Each line is one model predicting this same day,\nmade 1–7 days earlier.'`).
4. Added `footerFont: { weight: 'normal' }` and `footerColor: '#94a3b8'` so the footer
   renders as a caption, not a heading.

---

## Fix P3 — Sentiment strip double-scaling

**Location:** `renderSentimentInsightStrip()` function.

Root cause: `positive_prob` / `negative_prob` are already on a 0–100 scale (confirmed by
the sentiment chart's `y: { min: 0, max: 100 }` axis and its tooltip callback
`.toFixed(2)%`). The strip code called `kiMean(…) * 100`, doubling the already-percent
value to ~4 239%.

Fix:
- Removed the `* 100` multiplications.
- Introduced a `mean(k)` helper for DRY computation.
- Applied scale-invariant normalization: compute raw means `p`, `n`, `u`; divide each
  by `total = p + n + u`; round to integers; derive `neu = 100 - pos - neg` so the
  three values always sum to exactly 100 regardless of floating-point rounding.
- Added a one-line comment noting the 0–100 scale to prevent regression.
- Coverage/volume chip logic is unchanged.

---

## Fix P2 — Layout restructure (collision removal + paired tiles)

### 4a. Three paired rows in the right column

Replaced the flat `space-y-5` stack in `lg:col-span-3` with three inner-grid rows
(`grid grid-cols-1 lg:grid-cols-3 gap-5`). Each chart/table takes `lg:col-span-2` and
its insight tile takes `lg:col-span-1`.

| Row | `lg:col-span-2` | `lg:col-span-1` |
|-----|-----------------|-----------------|
| A   | Price chart (+ strip, legend) | Market Regime · Price tile (`#ki-hero-price`) |
| B   | Sentiment chart (+ strip) | News Sentiment tile (`#ki-hero-sentiment`) |
| C   | Configuration Comparison table | Stacked pair: Feature Comparison (`#ki-hero-comparison`) + Forecast Reliability (`#ki-hero-metrics`) |

### 4b. Per-tile collapsible details

Dissolved the single global `<details>` "full breakdown" dropdown. Each tile now has its
own `<details class="mt-3"><summary class="details-summary">Details</summary><ul id="ki-…">`.
The `<ul>` ids are unchanged — no JS was modified.

Added a CSS class `.details-summary` (in the `<style>` block) for consistent styling and
a visible `focus-visible` outline (`2px solid #0f766e`, matching the existing
`horizon-toggle` focus treatment).

### 4c. Sidebar declutter

The `#key-insights` section was removed from the sidebar. The sidebar retains:
Price Stats, Forecast Models, Feature Set, and the Best Model / Metrics card (already a
single consolidated card with a 3-column stat grid, not requiring further consolidation).
The `<aside>` element was already present.

### 4d. Balance

Sidebar is now shorter (4 cards). Right column carries 3 rows, each with a chart
`lg:col-span-2` section and a tile `lg:col-span-1`. At `lg`/`xl` widths the price chart
is tall (~420 px) so Row A is well balanced. Row B's shorter sentiment chart means the
tile column may be taller — this is fine as both sides have content. Row C's comparison
table scales with data; the stacked tiles on the right fill naturally.

---

## Verification checklist status

- [ ] `cd website && python manage.py runserver` — console clean
- [ ] Both charts render; horizon toggles, fan-mode-toggle, metric-select, `.tip` all work
- [ ] Tooltip: header "All forecasts for {date}", model lines show `(±x% vs close)`, footer text present, no `NaN`/`undefined`
- [ ] Sentiment strip: percentages sum to 100, no thousands-of-percent values
- [ ] Layout: tiles paired beside their charts; sidebar no longer crammed
- [ ] Per-tile details: each `<details>` expands to its populated list; old single dropdown gone
- [ ] Mobile (<768px): rows stack, tiles fall under charts, no overflow
- [ ] Keyboard: `<summary>` elements show visible focus ring
