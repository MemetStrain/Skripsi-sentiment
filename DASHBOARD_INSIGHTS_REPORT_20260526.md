# Dashboard Insights Report — 2026-05-26

Single file edited: `website/web/templates/dashboard.html`

---

## Task A — Quick-insight caption strips

### New DOM elements

| Element | Placement | Trigger |
|---------|-----------|---------|
| `<div id="price-insight-strip">` | After `<canvas id="priceChart">` container, before the State legend row | HTML placeholder on page load; overwritten by `renderPriceInsightStrip(data)` after `/api/forecasts/` succeeds |
| `<div id="sentiment-insight-strip">` | After `<canvas id="sentimentChart">` container, before `</section>` | `renderSentimentInsightStrip()` called immediately after `dismissSkeleton('sentimentChart-skeleton')` in both the `if` and `else` branches |

### New JS functions

#### `renderPriceInsightStrip(forecasts)`
Data fields read:

| Chip | Field(s) |
|------|----------|
| Next day (h=1) | `forward_fan[0].anchor_price`, `forward_fan.find(h===1).predicted_price` (fallback: last point of `trails` h=1), `configs['1']`, `winners['1']`, `metrics[winner]['1'].CSA.mape ?? .BASE.mape` |
| Direction consensus | `forward_fan[*].predicted_price` vs anchor; falls back to `trails[*].points[-1].predicted_price` when fan is empty |
| Window trend | `prices` global (chartData actuals), `chartData.length` for n |

Call site: inside `loadForecasts()` success handler, after `renderKeyInsightsHero(data)`.

Loading placeholder: HTML is pre-rendered in the `<div>` tag (pulse dot + "Loading forecasts…") so the strip is never blank before the API responds. On API error the placeholder stays; the function is not called in the catch block (strip remains showing loading state, which is acceptable for a supplementary chip row).

#### `renderSentimentInsightStrip()`
Data fields read:

| Chip | Field(s) |
|------|----------|
| Last 7d composition | `sentimentData[*].positive_prob`, `.negative_prob`; `neutral_prob` used if any entry carries it, otherwise inferred as `max(0, 100 - pos - neg)` |
| Coverage | Checked in priority order: `article_count`, `count`, `n`, `num_articles`, `volume` — **none of these fields is present in the current `sentimentData` schema**, so the coverage chip is cleanly omitted |

**Sentiment article-count field status:** Absent. No count field was found on any `sentimentData` entry. The coverage chip is not rendered. A code comment in `renderSentimentInsightStrip` marks the detection list so the chip can be wired later by adding the field in the Django view.

Thin-coverage threshold: `< 5` articles over 7 days triggers an amber "thin coverage" pill. This threshold reflects that CPO news is sparse; fewer than one article per weekday over a week is practically no signal.

Guard: `sd.length < 3` renders "Not enough sentiment data." (same wording pattern as existing hero blocks).

### Role split (hero tiles vs caption strips)
- Hero tiles stay **qualitative**: regime label + duration, sentiment direction arrow, feature-comparison verdict, DA headline.
- Caption strips carry **specific numbers**: MYR forecast value, ±MAPE, direction count, trend %, composition percentages.
- Regime label and rising/falling direction arrow are NOT repeated in the caption strips.

---

## Task B — Key Insights hero strip relocated to sidebar

### Markup moved
The entire `<section id="key-insights">` block was removed from the top of the `lg:col-span-3` main column and appended as the last child of `<aside class="lg:col-span-1">`.

All inner ids are preserved intact:
`key-insights`, `ki-hero-price`, `ki-hero-sentiment`, `ki-hero-comparison`, `ki-hero-metrics`, `ki-price`, `ki-sentiment`, `ki-comparison`, `ki-metrics`.

### Grid classes adjusted for narrow sidebar
| Element | Old classes | New classes |
|---------|------------|-------------|
| Section padding | `p-6` | `p-5` |
| Hero tile grid | `grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-4` | `grid-cols-1 2xl:grid-cols-2 gap-3` |
| Hero tile padding | `p-4` | `p-3` |
| Hero tile label font | `text-[11px]` | `text-[10px]` |
| Details grid | `grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-5` | `grid-cols-1 gap-4` |
| Details `mt-5` | `mt-5` | `mt-4` |
| Header subtitle | `text-[11px] … Computed in-browser · no model re-run` | `text-[10px] … no model re-run` (condensed) |
| Body copy | `mb-4` "The one thing to read from each chart." | `mb-3` "One takeaway per chart." |

### Column balance decision
The sidebar already had four cards (Price Stats, Forecast Models, Feature Set, Best Model/Metrics) consolidated into one grouped block. Adding the four hero tiles does increase sidebar height, but the main column now carries two chart sections and a comparison table — these are tall enough at `lg+` that the columns remain roughly balanced. No further layout adjustment was needed; the sidebar's `space-y-4` spacing keeps the stacking compact.

The 2-column mini-grid inside `#key-insights` at `2xl+` (`2xl:grid-cols-2`) is available if a very wide viewport is used, but at typical `lg`/`xl` laptop widths the tiles stack vertically, which is fine for the thesis defence environment.

---

## Verification notes
- All required ids (`priceChart`, `sentimentChart`, `fan-mode-toggle`, `horizon-legend`, `.horizon-toggle`, `winners-summary`, `forecasts-status`, `metrics-table`, `metric-select`, `key-insights`, `ki-*`) are intact.
- No new external libraries introduced.
- No inputs that mutate model state or re-run inference were added.
- `renderSentimentInsightStrip` is called from both branches of the sentiment chart construction block (data-present and data-absent), ensuring the strip always renders.
- `renderPriceInsightStrip` only reads `lastForecasts`/`forward_fan`; it never touches `fanMode` or triggers any model call.
