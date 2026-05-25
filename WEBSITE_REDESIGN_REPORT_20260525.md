# Website Front-End Redesign Report — 2026-05-25

## Summary
- Branch: `fix/critical-major-cleanup`
- Scope: visual redesign only — typography, colour discipline, surface depth,
  interactive states, accessibility, semantics. No tech-stack change, no build
  step added, no framework migration, no view/data-flow change.
- Files modified (4): `website/web/templates/base.html`,
  `website/web/templates/dashboard.html`,
  `website/web/templates/news.html`,
  `website/web/templates/about.html`.
- Files NOT touched: `website/web/views.py`, any Python code, any settings, any
  Firestore schema. No new JS/CSS dependency beyond a Google Fonts `<link>`.
- Tailwind delivery is still the **Play CDN**; config is still inline in
  `base.html`. No `package.json`, no PostCSS, no Tailwind CLI.

## Sanity tests
| Test                                                          | Result |
|---------------------------------------------------------------|--------|
| `python manage.py check`                                      | PASS — 0 issues |
| `get_template(...)` parse for all 4 templates                 | PASS — base 10 612 / dashboard 64 528 / news 14 753 / about 7 027 chars |
| `runserver 8765 --noreload`, GET `/`                          | 200 (92 322 b) |
| `runserver 8765 --noreload`, GET `/news/`                     | 200 (39 392 b) |
| `runserver 8765 --noreload`, GET `/about/`                    | 200 (16 428 b) |
| All 17 dashboard load-bearing `id`s present in rendered HTML  | PASS (priceChart, sentimentChart, fan-mode-toggle, horizon-legend, winners-summary, ki-price/sentiment/comparison/metrics, ki-hero-price/sentiment/comparison/metrics, forecasts-status, metrics-table, metric-select, key-insights) |
| `#mobile-menu` on all three pages                             | PASS |
| `.tip` class + `data-tip` attribute preserved                 | PASS |

## Hard-constraint compliance (§2 of the brief)
- [x] All JS-referenced DOM `id`s preserved (verified via grep on rendered HTML).
- [x] `.tip` class name + `data-tip` contract preserved; only the visual style
      was tightened (radius, shadow, transition).
- [x] Django template syntax preserved: `{% extends %}`, `{% block %}`,
      `{% url %}`, `{{ ... }}`, filters (`floatformat`, `safe`, `escapejs`,
      `default`, `lower`), and all `{% if %}` / `{% for %}` (pagination markup
      in `news.html` is unchanged in semantics).
- [x] Data flow untouched: `JSON.parse('{{ chart_data|safe }}')`,
      `JSON.parse('{{ winners_data|safe }}')`,
      `JSON.parse('{{ sentiment_data|safe }}')`,
      `'{{ latest_date|escapejs }}'` — same shape, same source.
- [x] JS string-literal Tailwind classes audited and reconciled (see token
      table below). `accent-primary`, `focus-visible:ring-primary`,
      `text-primary` all resolve via the inline `tailwind.config`.
- [x] Tailwind still on the Play CDN. New `Outfit` + `JetBrains Mono` loaded
      via `<link>` from Google Fonts and wired through
      `tailwind.config.theme.extend.fontFamily`. No build step.
- [x] Chart.js construction unchanged — restyle is done through `options` only
      (font family, tooltip cornerRadius/displayColors, dataset colours,
      `animation` block).

---

## File 1 — `website/web/templates/base.html`

### What changed
**Head**
- Added: meta description, theme-color, full OG tag set
  (`og:title`, `og:description`, `og:type`, `og:image`), Twitter Card.
- Added: inline-SVG **favicon** (teal rounded square with an ascending line
  chart). Uses a `data:image/svg+xml;utf8,…` URL — no new file written.
- Added: Google Fonts `<link>` for **Outfit** (400/500/600/700) and
  **JetBrains Mono** (500/600), preceded by `preconnect` to both Google
  Fonts origins for a fast paint.
- Replaced `<title>` separator " - " → " · " (Unicode middle dot).

**Tailwind config (inline `<script>`)**
- Added `theme.extend.fontFamily.sans = ['Outfit', ui-sans-serif, system-ui, …]`.
- Added `theme.extend.fontFamily.mono = ['"JetBrains Mono"', ui-monospace, …]`.
- Kept all colour tokens (`primary`, `secondary`, `accent`, `bullish`,
  `bearish`, `neutral`) and **explicitly annotated** the `neutral` token as
  "matches the chart band and legend swatch; amber is reserved for warnings."
- Added `theme.extend.boxShadow.card` and `.card-hover` for tinted
  (slate-toned) shadows in place of pure-black ones.

**Body / layout**
- `min-h-screen` → `min-h-dvh` (fixes the iOS Safari viewport jump).
- Inline `<style>` adds `-webkit-font-smoothing`, `font-feature-settings`,
  and a `:focus-visible { outline: none }` reset (Tailwind's `focus-visible`
  ring takes over).

**Nav**
- Brand: now a flex row with an inline-SVG mini chart inside a 28 px teal
  rounded square — gives the page an actual mark. Brand has its own
  `focus-visible` ring.
- Nav links: added `rounded-md`, `hover:bg-slate-50`, and a 150 ms
  `transition-colors`. Each link has `focus-visible:ring-2 ring-primary`.
- Nav background: `bg-white/85 backdrop-blur` for a sticky-header feel
  without being heavy.

**Mobile menu button**
- Replaced raw `&#9776;` glyph with **two inline SVGs** (hamburger + close)
  that swap on click via an inline JS expression on the button. The handler
  also flips `aria-expanded`.
- Added `aria-controls="mobile-menu"`, `aria-expanded="false"`,
  `aria-label`, `focus-visible:ring-2 ring-primary ring-offset-2`,
  `hover:bg-slate-100`, and a 150 ms transition.

**Footer**
- Two-column footer (`flex … justify-between`): left = copyright, right =
  the new tagline "Daily-scheduled pipeline · HMM × FinBERT × XGBoost".
  Replaces the centred single line.

### Before/after rationale (per category)
- **A. Typography:** Outfit (a friendly geometric grotesk) replaces the
  default Tailwind sans-stack. Outfit has more character than the
  Inter/system fallback and reads well in headings.
- **B. Colour & surfaces:** kept the single teal accent; introduced
  `shadow-card` / `shadow-card-hover` so card shadows are slate-tinted
  instead of pure-black (subtler at low contrast).
- **C. Interactive states:** every interactive element in the chrome (brand
  link, nav links, mobile-menu button, mobile-menu links) gained a
  `focus-visible:ring-2 ring-primary` and a 150 ms transition.
- **D. Layout & spacing:** `min-h-dvh` fixes mobile viewport drift.
- **E. Accessibility / meta:** favicon, meta description, OG tags, ARIA
  attributes on the menu button, proper hamburger/close icon.

---

## File 2 — `website/web/templates/dashboard.html` (the biggest delta)

### What changed
**Inline `<style>` block**
- `.tip` polished — slightly larger max-width (240 px), softer radius
  (`0.5rem`), wider shadow, 4 px slide-in transform on hover, longer fade
  (140 ms). Class name and `data-tip` attribute contract preserved.
- New `.skeleton` shimmer (linear-gradient, `shimmer` keyframes, 1.4 s loop)
  and `.chart-loading` overlay rule (absolute-positioned, fades out via
  `.is-done` class) — used in three places (price chart, sentiment chart,
  metrics table).
- New `.eyebrow` utility — small medium-weight slate label, replaces the
  blanket `uppercase tracking-wide` treatment on sidebar/breakdown
  subheaders.
- New `.metric-select` styles — `appearance: none` plus an inline-SVG
  chevron, so the native `<select>` matches the design.
- New `.horizon-toggle` rule — 180 ms transitions plus a 1 px lift on hover,
  and a 2 px `outline` `focus-visible` ring (works on `<button>`).
- New `.stat-grid > * + *` rule — left-border divider for the consolidated
  best-model card.

**Header**
- `<div class="mb-6">` → `<header class="mb-7">`.
- Title moved to `text-slate-900` and an `h1.page-title { letter-spacing: -0.018em }` class.
- **Subtitle copy:** "Real-time commodity price analysis…" → "**Daily**
  commodity price analysis with Hidden Markov Model regime detection." (§F
  content honesty fix).
- Footer metadata row uses `tabular-nums` on the date and the day count.

**Sidebar (`<aside aria-label="Dashboard summary">`)**
- Wrapped the sidebar `<div>` in `<aside>`. Each panel inside is now
  `<section>` (a11y landmark separation).
- All four sidebar cards: `rounded-lg` → `rounded-xl`, `shadow-sm` →
  `shadow-card`, sub-headers use `.eyebrow` instead of
  `uppercase tracking-wide`.
- **Price Stats** — all four prices use `font-mono tabular-nums` so digits
  don't jitter when values update.
- **Forecast Models** — copy unchanged; subhead uses `.eyebrow`.
- **Feature Set** — table gets `tabular-nums`; subhead uses `.eyebrow`.
- **Best Model + MAPE + sMAPE + Directional Acc.** — **consolidated** from
  4 separate cards into **one** card with a 3-column `divide-x` grid (the
  `.stat-grid` rule). MAPE / sMAPE / Dir Acc each have a small grey
  uppercase label and a large `font-mono tabular-nums` numeric value with a
  smaller `%` suffix.

**Main content**
- Each major panel changed from `<div>` to `<section>`.
- All outer cards: `rounded-lg` → `rounded-xl`, `shadow-sm` → `shadow-card`.
- **Key Insights:** subtle right-aligned "Computed in-browser · no model
  re-run" eyebrow. Hero cards now use `bg-gradient-to-br from-slate-50
  to-white` instead of flat `bg-slate-50`, giving the hero strip slightly
  more presence than the four panels below it.
- **Key Insights collapsible "Details":** subhead uses `.eyebrow`. The
  `<summary>` row gets a chevron that flips on `details[open]` via
  `group-open:rotate-180`.
- **Price chart:** the bare "Loading forecasts…" text is now a small pill
  with an `animate-pulse` slate dot; on success it switches to a green dot
  + "Forecasts updated · N horizons" (with `tabular-nums` on the N); on
  failure it switches to a red dot + the error. The fan-mode `<label>`
  gained `focus-within:ring-2 ring-primary` so keyboard focus on the
  underlying checkbox is visible.
- **Price chart canvas:** wrapped in `relative`, with an
  `#priceChart-skeleton.chart-loading.skeleton` overlay that fades out
  after Chart.js paints (called via `dismissSkeleton('priceChart-skeleton')`).
- **Sentiment chart:** identical skeleton treatment with
  `#sentimentChart-skeleton`.
- **Configuration Comparison:** the `<select>` got a proper id-linked
  `<label>`, the new `.metric-select` styling (chevron + padding),
  `hover:border-slate-300`, and `focus-visible:ring-2 ring-primary
  focus-visible:border-primary`.
- **Metrics-table container** now ships **pre-populated with a 5-row
  skeleton** so the panel doesn't render empty between server response
  and JS execution.

**Chart.js options (no constructor change)**
- `Chart.defaults.font.family = "'Outfit', …"` set once globally so all
  charts inherit the new font.
- Price chart `borderColor` `#3b82f6` (blue) → `#0f766e` (primary teal) for
  brand cohesion (the previous blue clashed with the other 7 horizon
  colours).
- Price tooltip / sentiment tooltip: `cornerRadius: 6`, `boxPadding: 4`,
  `displayColors: true`, slightly heavier title weight. Same call sites,
  same callbacks.
- Added `animation: { duration: 600, easing: 'easeOutQuart' }` to both
  charts.

**JS string literals (reconciled — every class injected at runtime)**
| Where | Before | After |
|-------|--------|-------|
| `kiLi` value `<span>` | `font-semibold ${toneCls}` | `font-semibold tabular-nums ${toneCls}` |
| Hero **Regime** tone (neutral) | `text-amber-600` | `text-slate-700` |
| Hero **Regime** Neutral chip | `bg-amber-100 text-amber-700` | `bg-slate-200 text-slate-700` |
| Hero **Regime** title | `text-xl font-bold ${tone} mb-3` | `+ tabular-nums` |
| Hero **Regime** stale pill | `rounded` | `rounded-md` + `tabular-nums` on the day count |
| Hero **Regime** "as of" date | plain | wrapped in `<span class="tabular-nums">` |
| Hero **Sentiment** neutral tone | `text-slate-600` | `text-slate-700` |
| Hero **Metrics** head | `text-xl font-bold ${tone}` | `+ tabular-nums` |
| Hero **Metrics** legend row | plain | `+ tabular-nums` on the wrapper |
| `renderHorizonLegend` button | `rounded border` `px-1.5 py-0.5` | `rounded-md border` `px-2 py-1` `text-[11px] tabular-nums` + `aria-pressed` |
| `renderWinnersSummary` row | text-slate-600 | `+ tabular-nums` (h-label and config) |
| `renderMetricsTable` `<table>` | `text-xs` | `text-xs tabular-nums` |
| `renderMetricsTable` header cells | `font-medium` | `font-semibold text-[10px]` |
| `renderMetricsTable` body row | `border-t` | `+ hover:bg-slate-50/60 transition-colors` |
| `renderMetricsTable` cell | `text-center` | `+ font-mono`, cellCls promoted from `text-teal-700 bg-teal-50` → `+ rounded-md` for winner, `text-slate-600` for non-winner |
| `loadForecasts` status (ok) | plain text | `<span class="bg-green-500 rounded-full">` + `tabular-nums` on N |
| `loadForecasts` status (err) | plain red text | `<span class="bg-red-500 rounded-full">` + same flex layout |

**`stateColors` (Chart.js annotations)**
- `2: 'rgba(148,163,184,0.12)'` → `2: 'rgba(148,163,184,0.16)'` (slightly
  more visible). Stays slate — see §B fix below.

### Before/after rationale (per category)
- **A. Typography:** Outfit pulled in via base; every numeric value now has
  `tabular-nums` (or `font-mono tabular-nums`), so the sidebar prices, the
  best-model card's MAPE/sMAPE/Dir-Acc, the metrics-table cells, the
  horizon-toggle labels, the hero "1 day / 7 day" DA labels, and the
  in-status counter all align column-wise and stop jittering when refetched.
  The blanket `uppercase tracking-wide` subheaders are gone — replaced with
  a single `.eyebrow` treatment (small medium-weight slate). Page `<h1>`
  picks up `tracking-tight`.
- **B. Colour & surfaces:** **Neutral state colour resolved to slate.** The
  Tailwind `neutral` token was already slate, the chart band was already
  slate, and the bottom-of-chart legend swatch was already
  `bg-slate-200` — only the hero tile diverged into amber. Switched the
  hero tile (Neutral chip + Neutral tone) to slate so the meaning lines up
  across the page. Amber is now used **only** on actual warnings (the
  staleness pill and the near-coin-flip DA marker), which makes the colour
  read consistently as "caution / watch this." Card outer radius bumped
  from `rounded-lg` to `rounded-xl`; inner chips/badges stay tighter
  (`rounded-md`); shadows tinted slate via the new `shadow-card`. The four
  monotonous sidebar metric cards are now one cohesive block.
- **C. Interactive states:** `metric-select`, `fan-mode-toggle` (via
  `focus-within`), every `.horizon-toggle`, and the mobile-menu button all
  show a 2 px teal `focus-visible` ring. Hovers on cards, horizon toggles,
  and metrics-table rows get 150–220 ms transitions. The chart load state
  uses a real skeleton shimmer, and the forecast status pill switches
  between a pulsing slate dot, a green dot, or a red dot depending on
  state.
- **D. Layout & spacing:** sidebar is now `<aside>`; major panels are
  `<section>`. The header rhythm is tightened (`mb-7`), the hero strip
  vs. detail strip is more clearly differentiated (gradient surface).
- **E. Accessibility / semantics:** new landmarks, `aria-pressed` on the
  horizon toggle buttons, label-id pairing for the metric `<select>`.
- **F. Content honesty:** "Real-time" → "Daily" (subtitle line 49 in the
  original).

---

## File 3 — `website/web/templates/news.html`

### What changed
- Header: `<div>` → `<header>`; h1 `text-slate-800` → `text-slate-900
  tracking-tight`. Coverage dates use `tabular-nums`. External-link `<a>`
  gets a `focus-visible:ring-2` ring.
- "About sentiment analysis" panel: turned into a `<section>` with
  `aria-labelledby`. The three sentiment classes are now coloured tinted
  cards (`bg-green-50/60`, `bg-red-50/60`, `bg-slate-50`) so the legend
  reads at a glance, instead of three identical flat columns.
- Sentiment stat counts: **consolidated** from 4 separate cards into one
  `<section>` with internal `divide-x divide-y` borders. Numbers gain
  `font-mono tabular-nums` and a hue per class (green / red / slate / slate).
- Filter bar: `<section>` with `aria-label`. Active state gets a
  `shadow-sm`. Every filter pill has a coloured `focus-visible` ring keyed
  to its sentiment (`ring-green-500`, `ring-red-500`, `ring-slate-500`,
  `ring-primary`).
- News cards (`<div>` → `<article>`): `rounded-lg shadow-sm` →
  `rounded-xl shadow-card`; hover lift via `hover:shadow-card-hover` +
  `hover:border-slate-200` + `transition-all duration-200`. Card title is
  `text-slate-900` and the snippet uses `flex-grow` so footers line up.
  Sentiment badge now has a small coloured dot. Date uses `tabular-nums`.
  The "Read original →" arrow nudges right on hover via `group-hover:translate-x-0.5`.
- Empty-state card: `bg-slate-50` → `bg-white` `shadow-card` for visual
  consistency with the rest of the page.
- Pagination: all numeric cells use `tabular-nums`; current page uses
  `aria-current="page"`; every link has a `focus-visible:ring-2 ring-primary`
  ring and a 150 ms colour transition; the disabled `<span>`s are marked
  `aria-hidden="true"`.

---

## File 4 — `website/web/templates/about.html`

### What changed
- Header: `<div>` → `<header>`; added a small "Skripsi project" eyebrow
  above the page title; h1 → `text-slate-900 tracking-tight` at
  `sm:text-4xl`.
- Three feature cards (`<div>` → `<article>`): icon container `rounded-full`
  → `rounded-xl` (matches the design language elsewhere); each card gains
  `shadow-card`, `hover:shadow-card-hover`, and a 200 ms
  `transition-shadow`. Every decorative SVG is wrapped (and self-marked)
  with `aria-hidden="true"`. h3 → h2 (under the section landmark).
- Data sources card: numbered circular badges marked `aria-hidden="true"`,
  numbers use `tabular-nums`.
- Pipeline list: every `<span class="… rounded-full">` bullet marked
  `aria-hidden="true"`.
- Author card: `<div>` → `<section aria-label="Author">`. Author avatar is
  now `bg-gradient-to-br from-teal-100 to-teal-50 ring-1 ring-teal-100
  text-primary` (teal-on-teal) instead of `bg-slate-100 text-slate-400` —
  it now reads as a brand element rather than a missing image. h3 → h2.

---

## Tailwind tokens edited (auditable list)
| Token | Where defined | Status |
|-------|---------------|--------|
| `theme.extend.fontFamily.sans` | `base.html` `<script>` | **Added** — Outfit + system fallbacks |
| `theme.extend.fontFamily.mono` | `base.html` `<script>` | **Added** — JetBrains Mono + system fallbacks |
| `theme.extend.boxShadow.card` | `base.html` `<script>` | **Added** — slate-tinted card shadow |
| `theme.extend.boxShadow.card-hover` | `base.html` `<script>` | **Added** — slate-tinted hover shadow |
| `colors.primary` | `base.html` `<script>` | unchanged (`#0f766e`) |
| `colors.bullish` | `base.html` `<script>` | unchanged (`#22c55e`) |
| `colors.bearish` | `base.html` `<script>` | unchanged (`#ef4444`) |
| `colors.neutral` | `base.html` `<script>` | unchanged (`#94a3b8`); **JS callers updated** to stop using amber for this state — see hero tile + tone changes above |
| `colors.accent` | `base.html` `<script>` | unchanged |
| `colors.secondary` | `base.html` `<script>` | unchanged |

## Things to verify manually (browser)
1. Open `/`, `/news/`, `/about/` and confirm the new Outfit/JetBrains Mono
   render (Network tab → fonts.googleapis.com hit; FOUT acceptable).
2. Dashboard `priceChart` should briefly show a shimmer, then the chart
   draws (close-price line is now teal, regime bands unchanged).
3. Toggle the **horizon legend** buttons — trails appear/disappear; the
   button gets a 1 px hover lift; `Tab` over them shows a 2 px teal outline.
4. Toggle the **fan mode** checkbox — the focus ring appears on the
   surrounding label via `focus-within`.
5. Click the `metric-select` dropdown — chevron is part of the control,
   focus shows a teal ring + border.
6. Hover a row in the **Configuration Comparison** table — subtle slate
   wash. Winner cells are teal with a `rounded-md` highlight.
7. Hover the `.tip` markers (Feature Set headers, sidebar h=… rows, state
   legend, horizon toggles) — tooltip slides in with the new style.
8. Narrow to <768 px — nav collapses, hamburger SVG swaps to an X on tap,
   `aria-expanded` flips.
9. `/news/`: filter pills get their per-sentiment focus rings on `Tab`. The
   stat counts (positive / negative / neutral / total) align on the
   monospaced numerals.
10. `/about/`: brand author avatar is a teal-on-teal "M" instead of grey.
11. Lighthouse / axe should now find: meta description, theme-color, proper
    `<main>` `<nav>` `<header>` `<footer>` `<aside>` `<section>` landmarks
    on the dashboard, decorative SVGs marked `aria-hidden`.

## Compliance checklist
- [x] Only the 4 in-scope templates were modified.
- [x] No new JS/CSS library beyond a `<link>` to Google Fonts.
- [x] Tailwind stays on the Play CDN; config stays inline.
- [x] Chart.js construction unchanged — only `options` were tuned.
- [x] All DOM `id`s the JS depends on are still present (curl + grep verified).
- [x] `.tip` class + `data-tip` attribute contract preserved.
- [x] Every JS-injected Tailwind class reconciled against the (extended)
      config — `accent-primary`, `focus-visible:ring-primary`, `text-primary`
      all resolve.
- [x] No view code, no Firestore touched.
- [x] `manage.py check` PASS; all three URLs return 200; template loader
      parses all 4 files.
