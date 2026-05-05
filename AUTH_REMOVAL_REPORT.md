# Auth Removal Report — 2026-05-05

## Summary

- **Branch:** `cleanup/remove-auth`
- **Total commits:** 7 (atomic, on this branch)
- **Files archived:** 3 (in `_archive_auth_removal/`)
- **Files modified:** 6 (views.py, urls.py, settings.py, base.html, admin.py, diagrams/sequence.md) + ARCHITECTURE.md
- **Lines of code removed (net):** ~276 net deletions across 9 files (43 insertions, 319 deletions per `git diff --stat main..HEAD`)

## Commits (in order)

```
16df71a chore(archive): move login/register templates to _archive_auth_removal/ — public-facing scope
6ce5b8f chore(archive): move auth_backend.py to _archive_auth_removal/ — public-facing scope
d212890 refactor(views): remove auth views and decorators
91b0260 refactor(urls): remove auth URL patterns
3474024 refactor(settings): remove custom auth backend config
d2416fc refactor(template): simplify base.html navbar to 3 public links
d7afdb4 docs(diagrams,admin): drop auth-flow sequence diagrams and stale comment
```

(Phase 7 doc updates — this report and ARCHITECTURE.md changes — will be a final commit if you approve.)

## Archived Files (`_archive_auth_removal/website/...`)

- `web/auth_backend.py` — custom Firestore auth backend (FirestoreUser, AnonymousUser, authenticate, create_user, get_user, login, logout, FirestoreAuthMiddleware, user_context_processor, firestore_login_required).
- `web/templates/login.html`
- `web/templates/register.html`

All moves were done via `git mv` so file history is preserved.

## Modified Files

| File | Net change |
|------|-----------:|
| `website/web/views.py`              | −82 lines (dropped login_view/register_view/logout_view, 3 `@firestore_login_required` decorators, auth_backend imports, redirect/require_http_methods imports) |
| `website/web/urls.py`               | −5 lines (dropped login/logout/register URL patterns) |
| `website/config/settings.py`        | −5 net lines (dropped FirestoreAuthMiddleware, user_context_processor, LOGIN_URL/LOGIN_REDIRECT_URL/LOGOUT_REDIRECT_URL; added "auth removed 2026-05-05" comment) |
| `website/web/templates/base.html`   | −19 lines (dropped Hi/Logout/Login UI, removed login/register-page wrapper conditional) |
| `website/web/admin.py`              | comment now reads `# Django admin not used — public-facing read-only site.` |
| `diagrams/sequence.md`              | −165 net lines (removed sections 1 + 2 user flows; renumbered 3→1, 4→2; dropped Auth Middleware nodes from prediction API diagram) |
| `ARCHITECTURE.md`                   | Updated Scope, Module layout, Firestore collections table (`users` marked legacy), and added a "What changed in 2026-05-05" section. |

## Sanity Tests (Phase 6)

| Test | Expected | Actual | Status |
|------|----------|--------|--------|
| `python manage.py check` | "no issues" | `System check identified no issues (0 silenced).` | ✅ PASS |
| Import check `from web import views` | imports OK | `imports OK; Views: ['about', 'dashboard', 'datetime', 'news', 'prediction_api', 'render', 'timedelta']` | ✅ PASS |
| URL pattern enumeration | only dashboard/news/about/prediction_api | `URL names: ['about', 'dashboard', 'news', 'prediction_api']` | ✅ PASS |
| `python manage.py runserver 8001` | starts cleanly | started without errors | ✅ PASS |
| HTTP `GET /`        | 200 | **200** | ✅ PASS |
| HTTP `GET /news/`   | 200 | **200** | ✅ PASS |
| HTTP `GET /about/`  | 200 | **200** | ✅ PASS |
| HTTP `GET /login/`  | 404 | **404** | ✅ PASS |
| HTTP `GET /register/` | 404 | **404** | ✅ PASS |
| HTTP `GET /logout/` | 404 | **404** | ✅ PASS |

All sanity tests passed on first attempt — no rollback or retries.

## Issues Encountered

None. The cleanup was straightforward because:
- No `forms.py` to update.
- No project-level `static/` files for login/register (Tailwind / Chart.js are CDN-hosted).
- No external code reads the `users` Firestore collection — only `auth_backend.py` did.
- `_empty_dashboard_ctx()` did not embed `request.user`.

## Things to Verify Manually

- Open the dev server (`cd website && python manage.py runserver`) and visit `/`, `/news/`, `/about/` to confirm the navbar shows only **CPO Predictor / Dashboard / News / About** — no Hi/Logout/Login buttons.
- Visit `/login/`, `/logout/`, `/register/` — should be standard Django 404s.
- Confirm the Firestore `users` collection still exists in the GCP console (data preserved per constraint #4).
- Click "Get Prediction" on the dashboard to confirm the `/api/prediction/` endpoint still serves predictions for `xgboost_{base,csa}_Daily_h{1..7}`.

## Out-of-scope items left untouched (per constraints)

- `scheduler/`, `prediction/`, `markov/`, `news/`, `cpo/` — ML pipeline code.
- `requirements.txt` — Django auth machinery is built-in; nothing to remove.
- `Dockerfile`, `firebase-credentials.json`, `.env*`.
- `_archive_before_cleanup/` — left fully alone.
- Firestore `users` collection — data preserved.

## Rollback Path

If anything breaks in production:

```bash
# Reset to the last good main:
git fetch origin
git reset --hard origin/main

# Or, if a tag exists for pre-removal state:
git reset --hard pre-auth-removal-2026-05-05
```

Each of the 7 atomic commits is also individually revertable with
`git revert <sha>` if you only need to roll back one piece (e.g. only
the navbar change while keeping the backend cleanup).
