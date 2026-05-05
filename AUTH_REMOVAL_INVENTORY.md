# Auth Removal Inventory

Generated: 2026-05-05
Branch: `cleanup/remove-auth`
Scope: Convert the Django site to a public-facing read-only dashboard (Dashboard, News, About). Remove all login / register / logout flows and the custom Firestore auth backend.

> Note: `website/venv/` was excluded from search. The site has **no** `forms.py`, no `static/` files, and no `users` collection reads outside `auth_backend.py`.

---

## Files to DELETE (move to `_archive_auth_removal/`)

| File | Reason |
|------|--------|
| [website/web/auth_backend.py](website/web/auth_backend.py) | Custom Firestore auth backend — entire file is auth-only (FirestoreUser, AnonymousUser, authenticate, create_user, get_user, login, logout, FirestoreAuthMiddleware, user_context_processor, firestore_login_required). |
| [website/web/templates/login.html](website/web/templates/login.html) | Login form template — only used by `login_view`. |
| [website/web/templates/register.html](website/web/templates/register.html) | Registration form template — only used by `register_view`. |

**No static CSS/JS files are login/register-only** — Tailwind + Chart.js come from CDN, and there are no project-level static assets to archive.

---

## Files to MODIFY

| File | What to remove (line ranges) |
|------|------------------------------|
| [website/web/views.py](website/web/views.py) | • Lines 15-21: `from .auth_backend import (...)` block.<br>• Line 28: `@firestore_login_required` decorator on `dashboard`.<br>• Line 159: `@firestore_login_required` decorator on `prediction_api`.<br>• Line 211: `@firestore_login_required` decorator on `news`.<br>• Lines 295-358: `login_view`, `register_view`, `logout_view` functions (and `from django.shortcuts import redirect`, `from django.contrib import messages`, `from django.views.decorators.http import require_http_methods` if no longer used).<br>• Line 5 docstring mention of authentication backend. |
| [website/web/urls.py](website/web/urls.py) | • Lines 13-16: `path('login/', ...)`, `path('logout/', ...)`, `path('register/', ...)` and the "Authentication" comment. |
| [website/config/settings.py](website/config/settings.py) | • Line 41: `'web.auth_backend.FirestoreAuthMiddleware'` from `MIDDLEWARE`.<br>• Line 57: `'web.auth_backend.user_context_processor'` from `TEMPLATES.OPTIONS.context_processors`.<br>• Lines 77-82: `LOGIN_URL` / `LOGIN_REDIRECT_URL` / `LOGOUT_REDIRECT_URL` block.<br>• Line 4 docstring mention of "auth lives in Firestore".<br>• Add comment block: `# Auth removed 2026-05-05 — public-facing decision support tool`. |
| [website/web/templates/base.html](website/web/templates/base.html) | • Line 37: `{% if request.resolver_match.url_name != 'login' and ... != 'register' %}` wrapper (and matching `{% endif %}` line 97).<br>• Lines 62-84: entire "Right: User" `<div>` with `{% if user.is_authenticated %}` / Hi/Logout / Login link block — replace with just the mobile-menu hamburger button (or drop the right-side div entirely on desktop and keep only the hamburger).<br>• Keep nav links (Dashboard, News, About) untouched. |
| [website/web/views.py](website/web/views.py) | (same file, called out separately) — also drop unused imports after removal. |

> The remaining views (`dashboard`, `news`, `about`, `prediction_api`) and the `_empty_dashboard_ctx()` helper do not embed `request.user` or `'user': ...` in their render context, so no template-context cleanup is required there.

---

## REVIEW Items (need user approval)

| File / Code | Why ambiguous |
|-------------|---------------|
| `website/config/settings.py` — `SESSION_ENGINE = 'django.contrib.sessions.backends.signed_cookies'` (line 73), `MESSAGE_STORAGE = 'django.contrib.messages.storage.session.SessionStorage'` (line 102), and the `SessionMiddleware` (line 38). | These were added to support the Firestore login/session flow. With auth gone, sessions/messages aren't strictly needed. **However**, removing them is non-trivial: `django.contrib.messages` still uses session storage, and any future "flash message" use would break. **Recommendation:** keep them — they're harmless for read-only pages and removing them is out of scope for "auth removal". Confirm? |
| `website/web/admin.py` — currently a 1-line stub: `# Django admin not used — authentication is Firestore-backed.` | The comment references Firestore auth. The file is otherwise empty. **Recommendation:** update the comment to drop the auth reference (e.g. `# Django admin not used — public-facing read-only site.`). Trivial; including in MODIFY if you approve. |
| `website/web/services.py` (line 1-7 docstring) | Says "All data reads go directly through Firestore in views.py" — no auth reference. No action needed. |
| `diagrams/sequence.md:51` references the `users` collection write. | Out of scope per constraint #3 (`scheduler/**`, `prediction/**`, etc. are out of scope) — but this is a *diagrams* file, not one of those. **Recommendation:** leave untouched in this pass; it's a historical sequence diagram. The user can update docs separately. |
| `ARCHITECTURE.md` exists at project root. | Per Phase 7a, may need an "Authentication" section removed. Will inspect during Phase 7. |
| `views.py` line 5 — docstring: `Authentication uses the custom Firestore backend (no Django ORM).` | Trivial — update or remove the line. Including in MODIFY. |

---

## Statistics

- **Total files identified: 8** (3 DELETE + 4 MODIFY + 1 ambiguous-but-trivial admin.py)
- **DELETE count: 3**
- **MODIFY count: 4** (views.py, urls.py, settings.py, base.html)
- **REVIEW count: 5 items** (most are advisory — actual blocking decisions: SessionMiddleware/SESSION_ENGINE retention, admin.py comment update)

---

## Branch + Pre-conditions Verified

- ✅ Current branch is `cleanup/remove-auth`
- ✅ `_archive_before_cleanup/` exists at project root (will NOT be touched)
- ✅ `_archive_auth_removal/` does not yet exist (will be created in Phase 3)
- ✅ No `users` collection reads outside `auth_backend.py`
- ✅ No project-level static files for login/register
- ✅ No `forms.py` exists

---

## Awaiting your approval to proceed to Phase 2 (REVIEW resolution).
