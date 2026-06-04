from django.urls import path, re_path
from django.views.generic import RedirectView
from . import views

urlpatterns = [
    # Main pages
    path('', views.dashboard, name='dashboard'),
    path('news/', views.news, name='news'),
    path('about/', views.about, name='about'),

    # Forecasts API (called by dashboard JS — rolling forecast trails)
    path('api/forecasts/', views.forecasts_api, name='forecasts_api'),

    # Catch-all: any unknown route (e.g. /xyz/) redirects to the dashboard
    # instead of showing a 404. Kept last so it only matches paths none of
    # the routes above claimed. permanent=False → 302 so browsers don't cache
    # the redirect for paths that might become real routes later.
    re_path(r'^.*$', RedirectView.as_view(pattern_name='dashboard', permanent=False)),
]
