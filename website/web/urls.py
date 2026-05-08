from django.urls import path
from . import views

urlpatterns = [
    # Main pages
    path('', views.dashboard, name='dashboard'),
    path('news/', views.news, name='news'),
    path('about/', views.about, name='about'),

    # Forecasts API (called by dashboard JS — rolling forecast trails)
    path('api/forecasts/', views.forecasts_api, name='forecasts_api'),
]
