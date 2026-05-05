from django.urls import path
from . import views

urlpatterns = [
    # Main pages
    path('', views.dashboard, name='dashboard'),
    path('news/', views.news, name='news'),
    path('about/', views.about, name='about'),

    # Prediction API (called by dashboard JS)
    path('api/prediction/', views.prediction_api, name='prediction_api'),
]
