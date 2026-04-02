"""
URL Configuration - Function-Based Views
========================================
All views are now standard Python functions (not classes).
Routes are simple and explicit, matching the procedural programming style.
"""
from django.urls import path
from django.contrib.auth import views as auth_views
from . import views

urlpatterns = [
    # Main Views (Dashboard, News, About)
    path('', views.dashboard, name='dashboard'),
    path('news/', views.news, name='news'),
    path('about/', views.about, name='about'),
    
    # Admin Features (Price Upload with HMM Calculation)
    path('admin/upload-price/', views.admin_upload_price, name='admin_upload_price'),
    
    # Machine Learning Features (Prediction)
    path('predict/', views.predict_price, name='predict_price'),
    
    # Authentication URLs
    path('login/', auth_views.LoginView.as_view(template_name='login.html'), name='login'),
    path('logout/', auth_views.LogoutView.as_view(next_page='dashboard'), name='logout'),
    path('register/', views.register, name='register'),
]
