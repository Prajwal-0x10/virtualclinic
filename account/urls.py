from django.contrib import admin
from django.urls import path
from django.contrib.auth import views as auth_views
from . import views

app_name = "account"

urlpatterns = [
    # path("login/", auth_views.LoginView.as_view(template_name="account/login.html"),name='login'),
    path("do_login/", views.do_login, name='do_login'),
    path("logout/", views.logout, name="logout"),
    path("choose_register/", views.register, name="register"),
    path("pateint_register/", views.pateint_register, name="pateint_register"),
    path("doctor_register/", views.doctor_register, name="doctor_register"),
    
    
    path("", views.home, name="home"),
    path("aboutus/", views.aboutus, name="about-us"),
    path("contact/", views.contact, name="contact"),
]
