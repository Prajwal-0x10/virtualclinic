from django.contrib import admin
from django.urls import path
from django.contrib.auth import views as auth_views
from . import views

app_name = "user_profile"

urlpatterns = [
    path("profile/", views.UpdatedUserProfile, name="profile"),
    path("patient_profile_view/", views.patient_profile_view, name="patient_profile_view"),
    path("UpdatePatientProfile/", views.UpdatePatientProfile, name="patient-profile"),
  
]
