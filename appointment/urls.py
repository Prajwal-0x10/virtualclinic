from django.contrib import admin
from django.urls import path
from django.contrib.auth import views as auth_views
from . import views

app_name = "appointment"

urlpatterns = [

    path("delete_appointment/<int:pk>", views.delete_appointment, name="delete_appointment"),
    path("prescription/<int:pk>",       views.prescription, name="prescription"),
    path("update_prescription/<int:pk>",views.update_prescription, name="update_prescription"),
    path("appointments_list/",          views.appointments_list, name="appointments_list"),
    #path("appointment_book/",           views.appointment_book, name="appointment_book"),
    path("prescription_list/",          views.prescription_list, name="prescription_list"),
    path("medical_history/",            views.medical_history, name="medical_history"),
    path("book_an_appointment/",        views.book_an_appointment, name="book_an_appointment"),
    
    path('predict',                     views.predict, name="predict"),
    path('result/',                     views.result,  name="result" ),
]
