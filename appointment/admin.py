from django.contrib import admin
from .models import *
# Register your models here.


class OrderAdmin(admin.ModelAdmin):
    list_display = ('patient', 'doctor', 'book_time','appointment_time','book_date','symptom','department')
    default_fields = ('appointment_time')
    default_fieldset =('appointment_time')

admin.site.register(Appointment, OrderAdmin)


admin.site.register(Predict)
admin.site.register(Prescription)

