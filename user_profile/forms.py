from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.db import transaction
from django.db.models import fields
from django.forms.utils import ValidationError

from account.models import  User
from .models import *




# departments=[('Cardiologist','Cardiologist'),
# ('Dermatologists','Dermatologists'),
# ('Emergency Medicine Specialists','Emergency Medicine Specialists'),
# ('Allergists/Immunologists','Allergists/Immunologists'),
# ('Anesthesiologists','Anesthesiologists'),
# ]


# GENDER_CHOICES = [
#     ('M', 'Male'),
#     ('F', 'Female'),
#     ('O', 'Other')
# ]


class DoctorUpdateForm(forms.ModelForm):
    class Meta:
        model=DoctorProfile
        fields='__all__'
        exclude = ('user',)
        # exclude=('user')
class PatientUpdateForm(forms.ModelForm):
    class Meta:
        model=PatientProfile
        fields='__all__'
        exclude = ('user',)
