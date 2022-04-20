from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.db import transaction
from django.forms.utils import ValidationError
from django.utils import timezone
from account.models import  User
from .models import *
departments=[('Cardiologist','Cardiologist'),
('Dermatologists','Dermatologists'),
('Rheumatologist','Rheumatologist'),
('Orthopedist','Orthopedist'),
('ENT_Specialists','ENT_Specialists'),
('Allergists/Immunologists','Allergists/Immunologists'),
('Neurologist','Neurologist'),
('CGastroenterologist','Gastroenterologist'),
('Urologist','Urologist')
]


GENDER_CHOICES = [
    ('M', 'Male'),
    ('F', 'Female'),
    ('O', 'Other')
]


class DoctorSignUpForm(UserCreationForm):
    department=forms.ChoiceField(choices = departments)
    gender=forms.ChoiceField(choices=GENDER_CHOICES)
    mobile=forms.IntegerField()
    # shift_start_time=forms.TimeField()
    # shift_end_time=forms.TimeField()
    class Meta(UserCreationForm.Meta):
        model = User
    
    @transaction.atomic
    def save(self):
        user = super().save(commit=False)
        user.is_doctor = True
        user.save()
        return user


BLOOD_GROUPS = [
    ('O-', 'O-'),
    ('O+', 'O+'),
    ('A-', 'A-'),
    ('A+', 'A+'),
    ('B-', 'B-'),
    ('B+', 'B+'),
    ('AB-', 'AB-'),
    ('AB+', 'AB+'),
]
class PatientSignUpForm(UserCreationForm):
    address=forms.CharField(max_length=500)
    age=forms.IntegerField()
    gender=forms.ChoiceField(choices = GENDER_CHOICES)
    blood_group=forms.ChoiceField(choices = BLOOD_GROUPS)
    phone = forms.IntegerField()
    email = forms.EmailField(max_length=100)
    class Meta(UserCreationForm.Meta):
        model = User
    @transaction.atomic
    def save(self):
        user = super().save(commit=False)
        user.is_patient = True
        user.save()
        return user






# from django.contrib.auth import get_user_model
# from django.contrib.auth.forms import UserCreationForm
# from django import forms

# USER_CHOICES = [
#     ('D', 'Doctor'),
#     ('P', 'Patient')
# ]

# class UserCreateForm(UserCreationForm):
#     user_type = forms.ChoiceField(choices=USER_CHOICES, required=True, widget=forms.RadioSelect)
#     class Meta:
#         fields = ("first_name", "last_name", "username", "email", "password1", "password2", "user_type")
#         model = get_user_model()

#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.fields["username"].label = "Username"
#         self.fields["email"].label = "Email address"