from django.db import models
from django.core.validators import RegexValidator
from django.contrib.auth.models import AbstractUser
from django.conf import settings
from django.utils import timezone
from account.models import User
GENDER_CHOICES = [
    ('M', 'Male'),
    ('F', 'Female'),
    ('O', 'Other')
]

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



class PatientProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    gender = models.CharField(choices=GENDER_CHOICES, max_length=1, blank=True)
    age = models.IntegerField(blank=True, null=True)
    phone = models.CharField(max_length=10,null=False)
    email = models.CharField(max_length=100,)
    address = models.CharField(max_length=500, blank=True,null=True)
    blood_group = models.CharField(choices=BLOOD_GROUPS, max_length=3, blank=True)
    # symptom = models.CharField(max_length=200,default="routine checkup")

    class Meta:
        ordering = ('-id',)

    def __str__(self):
        return str(self.user.username)




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

class DoctorProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    gender = models.CharField(choices=GENDER_CHOICES, max_length=1, blank=True)
    mobile = models.CharField(max_length=10,null=False)
    department= models.CharField(max_length=50,choices=departments,default='Cardiologist')
    
    shift_start_time    =models.TimeField(default=timezone.now,null=True)
    shift_end_time      =models.TimeField(default=timezone.now,null=True)

    class Meta:
        ordering = ('-id',)

    def __str__(self):
        return str(self.user.username)



