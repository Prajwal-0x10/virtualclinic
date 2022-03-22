from django.db import models
from user_profile.models import *
from account.models import User
from django.utils import timezone
# Create your models here.
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


class Appointment(models.Model):
    book_date=models.DateField(default=timezone.now,null=True)
    book_time=models.TimeField(default=timezone.now,null=True)
    appointment_time=models.TimeField(default=timezone.now,null=True)
    symptom = models.CharField(max_length=200,default="routine checkup")
    patient = models.ForeignKey(PatientProfile, on_delete=models.CASCADE)
    doctor = models.ForeignKey(DoctorProfile, on_delete=models.CASCADE)
    department= models.CharField(max_length=50,choices=departments,default='Cardiologist')
    prescription_added = models.BooleanField(default=False)
    




class Prescription(models.Model):
    appointment = models.ForeignKey(Appointment, on_delete=models.SET_NULL , null=True,blank=True)
    patient = models.ForeignKey(PatientProfile, on_delete=models.CASCADE)
    doctor = models.ForeignKey(DoctorProfile, on_delete=models.CASCADE)
    date = models.DateTimeField(default=timezone.now,null=True)
    symptoms = models.CharField(max_length=200)
    prescription = models.TextField()

    class Meta:
        ordering = ('-id',)

    def __str__(self):
        return "Presciption Doc-{} Patient-{}".format(self.doctor, self.patient)

PAYMENT_TYPES = [
    ('I','Individual'),
    ('C','Consulting')
]

class Payment(models.Model):
    patient = models.ForeignKey(User, on_delete=models.CASCADE, related_name="patient_payments")
    date = models.DateField(auto_now_add=True)
    paid = models.IntegerField(null=True)
    outstanding = models.IntegerField(null=True)
    total = models.IntegerField(null=True)
    payment_type = models.CharField(choices=PAYMENT_TYPES, max_length=1, default="I")

    class Meta:
        ordering = ('-id',)

    def __str__(self):
        return "Payment Patient-{} Amount-{}".format(self.patient, self.total)


class Predict(models.Model):
    
    patient    = models.ForeignKey(PatientProfile, on_delete=models.CASCADE)

    symptoms = models.TextField(max_length=200,null=True,blank=True)

    # symptoms_1 = models.CharField(max_length=50,null=True,blank=True) 
    # symptoms_2 = models.CharField(max_length=50,null=True,blank=True) 
    # symptoms_3 = models.CharField(max_length=50,null=True,blank=True) 
    # symptoms_4 = models.CharField(max_length=50,null=True,blank=True) 

    Predicted_Disease = models.CharField(max_length=100,null=True,blank=True)

    predict_date=models.DateField(default=timezone.now,null=True)