from django.shortcuts import render, redirect
from .models import *
from .forms import *
from django.contrib.auth.decorators import login_required



@login_required(login_url='/login/')
def UpdatedUserProfile(request):
    user=request.user
    profile = DoctorProfile.objects.get(user=user)
    if request.method == 'POST':
        form = DoctorUpdateForm(request.POST, instance=profile)
        if form.is_valid():
            form.save()
            return redirect('user_profile:profile')
    else:
        form = DoctorUpdateForm(instance=profile)
    return render(request, 'user_profile/profile.html', {'form': form})



@login_required(login_url='/login/')
def UpdatePatientProfile(request):
    user=request.user
    profile = PatientProfile.objects.get(user=user)
    if request.method == 'POST':
        form = PatientUpdateForm(request.POST, instance=profile)
        if form.is_valid():
            form.save()
            return redirect('user_profile:patient-profile')
    else:
        form = PatientUpdateForm(instance=profile)
    return render(request, 'user_profile/patientprofile.html', {'form': form})


def patient_profile_view(request):
    get_user=DoctorProfile.objects.get(request.user)
    return render(request, 'user_profile/patient_profile_view.html')    

