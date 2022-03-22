from django.contrib.auth import login, logout as deauth
from django.urls import reverse_lazy
from django.views.generic import CreateView
from django.shortcuts import redirect, render
from .forms import * 

from django.contrib.auth import authenticate,logout 
from django.contrib import messages
from user_profile.forms import *
from user_profile.models import *
from datetime import datetime, timedelta,date
def home(request):
    return render(request, 'account/home.html')

def aboutus(request):
    return render(request, 'account/aboutus.html')

def contact(request):
    return render(request, 'account/contact.html')

def register(request):
    return render(request, 'account/choose_register.html' )
    

def do_login(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        user = authenticate(request , username =username , password=password)
        if user is not None:
            login(request , user)
            messages.info(request , 'You are successfully logged')
            return redirect('account:home')
        else:
            messages.info(request , 'Username or Password is incorrect')
            return redirect('account:do_login') 

    if request.user.is_authenticated:
        return redirect('account:home')

    return render(request , "account/login.html")	


def pateint_register(request):
    form=PatientSignUpForm()
    if request.POST:
        form=PatientSignUpForm(request.POST)
        if form.is_valid():
            user=form.save()
             
            age=form.cleaned_data['age']
            address=form.cleaned_data['address']
            gender=form.cleaned_data['gender']
            blood_group=form.cleaned_data['blood_group']
            

            print(age, address, gender, blood_group,user)

            PatientProfile.objects.create(user=user,age=age,
                            address=address,
                            gender=gender,
                            blood_group=blood_group,
                            )
            #form.save()
            messages.success(request,'Your Account Created Succesfully')
            return redirect('account:do_login')
        else:
            messages.success(request,form.errors)
    
        return redirect('account:pateint_register')            

    return render(request , "account/patient_register.html",{'form':form})    


def doctor_register(request):
    form=DoctorSignUpForm()
    if request.POST:
        form=DoctorSignUpForm(request.POST)
        if form.is_valid():
            user=form.save()
            gender=form.cleaned_data['gender']
            mobile=form.cleaned_data['mobile']
            department=form.cleaned_data['department']
            start_time=request.POST.get('start_time')
            end_time=request.POST.get('end_time')
           
            try: 
                DoctorProfile.objects.create(user=user,
                                        mobile=mobile,
                                        department=department,
                                        gender=gender,
                                        shift_end_time=end_time,
                            shift_start_time=start_time)
            except Exception as e:
                print(e)                

            #form.save()
            messages.success(request,'Your Account Created Succesfully')
            return redirect('account:do_login')
        else:
            return render(request , "account/signup.html",{'form':form})    
        
                  
    return render(request , "account/signup.html",{'form':form})


def logout(request):
    if request.user.is_authenticated:
        deauth(request)
        messages.info(request , "You have been suceesfully Logout")
        
        return redirect('account:home')
    return redirect('account:home')    
