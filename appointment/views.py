from django.shortcuts import render, redirect
from django.urls.conf import re_path
from .models import * 
from .forms import * 
from django.views.generic import ListView, CreateView
from django.urls import reverse_lazy
from django.contrib.auth.decorators import login_required
from django.contrib.auth.mixins import LoginRequiredMixin
from user_profile.models import *
from datetime import datetime, timedelta  
# Create your views here.
from datetime import date
from django.contrib import messages

from sklearn import tree
import numpy as np
import pandas as pd
from django.http import JsonResponse


def appointments_list(request):
    if request.user.is_doctor:
        get_doctor_instance=DoctorProfile.objects.get(user_id=request.user.id)
        apps=Appointment.objects.filter(doctor=get_doctor_instance)
        context={'apps': apps}
        return render(request, 'appointment/appointment_list.html',context)
    get_patient_instance=PatientProfile.objects.get(user_id=request.user.id)
    apps=Appointment.objects.filter(patient=get_patient_instance)
    context={'apps': apps}
    return render(request, 'appointment/appointment_list.html',context)



#############======================================================#################
#############============ Function for book an appointmen =========#################
#############======================================================#################
import pandas as pd
import time, datetime
# new appointment code start from here
def book_an_appointment(request):
    # current_time =  datetime.now().strftime('%H:%M:%S')
    patient=PatientProfile.objects.get(user=request.user) 
    if request.POST:

        form=DepartmentForm(request.POST)
        if form.is_valid():
            department=form.cleaned_data['department']
            symptoms=request.POST.get('symptoms')
            book_date=request.POST.get('book_date')
            

            if book_date=='':
                messages.success(request,'Please Choose a date')
                return redirect('appointment:book_an_appointment')
                
            get_doctors=DoctorProfile.objects.filter(department=department)
            
            # if doctor is available
            if get_doctors:
                for doctor in get_doctors:
                    print(doctor.user.username)
                    get_doctor_instance=DoctorProfile.objects.get(user_id=doctor.user.id)
                    get_shift_start_time=get_doctor_instance.shift_start_time
                    get_shift_end_time=get_doctor_instance.shift_end_time
                    print(get_shift_start_time,get_shift_end_time)
                    get_latest=Appointment.objects.filter(doctor=doctor,book_date=book_date).last()
                    
                    current_time = datetime.datetime.fromtimestamp(time.time()).strftime("%H:%M:%S")
                    get_today=datetime.datetime.now().date()


                    if book_date == str(get_today):
                        print("today date")
                        check_book_date_apt=Appointment.objects.filter(doctor=doctor,book_date=book_date).last()
                        if check_book_date_apt:

                            get_last_time=check_book_date_apt.appointment_time
                            
                            
                            if str(get_last_time) < current_time > str(get_shift_end_time):
                                extTime=datetime.datetime.now() + datetime.timedelta(minutes = 60)
                                if str(extTime) > get_shift_end_time :
                                    print("doctor is not available")
                                else:
                                    appoint=Appointment(
                                            book_time=current_time,
                                            appointment_time=extTime,
                                            symptom =symptoms,
                                            patient = patient,
                                            doctor = get_doctor_instance,
                                            department=department,
                                            book_date=book_date,
                                            )

                                    appoint.save()  
                                    break
                            elif str(get_last_time) > str(get_shift_end_time):
                                print("continue")
                                continue
                            
                            else:
                                print("if appointment is already booked for  this day and time is greater then current time")
                                extTime= get_last_time.replace(hour=(get_last_time.hour+1) % 24)
                                appoint=Appointment(
                                        book_time=current_time,
                                        appointment_time=extTime,
                                        symptom =symptoms,
                                        patient = patient,
                                        doctor = get_doctor_instance,
                                        department=department,
                                        book_date=book_date)
                                appoint.save()
                                messages.success(request, f'Your appointment is booked at  {extTime} with {doctor}')
                                break

                        else:
                            if str(get_shift_start_time) < current_time <= str(get_shift_end_time):
                                extTime = datetime.datetime.now() + datetime.timedelta(minutes = 60)
                                appoint=Appointment(
                                        book_time=current_time,
                                        appointment_time=extTime,
                                        symptom =symptoms,
                                        patient = patient,
                                        doctor = get_doctor_instance,
                                        department=department,
                                        book_date=book_date,)

                                appoint.save()
                                messages.success(request,f'Appointment booked sucessfully with {doctor} at {book_date} on {extTime}')
                                break
                            else:
                                print("today first")
                                extTime=get_shift_start_time .replace(minute=(get_shift_start_time.minute) % 1440)
                                appoint=Appointment(
                                        book_time=current_time,
                                        appointment_time=extTime,
                                        symptom =symptoms,
                                        patient = patient,
                                        doctor = get_doctor_instance,
                                        department=department,
                                        book_date=book_date,)

                                appoint.save()
                                messages.success(request, f"Yor Appointment booked sucessfully with {doctor} at {book_date} on {extTime}")
                                break
                                

                    
                    else:
                        if get_latest:
                            get_last_time=get_latest.appointment_time
                            #get_last_time=get_last_time.replace(hour=(get_last_time.hour+1) % 24)
                            if str(get_last_time.replace(hour=(get_last_time.hour+1) % 24))  >= str(get_shift_end_time):
                                continue
                            else:

                                extTime = get_last_time.replace(hour=(get_last_time.hour+1) % 24)
                                appoint=Appointment(
                                            book_time=current_time,
                                            appointment_time=extTime,
                                            symptom =symptoms,
                                            patient = patient,
                                            doctor = get_doctor_instance,
                                            department=department,
                                            book_date=book_date,)

                                appoint.save()
                                messages.success(request, f"Yor Appointment booked sucessfully with {doctor} at {book_date} on {extTime}") 
                                break


                        
                        
                        else:
                            extTime=get_shift_start_time .replace(minute=(get_shift_start_time.minute) % 1440)
                            appoint=Appointment(
                                        book_time=current_time,
                                        appointment_time=extTime,
                                        symptom =symptoms,
                                        patient = patient,
                                        doctor = get_doctor_instance,
                                        department=department,
                                        book_date=book_date,)

                            appoint.save()
                            messages.success(request, f"Yor Appointment booked sucessfully with {doctor} at {book_date} on {extTime}") 
                            break

                    
                         
                            #
            # if doctor is not available
            else:
                messages.success(request,'No Doctor Availble yet Try another time')
                        
            return redirect('appointment:book_an_appointment')      
            # for doctor in get_doctors:
            #     get_id=doctor.id
            #     get_doctor=DoctorProfile.objects.get(id=get_id)
            #     print("Doctor Instance IS",get_doctor)
            #     start_time=get_doctor.shift_start_time
            #     endtime=get_doctor.shift_end_time

            #     print("Timing",start_time,endtime)

            #     patient=PatientProfile.objects.get(user=request.user) 
            #     if book_date:
            #         if book_date < str(date.today()):
            #             messages.error(request,'Please choose  right date')
            #             context ={'form':DepartmentForm()}
            #             return render(request, 'appointment/appointment_create.html',context)

            #         else:
            #             date_appointment=Appointment.objects.filter(book_date=book_date).filter(doctor=get_doctor)
            #             print("date is this",date_appointment)
                        

            #             if date_appointment:
            #                 last_appointment_id=Appointment.objects.filter(book_date=book_date).filter(doctor=get_doctor).last()
            #                 last_appointment_time=last_appointment_id.appointment_time    
                            
            #                 update_time = last_appointment_time.replace(hour=(last_appointment_time.hour+1) % 24)

            #                 if str(update_time) > str(endtime) :
            #                     pass
            #                     # context ={'form':DepartmentForm()}
            #                     # messages.success(request,"No doctor available In this date please choose another date")
            #                     # return render(request, 'appointment/appointment_create.html',context)
            #                 #end from here
            #                 else:
  
            #                     appoint=Appointment(
            #                         book_time=current_time,
            #                         appointment_time=update_time,
            #                         symptom =symptoms,
            #                         patient = patient,
            #                         doctor = get_doctor,
            #                         department=department,
            #                         book_date=book_date,)

            #                     appoint.save()
                                

            #                 messages.success(request,f"You have an appointment on date {book_date} at {update_time}")
            #                 return redirect('appointment:book_an_appointment')
            #                 pass
            #             else:
            #                 print("today date is",date.today())    
            #                 print("book date is",book_date)    
            #                 update_time = start_time.replace(hour=(start_time.hour+1) % 24)
            #                 if str(date.today())==str(book_date):
            #                     print("yes it was")
            #                     if str(current_time) > str(update_time):
            #                         current_time =  datetime.now()
            #                         print(current_time)
            #                         #datetime.strptime(current_time,'%H:%M:%S').strftime('%H:%M:%S')
            #                         print(type(current_time))
                                
            #                         update_time = current_time.replace(hour=(current_time.hour+1) % 24)
            #                         appointment=Appointment(
            #                             book_time=current_time,
            #                             appointment_time=update_time,
            #                             symptom =symptoms,
            #                             patient = patient,
            #                             doctor = get_doctor,
            #                             department=department,
            #                             book_date=book_date,
            #                                 )
            #                         appointment.save()
            #                         messages.success(request,f'You have an appointment now at {update_time}')
            #                         return redirect('appointment:book_an_appointment') 
            #                     else:
            #                         appointment=Appointment(
            #                             book_time=current_time,
            #                             appointment_time=update_time,
            #                             symptom =symptoms,
            #                             patient = patient,
            #                             doctor = get_doctor,
            #                             department=department,
            #                             book_date=book_date,
            #                                 )
            #                         appointment.save()



            #             appoint=Appointment(
            #                     book_time=current_time,
            #                     appointment_time=update_time,
            #                     symptom =symptoms,
            #                     patient = patient,
            #                     doctor = get_doctor,
            #                     department=department,
            #                     book_date=book_date,
                                

            #                 )
            #             appoint.save()
            #             messages.success(request,f"You have an appointment on date {book_date} at {update_time}")
            #             return redirect('appointment:book_an_appointment')                              
                
               

        
    return render(request, 'appointment/appointment_create.html',{'form':DepartmentForm()})


#############============ End of Function to book an appointment ==========##########


def prescription(request ,pk):
    get_appointment=Appointment.objects.filter(id=pk).first()

    if Prescription.objects.filter(appointment=get_appointment).exists():
        messages.success(request,'You have already given a prescription')
        return redirect('appointment:appointments_list')

    


    if request.POST:
        patient = request.POST.get('patient')
        symptoms = request.POST.get('symptom')
        prescription = request.POST.get('prescription')

        print(patient,symptoms,prescription)

        get_patient=PatientProfile.objects.get(user__username=patient)
        get_doctor_instances=DoctorProfile.objects.get(user_id=request.user.id)
        

        pres=Prescription(
            appointment=get_appointment,
            patient= get_patient,
            doctor=get_doctor_instances,
            symptoms=symptoms,
            prescription=prescription,
            date =datetime.now()

        )
        pres.save()
        Appointment.objects.filter(id=pk).update(prescription_added=True)
        messages.info(request, "Prescription addes sucessfully")
        return redirect('appointment:prescription_list')


    context={'apts':get_appointment}
    return render(request, 'appointment/prescription_create.html',context)


def prescription_list(request):
    get_doctor=DoctorProfile.objects.get(user_id=request.user.id)
    lists=Prescription.objects.filter(doctor=get_doctor)
    context={'lists':lists}
    return render(request , 'appointment/prescription_list.html',context)


def medical_history(request):
    get_id=request.GET.get('appointment')
    get_patient_instance=''
    if get_id:
        appointment=Appointment.objects.filter(id=get_id).first()
        get_patient=appointment.patient.id
        get_patient_instance=PatientProfile.objects.filter(id=get_patient).first()
        lists=Prescription.objects.filter(patient=get_patient_instance)
        context={'lists':lists}
        return render(request , 'appointment/medical_history.html',context)


    get_patient_instance=PatientProfile.objects.filter(user_id=request.user.id).first()
    predicts=Predict.objects.filter(patient=get_patient_instance)
    
    lists=Prescription.objects.filter(patient=get_patient_instance)
    context={'lists':lists,'predicts':predicts}

    return render(request , 'appointment/medical_history.html',context)


def update_prescription(request,pk):

    pr=Prescription.objects.get(id=pk)

    if request.POST:
        form=PrescriptionForm(request.POST,instance=pr)
        if form.is_valid:
            form.save()

            messages.success(request, "Prescription updated sucessfully")
            return redirect('appointment:prescription_list')





    form=PrescriptionForm(instance=pr)

    return render(request , 'appointment/update_prescription.html',{'form':form})



def delete_appointment(request,pk):
    Appointment.objects.get(pk=pk).delete()
    messages.success(request, 'Deleted Successfully')
    return redirect('appointment:appointments_list')



# Disease Prediction 



##
def predict(request):
    return render(request, "Predict/predict.html")
def result(request):
    import numpy as np
    import pandas as pd
    from scipy.stats import mode

    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.svm import SVC
    from sklearn.naive_bayes import GaussianNB
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, confusion_matrix

    # Reading the train.csv by removing the
    # last column since it's an empty column
    DATA_PATH = "Training.csv"
    data = pd.read_csv(DATA_PATH).dropna(axis=1)


    # Encoding the target value into numerical
    # value using LabelEncoder
    encoder = LabelEncoder()
    data["prognosis"] = encoder.fit_transform(data["prognosis"])

    X = data.iloc[:,:-1]
    y = data.iloc[:,-1]
    X_train, X_test, y_train, y_test =train_test_split( X, y, test_size = 0.2, random_state = 24)

    # Defining scoring metric for k-fold cross validation
    def cv_scoring(estimator, X, y):
        return accuracy_score(y, estimator.predict(X))

    # Initializing Models
    models = {
        "SVC":SVC(),
        "Gaussian NB":GaussianNB(),
        "Random Forest":RandomForestClassifier(random_state=18)
    }

    # Producing cross validation score for the models
    for model_name in models:
        model = models[model_name]
        scores = cross_val_score(model, X, y, cv = 10,
                                n_jobs = -1,
                                scoring = cv_scoring)


    # Training and testing SVM Classifier
    svm_model = SVC()
    svm_model.fit(X_train, y_train)
    preds = svm_model.predict(X_test)


    cf_matrix = confusion_matrix(y_test, preds)


    # Training and testing Naive Bayes Classifier
    nb_model = GaussianNB()
    nb_model.fit(X_train, y_train)
    preds = nb_model.predict(X_test)

    cf_matrix = confusion_matrix(y_test, preds)


    # Training and testing Random Forest Classifier
    rf_model = RandomForestClassifier(random_state=18)
    rf_model.fit(X_train, y_train)
    preds = rf_model.predict(X_test)


    cf_matrix = confusion_matrix(y_test, preds)


    # Training the models on whole data
    final_svm_model = SVC()
    final_nb_model = GaussianNB()
    final_rf_model = RandomForestClassifier(random_state=18)
    final_svm_model.fit(X, y)
    final_nb_model.fit(X, y)
    final_rf_model.fit(X, y)

    # Reading the test data
    test_data = pd.read_csv("Testing.csv").dropna(axis=1)

    test_X = test_data.iloc[:, :-1]
    test_Y = encoder.transform(test_data.iloc[:, -1])

    # Making prediction by take mode of predictions
    # made by all the classifiers
    svm_preds = final_svm_model.predict(test_X)
    nb_preds = final_nb_model.predict(test_X)
    rf_preds = final_rf_model.predict(test_X)

    final_preds = [mode([i,j,k])[0][0] for i,j,
                k in zip(svm_preds, nb_preds, rf_preds)]



    cf_matrix = confusion_matrix(test_Y, final_preds)


    symptoms = X.columns.values

    # Creating a symptom index dictionary to encode the
    # input symptoms into numerical form
    symptom_index = {}
    for index, value in enumerate(symptoms):
         symptom_index[value] = index

    data_dict = {
        "symptom_index": symptom_index,
        "predictions_classes": encoder.classes_
    }

    #print(data_dict["symptom_index"]["Skin Rash"])
    # Defining the Function
    # Input: string containing symptoms separated by commmas
    # Output: Generated predictions by models
    def predictDisease(psymptoms):
        
        if len(psymptoms)>=4:
        #print(symptoms)
        # creating input data for the models
            input_data = [0] * len(data_dict["symptom_index"])
            
            for i in psymptoms:
                index = data_dict["symptom_index"][i]
                input_data[index] = 1
            #print(input_data)
            # reshaping the input data and converting it
            # into suitable format for model predictions
            input_data = np.array(input_data).reshape(1, -1)

            # generating individual outputs
            rf_prediction = data_dict["predictions_classes"][final_rf_model.predict(input_data)[0]]
            nb_prediction = data_dict["predictions_classes"][final_nb_model.predict(input_data)[0]]
            svm_prediction = data_dict["predictions_classes"][final_svm_model.predict(input_data)[0]]

            # making final prediction by taking mode of all predictions
            final_prediction = mode([rf_prediction, nb_prediction, svm_prediction])[0][0]
            
            Rheumatologist = [ ]
       
            Cardiologist = [ 'Heart attack','Bronchial Asthma','Hypertension ']
            
            ENT_specialist = ['(vertigo) Paroymsal  Positional Vertigo','Hypothyroidism' ]

            Orthopedist = ['Osteoarthristis','Arthritis']

            Neurologist = ['Varicose veins','Paralysis (brain hemorrhage)','Migraine','Cervical spondylosis']

            Allergist_Immunologist = ['Allergy','Pneumonia',
                'AIDS','Common Cold','Tuberculosis','Malaria','Dengue','Typhoid']

            Urologist = [ 'Urinary tract infection',
                'Dimorphic hemmorhoids(piles)']

            Dermatologist = [  'Acne','Chicken pox','Fungal infection','Psoriasis','Impetigo']

            Gastroenterologist = ['Peptic ulcer diseae', 'GERD','Chronic cholestasis','Drug Reaction','Gastroenteritis','Hepatitis E',
                'Alcoholic hepatitis','Jaundice','hepatitis A',
                'Hepatitis B', 'Hepatitis C', 'Hepatitis D','Diabetes ','Hypoglycemia']
                
            if final_prediction in Rheumatologist :
                consultdoctor = "Rheumatologist"
                
            if final_prediction in Cardiologist :
                consultdoctor = "Cardiologist"
                

            elif final_prediction in ENT_specialist :
                consultdoctor = "ENT specialist"
            
            elif final_prediction in Orthopedist :
                consultdoctor = "Orthopedist"
            
            elif final_prediction in Neurologist :
                consultdoctor = "Neurologist"
            
            elif final_prediction in Allergist_Immunologist :
                consultdoctor = "Allergist/Immunologist"
            
            elif final_prediction in Urologist :
                consultdoctor = "Urologist"
            
            elif final_prediction in Dermatologist :
                consultdoctor = "Dermatologist"
            
            elif final_prediction in Gastroenterologist :
                consultdoctor = "Gastroenterologist"
            
            else :
                consultdoctor = "other"
        
            return final_prediction,consultdoctor

        else:
            return "Please select 4 or more symptoms"    
    
    

    # Testing the function
    #print(predictDisease("Loss Of Appetite,Pain Behind The Eyes,Back Pain"))


    #val1 = request.GET['s1']
    #val2 = request.GET['s2']
    #val3 = request.GET['s3']
    #val4 = request.GET['s4']

    #print(val1,val2,val3,val4)
  
    user_input = request.POST
    data=dict(user_input.lists())
    data.pop('csrfmiddlewaretoken')
    
    symptoms_list=[]
    for k,v in data.items():
        
        symptoms_list.append(k)
    #print("this is symptoms_list",symptoms_list)    
    
    
    #pred=[NaiveBayes(symptoms_list),KNN(symptoms_list),randomforest(symptoms_list),DecisionTree(symptoms_list)]
    prediction,consult=predictDisease(symptoms_list)

    #print("this is prediction",prediction)

    
    get_patient_instance=PatientProfile.objects.get(user_id=request.user.id)

    pr=Predict(
        patient   =    get_patient_instance,
        symptoms =    str(symptoms_list),
        Predicted_Disease=prediction,
        predict_date=date.today(),
    )         
    pr.save()
    
    return JsonResponse({'status': 'success','prediction':prediction,'consult':consult})
    #return render(request, "Predict/predict.html", {"result2":prediction})


#Predicted Disease End here

