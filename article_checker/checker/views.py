from django.conf import settings
from django.contrib import messages
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.core.mail import EmailMessage
from django.shortcuts import redirect, render
from django.urls import reverse
from django.utils import timezone
from django.views.decorators.csrf import csrf_exempt

from .forms import AbstractForm
from .models import *


@login_required
def check_article(request):
    if request.method == "POST":
        title = request.POST.get("title")
        abstract = request.POST.get("abstract")

        # Data `title` dan `abstract` sudah bisa digunakan untuk NLP di sini
        # Model NLP yang akan memproses data ini ditangani tim lain

        # Kirim data ke halaman result untuk ditampilkan
        return render(
            request,
            "checker/result.html",
            {
                "title": title,
                "abstract": abstract,
                #"nlp_result": nlp_result,
            },
        )

    return render(request, "checker/check_article.html")


def RegisterView(request):
    if request.method == 'POST':
        # getting user inputs from frontend
        first_name = request.POST.get('first_name')
        last_name = request.POST.get('last_name')
        username = request.POST.get('username')
        email = request.POST.get('email')
        password = request.POST.get('password')

        user_data_has_error = False

        if User.objects.filter(username=username).exists():
            user_data_has_error = True
            messages.error(request, "Username already exists")

        if User.objects.filter(email=email).exists():
            user_data_has_error = True
            messages.error(request, "Email already exists")

        if len(password) < 8:
            user_data_has_error = True
            messages.error(request, "Password must be at least 8 characters")

        if user_data_has_error:
            return redirect("register")
        else:
            new_user = User.objects.create_user(
                first_name=first_name,
                last_name=last_name,
                email=email,
                username=username,
                password=password,
            )
            messages.success(request, "Account created. Login now")
            return redirect("login")

    return render(request, "checker/register.html")


def LoginView(request):
    if request.method == 'POST':

        # getting user inputs from frontend
        username = request.POST.get('username')
        password = request.POST.get('password')

        # authenticate credentials
        user = authenticate(request=request, username=username, password=password)

        if user is not None:
            # login user if login credentials are correct
            login(request, user)

            # ewdirect to check article page
            return redirect("check_article")
        else:
            # redirect back to the login page if credentials are wrong
            messages.error(request, "Invalid username or password")
            return redirect("login")

    return render(request, "checker/login.html")


def LogoutView(request):

    logout(request)

    # redirect to login page after logout
    return redirect("login")


def ForgotPassword(request):
    if request.method == 'POST':
        email = request.POST.get('email')
        # verify if email exists
        try:
            user = User.objects.get(email=email)
            # create a new reset id
            new_password_reset = PasswordReset(user=user)
            new_password_reset.save()

            # creat password reset url;
            password_reset_url = reverse('reset-password', kwargs={'reset_id': new_password_reset.reset_id})

            full_password_reset_url = (
                f"{request.scheme}://{request.get_host()}{password_reset_url}"
            )

            # email content
            email_body = f'Reset your password using the link below:\n\n\n{full_password_reset_url}'

            email_message = EmailMessage(
                'Reset your password', # email subject
                email_body,
                settings.EMAIL_HOST_USER, # email sender
                [email] # email  receiver 
            )

            email_message.fail_silently = True
            email_message.send()

            return redirect('password-reset-sent', reset_id=new_password_reset.reset_id)

        except User.DoesNotExist:
            messages.error(request, f"No user with email '{email}' found")
            return redirect("forgot-password")

    return render(request, "checker/forgot_password.html")


def PasswordResetSent(request, reset_id):
    if PasswordReset.objects.filter(reset_id=reset_id).exists():
        return render(request, "checker/password_reset_sent.html")
    else:
        # redirect to forgot password page if code does not exist
        messages.error(request, 'Invalid reset id')
        return redirect('forgot-password')


def ResetPassword(request, reset_id):

    try:
        reset_id = PasswordReset.objects.get(reset_id=reset_id)
        
        if request.method == 'POST':

            password = request.POST.get('password')
            confirm_password = request.POST.get('confirm_password')
            
            passwords_have_error = False

            if password != confirm_password:
                passwords_have_error = True
                messages.error(request, 'Passwords do not match')

            if len(password) < 8:
                passwords_have_error = True
                messages.error(request, 'Password must be at least 8 characters long')

            # check to make sure link has not expired
            expiration_time = reset_id.created_when + timezone.timedelta(minutes=10)

            if timezone.now() > expiration_time:
                # delete reset id if expired
                reset_id.delete()
                passwords_have_error = True
                messages.error(request, 'Reset link has expired')
                
            # reset password
            if not passwords_have_error:
                user = reset_id.user
                user.set_password(password)
                user.save()
                
                # delete reset id after use
                reset_id.delete()

                # redirect to login
                messages.success(request, 'Password reset. Proceed to login')
                return redirect('login')

            else:
                # redirect back to password reset page and display errors
                return redirect('reset-password', reset_id=reset_id)

    except PasswordReset.DoesNotExist:

        # redirect to forgot password page if code does not exist
        messages.error(request, "Invalid reset id")
        return redirect("forgot-password")

    return render(request, "checker/reset_password.html")


from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from django.shortcuts import redirect, render

from .trial import get_classifier


@login_required
def check_article(request):
    if request.method == "POST":
        # Get data from form submission
        title = request.POST.get("title")
        abstract = request.POST.get("abstract")
        keywords = request.POST.get("keywords")

        # Store data in session to pass it to the next view
        request.session['title'] = title
        request.session['abstract'] = abstract
        request.session['keywords'] = keywords

        print(title)
        print(abstract)
        print(keywords)

        # Redirect to result view
        return redirect("result")  # Use URL name for cleaner redirection

    return render(request, "checker/check_article.html")


@login_required
def result(request):
    # # Retrieve stored session data
    title = request.session.get("title", "")
    abstract = request.session.get("abstract", "")
    keywords = request.session.get("keywords", "")

    # if not abstract:
    #     return redirect("check-abstract")


    nlp_result = get_classifier().predict_abstract_sections(abstract)

    return render(
        request,
        "checker/result.html",
        {
            "title": title,
            "abstract": abstract,
            "keywords": keywords,
            "nlp_result": nlp_result,
        },
    )

import pandas as pd
from flask import Blueprint, jsonify, request

# Initialize Blueprint
views = Blueprint('views', __name__)

# CSV Storage for User Abstracts
abstracts_df = pd.DataFrame(columns=['Abstract'])

@views.route('/save_abstract', methods=['POST'])
def save_abstract():
    try:
        # Receive abstract from user input
        abstract = request.form.get('abstract')
        if not abstract or not abstract.strip():
            return jsonify({"error": "Abstract cannot be empty."}), 400
        
        # Save dataset
        global abstracts_df
        abstracts_df = pd.concat([abstracts_df, pd.DataFrame({'Abstract': [abstract]})], ignore_index=True)
        abstracts_df.to_csv("abstracts_by_user.csv", index=False)
        
        # Predict abstract sections
        predicted_sections = get_classifier().predict_abstract_sections(abstract)
        
        return jsonify({
            "message": "Input loaded successfully!",
            "predicted_sections": predicted_sections
        }), 200
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500
