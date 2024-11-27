from django.shortcuts import render, redirect
from checker.utils import check_abstract_structure
from .forms import AbstractForm
from django.contrib.auth.models import User
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.conf import settings
from django.core.mail import EmailMessage
from django.utils import timezone
from django.urls import reverse
from django.views.generic import ListView
from .models import *
from .models import Titles, Abstracts, Keywords
from .forms import TitleForm, AbstractForm, KeywordForm


@login_required
def check_article(request):
    if request.method == "POST":
        # Ambil input dari form
        title_text = request.POST.get("title")
        abstract_text = request.POST.get("abstract")
        keywords_text = request.POST.get("keywords")

        # Simpan data ke dalam model
        title_obj = Titles.objects.create(user=request.user, title=title_text)
        abstract_obj = Abstracts.objects.create(title=title_obj, abstract=abstract_text)
        Keywords.objects.create(abstract=abstract_obj, keyword_list=keywords_text)

        # Data berhasil disimpan, kirim ke halaman result untuk ditampilkan
        return render(
            request,
            "checker/result.html",
            {
                "title": title_text,
                "abstract": abstract_text,
                "keywords": keywords_text,
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