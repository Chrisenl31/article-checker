from django.urls import path

from . import views
from .models import AbstractInput

urlpatterns = [
    path("", views.input_article, name="input_article"),
    path("register/", views.RegisterView, name="register"),
    path("login/", views.LoginView, name="login"),
    path("logout/", views.LogoutView, name="logout"),
    path("forgot-password/", views.ForgotPassword, name="forgot-password"),
    path(
        "password-reset-sent/<str:reset_id>/",
        views.PasswordResetSent,
        name="password-reset-sent",
    ),
    path("reset-password/<str:reset_id>/", views.ResetPassword, name="reset-password"),
    path("result/", views.result, name="result"),
    path("check_article", views.result, name="check_article"),
    path("input_article/", views.input_article, name="input_article"),
    # path("input_article/", views.input_article, name="input_article"),
]
