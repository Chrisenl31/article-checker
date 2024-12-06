from django.urls import path

from . import views

urlpatterns = [
    path('check-abstract/', views.check_abstract, name='check_abstract'),
]

