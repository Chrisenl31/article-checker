from django.urls import path
from . import views

urlpatterns = [
    path('', views.check_article, name='check_article'),
    # path('result/', views.result, name='result'),
]
