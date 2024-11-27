from django.contrib import admin
from .models import *
from .models import PasswordReset, Titles, Abstracts, Keywords

@admin.register(PasswordReset)
class PasswordResetAdmin(admin.ModelAdmin):
    list_display = ('user', 'reset_id', 'created_when')

@admin.register(Titles)
class TitlesAdmin(admin.ModelAdmin):
    list_display = ('user', 'title')

@admin.register(Abstracts)
class AbstractsAdmin(admin.ModelAdmin):
    list_display = ('title',)

@admin.register(Keywords)
class KeywordsAdmin(admin.ModelAdmin):
    list_display = ('abstract', 'keyword_list')