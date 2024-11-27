from django import forms
from .models import Titles, Abstracts, Keywords

# class AbstractForm(forms.Form):
#     title = forms.CharField(label='Title', max_length=200)
#     abstract = forms.CharField(widget=forms.Textarea, label='Abstract')

# Form untuk model Titles
class TitleForm(forms.ModelForm):
    class Meta:
        model = Titles
        fields = ['title']  # Field 'user' tidak diinputkan langsung
        widgets = {
            'title': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': 'Enter the article title',
            }),
        }
        labels = {
            'title': 'Title',
        }

# Form untuk model Abstracts
class AbstractForm(forms.ModelForm):
    class Meta:
        model = Abstracts
        fields = ['abstract']
        widgets = {
            'abstract': forms.Textarea(attrs={
                'class': 'form-control',
                'placeholder': 'Enter the abstract here',
                'rows': 5,
            }),
        }
        labels = {
            'abstract': 'Abstract',
        }

# Form untuk model Keywords
class KeywordForm(forms.ModelForm):
    class Meta:
        model = Keywords
        fields = ['keyword_list']
        widgets = {
            'keyword_list': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': 'Enter keywords separated by commas',
            }),
        }
        labels = {
            'keyword_list': 'Keywords',
        }
