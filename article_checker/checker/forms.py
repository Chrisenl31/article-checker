from django import forms

class AbstractForm(forms.Form):
    title = forms.CharField(label='Title', max_length=200)
    abstract = forms.CharField(widget=forms.Textarea, label='Abstract')
