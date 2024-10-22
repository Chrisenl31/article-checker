from django.shortcuts import render, redirect
from .forms import AbstractForm

def check_article(request):
    if request.method == 'POST':
        form = AbstractForm(request.POST)
        if form.is_valid():
            title = form.cleaned_data['title']
            abstract = form.cleaned_data['abstract']

            # Redirect ke halaman 'result.html' dan kirim data
            return render(request, 'checker/result.html', {
                'title': title,
                'abstract': abstract
            })
    else:
        form = AbstractForm()

    return render(request, 'checker/check_article.html', {'form': form})
