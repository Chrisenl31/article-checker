from django.shortcuts import render
from .forms import AbstractForm


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
