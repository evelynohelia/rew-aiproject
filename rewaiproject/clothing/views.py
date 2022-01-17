from io import FileIO
import json
import cv2
import os
import numpy
from django.shortcuts import render
from django.http import HttpResponse
from django.http import JsonResponse
from .forms import ImageTestForm
from PIL import Image
from django.core.files.base import ContentFile
from django.core.files.storage import default_storage
from django.conf import settings
from django.conf.urls.static import static
from django.views.decorators.csrf import csrf_exempt


def index(request):
    return render(
        request, 'index.html', context={},
    )

def getImage(request):
    return JsonResponse({'message':'Hello api'})

@csrf_exempt
def imageTest(request):
    print(request)
    if request.method == 'POST':
        form = ImageTestForm(request.POST, request.FILES)
        if form.is_valid():
           file = request.FILES['imagen']
           source_folder = os.path.join(settings.STATICFILES_DIRS[0], "img/testImages/")
           file_name = default_storage.save(source_folder+file.name, file)
           return JsonResponse({'message':'imagen recibida', 'file': file_name})
        else:
            print(form.errors)    

