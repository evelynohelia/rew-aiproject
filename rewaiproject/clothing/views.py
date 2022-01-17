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

from django.views.decorators.csrf import csrf_exempt


def index(request):
    return HttpResponse("Hello, world. You're at the polls index.")

def getImage(request):
    return JsonResponse({'message':'Hello api'})

@csrf_exempt
def imageTest(request):
    print(request)
    if request.method == 'POST':
        form = ImageTestForm(request.POST, request.FILES)
        if form.is_valid():
           #img = form.cleaned_data.get("imagen")
           #img_read = img.read()
           file = request.FILES['imagen']
           source_folder = os.path.join(os.getcwd(), "assets/img/testImages")
           file_name = default_storage.save(source_folder+file.name, file)
           return JsonResponse({'message':'imagen recibida', 'file': file_name})
        else:
            print(form.errors)    

