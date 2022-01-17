import cv2
import os
from django.shortcuts import render
from django.http import HttpResponse
from django.http import JsonResponse
from .forms import ImageTestForm
from django.core.files.base import ContentFile
from django.core.files.storage import default_storage
from django.conf import settings
from django.conf.urls.static import static
from django.views.decorators.csrf import csrf_exempt
import utils.storeClassification as classification

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
           img = cv2.imread(file_name, 0)       
           img = cv2.resize(img, (256, 256))
           model = classification.run_model()
           file_name_ext = file.name.split('.')
           file_base_name = file_name_ext[0]
           result_camisa = classification.get_mask(img,model, 'blusa.jpeg',mode='camisa')
           result_pantalon = classification.get_mask(img,model, 'pantalon.jpeg',mode='pantalon')
          
           # metodo para obtener la nueva imagen
           # file_output_name
           return JsonResponse({'message':'imagen recibida', 'recibido': file_name, 'result_camisa': result_camisa,
           'result_pantalon': result_pantalon})
        else:
            print(form.errors)    

