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
import utils.clothingClassification as store

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
        print('form',form)
        if form.is_valid():
           file = request.FILES['imagen']
           source_folder = os.path.join(settings.STATICFILES_DIRS[0], "img/testImages/")
           
           file_name = default_storage.save(source_folder+file.name, file)
           img = cv2.imread(file_name, 0)       
           img = cv2.resize(img, (256, 256))
           model = classification.run_model()
           result_pre_camisa = classification.get_mask(img,model, 'blusa',mode='camisa')
           result_pre_pantalon = classification.get_mask(img,model, 'pantalon',mode='pantalon')
           result_pre_camisa_mask = cv2.imread(os.path.join(os.getcwd(),'assets/img/results/blusa_mask.jpeg'),0) 
           result_pre_pantalon_mask = cv2.imread(os.path.join(os.getcwd(),'assets/img/results/blusa_mask.jpeg'),0)
           img_camisa = cv2.resize(result_pre_camisa_mask, (256, 256))
           img_pantalon = cv2.resize(result_pre_pantalon_mask, (256, 256))
           y_result_camisa = store.get_class(img_camisa)     
           y_result_pantalon = store.get_class(img_pantalon)   
           
           list_camisa = y_result_camisa.tolist()
           tmp_camisa = max(list_camisa[0])
           index_camisa = list_camisa[0].index(tmp_camisa)

           list_pantalon  = y_result_pantalon.tolist()
           tmp_pantalon = max(list_pantalon[0])
           index_pantalon = list_pantalon[0].index(tmp_pantalon)
           
           stores = ['DePrati', 'EtaFashion', 'RMStore']
           store_camisa = stores[index_camisa]
           store_pantalon = stores[index_pantalon]

           return JsonResponse({'message':'imagen recibida', 'recibido': file_name, 
           'blusa': 'img/results/blusa.jpeg',
           'pantalon': 'img/results/pantalon.jpeg',
           'result_pre_camisa': result_pre_camisa,
           'result_pre_pantalon': result_pre_pantalon,
           'y_result_camisa': y_result_camisa.tolist(),
           'y_result_pantalon': y_result_pantalon.tolist(),
           'store_camisa' : store_camisa,
           'store_pantalon' : store_pantalon
           })

        else:
            print(form.errors)    

@csrf_exempt
def download(request, path):
    print('dowload',path)
    file_path = os.path.join(settings.MEDIA_ROOT, path)
    if os.path.exists(file_path):
        with open(file_path, 'rb') as fh:
            response = HttpResponse(fh.read(), content_type="application/form-data")
            response['Content-Disposition'] = 'inline; filename=' + os.path.basename(file_path)
            return response
