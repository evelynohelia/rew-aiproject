from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('api/result', views.getImage, name='getImage'), 
    path('api/testImage', views.imageTest, name='testImage'),
    path('api/download', views.download, name='dowload')
]