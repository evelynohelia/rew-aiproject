from distutils.command.upload import upload
from django import forms
from PIL import Image

class ImageTestForm(forms.Form):
    imagen = forms.ImageField()

class ImageOutPutTestForm(forms.Form):
    imagen = forms.ImageField()    