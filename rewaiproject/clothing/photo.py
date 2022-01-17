class Photo:
    def __init__(self, email, content, created=None):
        self.nombre = email
        self.photoUrl = content

photo = Photo(nombre='prueba', photoUrl='prueba.jpg')