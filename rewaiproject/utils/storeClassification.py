from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose, BatchNormalization, Dropout, Lambda
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.utils import normalize
import os
import glob
import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors
from matplotlib import cm

SIZE_X = 256 
SIZE_Y = 256
n_classes=59

def multi_unet_model(n_classes=4, IMG_HEIGHT=256, IMG_WIDTH=256, IMG_CHANNELS=1):
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    s = inputs

    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(s)
    c1 = Dropout(0.1)(c1)
    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)
    
    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = Dropout(0.1)(c2)
    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)
     
    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = Dropout(0.2)(c3)
    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)
     
    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = Dropout(0.2)(c4)
    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)
     
    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
    c5 = Dropout(0.3)(c5)
    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)
    
    #Expansive path 
    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = Dropout(0.2)(c6)
    c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)
     
    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = Dropout(0.2)(c7)
    c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)
     
    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
    c8 = Dropout(0.1)(c8)
    c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)
     
    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
    c9 = Dropout(0.1)(c9)
    c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)
     
    outputs = Conv2D(n_classes, (1, 1), activation='softmax')(c9)
     
    model = Model(inputs=[inputs], outputs=[outputs])
    
    return model
# 
def get_model():
    return multi_unet_model(n_classes=n_classes, IMG_HEIGHT=256, IMG_WIDTH=256, IMG_CHANNELS=1)

def run_model():
    model = get_model()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    model.load_weights('testmodel2.h5')
    return model


def get_mask(image,model,filename="prediccion.png",mode='camisa'):
    color_list = list(range(59))
    print(color_list)
    test_img_input=np.expand_dims(image, 0)
    prediction = (model.predict(test_img_input))
    predicted_img=np.argmax(prediction, axis=3)[0,:,:]
    def camisa(imgshow):
        new_image = imgshow
        for i in np.unique(new_image):
            if i > 18:
                new_image[np.where(new_image==i)] = 0
            else: 
                if i > 0:
                    new_image[np.where(new_image==i)] = 10
        return new_image


    def pantalon(imgshow):
        new_image = imgshow
        for i in np.unique(new_image):
            if i < 24 or i > 35:
                new_image[np.where(new_image==i)] = 0
            else:
                if i > 0:
                    new_image[np.where(new_image==i)] = 10

        return new_image

    def piel(imgshow):
        new_image = imgshow
        for i in np.unique(new_image):
            if i < 35:
                new_image[np.where(new_image==i)] = 0
            else:
                if i > 0:
                    new_image[np.where(new_image==i)] = 10
        return new_image
    plt.figure(figsize=(256, 256))
    if mode == 'camisa':
        predicted_img = camisa(predicted_img)
    elif mode == 'pantalon':
        predicted_img = pantalon(predicted_img)
    elif mode == 'piel':
        predicted_img = piel(predicted_img)
    else:
        predicted_img = predicted_img
    fig = plt.imshow(camisa(predicted_img),cmap='gray')
    plt.axis('off')
    plt.savefig(filename, bbox_inches='tight')
    plt.close()


'''
test_images = []
for directory_path in glob.glob(os.path.join(os.getcwd(), "clothing\IMAGES")):
    for img_path in glob.glob(os.path.join(directory_path, "*.png")):
        img = cv2.imread(img_path, 0)       
        img = cv2.resize(img, (256, 256))
        test_images.append(img)
test_images = np.expand_dims(test_images, axis=3)
test_images = normalize(test_images, axis=1)

color_list = list(range(59))
cmap = cm.get_cmap('jet')
norm = colors.Normalize(vmin=0, vmax=58)
test_img = np.array(test_images)
test_img = test_img[19]
test_img_input=np.expand_dims(test_img, 0)
prediction = (model.predict(test_img_input))
predicted_img=np.argmax(prediction, axis=3)[0,:,:]
print(np.unique(predicted_img))

plt.figure(figsize=(24, 16))
plt.subplot(231)
plt.title('Testing Image')
plt.imshow(test_img[:,:,0], cmap='gray')
plt.subplot(232)
plt.title('Prediction on test image')
imgs = plt.imshow(predicted_img, cmap='jet')

#FFE500
plt.colorbar(imgs, cmap=cmap, norm=norm, boundaries=np.unique(predicted_img), ticks=np.unique(predicted_img))
plt.show()

def camisas(imgshow):
    for i in np.unique(imgshow):
        if i > 20:
            imgshow[np.where(imgshow==i)] = 0
    
    
test_img = np.array(test_images)
test_img = test_img[18]
test_img_input=np.expand_dims(test_img, 0)
prediction = (model.predict(test_img_input))
predicted_img=np.argmax(prediction, axis=3)[0,:,:]

def camisas(imgshow):
    new_image = imgshow
    for i in np.unique(new_image):
        if i > 18:
            new_image[np.where(new_image==i)] = 0
        else: 
            if i > 0:
                new_image[np.where(new_image==i)] = 10
    return new_image


def pantalon(imgshow):
    new_image = imgshow
    for i in np.unique(new_image):
        if i < 24 or i > 35:
            new_image[np.where(new_image==i)] = 0
        else:
            if i > 0:
                new_image[np.where(new_image==i)] = 10

    return new_image

def piel(imgshow):
    new_image = imgshow
    for i in np.unique(new_image):
        if i < 35:
            new_image[np.where(new_image==i)] = 0
        else:
            if i > 0:
                new_image[np.where(new_image==i)] = 10
    return new_image



plt.figure(figsize=(22, 14))
plt.subplot(231)
plt.title('Testing Image')
plt.imshow(test_img[:,:,0], cmap='gray')
plt.subplot(232)
plt.title('Prediction on test image')
imgs = plt.imshow(predicted_img, cmap='jet')
plt.subplot(233)
plt.title('Shirts/Blouse')
solo_camisa = camisas(predicted_img)
imgs = plt.imshow(solo_camisa, cmap='jet')

plt.subplot(234)
plt.title('Pants/Leggins')
test_img_input=np.expand_dims(test_img, 0)
prediction = (model.predict(test_img_input))
predicted_img=np.argmax(prediction, axis=3)[0,:,:]
solo_pantalon = pantalon(predicted_img)
imgs = plt.imshow(solo_pantalon, cmap='jet')


plt.subplot(235)
plt.title('Skin')
test_img_input=np.expand_dims(test_img, 0)
prediction = (model.predict(test_img_input))
predicted_img=np.argmax(prediction, axis=3)[0,:,:]
solo_piel = piel(predicted_img)
imgs = plt.imshow(solo_piel, cmap='jet')

plt.savefig('result.png')
plt.show()
'''