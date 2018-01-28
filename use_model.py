from keras.models import model_from_json
import cv2
import numpy as np


json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

loaded_model.load_weights("model.h5")
print("Loaded model from disk")

'''loaded_model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
'''
loaded_model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

img = cv2.imread("/media/parth/06C20E27C20E1B95/machine_leaning/sentdex/deepLearning/dogs_cates/test/21.jpg")
img = cv2.resize(img, (50,50))
print(img.shape)
img = img.reshape(1, 50, 50, 3)

print(img.shape)
#print(np.argmax(loaded_model.predict(img)))
print(loaded_model.predict(img))
