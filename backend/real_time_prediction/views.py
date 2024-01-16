from django.shortcuts import render
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import time
from data_scrap.views import fetchRealTimeData
from .models import RealTimePrediction

def load_and_preprocess_single_image(image_path):
    try:
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, (256, 256))
        image = image / 255.0
        return image
    except Exception as err:
        print(err)
        return None

def predictRealTime():
    loaded_model = tf.keras.models.load_model('model.h5')

    fetchRealTimeData()

    new_image_path = 'static/real-time-scrap/real-time.png'

    new_image = load_and_preprocess_single_image(new_image_path)

    if new_image is not None:
        new_image = np.expand_dims(new_image, axis=0)
        prediction = loaded_model.predict(new_image)
        # plt.imshow(new_image[0])
        # plt.title(f'Predicted WMO_WIND: {prediction[0][0]}')
        # plt.axis('off')
        # plt.show()
        print(f'Predicted WMO_WIND: {prediction[0][0]}')
        if len(list(RealTimePrediction.objects.all())) == 0:
            cyclone_intensity = RealTimePrediction.objects.create(prediction = float(prediction[0][0]))
        else:
            cyclone_intensity = RealTimePrediction.objects.get(id=1)
            cyclone_intensity.prediction = float(prediction[0][0])
        cyclone_intensity.save()
    else:
        print("Image loading or preprocessing failed.")

def fetchPrediction(request):
    intensity = RealTimePrediction.objects.get(id=1)
    print(intensity.prediction)
    return render(request, "index.html", {"prediction": intensity.prediction})
