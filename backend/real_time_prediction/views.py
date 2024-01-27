from django.shortcuts import render, HttpResponse
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from data_scrap.views import fetchRealTimeData
from rest_framework import viewsets
from rest_framework.response import Response
from .models import RealTimePrediction
from .serializers import RealTimePredictionSerializer
from skimage.color import rgb2gray
from skimage.segmentation import chan_vese
from django.core.files.base import ContentFile
import io
from PIL import Image

# Load the model when the module is imported
loaded_model = tf.keras.models.load_model('model.h5')

def image_segmentation_level_sets(image):
    img_gray = rgb2gray(image)

    mask = chan_vese(img_gray, mu=0.25, lambda1=1, lambda2=1, tol=1e-3)

    segmented_image = np.zeros_like(image)
    segmented_image[mask] = [255, 255, 255]

    return segmented_image

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
    global loaded_model  # Access the globally loaded model
    fetchRealTimeData()

    new_image_path = 'static/real-time-scrap/real-time.png'

    original_image = Image.open(new_image_path)

    processed_image = load_and_preprocess_single_image(new_image_path)

    if processed_image is not None:
        processed_image = np.expand_dims(processed_image, axis=0)
        prediction = loaded_model.predict(processed_image)
        print(f'Predicted WMO_WIND: {prediction[0][0]}')

        # Save original image to in-memory file buffer
        original_buffer = io.BytesIO()
        original_image.save(original_buffer, format='PNG')
        original_buffer.seek(0)

        # Create Django ImageField instance for the original image
        original_img = ContentFile(original_buffer.read(), name='original.png')

        # Convert NumPy arrays to PIL images and save processed image
        processed_pil_image = Image.fromarray((processed_image[0] * 255).astype(np.uint8))
        processed_buffer = io.BytesIO()
        processed_pil_image.save(processed_buffer, format='PNG')
        processed_buffer.seek(0)
        processed_img = ContentFile(processed_buffer.read(), name='processed.png')

        # Perform image segmentation and save level sets image
        level_sets_image = image_segmentation_level_sets(processed_image[0])
        level_sets_pil_image = Image.fromarray((level_sets_image * 255).astype(np.uint8))
        level_sets_buffer = io.BytesIO()
        level_sets_pil_image.save(level_sets_buffer, format='PNG')
        level_sets_buffer.seek(0)
        level_sets_img = ContentFile(level_sets_buffer.read(), name='levelsets.png')

        # Create and save RealTimePrediction object
        cyclone_intensity = RealTimePrediction.objects.create(
            prediction=float(prediction[0][0]), 
            original_img=original_img, 
            processed_img=processed_img, 
            level_sets_img=level_sets_img
        )
        cyclone_intensity.save()
    else:
        print("Image loading or preprocessing failed.")
    return HttpResponse("Done")

class RealTimePredictionViewSet(viewsets.ModelViewSet):
    serializer_class = RealTimePredictionSerializer
    queryset = RealTimePrediction.objects.all().order_by('id')

    def list(self, request):
        return Response(RealTimePredictionSerializer(RealTimePrediction.objects.order_by('-timestamp').first()).data)