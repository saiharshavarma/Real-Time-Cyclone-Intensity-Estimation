# serializers.py
from rest_framework import serializers
from .models import RealTimePrediction

class RealTimePredictionSerializer(serializers.ModelSerializer):
    class Meta:
        model = RealTimePrediction
        fields = ['id', 'prediction', 'original_img', 'processed_img', 'level_sets_img', 'timestamp']
