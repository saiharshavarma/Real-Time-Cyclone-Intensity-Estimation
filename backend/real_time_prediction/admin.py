from django.contrib import admin
from .models import RealTimePrediction

class RealTimePredictionAdmin(admin.ModelAdmin):
    list_display = ('id', 'timestamp', 'prediction')

# Register your models here.
admin.site.register(RealTimePrediction, RealTimePredictionAdmin)