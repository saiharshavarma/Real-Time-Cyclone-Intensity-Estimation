from django.db import models

# Create your models here.
class RealTimePrediction(models.Model):
    prediction = models.DecimalField(decimal_places=5, max_digits=25)
    original_img = models.ImageField(upload_to='realtime/', default='image.jpg')
    processed_img = models.ImageField(upload_to='processed/', default='image.jpg')
    level_sets_img = models.ImageField(upload_to='levelsets/', default='image.jpg')
    timestamp = models.DateTimeField(auto_now_add=True, blank=True, null=True)
    
    def __str__(self):
        return "Prediction at " + str(self.timestamp)