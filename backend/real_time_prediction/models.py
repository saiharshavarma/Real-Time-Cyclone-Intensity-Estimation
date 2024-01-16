from django.db import models

# Create your models here.
class RealTimePrediction(models.Model):
    prediction = models.DecimalField(decimal_places=5, max_digits=25)
    
    def __str__(self):
        return self.prediction