# Generated by Django 5.0.1 on 2024-03-13 13:08

from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        (
            "real_time_prediction",
            "0002_rename_realtimeprediction_cycloneintensityprediction",
        ),
    ]

    operations = [
        migrations.CreateModel(
            name="RealTimePrediction",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                (
                    "wind",
                    models.DecimalField(decimal_places=5, default=0, max_digits=25),
                ),
                (
                    "pressure",
                    models.DecimalField(decimal_places=5, default=0, max_digits=25),
                ),
                (
                    "t_number",
                    models.DecimalField(decimal_places=5, default=0, max_digits=25),
                ),
                ("category", models.CharField(default=None, max_length=50)),
                (
                    "original_img",
                    models.ImageField(default="image.jpg", upload_to="realtime/"),
                ),
                (
                    "processed_img",
                    models.ImageField(default="image.jpg", upload_to="processed/"),
                ),
                ("timestamp", models.DateTimeField(auto_now_add=True, null=True)),
            ],
        ),
        migrations.DeleteModel(
            name="CycloneIntensityPrediction",
        ),
    ]
