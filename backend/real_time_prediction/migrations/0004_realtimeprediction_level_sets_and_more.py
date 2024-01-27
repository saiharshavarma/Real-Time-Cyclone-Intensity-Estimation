# Generated by Django 5.0.1 on 2024-01-27 12:55

from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("real_time_prediction", "0003_alter_realtimeprediction_prediction"),
    ]

    operations = [
        migrations.AddField(
            model_name="realtimeprediction",
            name="level_sets",
            field=models.ImageField(default="image.jpg", upload_to="levelsets/"),
        ),
        migrations.AddField(
            model_name="realtimeprediction",
            name="original_img",
            field=models.ImageField(default="image.jpg", upload_to="realtime/"),
        ),
        migrations.AddField(
            model_name="realtimeprediction",
            name="processed_img",
            field=models.ImageField(default="image.jpg", upload_to="processed/"),
        ),
        migrations.AddField(
            model_name="realtimeprediction",
            name="timestamp",
            field=models.DateTimeField(auto_now_add=True, null=True),
        ),
    ]