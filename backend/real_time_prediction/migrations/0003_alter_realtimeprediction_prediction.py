# Generated by Django 5.0.1 on 2024-01-16 16:40

from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("real_time_prediction", "0002_alter_realtimeprediction_prediction"),
    ]

    operations = [
        migrations.AlterField(
            model_name="realtimeprediction",
            name="prediction",
            field=models.DecimalField(decimal_places=5, max_digits=25),
        ),
    ]
