# Generated by Django 5.0.1 on 2024-01-27 13:03

from django.db import migrations


class Migration(migrations.Migration):
    dependencies = [
        ("real_time_prediction", "0004_realtimeprediction_level_sets_and_more"),
    ]

    operations = [
        migrations.RenameField(
            model_name="realtimeprediction",
            old_name="level_sets",
            new_name="level_sets_img",
        ),
    ]