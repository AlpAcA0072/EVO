# Generated by Django 4.0.4 on 2022-04-20 01:06

from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ('natsbenchsss', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='natsbenchresult',
            name='cost',
            field=models.JSONField(default=dict),
        ),
    ]
