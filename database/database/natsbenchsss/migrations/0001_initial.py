# Generated by Django 3.2.12 on 2022-04-04 13:09

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='NATSBenchResult',
            fields=[
                ('id', models.BigAutoField(primary_key=True, serialize=False)),
                ('phenotype', models.JSONField(default=dict)),
                ('genotype', models.JSONField(default=dict)),
                ('result', models.JSONField(default=dict)),
                ('cifar10_valid', models.JSONField()),
                ('cifar10', models.JSONField()),
                ('cifar100', models.JSONField()),
                ('ImageNet16_120', models.JSONField()),
                ('index', models.IntegerField()),
            ],
            options={
                'abstract': False,
            },
        ),
    ]
