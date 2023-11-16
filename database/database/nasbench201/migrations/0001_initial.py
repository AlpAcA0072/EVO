# Generated by Django 3.2.12 on 2022-03-20 14:40

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='NASBench201Result',
            fields=[
                ('id', models.BigAutoField(primary_key=True, serialize=False)),
                ('genotype', models.JSONField(default=dict)),
                ('result', models.JSONField(default=dict)),
                ('index', models.IntegerField(db_index=True)),
                ('phenotype', models.CharField(db_index=True, max_length=256)),
                ('cost12', models.JSONField(default=dict)),
                ('cost200', models.JSONField(default=dict)),
            ],
            options={
                'abstract': False,
            },
        ),
    ]
