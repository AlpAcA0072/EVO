# Generated by Django 3.2.12 on 2022-03-19 09:13

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='NASBench101Result',
            fields=[
                ('id', models.BigAutoField(primary_key=True, serialize=False)),
                ('index', models.TextField(db_index=True, max_length=256, null=True)),
                ('phenotype', models.JSONField(default=dict)),
                ('genotype', models.JSONField(default=dict)),
                ('result', models.JSONField(default=dict)),
                ('epoch4', models.JSONField(default=dict)),
                ('epoch12', models.JSONField(default=dict)),
                ('epoch36', models.JSONField(default=dict)),
                ('epoch108', models.JSONField(default=dict)),
            ],
            options={
                'abstract': False,
            },
        ),
    ]
