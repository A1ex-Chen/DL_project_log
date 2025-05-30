# Generated by Django 2.1 on 2020-01-03 17:56

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Rating',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('user_id', models.CharField(max_length=20)),
                ('item_id', models.CharField(max_length=20)),
                ('rating', models.DecimalField(decimal_places=2, max_digits=4)),
                ('rating_timestamp', models.CharField(max_length=100)),
                ('type', models.CharField(default='explicit', max_length=50)),
            ],
        ),
    ]