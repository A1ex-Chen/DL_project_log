from django.db import models

# Create your models here.

class User(models.Model):
    '''用户表'''
    gender = (
        ('male','男'),
        ('female','女'),
    )
    name = models.CharField(max_length=128,unique=True)
    password = models.CharField(max_length=256)
    email = models.EmailField(unique=True)
    sex = models.CharField(max_length=32,choices=gender,default='男')
    age = models.IntegerField()
    c_time = models.DateTimeField(auto_now_add=True)


    class Meta:
        ordering = ['c_time']
        verbose_name = '用户'
        verbose_name_plural = '用户'


class Post(models.Model):
    title = models.CharField(max_length=70)









