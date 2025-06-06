"""商品推荐网页 URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, include
from django.conf.urls import url
from login import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('index/', views.index),
    # url(r'^$', views.index, name='index'),
    path('login/', views.login),
    path('register/', views.register),
    path('logout/', views.logout),
    path('detail/', views.detail, name='detail'),
    url(r'^collect/', include('analytics.urls')),
    url(r'^front/', include('login.urls')),
]


##########################################################################

# urlpatterns = [
#     path('admin/', admin.site.urls),
#     path('index/', views.index),
#     path('login/', views.login),
#     path('register/', views.register),
#     path('logout/', views.logout),
#     path('captcha/', include('captcha.urls')),
#     path('search/', views.search),
#     path('pay/', views.pay),
#     path('success/', views.success),
#     path('car/', views.car),
#     path('item_detail/', views.item_detail),
# ]

