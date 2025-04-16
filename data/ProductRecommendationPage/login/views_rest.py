from django.core.paginator import Paginator
from django.core.paginator import EmptyPage
from django.core.paginator import PageNotAnInteger
from django.shortcuts import render,redirect
from login.models import User
from items.models import Item
from analytics.models import Rec_Items
from login.forms import UserForm, RegisterForm
import pandas as pd
# Create your views here.


    # return render(request, 'login/index.html', context)













###########################################################################################################################


# def hash_code(s, salt='mysite'):
#     h = hashlib.sha256()
#     s += salt
#     h.update(s.encode())
#     return h.hexdigest()

# def pay(request):
#     return render(request, 'pay.html')
#
# def item_detail(request):
#     return render(request, 'item_detail.html')
#
# def success(request):
#     return render(request, 'success.html')
#
# def car(request):
#     return render(request, 'car.html')
#
# def index(request):
#     if request.method == 'POST':
#         keyStr = request.POST.get('mykey')
#         limit = 16 # 每页显示的记录数
#         datas = Item.objects.filter(title__contains=keyStr)
#         tuijian=Item.objects.filter(title__contains=keyStr)[:6]
#         paginator = Paginator(datas, limit)  # 实例化一个分页对象
#         page = request.GET.get('page')  # 获取页码
#         try:
#             datas = paginator.page(page)  # 获取某页对应的记录
#         except PageNotAnInteger:  # 如果页码不是个整数
#             datas = paginator.page(1)  # 取第一页的记录
#         except EmptyPage:  # 如果页码太大，没有相应的记录
#             datas = paginator.page(paginator.num_pages)  # 取最后一页的记录
#
#         return render(request, 'search.html',{'datas':datas,'keyStr':keyStr,'tuijian':tuijian})
#     else:
#         return render(request, 'index.html')
#
# def search(request):
#     if request.method == 'POST':
#         keyStr = request.POST.get('mykey')
#         limit = 16 # 每页显示的记录数
#         datas = Item.objects.filter(title__contains=keyStr)
#         tuijian=Item.objects.filter(title__contains=keyStr)[:6]
#         paginator = Paginator(datas, limit)  # 实例化一个分页对象
#         page = request.GET.get('page')  # 获取页码
#         try:
#             datas = paginator.page(page)  # 获取某页对应的记录
#         except PageNotAnInteger:  # 如果页码不是个整数
#             datas = paginator.page(1)  # 取第一页的记录
#         except EmptyPage:  # 如果页码太大，没有相应的记录
#             datas = paginator.page(paginator.num_pages)  # 取最后一页的记录
#
#         return render(request, 'search.html',{'datas':datas,'keyStr':keyStr,'tuijian':tuijian})
#     else:
#         return render(request, 'search.html')
#
# def login(request):
#     if request.session.get('is_login', None):  # 不允许重复登录
#         return redirect('/index/')
#     if request.method == 'POST':
#         login_form = UserForm(request.POST)
#         message = '请检查填写的内容！'
#         if login_form.is_valid():
#             username = login_form.cleaned_data.get('username')
#             password = login_form.cleaned_data.get('password')
#
#             try:
#                 user = User.objects.get(name=username)
#             except :
#                 message = '用户不存在！'
#                 return render(request, 'login.html', locals())
#
#             if user.password == hash_code(password):
#                 request.session['is_login'] = True
#                 request.session['user_id'] = user.id
#                 request.session['user_name'] = user.name
#                 return redirect('/index/')
#             else:
#                 message = '密码不正确！'
#                 return render(request, 'login.html', locals())
#         else:
#             return render(request, 'login.html', locals())
#
#     login_form = UserForm()
#     return render(request, 'login.html', locals())
#
#
# def register(request):
#     if request.session.get('is_login', None):
#         return redirect('/index/')
#
#     if request.method == 'POST':
#         register_form = RegisterForm(request.POST)
#         message = "请检查填写的内容！"
#         if register_form.is_valid():
#             username = register_form.cleaned_data.get('username')
#             password1 = register_form.cleaned_data.get('password1')
#             password2 = register_form.cleaned_data.get('password2')
#             email = register_form.cleaned_data.get('email')
#             sex = register_form.cleaned_data.get('sex')
#             age = register_form.cleaned_data.get('age')
#
#             if password1 != password2:
#                 message = '两次输入的密码不同！'
#                 return render(request, 'register.html', locals())
#             else:
#                 same_name_user = User.objects.filter(name=username)
#                 if same_name_user:
#                     message = '用户名已经存在'
#                     return render(request, 'register.html', locals())
#                 same_email_user = User.objects.filter(email=email)
#                 if same_email_user:
#                     message = '该邮箱已经被注册了！'
#                     return render(request, 'register.html', locals())
#
#                 # new_user = User()
#                 new_user = User.objects.create(name=username,
#                                                password=password1,
#                                                email=email,
#                                                sex = sex,
#                                                age = age)
#                 # new_user.name = username
#                 # new_user.password = hash_code(password1)
#                 # new_user.email = emai
#                 # new_user.sex = sex
#                 new_user.save()
#
#                 return redirect('/login/')
#         else:
#             return render(request, 'register.html', locals())
#     register_form = RegisterForm()
#     return render(request, 'register.html', locals())
#
#
# def logout(request):
#     if not request.session.get('is_login', None):
#         return redirect('/login/')
#
#     request.session.flush()
#     # del request.session['is_login']
#     return redirect("/login/")

