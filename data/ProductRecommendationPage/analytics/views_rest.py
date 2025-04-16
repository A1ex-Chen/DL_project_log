from django.shortcuts import render
from django.http import HttpResponse
import datetime
from django.views.decorators.csrf import ensure_csrf_cookie
from analytics.models import Rating
from login.models import User
import pandas as pd
# Create your views here.




@ensure_csrf_cookie


