import os
import numpy as np
import pandas as pd
import django
from decimal import Decimal

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "RecSysInItem.settings")
django.setup()

from analytics.models import Rating
from items.models import Item


















#
# a = load_all_ratings()
# # print(a)
# a = dfToDict(a)
# print(a)
# # print(a.info())