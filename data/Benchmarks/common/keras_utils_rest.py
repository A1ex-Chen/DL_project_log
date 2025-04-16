from __future__ import absolute_import

import warnings

from default_utils import set_seed as set_seed_defaultUtils
from scipy.stats.stats import pearsonr
from tensorflow.keras import backend as K
from tensorflow.keras import initializers, optimizers
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.layers import Dropout
from tensorflow.keras.metrics import (
    binary_crossentropy,
    mean_absolute_error,
    mean_squared_error,
)
from tensorflow.keras.utils import get_custom_objects

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    from sklearn.metrics import r2_score

import os










# clipnorm=kerasDefaults['clipnorm'],
# clipvalue=kerasDefaults['clipvalue'])

# Not generally available
#    elif optimizer == 'adamax':
#        return optimizers.Adamax(lr=lr, beta_1=kerasDefaults['beta_1'],
#                               beta_2=kerasDefaults['beta_2'],
#                               epsilon=kerasDefaults['epsilon'],
#                               decay=kerasDefaults['decay_lr'])

#    elif optimizer == 'nadam':
#        return optimizers.Nadam(lr=lr, beta_1=kerasDefaults['beta_1'],
#                               beta_2=kerasDefaults['beta_2'],
#                               epsilon=kerasDefaults['epsilon'],
#                               schedule_decay=kerasDefaults['decay_schedule_lr'])


















class PermanentDropout(Dropout):
    def __init__(self, rate, **kwargs):
        super(PermanentDropout, self).__init__(rate, **kwargs)
        self.uses_learning_phase = False

    def call(self, x, mask=None):
        if 0.0 < self.rate < 1.0:
            noise_shape = self._get_noise_shape(x)
            x = K.dropout(x, self.rate, noise_shape)
        return x





def register_permanent_dropout():
    get_custom_objects()["PermanentDropout"] = PermanentDropout


class LoggingCallback(Callback):
