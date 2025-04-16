import collections
import logging
import os

from deepview_profile.exceptions import NoConnectionError

import deepview_profile.protocol_gen.innpv_pb2 as pm

logger = logging.getLogger(__name__)

RequestContext = collections.namedtuple(
    'RequestContext',
    ['address', 'state', 'sequence_number'],
)



class MessageHandler:


