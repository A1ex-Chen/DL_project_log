import json
import cherrypy
import torch
import pickle
from pathlib import Path
from models import BERTModel
from model_args import args
import pandas as pd
import os
import numpy as np
import cherrypy_cors
cherrypy_cors.install()


class Root(object):
        


    

    @cherrypy.expose()
    @cherrypy.tools.json_out()
    @cherrypy.tools.json_in()

    @cherrypy.expose()
    @cherrypy.tools.json_out()

if __name__ == '__main__':
    cherrypy.quickstart(Root(), '/', config="server.conf")
else:
    application = cherrypy.Application(Root(), '/', config="server.conf")