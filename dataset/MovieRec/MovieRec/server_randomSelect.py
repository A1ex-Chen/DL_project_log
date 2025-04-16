@cherrypy.expose()
@cherrypy.tools.json_out()
def randomSelect(self):
    return self.randomSelectMovies()
