@cherrypy.expose()
@cherrypy.tools.json_out()
@cherrypy.tools.json_in()
def inference(self):
    cherrypy_cors.preflight(allowed_methods=['GET', 'POST'])
    print(cherrypy.request.json)
    input_json = cherrypy.request.json
    return self.model_inference(input_json['movie_seq'])
