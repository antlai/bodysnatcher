import cherrypy
import time
import os
import json
from .calibrate import mainCalibrate
from .segment import mainSegment

class BodySnatcher(object):
    @cherrypy.expose
    def index(self):
        return "Hello from BodySnatcher"

    @cherrypy.expose
    @cherrypy.tools.json_out()
    def calibrate(self, options = None):
        cherrypy.response.headers['Content-Type'] = 'application/json'
        options = json.loads(options) if options != None else options
        return mainCalibrate(options)

    @cherrypy.expose
    def parts(self, options = None):
        cherrypy.response.headers['Content-Type'] = 'application/json'
        options = json.loads(options) if options != None else options
        return mainSegment(options)
    parts._cp_config = {'response.stream': True}

def run():
    script_dir = os.path.dirname(__file__)
    cherrypy.config.update(os.path.join(script_dir, "server.config"))
    cherrypy.quickstart(BodySnatcher())


if __name__ == '__main__':
    run()
