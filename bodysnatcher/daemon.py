import cherrypy
import time
import os
import sys
import json
from .calibrate import mainCalibrate
from .calibrate import mainCalibrateProjector
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
    @cherrypy.tools.json_in()
    @cherrypy.tools.json_out()
    def calibrateProjector(self):
        cherrypy.response.headers['Content-Type'] = 'application/json'
        data = cherrypy.request.json
        return mainCalibrateProjector(data['objPoints'], data['imagePoints'],
                                      data['shape'])

    @cherrypy.expose
    @cherrypy.tools.json_out()
    def points3D(self, options = None):
        cherrypy.response.headers['Content-Type'] = 'application/json'
        print options
        options = json.loads(options) if options != None else {}
        options['onlyPoints3D'] = True
        return mainCalibrate(options)

    @cherrypy.expose
    def parts(self, options = None):
        cherrypy.response.headers['Content-Type'] = 'application/json'
        options = json.loads(options) if options != None else options
        return mainSegment(options)
    parts._cp_config = {'response.stream': True}

    @cherrypy.expose
    def reset(self):
        sys.exit(0)


def run():
    script_dir = os.path.dirname(__file__)
    cherrypy.config.update(os.path.join(script_dir, "server.config"))
    cherrypy.quickstart(BodySnatcher())


if __name__ == '__main__':
    run()
