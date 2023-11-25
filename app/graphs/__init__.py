from flask import Blueprint
#import mimetypes
#mimetypes.add_type('application/javascript', '.js')
#mimetypes.add_type('text/css', '.css')
blueprint = Blueprint(
    'graphs_blueprint',
    __name__,
    url_prefix='/sensoranalysis/',
    template_folder='templates',
    static_folder='static'
)