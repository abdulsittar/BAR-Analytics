import os
from importlib import import_module
from flask import Flask


def register_blueprints(app):
    #for module_name in ('DashExample', 'home'):
    module = import_module('app.{}.routes'.format('graphs'))
    app.register_blueprint(module.blueprint)

def create_app(config):
    app = Flask(__name__, static_url_path="/static", static_folder= "/graphs/static")
    
    print(['%s' % rule for rule in app.url_map.iter_rules()]);
    app.config.from_object(config)
    register_blueprints(app)
    return app
