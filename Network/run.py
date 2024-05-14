from flask import Flask
import os
import sys
from configs.config import config_dict
from app import create_app

from flaskext.markdown import Markdown

get_config_mode = os.environ.get('GENTELELLA_CONFIG_MODE', 'Debug')

try:
    config_mode = config_dict[get_config_mode.capitalize()]
except KeyError:
    sys.exit('Error: Invalid GENTELELLA_CONFIG_MODE environment variable entry.')

app = create_app(config_mode)
Markdown(app)


#@app.route('/')
#def hello_world():  # put application's code here
#    return 'Hello World!'


if __name__ == '__main__':
    #app.run(debug=True, port=1000)
    #app.run()
    run_simple('localhost', 5000, app,use_reloader=True, use_debugger=True, use_evalex=True)
