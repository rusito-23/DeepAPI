import sys
import os

sys.path.append(os.path.join(os.getcwd(), 'deep_api'))
from deep_app import create_app

application = create_app()

if __name__ == '__main__':
    application.run()
