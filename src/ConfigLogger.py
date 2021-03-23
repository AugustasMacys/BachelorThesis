import logging
from logging.handlers import RotatingFileHandler

handler = RotatingFileHandler(filename='../logs/coviar_combined_testing.log', maxBytes=20000000, backupCount=10)
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s, %(name)s, %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    handlers=[handler])
