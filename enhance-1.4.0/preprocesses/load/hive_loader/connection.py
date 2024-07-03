from pyhive import hive
from sqlalchemy.engine import create_engine
import logging

class HiveConnection:
    def __init__(self, host, port, username, database):
        self.host = host
        self.port = port
        self.username = username
        self.database = database
        self.connection = None

    def connect(self):
        try:
            self.connection = hive.Connection(host=self.host, port=self.port, username=self.username, database=self.database)
            logging.info("Connected to Hive")
        except Exception as e:
            logging.error(f"Error connecting to Hive: {e}")
            raise

    def get_engine(self):
        if self.connection is None:
            self.connect()
        return create_engine(f'hive://{self.host}:{self.port}/{self.database}')
