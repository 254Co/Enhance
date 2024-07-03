import pandas as pd
import logging

class QueryExecutor:
    def __init__(self, connection):
        self.connection = connection

    def execute_query(self, query):
        try:
            cursor = self.connection.cursor()
            cursor.execute(query)
            results = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]
            df = pd.DataFrame(results, columns=columns)
            logging.info("Query executed successfully")
            return df
        except Exception as e:
            logging.error(f"Error executing query: {e}")
            raise
