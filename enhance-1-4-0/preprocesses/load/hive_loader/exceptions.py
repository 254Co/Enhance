class HiveLoaderException(Exception):
    pass

class ConnectionError(HiveLoaderException):
    pass

class QueryExecutionError(HiveLoaderException):
    pass
