from pymongo import MongoClient
from config.settings import MONGO_URI

_client = None
_db = None

def get_db():
    global _client, _db
    if _db is None:
        _client = MongoClient(MONGO_URI)
        # get_default_database reads DB from URI when provided (e.g. mongodb://host:port/dbname)
        try:
            _db = _client.get_default_database()
        except Exception:
            # fallback to database name 'neurosense'
            _db = _client['neurosense']
    return _db
