import datetime
import hashlib
from bson.objectid import ObjectId
from pymongo.errors import DuplicateKeyError
from utils.db import get_db
import bcrypt


class AuthService:
    def __init__(self):
        self.db = get_db()
        self.users = self.db['users']
        # ensure unique index on email
        try:
            self.users.create_index('email', unique=True)
        except Exception:
            pass

    def hash_password(self, password: str) -> bytes:
        return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

    def check_password(self, password: str, pw_hash: bytes) -> bool:
        return bcrypt.checkpw(password.encode('utf-8'), pw_hash)

    def signup(self, email: str, password: str) -> dict:
        now = datetime.datetime.now(datetime.timezone.utc)
        pw_hash = self.hash_password(password)
        user = {
            'email': email.lower(),
            'password_hash': pw_hash,
            'created_at': now,
        }
        try:
            res = self.users.insert_one(user)
            user['_id'] = res.inserted_id
            # don't return password hash
            del user['password_hash']
            return user
        except DuplicateKeyError:
            raise ValueError('email_exists')

    def login(self, email: str, password: str) -> dict:
        user = self.users.find_one({'email': email.lower()})
        if not user:
            raise ValueError('invalid_credentials')
        if not self.check_password(password, user['password_hash']):
            raise ValueError('invalid_credentials')
        # hide password_hash
        user_out = {k: v for k, v in user.items() if k != 'password_hash'}
        return user_out

    def save_prediction_history(self, user_id, record: dict):
        """Save a prediction record for the user. record is a dict with arbitrary fields."""
        coll = self.db['predictions']
        # Try to include the user's email for easier querying / display
        uid_obj = ObjectId(user_id) if not isinstance(user_id, ObjectId) else user_id
        user_doc = self.users.find_one({'_id': uid_obj}, {'email': 1})
        user_email = user_doc.get('email') if user_doc else None

        # store a timezone-aware UTC timestamp so clients can reliably interpret it
        doc = {
            'user_id': uid_obj,
            'user_email': user_email,
            'timestamp': datetime.datetime.now(datetime.timezone.utc),
            **record,
        }
        res = coll.insert_one(doc)
        # return inserted id and created timestamp for caller convenience
        return {
            'id': str(res.inserted_id),
            'created_at': doc['timestamp'] if 'timestamp' in doc else None
        }

    def get_prediction_history(self, user_id, limit=50, skip=0):
        coll = self.db['predictions']
        q = {'user_id': ObjectId(user_id) if not isinstance(user_id, ObjectId) else user_id}
        cursor = coll.find(q).sort('timestamp', -1).skip(int(skip)).limit(int(limit))
        results = []
        for d in cursor:
            d['id'] = str(d.pop('_id'))
            # Ensure timestamp is serialized with explicit timezone info (UTC)
            try:
                ts = d.get('timestamp')
                if isinstance(ts, datetime.datetime):
                    # convert to UTC and output an ISO string with offset (+00:00)
                    d['timestamp'] = ts.astimezone(datetime.timezone.utc).isoformat()
                else:
                    d['timestamp'] = str(ts)
            except Exception:
                d['timestamp'] = str(d.get('timestamp'))
            # convert ObjectId user_id to str
            if isinstance(d.get('user_id'), ObjectId):
                d['user_id'] = str(d['user_id'])
            results.append(d)
        return results


auth_service = AuthService()
