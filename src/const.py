import os

# load key=value pairs from a .env file at the repo root, if present,
# without overriding any variable already set in the shell environment
_env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
if os.path.exists(_env_path):
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if not _line or _line.startswith("#") or "=" not in _line:
                continue
            _key, _, _value = _line.partition("=")
            os.environ.setdefault(_key.strip(), _value.strip())

# about habitat scene
INVALID_SCENE_ID = []

# about chatgpt api
END_POINT = "https://api.openai.com/v1"
OPENAI_KEY = os.environ.get("OPENAI_API_KEY")

