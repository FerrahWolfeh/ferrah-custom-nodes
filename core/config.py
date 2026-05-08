import os
import json

class Config:
    _instance = None
    _config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.json")

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            cls._instance._load_config()
        return cls._instance

    def _load_config(self):
        if os.path.exists(self._config_path):
            try:
                with open(self._config_path, 'r') as f:
                    self.data = json.load(f)
            except Exception as e:
                print(f"FerrahNodes: Error loading config.json: {e}")
                self.data = {}
        else:
            self.data = {}
            # Optionally create default config if it doesn't exist
            self._save_config()

    def _save_config(self):
        try:
            with open(self._config_path, 'w') as f:
                json.dump(self.data, f, indent=4)
        except Exception as e:
            print(f"FerrahNodes: Error saving config.json: {e}")

    def get(self, key, default=None):
        return self.data.get(key, default)

    @property
    def immich_url(self):
        url = self.get("immich_url", "http://127.0.0.1:2283/api")
        return url.rstrip('/')

    @property
    def immich_api_key(self):
        return self.get("immich_api_key", "")

config = Config()
