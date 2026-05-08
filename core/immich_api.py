import requests
import json
from .config import config

class ImmichAPI:
    def __init__(self):
        self.url = config.immich_url
        self.api_key = config.immich_api_key

    def _get_headers(self):
        return {
            'Accept': 'application/json',
            'x-api-key': self.api_key
        }

    def get_albums(self):
        if not self.url or not self.api_key:
            return {"error": "Immich URL or API Key not configured"}
        
        try:
            response = requests.get(f"{self.url}/albums", headers=self._get_headers(), timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}

    def upload_asset(self, file_name, file_data, mime_type, created_at=None):
        if not self.url or not self.api_key:
            return {"error": "Immich URL or API Key not configured"}

        data = {
            'deviceAssetId': f"{file_name}-{created_at.timestamp() if created_at else 'now'}",
            'deviceId': 'ComfyUI',
            'fileCreatedAt': created_at.isoformat() if created_at else '',
            'fileModifiedAt': created_at.isoformat() if created_at else '',
            'isFavorite': 'false',
        }
        files = {'assetData': (file_name, file_data, mime_type)}

        try:
            response = requests.post(f'{self.url}/assets', data=data, files=files, headers=self._get_headers(), timeout=30)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}

    def add_to_album(self, album_id, asset_id):
        if not self.url or not self.api_key:
            return {"error": "Immich URL or API Key not configured"}

        try:
            url = f'{self.url}/albums/{album_id}/assets'
            data = {"ids": [asset_id]}
            response = requests.put(url, json=data, headers=self._get_headers(), timeout=30)
            response.raise_for_status()
            return {"status": "success"}
        except Exception as e:
            return {"error": str(e)}

immich_api = ImmichAPI()
