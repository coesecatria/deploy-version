"""
Application configuration — loads settings from .env file.
"""

import os
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Load .env from the attendance_ai root (four levels up from backend/app/core/)
_env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), ".env")
load_dotenv(_env_path)


class Settings(BaseSettings):
    MONGO_URI: str = "mongodb://localhost:27017"
    DB_NAME: str = "attendance_ai"
    SIMILARITY_THRESHOLD: float = 0.75
    LOGIN_TIME: str = "09:30:00"
    LOGOUT_TIME: str = "16:30:00"

    # ── CP Plus Camera (RTSP / IP Camera) ──────────────────────────────────────
    # subtype=00 → main stream (full resolution, best quality)
    # subtype=01 → sub stream  (low resolution, less bandwidth)
    USE_IP_CAMERA: bool = True
    IP_CAMERA_HOST: str = "172.16.11.254"
    IP_CAMERA_PORT: int = 1036
    IP_CAMERA_USER: str = "admin"
    IP_CAMERA_PASS: str = "Atria@2026"
    IP_CAMERA_CHANNEL: int = 1   # 1-based channel number
    IP_CAMERA_SUBTYPE: int = 1   # 0 = main stream (high quality), 1 = sub stream

    class Config:
        env_file = _env_path
        extra = "ignore"

    @property
    def ip_camera_url(self) -> str:
        """Build RTSP URL with properly percent-encoded credentials."""
        from urllib.parse import quote
        user = quote(self.IP_CAMERA_USER, safe="")
        passwd = quote(self.IP_CAMERA_PASS, safe="")
        # CP Plus working format verified via debug script (root path required)
        return f"rtsp://{user}:{passwd}@{self.IP_CAMERA_HOST}:{self.IP_CAMERA_PORT}/"


settings = Settings()
