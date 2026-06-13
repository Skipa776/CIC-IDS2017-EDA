"""API configuration settings."""

from pathlib import Path
from typing import List
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings."""

    # App info
    app_name: str = "CICIDS2017 IDS API"
    app_version: str = "1.0.0"
    debug: bool = False

    # Model paths
    model_dir: Path = Path(__file__).parent.parent / "models"

    # API settings
    api_prefix: str = "/api/v1"
    allowed_origins: str = "*"

    # Classification settings
    binary_threshold: float = 0.5
    confidence_high_threshold: float = 0.9
    confidence_medium_threshold: float = 0.7

    @property
    def cors_origins(self) -> List[str]:
        """Parse allowed_origins as comma-separated list."""
        return [o.strip() for o in self.allowed_origins.split(",")]

    class Config:
        env_prefix = "IDS_"


settings = Settings()
