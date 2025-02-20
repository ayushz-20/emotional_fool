from pathlib import Path

class Config:
    DATA_DIR = Path("Data")
    CHAT_LOG_PATH = DATA_DIR / "Chatlog.json"
    DATABASE_PATH = Path("Database.data")
    RESPONSES_PATH = Path("Responses.data")
    IMAGE_GENERATION_PATH = Path("Frontend/Files/ImageGeneration.data")
    
    LOGGING_CONFIG = {
        "version": 1,
        "formatters": {
            "standard": {
                "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
            },
        },
        "handlers": {
            "default": {
                "level": "INFO",
                "formatter": "standard",
                "class": "logging.StreamHandler",
            },
        },
        "loggers": {
            "": {
                "handlers": ["default"],
                "level": "INFO",
                "propagate": True
            },
        }
    }