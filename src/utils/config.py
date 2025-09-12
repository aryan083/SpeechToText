import os
from dotenv import load_dotenv
from pathlib import Path
import logging

def load_config():
    """
    Load configuration based on APP_ENV environment variable.
    Follows the user rules for environment-specific config files.
    """
    # Get environment
    app_env = os.getenv('APP_ENV', 'local')
    
    # Construct config file path following the user rules
    config_dir = Path(__file__).parent.parent.parent / 'configs' / 'envs'
    config_file = config_dir / f'.env.{app_env}'
    
    # Load environment variables
    if config_file.exists():
        load_dotenv(config_file)
        print(f"✅ Loaded config from {config_file}")
    else:
        print(f"⚠️ Config file {config_file} not found, using defaults")
        # Load default local config
        default_config = config_dir / '.env.local'
        if default_config.exists():
            load_dotenv(default_config)
    
    # Return configuration dictionary
    config = {
        'APP_ENV': os.getenv('APP_ENV', 'local'),
        'DEBUG': os.getenv('DEBUG', 'True').lower() == 'true',
        'LOG_LEVEL': os.getenv('LOG_LEVEL', 'INFO'),
        'MODEL_CACHE_DIR': os.getenv('MODEL_CACHE_DIR', './models'),
        'GRADIO_SERVER_NAME': os.getenv('GRADIO_SERVER_NAME', '127.0.0.1'),
        'GRADIO_SERVER_PORT': int(os.getenv('GRADIO_SERVER_PORT', 7860)),
        'GRADIO_SHARE': os.getenv('GRADIO_SHARE', 'False').lower() == 'true',
        'DEFAULT_MODEL': os.getenv('DEFAULT_MODEL', 'distil-whisper'),
        'DEFAULT_LANGUAGE': os.getenv('DEFAULT_LANGUAGE', 'hindi'),
        'MAX_AUDIO_LENGTH': int(os.getenv('MAX_AUDIO_LENGTH', 300)),
        'ENABLE_GPU': os.getenv('ENABLE_GPU', 'True').lower() == 'true'
    }
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, config['LOG_LEVEL']),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    return config
