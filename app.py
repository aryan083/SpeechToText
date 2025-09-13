#!/usr/bin/env python3
"""
Hugging Face Spaces optimized version of the Indian Speech-to-Text application.
This version is specifically configured for deployment on Hugging Face Spaces.
"""
import os
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv

# Explicitly load .env from ./config/env/.env
env_path = Path(__file__).parent / "config" / "env" / ".env"
load_dotenv(dotenv_path=env_path, override=True)


# Set up environment for Spaces
os.environ['APP_ENV'] = 'prod'
os.environ['GRADIO_SERVER_NAME'] = '0.0.0.0'
os.environ['GRADIO_SERVER_PORT'] = '7860'
os.environ['MODEL_CACHE_DIR'] = '/app/models'
os.environ['HF_HOME'] = '/app/models'
os.environ['TRANSFORMERS_CACHE'] = '/app/models'
os.environ['TORCH_HOME'] = '/app/models'
os.environ['XDG_CACHE_HOME'] = '/app/models'
os.environ['HF_TOKEN'] = os.getenv('HF_TOKEN') or os.getenv('HUGGINGFACE_HUB_TOKEN') or ""

# Add src to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

# Configure logging for Spaces
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_spaces_environment():
    """Set up the environment specifically for Hugging Face Spaces."""
    
    # Create model cache directory
    model_cache_dir = Path(os.environ['MODEL_CACHE_DIR'])
    model_cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Set Hugging Face token if available
    hf_token = os.getenv('HF_TOKEN') or os.getenv('HUGGINGFACE_HUB_TOKEN')
    if hf_token:
        os.environ['HF_TOKEN'] = hf_token
        logger.info("✅ HuggingFace token found")
    else:
        logger.warning("⚠️ No HuggingFace token found - some models may not be accessible")
    
    # GPU detection
    try:
        import torch
        if torch.cuda.is_available():
            logger.info(f"✅ GPU detected: {torch.cuda.get_device_name(0)}")
            os.environ['ENABLE_GPU'] = 'True'
        else:
            logger.info("💻 Running on CPU")
            os.environ['ENABLE_GPU'] = 'False'
    except ImportError:
        logger.warning("⚠️ PyTorch not available")
        os.environ['ENABLE_GPU'] = 'False'

def download_essential_models():
    """Download essential models for Spaces deployment."""
    try:
        from scripts.download_models import ModelDownloader
        
        logger.info("🔄 Downloading essential models for Spaces...")
        
        downloader = ModelDownloader(
            cache_dir=os.environ['MODEL_CACHE_DIR'],
            use_auth_token=os.getenv('HF_TOKEN')
        )
        
        # Download only essential models for Spaces (to save space and time)
        essential_models = ["distil-whisper", "whisper-small"]
        
        results = downloader.download_models(essential_models, force_download=False)
        
        successful = sum(1 for success in results.values() if success)
        logger.info(f"✅ Downloaded {successful}/{len(essential_models)} essential models")
        
        return successful > 0
        
    except PermissionError as e:
        logger.error(f"❌ Permission denied for model cache: {e}")
        logger.info("🔄 Using fallback model loading strategy...")
        return True  # Continue with app initialization
    except Exception as e:
        logger.error(f"❌ Error downloading models: {e}")
        return False

def create_spaces_gradio_app():
    """Create Gradio app optimized for Spaces."""
    try:
        from ui.gradio_app import GradioSpeechToTextApp
        
        logger.info("🚀 Initializing Gradio app for Spaces...")
        
        # Create app with Spaces-specific configuration
        app = GradioSpeechToTextApp()
        
        # Create interface with Spaces optimizations
        interface = app.create_interface()
        
        # Add Spaces-specific customizations
        interface.title = "🎤 Indian Speech-to-Text Models"
        interface.description = """
        ## Free Open-Source Speech-to-Text for Indian Languages
        
        This Space showcases multiple free, open-source speech-to-text models optimized for Indian languages.
        
        **Available Models:**
        - Distil-Whisper (6x faster than Whisper)
        - OpenAI Whisper (best accuracy)
        - Wav2Vec2 Hindi (specialized for Hindi)
        
        **Supported Languages:** Hindi, Tamil, Bengali, Telugu, Marathi, Gujarati, Kannada, Malayalam, Punjabi, Odia, Assamese, Urdu, English
        """
        
        return interface
        
    except Exception as e:
        logger.error(f"❌ Error creating Gradio app: {e}")
        raise

def main():
    """Main function for Spaces deployment."""
    logger.info("🎤 Starting Indian Speech-to-Text Models on Hugging Face Spaces...")
    
    # Set up Spaces environment
    setup_spaces_environment()
    
    # Download essential models
    models_available = download_essential_models()
    
    if not models_available:
        logger.warning("⚠️ No models downloaded, but continuing with app initialization...")
    
    # Create and launch Gradio app
    try:
        interface = create_spaces_gradio_app()
        
        logger.info("🌐 Launching Gradio interface...")
        
        # Launch with Spaces-specific settings
        interface.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,  # Spaces handles sharing
            show_error=True,
            quiet=False,
            max_threads=10,     # Limit concurrent threads
        )
        
    except Exception as e:
        logger.error(f"❌ Failed to launch Gradio app: {e}")
        
        # Fallback: Create a simple error page
        import gradio as gr
        
        def error_message():
            return "❌ Application failed to initialize. Please check the logs."
        
        error_interface = gr.Interface(
            fn=error_message,
            inputs=[],
            outputs=gr.Textbox(label="Status"),
            title="🎤 Indian Speech-to-Text Models - Error",
            description="There was an error initializing the application."
        )
        
        error_interface.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False
        )

if __name__ == "__main__":
    main()
