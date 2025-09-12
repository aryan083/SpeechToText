#!/usr/bin/env python3
"""
Model downloader for Hugging Face Spaces deployment.
Downloads and caches essential speech-to-text models.
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Optional
import torch
from transformers import (
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
    pipeline
)
from huggingface_hub import snapshot_download, login
import warnings

warnings.filterwarnings("ignore")

class ModelDownloader:
    """Downloads and manages speech-to-text models for Spaces deployment."""
    
    def __init__(self, cache_dir: str = "/tmp/models", use_auth_token: Optional[str] = None):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Login to HuggingFace if token provided
        if use_auth_token:
            try:
                login(token=use_auth_token)
                self.logger.info("✅ Logged in to HuggingFace Hub")
            except Exception as e:
                self.logger.warning(f"⚠️ HuggingFace login failed: {e}")
        
        # Model configurations optimized for Spaces
        self.model_configs = {
            "distil-whisper": {
                "model_id": "distil-whisper/distil-large-v3",
                "type": "whisper",
                "priority": 1,  # Highest priority for Spaces
                "size_mb": 769
            },
            "whisper-small": {
                "model_id": "openai/whisper-small",
                "type": "whisper", 
                "priority": 2,
                "size_mb": 244
            },
            "whisper-free": {
                "model_id": "openai/whisper-large-v3",
                "type": "whisper",
                "priority": 3,
                "size_mb": 1550
            },
            "wav2vec2-hindi": {
                "model_id": "ai4bharat/indicwav2vec-hindi",
                "type": "wav2vec2",
                "priority": 4,
                "size_mb": 300
            },
            "wav2vec2-improved": {
                "model_id": "yash072/wav2vec2-large-XLSR-Hindi-YashR",
                "type": "wav2vec2",
                "priority": 5,
                "size_mb": 300
            }
        }
    
    def download_model(self, model_name: str, force_download: bool = False) -> bool:
        """Download a specific model."""
        if model_name not in self.model_configs:
            self.logger.error(f"❌ Unknown model: {model_name}")
            return False
        
        config = self.model_configs[model_name]
        model_id = config["model_id"]
        model_type = config["type"]
        
        self.logger.info(f"🔄 Downloading {model_name} ({model_id})...")
        
        try:
            if model_type == "whisper":
                return self._download_whisper_model(model_id, force_download)
            elif model_type == "wav2vec2":
                return self._download_wav2vec2_model(model_id, force_download)
            else:
                return self._download_generic_model(model_id, force_download)
                
        except Exception as e:
            self.logger.error(f"❌ Failed to download {model_name}: {e}")
            return False
    
    def _download_whisper_model(self, model_id: str, force_download: bool) -> bool:
        """Download Whisper-based models."""
        try:
            # Ensure cache directory exists and has proper permissions
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            os.chmod(str(self.cache_dir), 0o777)
            
            # Download using pipeline (automatically handles model and processor)
            pipe = pipeline(
                "automatic-speech-recognition",
                model=model_id,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device="cpu",  # Use CPU for downloading to avoid GPU memory issues
                model_kwargs={
                    "cache_dir": str(self.cache_dir),
                    "use_safetensors": True,
                    "force_download": force_download
                }
            )
            
            # Test the model with a dummy input to ensure it's working
            import numpy as np
            dummy_audio = np.zeros(16000)  # 1 second of silence at 16kHz
            _ = pipe(dummy_audio)
            
            self.logger.info(f"✅ Successfully downloaded and tested {model_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Whisper model download failed: {e}")
            return False
    
    def _download_wav2vec2_model(self, model_id: str, force_download: bool) -> bool:
        """Download Wav2Vec2 models."""
        try:
            # Download model
            model = Wav2Vec2ForCTC.from_pretrained(
                model_id,
                cache_dir=str(self.cache_dir),
                force_download=force_download
            )
            
            # Download processor
            processor = Wav2Vec2Processor.from_pretrained(
                model_id,
                cache_dir=str(self.cache_dir),
                force_download=force_download
            )
            
            # Test the model
            import numpy as np
            dummy_audio = np.zeros(16000)
            input_values = processor(
                dummy_audio,
                return_tensors="pt",
                sampling_rate=16000
            ).input_values
            
            with torch.no_grad():
                _ = model(input_values).logits
            
            self.logger.info(f"✅ Successfully downloaded and tested {model_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Wav2Vec2 model download failed: {e}")
            return False
    
    def _download_generic_model(self, model_id: str, force_download: bool) -> bool:
        """Download generic models using snapshot_download."""
        try:
            snapshot_download(
                repo_id=model_id,
                cache_dir=str(self.cache_dir),
                force_download=force_download
            )
            
            self.logger.info(f"✅ Successfully downloaded {model_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Generic model download failed: {e}")
            return False
    
    def download_models(self, model_names: List[str], force_download: bool = False) -> Dict[str, bool]:
        """Download multiple models."""
        results = {}
        
        # Sort by priority (lower number = higher priority)
        sorted_models = sorted(
            model_names,
            key=lambda x: self.model_configs.get(x, {}).get("priority", 999)
        )
        
        self.logger.info(f"📦 Downloading {len(sorted_models)} models in priority order...")
        
        for model_name in sorted_models:
            if model_name in self.model_configs:
                success = self.download_model(model_name, force_download)
                results[model_name] = success
                
                if success:
                    size_mb = self.model_configs[model_name]["size_mb"]
                    self.logger.info(f"✅ {model_name} downloaded ({size_mb}MB)")
                else:
                    self.logger.error(f"❌ {model_name} download failed")
            else:
                self.logger.warning(f"⚠️ Unknown model: {model_name}")
                results[model_name] = False
        
        successful = sum(1 for success in results.values() if success)
        self.logger.info(f"📊 Download summary: {successful}/{len(model_names)} models successful")
        
        return results
    
    def download_essential_models(self, force_download: bool = False) -> Dict[str, bool]:
        """Download essential models for Spaces deployment."""
        essential_models = ["distil-whisper", "whisper-small"]
        self.logger.info("🚀 Downloading essential models for Spaces...")
        return self.download_models(essential_models, force_download)
    
    def get_cache_size(self) -> float:
        """Get total size of cached models in MB."""
        total_size = 0
        for path in self.cache_dir.rglob("*"):
            if path.is_file():
                total_size += path.stat().st_size
        return total_size / (1024 * 1024)  # Convert to MB
    
    def list_cached_models(self) -> List[str]:
        """List all cached models."""
        cached_models = []
        for model_dir in self.cache_dir.iterdir():
            if model_dir.is_dir():
                cached_models.append(model_dir.name)
        return cached_models
    
    def cleanup_cache(self, keep_essential: bool = True) -> None:
        """Clean up model cache, optionally keeping essential models."""
        if keep_essential:
            essential_models = ["distil-whisper", "openai--whisper-small"]
            for model_dir in self.cache_dir.iterdir():
                if model_dir.is_dir() and not any(essential in model_dir.name for essential in essential_models):
                    import shutil
                    shutil.rmtree(model_dir)
                    self.logger.info(f"🗑️ Removed {model_dir.name}")
        else:
            import shutil
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self.logger.info("🗑️ Cleared entire model cache")

def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Download speech-to-text models for Spaces")
    parser.add_argument(
        "--models",
        type=str,
        default="distil-whisper,whisper-small",
        help="Comma-separated list of models to download"
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="/tmp/models",
        help="Directory to cache models"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if models exist"
    )
    parser.add_argument(
        "--essential-only",
        action="store_true",
        help="Download only essential models for Spaces"
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Clean up model cache"
    )
    
    args = parser.parse_args()
    
    # Initialize downloader
    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    downloader = ModelDownloader(
        cache_dir=args.cache_dir,
        use_auth_token=hf_token
    )
    
    if args.cleanup:
        downloader.cleanup_cache()
        return
    
    # Download models
    if args.essential_only:
        results = downloader.download_essential_models(args.force)
    else:
        model_list = [m.strip() for m in args.models.split(",")]
        results = downloader.download_models(model_list, args.force)
    
    # Print summary
    successful = sum(1 for success in results.values() if success)
    total = len(results)
    cache_size = downloader.get_cache_size()
    
    print(f"\n📊 Download Summary:")
    print(f"✅ Successful: {successful}/{total}")
    print(f"💾 Cache size: {cache_size:.1f} MB")
    print(f"📁 Cache location: {args.cache_dir}")
    
    if successful == 0:
        sys.exit(1)

if __name__ == "__main__":
    main()
