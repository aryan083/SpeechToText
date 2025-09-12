import torch
import librosa
import tensorflow as tf
from transformers import (
    AutoModelForSpeechSeq2Seq, 
    AutoProcessor, 
    pipeline,
    Wav2Vec2ForCTC, 
    Wav2Vec2Processor
)
import warnings
import logging
import os
from typing import List, Dict, Optional, Union
import numpy as np

warnings.filterwarnings("ignore")

class FreeIndianSpeechToText:
    """
    Complete Speech-to-Text implementation for Indian languages using free open-source models.
    Supports multiple model architectures optimized for different use cases.
    """
    
    def __init__(self, model_type: str = "distil-whisper", language: str = "hindi", cache_dir: str = "./models"):
        self.language = language
        self.cache_dir = cache_dir
        self.device = "cuda" if torch.cuda.is_available() and os.getenv("ENABLE_GPU", "True") == "True" else "cpu"
        self.torch_dtype = torch.float16 if self.device == "cuda" else torch.float32
        
        # Configure logging
        logging.basicConfig(level=getattr(logging, os.getenv("LOG_LEVEL", "INFO")))
        self.logger = logging.getLogger(__name__)
        
        # Free model configurations with performance metrics
        self.model_configs = {
            "distil-whisper": {
                "model_id": "distil-whisper/distil-large-v3",
                "type": "whisper",
                "description": "6x faster than Whisper, 49% smaller, <1% WER difference",
                "languages": 99,
                "size": "769M"
            },
            "whisper-free": {
                "model_id": "openai/whisper-large-v3", 
                "type": "whisper",
                "description": "Best accuracy, supports 99 languages",
                "languages": 99,
                "size": "1550M"
            },
            "whisper-small": {
                "model_id": "openai/whisper-small",
                "type": "whisper", 
                "description": "Balanced performance, good for CPU",
                "languages": 99,
                "size": "244M"
            },
            "wav2vec2-hindi": {
                "model_id": "ai4bharat/indicwav2vec-hindi",
                "type": "wav2vec2",
                "description": "Specialized for Hindi, AI4Bharat model",
                "languages": 1,
                "size": "300M"
            },
            "wav2vec2-improved": {
                "model_id": "yash072/wav2vec2-large-XLSR-Hindi-YashR",
                "type": "wav2vec2",
                "description": "Improved Hindi model, 54% WER",
                "languages": 1,
                "size": "300M"
            },
            "wav2vec2-multilang": {
                "model_id": "theainerd/Wav2Vec2-large-xlsr-hindi",
                "type": "wav2vec2",
                "description": "Multi-language Wav2Vec2 for Hindi",
                "languages": 1,
                "size": "300M"
            },
            "seamless": {
                "model_id": "facebook/seamless-m4t-v2-large",
                "type": "seamless",
                "description": "Meta's unified model, 101 languages",
                "languages": 101,
                "size": "2.3B"
            },
            "speecht5": {
                "model_id": "microsoft/speecht5_asr", 
                "type": "speecht5",
                "description": "Microsoft's unified speech model",
                "languages": 10,
                "size": "200M"
            }
        }
        
        self.load_model(model_type)
    
    def load_model(self, model_type: str) -> None:
        """Load the specified model with TensorFlow optimization."""
        if model_type not in self.model_configs:
            raise ValueError(f"Model type '{model_type}' not supported. Available: {list(self.model_configs.keys())}")
        
        config = self.model_configs[model_type]
        self.model_id = config["model_id"]
        self.model_type = config["type"]
        self.current_model_name = model_type
        
        self.logger.info(f"Loading {model_type} model: {self.model_id}")
        self.logger.info(f"Description: {config['description']}")
        
        try:
            if self.model_type == "whisper":
                self._load_whisper_model()
            elif self.model_type == "wav2vec2":
                self._load_wav2vec2_model()
            elif self.model_type in ["seamless", "speecht5"]:
                self._load_pipeline_model()
                
            self.logger.info(f"Successfully loaded {model_type} on {self.device}")
            
        except Exception as e:
            self.logger.error(f"Error loading model {model_type}: {str(e)}")
            raise
    
    def _load_whisper_model(self) -> None:
        """Load Whisper-based models with optimization."""
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model_id,
            torch_dtype=self.torch_dtype,
            device=self.device,
            model_kwargs={"cache_dir": self.cache_dir, "use_safetensors": True},
            return_timestamps=True
        )
    
    def _load_wav2vec2_model(self) -> None:
        """Load Wav2Vec2 models with TensorFlow compatibility."""
        self.model = Wav2Vec2ForCTC.from_pretrained(
            self.model_id, 
            cache_dir=self.cache_dir
        ).to(self.device)
        self.processor = Wav2Vec2Processor.from_pretrained(
            self.model_id,
            cache_dir=self.cache_dir
        )
        
        # Enable TensorFlow optimization if available
        if hasattr(self.model, 'to_tf'):
            try:
                self.model = self.model.to_tf()
                self.logger.info("Converted model to TensorFlow for optimization")
            except Exception as e:
                self.logger.warning(f"TensorFlow conversion failed: {e}")
    
    def _load_pipeline_model(self) -> None:
        """Load pipeline-based models."""
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model_id,
            device=self.device,
            model_kwargs={"cache_dir": self.cache_dir}
        )
    
    def transcribe(self, audio_input: Union[str, np.ndarray], language_code: str = "hi") -> Dict:
        """
        Transcribe audio to text with detailed results.
        
        Args:
            audio_input: Path to audio file or numpy array
            language_code: Language code (hi=Hindi, ta=Tamil, bn=Bengali, etc.)
            
        Returns:
            Dictionary with transcription results and metadata
        """
        try:
            start_time = torch.cuda.Event(enable_timing=True) if self.device == "cuda" else None
            end_time = torch.cuda.Event(enable_timing=True) if self.device == "cuda" else None
            
            if start_time:
                start_time.record()
            
            if self.model_type == "whisper":
                result = self._transcribe_whisper(audio_input, language_code)
            elif self.model_type == "wav2vec2":
                result = self._transcribe_wav2vec2(audio_input)
            else:
                result = self._transcribe_pipeline(audio_input)
            
            if end_time:
                end_time.record()
                torch.cuda.synchronize()
                processing_time = start_time.elapsed_time(end_time) / 1000.0
            else:
                processing_time = 0.0
            
            return {
                "text": result,
                "model": self.current_model_name,
                "language": language_code,
                "processing_time": processing_time,
                "device": self.device,
                "success": True
            }
            
        except Exception as e:
            self.logger.error(f"Transcription error: {str(e)}")
            return {
                "text": "",
                "error": str(e),
                "model": self.current_model_name,
                "success": False
            }
    
    def _transcribe_whisper(self, audio_input: Union[str, np.ndarray], language_code: str) -> str:
        """Transcribe using Whisper-based models."""
        generate_kwargs = {}
        
        if language_code != "en":
            language_name = self._get_language_name(language_code)
            generate_kwargs = {
                "language": language_name,
                "task": "transcribe"
            }
        
        result = self.pipe(audio_input, generate_kwargs=generate_kwargs)
        
        # Handle different return formats
        if isinstance(result, dict):
            return result.get("text", "")
        elif isinstance(result, list) and len(result) > 0:
            return result[0].get("text", "")
        else:
            return str(result)
    
    def _transcribe_wav2vec2(self, audio_input: Union[str, np.ndarray]) -> str:
        """Transcribe using Wav2Vec2 models."""
        if isinstance(audio_input, str):
            audio, sr = librosa.load(audio_input, sr=16000)
        else:
            audio = audio_input
        
        input_values = self.processor(
            audio, 
            return_tensors="pt", 
            sampling_rate=16000
        ).input_values.to(self.device)
        
        with torch.no_grad():
            logits = self.model(input_values).logits
        
        prediction_ids = torch.argmax(logits, dim=-1)
        transcription = self.processor.batch_decode(prediction_ids)[0]
        
        return transcription
    
    def _transcribe_pipeline(self, audio_input: Union[str, np.ndarray]) -> str:
        """Transcribe using pipeline models."""
        result = self.pipe(audio_input)
        
        if isinstance(result, dict):
            return result.get("text", "")
        else:
            return str(result)
    
    def batch_transcribe(self, audio_paths: List[str], language_code: str = "hi") -> List[Dict]:
        """Transcribe multiple audio files efficiently."""
        results = []
        
        self.logger.info(f"Starting batch transcription of {len(audio_paths)} files")
        
        for i, audio_path in enumerate(audio_paths):
            self.logger.info(f"Processing file {i+1}/{len(audio_paths)}: {audio_path}")
            
            try:
                result = self.transcribe(audio_path, language_code)
                result["file"] = audio_path
                results.append(result)
            except Exception as e:
                results.append({
                    "file": audio_path, 
                    "error": str(e),
                    "success": False
                })
        
        return results
    
    def get_model_info(self) -> Dict:
        """Get information about the current model."""
        config = self.model_configs[self.current_model_name]
        return {
            "name": self.current_model_name,
            "model_id": self.model_id,
            "type": self.model_type,
            "description": config["description"],
            "languages_supported": config["languages"],
            "model_size": config["size"],
            "device": self.device,
            "torch_dtype": str(self.torch_dtype)
        }
    
    def get_available_models(self) -> Dict:
        """Get list of all available models."""
        return {name: config for name, config in self.model_configs.items()}
    
    def switch_model(self, model_type: str) -> bool:
        """Switch to a different model."""
        try:
            self.load_model(model_type)
            return True
        except Exception as e:
            self.logger.error(f"Failed to switch to model {model_type}: {e}")
            return False
    
    def _get_language_name(self, code: str) -> str:
        """Convert language code to language name for Whisper models."""
        lang_map = {
            "hi": "hindi",
            "ta": "tamil", 
            "bn": "bengali",
            "te": "telugu",
            "mr": "marathi",
            "gu": "gujarati",
            "kn": "kannada",
            "ml": "malayalam",
            "pa": "punjabi",
            "or": "odia",
            "as": "assamese",
            "ur": "urdu",
            "en": "english"
        }
        return lang_map.get(code, "hindi")
    
    def preprocess_audio(self, audio_path: str, target_sr: int = 16000) -> np.ndarray:
        """Preprocess audio file for optimal transcription."""
        try:
            # Load audio
            audio, sr = librosa.load(audio_path, sr=target_sr)
            
            # Normalize audio
            audio = librosa.util.normalize(audio)
            
            # Remove silence
            audio, _ = librosa.effects.trim(audio, top_db=20)
            
            return audio
            
        except Exception as e:
            self.logger.error(f"Audio preprocessing error: {e}")
            raise
    
    def get_supported_languages(self) -> List[str]:
        """Get list of supported Indian languages."""
        return [
            "hindi", "tamil", "bengali", "telugu", "marathi", 
            "gujarati", "kannada", "malayalam", "punjabi", "odia",
            "assamese", "urdu", "english"
        ]
