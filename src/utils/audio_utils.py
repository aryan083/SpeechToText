import librosa
import numpy as np
import soundfile as sf
from typing import Union, Tuple, Optional
import logging
import os
from pathlib import Path

class AudioProcessor:
    """
    Audio processing utilities for speech-to-text preprocessing.
    Optimizes audio for better transcription accuracy.
    """
    
    def __init__(self, target_sr: int = 16000):
        self.target_sr = target_sr
        self.logger = logging.getLogger(__name__)
    
    def preprocess_audio(self, audio_input: Union[str, np.ndarray], 
                        normalize: bool = True, 
                        trim_silence: bool = True,
                        noise_reduction: bool = False) -> np.ndarray:
        """
        Preprocess audio for optimal speech recognition.
        
        Args:
            audio_input: Path to audio file or numpy array
            normalize: Whether to normalize audio amplitude
            trim_silence: Whether to trim silence from beginning/end
            noise_reduction: Whether to apply basic noise reduction
            
        Returns:
            Preprocessed audio as numpy array
        """
        try:
            # Load audio if it's a file path
            if isinstance(audio_input, str):
                audio, sr = librosa.load(audio_input, sr=self.target_sr)
            else:
                audio = audio_input
                sr = self.target_sr
            
            # Resample if needed
            if sr != self.target_sr:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=self.target_sr)
            
            # Normalize audio
            if normalize:
                audio = librosa.util.normalize(audio)
            
            # Trim silence
            if trim_silence:
                audio, _ = librosa.effects.trim(audio, top_db=20)
            
            # Basic noise reduction using spectral gating
            if noise_reduction:
                audio = self._reduce_noise(audio)
            
            # Ensure audio is not empty
            if len(audio) == 0:
                self.logger.warning("Audio is empty after preprocessing")
                return np.zeros(1024)  # Return minimal audio
            
            return audio
            
        except Exception as e:
            self.logger.error(f"Audio preprocessing error: {e}")
            # Return original audio or minimal fallback
            if isinstance(audio_input, np.ndarray):
                return audio_input
            else:
                return np.zeros(1024)
    
    def _reduce_noise(self, audio: np.ndarray, noise_factor: float = 0.1) -> np.ndarray:
        """
        Simple noise reduction using spectral subtraction.
        
        Args:
            audio: Input audio signal
            noise_factor: Factor for noise reduction (0.0 to 1.0)
            
        Returns:
            Noise-reduced audio
        """
        try:
            # Compute STFT
            stft = librosa.stft(audio)
            magnitude = np.abs(stft)
            phase = np.angle(stft)
            
            # Estimate noise from first few frames
            noise_frames = min(10, magnitude.shape[1] // 4)
            noise_profile = np.mean(magnitude[:, :noise_frames], axis=1, keepdims=True)
            
            # Spectral subtraction
            clean_magnitude = magnitude - noise_factor * noise_profile
            clean_magnitude = np.maximum(clean_magnitude, 0.1 * magnitude)
            
            # Reconstruct audio
            clean_stft = clean_magnitude * np.exp(1j * phase)
            clean_audio = librosa.istft(clean_stft)
            
            return clean_audio
            
        except Exception as e:
            self.logger.warning(f"Noise reduction failed: {e}")
            return audio
    
    def validate_audio(self, audio_path: str) -> Tuple[bool, str]:
        """
        Validate audio file for processing.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Tuple of (is_valid, message)
        """
        try:
            if not os.path.exists(audio_path):
                return False, "Audio file does not exist"
            
            # Check file size
            file_size = os.path.getsize(audio_path)
            if file_size == 0:
                return False, "Audio file is empty"
            
            if file_size > 100 * 1024 * 1024:  # 100MB limit
                return False, "Audio file too large (>100MB)"
            
            # Try to load audio
            try:
                audio, sr = librosa.load(audio_path, duration=1.0)  # Load first second
                if len(audio) == 0:
                    return False, "Audio file contains no audio data"
            except Exception as e:
                return False, f"Cannot load audio file: {str(e)}"
            
            return True, "Audio file is valid"
            
        except Exception as e:
            return False, f"Audio validation error: {str(e)}"
    
    def get_audio_info(self, audio_path: str) -> dict:
        """
        Get information about audio file.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary with audio information
        """
        try:
            # Get file info
            file_size = os.path.getsize(audio_path)
            
            # Load audio to get properties
            audio, sr = librosa.load(audio_path, sr=None)
            duration = len(audio) / sr
            
            return {
                "file_path": audio_path,
                "file_size_mb": file_size / (1024 * 1024),
                "duration_seconds": duration,
                "sample_rate": sr,
                "channels": 1 if audio.ndim == 1 else audio.shape[0],
                "samples": len(audio),
                "format": Path(audio_path).suffix.lower()
            }
            
        except Exception as e:
            return {
                "error": f"Cannot get audio info: {str(e)}"
            }
    
    def convert_audio_format(self, input_path: str, output_path: str, 
                           target_format: str = "wav") -> bool:
        """
        Convert audio to different format.
        
        Args:
            input_path: Input audio file path
            output_path: Output audio file path
            target_format: Target format (wav, mp3, flac, etc.)
            
        Returns:
            Success status
        """
        try:
            # Load audio
            audio, sr = librosa.load(input_path, sr=self.target_sr)
            
            # Save in target format
            sf.write(output_path, audio, sr, format=target_format.upper())
            
            return True
            
        except Exception as e:
            self.logger.error(f"Audio conversion error: {e}")
            return False
    
    def split_audio(self, audio_path: str, chunk_duration: int = 30) -> list:
        """
        Split long audio into chunks for processing.
        
        Args:
            audio_path: Path to audio file
            chunk_duration: Duration of each chunk in seconds
            
        Returns:
            List of audio chunks as numpy arrays
        """
        try:
            # Load full audio
            audio, sr = librosa.load(audio_path, sr=self.target_sr)
            
            # Calculate chunk size in samples
            chunk_samples = chunk_duration * sr
            
            # Split audio into chunks
            chunks = []
            for i in range(0, len(audio), chunk_samples):
                chunk = audio[i:i + chunk_samples]
                if len(chunk) > sr:  # Only include chunks longer than 1 second
                    chunks.append(chunk)
            
            return chunks
            
        except Exception as e:
            self.logger.error(f"Audio splitting error: {e}")
            return []
