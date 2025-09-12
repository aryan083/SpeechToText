import gradio as gr
import os
import sys
import json
import time
from typing import List, Tuple, Optional
import numpy as np
import librosa
from pathlib import Path

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.speech_to_text import FreeIndianSpeechToText
from utils.config import load_config
from utils.audio_utils import AudioProcessor

class GradioSpeechToTextApp:
    """
    Gradio web interface for Indian Speech-to-Text models.
    Provides an intuitive UI for testing different models and languages.
    """
    
    def __init__(self):
        self.config = load_config()
        self.current_model = None
        self.audio_processor = AudioProcessor()
        self.supported_languages = {
            "Hindi": "hi",
            "Tamil": "ta", 
            "Bengali": "bn",
            "Telugu": "te",
            "Marathi": "mr",
            "Gujarati": "gu",
            "Kannada": "kn",
            "Malayalam": "ml",
            "Punjabi": "pa",
            "Odia": "or",
            "Assamese": "as",
            "Urdu": "ur",
            "English": "en"
        }
        
        # Initialize with default model
        self.initialize_model()
    
    def initialize_model(self):
        """Initialize the default model."""
        try:
            default_model = self.config.get("DEFAULT_MODEL", "distil-whisper")
            self.current_model = FreeIndianSpeechToText(
                model_type=default_model,
                cache_dir=self.config.get("MODEL_CACHE_DIR", "./models")
            )
            return f"✅ Initialized with {default_model} model"
        except Exception as e:
            return f"❌ Error initializing model: {str(e)}"
    
    def transcribe_audio(self, audio_input, model_choice: str, language_choice: str, 
                        enable_preprocessing: bool = True) -> Tuple[str, str, str]:
        """
        Main transcription function for Gradio interface.
        
        Returns:
            Tuple of (transcription_text, model_info, processing_stats)
        """
        if audio_input is None:
            return "❌ No audio provided", "", ""
        
        try:
            # Switch model if needed
            if not self.current_model or self.current_model.current_model_name != model_choice:
                status = self.switch_model(model_choice)
                if not status.startswith("✅"):
                    return f"❌ Model loading failed: {status}", "", ""
            
            # Get language code
            language_code = self.supported_languages.get(language_choice, "hi")
            
            # Preprocess audio if enabled
            if enable_preprocessing:
                try:
                    audio_data = self.audio_processor.preprocess_audio(audio_input)
                except Exception as e:
                    # Fallback to original audio if preprocessing fails
                    audio_data = audio_input
                    print(f"Preprocessing failed, using original: {e}")
            else:
                audio_data = audio_input
            
            # Perform transcription
            start_time = time.time()
            result = self.current_model.transcribe(audio_data, language_code)
            end_time = time.time()
            
            if result["success"]:
                # Format results
                transcription = result["text"]
                
                # Model information
                model_info = self.format_model_info(result)
                
                # Processing statistics
                processing_stats = self.format_processing_stats(result, end_time - start_time)
                
                return transcription, model_info, processing_stats
            else:
                return f"❌ Transcription failed: {result.get('error', 'Unknown error')}", "", ""
                
        except Exception as e:
            return f"❌ Error during transcription: {str(e)}", "", ""
    
    def switch_model(self, model_name: str) -> str:
        """Switch to a different model."""
        try:
            if self.current_model:
                success = self.current_model.switch_model(model_name)
                if success:
                    return f"✅ Switched to {model_name}"
                else:
                    return f"❌ Failed to switch to {model_name}"
            else:
                self.current_model = FreeIndianSpeechToText(
                    model_type=model_name,
                    cache_dir=self.config.get("MODEL_CACHE_DIR", "./models")
                )
                return f"✅ Loaded {model_name}"
        except Exception as e:
            return f"❌ Error switching model: {str(e)}"
    
    def batch_transcribe(self, files: List, model_choice: str, language_choice: str) -> str:
        """Batch transcription for multiple files."""
        if not files:
            return "❌ No files provided"
        
        try:
            # Switch model if needed
            if not self.current_model or self.current_model.current_model_name != model_choice:
                status = self.switch_model(model_choice)
                if not status.startswith("✅"):
                    return f"❌ Model loading failed: {status}"
            
            language_code = self.supported_languages.get(language_choice, "hi")
            
            # Process files
            file_paths = [file.name for file in files]
            results = self.current_model.batch_transcribe(file_paths, language_code)
            
            # Format results
            output = "# Batch Transcription Results\n\n"
            for i, result in enumerate(results, 1):
                if result["success"]:
                    output += f"## File {i}: {Path(result['file']).name}\n"
                    output += f"**Transcription:** {result['text']}\n"
                    output += f"**Processing Time:** {result.get('processing_time', 0):.2f}s\n\n"
                else:
                    output += f"## File {i}: {Path(result['file']).name}\n"
                    output += f"**Error:** {result.get('error', 'Unknown error')}\n\n"
            
            return output
            
        except Exception as e:
            return f"❌ Batch processing error: {str(e)}"
    
    def get_model_comparison(self) -> str:
        """Generate model comparison table."""
        if not self.current_model:
            return "❌ No model loaded"
        
        models = self.current_model.get_available_models()
        
        comparison = "# Available Models Comparison\n\n"
        comparison += "| Model | Type | Size | Languages | Description |\n"
        comparison += "|-------|------|------|-----------|-------------|\n"
        
        for name, config in models.items():
            comparison += f"| {name} | {config['type']} | {config['size']} | {config['languages']} | {config['description']} |\n"
        
        return comparison
    
    def format_model_info(self, result: dict) -> str:
        """Format model information for display."""
        model_info = f"""
**Model:** {result['model']}
**Device:** {result['device']}
**Language:** {result['language']}
"""
        return model_info.strip()
    
    def format_processing_stats(self, result: dict, total_time: float) -> str:
        """Format processing statistics."""
        stats = f"""
**Total Processing Time:** {total_time:.2f}s
**Model Processing Time:** {result.get('processing_time', 0):.2f}s
**Status:** {'✅ Success' if result['success'] else '❌ Failed'}
"""
        return stats.strip()
    
    def create_interface(self) -> gr.Blocks:
        """Create the Gradio interface."""
        
        # Custom CSS for better styling
        css = """
        .gradio-container {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        .model-info {
            background-color: #f0f8ff;
            padding: 10px;
            border-radius: 5px;
            border-left: 4px solid #4CAF50;
        }
        .stats-info {
            background-color: #fff8f0;
            padding: 10px;
            border-radius: 5px;
            border-left: 4px solid #ff9800;
        }
        """
        
        with gr.Blocks(css=css, title="Indian Speech-to-Text Models", theme=gr.themes.Soft()) as interface:
            
            gr.Markdown("""
            # 🎤 Complete Guide to Free Open-Source Speech-to-Text Models for Indian Languages
            
            This application provides access to multiple free, open-source speech-to-text models optimized for Indian languages.
            All models are completely free to use and can be deployed commercially.
            """)
            
            with gr.Tab("🎯 Single Audio Transcription"):
                with gr.Row():
                    with gr.Column(scale=2):
                        # Audio input
                        audio_input = gr.Audio(
                            label="Upload Audio File or Record",
                            type="filepath",
                            sources=["upload", "microphone"]
                        )
                        
                        # Model selection
                        model_choice = gr.Dropdown(
                            choices=[
                                "distil-whisper", "whisper-free", "whisper-small",
                                "wav2vec2-hindi", "wav2vec2-improved", "wav2vec2-multilang",
                                "seamless", "speecht5"
                            ],
                            value="distil-whisper",
                            label="Select Model",
                            info="Choose the speech-to-text model"
                        )
                        
                        # Language selection
                        language_choice = gr.Dropdown(
                            choices=list(self.supported_languages.keys()),
                            value="Hindi",
                            label="Select Language",
                            info="Choose the audio language"
                        )
                        
                        # Preprocessing option
                        enable_preprocessing = gr.Checkbox(
                            value=True,
                            label="Enable Audio Preprocessing",
                            info="Normalize and clean audio for better results"
                        )
                        
                        # Transcribe button
                        transcribe_btn = gr.Button("🎯 Transcribe Audio", variant="primary", size="lg")
                    
                    with gr.Column(scale=3):
                        # Results
                        transcription_output = gr.Textbox(
                            label="Transcription Result",
                            lines=6,
                            placeholder="Transcription will appear here..."
                        )
                        
                        with gr.Row():
                            model_info_output = gr.Markdown(
                                label="Model Information",
                                elem_classes=["model-info"]
                            )
                            
                            processing_stats = gr.Markdown(
                                label="Processing Statistics", 
                                elem_classes=["stats-info"]
                            )
            
            with gr.Tab("📁 Batch Processing"):
                with gr.Row():
                    with gr.Column():
                        # File upload for batch processing
                        batch_files = gr.File(
                            label="Upload Multiple Audio Files",
                            file_count="multiple",
                            file_types=["audio"]
                        )
                        
                        # Model and language for batch
                        batch_model = gr.Dropdown(
                            choices=[
                                "distil-whisper", "whisper-free", "whisper-small",
                                "wav2vec2-hindi", "wav2vec2-improved"
                            ],
                            value="distil-whisper",
                            label="Select Model for Batch Processing"
                        )
                        
                        batch_language = gr.Dropdown(
                            choices=list(self.supported_languages.keys()),
                            value="Hindi",
                            label="Select Language for All Files"
                        )
                        
                        batch_btn = gr.Button("🚀 Process Batch", variant="primary")
                    
                    with gr.Column():
                        batch_results = gr.Markdown(
                            label="Batch Results",
                            value="Upload files and click 'Process Batch' to see results here."
                        )
            
            with gr.Tab("📊 Model Comparison"):
                gr.Markdown("## Model Performance Comparison")
                
                comparison_btn = gr.Button("📊 Generate Comparison Table")
                comparison_output = gr.Markdown()
                
                gr.Markdown("""
                ### Model Recommendations:
                
                - **Distil-Whisper Large-v3**: Best overall choice - 6x faster, 49% smaller, <1% WER difference
                - **OpenAI Whisper Large-v3**: Best accuracy for complex audio
                - **Wav2Vec2 Hindi Models**: Specialized for Hindi language
                - **Whisper Small**: Good balance for CPU-only deployment
                - **SeamlessM4T**: Best for multilingual scenarios (101 languages)
                """)
            
            with gr.Tab("ℹ️ About & Setup"):
                gr.Markdown("""
                ## About This Application
                
                This application showcases free, open-source speech-to-text models specifically optimized for Indian languages.
                All models are available under permissive licenses (MIT, Apache 2.0) and can be used commercially.
                
                ### Supported Languages:
                - Hindi, Tamil, Bengali, Telugu, Marathi
                - Gujarati, Kannada, Malayalam, Punjabi, Odia
                - Assamese, Urdu, English
                
                ### Key Features:
                - ✅ Multiple free model architectures
                - ✅ Real-time and batch processing
                - ✅ Audio preprocessing and optimization
                - ✅ Performance metrics and comparison
                - ✅ Commercial use allowed
                
                ### Technical Stack:
                - **Models**: Transformers, PyTorch, TensorFlow
                - **Interface**: Gradio
                - **Audio Processing**: Librosa, SoundFile
                - **Optimization**: CUDA support, Mixed precision
                
                ### Setup Instructions:
                1. Install dependencies: `pip install -r requirements.txt`
                2. Set environment: `export APP_ENV=local`
                3. Run application: `python app.py`
                """)
            
            # Event handlers
            transcribe_btn.click(
                fn=self.transcribe_audio,
                inputs=[audio_input, model_choice, language_choice, enable_preprocessing],
                outputs=[transcription_output, model_info_output, processing_stats]
            )
            
            batch_btn.click(
                fn=self.batch_transcribe,
                inputs=[batch_files, batch_model, batch_language],
                outputs=[batch_results]
            )
            
            comparison_btn.click(
                fn=self.get_model_comparison,
                outputs=[comparison_output]
            )
        
        return interface
    
    def launch(self, share: bool = None, server_name: str = None, server_port: int = None):
        """Launch the Gradio application."""
        interface = self.create_interface()
        
        # Use config values or defaults
        share = share if share is not None else self.config.get("GRADIO_SHARE", False)
        server_name = server_name or self.config.get("GRADIO_SERVER_NAME", "127.0.0.1")
        server_port = server_port or int(self.config.get("GRADIO_SERVER_PORT", 7860))
        
        print(f"🚀 Launching Speech-to-Text Application...")
        print(f"📍 Server: http://{server_name}:{server_port}")
        print(f"🌐 Share: {share}")
        
        interface.launch(
            share=share,
            server_name=server_name,
            server_port=server_port,
            show_error=True,
            quiet=False
        )

def main():
    """Main function to run the application."""
    app = GradioSpeechToTextApp()
    app.launch()

if __name__ == "__main__":
    main()
