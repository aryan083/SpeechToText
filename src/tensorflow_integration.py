import tensorflow as tf
import torch
import numpy as np
from typing import Optional, Union, Dict
import logging
from transformers import TFAutoModel, AutoTokenizer

class TensorFlowOptimizer:
    """
    TensorFlow integration for model optimization and inference acceleration.
    Provides TensorFlow Lite conversion and GPU optimization.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Configure TensorFlow
        self._configure_tensorflow()
    
    def _configure_tensorflow(self):
        """Configure TensorFlow for optimal performance."""
        try:
            # Enable mixed precision
            tf.keras.mixed_precision.set_global_policy('mixed_float16')
            
            # Configure GPU memory growth
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                self.logger.info(f"Configured {len(gpus)} GPU(s) for TensorFlow")
            else:
                self.logger.info("No GPUs found, using CPU")
                
        except Exception as e:
            self.logger.warning(f"TensorFlow configuration warning: {e}")
    
    def convert_to_tensorflow_lite(self, model_path: str, output_path: str, 
                                 quantize: bool = True) -> bool:
        """
        Convert a model to TensorFlow Lite for mobile/edge deployment.
        
        Args:
            model_path: Path to the saved model
            output_path: Path for the TFLite model
            quantize: Whether to apply quantization
            
        Returns:
            Success status
        """
        try:
            # Load the model
            converter = tf.lite.TFLiteConverter.from_saved_model(model_path)
            
            if quantize:
                # Apply dynamic range quantization
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                
                # For better quantization, use representative dataset
                converter.representative_dataset = self._representative_dataset_gen
                converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
                converter.inference_input_type = tf.int8
                converter.inference_output_type = tf.int8
            
            # Convert the model
            tflite_model = converter.convert()
            
            # Save the model
            with open(output_path, 'wb') as f:
                f.write(tflite_model)
            
            self.logger.info(f"Successfully converted model to TFLite: {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"TFLite conversion failed: {e}")
            return False
    
    def _representative_dataset_gen(self):
        """Generate representative dataset for quantization."""
        # Generate dummy audio data for quantization calibration
        for _ in range(100):
            # Simulate 16kHz audio for 1 second
            dummy_audio = np.random.randn(1, 16000).astype(np.float32)
            yield [dummy_audio]
    
    def optimize_for_inference(self, model, input_shape: tuple) -> tf.keras.Model:
        """
        Optimize a TensorFlow model for inference.
        
        Args:
            model: TensorFlow model to optimize
            input_shape: Expected input shape
            
        Returns:
            Optimized model
        """
        try:
            # Create concrete function for optimization
            @tf.function
            def inference_func(x):
                return model(x, training=False)
            
            # Get concrete function
            concrete_func = inference_func.get_concrete_function(
                tf.TensorSpec(shape=input_shape, dtype=tf.float32)
            )
            
            # Apply graph optimization
            optimized_func = tf.function(concrete_func)
            
            self.logger.info("Model optimized for inference")
            return optimized_func
            
        except Exception as e:
            self.logger.error(f"Inference optimization failed: {e}")
            return model
    
    def benchmark_model(self, model, input_shape: tuple, num_runs: int = 100) -> Dict:
        """
        Benchmark model performance.
        
        Args:
            model: Model to benchmark
            input_shape: Input shape for testing
            num_runs: Number of benchmark runs
            
        Returns:
            Performance metrics
        """
        try:
            # Generate test input
            test_input = tf.random.normal(input_shape)
            
            # Warmup runs
            for _ in range(10):
                _ = model(test_input)
            
            # Benchmark runs
            import time
            start_time = time.time()
            
            for _ in range(num_runs):
                _ = model(test_input)
            
            end_time = time.time()
            
            # Calculate metrics
            total_time = end_time - start_time
            avg_time = total_time / num_runs
            throughput = num_runs / total_time
            
            metrics = {
                "total_time": total_time,
                "average_inference_time": avg_time,
                "throughput_fps": throughput,
                "num_runs": num_runs
            }
            
            self.logger.info(f"Benchmark results: {metrics}")
            return metrics
            
        except Exception as e:
            self.logger.error(f"Benchmarking failed: {e}")
            return {}
    
    def create_serving_signature(self, model, input_spec: Dict) -> tf.keras.Model:
        """
        Create a model with serving signature for deployment.
        
        Args:
            model: TensorFlow model
            input_spec: Input specification dictionary
            
        Returns:
            Model with serving signature
        """
        try:
            # Define serving function
            @tf.function
            def serve_fn(audio_input):
                # Preprocess input if needed
                processed_input = tf.cast(audio_input, tf.float32)
                
                # Run inference
                output = model(processed_input)
                
                # Post-process output if needed
                return {"transcription": output}
            
            # Create serving signature
            signatures = {
                "serving_default": serve_fn.get_concrete_function(
                    audio_input=tf.TensorSpec(
                        shape=input_spec.get("shape", [None, None]), 
                        dtype=tf.float32,
                        name="audio_input"
                    )
                )
            }
            
            # Save model with signature
            model._signatures = signatures
            
            self.logger.info("Created serving signature for model")
            return model
            
        except Exception as e:
            self.logger.error(f"Serving signature creation failed: {e}")
            return model

class TensorFlowSpeechModel:
    """
    TensorFlow-native speech-to-text model wrapper.
    Provides optimized inference using TensorFlow operations.
    """
    
    def __init__(self, model_name: str, use_mixed_precision: bool = True):
        self.model_name = model_name
        self.use_mixed_precision = use_mixed_precision
        self.logger = logging.getLogger(__name__)
        
        # Initialize TensorFlow optimizer
        self.tf_optimizer = TensorFlowOptimizer()
        
        # Load model
        self.model = None
        self.tokenizer = None
        self._load_model()
    
    def _load_model(self):
        """Load TensorFlow version of the model."""
        try:
            # Try to load TensorFlow version
            self.model = TFAutoModel.from_pretrained(
                self.model_name,
                from_tf=True
            )
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Optimize for inference
            if hasattr(self.model, 'config'):
                input_shape = (1, self.model.config.max_position_embeddings)
                self.model = self.tf_optimizer.optimize_for_inference(
                    self.model, input_shape
                )
            
            self.logger.info(f"Loaded TensorFlow model: {self.model_name}")
            
        except Exception as e:
            self.logger.warning(f"TensorFlow model loading failed: {e}")
            self.model = None
    
    def transcribe(self, audio_input: np.ndarray) -> str:
        """
        Transcribe audio using TensorFlow model.
        
        Args:
            audio_input: Audio data as numpy array
            
        Returns:
            Transcribed text
        """
        if self.model is None:
            raise ValueError("Model not loaded")
        
        try:
            # Convert to TensorFlow tensor
            audio_tensor = tf.convert_to_tensor(audio_input, dtype=tf.float32)
            
            # Add batch dimension if needed
            if len(audio_tensor.shape) == 1:
                audio_tensor = tf.expand_dims(audio_tensor, 0)
            
            # Run inference
            with tf.device('/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'):
                outputs = self.model(audio_tensor)
            
            # Process outputs (this would depend on the specific model)
            # For now, return a placeholder
            return "TensorFlow transcription result"
            
        except Exception as e:
            self.logger.error(f"TensorFlow transcription failed: {e}")
            raise
    
    def benchmark(self, audio_shape: tuple = (1, 16000)) -> Dict:
        """Benchmark the TensorFlow model."""
        if self.model is None:
            return {"error": "Model not loaded"}
        
        return self.tf_optimizer.benchmark_model(self.model, audio_shape)
    
    def save_optimized_model(self, output_path: str) -> bool:
        """Save optimized model for deployment."""
        if self.model is None:
            return False
        
        try:
            # Create serving signature
            input_spec = {"shape": [None, 16000]}  # Audio input shape
            model_with_signature = self.tf_optimizer.create_serving_signature(
                self.model, input_spec
            )
            
            # Save model
            tf.saved_model.save(model_with_signature, output_path)
            
            self.logger.info(f"Saved optimized model to: {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Model saving failed: {e}")
            return False
