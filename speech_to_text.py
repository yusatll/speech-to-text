import whisper
import os
import time
from datetime import datetime
import torch

def check_gpu():
    """
    Check GPU availability and return device information
    
    Returns:
        bool: True if GPU is available, False otherwise
    """
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"GPU detected: {gpu_name}")
        return True
    else:
        print("No GPU found, using CPU")
        return False

def convert_audio_to_text(audio_path, model_size="small", language="en"):
    """
    Convert audio file to text using Whisper model with GPU acceleration if available
    
    Parameters:
        audio_path (str): Path to the audio file
        model_size (str): Size of the Whisper model ('tiny', 'base', 'small', 'medium', 'large')
        language (str): Language code (e.g., 'en' for English, 'tr' for Turkish)
    
    Returns:
        str: Transcribed text if successful, None otherwise
    """
    print(f"Starting process: {datetime.now()}")
    
    # Check GPU availability
    use_gpu = check_gpu()
    
    try:
        # Load the model
        print(f"Loading model: {model_size}")
        model = whisper.load_model(model_size)
        
        # Move model to GPU if available
        if use_gpu:
            model = model.cuda()
            print("Model moved to GPU")
        
        print("\nProcessing audio file...")
        start_time = time.time()
        
        # Transcribe audio
        result = model.transcribe(
            audio_path,
            language=language,
            fp16=use_gpu  # Use float16 on GPU for better performance
        )
        
        # Calculate processing time
        process_time = time.time() - start_time
        print(f"Processing time: {process_time:.2f} seconds")
        
        # Save result to file
        base_name, _ = os.path.splitext(audio_path)
        output_file = f"{base_name}_{model_size}_{process_time}.txt"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(result["text"])
        
        print(f"\nProcess completed: {datetime.now()}")
        print(f"Text saved to {output_file}")
        
        return result["text"]
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return None
    finally:
        # Clean up GPU memory
        if use_gpu:
            torch.cuda.empty_cache()
            print("GPU memory cleared")

if __name__ == "__main__":
    # Clear CUDA memory at start
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Display system information
    print("\nSystem Information:")
    print(f"PyTorch version: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Available GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # Audio file path
    file_name = "Source/Liderlik-3. Ders.m4a"  # Change this to your audio file name
    audio_file = os.path.join(os.getcwd(), file_name)
    
    # Select model size based on GPU availability
    # Use larger model if GPU is available for better accuracy
    MODEL_SIZE = "medium" if torch.cuda.is_available() else "turbo"
    
    # Select language (default: English)
    LANGUAGE = "tr"  # Change to "tr" for Turkish, "fr" for French, etc.
    
    print(f"\nProcessing file: {audio_file}")
    result = convert_audio_to_text(audio_file, MODEL_SIZE, LANGUAGE)
    
    if result:
        print("\nTranscription successful!")
        print("\nFirst 500 characters:")
        print(result[:500] + "...")
