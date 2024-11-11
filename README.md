# GPU-Accelerated Speech-to-Text Converter

This project provides a Python script that converts audio files to text using OpenAI's Whisper model with GPU acceleration. It's especially optimized for processing large audio files and supports multiple languages (default is Turkish, but can be easily modified).

## Features

- GPU acceleration support for faster processing
- Multiple model sizes (tiny, base, small, medium, large)
- Automatic GPU/CPU detection
- Progress tracking and timing information
- Memory management for GPU
- Support for various audio formats (m4a, mp3, wav, etc.)

## Requirements

### Hardware
- NVIDIA GPU (optional, but recommended for faster processing)
- Minimum 4GB RAM (8GB recommended)
- For large models: 8GB+ GPU VRAM recommended

### Software
- Python 3.8+
- CUDA Toolkit (for GPU support)
- FFmpeg

# Installation

## 1. Install CUDA Toolkit (if using GPU):
   - Download from [NVIDIA CUDA](https://developer.nvidia.com/cuda-downloads)
   - Follow installation instructions for your system

## 2. Install FFmpeg:

### Windows
1. Download FFmpeg:
   * Go to https://github.com/BtbN/FFmpeg-Builds/releases
   * Download the latest version (ffmpeg-master-latest-win64-gpl.zip)

2. Extract the ZIP file:
   * Extract the downloaded ZIP file
   * Move the extracted folder to an easy-to-find location (e.g., `C:\ffmpeg`)

3. Add to System PATH:
   * Type "Environment Variables" in Windows search
   * Click "Edit the system environment variables"
   * Click "Environment Variables" button
   * Under "System variables", find and click "Path"
   * Click "New"
   * Add the path to FFmpeg's bin folder (e.g., C:\ffmpeg\bin)
   * Click "OK" to close all windows

4. Test the installation:
   * Open a new Command Prompt (CMD) (close existing ones first)
   * Type the following command:
     ```bash
     ffmpeg -version
     ```
   * If you see version information, the installation was successful

### macOS
```bash
brew install ffmpeg
```

### Linux (Ubuntu/Debian)
```bash
sudo apt-get update
sudo apt-get install ffmpeg
```

### Troubleshooting FFmpeg Installation

If you get an error about FFmpeg not being found:
1. Make sure you've added the correct path to system PATH
2. Verify that the bin folder contains ffmpeg.exe
3. Try restarting your computer
4. Run Command Prompt as administrator and try the test command again

You can verify the PATH setting by:
```bash
echo %PATH%
```
This should show the FFmpeg bin directory in the output.

## 3. Install required Python packages:
   ```bash
   # For GPU support with CUDA 11.8
   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

   # For GPU support with CUDA 12.1
   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

   # Install other requirements
   pip install openai-whisper
   ```

# Usage

1. Place your audio file in the same directory as the script
2. Update the audio filename in the script
3. Run the script:
   ```bash
   python speech_to_text.py
   ```

## Model Sizes

Choose the appropriate model size based on your needs:

| Model  | Size   | Memory Required | Speed     | Accuracy |
|--------|--------|-----------------|-----------|----------|
| tiny   | 39M    | ~1GB VRAM      | Fastest   | Basic    |
| base   | 74M    | ~1GB VRAM      | Fast      | Good     |
| small  | 244M   | ~2GB VRAM      | Moderate  | Better   |
| medium | 769M   | ~5GB VRAM      | Slow      | Best     |
| large  | 1.5GB  | ~10GB VRAM     | Slowest   | Premium  |
| turbo	| 809 M	| ~6 GB VRAM     |turbo	     | Best     |

## Performance Benchmarks

Here are real-world performance tests comparing RTX 4060 and Google Colab (T4 GPU):

| Audio Duration | Model  | Environment | Processing Time | Processing Ratio* |
|---------------|--------|-------------|-----------------|-------------------|
| 2h 36min      | turbo  | RTX 4060    | 705 sec (~12min)| 0.075 |
| 2h            | turbo  | RTX 4060    | 500 sec (~8min) | 0.069 |
| 2h            | medium | RTX 4060    | 1750 sec (~29min)| 0.243 |
| 1h 30min      | large  | RTX 4060    | 27428 sec (~7.6h)| 5.07 |
| 15min         | large  | RTX 4060    | 3928 sec (~65min)| 4.35 |
| 15min         | medium | RTX 4060    | 370 sec (~6min)  | 0.4  |
| 2h 36min      | large  | Colab GPU   | 2400 sec (~40min)| 0.26 |

*Processing Ratio: Processing time / Audio duration. Lower is better.

Key Observations:
1. RTX 4060:
   - Best with turbo model: ~0.07 processing ratio
   - Large model significantly slower: ~4-5 processing ratio
   - Medium model offers good balance: ~0.2-0.4 processing ratio

2. Google Colab (T4 GPU):
   - Significantly faster for large model
   - ~20x faster than RTX 4060 for large model processing
   - Recommended for large model usage

### Google Colab (T4 GPU) Performance
When using Google Colab's GPU:
- 2h 36min audio with large model: ~40 minutes
- Processing Ratio: ~0.26

This shows that Colab processing is significantly faster (approximately 20x) than local GPU processing for the large model.

### Model Comparison
Based on these benchmarks:
1. **Turbo Model**:
   - Fastest processing (ratio: ~0.07)
   - Best for long recordings
   - Good for quick drafts

2. **Medium Model**:
   - Balanced performance (ratio: ~0.2-0.4)
   - Recommended for most use cases
   - Good accuracy/speed trade-off

3. **Large Model**:
   - Slowest processing (ratio: ~4-5)
   - Best accuracy
   - Recommended for:
     - Short recordings on local GPU
     - Any length on Google Colab
     - Critical accuracy requirements

## Google Colab Integration

For faster processing, especially with the large model, you can use our Google Colab script. The script provides:

1. Direct Google Drive integration
2. Automatic GPU utilization
3. Progress tracking
4. Error handling
5. Automatic output saving to Drive

### Using Colab Script
1. Copy `speech_to_text_whisper.ipynb` to your Colab notebook
2. Mount Google Drive
3. Specify input/output paths
4. Run with GPU acceleration

Example workflow:
```python
AUDIO_FILE = "/content/drive/MyDrive/audio.m4a"
OUTPUT_FOLDER = "/content/drive/MyDrive/transcriptions"
```

Full Colab implementation can be found in `speech_to_text_whisper.ipynb`.

### Colab vs Local Processing
Choose based on your needs:

| Aspect        | Local Processing | Google Colab |
|---------------|------------------|--------------|
| Speed (Large) | ~4-5x audio length | ~0.26x audio length |
| Setup         | More complex     | Simple |
| Dependencies  | Local install    | Auto-managed |
| File Size     | Unlimited       | <500MB |
| Availability  | Always          | Internet required |
| Cost          | Free            | Free (with limits) |


## Troubleshooting

1. **CUDA/GPU Issues**:
   ```python
   import torch
   print(torch.cuda.is_available())  # Should return True if GPU is available
   print(torch.cuda.get_device_properties(0))  # Shows GPU properties
   ```

2. **Memory Issues**:
   - Reduce model size
   - Clear GPU memory: `torch.cuda.empty_cache()`
   - Monitor GPU memory usage during processing

## Contributing

Feel free to open issues or submit pull requests for improvements.

## License

MIT License

## Acknowledgments

- OpenAI Whisper team for the amazing model
- PyTorch team for GPU support
