# 🚀 Deploying to Hugging Face Spaces

This guide will help you deploy the Indian Speech-to-Text application to Hugging Face Spaces.

## 📋 Prerequisites

1. **Hugging Face Account**: Create an account at [huggingface.co](https://huggingface.co)
2. **Git**: Install Git on your system
3. **Git LFS**: Install Git Large File Storage for model files

## 🔧 Setup Instructions

### Step 1: Create a New Space

1. Go to [huggingface.co/new-space](https://huggingface.co/new-space)
2. Fill in the details:
   - **Space name**: `indian-speech-to-text`
   - **License**: MIT
   - **SDK**: Docker
   - **Hardware**: CPU Basic (or GPU if available)
   - **Visibility**: Public

### Step 2: Clone Your Space Repository

```bash
git clone https://huggingface.co/spaces/YOUR_USERNAME/indian-speech-to-text
cd indian-speech-to-text
```

### Step 3: Copy Files for Spaces Deployment

Copy these files from your project to the Spaces repository:

**Essential Files:**
```bash
# Copy the Spaces-optimized files
cp README.md indian-speech-to-text/
cp app_spaces.py indian-speech-to-text/app.py
cp Dockerfile.spaces indian-speech-to-text/Dockerfile
cp requirements_spaces.txt indian-speech-to-text/requirements.txt
cp .gitignore indian-speech-to-text/

# Copy source code
cp -r src/ indian-speech-to-text/
cp -r scripts/ indian-speech-to-text/
cp -r configs/ indian-speech-to-text/
```

### Step 4: Set Up Environment Variables (Optional)

In your Space settings, you can add these environment variables:

- `HF_TOKEN`: Your Hugging Face token (for private models)
- `HUGGINGFACE_HUB_TOKEN`: Alternative token name
- `DEFAULT_MODEL`: `distil-whisper` (default model to load)
- `DEFAULT_LANGUAGE`: `hindi` (default language)

### Step 5: Deploy to Spaces

```bash
cd indian-speech-to-text

# Add all files
git add .

# Commit changes
git commit -m "Initial deployment of Indian Speech-to-Text models"

# Push to Spaces
git push origin main
```

## 🎯 Spaces-Specific Optimizations

### File Structure for Spaces:
```
indian-speech-to-text/
├── README.md                 # With Spaces metadata
├── app.py                    # Spaces-optimized entry point
├── Dockerfile                # Spaces-compatible Docker config
├── requirements.txt          # Essential dependencies only
├── .gitignore               # Spaces-specific ignores
├── src/
│   ├── models/
│   │   └── speech_to_text.py
│   ├── ui/
│   │   └── gradio_app.py
│   └── utils/
│       ├── config.py
│       └── audio_utils.py
├── scripts/
│   └── download_models.py
└── configs/
    └── envs/
        └── .env.prod
```

### Key Optimizations:

1. **Memory Efficient**: Uses `/tmp` for model cache
2. **Fast Startup**: Downloads only essential models
3. **Gradio Queue**: Enables queuing for better performance
4. **Error Handling**: Graceful fallbacks if models fail to load
5. **Spaces Integration**: Proper metadata and configuration

## 🔍 Monitoring Your Space

### Check Deployment Status:
1. Go to your Space URL: `https://huggingface.co/spaces/YOUR_USERNAME/indian-speech-to-text`
2. Monitor the build logs in the "Logs" tab
3. Wait for the "Running" status

### Common Issues and Solutions:

**Build Timeout:**
- Reduce the number of models downloaded
- Use smaller models (whisper-small instead of whisper-large)

**Memory Issues:**
- Upgrade to GPU hardware
- Reduce model size in `app_spaces.py`

**Model Download Failures:**
- Add HF_TOKEN in Space settings
- Check internet connectivity in build logs

## 🎉 Success!

Once deployed, your Space will be available at:
`https://huggingface.co/spaces/YOUR_USERNAME/indian-speech-to-text`

### Features Available:
- ✅ Real-time speech-to-text conversion
- ✅ Multiple Indian language support
- ✅ Model comparison interface
- ✅ Batch processing capabilities
- ✅ Audio preprocessing options

## 🔄 Updating Your Space

To update your Space:

```bash
# Make changes to your files
# Then commit and push
git add .
git commit -m "Update: description of changes"
git push origin main
```

The Space will automatically rebuild and redeploy.

## 📞 Support

If you encounter issues:
1. Check the Space logs for error messages
2. Verify all files are correctly copied
3. Ensure the Dockerfile builds successfully locally
4. Contact Hugging Face support if needed

## 🏷️ Space Metadata

The README.md includes this metadata for Spaces:

```yaml
---
title: Indian Speech-to-Text Models
emoji: 🎤
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
license: mit
tags:
  - speech-to-text
  - indian-languages
  - hindi
  - whisper
  - wav2vec2
  - gradio
models:
  - distil-whisper/distil-large-v3
  - openai/whisper-large-v3
  - ai4bharat/indicwav2vec-hindi
---
```

This ensures proper categorization and discoverability on Hugging Face Spaces!
