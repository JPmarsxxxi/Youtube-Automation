# Automated Video Production Pipeline

A comprehensive AI-powered video production system that automates the entire workflow from script writing to final video asset collection. Built for Google Colab with integration to Google Drive, this pipeline uses multiple AI services (Groq, Google Gemini, ElevenLabs) to create professional video content with minimal manual intervention.

![Pipeline Status](https://img.shields.io/badge/status-active-brightgreen)
![Python](https://img.shields.io/badge/python-3.8+-blue)
![License](https://img.shields.io/badge/license-MIT-green)

## 🎯 Overview

This production pipeline consists of five interconnected modules that handle the complete video production workflow:

1. **Script Writing** - AI-powered script generation using Groq's LLaMA models
2. **Pre-Production** - Audio tag enhancement, TTS generation, transcription, and pose mapping
3. **Real Image Collection** - Automated image search and AI-curated selection
4. **PNG Sequence Collection** - Video downloading and frame extraction with AI validation
5. **Video Section Collection** - YouTube video downloading and trimming

## 📋 Table of Contents

- [Features](#-features)
- [Architecture](#-architecture)
- [Prerequisites](#-prerequisites)
- [Installation](#-installation)
- [Usage](#-usage)
- [Module Documentation](#-module-documentation)
- [Configuration](#-configuration)
- [Directory Structure](#-directory-structure)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)
- [License](#-license)

## ✨ Features

### Script Writing Module
- **AI-Powered Generation**: Uses Groq's LLaMA 3.3 70B model for high-quality script writing
- **MCP Integration**: Loads custom Model Context Protocols for style consistency
- **Iterative Refinement**: Multi-stage script enhancement and polishing
- **RSS Feed Integration**: Fetches fresh content ideas from configured feeds

### Pre-Production Module
- **Audio Tag Enhancement**: Automatically adds pose and audio effect tags to scripts
- **Text-to-Speech**: Integrates with ElevenLabs for professional voiceover generation
- **Transcription**: Precise word-level timing using Gemini's multimodal capabilities
- **Pose Mapping**: Maps script segments to character poses with timing
- **Lip Sync XML**: Generates Adobe Character Animator-compatible XML files

### Real Image Collection
- **Multi-Source Search**: Searches Unsplash, Pexels, and meme databases
- **AI Curation**: Uses Google Gemini to select contextually appropriate images
- **Smart Fallbacks**: Multiple fallback strategies ensure content is always found
- **Progress Tracking**: Resumable with comprehensive progress tracking
- **Metadata Management**: Detailed logging of all collected assets

### PNG Sequence Collection
- **Video Download**: Supports YouTube, TikTok, Instagram Reels
- **Frame Extraction**: Intelligent frame selection with scene detection
- **YOLO Integration**: Human presence validation using YOLOv8
- **Quality Filtering**: AI-powered quality assessment of frames
- **Duplicate Prevention**: Tracks used content to avoid repetition

### Video Section Collection
- **YouTube Integration**: Cookie-based authenticated downloading
- **Precise Trimming**: FFmpeg-based exact timestamp extraction
- **Keep-Alive System**: Prevents Colab disconnection during long operations
- **Metadata Tracking**: Comprehensive video metadata preservation

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    WRITING SECTION                          │
│  ┌────────────────────────────────────────────────────┐    │
│  │ RSS Feed → Groq LLaMA → Script Generation          │    │
│  │ MCP Loading → Style Refinement → Draft Output      │    │
│  └────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                 PRE-PRODUCTION SECTION                      │
│  ┌────────────────────────────────────────────────────┐    │
│  │ Phase 1: Audio Tag Enhancement                     │    │
│  │ Phase 2: ElevenLabs TTS Generation                 │    │
│  │ Phase 3: Gemini Transcription & Pose Mapping       │    │
│  │ Phase 4: Video Plan Generation                     │    │
│  └────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                  ASSET COLLECTION SECTION                   │
│  ┌────────────────────────────────────────────────────┐    │
│  │ Real Images: Unsplash/Pexels + Gemini Selection    │    │
│  │ PNG Sequences: YT-DLP + YOLO + Frame Extraction    │    │
│  │ Video Sections: Cookie Auth + FFmpeg Trimming      │    │
│  └────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

## 🔧 Prerequisites

### Required Accounts & API Keys
- [Groq API Key](https://console.groq.com/) - Free tier available
- [Google AI API Key](https://makersuite.google.com/app/apikey) - For Gemini
- [ElevenLabs API Key](https://elevenlabs.io/) - For TTS (paid)
- [Unsplash API Key](https://unsplash.com/developers) - For image search
- [Pexels API Key](https://www.pexels.com/api/) - For image search
- [Giphy API Key](https://developers.giphy.com/) - Optional for GIFs

### Environment
- Google Colab (recommended) or local Python 3.8+
- Google Drive for storage
- ~5GB free storage space
- Stable internet connection

### Browser Extensions (for YouTube cookie extraction)
- [Get cookies.txt LOCALLY](https://chrome.google.com/webstore/detail/get-cookiestxt-locally/cclelndahbckbenkjhflpdbgdldlbecc) - Chrome extension

## 📥 Installation

### 1. Clone Repository
```bash
git clone https://github.com/yourusername/video-production-pipeline.git
cd video-production-pipeline
```

### 2. Set Up Google Drive Structure

Create the following directory structure in your Google Drive:

```
MyDrive/Angry Dude Scripts/
├── mcps/                      # Model Context Protocols (style guides)
├── drafts/                    # Script generation output
├── raw_scripts/               # Scripts ready for processing
├── processed_scripts/         # Audio-tagged scripts
├── voiceovers/                # Generated audio files
├── pose_data/                 # Transcription data
├── final_videos/              # Lip-sync XML files
├── video_plans/               # Video production plans
├── pose_assets/
│   ├── poses/                 # Character pose PNG files
│   └── pose_descriptions.txt  # Pose tag definitions
├── temp/                      # Temporary files
└── backups/                   # Backup files
```

### 3. Configure API Keys

Create a `.env` file in `MyDrive/Angry Dude Scripts/`:

```env
# ElevenLabs TTS
ELEVENLABS_API_KEY=your_elevenlabs_key_here

# Google AI (Gemini)
GOOGLE_AI_API_KEY=your_google_ai_key_here

# Groq (LLaMA)
GROQ_API_KEY=your_groq_key_here

# Image Search APIs
UNSPLASH_ACCESS_KEY=your_unsplash_key_here
PEXELS_API_KEY=your_pexels_key_here
GIPHY_API_KEY=your_giphy_key_here
```

Or add keys directly to each script (see Configuration section).

## 🚀 Usage

### Quick Start - Complete Workflow

#### Step 1: Write Script
```python
# Upload writing_section.py to Google Colab
# Run all cells
# Script will be saved to /drafts/
```

#### Step 2: Pre-Production
```python
# Upload preproduction_after_writing.py to Google Colab
# Move script from /drafts/ to /raw_scripts/
# Select phases to run (or run all)
# Outputs: TTS audio, transcripts, lip-sync XML, video plan
```

#### Step 3: Collect Assets
```python
# Upload asset collection scripts:
# - real_image_collector.py
# - png_sequence_collector.py
# - video_section_collector.py

# Run each based on your video plan needs
```

### Detailed Usage by Module

#### Writing Section
```python
# 1. Configure RSS feeds in the script
# 2. Set up MCPs (Model Context Protocols)
# 3. Run script generation
# 4. Review and refine output in /drafts/
```

#### Pre-Production
```python
# Phase 1: Audio Tag Enhancement
- Adds [POSE:angry], [SOUND:explosion] tags
- Output: processed_scripts/

# Phase 2: TTS Generation
- Generates voiceover with ElevenLabs
- Output: voiceovers/

# Phase 3: Transcription & Mapping
- Word-level timing
- Pose mapping
- Lip-sync XML generation
- Output: final_videos/

# Phase 4: Video Planning
- Creates shot-by-shot breakdown
- Output: video_plans/
```

#### Asset Collection

**Real Images:**
```python
# Automatically finds and downloads images matching video plan
python real_image_collector.py
# Select video plan file
# Script runs automated search and AI curation
# Output: collected_assets/{script_name}/real_images/
```

**PNG Sequences:**
```python
# Downloads videos and extracts frames
python png_sequence_collector.py
# Select video plan file
# Script searches, downloads, validates frames
# Output: collected_assets/{script_name}/png_sequences/
```

**Video Sections:**
```python
# Downloads and trims YouTube videos
python video_section_collector.py
# Upload YouTube cookies.txt
# Select video plan file
# Script downloads and trims videos precisely
# Output: collected_assets/{script_name}/videos/
```

## 📚 Module Documentation

### writing_section.py
**Purpose**: AI-powered script generation using Groq's LLaMA models

**Key Classes:**
- Script generator with MCP integration
- RSS feed parser for content ideas
- Multi-stage refinement system

**Outputs:**
- Draft scripts in markdown format
- Saved to `drafts/` directory

### preproduction_after_writing.py
**Purpose**: Complete pre-production pipeline from script to video plan

**Key Classes:**
- `AudioTagProcessor`: Adds pose and sound effect tags
- `TTSGenerator`: ElevenLabs voice synthesis
- `GeminiTranscriber`: Word-level transcription
- `PoseMapper`: Maps scripts to character animations
- `VideoPlanner`: Creates shot-by-shot breakdown

**Outputs:**
- Audio-tagged scripts
- MP3 voiceovers
- Word-level transcripts
- Lip-sync XML files
- Video plan documents

### real_image_collector.py
**Purpose**: Automated image search and AI-curated selection

**Key Classes:**
- `ImageSearchAPI`: Multi-source search (Unsplash, Pexels)
- `GeminiImageSelector`: AI-powered relevance scoring
- `ProgressTracker`: Resume-capable progress tracking

**Features:**
- Smart fallback strategies (memes, emergency modes)
- Duplicate prevention
- Quality filtering
- Metadata logging

**Outputs:**
- High-resolution images
- Metadata JSON files
- Progress tracking

### png_sequence_collector.py
**Purpose**: Video downloading and intelligent frame extraction

**Key Classes:**
- `VideoDownloader`: Multi-platform support (YouTube, TikTok, Instagram)
- `FrameExtractor`: Scene detection and frame selection
- `YOLOValidator`: Human presence detection
- `GeminiQualityChecker`: Frame quality assessment

**Features:**
- Adaptive frame selection
- Quality filtering
- Duplicate prevention
- Comprehensive error handling

**Outputs:**
- PNG frame sequences
- Frame metadata
- Quality scores

### video_section_collector.py
**Purpose**: YouTube video downloading with precise trimming

**Key Classes:**
- `YouTubeDownloader`: Cookie-authenticated downloading
- `VideoTrimmer`: FFmpeg-based precise cutting
- `ProgressTracker`: Resumable operations

**Features:**
- Keep-alive mechanism for long operations
- Cookie-based authentication bypass
- Metadata preservation
- Error recovery

**Outputs:**
- Trimmed video clips
- Video metadata
- Progress tracking

## ⚙️ Configuration

### API Key Configuration

**Method 1: Environment File** (Recommended)
```bash
# In Google Drive: Angry Dude Scripts/.env
ELEVENLABS_API_KEY=sk_...
GOOGLE_AI_API_KEY=AIza...
GROQ_API_KEY=gsk_...
```

**Method 2: Direct in Script**
```python
# In each script, find and update:
API_KEY = "your_api_key_here"
```

### Model Configuration

**Groq Models:**
```python
model = "llama-3.3-70b-versatile"  # Default, best quality
# Alternative: "llama-3.1-8b-instant" for faster generation
```

**Gemini Models:**
```python
model = "gemini-2.0-flash-exp"  # Default
# For production: "gemini-1.5-pro"
```

**ElevenLabs Voices:**
```python
voice_id = "onwK4e9ZLuTAKqWW03F9"  # Daniel (default)
# Find more: https://elevenlabs.io/voice-library
```

### Search Configuration

**Real Image Search:**
```python
# Adjust search parameters
max_results = 10
min_resolution = (1920, 1080)
confidence_threshold = 70  # AI selection confidence (0-100)
```

**PNG Sequence Search:**
```python
# Frame extraction settings
target_frame_count = 30  # Frames to extract
quality_threshold = 70   # Minimum quality score
```

## 📁 Directory Structure

```
video-production-pipeline/
├── writing_section.py                # Script generation module
├── preproduction_after_writing.py    # Pre-production pipeline
├── real_image_collector.py           # Image collection module
├── png_sequence_collector.py         # Frame extraction module
├── video_section_collector.py        # Video trimming module
├── README.md                          # This file
├── PORTFOLIO.md                       # Portfolio documentation
├── LICENSE                            # MIT License
└── requirements.txt                   # Python dependencies

Google Drive Structure:
MyDrive/Angry Dude Scripts/
├── .env                               # API keys
├── mcps/                              # Style guides
├── drafts/                            # Generated scripts
├── raw_scripts/                       # Input scripts
├── processed_scripts/                 # Tagged scripts
├── voiceovers/                        # Audio files
├── pose_data/                         # Transcription data
├── final_videos/                      # Lip-sync XML
├── video_plans/                       # Production plans
├── pose_assets/                       # Character assets
├── collected_assets/                  # Downloaded media
│   └── {script_name}/
│       ├── real_images/
│       ├── png_sequences/
│       └── videos/
├── temp/                              # Temporary files
└── backups/                           # Backup files
```

## 🔍 Troubleshooting

### Common Issues

**Issue: "Module not found" errors**
```bash
# Solution: Run the installation cells in each script
!pip install -U google-generativeai pillow requests playwright
```

**Issue: YouTube downloads failing**
```bash
# Solution: Update yt-dlp and use cookies
!pip install -U yt-dlp
# Upload fresh cookies.txt from YouTube
```

**Issue: API rate limits**
```python
# Solution: Add delays between requests
time.sleep(2)  # 2 second delay
```

**Issue: Colab disconnecting**
```python
# Solution: Keep-alive mechanism (included in video_section_collector)
# Runs automatically when script starts
```

**Issue: Out of memory**
```bash
# Solution: Clear runtime memory
from IPython.display import clear_output
clear_output()
```

**Issue: Gemini quota exceeded**
```python
# Solution: Switch to alternative model or wait
# Free tier: 15 requests/min, 1500 requests/day
```

### Error Recovery

All collection scripts include progress tracking:
```python
# Progress is saved automatically
# On restart, script will skip completed sections
# Check progress file: collected_assets/{script_name}/progress_tracker.json
```

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Development Guidelines
- Follow PEP 8 style guidelines
- Add docstrings to all functions and classes
- Include error handling for external API calls
- Update documentation for new features
- Test on Google Colab before submitting

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Groq** - Fast LLaMA inference
- **Google Gemini** - Multimodal AI capabilities
- **ElevenLabs** - Professional TTS generation
- **Ultralytics** - YOLO object detection
- **FFmpeg** - Video processing
- **Playwright** - Web automation
- **yt-dlp** - Universal video downloader

## 📧 Contact

For questions, issues, or suggestions:
- Open an issue on GitHub
- Email: your.email@example.com
- Twitter: [@yourhandle](https://twitter.com/yourhandle)

## 🗺️ Roadmap

- [ ] Add support for Claude API as alternative to Gemini
- [ ] Implement automatic background music selection
- [ ] Add video effect automation
- [ ] Create web UI for easier configuration
- [ ] Support for multiple languages
- [ ] Batch processing multiple scripts
- [ ] Integration with video editing software APIs
- [ ] Real-time preview generation

---

**Made with ❤️ for automated video production**
