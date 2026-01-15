
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#
#      PREPRODUCTION: AFTER WRITING
#
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++







# ═══════════════════════════════════════════════════════════════════════════════
# AUTOMATED VIDEO PRODUCTION PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════
# Multi-Phase Workflow: Raw Scripts → Audio Tags → TTS → Transcription →
#                       Pose Mapping → Lip Sync XML → Video Planning
# ═══════════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════════
# 📦 SECTION 1: CORE DEPENDENCIES ONLY
# ══════════════════════════════════════════════════════════════════════════════

print("🔧 Installing core packages...")
print("This may take 1-2 minutes on first run...\n")

# Core dependencies (always needed)
!pip install -q python-dotenv elevenlabs pydub watchdog
!pip install -q google-generativeai

print("✅ Core packages installed!\n")
print("ℹ️  Additional packages will be installed only when needed by each phase\n")

# ══════════════════════════════════════════════════════════════════════════════
# 📂 SECTION 2: GOOGLE DRIVE MOUNT & DIRECTORY SETUP
# ══════════════════════════════════════════════════════════════════════════════

from google.colab import drive, files
import os
from pathlib import Path
import json
from datetime import datetime
import shutil
import re

# Mount Google Drive
print("📁 Mounting Google Drive...")
drive.mount('/content/drive', force_remount=False)
print("✅ Google Drive mounted!\n")

# Define base directory
BASE_DIR = Path('/content/drive/MyDrive/Angry Dude Scripts')

# Required directory structure
REQUIRED_DIRS = {
    'raw_scripts': 'Place your raw script .txt files here',
    'processed_scripts': 'Audio Tags enhanced scripts (Phase 1 output)',
    'voiceovers': 'Generated voiceover MP3 files (Phase 2 output)',
    'pose_data': 'Transcripts and mapping data (Phase 3 intermediate)',
    'final_videos': 'Final lip-sync XML files (Phase 3 output)',
    'video_plans': 'Video planning documents (Phase 4 output)',
    'temp': 'Temporary processing files',
    'backups': 'Tracking file backups',
    'pose_assets/poses': 'Pose PNG files (user must provide)',
    'pose_assets/poses/mouth_movements': 'Mouth movement PNG files (user must provide)',
}

# Required files
REQUIRED_FILES = {
    '.env': 'ElevenLabs API credentials (ELEVENLABS_API_KEY=xxx)',
    'pose_assets/pose_descriptions.txt': 'Pose tag descriptions file',
}

print("🔍 Validating directory structure...\n")

# Create base directory if doesn't exist
if not BASE_DIR.exists():
    print(f"📁 Creating base directory: {BASE_DIR}")
    BASE_DIR.mkdir(parents=True, exist_ok=True)
else:
    print(f"✅ Base directory exists: {BASE_DIR}")

# Create/validate subdirectories
missing_dirs = []
for dir_path, description in REQUIRED_DIRS.items():
    full_path = BASE_DIR / dir_path
    if not full_path.exists():
        full_path.mkdir(parents=True, exist_ok=True)
        missing_dirs.append((dir_path, description))
        print(f"  📁 Created: {dir_path}")
    else:
        print(f"  ✅ Exists: {dir_path}")

# Check required files
missing_files = []
for file_path, description in REQUIRED_FILES.items():
    full_path = BASE_DIR / file_path
    if not full_path.exists():
        missing_files.append((file_path, description))
        print(f"  ❌ Missing: {file_path}")
    else:
        print(f"  ✅ Exists: {file_path}")

# Validate pose assets
pose_assets_dir = BASE_DIR / 'pose_assets' / 'poses'
mouth_dir = pose_assets_dir / 'mouth_movements'

pose_pngs = list(pose_assets_dir.glob('*.png'))
mouth_pngs = list(mouth_dir.glob('*.png'))

print(f"\n📊 Pose Assets Check:")
print(f"  Pose PNGs: {len(pose_pngs)} files")
print(f"  Mouth PNGs: {len(mouth_pngs)} files")

if len(pose_pngs) == 0:
    missing_files.append(('pose_assets/poses/*.png', 'Pose image files (e.g., 4.png, 5.png, Q.png)'))
if len(mouth_pngs) == 0:
    missing_files.append(('pose_assets/poses/mouth_movements/*.png', 'Mouth movement files (Aa.png, Ee.png, etc.)'))

# Display setup status
print("\n" + "="*70)
if missing_files:
    print("⚠️  SETUP INCOMPLETE - MISSING FILES:")
    print("="*70)
    for file_path, description in missing_files:
        print(f"\n📄 {file_path}")
        print(f"   Description: {description}")

    print("\n" + "="*70)
    print("📋 SETUP INSTRUCTIONS:")
    print("="*70)
    print(f"""
1. Navigate to: {BASE_DIR}

2. Create .env file with:
   ELEVENLABS_API_KEY=your_api_key_here
   ELEVENLABS_VOICE_ID=Joseph

3. Create pose_descriptions.txt in pose_assets/ with format:
   [4] Pose Name – Description here
   [5] Another Pose – Description here

4. Upload pose PNG files to pose_assets/poses/
   (e.g., 4.png, 5.png, 6.png, N.png, Q.png, T.png, E.png, W.png)

5. Upload mouth movement PNGs to pose_assets/poses/mouth_movements/
   (Aa.png, Ee.png, Uh.png, Oh.png, W-oo.png, M.png, L.png, F.png, S.png, D.png, R.png, Neutral.png)

6. After setup, re-run this cell to validate.
    """)

    print("⛔ Cannot proceed without required files. Please complete setup.\n")
    raise SystemExit("Setup incomplete")
else:
    print("✅ SETUP COMPLETE - ALL REQUIRED FILES PRESENT")
    print("="*70)
    print(f"📁 Base Directory: {BASE_DIR}")
    print(f"🎨 Pose PNGs: {len(pose_pngs)}")
    print(f"🗣️  Mouth PNGs: {len(mouth_pngs)}")
    print("="*70)

print("\n🚀 Ready to process scripts!\n")

# ══════════════════════════════════════════════════════════════════════════════
# 🤖 SECTION 3: GEMINI API CONFIGURATION WITH GOOGLE SEARCH
# ══════════════════════════════════════════════════════════════════════════════

import google.generativeai as genai
import time

# Configure Gemini API
GEMINI_API_KEY = "" #input api key here or setup with .env
# ══════════════════════════════════════════════════════════════════════════════
# 🔍 LANGSEARCH API CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

import requests

# LangSearch API Configuration
LANGSEARCH_API_KEY = "" #input api key here or setup with .env
LANGSEARCH_API_URL = "" #input api key here or setup with .env

def langsearch_query(query, count=5):
    """
    Execute a web search using LangSearch API with detailed debugging.
    """
    try:
        headers = {
            "Authorization": f"Bearer {LANGSEARCH_API_KEY}",
            "Content-Type": "application/json"
        }

        payload = {
            "query": query,
            "freshness": "noLimit",
            "summary": True,
            "count": count
        }

        print(f"            📡 Calling: {LANGSEARCH_API_URL}")
        print(f"            🔑 Auth: Bearer {LANGSEARCH_API_KEY[:15]}...")

        response = requests.post(
            LANGSEARCH_API_URL,
            headers=headers,
            json=payload,
            timeout=15
        )

        print(f"            📊 Status: {response.status_code}")
        print(f"            📄 Response: {response.text[:300]}")

        if response.status_code == 200:
            data = response.json()

            if data.get("code") == 200 and data.get("data"):
                webpages = data.get("data", {}).get("webPages", {}).get("value", [])

                if webpages:
                    results = []
                    for page in webpages[:count]:
                        results.append({
                            'title': page.get('name', ''),
                            'url': page.get('url', ''),
                            'snippet': page.get('snippet', ''),
                            'summary': page.get('summary', '')
                        })

                    return {
                        'found': True,
                        'results': results,
                        'query': query
                    }

            return {'found': False, 'query': query, 'error': f"API returned code: {data.get('code')}"}
        else:
            return {'found': False, 'query': query, 'error': f'HTTP {response.status_code}: {response.text[:200]}'}

    except Exception as e:
        return {'found': False, 'query': query, 'error': str(e)}

def search_match_info(team1, team2, month, year):
    """
    Search for match information using LangSearch.
    Returns: dict with 'date', 'score', 'found' keys
    """
    query = f"{team1} vs {team2} {month} {year} final score result"

    try:
        response = requests.get(
            LANGSEARCH_API_URL,
            params={"q": query, "num": 5},
            timeout=10
        )

        if response.status_code == 200:
            results = response.json()

            # Extract from top result
            if results and len(results) > 0:
                top_result = results[0]
                snippet = top_result.get('snippet', '').lower()

                # Simple extraction logic
                match_info = {
                    'found': True,
                    'snippet': snippet,
                    'query': query
                }

                return match_info

        return {'found': False, 'query': query}

    except Exception as e:
        print(f"      ⚠️  LangSearch error: {e}")
        return {'found': False, 'query': query}

print("✅ LangSearch configured (free web search API)\n")

genai.configure(api_key=GEMINI_API_KEY)

# Use Gemini 2.5 Flash (grounding is enabled per-request)
GEMINI_MODEL = genai.GenerativeModel("gemini-2.5-flash")

# Rate limiting settings (free tier: 15 RPM, 1M TPM, 1500 RPD)
REQUEST_DELAY = 4.5  # seconds between requests (safety margin for 15 RPM)
MAX_RETRIES = 15

print("✅ Gemini API configured")
print(f"   Model: gemini-2.5-flash")
print(f"   Google Search: Available via 'google_search_retrieval'")
print(f"   Max retries: {MAX_RETRIES}")
print(f"   Rate limit: 15 requests/minute\n")

def call_gemini_with_retry(prompt, max_retries=MAX_RETRIES, validation_func=None,
                           validation_name="output", enable_grounding=False):
    """
    Call Gemini API with retry logic for strict validations.

    Args:
        prompt: The prompt to send to Gemini
        max_retries: Maximum number of retry attempts
        validation_func: Optional function(response) -> (is_valid, errors)
        validation_name: Name of validation for logging
        enable_grounding: Enable Google Search grounding for this request

    Returns:
        Gemini response text if successful, None if all retries failed
    """
    original_prompt = prompt

    for attempt in range(1, max_retries + 1):
        try:
            if enable_grounding and attempt == 1:
                print(f"   🔍 Gemini API call with Google Search (attempt {attempt}/{max_retries})...")
            else:
                print(f"   🤖 Gemini API call (attempt {attempt}/{max_retries})...")

            # Call with or without grounding
            if enable_grounding:
                # Use google_search_retrieval for OLD SDK
                response = GEMINI_MODEL.generate_content(
                    prompt,
                    tools='google_search_retrieval'
                )
            else:
                # Regular call without grounding
                response = GEMINI_MODEL.generate_content(prompt)

            response_text = response.text.strip()

            # If no validation function, return immediately
            if validation_func is None:
                print(f"   ✅ Response received ({len(response_text)} chars)")
                time.sleep(REQUEST_DELAY)  # Rate limiting
                return response_text

            # Validate response
            is_valid, errors = validation_func(response_text)

            if is_valid:
                print(f"   ✅ {validation_name} validation passed!")
                time.sleep(REQUEST_DELAY)  # Rate limiting
                return response_text
            else:
                print(f"   ❌ {validation_name} validation failed (attempt {attempt}/{max_retries})")
                print(f"   Errors: {errors[:200]}..." if len(str(errors)) > 200 else f"   Errors: {errors}")

                # Create retry prompt with quoted errors
                prompt = f"""Your previous response had validation errors. Please fix them and try again.

ERRORS FOUND:
{errors}

ORIGINAL TASK:
{original_prompt}

Please provide a corrected response that addresses all the errors above."""

                # Continue to next attempt
                time.sleep(REQUEST_DELAY)

        except Exception as e:
            error_msg = str(e)
            print(f"   ⚠️  API error (attempt {attempt}/{max_retries}): {error_msg}")

            # If grounding not available, fall back and output prompt for Claude
            if enable_grounding and ('google_search_retrieval' in error_msg.lower() or 'not available' in error_msg.lower() or 'free tier' in error_msg.lower()):
                print(f"\n" + "="*70)
                print("⚠️  GOOGLE SEARCH GROUNDING NOT AVAILABLE IN FREE TIER")
                print("="*70)
                print("\n📋 FALLBACK: Copy this prompt and use Claude to research:")
                print("\n" + "-"*70)
                print(original_prompt)
                print("-"*70)
                print("\n💡 Instructions:")
                print("1. Copy the prompt above")
                print("2. Paste into Claude and get response")
                print("3. Save Claude's response to a text file")
                print("4. Upload the file when prompted below\n")

                # Use file upload instead of text paste
                from google.colab import files

                print("📁 Upload Claude's response as a .txt file:")
                uploaded = files.upload()

                if uploaded:
                    # Get the first uploaded file
                    filename = list(uploaded.keys())[0]
                    claude_response = uploaded[filename].decode('utf-8').strip()

                    print(f"   ✅ File uploaded: {filename}")
                    print(f"   ✅ Content length: {len(claude_response)} characters")

                    # CRITICAL: Validate Claude's response if validation function exists
                    if validation_func is not None:
                        print(f"   🔍 Validating Claude's response...")
                        is_valid, errors = validation_func(claude_response)

                        if is_valid:
                            print(f"   ✅ {validation_name} validation passed!")
                            time.sleep(REQUEST_DELAY)
                            return claude_response
                        else:
                            print(f"   ❌ {validation_name} validation failed!")
                            print(f"   Errors: {errors[:200]}..." if len(str(errors)) > 200 else f"   Errors: {errors}")
                            print(f"\n   ⚠️  Claude's response didn't pass validation.")
                            print(f"   💡 Please fix the errors and upload again, or type 'skip' to continue without grounding\n")

                            # Ask what to do
                            action = input("Type 'retry' to upload again, or 'skip' to continue: ").strip().lower()

                            if action == 'retry':
                                continue  # Go back to retry (file upload will happen again on next attempt)
                            else:
                                print(f"   ⚠️  Continuing without grounding")
                                enable_grounding = False
                                continue
                    else:
                        # No validation function, return as-is
                        time.sleep(REQUEST_DELAY)
                        return claude_response
                else:
                    print(f"   ⚠️  No file uploaded, continuing without grounding")
                    enable_grounding = False
                    continue

            time.sleep(REQUEST_DELAY * 2)

            if attempt == max_retries:
                print(f"   ❌ All {max_retries} attempts failed")
                return None

    print(f"   ❌ Validation failed after {max_retries} attempts")
    return None

print("✅ Gemini retry logic configured\n")

# ══════════════════════════════════════════════════════════════════════════════
# 📊 SECTION 4: TRACKING SYSTEM WITH BACKUP
# ══════════════════════════════════════════════════════════════════════════════

import hashlib
import shutil

class TrackingSystem:
    """Unified tracking system with automatic backups."""

    def __init__(self, tracking_name, base_dir):
        self.tracking_name = tracking_name
        self.base_dir = Path(base_dir)
        self.tracking_file = self.base_dir / f"{tracking_name}_tracking.json"
        self.backup_dir = self.base_dir / "backups"
        self.backup_dir.mkdir(exist_ok=True)

        # Backup settings
        self.max_backups = 5

        # Load or create tracking data
        self.data = self._load_tracking()

    def _load_tracking(self):
        """Load tracking data with corruption recovery."""
        if self.tracking_file.exists():
            try:
                with open(self.tracking_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                print(f"   📋 Loaded {self.tracking_name}: {len(data.get('processed_files', {}))} files tracked")
                return data
            except json.JSONDecodeError:
                print(f"   ⚠️  Corrupted tracking file, attempting recovery from backup...")
                return self._recover_from_backup()
        else:
            print(f"   📋 Creating new {self.tracking_name} tracking")
            return {"last_updated": "", "processed_files": {}}

    def _recover_from_backup(self):
        """Recover from most recent backup."""
        backups = sorted(self.backup_dir.glob(f"{self.tracking_name}_backup_*.json"))

        if backups:
            latest_backup = backups[-1]
            try:
                with open(latest_backup, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                print(f"   ✅ Recovered from backup: {latest_backup.name}")
                return data
            except:
                pass

        print(f"   ⚠️  No valid backups, starting fresh")
        return {"last_updated": "", "processed_files": {}}

    def save(self):
        """Save tracking data with automatic backup."""
        self.data["last_updated"] = datetime.now().isoformat()

        # Save main file
        with open(self.tracking_file, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)

        # Create backup
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_file = self.backup_dir / f"{self.tracking_name}_backup_{timestamp}.json"
        shutil.copy2(self.tracking_file, backup_file)

        # Cleanup old backups
        self._cleanup_old_backups()

    def _cleanup_old_backups(self):
        """Keep only last N backups."""
        backups = sorted(self.backup_dir.glob(f"{self.tracking_name}_backup_*.json"))

        if len(backups) > self.max_backups:
            for old_backup in backups[:-self.max_backups]:
                old_backup.unlink()

    def get_file_key(self, file_path):
        """Generate unique file key."""
        try:
            stat = file_path.stat()
            content = f"{file_path.name}_{stat.st_size}_{stat.st_mtime}"
            return hashlib.md5(content.encode()).hexdigest()
        except:
            return None

    def is_processed(self, file_path, stage=None):
        """Check if file/stage is processed."""
        file_key = self.get_file_key(file_path)
        if not file_key or file_key not in self.data["processed_files"]:
            return False

        file_data = self.data["processed_files"][file_key]

        if stage:
            return file_data.get(f"{stage}_complete", False)
        else:
            return file_data.get("completed", False)

    def mark_processed(self, file_path, stage=None, **extra_data):
        """Mark file/stage as processed."""
        file_key = self.get_file_key(file_path)
        if not file_key:
            return

        if file_key not in self.data["processed_files"]:
            self.data["processed_files"][file_key] = {
                "file_name": file_path.name,
                "started": datetime.now().isoformat()
            }

        if stage:
            self.data["processed_files"][file_key][f"{stage}_complete"] = True
            self.data["processed_files"][file_key][f"{stage}_completed_at"] = datetime.now().isoformat()
        else:
            self.data["processed_files"][file_key]["completed"] = True
            self.data["processed_files"][file_key]["completed_at"] = datetime.now().isoformat()

        # Add extra data
        self.data["processed_files"][file_key].update(extra_data)

        # Save with backup
        self.save()

    def get_stats(self):
        """Get processing statistics."""
        total = len(self.data["processed_files"])
        completed = sum(1 for f in self.data["processed_files"].values()
                       if f.get("completed", False))
        return {"total": total, "completed": completed, "pending": total - completed}

print("✅ Tracking system with backup configured\n")

# ══════════════════════════════════════════════════════════════════════════════
# 🎯 PHASE 1: AUDIO TAGS ENHANCEMENT (RAW SCRIPT → PROCESSED SCRIPT)
# ══════════════════════════════════════════════════════════════════════════════

print("="*70)
print("🎤 PHASE 1: AUDIO TAGS ENHANCEMENT")
print("="*70)

# Quick check: Skip if all files already processed
raw_files = list((BASE_DIR / "raw_scripts").glob("*.txt"))
print(f"📁 Found {len(raw_files)} raw script files")

if len(raw_files) == 0:
    print("ℹ️  No raw scripts to process\n")
    phase1_success = 0
    unprocessed_phase1 = []
else:
    phase1_tracking_temp = TrackingSystem("phase1_audio_tags", BASE_DIR)
    unprocessed_phase1 = [f for f in raw_files if not phase1_tracking_temp.is_processed(f)]

    if not unprocessed_phase1:
        print("✅ Phase 1 already complete - skipping imports and initialization\n")
        phase1_success = 0
    else:
        print(f"🔄 {len(unprocessed_phase1)} files need processing\n")

    # Only import and run if needed

    class AudioTagsProcessor:
        """Process raw scripts with Gemini to add ElevenLabs v3 Audio Tags."""

        def __init__(self, base_dir):
            self.base_dir = Path(base_dir)
            self.raw_scripts_path = self.base_dir / "raw_scripts"
            self.processed_scripts_path = self.base_dir / "processed_scripts"
            self.tracking = TrackingSystem("phase1_audio_tags", self.base_dir)

            # Audio Tags prompt template
            self.audio_tags_prompt = """# 🎤 ElevenLabs v3 Audio Tags Enhancement Prompt for YouTube Football Scripts

    **Your mission**: Transform the raw football script using ElevenLabs v3 Audio Tags to create expressive, performance-driven speech that captures every emotion, delivery nuance, and sound effect.

    ## 🎬 **AUDIO TAGS MASTERY APPROACH:**

    **You are an expert scriptwriter enhancing YouTube football scripts with ElevenLabs v3 Audio Tags. Your goal is to use Audio Tags to control emotion, pacing, delivery, and sound effects for breathtaking AI voice generation.**

    **🔥 AUDIO TAGS CATEGORIES TO MASTER:**

    **1. EMOTIONS:**
    Use emotional tags to set the tone: [excited], [angry], [happily], [sad], [sorrowful], [intense], [upbeat]

    **2. DELIVERY DIRECTION:**
    Control volume and energy: [whispers], [shouts], [quietly], [dismissive], [cheeky], [cautiously], [panicking], [surprised]

    **3. HUMAN REACTIONS:**
    Add natural reactions: [laughs], [sighs], [clears throat], [chuckles], [light chuckle], [laughs hard], [gasps]

    **4. ACCENT CHANGES:**
    Switch accents mid-script: [Australian accent], [French accent], [British accent], [American accent], [Southern US accent]

    **5. CONVERSATIONAL FLOW:**
    Handle multi-speaker elements: [interrupting], [overlapping], [jumping in], [starting to speak]

    **6. PACING CONTROL:**
    Fine-tune timing: [pause], [rushed], [drawn out], [dramatic tone], [building excitement]

    **🎯 AUDIO TAGS PLACEMENT RULES:**

    - **Place tags ANYWHERE** in your script to shape delivery in real time
    - **Use combinations** of tags within sentences for complex emotions
    - **Mix categories** for rich, layered performance: [excited] [building excitement] text here [laughs]
    - **Control speaker transitions** for multi-character dialogue
    - **Add sound effects** when contextually relevant: [clapping], [crowd noise], [whistle]

    **📝 ENHANCEMENT EXAMPLES:**

    **❌ BEFORE (Plain Text):**
    ```
    Haaland scored another goal. The crowd went wild. This is unbelievable.
    ```

    **✅ AFTER (Audio Tags Enhanced):**
    ```
    [excited] Haaland scored another goal! [crowd cheering] [shouting] The crowd went wild! [awe] This is absolutely unbelievable!
    ```

    **🔧 ADVANCED TECHNIQUES:**

    **Multi-layered Emotions:**
    - [excited] [building excitement] for escalating energy
    - [sad] [sorrowful] for deep emotional moments
    - [cheeky] [dismissive] for playful mockery

    **Dynamic Delivery Changes:**
    - Start [quietly] then build to [shouting]
    - Use [whispers] for conspiracy-like moments
    - [dramatic pause] before big reveals

    **Natural Human Reactions:**
    - [sighs] before disappointing news
    - [laughs] or [chuckles] for humor
    - [clears throat] before important points

    **Accent Switching:**
    - [British accent] for formal commentary
    - [American accent] for casual analysis
    - [Australian accent] for different perspectives

    **Raw Script:**
    ```
    {script_content}
    ```

    **🎤 CRITICAL INSTRUCTIONS:**

    1. **Your response must contain ONLY the enhanced script**
    2. **NO headers, explanations, or introductory text**
    3. **Start immediately with the script text enhanced with Audio Tags**
    4. **Use Audio Tags liberally** - this is ElevenLabs v3, it handles them beautifully
    5. **Focus on performance** - make it sound like a passionate commentator
    6. **Layer emotions** for complex delivery
    7. **Add natural reactions** throughout for realism

    **Enhanced Script with Audio Tags:**"""

        def process_script(self, script_path):
            """Process a single raw script through Gemini."""
            script_name = script_path.stem

            # Check if already processed
            if self.tracking.is_processed(script_path):
                print(f"   ⏭️  {script_name} already processed")
                return True

            print(f"   📝 Processing: {script_name}")

            # Read raw script
            try:
                with open(script_path, 'r', encoding='utf-8') as f:
                    raw_content = f.read().strip()

                if not raw_content:
                    print(f"   ❌ Empty script file")
                    return False

                print(f"   📊 Raw script: {len(raw_content)} characters")
            except Exception as e:
                print(f"   ❌ Error reading script: {e}")
                return False

            # Create prompt
            prompt = self.audio_tags_prompt.format(script_content=raw_content)

            # Call Gemini (no strict validation for Audio Tags)
            print(f"   🤖 Sending to Gemini for Audio Tags enhancement...")
            enhanced_script = call_gemini_with_retry(prompt)

            if not enhanced_script:
                print(f"   ❌ Failed to get enhanced script from Gemini")
                return False

            # Clean response (remove any headers/explanations)
            enhanced_script = self.clean_response(enhanced_script)

            print(f"   📊 Enhanced script: {len(enhanced_script)} characters")

            # Save enhanced script
            output_path = self.processed_scripts_path / f"{script_name}.txt"
            try:
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(enhanced_script)
                print(f"   ✅ Saved: {output_path.name}")
            except Exception as e:
                print(f"   ❌ Error saving: {e}")
                return False

            # Mark as processed
            self.tracking.mark_processed(script_path, output_file=str(output_path))

            return True

        def clean_response(self, response):
            """Remove headers/explanations from Gemini response."""
            lines = response.split('\n')
            cleaned_lines = []
            found_content = False

            for line in lines:
                line_stripped = line.strip()

                # Skip empty lines before content
                if not found_content and not line_stripped:
                    continue

                # Skip header lines
                if not found_content and any(pattern in line_stripped.lower() for pattern in
                    ['enhanced script', 'audio tags', "here's the", '**enhanced', '# enhanced']):
                    continue

                # Once we find actual content, start including
                if line_stripped and ('[' in line_stripped or len(line_stripped) > 20):
                    found_content = True
                    cleaned_lines.append(line)
                elif found_content:
                    cleaned_lines.append(line)

            return '\n'.join(cleaned_lines).strip()

        def process_all_unprocessed(self):
            """Process all unprocessed raw scripts."""
            print("\n🔍 Scanning for unprocessed raw scripts...")

            raw_files = list(self.raw_scripts_path.glob("*.txt"))
            print(f"   Found {len(raw_files)} .txt files in raw_scripts/")

            if not raw_files:
                print("   ℹ️  No raw scripts found")
                return 0

            unprocessed = [f for f in raw_files if not self.tracking.is_processed(f)]

            if not unprocessed:
                print("   ✅ All raw scripts already processed")
                return 0

            print(f"   🎯 {len(unprocessed)} scripts need processing:\n")

            successful = 0
            for i, script_path in enumerate(unprocessed, 1):
                print(f"\n📄 Script {i}/{len(unprocessed)}: {script_path.name}")

                if self.process_script(script_path):
                    successful += 1
                    print(f"   ✅ Success")
                else:
                    print(f"   ❌ Failed")

                # Checkpoint
                print(f"\n✅ Checkpoint: Processed {i}/{len(unprocessed)} scripts. Safe to disconnect.")

            return successful

    # Run Phase 1
    phase1_processor = AudioTagsProcessor(BASE_DIR)
    phase1_success = phase1_processor.process_all_unprocessed()

print("\n" + "="*70)
print(f"🎉 PHASE 1 COMPLETE: {phase1_success} scripts enhanced")
print("="*70)
print(f"📁 Output location: {BASE_DIR / 'processed_scripts'}")
print("="*70)

# Cleanup Phase 1 (only if it ran)
if unprocessed_phase1:
    try:
        del phase1_processor
        import gc
        gc.collect()
        time.sleep(0.5)
    except:
        pass

# ══════════════════════════════════════════════════════════════════════════════
# 🎵 PHASE 2: TTS GENERATION (PROCESSED SCRIPT → VOICEOVER)
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("🎵 PHASE 2: TTS GENERATION WITH SILENCE REMOVAL")
print("="*70)

# Quick check: Skip if all files already processed
script_files = list((BASE_DIR / "processed_scripts").glob("*.txt"))
print(f"📁 Found {len(script_files)} processed script files")

if len(script_files) == 0:
    print("ℹ️  No processed scripts to generate TTS\n")
    phase2_success = 0
    unprocessed_phase2 = []
else:
    phase2_tracking_temp = TrackingSystem("phase2_tts", BASE_DIR)
    unprocessed_phase2 = [f for f in script_files if not phase2_tracking_temp.is_processed(f)]

    if not unprocessed_phase2:
        print("✅ Phase 2 already complete - skipping ElevenLabs initialization\n")
        phase2_success = 0
    else:
        print(f"🔄 {len(unprocessed_phase2)} files need TTS generation\n")

    # Only load ElevenLabs if needed
    from dotenv import load_dotenv
    from elevenlabs.client import ElevenLabs
    from pydub import AudioSegment
    from pydub.silence import detect_nonsilent
    import io

    # Load environment variables
    env_path = BASE_DIR / '.env'
    load_dotenv(dotenv_path=env_path)

    class TTSGenerator:
        """Generate voiceovers using ElevenLabs v3 with silence removal."""

        def __init__(self, base_dir):
            self.base_dir = Path(base_dir)
            self.processed_scripts_path = self.base_dir / "processed_scripts"
            self.voiceovers_path = self.base_dir / "voiceovers"
            self.tracking = TrackingSystem("phase2_tts", self.base_dir)

            # ElevenLabs configuration
            self.api_key = os.getenv('ELEVENLABS_API_KEY')
            self.voice_id = os.getenv('ELEVENLABS_VOICE_ID', 'Joseph')

            if not self.api_key:
                raise ValueError("ELEVENLABS_API_KEY not found in .env file")

            self.client = ElevenLabs(api_key=self.api_key)

            # TTS settings
            self.model = "eleven_v3"  # Dedicated Eleven v3 model with full audio tags support
            self.chunk_size = 1200
            self.voice_settings = {
                "stability": 0.5,
                "similarity_boost": 0.8,
                "style": 0.0,
                "use_speaker_boost": True
            }

            # Silence removal settings (from your Colab code)
            self.silence_thresh = -40  # dB
            self.min_silence_len = 500  # milliseconds
            self.padding = 100  # milliseconds

            print(f"   ✅ ElevenLabs initialized")
            print(f"   Voice: {self.voice_id}")
            print(f"   Model: {self.model}")

        def chunk_text(self, text):
            """Split text into optimal chunks for TTS."""
            # Remove SSML tags for chunking
            clean_text = text.replace('<speak>', '').replace('</speak>', '').strip()

            if len(clean_text) <= self.chunk_size:
                return [f"<speak>{clean_text}</speak>"]

            chunks = []
            current_chunk = ""

            # Split by paragraphs
            paragraphs = [p.strip() for p in clean_text.split('\n\n') if p.strip()]

            for paragraph in paragraphs:
                if current_chunk and len(current_chunk + "\n\n" + paragraph) > self.chunk_size:
                    chunks.append(f"<speak>{current_chunk.strip()}</speak>")
                    current_chunk = paragraph
                elif len(paragraph) > self.chunk_size:
                    if current_chunk:
                        chunks.append(f"<speak>{current_chunk.strip()}</speak>")
                        current_chunk = ""

                    # Split long paragraph by sentences
                    import re
                    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', paragraph) if s.strip()]

                    for sentence in sentences:
                        if current_chunk and len(current_chunk + " " + sentence) > self.chunk_size:
                            chunks.append(f"<speak>{current_chunk.strip()}</speak>")
                            current_chunk = sentence
                        else:
                            current_chunk = current_chunk + " " + sentence if current_chunk else sentence
                else:
                    current_chunk = current_chunk + "\n\n" + paragraph if current_chunk else paragraph

            if current_chunk.strip():
                chunks.append(f"<speak>{current_chunk.strip()}</speak>")

            return chunks

        def generate_tts(self, text, output_path):
            """Generate TTS audio from text."""
            try:
                chunks = self.chunk_text(text)
                print(f"      Split into {len(chunks)} chunks")

                audio_segments = []

                for i, chunk in enumerate(chunks, 1):
                    try:
                        audio = self.client.text_to_speech.convert(
                            text=chunk,
                            voice_id=self.voice_id,
                            model_id=self.model,
                            output_format="mp3_44100_128",
                            voice_settings=self.voice_settings
                        )

                        chunk_audio = b''
                        for audio_chunk in audio:
                            chunk_audio += audio_chunk

                        audio_segments.append(chunk_audio)

                    except Exception as e:
                        print(f"      ❌ Chunk {i} failed: {e}")
                        return False

                    if i < len(chunks):
                        time.sleep(0.1)

                # Concatenate audio
                if len(audio_segments) == 1:
                    with open(output_path, 'wb') as f:
                        f.write(audio_segments[0])
                else:
                    try:
                        combined = AudioSegment.from_mp3(io.BytesIO(audio_segments[0]))
                        for segment in audio_segments[1:]:
                            combined += AudioSegment.from_mp3(io.BytesIO(segment))
                        combined.export(output_path, format="mp3")
                    except:
                        # Fallback: binary concatenation
                        with open(output_path, 'wb') as f:
                            for segment in audio_segments:
                                f.write(segment)

                return True

            except Exception as e:
                print(f"      ❌ TTS generation failed: {e}")
                return False

        def remove_silence(self, audio_path, output_path):
            """Remove silence from audio."""
            try:
                audio = AudioSegment.from_mp3(str(audio_path))

                nonsilent_chunks = detect_nonsilent(
                    audio,
                    min_silence_len=self.min_silence_len,
                    silence_thresh=self.silence_thresh,
                    seek_step=10
                )

                processed = AudioSegment.empty()
                for start, end in nonsilent_chunks:
                    padded_start = max(0, start - self.padding)
                    padded_end = min(len(audio), end + self.padding)
                    chunk = audio[padded_start:padded_end]
                    processed += chunk

                processed.export(str(output_path), format="mp3")
                return True

            except Exception as e:
                print(f"      ❌ Silence removal failed: {e}")
                return False

        def process_script(self, script_path):
            """Process single script to generate voiceovers."""
            script_name = script_path.stem

            if self.tracking.is_processed(script_path):
                print(f"   ⏭️  {script_name} already processed")
                return True

            print(f"   📝 Processing: {script_name}")

            # Read script
            try:
                with open(script_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()

                if not content:
                    print(f"      ❌ Empty script")
                    return False
            except Exception as e:
                print(f"      ❌ Error reading: {e}")
                return False

            # Generate original version
            original_path = self.voiceovers_path / f"{script_name}.mp3"
            print(f"      🎙️  Generating TTS...")

            if not self.generate_tts(content, original_path):
                return False

            print(f"      ✅ Original saved: {original_path.name}")

            # Generate clean version (silence removed)
            clean_path = self.voiceovers_path / f"{script_name}_clean.mp3"
            print(f"      🧹 Removing silence...")

            if not self.remove_silence(original_path, clean_path):
                return False

            print(f"      ✅ Clean saved: {clean_path.name}")

            # Mark as processed
            self.tracking.mark_processed(
                script_path,
                original_file=str(original_path),
                clean_file=str(clean_path)
            )

            return True

        def process_all_unprocessed(self):
            """Process all unprocessed scripts."""
            print("\n🔍 Scanning for unprocessed scripts...")

            script_files = list(self.processed_scripts_path.glob("*.txt"))
            print(f"   Found {len(script_files)} processed scripts")

            if not script_files:
                print("   ℹ️  No processed scripts found")
                return 0

            unprocessed = [f for f in script_files if not self.tracking.is_processed(f)]

            if not unprocessed:
                print("   ✅ All scripts already have voiceovers")
                return 0

            print(f"   🎯 {len(unprocessed)} scripts need TTS:\n")

            successful = 0
            for i, script_path in enumerate(unprocessed, 1):
                print(f"\n🎵 Script {i}/{len(unprocessed)}: {script_path.name}")

                if self.process_script(script_path):
                    successful += 1
                    print(f"   ✅ Success")
                else:
                    print(f"   ❌ Failed")

                # Checkpoint
                print(f"\n✅ Checkpoint: Generated {i}/{len(unprocessed)} voiceovers. Safe to disconnect.")

            return successful

    # Run Phase 2
    phase2_generator = TTSGenerator(BASE_DIR)
    phase2_success = phase2_generator.process_all_unprocessed()

print("\n" + "="*70)
print(f"🎉 PHASE 2 COMPLETE: {phase2_success} voiceovers generated")
print("="*70)
print(f"📁 Output location: {BASE_DIR / 'voiceovers'}")
print("="*70)

# Cleanup Phase 2 (only if it ran)
if unprocessed_phase2:
    try:
        del phase2_generator
        import gc
        gc.collect()
        time.sleep(0.5)
    except:
        pass

# ══════════════════════════════════════════════════════════════════════════════
# 🎤 WHISPER TRANSCRIPTION (VOICEOVER → TRANSCRIPT)
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("🎤 WHISPER TRANSCRIPTION")
print("="*70)

# Quick check: Skip if all files already transcribed
clean_files = list((BASE_DIR / "voiceovers").glob("*_clean.mp3"))
print(f"📁 Found {len(clean_files)} clean audio files")

if len(clean_files) == 0:
    print("ℹ️  No clean audio files to transcribe\n")
    whisper_success = 0
    whisper_model = None
    whisper_transcriber = None
    unprocessed_whisper = []
else:
    whisper_tracking_temp = TrackingSystem("whisper_transcription", BASE_DIR)

    # Check for existing transcripts FIRST (crash recovery)
    actually_unprocessed = []
    for f in clean_files:
        script_name = f.stem.replace('_clean', '')
        transcript_file = BASE_DIR / "pose_data" / f"{script_name}_transcript.json"

        # Only add if transcript doesn't exist (ignore tracking file)
        if not transcript_file.exists():
            actually_unprocessed.append(f)

    if not actually_unprocessed:
        print("✅ All transcripts exist - skipping Whisper\n")
        whisper_success = 0
        whisper_model = None
        whisper_transcriber = None
        unprocessed_whisper = []
    else:
        print(f"🔄 {len(actually_unprocessed)} files need transcription\n")
        unprocessed_whisper = actually_unprocessed

        # ========== INSTALL WHISPER DEPENDENCIES ==========
        print("📦 Installing Whisper dependencies...")
        !pip install -q phonemizer
        !pip install -q --force-reinstall "numpy==1.23.5"
        !pip install -q --force-reinstall librosa soundfile numba torchaudio
        !pip install -q --force-reinstall openai-whisper
        print("   ✅ Whisper packages installed\n")

        # Only load Whisper if needed
        import torch
        import whisper
        import librosa
        import soundfile as sf
        import numpy as np
        import numba
        from concurrent.futures import ThreadPoolExecutor
        import gc

        # Memory settings (from your working Colab code)
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:2048,roundup_power2_divisions:8"
        MAX_CONCURRENT_CHUNKS = 4
        MAX_PROCESS_WORKERS = 1

        # JIT-optimized functions (YOUR CODE)
        @numba.jit(nopython=True)
        def optimize_chunk_splitting(audio_length, sample_rate, max_duration):
            if audio_length <= max_duration * sample_rate:
                return np.array([0]), np.array([audio_length])
            chunk_samples = int(max_duration * sample_rate)
            starts = np.arange(0, audio_length, chunk_samples)
            ends = np.minimum(starts + chunk_samples, audio_length)
            return starts, ends

        @numba.jit(nopython=True)
        def has_speech_jit(audio_chunk, threshold=0.01):
            return np.max(np.abs(audio_chunk)) > threshold

        # Memory management (YOUR CODE)
        def clear_gpu_memory_aggressive():
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            gc.collect()
            time.sleep(0.1)

        def get_gpu_memory_info():
            if torch.cuda.is_available():
                total = torch.cuda.get_device_properties(0).total_memory
                reserved = torch.cuda.memory_reserved(0)
                allocated = torch.cuda.memory_allocated(0)
                free = total - allocated
                return {'total_gb': total/1024**3, 'allocated_gb': allocated/1024**3,
                        'free_gb': free/1024**3, 'reserved_gb': reserved/1024**3}
            return None

        def check_gpu_memory_for_model(required_gb=3.0):
            if not torch.cuda.is_available():
                return False, "CUDA not available"
            clear_gpu_memory_aggressive()
            mem_info = get_gpu_memory_info()
            if mem_info is None:
                return False, "Could not get memory info"
            if mem_info['free_gb'] >= required_gb:
                return True, f"Sufficient memory available: {mem_info['free_gb']:.1f}GB"
            return False, f"Insufficient memory: {mem_info['free_gb']:.1f}GB < {required_gb}GB required"

        # Device setup (YOUR CODE)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda":
            try:
                torch.cuda.set_per_process_memory_fraction(0.7)
            except:
                pass
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            # YOUR FIX for tensor dimension errors
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_math_sdp(True)
            clear_gpu_memory_aggressive()

        print(f"   Device: {device}")

        # Load Whisper model (YOUR CODE)
        memory_ok, memory_msg = check_gpu_memory_for_model(required_gb=3.0)
        if memory_ok:
            print(f"   Loading Whisper Medium on GPU: {memory_msg}")
        else:
            device = "cpu"
            print(f"   GPU memory insufficient, using CPU: {memory_msg}")

        print("   Loading Whisper Medium model...")
        whisper_model = whisper.load_model("medium", device=device)
        print("   ✅ Whisper model loaded")

        class WhisperTranscriber:
            """Transcribe audio files using Whisper - YOUR WORKING METHODOLOGY."""

            def __init__(self, base_dir, whisper_model):
                self.base_dir = Path(base_dir)
                self.voiceovers_path = self.base_dir / "voiceovers"
                self.pose_data_path = self.base_dir / "pose_data"
                self.temp_path = self.base_dir / "temp"
                self.temp_path.mkdir(exist_ok=True)

                self.model = whisper_model
                self.tracking = TrackingSystem("whisper_transcription", self.base_dir)

            def chunk_if_needed(self, audio_path, max_duration=180):
                """Split audio if longer than 3 minutes (YOUR CODE)."""
                try:
                    duration = librosa.get_duration(path=str(audio_path))
                    print(f"      Audio duration: {duration:.1f}s")
                except:
                    duration = 0

                if duration <= max_duration:
                    return [audio_path], [0]

                print(f"      Chunking needed ({duration:.1f}s > {max_duration}s)")

                # Load and split
                y, sr = librosa.load(str(audio_path), sr=None)
                chunk_samples = int(max_duration * sr)

                chunks = []
                start_times = []

                for i in range(0, len(y), chunk_samples):
                    chunk_data = y[i:i + chunk_samples]
                    start_time = i / sr

                    chunk_path = self.temp_path / f"{audio_path.stem}_chunk_{len(chunks):03d}.wav"
                    sf.write(str(chunk_path), chunk_data, sr)

                    chunks.append(chunk_path)
                    start_times.append(start_time)

                    print(f"         Chunk {len(chunks)}: {start_time:.1f}s - {(i+len(chunk_data))/sr:.1f}s")

                return chunks, start_times

            def transcribe_chunk_p100_optimized(self, chunk_info):
                """Transcribe single chunk (YOUR EXACT WORKING CODE)."""
                chunk_path, chunk_num = chunk_info
                clear_gpu_memory_aggressive()

                quality_configs = [
                    {"beam_size": 5, "best_of": 3, "temperature": 0},
                    {"beam_size": 3, "best_of": 2, "temperature": 0},
                    {"beam_size": 1, "best_of": 1, "temperature": 0}
                ]

                result = None
                for config in quality_configs:
                    try:
                        result = self.model.transcribe(
                            str(chunk_path),
                            language="en",
                            word_timestamps=True,
                            fp16=(device == "cuda"),
                            **config
                        )
                        break
                    except RuntimeError as e:
                        if "out of memory" in str(e).lower() and device == "cuda":
                            clear_gpu_memory_aggressive()
                            time.sleep(1)
                            continue
                        else:
                            raise e

                if result is None:
                    raise RuntimeError("Transcription failed for chunk and all fallbacks.")

                clear_gpu_memory_aggressive()

                words = []
                for segment in result.get('segments', []):
                    if 'words' in segment:
                        for w in segment['words']:
                            words.append({
                                'word': w['word'].strip(),
                                'start': w['start'],
                                'end': w['end'],
                                'probability': w.get('probability', 0.0)
                            })

                return {'words': words, 'chunk_num': chunk_num}

            def process_chunks_in_batches_p100(self, chunk_tasks, batch_size=MAX_CONCURRENT_CHUNKS):
                """Process chunks in batches (YOUR CODE)."""
                results = []
                for i in range(0, len(chunk_tasks), batch_size):
                    batch = chunk_tasks[i:i+batch_size]
                    clear_gpu_memory_aggressive()

                    with ThreadPoolExecutor(max_workers=len(batch)) as executor:
                        batch_results = list(executor.map(self.transcribe_chunk_p100_optimized, batch))

                    results.extend(batch_results)
                    clear_gpu_memory_aggressive()
                    time.sleep(0.5)

                return results

            def merge_results_p100_optimized(self, chunk_results, start_times):
                """Merge chunk results (YOUR CODE)."""
                merged_words = []
                for result, start_offset in zip(chunk_results, start_times):
                    if result and result.get('words'):
                        for word in result['words']:
                            merged_words.append({
                                'word': word['word'],
                                'start_time': word['start'] + start_offset,
                                'end_time': word['end'] + start_offset,
                                'confidence': word.get('probability', 0.0)
                            })
                return merged_words

            def transcribe_audio(self, audio_path):
                """Transcribe single audio file (YOUR METHODOLOGY)."""
                script_name = audio_path.stem.replace('_clean', '')

                if self.tracking.is_processed(audio_path):
                    print(f"   ⏭️  {script_name} already transcribed")
                    return True

                print(f"   🎤 Transcribing: {script_name}")

                try:
                    # Chunk if needed (YOUR CODE)
                    chunks, start_times = self.chunk_if_needed(audio_path)

                    if not chunks:
                        print(f"      ❌ No chunks generated")
                        return False

                    # Transcribe (YOUR CODE)
                    print(f"      🎙️  Transcribing...")
                    if len(chunks) == 1:
                        chunk_results = [self.transcribe_chunk_p100_optimized((chunks[0], 1))]
                    else:
                        chunk_tasks = [(chunk, i+1) for i, chunk in enumerate(chunks)]
                        chunk_results = self.process_chunks_in_batches_p100(chunk_tasks)

                    chunk_results.sort(key=lambda x: x['chunk_num'])

                    # Merge results (YOUR CODE)
                    final_words = self.merge_results_p100_optimized(chunk_results, start_times)

                    if not final_words:
                        print(f"      ❌ No words transcribed")
                        return False

                    print(f"      ✅ Transcribed {len(final_words)} words")

                    # Save transcript
                    transcript_file = self.pose_data_path / f"{script_name}_transcript.json"
                    with open(transcript_file, 'w', encoding='utf-8') as f:
                        json.dump(final_words, f, indent=2)

                    print(f"      ✅ Saved: {transcript_file.name}")

                    # Cleanup chunks
                    for chunk in chunks:
                        try:
                            if chunk != audio_path and chunk.exists():
                                chunk.unlink()
                        except:
                            pass

                    # Mark as processed
                    self.tracking.mark_processed(audio_path, transcript_file=str(transcript_file))

                    return True

                except Exception as e:
                    print(f"      ❌ Transcription error: {e}")
                    import traceback
                    traceback.print_exc()
                    return False

            def process_all_unprocessed(self):
                """Process all unprocessed audio files."""
                print("\n🔍 Scanning for untranscribed audio...")

                clean_files = list(self.voiceovers_path.glob("*_clean.mp3"))
                print(f"   Found {len(clean_files)} clean audio files")

                if not clean_files:
                    print("   ℹ️  No clean audio files found")
                    return 0

                unprocessed = [f for f in clean_files if not self.tracking.is_processed(f)]

                if not unprocessed:
                    print("   ✅ All audio files already transcribed")
                    return 0

                print(f"   🎯 {len(unprocessed)} files need transcription:\n")

                successful = 0
                for i, audio_path in enumerate(unprocessed, 1):
                    print(f"\n🎤 Audio {i}/{len(unprocessed)}: {audio_path.name}")

                    if self.transcribe_audio(audio_path):
                        successful += 1
                        print(f"   ✅ Success")
                    else:
                        print(f"   ❌ Failed")

                    # Checkpoint
                    print(f"\n✅ Checkpoint: Transcribed {i}/{len(unprocessed)} files. Safe to disconnect.")

                return successful

        # Run Whisper transcription
        whisper_transcriber = WhisperTranscriber(BASE_DIR, whisper_model)
        whisper_success = whisper_transcriber.process_all_unprocessed()

print("\n" + "="*70)
print(f"🎉 WHISPER TRANSCRIPTION COMPLETE: {whisper_success} files transcribed")
print("="*70)
print(f"📁 Output location: {BASE_DIR / 'pose_data'}")
print("="*70)

# Only cleanup if Whisper was actually loaded
if unprocessed_whisper and whisper_model is not None:
    print("\n🧹 Freeing memory (unloading Whisper model)...")

    # Delete Whisper objects
    del whisper_model
    del whisper_transcriber

    # Force garbage collection
    import gc
    gc.collect()

    # Clear GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    print("   ✅ Whisper model unloaded")
    print("   💾 ~3-4GB RAM freed\n")
    time.sleep(2)
    print("\n🔄 Auto-restarting runtime in 5 seconds...")
    print("(Phase 3 will run automatically after restart)\n")
    time.sleep(5)

    import os
    os.kill(os.getpid(), 9)

# ══════════════════════════════════════════════════════════════════════════════
# 🎭 PHASE 3: POSE PROCESSING & LIP SYNC XML GENERATION
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("🎭 PHASE 3: POSE PROCESSING & LIP SYNC")
print("="*70)

# Quick check: Skip if all files already processed
audio_files = list((BASE_DIR / "voiceovers").glob("*_clean.mp3"))
print(f"📁 Found {len(audio_files)} clean audio files")

if len(audio_files) == 0:
    print("ℹ️  No audio files to process\n")
    phase3_success = 0
    unprocessed_phase3 = []
else:
    phase3_tracking_temp = TrackingSystem("phase3_pose_lipsync", BASE_DIR)
    unprocessed_phase3 = [f for f in audio_files if not phase3_tracking_temp.is_processed(f)]

    if not unprocessed_phase3:
        print("✅ Phase 3 already complete - skipping all imports\n")
        phase3_success = 0
    else:
        print(f"🔄 {len(unprocessed_phase3)} files need processing\n")

    # ══════════════════════════════════════════════════════════════════════════════
# 🎭 PHASE 3: POSE PROCESSING & LIP SYNC
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("🎭 PHASE 3: POSE PROCESSING & LIP SYNC")
print("="*70)

# Quick check: Skip if all files already processed
audio_files = list((BASE_DIR / "voiceovers").glob("*_clean.mp3"))
print(f"📁 Found {len(audio_files)} clean audio files")

if len(audio_files) == 0:
    print("ℹ️  No audio files to process\n")
    phase3_success = 0
    unprocessed_phase3 = []
else:
    phase3_tracking_temp = TrackingSystem("phase3_pose_lipsync", BASE_DIR)
    unprocessed_phase3 = [f for f in audio_files if not phase3_tracking_temp.is_processed(f)]

    if not unprocessed_phase3:
        print("✅ Phase 3 already complete - skipping all imports\n")
        phase3_success = 0
    else:
        print(f"🔄 {len(unprocessed_phase3)} files need processing\n")

        # ========== CRITICAL: CHECK NUMPY BEFORE ANY IMPORTS ==========
        print("📦 Checking numpy environment for Phase 3...")
        import sys

        numpy_is_corrupted = False
        transformers_already_loaded = 'transformers' in sys.modules

        try:
            import numpy as np

            # Check if numpy is old version from Whisper
            if np.__version__.startswith('1.23'):
                numpy_is_corrupted = True
                print(f"⚠️  Detected Whisper's old numpy: {np.__version__}")

            # Check if transformers was already imported with old numpy
            elif transformers_already_loaded:
                # Try to actually use numpy in transformers context
                try:
                    from numpy._core.umath import isalpha  # This will fail if incompatible
                except ImportError:
                    numpy_is_corrupted = True
                    print(f"⚠️  Numpy binaries are incompatible with cached modules")
            else:
                print(f"✅ Numpy environment healthy: {np.__version__}")

        except ImportError as e:
            print(f"⚠️  Numpy check failed: {e}")
            numpy_is_corrupted = True

        # ========== RESTART IF NEEDED ==========
        if numpy_is_corrupted:
            print("\n" + "="*70)
            print("🔧 FIXING NUMPY ENVIRONMENT")
            print("="*70)

            # Uninstall problematic packages
            print("Uninstalling incompatible packages...")
            !pip uninstall -y numpy numba transformers -qq

            # Reinstall with correct versions
            print("Reinstalling with correct versions...")
            !pip install -q "numpy>=1.26.0,<2.0.0"
            !pip install -q "numba>=0.59.0,<0.62.0"
            !pip install -q transformers pandas

            print("\n" + "="*70)
            print("🔄 AUTO-RESTARTING RUNTIME")
            print("="*70)
            print("✅ Numpy fixed - Phase 3 will resume after restart")
            print("⚠️  Please re-run this cell after restart\n")
            print("="*70)

            time.sleep(3)

            # Force restart
            import os
            os.kill(os.getpid(), 9)

        # ========== SAFE TO IMPORT NOW ==========
        print("📦 Installing Phase 3 dependencies...")

        # Import all Phase 3 packages
        import random
        import re
        import xml.etree.ElementTree as ET
        import xml.dom.minidom as minidom
        import urllib.parse
        from transformers import pipeline  # NOW SAFE



    # ══════════════════════════════════════════════════════════════════════════════
    # STAGE A: POSE TAGGING
    # ══════════════════════════════════════════════════════════════════════════════

    class PoseTagger:
        """Automatic pose tagging using sentiment analysis and content detection."""

        def __init__(self, base_dir):
            self.base_dir = Path(base_dir)
            self.processed_scripts_path = self.base_dir / "processed_scripts"
            self.pose_data_path = self.base_dir / "pose_data"
            self.pose_descriptions_file = self.base_dir / "pose_assets" / "pose_descriptions.txt"

            print("      Loading pose descriptions...")
            # Load pose descriptions
            self.pose_dict = {}
            self.load_pose_descriptions()

            print("      Setting up pose categories...")

            # Pose categories from your original code
            self.DIALOGUE_POSES = ['[4]', '[5]', '[6]', '[N]', '[Q]']
            self.NARRATION_POSES = ['[4]', '[5]', '[E]', '[Q]', '[T]']
            self.ACTION_POSES = ['[W]', '[T]', '[E]']

            # Rare poses with triggers and limits
            self.RARE_POSES = {
                '[2]': {'triggers': ["wow!", "so cute!", "amazing!", "giggle", "laugh", "adorable", "awe"], 'limit': 2, 'count': 0},
                '[3]': {'triggers': ["wait", "what?", "how", "don't get it", "confused", "huh?", "unsure"], 'limit': 2, 'count': 0},
                '[7]': {'triggers': ["oops", "my bad", "sorry", "didn't mean to", "awkward", "guilty"], 'limit': 1, 'count': 0},
                '[8]': {'triggers': ["can't even", "ridiculous", "seriously?", "enough", "this is insane", "come on", "no way", "unbelievable", "fed up"], 'limit': 2, 'count': 0},
                '[R]': {'triggers': ["wait, what?", "no way", "you're joking", "wow", "incredible", "unbelievable"], 'limit': 2, 'count': 0}
            }

            # Load sentiment classifier
            self.sentiment_classifier = None

        def load_pose_descriptions(self):
            """Load pose descriptions from file."""
            if not self.pose_descriptions_file.exists():
                print(f"   ⚠️  Pose descriptions file not found: {self.pose_descriptions_file}")
                return

            with open(self.pose_descriptions_file, 'r', encoding='utf-8') as f:
                for line in f:
                    match = re.match(r"(\[[A-Z0-9]+\])\s+(.+?)\s+–\s+(.+)", line.strip())
                    if match:
                        tag, name, desc = match.groups()
                        self.pose_dict[tag] = desc.lower()

            print(f"   ✅ Loaded {len(self.pose_dict)} pose descriptions")

        def load_sentiment_classifier(self):
            """Load sentiment analysis model."""
            if self.sentiment_classifier is None:
                print("      🧠 Loading sentiment analysis model (this may take 30-60 seconds on CPU)...")
                try:
                    # Use smaller, faster model on CPU
                    from transformers import pipeline
                    self.sentiment_classifier = pipeline(
                        'sentiment-analysis',
                        model='distilbert-base-uncased-finetuned-sst-2-english',  # Smaller model
                        device=-1  # Force CPU
                    )
                    print("      ✅ Sentiment model loaded")
                except Exception as e:
                    print(f"      ⚠️  Sentiment model failed to load: {e}")
                    print("      Using rule-based sentiment detection instead")
                    self.sentiment_classifier = "FALLBACK"
            return self.sentiment_classifier

        def detect_content_type(self, text):
            """Detect if text is dialogue, narration, or action."""
            text_clean = text.strip().lower()

            action_words = ['moves', 'walks', 'runs', 'grabs', 'takes', 'puts', 'opens', 'closes', 'says', 'shouts', 'whispers']
            if any(word in text_clean for word in action_words):
                return 'action'

            dialogue_words = ['i', 'you', 'we', 'they', 'me', 'him', 'her', 'us', 'them', 'yes', 'no', 'well', 'oh', 'ah']
            if any(text_clean.startswith(word + ' ') for word in dialogue_words):
                return 'dialogue'

            return 'narration'

        def predict_pose(self, text, content_type, last_pose=None):
            """Predict appropriate pose for text segment."""
            text_lc = text.lower()

            # Check for rare poses first
            for tag, data in self.RARE_POSES.items():
                if any(trigger in text_lc for trigger in data['triggers']):
                    if data['count'] < data['limit'] and tag != last_pose:
                        self.RARE_POSES[tag]['count'] += 1
                        return tag

            # Sentiment analysis
            try:
                classifier = self.load_sentiment_classifier()
                if classifier == "FALLBACK":
                    # Rule-based fallback
                    sentiment = 'NEUTRAL'
                    confidence = 0.5
                else:
                    result = classifier(text[:512])[0]
                    sentiment = result['label'].upper()
                    confidence = result['score']
            except Exception as e:
                print(f"      ⚠️  Sentiment analysis error: {e}")
                sentiment = 'NEUTRAL'
                confidence = 0.5

            # Choose pose pool based on content type
            if content_type == 'dialogue':
                available_poses = self.DIALOGUE_POSES
            elif content_type == 'action':
                available_poses = self.ACTION_POSES
            else:
                available_poses = self.NARRATION_POSES

            # Filter out last pose to avoid repetition
            available_poses = [p for p in available_poses if p != last_pose]

            # Sentiment-based selection
            if sentiment == 'POSITIVE' and confidence > 0.7:
                positive_poses = [p for p in available_poses if p in ['[4]', '[5]', '[Q]']]
                if positive_poses:
                    return random.choice(positive_poses)

            elif sentiment == 'NEGATIVE' and confidence > 0.7:
                negative_poses = [p for p in available_poses if p in ['[6]', '[N]', '[T]']]
                if negative_poses:
                    return random.choice(negative_poses)

            # Default: random selection
            return random.choice(available_poses) if available_poses else '[4]'

        def split_at_punctuation(self, text):
            """Split text at punctuation marks."""
            # Remove existing pose tags
            text = re.sub(r'\[[A-Z0-9]{1,2}\]\s*', '', text).strip()

            if not text:
                return []

            # Split at punctuation
            segments = re.split(r'[,.!?;:]', text)

            # Clean segments
            cleaned_segments = []
            for segment in segments:
                cleaned_segment = re.sub(r'[^\w\s]', ' ', segment)
                cleaned_segment = re.sub(r'\s+', ' ', cleaned_segment).strip()

                if cleaned_segment:
                    cleaned_segments.append(cleaned_segment)

            return cleaned_segments

        def split_long_segment(self, segment, content_type, prev_pose):
            """Split segments longer than 7 words."""
            words = segment.split()
            if len(words) <= 7:
                return [(segment, None)]

            # Split into chunks of ~7 words
            chunks = []
            current_chunk = []

            for word in words:
                current_chunk.append(word)
                if len(current_chunk) >= 7:
                    chunks.append(' '.join(current_chunk))
                    current_chunk = []

            # Add remaining words to last chunk
            if current_chunk:
                if chunks:
                    chunks[-1] += ' ' + ' '.join(current_chunk)
                else:
                    chunks.append(' '.join(current_chunk))

            # Assign poses
            result = []
            last_pose = prev_pose

            for chunk in chunks:
                pose = self.predict_pose(chunk, content_type, last_pose)
                result.append((chunk, pose))
                last_pose = pose

            return result

        def tag_script(self, script_path):
            """Tag a script with poses."""
            script_name = script_path.stem
            poses_output = self.pose_data_path / f"{script_name}_poses.txt"

            # Check if already exists
            if poses_output.exists():
                print(f"   ⏭️  {script_name} already has pose tags")
                return True

            print(f"   🎭 Tagging poses for: {script_name}")

            # Read script
            try:
                with open(script_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            except Exception as e:
                print(f"   ❌ Error reading script: {e}")
                return False

            # Reset rare pose counters
            for pose_data in self.RARE_POSES.values():
                pose_data['count'] = 0

            # Process script
            lines = content.split('\n')
            output_lines = []
            prev_pose = None

            for line in lines:
                line = line.strip()
                if not line:
                    output_lines.append("")
                    continue

                # Split at punctuation
                segments = self.split_at_punctuation(line)

                for segment in segments:
                    if not segment.strip():
                        continue

                    # Detect content type
                    content_type = self.detect_content_type(segment)

                    # Split long segments
                    segment_chunks = self.split_long_segment(segment, content_type, prev_pose)

                    for chunk, chunk_pose in segment_chunks:
                        if chunk_pose is None:
                            pose = self.predict_pose(chunk, content_type, prev_pose)
                        else:
                            pose = chunk_pose

                        output_lines.append(f"{pose} {chunk}")
                        prev_pose = pose

            # Save pose-tagged output
            try:
                with open(poses_output, "w", encoding='utf-8') as f:
                    for line in output_lines:
                        f.write(line + "\n")
                print(f"   ✅ Saved: {poses_output.name}")
                return True
            except Exception as e:
                print(f"   ❌ Error saving: {e}")
                return False

    # ══════════════════════════════════════════════════════════════════════════════
    # STAGE C: TRANSCRIPT-PRIMARY POSE MAPPING WITH GEMINI
    # ══════════════════════════════════════════════════════════════════════════════

    class PoseMapper:
        """Map poses to transcript using Gemini with strict validation."""

        def __init__(self, base_dir):
            self.base_dir = Path(base_dir)
            self.pose_data_path = self.base_dir / "pose_data"

        def create_mapping_prompt(self, transcript_text, poses_content):
            """Create prompt for Gemini to map poses to transcript."""

            prompt = f"""# POSE MAPPING: PRESERVE TRANSCRIPT EXACTLY

    ## CRITICAL REQUIREMENTS:
    1. PRESERVE EVERY CHARACTER of the transcript text below (spaces, punctuation, capitalization, commas, periods, apostrophes, hyphens - EVERYTHING)
    2. Only ADD pose tags in square brackets like [4], [5], [Q], etc.
    3. Put each pose segment on a NEW LINE
    4. DO NOT change, remove, or modify ANY words from the transcript
    5. DO NOT add or remove spaces, punctuation, or any characters

    ## TRANSCRIPT TEXT (preserve EXACTLY as written):
    {transcript_text}

    ## POSE-TAGGED SCRIPT (reference for pose placement only):
    {poses_content}

    ## MANDATORY VERIFICATION PROCESS:
    1. Before starting, scan the transcript for unique formatting patterns:
      - How are numbers formatted? (86,000 vs 86 ,000 vs 86, 000)
      - How are hyphens spaced? (-1 vs - 1 vs -1)
      - What contractions exist? (can't vs can t)
      - Any unusual punctuation spacing?

    2. Create a "formatting checklist" of 10-15 tricky strings from YOUR transcript

    3. After pose-mapping, verify each item on your checklist appears exactly as in original

    4. Only output after 100% verification

    ## EXACT OUTPUT FORMAT REQUIRED:
    Each line should be: [POSE_TAG] transcript_text_segment

    ## CHARACTER PRESERVATION RULE:
    Preserve ALL spacing, punctuation, and formatting EXACTLY as it appears in YOUR specific transcript, including:
    - Any spaces before or after punctuation (commas, hyphens, periods)
    - All apostrophes and contractions as written
    - All capitalization as written
    - All number formatting as written
    - All hyphenation patterns as written

    Do NOT assume standard formatting - follow what's actually in the transcript.

    ## POSE PLACEMENT RULES:
    - Insert poses at natural speech breaks (periods, exclamation marks, question marks)
    - Use pose tags from the reference script: [4], [5], [6], [N], [Q], [T], [E], [W], etc.
    - Aim for 5-15 words per pose segment
    - Each pose segment on its own line

    ## DYNAMIC VERIFICATION:
    Instead of checking pre-defined examples, identify and verify the 10 most formatting-sensitive phrases from THIS specific transcript, such as:
    - All monetary amounts and their exact spacing
    - All hyphenated words and their exact spacing
    - All contractions and apostrophes
    - All numbers and scores
    - Any unusual punctuation patterns

    ## SELF-CHECK REQUIREMENT:
    Before outputting, ask yourself: "If I randomly selected any 20-character substring from my output and compared it to the original transcript, would they match character-for-character including all spaces and punctuation?"

    If the answer is not "yes," continue checking and correcting.

    ## OUTPUT INSTRUCTION:
    Return ONLY the pose-mapped transcript with each [POSE] segment on a new line, preserving every character of the original transcript text exactly.

    Pose-mapped transcript:"""

            return prompt

        def validate_mapping_strict(self, response, original_transcript):
            """STRICT validation: transcript text must be preserved exactly."""

            def extract_text_only(content):
                """Extract text without pose tags."""
                text = re.sub(r'\[[^\]]+\]', '', content)
                text = re.sub(r'\s+', ' ', text).strip()
                return text

            def normalize_for_comparison(text):
                """Normalize for comparison."""
                return re.sub(r'\s+', ' ', text.lower().strip())

            # Extract text from response
            response_text = extract_text_only(response)
            original_text = extract_text_only(original_transcript)

            # Normalize for comparison
            response_norm = normalize_for_comparison(response_text)
            original_norm = normalize_for_comparison(original_text)

            if response_norm == original_norm:
                return True, None
            else:
                # Find first difference
                error_msg = f"Transcript text was modified.\n"
                error_msg += f"Original length: {len(original_text)} chars\n"
                error_msg += f"Response length: {len(response_text)} chars\n"

                # Show first difference
                for i, (a, b) in enumerate(zip(original_norm, response_norm)):
                    if a != b:
                        start = max(0, i - 30)
                        end = min(len(original_norm), i + 30)
                        error_msg += f"\nFirst difference at position {i}:\n"
                        error_msg += f"Original: ...{original_norm[start:end]}...\n"
                        error_msg += f"Response: ...{response_norm[start:end]}...\n"
                        break

                return False, error_msg

        def map_poses_to_transcript(self, script_name):
            """Map poses to transcript using Gemini."""

            # Find pose file
            possible_patterns = [
                f"{script_name}_poses.txt",
                f"{script_name}_clean_poses.txt",
            ]

            poses_file = None
            for pattern in possible_patterns:
                test_file = self.pose_data_path / pattern
                if test_file.exists():
                    poses_file = test_file
                    break

            if not poses_file:
                print(f"   ❌ Pose file not found for {script_name}")
                return False

            # Find transcript file
            possible_patterns = [
                f"{script_name}_transcript.json",
                f"{script_name}_clean_transcript.json",
            ]

            transcript_file = None
            for pattern in possible_patterns:
                test_file = self.pose_data_path / pattern
                if test_file.exists():
                    transcript_file = test_file
                    break

            if not transcript_file:
                print(f"   ❌ Transcript file not found for {script_name}")
                return False

            print(f"   🗺️  Mapping poses for: {script_name}")

            # Read files
            with open(poses_file, 'r', encoding='utf-8') as f:
                poses_content = f.read()

            with open(transcript_file, 'r', encoding='utf-8') as f:
                transcript_data = json.load(f)

            # Extract transcript text
            transcript_text = " ".join(item.get('word', '').strip() for item in transcript_data if item.get('word', '').strip())

            print(f"   📄 Transcript: {len(transcript_text)} characters")
            print(f"   📄 Poses: {len(poses_content)} characters")

            # Create prompt
            prompt = self.create_mapping_prompt(transcript_text, poses_content)

            # Call Gemini with strict validation
            print(f"   🤖 Calling Gemini for pose mapping...")

            mapped_result = call_gemini_with_retry(
                prompt,
                validation_func=lambda resp: self.validate_mapping_strict(resp, transcript_text),
                validation_name="Pose mapping"
            )

            if not mapped_result:
                print(f"   ❌ Pose mapping failed after all retries")
                return False

            # Save mapped script
            mapped_file = self.pose_data_path / f"{script_name}_poses.txt"
            with open(mapped_file, 'w', encoding='utf-8') as f:
                f.write(mapped_result)

            print(f"   ✅ Pose mapping saved: {mapped_file.name}")

            return True

    # ══════════════════════════════════════════════════════════════════════════════
    # STAGE C2: AUTOMATED TIMING ALIGNMENT
    # ══════════════════════════════════════════════════════════════════════════════

    class TimingAligner:
        """Align pose-mapped script with transcript timestamps."""

        def __init__(self, base_dir):
            self.base_dir = Path(base_dir)
            self.pose_data_path = self.base_dir / "pose_data"

        def normalize(self, text):
            """Normalize text for comparison."""
            return re.sub(r"[^a-z0-9]", "", text.lower())

        def align_timing(self, script_name):
            """Align pose segments with transcript timestamps."""

            aligned_script_file = self.pose_data_path / f"{script_name}_poses.txt"
            transcript_file = self.pose_data_path / f"{script_name}_transcript.json"

            # Alternative transcript patterns
            if not transcript_file.exists():
                transcript_file = self.pose_data_path / f"{script_name}_clean_transcript.json"

            if not aligned_script_file.exists() or not transcript_file.exists():
                print(f"   ❌ Missing files for timing alignment")
                return False

            print(f"   ⏱️  Aligning timing for: {script_name}")

            # Load files
            with open(aligned_script_file, 'r', encoding='utf-8') as f:
                script_lines = f.read().splitlines()

            with open(transcript_file, 'r', encoding='utf-8') as f:
                words_data = json.load(f)

            # Normalize transcript words
            norm_words = []
            for w in words_data:
                if w.get('word', '').strip():
                    norm_words.append({
                        "orig": w["word"],
                        "norm": self.normalize(w["word"]),
                        "start": float(w.get("start_time", 0)),
                        "end": float(w.get("end_time", 0))
                    })

            # Parse script segments
            segments = []
            for line in script_lines:
                line = line.strip()
                if not line:
                    continue

                m = re.match(r"^\[(.*?)\]\s*(.+)$", line)
                if m:
                    tag = m.group(1).strip()
                    text = m.group(2).strip()
                    words_norm = [self.normalize(w) for w in text.split()]
                    segments.append({"tag": tag, "text": text, "norm_words": words_norm})

            print(f"      Found {len(segments)} segments to align")

            if not segments:
                print(f"   ❌ No valid segments found")
                return False

            # Align segments with transcript
            aligned_segments = []
            idx = 0  # pointer into norm_words

            for i, seg in enumerate(segments):
                # Find start
                start_time = None
                for nw in seg["norm_words"]:
                    while idx < len(norm_words) and norm_words[idx]["norm"] != nw:
                        idx += 1
                    if idx < len(norm_words):
                        start_time = norm_words[idx]["start"]
                        break

                if start_time is None:
                    continue

                # Find end
                if i + 1 < len(segments):
                    # Lookahead to next segment's first word
                    next_first = segments[i+1]["norm_words"][0]
                    j = idx
                    next_start = None
                    while j < len(norm_words):
                        if norm_words[j]["norm"] == next_first:
                            next_start = norm_words[j]["start"]
                            break
                        j += 1
                    end_time = next_start if next_start else norm_words[idx]["end"]
                else:
                    # Last segment
                    last_match = None
                    for nw in seg["norm_words"]:
                        while idx < len(norm_words) and norm_words[idx]["norm"] != nw:
                            idx += 1
                        if idx < len(norm_words):
                            last_match = norm_words[idx]["end"]
                    end_time = last_match if last_match else norm_words[-1]["end"]

                aligned_segments.append({
                    "tag": seg["tag"],
                    "text": seg["text"],
                    "start": start_time,
                    "end": end_time
                })

            # Format as MM:SS.mmm
            def format_time(t):
                m = int(t // 60)
                s = int(t % 60)
                ms = int((t - int(t)) * 1000)
                return f"{m:02d}:{s:02d}.{ms:03d}"

            # Create mapping output
            output_lines = []
            for seg in aligned_segments:
                output_lines.append(
                    f"[{format_time(seg['start'])} - {format_time(seg['end'])}] [{seg['tag']}]: {seg['text']}"
                )

            final_output = "\n".join(output_lines)

            # Save mapping file
            mapping_output = self.pose_data_path / f"{script_name}_mapping.txt"
            with open(mapping_output, 'w', encoding='utf-8') as f:
                f.write(final_output)

            print(f"   ✅ Aligned {len(aligned_segments)}/{len(segments)} segments")
            print(f"   ✅ Saved: {mapping_output.name}")

            return True

    # ══════════════════════════════════════════════════════════════════════════════
    # LIP SYNC: PHONEME MAPPING & MOUTH TIMING
    # ══════════════════════════════════════════════════════════════════════════════

    class PhonemeMapper:
        """Map words to mouth movements."""

        def __init__(self):
            self.special_words = {
                'THE': [('D', 0.3), ('Uh', 0.7)],
                'YOU': [('Ee', 0.4), ('W-oo', 0.6)],
                'YOUR': [('Ee', 0.3), ('R', 0.7)],
                'WHAT': [('W-oo', 0.4), ('Aa', 0.3), ('D', 0.3)],
                'WHO': [('W-oo', 1.0)],
                'WHERE': [('W-oo', 0.3), ('Ee', 0.3), ('R', 0.4)],
                'WHEN': [('W-oo', 0.3), ('Ee', 0.3), ('D', 0.4)],
                'WHY': [('W-oo', 0.3), ('Aa', 0.7)],
                'HOW': [('Uh', 0.3), ('W-oo', 0.7)],
                'YEAH': [('Ee', 0.4), ('Aa', 0.6)],
                'NAH': [('D', 0.3), ('Aa', 0.7)],
            }

            self.use_phonemizer = False
            try:
                from phonemizer import phonemize
                self.use_phonemizer = True
            except:
                pass

        def word_to_phonemes(self, word):
            """Convert word to phonemes."""
            word = word.strip().upper()

            if not word:
                return []

            clean_word = re.sub(r"[^A-Z']", '', word)

            if not clean_word:
                return []

            word = clean_word

            if word in self.special_words:
                return self.special_words[word]

            if self.use_phonemizer:
                return self._phonemize_with_library(word)
            else:
                return self._phonemize_fallback(word)

        def _phonemize_with_library(self, word):
            """Use phonemizer library."""
            try:
                from phonemizer import phonemize
                ipa = phonemize(word.lower(), language='en-us', backend='espeak')

                ipa_map = {
                    'ɑ': ('Aa', 1.5), 'æ': ('Aa', 1.5), 'ʌ': ('Uh', 1.5),
                    'ɔ': ('Oh', 1.5), 'aʊ': ('W-oo', 1.5), 'aɪ': ('Aa', 1.5),
                    'ɛ': ('Ee', 1.5), 'ɜ': ('R', 1.5), 'eɪ': ('Ee', 1.5),
                    'ɪ': ('Ee', 1.5), 'i': ('Ee', 1.5), 'oʊ': ('Oh', 1.5),
                    'ɔɪ': ('Oh', 1.5), 'ʊ': ('Uh', 1.5), 'u': ('W-oo', 1.5),
                    'b': ('M', 0.5), 'tʃ': ('S', 1.0), 'd': ('D', 0.5),
                    'ð': ('D', 1.0), 'f': ('F', 1.0), 'g': ('D', 0.5),
                    'h': ('Uh', 0.5), 'dʒ': ('S', 1.0), 'k': ('D', 0.5),
                    'l': ('L', 1.0), 'm': ('M', 1.0), 'n': ('D', 0.5),
                    'ŋ': ('D', 0.5), 'p': ('M', 0.5), 'r': ('R', 1.0),
                    's': ('S', 1.0), 'ʃ': ('S', 1.0), 't': ('D', 0.5),
                    'θ': ('F', 1.0), 'v': ('F', 1.0), 'w': ('W-oo', 1.0),
                    'j': ('Ee', 0.5), 'z': ('S', 1.0), 'ʒ': ('S', 1.0),
                }

                phonemes = []
                for char in ipa:
                    if char in ipa_map:
                        phonemes.append(ipa_map[char])

                return phonemes if phonemes else [('Uh', 1.0)]
            except:
                return self._phonemize_fallback(word)

        def _phonemize_fallback(self, word):
            """Simple fallback phoneme mapping."""
            phonemes = []
            i = 0

            while i < len(word):
                # Two-letter combinations
                if i < len(word) - 1:
                    two = word[i:i+2]
                    if two in ['TH', 'CH', 'SH', 'PH', 'WH']:
                        if two == 'TH':
                            phonemes.append(('F', 1.0))
                        elif two in ['CH', 'SH']:
                            phonemes.append(('S', 1.0))
                        elif two == 'PH':
                            phonemes.append(('F', 1.0))
                        elif two == 'WH':
                            phonemes.append(('W-oo', 1.0))
                        i += 2
                        continue
                    elif two in ['OO', 'EE']:
                        if two == 'OO':
                            phonemes.append(('W-oo', 1.5))
                        else:
                            phonemes.append(('Ee', 1.5))
                        i += 2
                        continue

                # Single letters
                char = word[i]
                if char == 'A':
                    phonemes.append(('Aa', 1.5))
                elif char == 'E':
                    phonemes.append(('Ee', 1.5))
                elif char == 'I':
                    phonemes.append(('Ee', 1.5))
                elif char == 'O':
                    phonemes.append(('Oh', 1.5))
                elif char == 'U':
                    phonemes.append(('Uh', 1.5))
                elif char in 'BP':
                    phonemes.append(('M', 0.5))
                elif char in 'CDGKQT':
                    phonemes.append(('D', 0.5))
                elif char in 'FV':
                    phonemes.append(('F', 1.0))
                elif char == 'L':
                    phonemes.append(('L', 1.0))
                elif char == 'M':
                    phonemes.append(('M', 1.0))
                elif char == 'R':
                    phonemes.append(('R', 1.0))
                elif char in 'SZ':
                    phonemes.append(('S', 1.0))
                elif char == 'W':
                    phonemes.append(('W-oo', 1.0))

                i += 1

            return phonemes if phonemes else [('Uh', 1.0)]

    class MouthTimingGenerator:
        """Generate mouth timeline from transcript."""

        def __init__(self):
            self.phoneme_mapper = PhonemeMapper()

        def generate_mouth_timeline(self, word_transcript):
            """Generate mouth movements with Neutral filling gaps."""
            mouth_timeline = []

            # Process each word
            for word_data in word_transcript:
                word = word_data['word'].strip()
                start_time = word_data['start_time']
                end_time = word_data['end_time']
                word_duration = end_time - start_time

                if word_duration <= 0:
                    continue

                phonemes = self.phoneme_mapper.word_to_phonemes(word)

                if not phonemes:
                    phonemes = [('Uh', 1.0)]

                total_weight = sum(weight for _, weight in phonemes)

                # Add phoneme movements
                current_time = start_time
                for phoneme, weight in phonemes:
                    phoneme_duration = (weight / total_weight) * word_duration

                    if phoneme_duration < 0.04:
                        current_time += phoneme_duration
                        continue

                    phoneme_end = current_time + phoneme_duration

                    mouth_timeline.append({
                        'mouth_png': f'{phoneme}.png',
                        'start': current_time,
                        'end': phoneme_end,
                        'duration': phoneme_duration
                    })

                    current_time = phoneme_end

            # Sort by start time
            mouth_timeline.sort(key=lambda x: x['start'])

            # Fill gaps with Neutral
            filled_timeline = []

            # Add Neutral at beginning if needed
            if mouth_timeline and mouth_timeline[0]['start'] > 0:
                filled_timeline.append({
                    'mouth_png': 'Neutral.png',
                    'start': 0,
                    'end': mouth_timeline[0]['start'],
                    'duration': mouth_timeline[0]['start']
                })

            # Fill gaps between segments
            for i, segment in enumerate(mouth_timeline):
                filled_timeline.append(segment)

                if i < len(mouth_timeline) - 1:
                    current_end = segment['end']
                    next_start = mouth_timeline[i + 1]['start']
                    gap = next_start - current_end

                    if gap > 0.01:
                        filled_timeline.append({
                            'mouth_png': 'Neutral.png',
                            'start': current_end,
                            'end': next_start,
                            'duration': gap
                        })

            # Add Neutral at end
            if filled_timeline:
                last_end = filled_timeline[-1]['end']
                filled_timeline.append({
                    'mouth_png': 'Neutral.png',
                    'start': last_end,
                    'end': last_end + 2.0,
                    'duration': 2.0
                })

            # Optimize: merge consecutive Neutrals
            optimized = self._optimize_timeline(filled_timeline)

            return optimized

        def _optimize_timeline(self, timeline):
            """Merge consecutive identical mouths."""
            if not timeline:
                return []

            optimized = []
            current = timeline[0].copy()

            for item in timeline[1:]:
                if (item['mouth_png'] == current['mouth_png'] and
                    abs(item['start'] - current['end']) < 0.01):
                    current['end'] = item['end']
                    current['duration'] = current['end'] - current['start']
                else:
                    optimized.append(current)
                    current = item.copy()

            optimized.append(current)
            return optimized

    # ══════════════════════════════════════════════════════════════════════════════
    # XML GENERATION FOR DAVINCI RESOLVE
    # ══════════════════════════════════════════════════════════════════════════════

    class DaVinciXMLGenerator:
        """Generate DaVinci Resolve XML for lip sync."""

        def __init__(self, base_dir, fps=30, local_base_dir=None):
            self.base_dir = Path(base_dir)
            self.fps = fps
            self.width = 1920
            self.height = 1080

            # Use local directory for XML paths (for DaVinci Resolve on local machine)
            if local_base_dir is None:
                # Default to Windows local path
                local_base_dir = r"C:\Users\JESUPELUMI\Documents\Angry Dude Scripts"

            self.local_base = Path(local_base_dir)

            # Use LOCAL paths for XML (where DaVinci will find them)
            self.pose_dir = str(self.local_base / "pose_assets" / "poses")
            self.mouth_dir = str(self.local_base / "pose_assets" / "poses" / "mouth_movements")
            self.audio_dir = str(self.local_base / "voiceovers")

        def seconds_to_frames(self, seconds):
            return int(round(seconds * self.fps))

        def encode_path(self, path):
            """Convert path to proper file:// URL format for DaVinci Resolve."""
            # Normalize to forward slashes
            normalized = path.replace('\\', '/')

            # Return proper file URL format
            if normalized.startswith('/'):
                # Unix path
                return f"file://{normalized}"
            else:
                # Windows path - needs three slashes
                return f"file:///{normalized}"

        def generate_xml(self, script_name, pose_timeline, mouth_timeline, audio_file=None):
            """Generate DaVinci XML."""

            max_pose = max([s['end'] for s in pose_timeline])
            max_mouth = max([s['end'] for s in mouth_timeline]) if mouth_timeline else 0
            total_duration = max(max_pose, max_mouth)
            total_frames = self.seconds_to_frames(total_duration)

            root = ET.Element("xmeml", version="4")
            seq = ET.SubElement(root, "sequence", id="sequence-1")
            ET.SubElement(seq, "name").text = f"{script_name}_lipsync"
            ET.SubElement(seq, "duration").text = str(total_frames)

            rate = ET.SubElement(seq, "rate")
            ET.SubElement(rate, "timebase").text = str(self.fps)
            ET.SubElement(rate, "ntsc").text = "FALSE"

            media = ET.SubElement(seq, "media")
            video = ET.SubElement(media, "video")

            fmt = ET.SubElement(video, "format")
            sc = ET.SubElement(fmt, "samplecharacteristics")
            rate2 = ET.SubElement(sc, "rate")
            ET.SubElement(rate2, "timebase").text = str(self.fps)
            ET.SubElement(rate2, "ntsc").text = "FALSE"
            ET.SubElement(sc, "width").text = str(self.width)
            ET.SubElement(sc, "height").text = str(self.height)

            # Pose track
            pose_track = ET.SubElement(video, "track")

            for i, seg in enumerate(pose_timeline, 1):
                start_f = self.seconds_to_frames(seg['start'])
                end_f = self.seconds_to_frames(seg['end'])
                dur_f = end_f - start_f

                if dur_f <= 0:
                    continue

                pose_tag = seg['pose'].replace('[', '').replace(']', '')
                pose_file = f"{pose_tag}.png"
                pose_path = f"{self.pose_dir}/{pose_file}"

                clip = ET.SubElement(pose_track, "clipitem", id=f"pose-{i}")
                ET.SubElement(clip, "name").text = pose_file
                ET.SubElement(clip, "start").text = str(start_f)
                ET.SubElement(clip, "end").text = str(end_f)
                ET.SubElement(clip, "in").text = "0"
                ET.SubElement(clip, "out").text = str(dur_f)

                file_elem = ET.SubElement(clip, "file", id=f"pfile-{i}")
                ET.SubElement(file_elem, "name").text = pose_file
                ET.SubElement(file_elem, "pathurl").text = self.encode_path(pose_path)

            # Mouth track
            mouth_track = ET.SubElement(video, "track")

            for i, seg in enumerate(mouth_timeline, 1):
                start_f = self.seconds_to_frames(seg['start'])
                end_f = self.seconds_to_frames(seg['end'])
                dur_f = end_f - start_f

                if dur_f <= 0:
                    continue

                mouth_file = seg['mouth_png']
                mouth_path = f"{self.mouth_dir}/{mouth_file}"

                clip = ET.SubElement(mouth_track, "clipitem", id=f"mouth-{i}")
                ET.SubElement(clip, "name").text = mouth_file
                ET.SubElement(clip, "start").text = str(start_f)
                ET.SubElement(clip, "end").text = str(end_f)
                ET.SubElement(clip, "in").text = "0"
                ET.SubElement(clip, "out").text = str(dur_f)

                file_elem = ET.SubElement(clip, "file", id=f"mfile-{i}")
                ET.SubElement(file_elem, "name").text = mouth_file
                ET.SubElement(file_elem, "pathurl").text = self.encode_path(mouth_path)

            # Audio
            if audio_file:
                audio = ET.SubElement(media, "audio")
                audio_track = ET.SubElement(audio, "track")

                audio_path = f"{self.audio_dir}/{audio_file}"

                clip = ET.SubElement(audio_track, "clipitem", id="audio-1")
                ET.SubElement(clip, "name").text = audio_file
                ET.SubElement(clip, "start").text = "0"
                ET.SubElement(clip, "end").text = str(total_frames)
                ET.SubElement(clip, "in").text = "0"
                ET.SubElement(clip, "out").text = str(total_frames)

                file_elem = ET.SubElement(clip, "file", id="afile-1")
                ET.SubElement(file_elem, "name").text = audio_file
                ET.SubElement(file_elem, "pathurl").text = self.encode_path(audio_path)

            rough = ET.tostring(root, 'unicode')
            parsed = minidom.parseString(rough)
            pretty = parsed.toprettyxml(indent="  ")

            lines = [line for line in pretty.split('\n') if line.strip()]
            return '\n'.join(lines)

    # ══════════════════════════════════════════════════════════════════════════════
    # PHASE 3 ORCHESTRATOR
    # ══════════════════════════════════════════════════════════════════════════════

    class Phase3Orchestrator:
        """Orchestrate all Phase 3 stages."""

        def __init__(self, base_dir):
            self.base_dir = Path(base_dir)
            self.processed_scripts_path = self.base_dir / "processed_scripts"
            self.voiceovers_path = self.base_dir / "voiceovers"
            self.pose_data_path = self.base_dir / "pose_data"
            self.final_videos_path = self.base_dir / "final_videos"

            self.tracking = TrackingSystem("phase3_pose_lipsync", self.base_dir)

            # Initialize components
            self.pose_tagger = PoseTagger(self.base_dir)
            self.pose_mapper = PoseMapper(self.base_dir)
            self.timing_aligner = TimingAligner(self.base_dir)
            self.mouth_generator = MouthTimingGenerator()
            self.xml_generator = DaVinciXMLGenerator(self.base_dir)

        def process_script(self, audio_file):
            """Process single script through all Phase 3 stages."""
            script_name = audio_file.stem.replace('_clean', '')

            if self.tracking.is_processed(audio_file):
                print(f"   ⏭️  {script_name} already processed")
                return True

            print(f"   🎬 Processing: {script_name}")

            # Find script file
            possible_patterns = [
                f"{script_name}.txt",
                f"{script_name}_clean.txt",
            ]

            script_file = None
            for pattern in possible_patterns:
                test_file = self.processed_scripts_path / pattern
                if test_file.exists():
                    script_file = test_file
                    break

            if not script_file:
                print(f"      ❌ Script file not found")
                return False

            # Stage A: Pose Tagging
            if not self.tracking.is_processed(audio_file, "stage_a"):
                print(f"      🎭 Stage A: Pose tagging...")
                if not self.pose_tagger.tag_script(script_file):
                    return False
                self.tracking.mark_processed(audio_file, "stage_a")
            else:
                print(f"      ✅ Stage A already complete")

            # Stage B is Whisper transcription (already done in previous section)
            if not self.tracking.is_processed(audio_file, "stage_b"):
                # Check if transcript exists
                transcript_file = self.pose_data_path / f"{script_name}_transcript.json"
                if not transcript_file.exists():
                    print(f"      ❌ Transcript not found (should have been created by Whisper)")
                    return False
                self.tracking.mark_processed(audio_file, "stage_b")
            else:
                print(f"      ✅ Stage B already complete")

            # Stage C: Pose Mapping with Gemini
            if not self.tracking.is_processed(audio_file, "stage_c"):
                print(f"      🗺️  Stage C: Pose mapping...")
                if not self.pose_mapper.map_poses_to_transcript(script_name):
                    return False
                self.tracking.mark_processed(audio_file, "stage_c")
            else:
                print(f"      ✅ Stage C already complete")

            # Stage C2: Timing Alignment
            if not self.tracking.is_processed(audio_file, "stage_c2"):
                print(f"      ⏱️  Stage C2: Timing alignment...")
                if not self.timing_aligner.align_timing(script_name):
                    return False
                self.tracking.mark_processed(audio_file, "stage_c2")
            else:
                print(f"      ✅ Stage C2 already complete")

            # Stage D: XML Generation with Lip Sync
            if not self.tracking.is_processed(audio_file, "stage_d"):
                print(f"      🎬 Stage D: XML generation...")

                # Load mapping file
                mapping_file = self.pose_data_path / f"{script_name}_mapping.txt"
                if not mapping_file.exists():
                    print(f"      ❌ Mapping file not found")
                    return False

                with open(mapping_file, 'r', encoding='utf-8') as f:
                    mapping_content = f.read()

                # Parse pose timeline
                pose_timeline = []
                pattern = r'\[(\d{2}:\d{2}\.\d{3})\s*-\s*(\d{2}:\d{2}\.\d{3})\]\s*\[([^\]]+)\]'

                for match in re.finditer(pattern, mapping_content):
                    start_str, end_str, pose = match.groups()

                    def parse_time(t):
                        parts = t.split(':')
                        minutes = int(parts[0])
                        seconds = float(parts[1])
                        return minutes * 60 + seconds

                    start = parse_time(start_str)
                    end = parse_time(end_str)

                    pose_timeline.append({
                        'pose': pose.strip(),
                        'start': start,
                        'end': end,
                        'duration': end - start
                    })

                # Load transcript for mouth movements
                transcript_file = self.pose_data_path / f"{script_name}_transcript.json"
                if not transcript_file.exists():
                    transcript_file = self.pose_data_path / f"{script_name}_clean_transcript.json"

                if not transcript_file.exists():
                    print(f"      ❌ Transcript not found")
                    return False

                with open(transcript_file, 'r', encoding='utf-8') as f:
                    transcript = json.load(f)

                # Generate mouth timeline
                print(f"      🗣️  Generating mouth movements...")
                mouth_timeline = self.mouth_generator.generate_mouth_timeline(transcript)

                # Generate XML
                print(f"      📄 Generating XML...")
                audio_filename = audio_file.name

                xml_content = self.xml_generator.generate_xml(
                    script_name=script_name,
                    pose_timeline=pose_timeline,
                    mouth_timeline=mouth_timeline,
                    audio_file=audio_filename
                )

                # Save XML
                xml_output = self.final_videos_path / f"{script_name}_lipsync.xml"
                with open(xml_output, 'w', encoding='utf-8') as f:
                    f.write(xml_content)

                print(f"      ✅ XML saved: {xml_output.name}")
                print(f"      📊 Pose segments: {len(pose_timeline)}")
                print(f"      📊 Mouth segments: {len(mouth_timeline)}")

                self.tracking.mark_processed(audio_file, "stage_d", xml_file=str(xml_output))
            else:
                print(f"      ✅ Stage D already complete")

            # Mark as fully processed
            self.tracking.mark_processed(audio_file)

            return True

        def process_all_unprocessed(self):
            """Process all unprocessed audio files."""
            print("\n🔍 Scanning for unprocessed audio files...")

            clean_files = list(self.voiceovers_path.glob("*_clean.mp3"))
            print(f"   Found {len(clean_files)} clean audio files")

            if not clean_files:
                print("   ℹ️  No clean audio files found")
                return 0

            unprocessed = [f for f in clean_files if not self.tracking.is_processed(f)]

            if not unprocessed:
                print("   ✅ All files already processed")
                return 0

            print(f"   🎯 {len(unprocessed)} files need processing:\n")

            successful = 0
            for i, audio_file in enumerate(unprocessed, 1):
                print(f"\n🎬 File {i}/{len(unprocessed)}: {audio_file.name}")

                if self.process_script(audio_file):
                    successful += 1
                    print(f"   ✅ Success")
                else:
                    print(f"   ❌ Failed")

                # Checkpoint
                print(f"\n✅ Checkpoint: Processed {i}/{len(unprocessed)} files. Safe to disconnect.")

            return successful

    # Run Phase 3
    phase3_orchestrator = Phase3Orchestrator(BASE_DIR)
    phase3_success = phase3_orchestrator.process_all_unprocessed()

print("\n" + "="*70)
print(f"🎉 PHASE 3 COMPLETE: {phase3_success} XMLs generated")
print("="*70)
print(f"📁 Output location: {BASE_DIR / 'final_videos'}")
print("="*70)

# ========== CLEANUP PHASE 3 (only if it ran) ==========
if unprocessed_phase3:
    print("\n🧹 Freeing memory (unloading Phase 3 models)...")

    try:
        del phase3_orchestrator
        del PoseTagger
        del PoseMapper
        del TimingAligner
        del PhonemeMapper
        del MouthTimingGenerator
        del DaVinciXMLGenerator
        del Phase3Orchestrator
    except:
        pass

    # Force garbage collection
    import gc
    gc.collect()

    # If transformers was loaded, try to clear cache
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except:
        pass

    print("   ✅ Phase 3 cleaned up")
    print("   💾 ~1-2GB RAM freed\n")
    time.sleep(5)

# ══════════════════════════════════════════════════════════════════════════════
# 🎬 PHASE 4: VIDEO PLANNING WITH GROQ + MCP + LANGSEARCH
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("🎬 PHASE 4: VIDEO PLANNING WITH GROQ")
print("="*70)

# Quick check: Skip if all files already planned
script_files = list((BASE_DIR / "processed_scripts").glob("*.txt"))
print(f"📁 Found {len(script_files)} processed script files")

if len(script_files) == 0:
    print("ℹ️  No processed scripts to plan\n")
    phase4_success = 0
    unprocessed_phase4 = []
else:
    phase4_tracking_temp = TrackingSystem("phase4_video_planning", BASE_DIR)
    unprocessed_phase4 = [f for f in script_files if not phase4_tracking_temp.is_processed(f)]

    if not unprocessed_phase4:
        print("✅ Phase 4 already complete - skipping\n")
        phase4_success = 0
    else:
        print(f"🔄 {len(unprocessed_phase4)} files need video planning\n")

        # ========== INSTALL PHASE 4 DEPENDENCIES ==========
        print("📦 Installing Groq SDK and LangSearch dependencies...")
        !pip install -q groq requests
        print("   ✅ Dependencies installed\n")

        import requests
        from groq import Groq
        import re
        import json
        import time

        # Groq API Configuration
        GROQ_API_KEY = "" #input api key here or setup with .env
        groq_client = Groq(api_key=GROQ_API_KEY)
        GROQ_MODEL = "llama-3.3-70b-versatile"

        print(f"✅ Groq configured")
        print(f"   Model: {GROQ_MODEL}\n")

        # ══════════════════════════════════════════════════════════════════════════════
        # 📚 MCP TRAINING DATA LOADER
        # ══════════════════════════════════════════════════════════════════════════════

        class MCPTrainer:
            """Load and format MCP training examples."""

            def __init__(self, base_dir):
                self.base_dir = Path(base_dir)
                self.mcp_dir = self.base_dir / "Video_planning_trainer"

                # Load rules
                self.rules = self.load_rules()

                # Load example video plans
                self.examples = self.load_examples()

            def load_rules(self):
                """Load rules from MCP."""
                rules_file = self.mcp_dir / "rules.txt"

                if not rules_file.exists():
                    print(f"   ⚠️  MCP rules.txt not found at {rules_file}")
                    return ""

                with open(rules_file, 'r', encoding='utf-8') as f:
                    return f.read().strip()

            def load_examples(self):
                """Load example video plans from MCP."""
                examples_dir = self.mcp_dir / "examples"

                if not examples_dir.exists():
                    print(f"   ⚠️  MCP examples directory not found at {examples_dir}")
                    return []

                examples = []
                for example_file in examples_dir.glob("*_video_plan.txt"):
                    with open(example_file, 'r', encoding='utf-8') as f:
                        content = f.read().strip()
                        examples.append({
                            'name': example_file.stem,
                            'content': content
                        })

                print(f"   ✅ Loaded {len(examples)} MCP training examples")
                return examples

            def format_training_context(self):
                """Format MCP data for Groq prompt."""
                context = f"""# YOUR TRAINING - VIDEO PLANNING MCP

## RULES YOU MUST FOLLOW:
{self.rules}

## EXAMPLE VIDEO PLANS (YOUR STYLE):
"""

                for i, example in enumerate(self.examples, 1):
                    context += f"\n### Example {i}: {example['name']}\n"
                    context += f"{example['content']}\n"

                return context

        # Load MCP
        mcp_trainer = MCPTrainer(BASE_DIR)
        mcp_context = mcp_trainer.format_training_context()

        print(f"✅ MCP training data loaded\n")


        # ══════════════════════════════════════════════════════════════════════════════
        # SCORE VERIFICATION SYSTEM
        # ══════════════════════════════════════════════════════════════════════════════

        def verify_match_score_with_gemini(teams, date, script_content):
            """
            Verify match score using Gemini with Google Search.
            Also finds the ACTUAL date if Stage 3 got it wrong.
            """
            print(f"\n         🔍 Verifying: {teams} on {date}")

            prompt = f"""You are verifying a football match score and date.

        USER'S SCRIPT CONTEXT (for understanding what match they're talking about):
        {script_content[:1500]}

        VERIFICATION TASK:
        Teams mentioned: {teams}
        Date suggested by AI: {date}

        YOUR TASK:
        1. Search Google for: "{teams} football match recent"
        2. Find the ACTUAL match that the script is talking about
        3. The date "{date}" might be WRONG - find the real date
        4. Return the verified score AND correct date

        IMPORTANT:
        - The script might reference a historical match (2024, 2023, etc.)
        - The script might reference a current/recent match (late 2025)
        - Find the MOST RECENT or MOST RELEVANT match between these teams
        - If multiple matches exist, pick the one that matches the script context (check player names, scores mentioned)

        STRATEGY:
        - Read the script context above carefully
        - Look for player names, events, or details mentioned
        - Search for matches between these teams that match the script's description
        - If the script mentions specific players/events, use that to find the right match
        - If multiple matches exist, pick the most recent or most relevant one

        Return JSON:
        {{
          "status": "found/not_found",
          "score": "2-1" or null,
          "correct_date": "December 15 2024" or null,
          "source": "BBC Sport / ESPN / Official source",
          "reasoning": "Found Nigeria vs Tanzania AFCON qualifier on [date] where Osimhen, Ajayi, and Lookman all featured. Final score 2-1."
        }}

        Search Google and find the REAL match:"""

            try:
                response = GEMINI_MODEL.generate_content(prompt)

                import json
                response_text = response.text.strip()

                # DEBUG
                print(f"            🐛 DEBUG: Gemini response:")
                print(f"            {response_text[:500]}")

                # Remove markdown fences
                if response_text.startswith('```'):
                    response_text = response_text.split('\n', 1)[1] if '\n' in response_text else response_text[3:]
                    if response_text.endswith('```'):
                        response_text = response_text.rsplit('```', 1)[0]
                    response_text = response_text.strip()

                # Parse JSON
                result = json.loads(response_text)

                status = result.get('status', 'unknown')
                score = result.get('score')
                correct_date = result.get('correct_date')
                reasoning = result.get('reasoning', 'No reasoning provided')

                if status == 'found' and score:
                    print(f"            ✅ VERIFIED: {score}")
                    if correct_date and correct_date != date:
                        print(f"            📅 Corrected date: {date} → {correct_date}")
                    print(f"            📊 Source: {result.get('source', 'N/A')}")

                    return (score, correct_date or date, "high", reasoning)

                else:
                    print(f"            ℹ️  Match not found or no score available")
                    return (None, None, "not_found", reasoning)

            except Exception as e:
                print(f"            ❌ Gemini error: {e}")
                return (None, None, "failed", str(e))


        def extract_matches_from_plan(video_plan_lines):
            """Extract unique (teams, date) from PNG sequences."""
            import re

            matches = []
            seen = set()

            for line in video_plan_lines:
                if 'PNG sequence' in line:
                    # Extract teams pattern: "TeamA vs TeamB"
                    teams_match = re.search(r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+vs\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)', line)

                    # Extract date pattern: "Month DD YYYY"
                    date_match = re.search(r'(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}\s+\d{4}', line)

                    if teams_match and date_match:
                        teams = f"{teams_match.group(1)} vs {teams_match.group(2)}"
                        date = date_match.group(0)

                        key = f"{teams}_{date}"
                        if key not in seen:
                            matches.append({"teams": teams, "date": date, "key": key})
                            seen.add(key)

            return matches


        def apply_verified_scores(video_plan_lines, verified_scores):
            """Replace scores and dates in PNG sequences with verified ones."""
            import re

            corrected_lines = []
            corrections_made = 0

            for line in video_plan_lines:
                if 'PNG sequence' not in line:
                    corrected_lines.append(line)
                    continue

                # Extract teams and date from line
                teams_match = re.search(r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+vs\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)', line)
                date_match = re.search(r'(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}\s+\d{4}', line)

                if teams_match and date_match:
                    teams = f"{teams_match.group(1)} vs {teams_match.group(2)}"
                    date = date_match.group(0)
                    key = f"{teams}_{date}"

                    # Try exact match first
                    score_info = verified_scores.get(key)

                    # If not found, try matching just by teams (ignoring date)
                    if not score_info or not score_info.get('score'):
                        for verified_key, verified_data in verified_scores.items():
                            verified_teams = verified_key.split('_')[0]
                            # Check if teams match (handle both "A vs B" and "B vs A")
                            if verified_teams == teams or verified_teams == f"{teams_match.group(2)} vs {teams_match.group(1)}":
                                if verified_data.get('score'):
                                    score_info = verified_data
                                    print(f"         🔄 Using verified data from {verified_key} for {key}")
                                    break

                    if score_info and score_info.get('score'):
                        verified_score = score_info['score']
                        correct_date = score_info.get('correct_date')

                        # Extract current score and date
                        current_score_match = re.search(r'(\d+)-(\d+)', line)
                        current_score = f"{current_score_match.group(1)}-{current_score_match.group(2)}" if current_score_match else None

                        current_date_match = re.search(r'(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}\s+\d{4}', line)
                        current_date = current_date_match.group(0) if current_date_match else None

                        # Replace score
                        if current_score != verified_score:
                            line = re.sub(r'\d+-\d+', verified_score, line)
                            print(f"         🔧 Corrected score: {current_score} → {verified_score}")
                            corrections_made += 1

                        # Replace date if Gemini found the correct one
                        if correct_date and current_date and correct_date != current_date:
                            line = re.sub(
                                r'(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}\s+\d{4}',
                                correct_date,
                                line
                            )
                            print(f"         📅 Corrected date: {current_date} → {correct_date}")
                            corrections_made += 1

                        corrected_lines.append(line)
                    else:
                        # No verified score
                        if score_info and score_info.get('confidence') == 'not_found':
                            print(f"         ℹ️  Skipped {key}: Match doesn't exist")

                        corrected_lines.append(line)
                else:
                    corrected_lines.append(line)

            print(f"         ✅ Made {corrections_made} score corrections")
            return corrected_lines


        # ══════════════════════════════════════════════════════════════════════════════
        # 🎬 VIDEO PLANNER CLASS
        # ══════════════════════════════════════════════════════════════════════════════

        class VideoPlanner:
            """3-Stage video planning with Pass 2 enhancement."""

            def __init__(self, base_dir, mcp_context):
                self.base_dir = Path(base_dir)
                self.processed_scripts_path = self.base_dir / "processed_scripts"
                self.pose_data_path = self.base_dir / "pose_data"
                self.video_plans_path = self.base_dir / "video_plans"

                self.tracking = TrackingSystem("phase4_video_planning", self.base_dir)
                self.stage3_cache_path = self.base_dir / "video_plans_cache"
                self.stage3_cache_path.mkdir(parents=True, exist_ok=True)

                # MCP training context
                self.mcp_context = mcp_context

                # Optimization settings
                self.min_duration = 2.0
                self.max_duration = 6.0
                self.target_min = 4.0

            def parse_mapping_timestamp(self, time_str):
                """Parse MM:SS.mmm format to seconds."""
                try:
                    if ':' in time_str:
                        parts = time_str.split(':')
                        minutes = int(parts[0])
                        seconds = float(parts[1])
                        return minutes * 60 + seconds
                    else:
                        return float(time_str)
                except:
                    return 0.0

            def load_mapping_data(self, script_name):
                """Load mapping file data."""
                mapping_file = self.pose_data_path / f"{script_name}_mapping.txt"

                if not mapping_file.exists():
                    return None

                with open(mapping_file, 'r', encoding='utf-8') as f:
                    content = f.read()

                pattern = r'\[(\d{2}:\d{2}\.\d{3})\s*-\s*(\d{2}:\d{2}\.\d{3})\]\s*\[([^\]]+)\]:\s*(.+?)(?=\n\[|\Z)'
                matches = re.findall(pattern, content, re.DOTALL | re.MULTILINE)

                segments = []
                for i, (start_str, end_str, pose_tag, text_content) in enumerate(matches):
                    start_time = self.parse_mapping_timestamp(start_str)
                    end_time = self.parse_mapping_timestamp(end_str)
                    duration = end_time - start_time

                    segments.append({
                        'index': i,
                        'start': start_time,
                        'end': end_time,
                        'duration': duration,
                        'pose': pose_tag.strip(),
                        'text': text_content.strip()
                    })

                return segments

            def load_transcript_data(self, script_name):
                """Load transcript JSON data."""
                possible_patterns = [
                    f"{script_name}_transcript.json",
                    f"{script_name}_clean_transcript.json",
                ]

                for pattern in possible_patterns:
                    test_file = self.pose_data_path / pattern
                    if test_file.exists():
                        with open(test_file, 'r') as f:
                            return json.load(f)

                return None

            def get_words_in_range(self, transcript_data, start_time, end_time):
                """Get words that start within time range."""
                words_in_range = []

                for word_data in transcript_data:
                    word_start = float(word_data['start_time'])
                    if start_time <= word_start < end_time:
                        words_in_range.append(word_data)

                return words_in_range

            def merge_segments(self, transcript_data, segments_to_merge):
                """Merge multiple segments."""
                if len(segments_to_merge) < 2:
                    return segments_to_merge[0] if segments_to_merge else None

                first_segment = segments_to_merge[0]
                last_segment = segments_to_merge[-1]

                merged_start = first_segment['start']
                merged_end = last_segment['end']
                merged_duration = merged_end - merged_start

                words_in_range = self.get_words_in_range(transcript_data, merged_start, merged_end)

                if not words_in_range:
                    return None

                words_in_range.sort(key=lambda w: float(w['start_time']))
                combined_text = " ".join(w["word"] for w in words_in_range)

                merged_segment = {
                    "index": f"{first_segment['index']}-{last_segment['index']}",
                    "start": merged_start,
                    "end": merged_end,
                    "duration": merged_duration,
                    "pose": first_segment["pose"],
                    "text": combined_text,
                    "action": f"MERGED_{len(segments_to_merge)}_SEGMENTS"
                }

                return merged_segment

            def split_segment(self, transcript_data, segment):
                """Split a long segment at word boundary."""
                words = self.get_words_in_range(transcript_data, segment['start'], segment['end'])
                words.sort(key=lambda w: float(w['start_time']))

                if len(words) < 2:
                    return None

                # Find midpoint
                mid_idx = len(words) // 2
                split_word = words[mid_idx]
                split_time = float(split_word['start_time'])

                part1_words = words[:mid_idx]
                part1_text = " ".join(w['word'] for w in part1_words)

                part2_words = words[mid_idx:]
                part2_text = " ".join(w['word'] for w in part2_words)

                part1 = {
                    "index": f"{segment['index']}.1",
                    "start": segment['start'],
                    "end": split_time,
                    "duration": split_time - segment['start'],
                    "pose": segment["pose"],
                    "text": part1_text,
                    "action": "SPLIT_PART_1"
                }

                part2 = {
                    "index": f"{segment['index']}.2",
                    "start": split_time,
                    "end": segment['end'],
                    "duration": segment['end'] - split_time,
                    "pose": segment["pose"],
                    "text": part2_text,
                    "action": "SPLIT_PART_2"
                }

                return [part1, part2]

            def optimize_segments(self, mapping_data, transcript_data):
                """Optimize segment timing using transcript boundaries."""
                if not mapping_data or not transcript_data:
                    return None

                print("      🔧 Optimizing segment timing...")

                # Phase 1: Split long segments
                segments_after_splits = []

                for segment in mapping_data:
                    if segment['duration'] > self.max_duration:
                        split_parts = self.split_segment(transcript_data, segment)
                        if split_parts:
                            segments_after_splits.extend(split_parts)
                        else:
                            segment['action'] = 'SPLIT_FAILED'
                            segments_after_splits.append(segment)
                    else:
                        segments_after_splits.append(segment)

                # Phase 2: Merge short segments
                final_segments = []
                i = 0

                while i < len(segments_after_splits):
                    current_segment = segments_after_splits[i]

                    if current_segment['duration'] < self.target_min:
                        merge_candidates = [current_segment]
                        total_duration = current_segment['duration']
                        j = i + 1

                        while j < len(segments_after_splits) and total_duration < self.target_min:
                            next_segment = segments_after_splits[j]
                            potential_duration = next_segment['end'] - current_segment['start']

                            if potential_duration <= self.max_duration:
                                merge_candidates.append(next_segment)
                                total_duration = potential_duration
                                j += 1
                            else:
                                break

                        if len(merge_candidates) > 1:
                            merged = self.merge_segments(transcript_data, merge_candidates)
                            if merged:
                                final_segments.append(merged)
                                i = j
                                continue

                        current_segment['action'] = 'TOO_SHORT_CANNOT_MERGE'
                        final_segments.append(current_segment)
                        i += 1
                    else:
                        if 'action' not in current_segment:
                            current_segment['action'] = 'UNCHANGED'
                        final_segments.append(current_segment)
                        i += 1

                print(f"      ✅ Optimized to {len(final_segments)} segments")

                return final_segments

            def format_time_seconds(self, seconds):
                """Format seconds with 2 decimal places."""
                return f"{seconds:.2f}"

            def generate_timing_document(self, optimized_segments):
                """Generate timing document."""
                output_lines = []

                for segment in optimized_segments:
                    start_formatted = self.format_time_seconds(segment['start'])
                    end_formatted = self.format_time_seconds(segment['end'])
                    line = f"{start_formatted}-{end_formatted}: {segment['text']}"
                    output_lines.append(line)

                return '\n'.join(output_lines)

            # ==================== STAGE 1: ASSET TYPE CLASSIFICATION ====================

            def stage1_classify_assets(self, timing_document, script_content):
                """Stage 1: Groq classifies each segment into asset types."""
                print(f"      🎯 STAGE 1: Classifying asset types...")

                prompt = f"""# STAGE 1: ASSET TYPE CLASSIFICATION

You are analyzing a football commentary script to determine what TYPE of visual asset each segment needs.

## ASSET TYPE PRIORITY RULES:

### **1. Check for PNG SEQUENCE first:**
Use when transcript describes:
- Specific match actions (goals, tackles, saves, assists, shots)
- Player movements on the pitch
- Concrete events you could see on a highlight reel
- ANY mention of scoring, assisting, shooting, saving

Keywords: "scores", "goal", "assist", "shot", "tackle", "save", "header", "penalty"

Examples:
- "scores in 21 minutes" → PNG sequence
- "Malogusto also gets an assist and a goal" → PNG sequence
- "shot on target" → PNG sequence

### **2. Check for REAL IMAGE second:**
Use when transcript describes:
- **Named players** with emotional/physical context
- **Specific statistics or data** (possession %, shots, passes)
- **Named people** (managers, coaches, referees)
- **Concrete visual scenes** with specific identifiers
- **Achievements/milestones** (top four, records, titles)

Keywords: Player names, coach names, "possession", "stats", "%", "top four", "league table"

Examples:
- "Cole Palmer, their usual messiah" → Real image (named player)
- "Fofana gets a yellow card" → Real image (named player, specific moment)
- "57.7% possession and 17 shots" → Real image (statistics)
- "Chelsea moves back into the top four" → Real image (achievement milestone)

### **3. Use VIDEO as fallback:**
Use when transcript is:
- Generic sarcasm/mockery WITHOUT named subjects
- Viewer reactions/opinions
- Comedy punchlines without specific context
- Exaggeration or metaphors
- Generic emotional reactions

Examples:
- "Blud, Chelsea just scraped a win" → Video (generic sarcasm)
- "right? Everton's shooting was laughable" → Video (viewer reaction)
- "I'm not impressed, man" → Video (opinion without names)
- "Shot more blanks than a vasectomy clinic" → Video (exaggeration)

## DECISION TREE:

1. Does it mention a MATCH ACTION (goal, shot, save, tackle)? → **PNG SEQUENCE**
2. Does it mention a NAMED PLAYER/PERSON or SPECIFIC STATS? → **REAL IMAGE**
3. Is it generic sarcasm/reaction/opinion? → **VIDEO**

## TARGET DISTRIBUTION (for balance):
- PNG Sequences: 20-30%
- Real Images: 20-30%
- Videos: 50-60%

## YOUR TASK:

Analyze each segment using the decision tree above and output ONLY valid JSON:
```json
{{
  "segments": [
    {{
      "timestamp": "0.00-4.98",
      "transcript": "transcript text here",
      "asset_type": "Video",
      "reasoning": "Generic sarcasm without named subject"
    }},
    {{
      "timestamp": "4.98-9.24",
      "transcript": "Cole Palmer, their usual messiah",
      "asset_type": "Real image",
      "reasoning": "Named player with context"
    }},
    {{
      "timestamp": "9.24-13.50",
      "transcript": "scores in 21 minutes",
      "asset_type": "PNG sequence",
      "reasoning": "Describes specific goal action"
    }}
  ]
}}
```

## SCRIPT CONTEXT:
{script_content[:500]}

## TIMING DOCUMENT TO CLASSIFY:
{timing_document}

Output valid JSON only (no markdown, no explanation):"""

                # Try Groq first
                response = self.call_groq_with_retry(prompt, max_retries=2)

                # Fallback to Gemini if Groq fails
                if not response:
                    print(f"         ⚠️  Groq failed, trying Gemini...")
                    response = call_gemini_with_retry(prompt, max_retries=2)

                if not response:
                    print(f"         ❌ Stage 1 failed")
                    return None

                # Parse JSON
                try:
                    clean_response = response.strip()
                    if clean_response.startswith('```'):
                        clean_response = '\n'.join(clean_response.split('\n')[1:-1])

                    classification = json.loads(clean_response)

                    print(f"         ✅ Classified {len(classification.get('segments', []))} segments")

                    # Debug: Show asset type distribution
                    asset_counts = {}
                    for seg in classification.get('segments', []):
                        asset_type = seg.get('asset_type', 'Unknown')
                        asset_counts[asset_type] = asset_counts.get(asset_type, 0) + 1

                    print(f"         📊 Asset distribution: {asset_counts}")

                    return classification

                except json.JSONDecodeError as e:
                    print(f"         ❌ JSON parse error: {e}")
                    print(f"         Response preview: {response[:200]}")
                    return None

            # ==================== STAGE 2: SEARCH QUERY GENERATION ====================

            def stage2_generate_searches(self, classification, script_content):
                """Stage 2: Generate specific search queries based on asset types."""
                print(f"      🔍 STAGE 2: Generating search queries...")

                # Get current date dynamically
                from datetime import datetime
                current_date = datetime.now().strftime("%B %d, %Y")
                current_year = datetime.now().year
                current_month = datetime.now().strftime("%B")

                prompt = f"""# STAGE 2: SEARCH QUERY GENERATION

**TODAY IS {current_date}. CURRENT YEAR: {current_year}.**
You have classified segments into asset types. Now generate SPECIFIC search queries for segments that need them.

## SEARCH RULES BY ASSET TYPE:

### **PNG SEQUENCE** - REQUIRES SEARCH:
CRITICAL: Always include BOTH TEAMS in every query for match-specific data

✓ CORRECT queries:
- "Cole Palmer goal Chelsea vs Everton December 2025"
- "Chelsea vs Everton December 2025 final score match report"
- "Malo Gusto assist Chelsea vs Everton December 22 2025"

✗ WRONG queries (too generic, returns season stats):
- "Cole Palmer goals Chelsea 2025"
- "Malo Gusto assists 2025"

### **REAL IMAGE** - REQUIRES SEARCH:
Include match context when relevant:
✓ "Cole Palmer celebrating Chelsea vs Everton December 2025"
✓ "Wesley Fofana yellow card Chelsea vs Everton 2025"
✓ "Everton players dejected Chelsea vs Everton December 2025"

### **VIDEO** (Memes) - NO SEARCH NEEDED:
These are generic memes, no search required

## MANDATORY SEARCHES FOR EVERY MATCH SCRIPT:
You MUST generate these searches:
1. "[Team A] vs [Team B] [Month Year] final score match report"
2. "[Team A] vs [Team B] [Month Year] lineups starting XI"
3. "[Team A] vs [Team B] [Month Year] goals scorers highlights"

## SCRIPT TYPE DETECTION:
Analyze if this is:
- **PREVIEW**: Future match (use recent form searches)
- **REVIEW**: Past match (search for that specific match)

Full script context:
{script_content[:1000]}

## CLASSIFIED SEGMENTS:
{json.dumps(classification, indent=2)}

## OUTPUT FORMAT (valid JSON only):
```json
{{
  "script_type": "PREVIEW or REVIEW",
  "primary_teams": ["Team A", "Team B"],
  "searches": [
    {{
      "query": "Chelsea vs Everton December 2025 final score match report",
      "purpose": "Get match result, date, scorers for all segments"
    }},
    {{
      "query": "Cole Palmer goal Chelsea vs Everton December 2025",
      "purpose": "PNG sequence asset for segment X"
    }}
  ]
}}
```

Output valid JSON only:"""

                # Try Groq first
                response = self.call_groq_with_retry(prompt, max_retries=2)

                # Fallback to Gemini if Groq fails
                if not response:
                    print(f"         ⚠️  Groq failed, trying Gemini...")
                    response = call_gemini_with_retry(prompt, max_retries=2)

                if not response:
                    print(f"         ❌ Stage 2 failed")
                    return None

                # Parse JSON
                try:
                    clean_response = response.strip()
                    if clean_response.startswith('```'):
                        clean_response = '\n'.join(clean_response.split('\n')[1:-1])

                    search_plan = json.loads(clean_response)

                    script_type = search_plan.get('script_type', 'UNKNOWN')
                    num_searches = len(search_plan.get('searches', []))

                    print(f"         ✅ Script type: {script_type}")
                    print(f"         ✅ Generated {num_searches} search queries")

                    return search_plan

                except json.JSONDecodeError as e:
                    print(f"         ❌ JSON parse error: {e}")
                    return None

            # ==================== STAGE 2.5: EXECUTE SEARCHES ====================

            def execute_langsearch_queries(self, search_plan):
                """Execute all LangSearch queries using corrected API."""
                print(f"      🌐 Executing LangSearch queries...")

                searches = search_plan.get('searches', [])

                if not searches:
                    print(f"         ℹ️  No searches needed")
                    return {}

                results = {}

                for i, search_item in enumerate(searches, 1):
                    query = search_item.get('query', '')
                    purpose = search_item.get('purpose', '')

                    if not query:
                        continue

                    print(f"         🔍 Search {i}/{len(searches)}: {query}")

                    # Use corrected LangSearch API
                    search_result = langsearch_query(query, count=5)

                    if search_result.get('found'):
                        # Extract snippets and summaries
                        snippets = []
                        for page in search_result.get('results', []):
                            # Prefer summary, fallback to snippet
                            text = page.get('summary') or page.get('snippet', '')
                            if text:
                                snippets.append(f"{page.get('title', 'Untitled')}: {text}")

                        if snippets:
                            combined = " | ".join(snippets)
                            results[query] = combined
                            print(f"            ✅ Found {len(snippets)} results")
                        else:
                            results[query] = "No content in results"
                            print(f"            ⚠️  Empty results")
                    else:
                        error = search_result.get('error', 'Unknown error')
                        results[query] = f"Search failed: {error}"
                        print(f"            ❌ Failed: {error}")

                    time.sleep(1.5)  # Rate limiting

                print(f"         ✅ Completed {len(results)} searches")

                # Check if all failed
                failed_count = sum(1 for v in results.values() if 'failed' in v.lower() or 'error' in v.lower())
                if failed_count == len(results):
                    print(f"         ⚠️  ALL SEARCHES FAILED - Check API key/quota")

                return results

            # ==================== STAGE 3: FINAL VIDEO PLAN ====================

            def stage3_generate_final_plan(self, classification, search_results, timing_document):
                """Stage 3: Generate final video plan one segment at a time with smart context filtering."""
                print(f"      📝 STAGE 3: Generating final video plan (smart context filtering)...")

                segments = classification.get('segments', [])

                if not segments:
                    print(f"         ❌ No segments to process")
                    return None

                final_plan_lines = []

                # Asset-specific MCP examples
                mcp_examples = {
                    'PNG sequence': """
**PNG SEQUENCE format:**
PNG sequence of [PLAYER NAME] [SPECIFIC ACTION] [TEAM A] vs [TEAM B] [SCORE] [FULL DATE] [transcript: exact text]

Examples from your training:
- PNG sequence of Conor Bradley tackle on Mbappe Liverpool vs Real Madrid 1-0 November 4 2025 [transcript: Bradley flattened this man in the 30th minute.]
- PNG sequence of Mac Allister header goal Liverpool vs Real Madrid 1-0 November 4 2025 [transcript: a header through Courtois arms.]
- PNG sequence of Courtois save on Van Dijk header Liverpool vs Real Madrid 1-0 November 4 2025 [transcript: Van Dyke header. Saved with his forehead.]
""",
                    'Real image': """
**REAL IMAGE format:**
Real image of [SPECIFIC SUBJECT] [DESCRIPTION] [OPTIONAL: match context] [transcript: exact text]

Examples from your training:
- Real image of Trent Alexander-Arnold in Real Madrid kit funny [transcript: Gave a whole speech in fluent Espanol and Liverpool fans said,]
- Real image of angry Liverpool fans with banners [transcript: Liverpool fans had zero chill. Negative chill.]
- Real image of vandalized Trent Alexander-Arnold mural Liverpool November 2025 [transcript: Adios el rata. That's goodbye rat for those of you who skipped Spanish class.]
""",
                    'Video': """
**VIDEO format:**
Video of [SPECIFIC MEME NAME] [transcript: exact text]

Examples from your training:
- Video of chef kiss meme [transcript: Beautiful finish. Clinical. Composed. His first La Liga goal since August.]
- Video of understanding meme [transcript: The math is mathing.]
- Video of shocked realization meme [transcript: She's closer to 30 than this man to being trusted with a rental car.]
- Video of facepalm meme [transcript: That Bro's stuck with permanent evidence of his L. Oh boy,]
- Video of No meme [transcript: The internet said no. Bro, please, just focus on the football.]

CRITICAL: Use REAL, SEARCHABLE internet meme names. Not literal descriptions like "egg boiling" or "time bomb ticking".
"""
                }

                # Process each segment individually
                for i, segment in enumerate(segments, 1):
                    timestamp = segment.get('timestamp', '')
                    transcript = segment.get('transcript', '')
                    asset_type = segment.get('asset_type', '')
                    reasoning = segment.get('reasoning', '')

                    print(f"         🎬 Segment {i}/{len(segments)}: {timestamp} ({asset_type})")

                    # STEP 1: Filter relevant search results
                    relevant_searches = self.filter_relevant_searches(transcript, search_results)

                    # STEP 2: Get asset-specific MCP
                    asset_mcp = mcp_examples.get(asset_type, mcp_examples['Real image'])

                    # STEP 3: Create focused prompt
                    prompt = f"""{asset_mcp}

## RELEVANT SEARCH RESULTS:
{json.dumps(relevant_searches, indent=2) if relevant_searches else "No specific search data needed for this segment."}

## SEGMENT TO PROCESS:
- Timestamp: {timestamp}
- Transcript: "{transcript}"
- Asset Type: {asset_type}

## FORMAT RULES:
CRITICAL FORMAT RULES:
1. EVERY line MUST start with timestamp: XX.XX-XX.XX:
2. Asset type immediately after timestamp
3. Description follows asset type
4. Transcript ALWAYS in square brackets at end: [transcript: exact words]

ASSET FORMATS:
- PNG sequence: "XX.XX-XX.XX: PNG sequence of [PLAYER NAME] [ACTION] [TEAM A] vs [TEAM B] [SCORE] [DATE] [transcript: ...]"
- Real image: "XX.XX-XX.XX: Real image of [SPECIFIC SUBJECT] [DESCRIPTION] [CONTEXT] [transcript: ...]"
- Video: "XX.XX-XX.XX: Video of [EXACT MEME NAME] [transcript: ...]"

REAL IMAGE RULES:
- Use SPECIFIC names (players, coaches, managers, referees)
- Include match context when relevant (team names, dates in "December 6 2025" format)
- Add descriptive details (facial expressions, body language, emotions)
- Examples:
  ✓ "Real image of Mikel Arteta looking frustrated on touchline Arsenal vs Villa December 6 2025"
  ✓ "Real image of David Raya diving save Villa Park December 6 2025"
  ✗ "Real image of a goalkeeper making a save"
  ✗ "Real image of coach on sideline"

VIDEO RULES:
- Use REAL, SEARCHABLE meme names from internet culture
- Examples: "chef kiss meme", "shocked Pikachu meme", "disappointed cricket fan meme"
- DO NOT use literal descriptions

PNG SEQUENCE RULES:
- MUST include: Player name, action, both teams, score, date
- Format date as: "December 6 2025" or "Dec 6 2025"
- Example: "PNG sequence of Ollie Watkins header goal Arsenal vs Aston Villa 2-1 December 6 2025"

DATE FORMAT:
- Always use: [Month name] [Day] [Year]
- ✓ December 6 2025
- ✓ Dec 6 2025
- ✗ 6/12/2025
- ✗ 2025-12-06

TRANSCRIPT PRESERVATION:
- NEVER modify transcript text
- Keep exact punctuation and capitalization
- Always wrap in square brackets: [transcript: ...]


Generate ONE line following your training examples:"""

                    # Try Groq first
                    response = self.call_groq_with_retry(prompt, max_retries=2)

                    # Fallback to Gemini
                    if not response:
                        print(f"            ⚠️  Groq failed, trying Gemini...")
                        response = call_gemini_with_retry(prompt, max_retries=2)

                    if response:
                        # Extract and clean the line
                        import re
                        lines = [l.strip() for l in response.split('\n') if l.strip()]

                        plan_line = None
                        for line in lines:
                            if re.match(r'^\d+\.\d{2}-\d+\.\d{2}:', line):
                                plan_line = line
                                break

                        # Fallback: use first line
                        if not plan_line and lines:
                            plan_line = lines[0]

                        if plan_line:
                            # Fix formatting
                            plan_line = re.sub(r'\(transcript:', '[transcript:', plan_line)
                            plan_line = re.sub(r'\)\s*$', ']', plan_line)
                            plan_line = re.sub(r'transcript:\s*"([^"]+)"', r'[transcript: \1]', plan_line)

                            # Ensure [transcript: ...] exists
                            if '[transcript:' not in plan_line:
                                if plan_line.endswith(']'):
                                    plan_line = plan_line[:-1] + f" [transcript: {transcript}]"
                                else:
                                    plan_line = plan_line + f" [transcript: {transcript}]"

                            # ========== ADD THIS VALIDATION BLOCK ==========
                            # Validate format based on asset type
                            is_valid = True

                            if 'PNG sequence' in plan_line:
                                # Check for "unspecified"
                                if 'unspecified' in plan_line.lower():
                                    print(f"            ⚠️  PNG sequence has 'unspecified', will be fixed in Pass 2")
                                    # Don't reject, Pass 2 will fix it

                            elif 'Real image' in plan_line:
                                # Check for generic terms
                                generic_terms = [
                                    'a player', 'a coach', 'a fan', 'a goalkeeper', 'a defender',
                                    'a midfielder', 'a striker', 'a manager', 'football player',
                                    'soccer player', 'football players'
                                ]
                                if any(term in plan_line.lower() for term in generic_terms):
                                    print(f"            ⚠️  Real image has generic terms, will be fixed in Pass 2")
                                    # Don't reject, Pass 2 will fix it
                            # ========== END VALIDATION BLOCK ==========

                            final_plan_lines.append(plan_line)
                            print(f"            ✅ Generated")
                        else:
                            print(f"            ⚠️  Using fallback")
                            final_plan_lines.append(f"{timestamp}: Real image placeholder [transcript: {transcript}]")
                    else:
                        print(f"            ❌ All APIs failed, using fallback")
                        final_plan_lines.append(f"{timestamp}: Real image placeholder [transcript: {transcript}]")

                    # Rate limiting delay
                    time.sleep(0.8)

                if not final_plan_lines:
                    print(f"         ❌ No plan lines generated")
                    return None

                final_plan = '\n'.join(final_plan_lines)

                print(f"         ✅ Final plan complete: {len(final_plan_lines)} lines")

                return final_plan

            def filter_relevant_searches(self, transcript, search_results):
                """Filter search results to only include relevant ones for this segment."""
                import re

                # Extract keywords from transcript (nouns, proper nouns)
                transcript_lower = transcript.lower()
                transcript_words = set(re.findall(r'\b[a-z]{3,}\b', transcript_lower))

                relevant = {}

                for query, result in search_results.items():
                    query_lower = query.lower()
                    query_words = set(re.findall(r'\b[a-z]{3,}\b', query_lower))

                    # Check for overlap between transcript and query
                    overlap = transcript_words & query_words

                    # If 2+ words match, it's relevant
                    if len(overlap) >= 2:
                        relevant[query] = result
                    # Also include if specific player/team names match
                    elif any(word in transcript_lower for word in ['arsenal', 'villa', 'buendia', 'yamal', 'barcelona']):
                        if any(word in query_lower for word in ['arsenal', 'villa', 'buendia', 'yamal', 'barcelona']):
                            relevant[query] = result

                # Limit to max 3 most relevant searches to save tokens
                if len(relevant) > 3:
                    # Keep first 3
                    relevant = dict(list(relevant.items())[:3])

                return relevant

            def call_groq_with_retry(self, prompt, max_retries=3):
                """Call Groq API with retry logic."""

                for attempt in range(1, max_retries + 1):
                    try:
                        print(f"         🤖 Groq API call (attempt {attempt}/{max_retries})...")

                        response = groq_client.chat.completions.create(
                            model=GROQ_MODEL,
                            messages=[
                                {
                                    "role": "system",
                                    "content": "You are a video planning expert. You output valid JSON when requested, and follow training examples exactly for video plans."
                                },
                                {
                                    "role": "user",
                                    "content": prompt
                                }
                            ],
                            temperature=0.7,
                            max_tokens=4000
                        )

                        result = response.choices[0].message.content.strip()
                        print(f"         ✅ Response received ({len(result)} chars)")

                        return result

                    except Exception as e:
                        print(f"         ⚠️  Groq API error: {e}")

                        if attempt < max_retries:
                            time.sleep(2)
                        else:
                            print(f"         ❌ All attempts failed")
                            return None

                return None

            # ==================== ORCHESTRATION ====================

            def extract_match_context(self, search_plan, search_results):
                """Auto-extract match context from search plan and results."""
                import re
                from datetime import datetime

                # Extract teams
                teams = search_plan.get('primary_teams', [])
                teams_str = f"{teams[0]} vs {teams[1]}" if len(teams) >= 2 else "Unknown teams"

                # Extract date from search results
                all_results_text = " ".join(str(v) for v in search_results.values())
                date_patterns = [
                    r'(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2}),?\s+(\d{4})',
                    r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+(\d{1,2}),?\s+(\d{4})'
                ]

                date_str = None
                for pattern in date_patterns:
                    match = re.search(pattern, all_results_text)
                    if match:
                        month, day, year = match.groups()
                        month_map = {'Jan':'January','Feb':'February','Mar':'March','Apr':'April',
                                    'May':'May','Jun':'June','Jul':'July','Aug':'August',
                                    'Sep':'September','Oct':'October','Nov':'November','Dec':'December'}
                        month = month_map.get(month, month)
                        date_str = f"{month} {day} {year}"
                        break

                if not date_str:
                    date_str = datetime.now().strftime("%B %d %Y")

                return {'teams': teams_str, 'date': date_str, 'venue': 'N/A'}

            def extract_main_subjects(self, script_content):
                """Extract main players/teams mentioned in script."""
                import re
                from collections import Counter

                # Find capitalized names (2-3 words)
                names = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b', script_content[:2000])

                # Filter for player names (not common words)
                likely_names = []
                exclude_words = ['Man United', 'Manchester United', 'Premier League', 'Saudi Arabia', 'Real Madrid']

                for name in names:
                    word_count = len(name.split())
                    if 2 <= word_count <= 3 and name not in exclude_words:
                        likely_names.append(name)

                # Get top 5 most mentioned
                name_counts = Counter(likely_names)
                main_subjects = [name for name, count in name_counts.most_common(5)]

                # Always include team names from context
                if 'Man United' in script_content or 'Manchester United' in script_content:
                    main_subjects.append('Man United')
                if 'Al-Hilal' in script_content or 'Al Hilal' in script_content:
                    main_subjects.append('Al-Hilal')

                return main_subjects

            def enhance_real_images_pass2(self, video_plan_lines, match_context, search_results, output_path, main_subjects=None):
                """Pass 2: Enhance Real Image lines for specificity and eliminate duplicates."""

                # Extract ALL lines that need enhancement
                lines_to_enhance = []
                for idx, line in enumerate(video_plan_lines):
                    # Process Real Images and PNG sequences (Videos are usually fine)
                    if any(asset_type in line for asset_type in ['Real image', 'PNG sequence']):
                        lines_to_enhance.append((idx, line))

                if not lines_to_enhance:
                    print("      No lines need enhancement, skipping Pass 2")
                    return video_plan_lines

                print(f"\n      🎯 PASS 2: Enhancing {len(lines_to_enhance)} lines...")

                enhanced_lines = video_plan_lines.copy()
                chunk_size = 10

                for chunk_start in range(0, len(lines_to_enhance), chunk_size):
                    chunk = lines_to_enhance[chunk_start:chunk_start + chunk_size]
                    chunk_indices = [idx for idx, _ in chunk]
                    chunk_lines = [line for _, line in chunk]

                    print(f"\n         📦 Processing chunk {chunk_start//chunk_size + 1}/{(len(lines_to_enhance)-1)//chunk_size + 1}")

                    # Filter relevant searches for this chunk
                    all_chunk_text = " ".join(chunk_lines)
                    relevant_searches = self.filter_relevant_searches(all_chunk_text, search_results)

                    prompt = f"""You are enhancing video plan lines for a comedy football commentary video.

MATCH CONTEXT:
- Teams: {match_context['teams']}
- Date: {match_context['date']}

SCRIPT SUBJECTS (ONLY USE THESE):
{', '.join(main_subjects) if main_subjects else 'Not available'}

CRITICAL: NEVER mention players not in the script. If original line has wrong player, REPLACE with correct context.

## ASSET-SPECIFIC FORMAT RULES (DO NOT CHANGE THESE):

### PNG SEQUENCE FORMAT:
XX.XX-XX.XX: PNG sequence of [PLAYER NAME] [ACTION] [TEAM A] vs [TEAM B] [SCORE] [DATE] [transcript: ...]
- MUST include: Player name, specific action, both teams, score, date
- Example: "PNG sequence of Cole Palmer penalty kick Chelsea vs Everton 2-0 December 22 2025"
- FORBIDDEN: "PNG sequence of unspecified player" or "unspecified team" or "unspecified score"

### REAL IMAGE FORMAT:
XX.XX-XX.XX: Real image of [SPECIFIC SUBJECT] [DESCRIPTION] [CONTEXT] [transcript: ...]
- Use specific names (players, coaches) - NEVER "a player", "a coach", "a fan"
- Max 15 words before [transcript]
- Example: "Real image of Cole Palmer celebrating penalty Chelsea December 22 2025"
- If no name in search data, use: "Chelsea midfielder" or "Everton striker" (team + position)

### VIDEO FORMAT:
XX.XX-XX.XX: Video of [REAL MEME NAME] [transcript: ...]
- Keep format unchanged, just verify meme name is real and searchable
- Example: "Video of not impressed meme" or "Video of disappointed fan meme"

## YOUR ENHANCEMENT TASKS:

### For PNG SEQUENCES:
1. If line contains "unspecified", fill in from search data:
   - Find player name, teams, score, date
   - If not in search data, mark as "[NEEDS MORE DATA]" but keep proper format
2. Ensure both teams are present
3. Ensure date is in correct format: "December 22 2025" or "Dec 22 2025"

### For REAL IMAGES:
1. Replace ALL generic terms:
   - ❌ "a player", "a football player", "soccer player", "football players"
   - ❌ "a coach", "a manager", "the coach", "the manager"
   - ❌ "a goalkeeper", "a defender", "a midfielder", "a striker"
   - ❌ "a fan", "fans", "Chelsea fan", "Everton fan"
   - ✓ Use specific names OR "Chelsea midfielder", "Everton striker"
2. Keep descriptions SHORT (max 15 words before [transcript])
3. Make SEARCHABLE (specific names, dates, emotions)
4. Eliminate duplicates with different angles/moments

### For ALL ASSETS:
- PRESERVE timestamps: XX.XX-XX.XX:
- NEVER change [transcript: ...] text
- Keep exact punctuation in transcript

SEARCH DATA:
{json.dumps(relevant_searches, indent=2)[:3000]}

LINES TO ENHANCE:
{json.dumps(chunk_lines, indent=2)}

RULES:
- Return EXACTLY {len(chunk_lines)} lines in same order
- Follow format rules for each asset type above
- If already 8/10+ specific and follows format, keep as-is

Return JSON array of enhanced lines:
["enhanced line 1", "enhanced line 2", ...]
"""

                    try:
                        # Try Groq first
                        response = groq_client.chat.completions.create(
                            model="llama-3.3-70b-versatile",
                            messages=[{"role": "user", "content": prompt}],
                            response_format={"type": "json_object"},
                            temperature=0.3
                        )

                        response_content = response.choices[0].message.content

                        # Parse JSON - handle both array and object formats
                        enhanced_data = json.loads(response_content)

                        # If it's wrapped in an object, extract the array
                        if isinstance(enhanced_data, dict):
                            # Look for common key names
                            for key in ['lines', 'enhanced_lines', 'segments', 'results']:
                                if key in enhanced_data:
                                    enhanced_chunk = enhanced_data[key]
                                    break
                            else:
                                # If no recognized key, take the first list value
                                for v in enhanced_data.values():
                                    if isinstance(v, list):
                                        enhanced_chunk = v
                                        break
                                else:
                                    enhanced_chunk = chunk_lines
                        else:
                            enhanced_chunk = enhanced_data

                    except Exception as e:
                        print(f"            ⚠️ Groq failed: {e}, trying Gemini...")
                        try:
                            response = GEMINI_MODEL.generate_content(
                                prompt,
                                generation_config=genai.types.GenerationConfig(
                                    response_mime_type="application/json",
                                    temperature=0.3
                                )
                            )
                            enhanced_data = json.loads(response.text)

                            # Same parsing logic for Gemini
                            if isinstance(enhanced_data, dict):
                                for key in ['lines', 'enhanced_lines', 'segments', 'results']:
                                    if key in enhanced_data:
                                        enhanced_chunk = enhanced_data[key]
                                        break
                                else:
                                    for v in enhanced_data.values():
                                        if isinstance(v, list):
                                            enhanced_chunk = v
                                            break
                                    else:
                                        enhanced_chunk = chunk_lines
                            else:
                                enhanced_chunk = enhanced_data

                        except Exception as e2:
                            print(f"            ❌ Both failed: {e2}, keeping original chunk")
                            enhanced_chunk = chunk_lines

                    # Replace enhanced lines in the full list
                    for i, enhanced_line in enumerate(enhanced_chunk):
                        if i < len(chunk_indices):
                            original_idx = chunk_indices[i]
                            enhanced_lines[original_idx] = enhanced_line

                            # Show changes
                            if chunk_lines[i] != enhanced_line:
                                print(f"            ✨ Enhanced line {original_idx}:")
                                print(f"               Before: {chunk_lines[i][:100]}...")
                                print(f"               After:  {enhanced_line[:100]}...")

                    # Rate limit delay
                    time.sleep(1)

                # Save enhanced version
                with open(output_path, 'w') as f:
                    f.write('\n'.join(enhanced_lines))

                print(f"\n      ✅ Pass 2 complete! Saved to {output_path.name}")
                return enhanced_lines

            def assign_assets_3stage(self, script_name, timing_document):
                """3-stage asset assignment process."""
                print(f"      🎬 Starting 3-stage video planning...")

                # Load original script
                script_file = self.processed_scripts_path / f"{script_name}.txt"
                if not script_file.exists():
                    print(f"         ⚠️  Script not found")
                    script_content = timing_document
                else:
                    with open(script_file, 'r', encoding='utf-8') as f:
                        script_content = f.read()

                # ========== ADD THIS BLOCK ==========
                # Extract main subjects from script
                main_subjects = self.extract_main_subjects(script_content)
                print(f"         📌 Main subjects detected: {', '.join(main_subjects)}")
                # ========== END BLOCK ==========
                if not script_file.exists():
                    print(f"         ⚠️  Script not found")
                    script_content = timing_document
                else:
                    with open(script_file, 'r', encoding='utf-8') as f:
                        script_content = f.read()

                # STAGE 1: Classify asset types
                classification = self.stage1_classify_assets(timing_document, script_content)
                if not classification:
                    print(f"      ❌ Stage 1 failed - cannot continue")
                    return None

                # STAGE 2: Generate search queries
                search_plan = self.stage2_generate_searches(classification, script_content)
                if not search_plan:
                    print(f"      ❌ Stage 2 failed - cannot continue")
                    return None

                # STAGE 2.5: Execute searches
                search_results = self.execute_langsearch_queries(search_plan)

                # ========== CHECK STAGE 3 CACHE ==========
                cache_file = self.stage3_cache_path / f"{script_name}_stage3_cache.txt"

                if cache_file.exists():
                    print(f"      📦 Loading cached Stage 3 plan...")
                    with open(cache_file, 'r', encoding='utf-8') as f:
                        video_plan = f.read()
                    print(f"      ✅ Loaded from cache ({len(video_plan.split(chr(10)))} lines)")
                else:
                    # STAGE 3: Generate final plan
                    print(f"      📝 STAGE 3: Generating final video plan (smart context filtering)...")
                    video_plan = self.stage3_generate_final_plan(classification, search_results, timing_document)

                    if not video_plan:
                        print(f"      ❌ Stage 3 failed")
                        return None

                    # Save to cache
                    print(f"      💾 Saving Stage 3 to cache...")
                    with open(cache_file, 'w', encoding='utf-8') as f:
                        f.write(video_plan)

                print(f"      ✅ 3-stage planning complete")
                # ========== END CACHE CHECK ==========

                # ========== SCORE VERIFICATION LAYER ==========
                print(f"\n      🔍 SCORE VERIFICATION LAYER")
                print(f"      " + "="*50)

                # Convert to lines for processing
                video_plan_lines = video_plan.split('\n')

                # Extract unique matches from PNG sequences
                unique_matches = extract_matches_from_plan(video_plan_lines)

                if unique_matches:
                    print(f"      📊 Found {len(unique_matches)} unique match(es) to verify")

                    # Load raw script for context
                    script_file = self.processed_scripts_path / f"{script_name}.txt"
                    script_content = ""
                    if script_file.exists():
                        with open(script_file, 'r', encoding='utf-8') as f:
                            script_content = f.read()

                    verified_scores = {}

                    for match in unique_matches:
                        teams = match['teams']
                        date = match['date']
                        key = match['key']

                        # Verify score AND date with Gemini (with script context)
                        verified_score, correct_date, confidence, reasoning = verify_match_score_with_gemini(
                            teams, date, script_content
                        )

                        verified_scores[key] = {
                            'score': verified_score,
                            'correct_date': correct_date,  # Add this
                            'confidence': confidence,
                            'reasoning': reasoning
                        }

                        time.sleep(2)  # Rate limiting

                    # Apply verified scores to plan
                    print(f"\n      🔧 Applying verified scores to plan...")
                    video_plan_lines = apply_verified_scores(video_plan_lines, verified_scores)

                    # Convert back to string
                    video_plan = '\n'.join(video_plan_lines)

                    print(f"      ✅ Score verification complete")
                else:
                    print(f"      ℹ️  No PNG sequences found, skipping score verification")

                print(f"      " + "="*50)
                # ========== END VERIFICATION LAYER ==========

                # PASS 2: Enhance Real Images
                print(f"\n      🔧 Starting Pass 2: Real Image Enhancement")

                # Extract match context
                match_context = self.extract_match_context(search_plan, search_results)
                print(f"         Context: {match_context['teams']} on {match_context['date']}")

                # Convert video_plan string to list
                video_plan_lines = video_plan.split('\n')

                # Enhance
                enhanced_lines = self.enhance_real_images_pass2(
                    video_plan_lines=video_plan_lines,
                    match_context=match_context,
                    search_results=search_results,
                    output_path=self.video_plans_path / f"{script_name}_video_plan.txt",
                    main_subjects=main_subjects  # Add this parameter
                )

                # Join to string
                if isinstance(enhanced_lines, list):
                    enhanced_plan = '\n'.join(enhanced_lines)
                else:
                    enhanced_plan = enhanced_lines

                return enhanced_plan

            def process_script(self, script_path):
                """Process single script for video planning."""
                script_name = script_path.stem

                if self.tracking.is_processed(script_path):
                    print(f"   ⏭️  {script_name} already planned")
                    return True

                print(f"   📋 Planning: {script_name}")

                # Load mapping data
                mapping_data = self.load_mapping_data(script_name)
                if not mapping_data:
                    print(f"      ❌ Mapping file not found")
                    return False

                # Load transcript data
                transcript_data = self.load_transcript_data(script_name)
                if not transcript_data:
                    print(f"      ❌ Transcript file not found")
                    return False

                # Optimize segments
                optimized_segments = self.optimize_segments(mapping_data, transcript_data)
                if not optimized_segments:
                    print(f"      ❌ Segment optimization failed")
                    return False

                # Generate timing document
                timing_document = self.generate_timing_document(optimized_segments)

                # 3-STAGE PLANNING
                video_plan = self.assign_assets_3stage(script_name, timing_document)
                if not video_plan:
                    return False

                # Save video plan
                video_plan_file = self.video_plans_path / f"{script_name}_video_plan.txt"
                with open(video_plan_file, 'w', encoding='utf-8') as f:
                    f.write(video_plan)

                print(f"      ✅ Saved: {video_plan_file.name}")

                # Mark as processed
                self.tracking.mark_processed(script_path, video_plan_file=str(video_plan_file))

                return True

            def process_all_unprocessed(self):
                """Process all unprocessed scripts."""
                print("\n🔍 Scanning for scripts needing video planning...")

                script_files = list(self.processed_scripts_path.glob("*.txt"))
                print(f"   Found {len(script_files)} processed scripts")

                if not script_files:
                    print("   ℹ️  No processed scripts found")
                    return 0

                unprocessed = [f for f in script_files if not self.tracking.is_processed(f)]

                if not unprocessed:
                    print("   ✅ All scripts already have video plans")
                    return 0

                print(f"   🎯 {len(unprocessed)} scripts need planning:\n")

                successful = 0
                for i, script_path in enumerate(unprocessed, 1):
                    print(f"\n🎬 Script {i}/{len(unprocessed)}: {script_path.name}")

                    if self.process_script(script_path):
                        successful += 1
                        print(f"   ✅ Success")
                    else:
                        print(f"   ❌ Failed")

                    # Checkpoint
                    print(f"\n✅ Checkpoint: Planned {i}/{len(unprocessed)} scripts. Safe to disconnect.")

                return successful

        # Run Phase 4
        phase4_planner = VideoPlanner(BASE_DIR, mcp_context)
        phase4_success = phase4_planner.process_all_unprocessed()

print("\n" + "="*70)
print(f"🎉 PHASE 4 COMPLETE: {phase4_success} video plans created")
print("="*70)
print(f"📁 Output location: {BASE_DIR / 'video_plans'}")
print("="*70)

# ══════════════════════════════════════════════════════════════════════════════
# 🎊 FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("🎊 COMPLETE PIPELINE SUMMARY")
print("="*70)
print(f"✅ Phase 1: {phase1_success} scripts enhanced with Audio Tags")
print(f"✅ Phase 2: {phase2_success} voiceovers generated")
print(f"✅ Whisper: {whisper_success} transcripts created")
print(f"✅ Phase 3: {phase3_success} lip-sync XMLs generated")
print(f"✅ Phase 4: {phase4_success} video plans created")
print("="*70)
print(f"📁 All outputs saved to: {BASE_DIR}")
print("="*70)
print("\n🚀 PIPELINE COMPLETE! All files ready for video production.")

import google.generativeai as genai
print(genai.__version__)