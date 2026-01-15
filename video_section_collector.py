#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#
#      VIDEO SECTION
#
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


# Install required packages
!pip install -U google-generativeai opencv-python matplotlib yt-dlp ultralytics giphy-client

import os
import shutil
from google.colab import files, drive
import google.generativeai as genai
import cv2
import json
import re
import time
import threading
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np
from dataclasses import dataclass
from IPython.display import display, Javascript
import requests

# Mount Google Drive
print("📂 Mounting Google Drive...")
drive.mount('/content/drive')
print("✅ Drive mounted!\n")

# ========================================
# COOKIE FILE UPLOAD
# ========================================
print("=" * 70)
print("🍪 YOUTUBE COOKIE FILE UPLOAD")
print("=" * 70)
print("Please upload your YouTube cookies.txt file")
print("(Download from youtube.com using 'Get cookies.txt LOCALLY' extension)\n")

uploaded_cookies = files.upload()

if not uploaded_cookies:
    print("⚠️  WARNING: No cookie file uploaded!")
    print("   The script may face YouTube blocking issues.\n")
    COOKIE_FILE = None
else:
    cookie_filename = list(uploaded_cookies.keys())[0]
    COOKIE_FILE = f"/content/{cookie_filename}"
    print(f"✅ Cookie file uploaded: {COOKIE_FILE}\n")

# ========================================
# KEEP-ALIVE MECHANISM
# ========================================
def keep_runtime_alive():
    """Keep Colab runtime alive by simulating activity"""
    while True:
        time.sleep(240)  # Every 4 minutes
        display(Javascript('console.log("Runtime keep-alive ping")'))

# Start keep-alive thread in background
print("🔄 Starting keep-alive mechanism...")
keep_alive_thread = threading.Thread(target=keep_runtime_alive, daemon=True)
keep_alive_thread.start()
print("✅ Keep-alive active - runtime won't disconnect during long operations\n")

# ========================================
# DATA CLASSES
# ========================================
@dataclass
class VideoMetadata:
    """Metadata for downloaded video"""
    video_id: str
    title: str
    url: str
    duration: float
    file_path: str
    source: str = "youtube"  # youtube or giphy

@dataclass
class VideoSection:
    """Represents a video section from the plan"""
    start_time: float
    end_time: float
    description: str
    transcript: str
    section_type: str  # "Video", "Real image", "PNG sequence", etc.

# ========================================
# PROGRESS TRACKER
# ========================================
class ProgressTracker:
    """Track progress to handle interruptions"""

    def __init__(self, script_name: str, output_base_dir: str):
        self.script_name = script_name
        self.progress_dir = os.path.join(output_base_dir, script_name, "videos")
        os.makedirs(self.progress_dir, exist_ok=True)

        self.progress_file = os.path.join(self.progress_dir, "progress_tracker.json")
        self.temp_file = os.path.join(self.progress_dir, "progress_tracker_temp.json")
        self.backup_file = os.path.join(self.progress_dir, "progress_tracker_backup.json")

        self.data = self._load_or_create()

    def _load_or_create(self) -> Dict:
        """Load existing progress or create new"""
        if os.path.exists(self.progress_file):
            print("📊 Found existing progress file - loading...")
            with open(self.progress_file, 'r') as f:
                return json.load(f)
        else:
            print("📊 Creating new progress tracker...")
            return {
                'script_name': self.script_name,
                'started_at': datetime.now().isoformat(),
                'sections': {},
                'giphy_used_ids': [],  # Track used Giphy GIFs
                'stats': {
                    'total_sections': 0,
                    'completed': 0,
                    'failed': 0,
                    'skipped': 0
                }
            }

    def is_section_complete(self, section_key: str) -> bool:
        """Check if section is already completed"""
        return section_key in self.data['sections'] and self.data['sections'][section_key]['status'] == 'completed'

    def mark_section_started(self, section: VideoSection):
        """Mark section as started"""
        section_key = f"{section.start_time:.2f}-{section.end_time:.2f}"
        self.data['sections'][section_key] = {
            'status': 'started',
            'description': section.description,
            'transcript': section.transcript,
            'started_at': datetime.now().isoformat(),
            'attempts': []
        }
        self._save()

    def mark_section_completed(self, section: VideoSection, metadata: Dict):
        """Mark section as completed"""
        section_key = f"{section.start_time:.2f}-{section.end_time:.2f}"
        if section_key in self.data['sections']:
            self.data['sections'][section_key]['status'] = 'completed'
            self.data['sections'][section_key]['completed_at'] = datetime.now().isoformat()
            self.data['sections'][section_key]['metadata'] = metadata
            self.data['stats']['completed'] += 1
        self._save()

    def mark_section_failed(self, section: VideoSection, error: str):
        """Mark section as failed"""
        section_key = f"{section.start_time:.2f}-{section.end_time:.2f}"
        if section_key in self.data['sections']:
            self.data['sections'][section_key]['status'] = 'failed'
            self.data['sections'][section_key]['error'] = error
            self.data['sections'][section_key]['failed_at'] = datetime.now().isoformat()
            self.data['stats']['failed'] += 1
        self._save()

    def log_attempt(self, section: VideoSection, attempt_data: Dict):
        """Log an attempt for a section"""
        section_key = f"{section.start_time:.2f}-{section.end_time:.2f}"
        if section_key in self.data['sections']:
            self.data['sections'][section_key]['attempts'].append({
                'timestamp': datetime.now().isoformat(),
                **attempt_data
            })
        self._save()

    def add_used_giphy_id(self, giphy_id: str):
        """Track used Giphy ID"""
        if giphy_id not in self.data['giphy_used_ids']:
            self.data['giphy_used_ids'].append(giphy_id)
            self._save()

    def is_giphy_id_used(self, giphy_id: str) -> bool:
        """Check if Giphy ID already used"""
        return giphy_id in self.data['giphy_used_ids']

    def _save(self):
        """Save progress with backup strategy"""
        # Backup current file if it exists
        if os.path.exists(self.progress_file):
            shutil.copy(self.progress_file, self.backup_file)

        # Write to temp file first
        with open(self.temp_file, 'w') as f:
            json.dump(self.data, f, indent=2)

        # If successful, move temp to main
        shutil.move(self.temp_file, self.progress_file)

        print(f"💾 Progress saved: {self.data['stats']['completed']}/{self.data['stats']['total_sections']} completed")

# ========================================
# VIDEO PLAN PARSER
# ========================================
class VideoPlanParser:
    """Parse video plan TXT files"""

    def __init__(self):
        print("✅ VideoPlanParser initialized\n")

    def parse_plan(self, file_path: str) -> tuple[str, List[VideoSection]]:
        """
        Parse video plan file

        Returns:
            (script_name, list of VideoSection objects)
        """
        print(f"📄 Parsing video plan: {file_path}")

        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Extract script name from filename
        script_name = Path(file_path).stem

        # Parse each line
        sections = []
        lines = content.strip().split('\n')

        for line in lines:
            if not line.strip():
                continue

            # Pattern: 0.00-1.84: Real image of Virgil van Dijk [transcript: ...]
            match = re.match(r'([\d.]+)-([\d.]+):\s*(.+?)\s*\[transcript:\s*(.+?)\]', line)

            if match:
                start_time = float(match.group(1))
                end_time = float(match.group(2))
                description = match.group(3).strip()
                transcript = match.group(4).strip()

                # Extract section type (e.g., "Video", "Real image", "PNG sequence")
                type_match = re.match(r'^(Video|Real image|PNG sequence|Meme Clip)', description)
                section_type = type_match.group(1) if type_match else "Unknown"

                section = VideoSection(
                    start_time=start_time,
                    end_time=end_time,
                    description=description,
                    transcript=transcript,
                    section_type=section_type
                )
                sections.append(section)

        print(f"   ✅ Parsed {len(sections)} sections")

        # Filter for video sections only
        video_sections = [s for s in sections if s.section_type == "Video"]
        print(f"   🎬 Found {len(video_sections)} video sections to process\n")

        return script_name, video_sections

# ========================================
# GIPHY HANDLER
# ========================================
class GiphyHandler:
    """Handle Giphy searches and downloads"""

    def __init__(self, api_key: str):
        self.api_key = api_key
        print("✅ GiphyHandler initialized\n")

    def search_giphy(self, query: str, limit: int = 10) -> List[Dict]:
        """Search Giphy for GIFs/videos"""
        print(f"🎭 Searching Giphy: '{query}'")

        try:
            # Search both GIFs and clips
            url = f"https://api.giphy.com/v1/gifs/search"
            params = {
                'api_key': self.api_key,
                'q': query,
                'limit': limit,
                'rating': 'pg-13'
            }

            response = requests.get(url, params=params, timeout=10)

            if response.status_code != 200:
                print(f"   ❌ Giphy API error: {response.status_code}")
                return []

            data = response.json()
            results = []

            for item in data.get('data', []):
                results.append({
                    'id': item['id'],
                    'title': item.get('title', ''),
                    'url': item['images']['original']['url'],
                    'mp4_url': item['images']['original'].get('mp4', ''),
                    'width': int(item['images']['original'].get('width', 0)),
                    'height': int(item['images']['original'].get('height', 0))
                })

            print(f"   ✅ Found {len(results)} Giphy results")
            return results

        except Exception as e:
            print(f"   ❌ Giphy search error: {e}")
            return []

    def download_giphy(self, giphy_data: Dict, output_path: str) -> bool:
        """Download and convert Giphy to MP4"""
        try:
            # Use MP4 version if available
            download_url = giphy_data.get('mp4_url') or giphy_data.get('url')

            print(f"   📥 Downloading from Giphy...")
            response = requests.get(download_url, timeout=30, stream=True)

            if response.status_code != 200:
                return False

            # Save temporarily
            temp_path = output_path + ".temp"
            with open(temp_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            # If it's a GIF, convert to MP4
            if download_url.endswith('.gif'):
                print(f"   🔄 Converting GIF to MP4...")
                cap = cv2.VideoCapture(temp_path)
                fps = 15  # Standard GIF framerate

                frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    out.write(frame)

                cap.release()
                out.release()
                os.remove(temp_path)
            else:
                # Already MP4, just rename
                shutil.move(temp_path, output_path)

            print(f"   ✅ Giphy downloaded and converted to MP4")
            return True

        except Exception as e:
            print(f"   ❌ Giphy download error: {e}")
            return False



# ========================================
# PEXELS HANDLER
# ========================================
class PexelsHandler:
    """Handle Pexels video searches and downloads"""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.pexels.com/videos"
        print("✅ PexelsHandler initialized\n")

    def search_videos(self, query: str, per_page: int = 15) -> List[Dict]:
        """Search Pexels for videos"""
        print(f"🎥 Searching Pexels: '{query}'")

        try:
            headers = {'Authorization': self.api_key}
            params = {'query': query, 'per_page': per_page, 'orientation': 'landscape'}

            response = requests.get(f"{self.base_url}/search",
                                  headers=headers, params=params, timeout=10)

            if response.status_code != 200:
                print(f"   ❌ Pexels API error: {response.status_code}")
                return []

            data = response.json()
            results = []

            for video in data.get('videos', []):
                # Get best quality video file
                video_files = video.get('video_files', [])
                hd_file = next((f for f in video_files if f.get('quality') == 'hd'),
                              video_files[0] if video_files else None)

                if hd_file:
                    results.append({
                        'id': video['id'],
                        'url': video['url'],
                        'download_url': hd_file['link'],
                        'duration': video.get('duration', 0),
                        'width': video.get('width', 0),
                        'height': video.get('height', 0),
                        'user': video.get('user', {}).get('name', '')
                    })

            print(f"   ✅ Found {len(results)} Pexels videos")
            return results

        except Exception as e:
            print(f"   ❌ Pexels search error: {e}")
            return []

    def download_video(self, video_data: Dict, output_path: str) -> bool:
        """Download Pexels video"""
        try:
            print(f"   📥 Downloading from Pexels...")
            response = requests.get(video_data['download_url'],
                                  timeout=60, stream=True)

            if response.status_code != 200:
                return False

            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            print(f"   ✅ Pexels video downloaded")
            return True

        except Exception as e:
            print(f"   ❌ Pexels download error: {e}")
            return False

# ========================================
# PIXABAY HANDLER
# ========================================
class PixabayHandler:
    """Handle Pixabay video searches and downloads"""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://pixabay.com/api/videos/"
        print("✅ PixabayHandler initialized\n")

    def search_videos(self, query: str, per_page: int = 20) -> List[Dict]:
        """Search Pixabay for videos"""
        print(f"🎥 Searching Pixabay: '{query}'")

        try:
            params = {
                'key': self.api_key,
                'q': query,
                'per_page': per_page,
                'video_type': 'all'
            }

            response = requests.get(self.base_url, params=params, timeout=10)

            if response.status_code != 200:
                print(f"   ❌ Pixabay API error: {response.status_code}")
                return []

            data = response.json()
            results = []

            for video in data.get('hits', []):
                # Get medium quality video
                videos = video.get('videos', {})
                medium = videos.get('medium', videos.get('large', {}))

                if medium:
                    results.append({
                        'id': video['id'],
                        'download_url': medium.get('url'),
                        'duration': video.get('duration', 0),
                        'tags': video.get('tags', ''),
                        'user': video.get('user', '')
                    })

            print(f"   ✅ Found {len(results)} Pixabay videos")
            return results

        except Exception as e:
            print(f"   ❌ Pixabay search error: {e}")
            return []

    def download_video(self, video_data: Dict, output_path: str) -> bool:
        """Download Pixabay video"""
        try:
            print(f"   📥 Downloading from Pixabay...")
            response = requests.get(video_data['download_url'],
                                  timeout=60, stream=True)

            if response.status_code != 200:
                return False

            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            print(f"   ✅ Pixabay video downloaded")
            return True

        except Exception as e:
            print(f"   ❌ Pixabay download error: {e}")
            return False

# ========================================
# GEMINI ANALYZER WITH STRICT VERIFICATION
# ========================================
class GeminiVideoAnalyzer:
    """Analyze videos with Gemini to find specific moments with strict verification"""

    def __init__(self, api_key: str):
        print("🤖 Initializing Gemini analyzer...")
        self.api_key = api_key
        genai.configure(api_key=api_key)
        # Use gemini-2.5-flash (same as original code)
        self.model = genai.GenerativeModel("gemini-2.5-flash")
        print("✅ Gemini initialized!\n")

    def extract_search_query(self, section: VideoSection) -> str:
        """
        Extract search query from description by removing 'Video of' prefix
        NO Gemini modification - use exact text after 'Video of'
        """
        description = section.description

        # Remove "Video of", "Video:", "Video " prefix
        query = re.sub(r'^Video\s*(?:of|:)?\s*', '', description, flags=re.IGNORECASE).strip()

        print(f"🔍 Search query: '{query}' (extracted directly from plan)")
        return query

    def generate_simplified_search(self, section: VideoSection, previous_query: str) -> Optional[str]:
        """
        Ask Gemini to generate a simpler, more generic search query
        that's guaranteed to get results
        """
        print(f"🤖 Asking Gemini to simplify search query...")

        prompt = f"""The following search query failed to find suitable videos across multiple sources:

    ORIGINAL QUERY: {previous_query}
    FULL DESCRIPTION: {section.description}
    TRANSCRIPT CONTEXT: {section.transcript}

    Your task: Generate a SIMPLER, more GENERIC search query that will definitely return results.

    Rules:
    - Remove specific names/brands/teams
    - Use general concepts instead of specifics
    - Focus on the ACTION or EMOTION rather than the subject
    - Make it 1-3 words maximum
    - Think about what stock footage would exist

    Examples:
    - "Virgil van Dijk celebrating" → "soccer celebration"
    - "Tesla Cybertruck reveal" → "futuristic vehicle"
    - "Joe Rogan laughing" → "man laughing"

    RESPOND WITH ONLY THE SIMPLIFIED QUERY (no quotes, no explanation):"""

        try:
            response = self.model.generate_content(prompt)
            simplified_query = response.text.strip().strip('"\'')

            print(f"   ✅ Simplified query: '{simplified_query}'")
            return simplified_query

        except Exception as e:
            print(f"   ❌ Gemini simplification error: {e}")
            return None

    def verify_video_metadata(self, video_info: Dict, section: VideoSection) -> Tuple[bool, float, str]:
        """
        Verify video matches description using metadata (title, description)

        Returns:
            (matches, confidence_score, reasoning)
        """
        print(f"🔍 Verifying video metadata...")

        # Skip metadata check if no description available
        if not video_info.get('description'):
            print(f"   ⏭️  No description available - skipping metadata check")
            return True, 70, "No description to verify, proceeding to frame check"

        is_meme = 'meme' in section.description.lower()

        prompt = f"""You are verifying if a YouTube video matches a specific description.

VIDEO METADATA:
- Title: {video_info.get('title', '')}
- Channel: {video_info.get('uploader', '')}
- Description: {video_info.get('description', '')[:500]}

REQUIRED DESCRIPTION: {section.description}
TRANSCRIPT CONTEXT: {section.transcript}

IS THIS A MEME? {is_meme}

VERIFICATION RULES:
{'- For MEMES: Check if most key elements are present (looser matching for subjective content)' if is_meme else '- For REGULAR VIDEOS: ALL elements in the description must be verifiable from metadata'}
- Consider title, channel name, and description
- Be strict - if metadata doesn't suggest a clear match, reject it

RESPOND IN THIS EXACT JSON FORMAT (no markdown, no extra text):
{{
    "matches": true,
    "confidence": 85,
    "reasoning": "brief explanation"
}}
"""

        # Try 3 times with error handling
        for attempt in range(3):
            try:
                if attempt == 0:
                    generation_config = genai.types.GenerationConfig(
                        response_mime_type="application/json",
                        temperature=0.1
                    )
                    response = self.model.generate_content(prompt, generation_config=generation_config)
                else:
                    response = self.model.generate_content(prompt)

                response_text = response.text.strip()

                # Try to extract JSON from response
                json_match = re.search(r'\{[^{}]*"matches"[^{}]*\}', response_text, re.DOTALL)
                if json_match:
                    json_text = json_match.group(0)
                else:
                    json_text = response_text

                result = json.loads(json_text)

                matches = result.get('matches', False)
                confidence = result.get('confidence', 0)
                reasoning = result.get('reasoning', '')

                print(f"   {'✅' if matches else '❌'} Metadata check: {confidence}% confidence")
                print(f"   💭 Reasoning: {reasoning}")

                return matches, confidence, reasoning

            except json.JSONDecodeError as e:
                print(f"   ⚠️ JSON parse error (attempt {attempt+1}/3)")
                if attempt == 2:
                    # Last attempt - skip metadata check
                    print(f"   ⏭️  Skipping metadata verification, will rely on frame check")
                    return True, 50, "Metadata check failed, proceeding to frame verification"
                time.sleep(2)

            except Exception as e:
                print(f"   ⚠️ Attempt {attempt+1} error: {e}")
                if attempt == 2:
                    print(f"   ⏭️  Skipping metadata verification")
                    return True, 50, "Metadata check failed, proceeding to frame verification"
                time.sleep(2)

        return True, 50, "Metadata verification skipped after errors"

    def verify_video_frames(self, video_path: str, section: VideoSection) -> Tuple[bool, float, str]:
        """
        Verify video matches description by analyzing actual video frames
        This is the CRITICAL verification step

        Returns:
            (matches, confidence_score, reasoning)
        """
        print(f"🎬 Analyzing video frames for strict verification...")

        # Extract frames from video
        frames = self._extract_frames(video_path, num_frames=10)
        if not frames:
            print(f"   ❌ Could not extract frames")
            return False, 0, "Failed to extract frames"

        # Upload frames to Gemini
        print(f"   📤 Uploading {len(frames)} frames to Gemini...")
        uploaded_files = []
        for i, frame_path in enumerate(frames):
            uploaded_file = genai.upload_file(frame_path)
            uploaded_files.append(uploaded_file)
            print(f"      Frame {i+1}/{len(frames)} uploaded")

        is_meme = 'meme' in section.description.lower()

        prompt = f"""You are strictly verifying if this video matches the required description by analyzing its frames.

REQUIRED DESCRIPTION: {section.description}
TRANSCRIPT CONTEXT: {section.transcript}

IS THIS A MEME? {is_meme}

VERIFICATION RULES:
{'''- For MEMES: Verify that MOST KEY ELEMENTS are visually present
- Memes are subjective, so allow some flexibility
- But the overall vibe and main elements must match''' if is_meme else '''- For REGULAR VIDEOS: ALL elements must be clearly visible and verifiable
- Person identification must be accurate
- Clothing/kit colors must match
- Actions/emotions must be clearly visible
- Be VERY STRICT - if any key element is missing or unclear, REJECT'''}

ANALYZE THE FRAMES AND VERIFY:
- What do you see in these frames?
- Does it match ALL (or most for memes) required elements?
- Be honest and strict - only approve clear matches

RESPOND IN THIS EXACT JSON FORMAT:
{{
    "matches": true/false,
    "confidence": 0-100,
    "reasoning": "detailed explanation of what you see and why it does/doesn't match",
    "missing_elements": ["list any required elements that are missing or unclear"]
}}
"""

        try:
            # Force JSON response from Gemini
            generation_config = genai.types.GenerationConfig(
                response_mime_type="application/json"
            )

            response = self.model.generate_content(
                [prompt] + uploaded_files,
                generation_config=generation_config
            )
            response_text = response.text.strip()

            print(f"   📝 Gemini response preview: {response_text[:200]}...")

            # Try to extract JSON from response (in case it's wrapped in markdown)
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                json_text = json_match.group(0)
            else:
                json_text = response_text

            try:
                result = json.loads(json_text)
            except json.JSONDecodeError:
                print(f"   ⚠️ Could not parse JSON, trying to interpret response...")
                # If JSON parsing fails, try to interpret the response
                if 'match' in response_text.lower() and ('yes' in response_text.lower() or 'true' in response_text.lower()):
                    result = {'matches': True, 'confidence': 70, 'reasoning': response_text[:200]}
                else:
                    result = {'matches': False, 'confidence': 0, 'reasoning': response_text[:200]}

            matches = result.get('matches', False)
            confidence = result.get('confidence', 0)
            reasoning = result.get('reasoning', '')
            missing = result.get('missing_elements', [])

            print(f"   {'✅' if matches else '❌'} Frame analysis: {confidence}% confidence")
            print(f"   💭 Reasoning: {reasoning}")
            if missing:
                print(f"   ⚠️ Missing elements: {', '.join(missing)}")

            # Cleanup uploaded files
            for uploaded_file in uploaded_files:
                try:
                    genai.delete_file(uploaded_file.name)
                except:
                    pass

            # Cleanup frame files
            for frame_path in frames:
                try:
                    os.remove(frame_path)
                except:
                    pass

            return matches, confidence, reasoning

        except Exception as e:
            print(f"   ❌ Frame verification error: {e}")
            print(f"   📄 Full error details: {str(e)[:500]}")

            # Cleanup on error
            for uploaded_file in uploaded_files:
                try:
                    genai.delete_file(uploaded_file.name)
                except:
                    pass
            for frame_path in frames:
                try:
                    os.remove(frame_path)
                except:
                    pass

            return False, 0, str(e)

    def _extract_frames(self, video_path: str, num_frames: int = 10) -> List[str]:
        """Extract evenly spaced frames from video"""
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames == 0:
            cap.release()
            return []

        # Calculate frame indices to extract
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

        extracted_frames = []
        temp_dir = "/tmp/gemini_frames"
        os.makedirs(temp_dir, exist_ok=True)

        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()

            if ret:
                frame_path = f"{temp_dir}/frame_{idx}.jpg"
                cv2.imwrite(frame_path, frame)
                extracted_frames.append(frame_path)

        cap.release()
        return extracted_frames

    def analyze_and_find_moment(self, video_path: str, section: VideoSection) -> Dict:
        """Analyze video and find the best moment matching the transcript"""
        print(f"🎬 Analyzing video to find best moment...")

        # Upload full video to Gemini
        print(f"   📤 Uploading video to Gemini...")
        video_file = genai.upload_file(video_path)
        print(f"   ✅ Video uploaded")

        prompt = f"""Analyze this video and find the BEST moment that matches this description.

DESCRIPTION: {section.description}
TRANSCRIPT: {section.transcript}

Your task:
1. Watch the video carefully
2. Find the moment that best matches the description and transcript
3. Return the timestamp range (start and end in seconds)

The moment should:
- Match the visual action/scene described
- Fit with the transcript context
- Be engaging and clear

RESPOND IN THIS EXACT JSON FORMAT:
{{
    "start_second": 0.0,
    "end_second": 5.0,
    "reasoning": "why this moment is the best match"
}}
"""

        try:
            response = self.model.generate_content([prompt, video_file])
            result = json.loads(response.text.strip())

            start_second = float(result.get('start_second', 0))
            end_second = float(result.get('end_second', 5))
            reasoning = result.get('reasoning', '')

            print(f"   ✅ Found moment: {start_second:.2f}s - {end_second:.2f}s")
            print(f"   💭 Reasoning: {reasoning}")

            # Cleanup
            genai.delete_file(video_file.name)

            return {
                'start_second': start_second,
                'end_second': end_second,
                'reasoning': reasoning
            }

        except Exception as e:
            print(f"   ⚠️ Error analyzing video: {e}")
            # Fallback to using first 5 seconds
            return {
                'start_second': 0.0,
                'end_second': 5.0,
                'reasoning': f"Error in analysis, using default: {e}"
            }

    def extract_video_clip(self, video_path: str, start_second: float, end_second: float, output_path: str) -> str:
        """Extract a clip from video between start and end seconds"""
        cap = cv2.VideoCapture(video_path)

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Define codec and create VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # Calculate frame numbers
        start_frame = int(start_second * fps)
        end_frame = int(end_second * fps)

        # Set to start frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        # Extract frames
        current_frame = start_frame
        while current_frame <= end_frame:
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
            current_frame += 1

        cap.release()
        out.release()

        return output_path

# ========================================
# MAIN ORCHESTRATOR
# ========================================
class AssetCollector:
    """Main orchestrator for collecting video assets"""

    def __init__(self, api_key: str, giphy_key: str, pexels_key: str, pixabay_key: str,
             output_base_dir: str = "/content/drive/MyDrive/Angry Dude Scripts/collected_assets"):
        self.api_key = api_key
        self.giphy_key = giphy_key
        self.output_base_dir = output_base_dir

        # Initialize components
        self.parser = VideoPlanParser()
        self.analyzer = GeminiVideoAnalyzer(api_key)
        self.giphy = GiphyHandler(giphy_key)
        self.pexels = PexelsHandler(pexels_key)
        self.pixabay = PixabayHandler(pixabay_key)

        Path(output_base_dir).mkdir(parents=True, exist_ok=True)
        print(f"✅ AssetCollector initialized")
        print(f"   Output base: {output_base_dir}\n")

    def process_video_plan(self, plan_file_path: str, progress_tracker: Optional[ProgressTracker] = None):
        """Process entire video plan"""
        print("="*70)
        print("🎬 STARTING ASSET COLLECTION")
        print("="*70)
        print()

        # Parse plan
        script_name, video_sections = self.parser.parse_plan(plan_file_path)

        # Initialize or use existing progress tracker
        if not progress_tracker:
            progress_tracker = ProgressTracker(script_name, self.output_base_dir)

        progress_tracker.data['stats']['total_sections'] = len(video_sections)

        # Create output directory for this script
        script_output_dir = os.path.join(self.output_base_dir, script_name, "videos")
        Path(script_output_dir).mkdir(parents=True, exist_ok=True)

        print(f"📂 Output directory: {script_output_dir}\n")
        print(f"📋 Processing {len(video_sections)} video sections...\n")

        # Process each video section
        for i, section in enumerate(video_sections, 1):
            section_key = f"{section.start_time:.2f}-{section.end_time:.2f}"

            # Check if already completed
            if progress_tracker.is_section_complete(section_key):
                print("="*70)
                print(f"SECTION {i}/{len(video_sections)} - ✅ ALREADY COMPLETED")
                print("="*70)
                print(f"📝 Description: {section.description}")
                print(f"⏭️  Skipping...\n")
                continue

            print("="*70)
            print(f"SECTION {i}/{len(video_sections)}")
            print("="*70)
            print(f"📝 Description: {section.description}")
            print(f"💬 Transcript: {section.transcript}")
            print(f"⏱️  Plan timestamp: {section.start_time}-{section.end_time}\n")

            progress_tracker.mark_section_started(section)

            try:
                # Check if this is a meme segment
                is_meme_segment = 'meme' in section.description.lower()

                if is_meme_segment:
                    # Use Giphy for meme segments
                    success = self._process_meme_segment(section, script_output_dir, progress_tracker)
                else:
                    # Use stock videos for regular segments
                    success = self._process_stock_video_segment(section, script_output_dir, progress_tracker)

                if success:
                    print(f"✅ Section completed successfully!\n")
                else:
                    print(f"❌ Section failed after all attempts\n")
                    progress_tracker.mark_section_failed(section, "All attempts exhausted")

            except Exception as e:
                print(f"❌ Error processing section: {e}")
                print("⚠️ Skipping to next section...\n")
                progress_tracker.mark_section_failed(section, str(e))
                continue

        print("="*70)
        print("🎉 ASSET COLLECTION COMPLETE!")
        print("="*70)
        print(f"📊 Stats:")
        print(f"   ✅ Completed: {progress_tracker.data['stats']['completed']}")
        print(f"   ❌ Failed: {progress_tracker.data['stats']['failed']}")
        print(f"   📁 Saved to: {script_output_dir}\n")

    def _process_meme_segment(self, section: VideoSection, output_dir: str, tracker: ProgressTracker) -> bool:
        """Process meme segment using Giphy with YouTube fallback"""
        print(f"🎭 Processing MEME segment with Giphy...\n")

        # Extract search query (remove "Video of" prefix)
        search_query = self.analyzer.extract_search_query(section)

        # Try Giphy first
        giphy_results = self.giphy.search_giphy(search_query, limit=20)

        if giphy_results:
            # Try up to 5 different GIFs
            for attempt in range(min(5, len(giphy_results))):
                giphy_item = giphy_results[attempt]

                # Check if already used
                if tracker.is_giphy_id_used(giphy_item['id']):
                    print(f"   ⏭️  GIF {attempt + 1} already used, skipping...")
                    continue

                print(f"\n🎬 Trying Giphy option {attempt + 1}/{min(5, len(giphy_results))}:")
                print(f"   Title: {giphy_item['title']}")

                tracker.log_attempt(section, {
                    'attempt_number': attempt + 1,
                    'source': 'giphy',
                    'giphy_id': giphy_item['id'],
                    'title': giphy_item['title']
                })

                # Download GIF/video
                output_filename = f"{section.start_time:.2f}-{section.end_time:.2f}.mp4"
                temp_path = os.path.join(output_dir, f"temp_{output_filename}")

                success = self.giphy.download_giphy(giphy_item, temp_path)

                if not success:
                    print(f"   ❌ Download failed")
                    continue

                # Verify with Gemini (frame analysis only for memes)
                matches, confidence, reasoning = self.analyzer.verify_video_frames(temp_path, section)

                if matches and confidence >= 70:  # Lower threshold for memes
                    # This is a good match!
                    final_path = os.path.join(output_dir, output_filename)
                    shutil.move(temp_path, final_path)

                    # Mark as used
                    tracker.add_used_giphy_id(giphy_item['id'])

                    # Mark section complete
                    tracker.mark_section_completed(section, {
                        'source': 'giphy',
                        'giphy_id': giphy_item['id'],
                        'file_path': final_path,
                        'confidence': confidence
                    })

                    file_size = os.path.getsize(final_path) / (1024 * 1024)
                    print(f"\n✅ SAVED: {output_filename} ({file_size:.2f} MB)")
                    print(f"   Source: Giphy")
                    print(f"   Confidence: {confidence}%")
                    print(f"   Path: {final_path}\n")

                    return True
                else:
                    print(f"   ❌ Verification failed (confidence: {confidence}%)")
                    print(f"   💭 {reasoning}")
                    os.remove(temp_path)
                    continue

            print(f"\n⚠️ No suitable Giphy GIF found after trying {min(5, len(giphy_results))} options")
        else:
            print(f"❌ No Giphy results found for: '{search_query}'")

        # FALLBACK TO STOCK VIDEOS for memes (which includes Gemini fallback)
        print(f"\n🔄 Falling back to stock videos for meme segment...")
        print(f"   (This will also try Gemini simplification if needed)\n")
        return self._process_stock_video_segment(section, output_dir, tracker)

    def _process_stock_video_segment(self, section: VideoSection, output_dir: str, tracker: ProgressTracker) -> bool:
        """Process segment using stock video sources with Gemini fallback"""
        print(f"🎥 Processing with stock video sources...\n")

        # Extract search query
        search_query = self.analyzer.extract_search_query(section)

        # Try with original query
        success = self._try_all_sources(section, search_query, output_dir, tracker)
        if success:
            return True

        # GEMINI FALLBACK: Try up to 3 simplified searches
        print(f"\n🔄 All sources failed with original query")
        print(f"🤖 Activating Gemini simplification fallback...\n")

        for attempt in range(3):
            print(f"="*70)
            print(f"GEMINI FALLBACK ATTEMPT {attempt + 1}/3")
            print(f"="*70)

            # Get simplified query from Gemini
            simplified_query = self.analyzer.generate_simplified_search(section, search_query)

            if not simplified_query:
                print(f"   ❌ Could not generate simplified query")
                continue

            print(f"   🔍 Trying simplified query: '{simplified_query}'\n")

            # Try all sources with simplified query
            success = self._try_all_sources(section, simplified_query, output_dir, tracker)
            if success:
                print(f"\n🎉 SUCCESS with simplified query!")
                return True

            print(f"   ❌ Attempt {attempt + 1} failed, trying again...\n")

        print(f"\n❌ All attempts exhausted (original + 3 Gemini fallbacks)")
        return False

    def _try_all_sources(self, section: VideoSection, search_query: str, output_dir: str, tracker: ProgressTracker) -> bool:
        """Try all video sources with given query"""
        # Try Pexels first
        print(f"📹 Trying Pexels with: '{search_query}'")
        success = self._try_pexels(section, search_query, output_dir, tracker)
        if success:
            return True

        # Fallback to Pixabay
        print(f"\n🔄 Pexels didn't work, trying Pixabay with: '{search_query}'")
        success = self._try_pixabay(section, search_query, output_dir, tracker)
        if success:
            return True

        return False

    def _try_pexels(self, section: VideoSection, search_query: str, output_dir: str, tracker: ProgressTracker) -> bool:
        """Try Pexels videos"""
        pexels_results = self.pexels.search_videos(search_query, per_page=15)

        if not pexels_results:
            print(f"   ❌ No Pexels results found")
            return False

        # Try up to 5 videos
        for attempt in range(min(5, len(pexels_results))):
            video_item = pexels_results[attempt]

            print(f"\n🎬 Trying Pexels video {attempt + 1}/{min(5, len(pexels_results))}:")
            print(f"   Duration: {video_item['duration']:.1f}s")

            tracker.log_attempt(section, {
                'attempt_number': attempt + 1,
                'source': 'pexels',
                'video_id': video_item['id']
            })

            # Download video
            output_filename = f"{section.start_time:.2f}-{section.end_time:.2f}.mp4"
            temp_path = os.path.join(output_dir, f"temp_{output_filename}")

            success = self.pexels.download_video(video_item, temp_path)

            if not success:
                print(f"   ❌ Download failed")
                continue

            # Verify with Gemini
            matches, confidence, reasoning = self.analyzer.verify_video_frames(temp_path, section)

            if matches and confidence >= 65:  # Reasonable threshold for stock footage
                final_path = os.path.join(output_dir, output_filename)
                shutil.move(temp_path, final_path)

                # Mark section complete
                tracker.mark_section_completed(section, {
                    'source': 'pexels',
                    'video_id': video_item['id'],
                    'file_path': final_path,
                    'confidence': confidence
                })

                file_size = os.path.getsize(final_path) / (1024 * 1024)
                print(f"\n✅ SAVED: {output_filename} ({file_size:.2f} MB)")
                print(f"   Source: Pexels")
                print(f"   Confidence: {confidence}%")
                print(f"   Path: {final_path}\n")

                return True
            else:
                print(f"   ❌ Verification failed (confidence: {confidence}%)")
                print(f"   💭 {reasoning}")
                os.remove(temp_path)
                continue

        return False

    def _try_pixabay(self, section: VideoSection, search_query: str, output_dir: str, tracker: ProgressTracker) -> bool:
        """Try Pixabay videos"""
        pixabay_results = self.pixabay.search_videos(search_query, per_page=20)

        if not pixabay_results:
            print(f"   ❌ No Pixabay results found")
            return False

        # Try up to 5 videos
        for attempt in range(min(5, len(pixabay_results))):
            video_item = pixabay_results[attempt]

            print(f"\n🎬 Trying Pixabay video {attempt + 1}/{min(5, len(pixabay_results))}:")
            print(f"   Tags: {video_item.get('tags', 'N/A')}")

            tracker.log_attempt(section, {
                'attempt_number': attempt + 1,
                'source': 'pixabay',
                'video_id': video_item['id']
            })

            # Download video
            output_filename = f"{section.start_time:.2f}-{section.end_time:.2f}.mp4"
            temp_path = os.path.join(output_dir, f"temp_{output_filename}")

            success = self.pixabay.download_video(video_item, temp_path)

            if not success:
                print(f"   ❌ Download failed")
                continue

            # Verify with Gemini
            matches, confidence, reasoning = self.analyzer.verify_video_frames(temp_path, section)

            if matches and confidence >= 65:
                final_path = os.path.join(output_dir, output_filename)
                shutil.move(temp_path, final_path)

                # Mark section complete
                tracker.mark_section_completed(section, {
                    'source': 'pixabay',
                    'video_id': video_item['id'],
                    'file_path': final_path,
                    'confidence': confidence
                })

                file_size = os.path.getsize(final_path) / (1024 * 1024)
                print(f"\n✅ SAVED: {output_filename} ({file_size:.2f} MB)")
                print(f"   Source: Pixabay")
                print(f"   Confidence: {confidence}%")
                print(f"   Path: {final_path}\n")

                return True
            else:
                print(f"   ❌ Verification failed (confidence: {confidence}%)")
                print(f"   💭 {reasoning}")
                os.remove(temp_path)
                continue

        return False

# ========================================
# MAIN EXECUTION
# ========================================

print("="*70)
print("🎬 ENHANCED VIDEO PLAN ASSET COLLECTOR")
print("="*70)
print()

# Step 1: List and select video plan from Drive
video_plans_dir = "/content/drive/MyDrive/Angry Dude Scripts/video_plans"

print(f"📂 Scanning for video plans in: {video_plans_dir}\n")

if not os.path.exists(video_plans_dir):
    print(f"❌ Directory not found: {video_plans_dir}")
    print("Creating directory...")
    os.makedirs(video_plans_dir, exist_ok=True)
    print("✅ Directory created. Please add your video plan files there and re-run.\n")
    raise SystemExit()

# Get all .txt files
video_plans = [f for f in os.listdir(video_plans_dir) if f.endswith('.txt')]

if not video_plans:
    print("❌ No video plan files (.txt) found!")
    print(f"Please add your video plan files to: {video_plans_dir}\n")
    raise SystemExit()

print("📋 Available video plans:")
for i, plan in enumerate(video_plans, 1):
    print(f"   {i}. {plan}")

print()
selection = input(f"Select video plan (1-{len(video_plans)}): ").strip()

try:
    plan_index = int(selection) - 1
    if plan_index < 0 or plan_index >= len(video_plans):
        raise ValueError()
    plan_file = os.path.join(video_plans_dir, video_plans[plan_index])
    print(f"\n✅ Selected: {video_plans[plan_index]}\n")
except:
    print("❌ Invalid selection!")
    raise SystemExit()

# Step 2: Check for existing progress tracker
script_name = Path(plan_file).stem
progress_file_path = f"/content/drive/MyDrive/Angry Dude Scripts/collected_assets/{script_name}/videos/progress_tracker.json"

tracker = None
if os.path.exists(progress_file_path):
    print(f"📊 Found existing progress tracker!")
    print(f"   Path: {progress_file_path}")

    resume = input("\nResume from previous progress? (y/n): ").strip().lower()

    if resume == 'y':
        try:
            with open(progress_file_path, 'r') as f:
                progress_data = json.load(f)

            # Create tracker with existing data
            tracker = ProgressTracker(script_name, "/content/drive/MyDrive/Angry Dude Scripts/collected_assets")
            tracker.data = progress_data

            print(f"\n📊 Resuming from previous progress:")
            print(f"   Completed: {tracker.data['stats']['completed']}")
            print(f"   Failed: {tracker.data['stats']['failed']}\n")
        except Exception as e:
            print(f"⚠️  Error loading progress file: {e}")
            print("Starting fresh...\n")
            tracker = None
    else:
        print("Starting fresh...\n")
else:
    print("📊 No previous progress found - starting fresh\n")

# Step 3: API Keys
GEMINI_API_KEY = "" #input api key here or setup with .env
GIPHY_API_KEY = ""  #input api key here or setup with .env
PEXELS_API_KEY = "" #input api key here or setup with .env
PIXABAY_API_KEY = "" #input api key here or setup with .env

# Step 4: Initialize and run
collector = AssetCollector(GEMINI_API_KEY, GIPHY_API_KEY, PEXELS_API_KEY, PIXABAY_API_KEY)

# Process the plan
collector.process_video_plan(plan_file, progress_tracker=tracker)

print("\n🎊 ALL DONE! Check your Google Drive at:")
print(f"   /content/drive/MyDrive/Angry Dude Scripts/collected_assets/")