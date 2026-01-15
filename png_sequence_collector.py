#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#
#      PNG SEQUENCE SECTION
#
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++




# ========================================
# MINIMAL PNG SEQUENCE TEST SCRIPT
# ========================================
# Perfect this logic, then integrate into main code

!pip install -q google-generativeai opencv-python-headless ultralytics pillow requests ipywidgets decord playwright undetected-chromedriver
# Upgrade to latest yt-dlp
!pip install -U yt-dlp

# Verify version
!yt-dlp --version

# Install Playwright browsers and dependencies
!playwright install chromium
!playwright install-deps chromium

# Install Chrome for undetected-chromedriver
!apt-get update
!apt-get install -y chromium-browser chromium-chromedriver

# Install deno (JavaScript runtime for yt-dlp)
!curl -fsSL https://deno.land/install.sh | sh
!ln -sf /root/.deno/bin/deno /usr/local/bin/deno

# Verify installation
!deno --version

# Set Chrome binary location for Colab
import os
os.environ['CHROME_BIN'] = '/usr/bin/chromium-browser'
from google.colab import drive, files
import threading
import json
import shutil
from pathlib import Path
from datetime import datetime
from IPython.display import display, Javascript
import yt_dlp
from decord import VideoReader, cpu

# Mount Google Drive
print("📂 Mounting Google Drive...")
drive.mount('/content/drive')
print("✅ Drive mounted!\n")

import cv2
import numpy as np
from collections import deque
from ultralytics import YOLO
import google.generativeai as genai
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import yt_dlp
import os
import re
import requests  # ← ADD THIS
import time
import random
from PIL import Image
import io
import base64
import nest_asyncio
nest_asyncio.apply()

# ========================================
# DATA CLASSES
# ========================================
@dataclass
class ZoomSegment:
    start_time: float
    end_time: float
    duration: float

@dataclass
class PlanSection:
    description: str
    transcript: str
    line_number: int

@dataclass
class PNGSequenceSection:
    """Represents a PNG sequence section from the plan"""
    start_time: float
    end_time: float
    description: str
    transcript: str
    line_number: int
    script_name: str

@dataclass
class MatchContext:
    """Track match continuity across sections"""
    teams: List[str]  # ["Barcelona", "Real Madrid"]
    date: str  # "October 26, 2024"
    home_jersey_color: str  # "blue and red"
    away_jersey_color: str  # "white"
    stadium: str  # "Camp Nou"
    competition: str  # "La Liga"
    last_section_line: int  # Track which section this came from

# ADD THIS CLASS:
@dataclass
class ImageCandidate:
    """Single downloaded image candidate"""
    path: str
    url: str
    width: int
    height: int
    file_size: int

@dataclass
class ImageSequenceCandidate:
    """Single image in a 3-image sequence"""
    path: str
    url: str
    width: int
    height: int
    file_size: int
    sequence_position: str  # "beginning", "middle", or "end"
    narrative_score: float  # How well it fits the story progression

@dataclass
class VideoDiscoveryResult:
    """Result from video discovery phase"""
    section: PNGSequenceSection
    video_url: Optional[str]
    channel_handle: Optional[str]
    title: str
    duration: int
    upload_date: str
    search_method: str  # "team_a_exact", "team_b_exact", "simplified", "player_highlight", "failed"
    fallback_reason: Optional[str] = None
    is_player_highlight: bool = False

@dataclass
class BatchDownloadQueue:
    """Queue of videos to download in one session"""
    videos: List[VideoDiscoveryResult]
    total_duration_estimate: int
    estimated_download_time: int


# ========================================
# COOKIE UPLOAD FOR YOUTUBE
# ========================================
def upload_youtube_cookies():
    """
    Prompt user to upload YouTube cookies for bot detection bypass
    Returns path to cookies file or None
    """
    print("="*70)
    print("🍪 YOUTUBE COOKIE UPLOAD (OPTIONAL)")
    print("="*70)
    print("YouTube blocks automated downloads from Colab.")
    print("To bypass this, you can upload your YouTube cookies.\n")
    print("📝 How to export cookies:")
    print("   1. Install browser extension: 'Get cookies.txt LOCALLY'")
    print("      Chrome: https://chrome.google.com/webstore/detail/get-cookiestxt-locally/cclelndahbckbenkjhflpdbgdldlbecc")
    print("      Firefox: https://addons.mozilla.org/en-US/firefox/addon/cookies-txt/")
    print("   2. Go to youtube.com and make sure you're logged in")
    print("   3. Click the extension icon → Export cookies")
    print("   4. Save as 'youtube_cookies.txt' (or any name)")
    print("   5. Upload it when prompted below\n")

    upload_choice = input("Upload YouTube cookies? (y/n): ").strip().lower()

    if upload_choice != 'y':
        print("⏭️  Skipping cookie upload - YouTube downloads will be skipped\n")
        return None

    print("\n📤 Upload your cookies file now...")
    try:
        from google.colab import files
        uploaded = files.upload()

        if not uploaded:
            print("❌ No file uploaded")
            return None

        # Get the filename (whatever user named it)
        cookie_filename = list(uploaded.keys())[0]
        cookie_path = f"/content/{cookie_filename}"

        # Verify it's a valid cookies file
        if not os.path.exists(cookie_path):
            print(f"❌ File not found: {cookie_path}")
            return None

        # Check if it looks like a cookies file
        with open(cookie_path, 'r') as f:
            first_line = f.readline()
            if not ('youtube.com' in first_line or 'Netscape' in first_line or first_line.startswith('#')):
                print("⚠️  Warning: This doesn't look like a Netscape cookies file")
                print("   Continuing anyway...\n")

        print(f"✅ Cookies loaded: {cookie_filename}")
        print(f"📁 Path: {cookie_path}\n")

        return cookie_path

    except Exception as e:
        print(f"❌ Cookie upload failed: {str(e)}")
        return None


class GeminiValidator:
    """
    Validates match information and finds videos using Gemini
    """

    def __init__(self, api_key: str, raw_scripts_dir: str):
        from google import genai
        from google.genai import types

        self.client = genai.Client(api_key=api_key)
        self.raw_scripts_dir = Path(raw_scripts_dir)
        self.MODEL_ID = "gemini-2.0-flash-exp"
        self.types = types

    def get_raw_script_path(self, video_plan_name: str) -> Optional[Path]:
        """
        Get raw script path from video plan name

        Args:
            video_plan_name: e.g., "Fergie Time_video_plan.txt"

        Returns:
            Path to raw script or None
        """
        # Remove _video_plan suffix
        if video_plan_name.endswith('_video_plan.txt'):
            base_name = video_plan_name.replace('_video_plan.txt', '.txt')
        elif video_plan_name.endswith('.txt'):
            base_name = video_plan_name
        else:
            base_name = f"{video_plan_name}.txt"

        raw_script_path = self.raw_scripts_dir / base_name

        if raw_script_path.exists():
            return raw_script_path
        else:
            print(f"   ⚠️ Raw script not found: {raw_script_path}")
            return None

    def tier1_validate(self, teams: List[str], date: str, team_channels: dict) -> str:
        """
        Tier 1: FREE validation (no API)
        Quick checks without calling Gemini

        Returns:
            "high" or "low" confidence
        """
        confidence = "high"

        # Check if teams are in known channels
        team1_known = any(team.lower() in team_channels for team in teams[0].lower().split())
        team2_known = any(team.lower() in team_channels for team in teams[1].lower().split()) if len(teams) > 1 else False

        if not team1_known and not team2_known:
            confidence = "low"
            print(f"   ⚠️ Tier 1: Unknown teams detected")

        # Check date format
        if date == "unknown" or not date:
            confidence = "low"
            print(f"   ⚠️ Tier 1: Invalid date")

        # Check for player names contamination (common player surnames)
        common_players = ['reyes', 'neville', 'cole', 'campbell', 'rooney', 'fabregas',
                          'vieira', 'keane', 'carragher', 'wilshere', 'palmer', 'gusto']
        team_words = ' '.join(teams).lower()
        if any(player in team_words for player in common_players):
            confidence = "low"
            print(f"   ⚠️ Tier 1: Possible player name in team name")

        if confidence == "high":
            print(f"   ✅ Tier 1: High confidence - teams and date look valid")

        return confidence

    def tier2_validate(self, description: str, raw_script_path: Optional[Path],
                       match_tracker) -> Dict[str, any]:
        """
        Tier 2: LIGHT validation (cheap API call)
        Validate match info only, no video search, no Google grounding

        Returns:
            Dict with corrected match info
        """
        print(f"   🔍 Tier 2: Light validation (correcting match info)...")

        # Read raw script if available
        context_text = ""
        if raw_script_path and raw_script_path.exists():
            with open(raw_script_path, 'r', encoding='utf-8') as f:
                context_text = f.read()[:2000]  # First 2000 chars for context

        prompt = f"""You are a football match validator.

MATCH DESCRIPTION: "{description}"

CONTEXT FROM SCRIPT:
{context_text if context_text else "No context available"}

TASK:
1. Extract the correct team names (remove any player names)
2. Verify the date format
3. Verify the score if mentioned

Return ONLY a JSON object with this exact format:
{{
  "teams": ["Team A", "Team B"],
  "date": "Month Day Year",
  "score": "X-X",
  "confidence": "high/medium/low"
}}

Example: {{"teams": ["Manchester United", "Arsenal"], "date": "October 24 2004", "score": "0-0", "confidence": "high"}}

Be concise. Return ONLY the JSON, no explanation."""

        try:
            response = self.client.models.generate_content(
                model=self.MODEL_ID,
                contents=prompt,
                config=self.types.GenerateContentConfig(
                    temperature=0.0,
                    response_mime_type="application/json"
                )
            )

            import json
            print(f"   🔍 DEBUG: Gemini response length: {len(response.text)}")
            print(f"   🔍 DEBUG: First 200 chars: {response.text[:200]}")

            try:
                result = json.loads(response.text)
            except json.JSONDecodeError as e:
                print(f"   ❌ JSON decode error: {e}")
                print(f"   📄 Full response: {response.text}")
                return None

            print(f"   ✅ Tier 2: Corrected to {result['teams'][0]} vs {result['teams'][1]}, {result['date']}, {result.get('score', 'N/A')}")

            return result

        except Exception as e:
            print(f"   ❌ Tier 2 validation failed: {str(e)[:100]}")
            # Fallback to original extraction
            teams = match_tracker._extract_teams(description)
            date = match_tracker._extract_date(description)
            score = match_tracker._extract_score(description)
            return {
                "teams": list(teams),
                "date": date,
                "score": score,
                "confidence": "low"
            }

    def tier3_validate(self, description: str, raw_script_path: Optional[Path],
                       corrected_info: Dict) -> Optional[Dict]:
        """
        Tier 3: FULL validation (expensive - with Google Search grounding)
        Search for video on YouTube with Gemini

        Returns:
            Video info dict or None
        """
        print(f"   🔍 Tier 3: Full Gemini search with grounding...")

        # Read raw script
        context_text = ""
        if raw_script_path and raw_script_path.exists():
            with open(raw_script_path, 'r', encoding='utf-8') as f:
                context_text = f.read()

        teams = corrected_info.get('teams', [])
        date = corrected_info.get('date', '')
        score = corrected_info.get('score', '')
        mandatory_search_query = f"{teams[0]} vs {teams[1]} {date} scoreline"

        prompt = f"""You are a football highlight video finder with match verification.

USER'S REQUEST:
Teams: {teams[0]} vs {teams[1]}
Date: {date}
Claimed Score: {score}

YOUR PRIMARY GOAL: Find the best YouTube highlight video for this match.

STEP 1 - VERIFY MATCH (Quick Google check):
Search: "{mandatory_search_query}"
Find the ACTUAL score from a reliable source.
If match doesn't exist or score is different, note it AND use the correct version.

STEP 2 - FIND VIDEO (Main task):
A) Search official YouTube channels:
   - {teams[0]} official channel
   - {teams[1]} official channel
   Search for: "highlights {date}" or "{teams[0]} vs {teams[1]}"

B) If not found on official channels:
   - Do broad YouTube search: "{teams[0]} {teams[1]} {date} highlights"
   - Look for high-quality highlight videos (avoid interviews, reactions)

C) Return the BEST video you find (prefer official channels, 2-10 min duration)

REASONING FORMAT:
Line 1: "Match verification: [Google result - actual score]"
Line 2: "Score check: USER CLAIMED {score}, ACTUAL WAS X-X" OR "Scores match" OR "Match not found but searching anyway"
Line 3: "Video search: Searched [{teams[0]}] official → [result]"
Line 4: "Video search: Searched [{teams[1]}] official → [result]"
Line 5: "Final result: [Video found/not found - source]"

EXAMPLE (video found, score different):
"Match verification: BBC Sport reports Chelsea 2-2 Newcastle on Dec 22, 2025
Score check: USER CLAIMED 5-0 on Dec 20, ACTUAL WAS 2-2 on Dec 22
Video search: Searched Newcastle United official channel → Found 'Newcastle 2-2 Chelsea Extended Highlights'
Video search: Searched Chelsea official channel → No Dec 22 highlight found
Final result: Video found on Newcastle channel (extended highlights, 7min)"

EXAMPLE (no video, match verified):
"Match verification: Premier League site confirms Chelsea 2-2 Newcastle Dec 22
Score check: USER CLAIMED 5-0 on Dec 20, ACTUAL WAS 2-2 on Dec 22
Video search: Searched Newcastle official → No highlight video
Video search: Searched Chelsea official → No highlight video
Final result: No highlight video available on official channels or broad search"

EXAMPLE (no match exists):
"Match verification: No Chelsea vs Newcastle match on December 20, 2025
Score check: Match did not occur on this date
Video search: Searched for any Chelsea-Newcastle December 2025 videos → None
Video search: Broadened search to Q4 2025 → Found unrelated matches only
Final result: No video found (match didn't happen)"

Return JSON:
{{
  "status": "verified/corrected/not_found",
  "match_confirmed": "Actual match details from Google",
  "video_url": "https://youtube.com/watch?v=..." or null,
  "video_title": "Full video title" or null,
  "source": "official_channel/broad_search/not_found",
  "reasoning": "Follow 5-line format: verification + score check + 2 channel searches + final result"
}}

REMEMBER: Finding the video is VERY IMPORTANT"""


        try:
            response = self.client.models.generate_content(
                model=self.MODEL_ID,
                contents=prompt,
                config=self.types.GenerateContentConfig(
                    tools=[self.types.Tool(google_search=self.types.GoogleSearch())],
                    temperature=0.0,
                )
            )

            import json
            print(f"   🔍 DEBUG Tier 3: Response length = {len(response.text)}")
            print(f"   🔍 DEBUG Tier 3: First 500 chars = {response.text[:500]}")

            # Parse response text as JSON
            response_text = response.text.strip()

            # Remove markdown fences
            if response_text.startswith('```'):
                response_text = response_text.split('\n', 1)[1] if '\n' in response_text else response_text[3:]
                if response_text.endswith('```'):
                    response_text = response_text.rsplit('```', 1)[0]
                response_text = response_text.strip()

            # Handle multiple JSON objects
            if response_text.count('{') > 1:
                # Extract first complete JSON
                brace_count = 0
                end_pos = 0
                for i, char in enumerate(response_text):
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            end_pos = i + 1
                            break
                if end_pos > 0:
                    response_text = response_text[:end_pos]

            # Clean the JSON text
            import re

            # Replace unescaped newlines in string values
            response_text = re.sub(r'(?<!\\)\n', ' ', response_text)

            # Remove any control characters
            response_text = ''.join(char for char in response_text if ord(char) >= 32 or char in '\n\r\t')

            try:
                result = json.loads(response_text)
            except json.JSONDecodeError as e:
                print(f"   ❌ JSON parse error: {e}")
                print(f"   📄 Cleaned response: {response_text[:500]}")
                return None

            if result.get('video_title'):
                print(f"   ✅ Tier 3 Validation Complete:")
                print(f"   📊 Status: {result.get('status', 'unknown')}")
                print(f"   ⚽ Match confirmed: {result.get('match_confirmed', 'N/A')}")
                print(f"   💡 Reasoning: {result.get('reasoning', 'No reasoning provided')}")
                print(f"   🎬 Video title from Gemini: {result['video_title'][:80]}")
                print(f"   🔍 Now searching YouTube for this exact video...")

                # Search YouTube for the video title Gemini found
                try:
                    search_opts = {
                        'quiet': True,
                        'extract_flat': 'in_playlist',
                        'http_headers': {
                            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                        }
                    }

                    if hasattr(self, 'cookies_path') and self.cookies_path:
                        search_opts['cookiefile'] = self.cookies_path

                    # Search for exact video title
                    search_query = result['video_title']

                    with yt_dlp.YoutubeDL(search_opts) as ydl:
                        search_result = ydl.extract_info(f"ytsearch1:{search_query}", download=False)

                        if search_result and 'entries' in search_result and search_result['entries']:
                            video = search_result['entries'][0]
                            video_url = f"https://www.youtube.com/watch?v={video['id']}"

                            print(f"   ✅ Found video on YouTube!")
                            print(f"   📺 Title: {video.get('title', '')[:60]}")
                            print(f"   🔗 URL: {video_url}")

                            return {
                                'url': video_url,
                                'title': video.get('title', result['video_title']),
                                'duration': video.get('duration', 0),
                                'upload_date': video.get('upload_date', ''),
                                'channel': 'gemini_tier3'
                            }
                        else:
                            print(f"   ⚠️ Could not find video on YouTube with that title")
                            return None

                except Exception as e:
                    print(f"   ❌ YouTube search failed: {str(e)[:100]}")
                    return None
            else:
                print(f"   ❌ Tier 3: {result.get('reasoning', 'No video found')}")
                return None

        except Exception as e:
            print(f"   ❌ Tier 3 failed: {str(e)[:100]}")
            return None

# ========================================
# GEO-BLOCK MANAGER
# ========================================
class GeoBlockManager:
    """Manage geo-blocked channels list"""

    def __init__(self, drive_path="/content/drive/MyDrive/Angry Dude Scripts/geoblocked_channels.json"):
        self.file_path = drive_path
        self.blocked_channels = self._load()

        # Pre-populate known blocked channels
        if "@tntsportsfootball" not in self.blocked_channels:
            self.add_blocked_channel("@tntsportsfootball", "Pre-known geo-blocked channel")

    def _load(self) -> Dict:
        """Load blocked channels from Drive"""
        if os.path.exists(self.file_path):
            with open(self.file_path, 'r') as f:
                return json.load(f)
        return {}

    def _save(self):
        """Save blocked channels to Drive"""
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
        with open(self.file_path, 'w') as f:
            json.dump(self.blocked_channels, f, indent=2)

    def is_blocked(self, channel_handle: str) -> bool:
        """Check if channel is geo-blocked"""
        return channel_handle in self.blocked_channels

    def add_blocked_channel(self, channel_handle: str, error_msg: str):
        """Add channel to blocked list"""
        if channel_handle not in self.blocked_channels:
            self.blocked_channels[channel_handle] = {
                'blocked': True,
                'detected_date': datetime.now().isoformat(),
                'error': error_msg
            }
            self._save()
            print(f"⛔ Added {channel_handle} to geo-block list")

    def get_blocked_list(self) -> List[str]:
        """Get list of blocked channel handles"""
        return list(self.blocked_channels.keys())

class VideoDiscoveryProgress:
    """Track video discovery progress with backup"""

    def __init__(self, script_name: str, base_dir: str = "/content/drive/MyDrive/Angry Dude Scripts/collected_assets"):
        self.script_name = script_name
        self.output_dir = Path(base_dir) / script_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Two progress files for redundancy
        self.progress_file = self.output_dir / "video_discovery_progress.json"
        self.backup_file = self.output_dir / "video_discovery_progress_backup.json"

        self.data = self._load()

    def _load(self) -> dict:
        """Load from main file, fallback to backup if corrupted"""
        # Try main file first
        if self.progress_file.exists():
            try:
                with open(self.progress_file, 'r') as f:
                    data = json.load(f)
                    print(f"📊 Loaded video discovery progress: {len(data.get('completed', {}))} sections completed")
                    return data
            except:
                print(f"⚠️ Main progress file corrupted, trying backup...")

        # Try backup file
        if self.backup_file.exists():
            try:
                with open(self.backup_file, 'r') as f:
                    data = json.load(f)
                    print(f"📊 Loaded from backup: {len(data.get('completed', {}))} sections completed")
                    # Restore main file from backup
                    self._save(data)
                    return data
            except:
                print(f"⚠️ Backup file also corrupted, starting fresh")

        return {
            'script_name': self.script_name,
            'completed': {},  # Key: line_number -> VideoDiscoveryResult dict
            'failed': {},     # Key: line_number -> reason
            'total_sections': 0,
            'last_updated': datetime.now().isoformat()
        }

    def _save(self, data=None):
        """Save to both main and backup files"""
        if data is None:
            data = self.data

        data['last_updated'] = datetime.now().isoformat()

        # Save to main file
        with open(self.progress_file, 'w') as f:
            json.dump(data, f, indent=2)

        # Save to backup file
        with open(self.backup_file, 'w') as f:
            json.dump(data, f, indent=2)

    def is_completed(self, line_number: int) -> bool:
        """Check if line already has video discovered"""
        return str(line_number) in self.data['completed']

    def is_failed(self, line_number: int) -> bool:
        """Check if line previously failed"""
        return str(line_number) in self.data['failed']

    def get_result(self, line_number: int) -> Optional[dict]:
        """Get discovered video for a line"""
        return self.data['completed'].get(str(line_number))

    def mark_completed(self, line_number: int, discovery_result: dict):
        """Mark line as completed with video info"""
        self.data['completed'][str(line_number)] = {
            'video_url': discovery_result.get('url'),
            'title': discovery_result.get('title'),
            'channel': discovery_result.get('channel'),
            'duration': discovery_result.get('duration', 0),  # Add
            'upload_date': discovery_result.get('upload_date', ''),  # Add
            'search_method': discovery_result.get('search_method', 'unknown'),
            'is_player_highlight': discovery_result.get('is_player_highlight', False),  # Add
            'timestamp': datetime.now().isoformat()
        }
        self._save()
        print(f"   ✅ Saved discovery for line {line_number}")

    def mark_failed(self, line_number: int, reason: str):
        """Mark line as failed"""
        self.data['failed'][str(line_number)] = {
            'reason': reason,
            'timestamp': datetime.now().isoformat()
        }
        self._save()
        print(f"   ❌ Marked line {line_number} as failed: {reason}")

    def get_stats(self) -> dict:
        """Get progress statistics"""
        return {
            'total': self.data.get('total_sections', 0),
            'completed': len(self.data['completed']),
            'failed': len(self.data['failed']),
            'remaining': self.data.get('total_sections', 0) - len(self.data['completed']) - len(self.data['failed'])
        }

class VideoProcessingCache:
    """
    Comprehensive cache for video downloads and analysis results
    Saves everything to Drive for persistence across runtime disconnects
    """

    def __init__(self, script_name: str, base_dir: str = "/content/drive/MyDrive/Angry Dude Scripts"):
        self.script_name = script_name
        self.base_dir = Path(base_dir)

        # Cache directories
        self.cache_dir = self.base_dir / "cache" / script_name
        self.videos_dir = self.cache_dir / "videos"
        self.analysis_dir = self.cache_dir / "analysis"
        self.keyframes_dir = self.cache_dir / "keyframes"

        # Create all directories
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.videos_dir.mkdir(exist_ok=True)
        self.analysis_dir.mkdir(exist_ok=True)
        self.keyframes_dir.mkdir(exist_ok=True)

        # Cache index files
        self.video_cache_file = self.cache_dir / "video_downloads.json"
        self.analysis_cache_file = self.cache_dir / "video_analysis.json"

        # Load caches
        self.video_cache = self._load_json(self.video_cache_file)
        self.analysis_cache = self._load_json(self.analysis_cache_file)

        print(f"📦 Cache initialized: {len(self.video_cache)} videos, {len(self.analysis_cache)} analyses")

    def _load_json(self, filepath: Path) -> dict:
        """Load JSON file or return empty dict"""
        if filepath.exists():
            try:
                with open(filepath, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}

    def _save_json(self, filepath: Path, data: dict):
        """Save dict to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    # ========== VIDEO DOWNLOAD CACHE ==========

    def get_video_path(self, video_url: str) -> Optional[str]:
        """
        Get cached video path if exists
        Returns path if video is already downloaded, None otherwise
        """
        if video_url in self.video_cache:
            cached_path = self.video_cache[video_url]['path']
            if os.path.exists(cached_path):
                print(f"   ♻️ Using cached video: {Path(cached_path).name}")
                return cached_path
            else:
                # Cache entry exists but file missing - remove from cache
                del self.video_cache[video_url]
                self._save_json(self.video_cache_file, self.video_cache)
        return None

    def cache_video(self, video_url: str, source_path: str) -> str:
        """
        Copy video to cache and return cached path
        """
        # Generate safe filename from URL
        import hashlib
        url_hash = hashlib.md5(video_url.encode()).hexdigest()[:12]
        filename = f"{url_hash}.mp4"
        cached_path = self.videos_dir / filename

        # Copy to cache
        shutil.copy(source_path, cached_path)

        # Update cache index
        self.video_cache[video_url] = {
            'path': str(cached_path),
            'original_url': video_url,
            'cached_at': datetime.now().isoformat(),
            'file_size': os.path.getsize(cached_path)
        }
        self._save_json(self.video_cache_file, self.video_cache)

        print(f"   💾 Cached video: {filename}")
        return str(cached_path)

    def get_all_cached_videos(self) -> dict:
        """Get all cached video URLs and their paths"""
        valid_cache = {}
        for url, info in self.video_cache.items():
            if os.path.exists(info['path']):
                valid_cache[url] = info['path']
        return valid_cache

    # ========== VIDEO ANALYSIS CACHE ==========

    def get_analysis(self, video_url: str, section_line: int) -> Optional[dict]:
        """
        Get cached analysis results for a video section
        Returns cached analysis if exists, None otherwise
        """
        cache_key = f"{video_url}_{section_line}"

        if cache_key in self.analysis_cache:
            cached = self.analysis_cache[cache_key]

            # Verify all cached files exist
            files_exist = True
            for key in ['keyframes_path', 'event_timeline_path', 'exact_moment_path']:
                if key in cached and not os.path.exists(cached[key]):
                    files_exist = False
                    break

            if files_exist:
                print(f"   ♻️ Using cached analysis (keyframes, timeline, moment)")
                return cached
            else:
                # Cache corrupted, remove entry
                del self.analysis_cache[cache_key]
                self._save_json(self.analysis_cache_file, self.analysis_cache)

        return None

    def cache_analysis(self, video_url: str, section_line: int,
                      keyframes: list, event_timeline: list, exact_moment: dict) -> dict:
        """
        Cache video analysis results (keyframes, timeline, exact moment)
        Returns paths to cached files
        """
        cache_key = f"{video_url}_{section_line}"

        # Save keyframes list
        keyframes_file = self.keyframes_dir / f"keyframes_{section_line:03d}.json"
        with open(keyframes_file, 'w') as f:
            json.dump(keyframes, f, indent=2)

        # Save event timeline
        timeline_file = self.analysis_dir / f"timeline_{section_line:03d}.json"
        with open(timeline_file, 'w') as f:
            json.dump(event_timeline, f, indent=2)

        # Save exact moment
        moment_file = self.analysis_dir / f"moment_{section_line:03d}.json"
        with open(moment_file, 'w') as f:
            json.dump(exact_moment, f, indent=2)

        # Update cache index
        cached_data = {
            'video_url': video_url,
            'section_line': section_line,
            'keyframes_path': str(keyframes_file),
            'event_timeline_path': str(timeline_file),
            'exact_moment_path': str(moment_file),
            'cached_at': datetime.now().isoformat()
        }

        self.analysis_cache[cache_key] = cached_data
        self._save_json(self.analysis_cache_file, self.analysis_cache)

        print(f"   💾 Cached analysis results")
        return cached_data

    def load_cached_analysis(self, cached_data: dict) -> tuple:
        """
        Load analysis results from cache
        Returns (keyframes, event_timeline, exact_moment)
        """
        with open(cached_data['keyframes_path'], 'r') as f:
            keyframes = json.load(f)

        with open(cached_data['event_timeline_path'], 'r') as f:
            event_timeline = json.load(f)

        with open(cached_data['exact_moment_path'], 'r') as f:
            exact_moment = json.load(f)

        return keyframes, event_timeline, exact_moment

    # ========== CLEANUP ==========

    def cleanup(self):
        """Delete all cache files and directories"""
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            print(f"   ✅ Deleted cache directory: {self.cache_dir}")

    def get_cache_size(self) -> dict:
        """Get cache statistics"""
        total_size = 0
        video_count = 0
        analysis_count = 0

        # Count videos
        if self.videos_dir.exists():
            for f in self.videos_dir.glob("*.mp4"):
                total_size += f.stat().st_size
                video_count += 1

        # Count analysis files
        if self.analysis_dir.exists():
            for f in self.analysis_dir.glob("*.json"):
                total_size += f.stat().st_size
                analysis_count += 1

        return {
            'total_size_mb': total_size / (1024 * 1024),
            'video_count': video_count,
            'analysis_count': analysis_count
        }


# ========================================
# ESPN MATCH DATA SCRAPER
# ========================================
class ESPNMatchScraper:
    """Scrape match data from ESPN API for goal context"""

    LEAGUE_IDS = {
        'premier league': 'eng.1',
        'la liga': 'esp.1',
        'bundesliga': 'ger.1',
        'serie a': 'ita.1',
        'ligue 1': 'fra.1',
        'champions league': 'uefa.champions',
        'europa league': 'uefa.europa',
    }

    def __init__(self):
        self.base_url = "https://site.api.espn.com/apis/site/v2/sports/soccer"

    def get_match_data(self, team1: str, team2: str, match_date: str) -> Optional[Dict]:
        """
        Get match data from ESPN API

        Args:
            team1: First team name (e.g., "Aston Villa")
            team2: Second team name (e.g., "Arsenal")
            match_date: Date string (e.g., "december 6 2025")

        Returns:
            Dict with goals and match info, or None if not found
        """
        print(f"🔍 Searching ESPN for: {team1} vs {team2} on {match_date}")

        # Parse date
        try:
            if isinstance(match_date, str):
                try:
                    date_obj = datetime.strptime(match_date, "%Y-%m-%d")
                except:
                    try:
                        date_obj = datetime.strptime(match_date.lower(), "%B %d %Y")
                    except:
                        date_obj = datetime.strptime(match_date.lower().replace(',', ''), "%B %d %Y")
        except:
            print(f"   ❌ Could not parse date: {match_date}")
            return None

        # Try Premier League first (most common), then others
        for league_name, league_id in self.LEAGUE_IDS.items():
            try:
                event_id = self._find_event_id(league_id, team1, team2, date_obj)
                if event_id:
                    print(f"   ✅ Found in {league_name.title()}")
                    return self._get_match_summary(league_id, event_id)
            except Exception as e:
                continue

        print(f"   ⚠️ Match not found on ESPN")
        return None

    def _find_event_id(self, league_id: str, team1: str, team2: str, date_obj: datetime) -> Optional[str]:
        """Find event ID by searching scoreboard for date"""

        # Format date for ESPN API (YYYYMMDD)
        date_str = date_obj.strftime("%Y%m%d")

        url = f"{self.base_url}/{league_id}/scoreboard?dates={date_str}"

        try:
            response = requests.get(url, timeout=10)
            if response.status_code != 200:
                return None

            data = response.json()

            if 'events' not in data or not data['events']:
                return None

            # Normalize team names for matching
            team1_norm = team1.lower().replace(' ', '')
            team2_norm = team2.lower().replace(' ', '')

            # Search through events
            for event in data['events']:
                try:
                    competitors = event['competitions'][0]['competitors']

                    teams_in_event = []
                    for comp in competitors:
                        team_name = comp['team']['displayName'].lower().replace(' ', '')
                        teams_in_event.append(team_name)

                    # Check if both teams are in this event
                    if team1_norm in teams_in_event[0] or team1_norm in teams_in_event[1]:
                        if team2_norm in teams_in_event[0] or team2_norm in teams_in_event[1]:
                            return event['id']

                except:
                    continue

            return None

        except Exception as e:
            return None

    def _get_match_summary(self, league_id: str, event_id: str) -> Dict:
        """Get match summary with goals"""

        url = f"{self.base_url}/{league_id}/summary?event={event_id}"

        try:
            response = requests.get(url, timeout=10)
            if response.status_code != 200:
                return None

            data = response.json()
            # DEBUG: Print raw status data
            print(f"   🔍 DEBUG: Full status data:")
            print(f"      {json.dumps(data['header']['competitions'][0]['status'], indent=2)}")

            # Check if match is finished
            status = data['header']['competitions'][0]['status']['type']['name']
            completed = data['header']['competitions'][0]['status']['type']['completed']

            print(f"   📊 Match status: {status}, Completed: {completed}")

            if not completed:
                print(f"   ⚠️ Match not finished yet")
                return None

            # Extract teams and score
            competitors = data['header']['competitions'][0]['competitors']
            home_team = [t for t in competitors if t['homeAway'] == 'home'][0]
            away_team = [t for t in competitors if t['homeAway'] == 'away'][0]

            home_name = home_team['team']['displayName']
            away_name = away_team['team']['displayName']
            home_score = int(home_team['score'])
            away_score = int(away_team['score'])

            # Extract goals from commentary
            goals = []
            if 'commentary' in data:
                for comment in data['commentary']:
                    if comment.get('scoringPlay'):
                        time_str = comment['clock']['displayValue']
                        text = comment.get('text', '')

                        # Parse scorer from text (simple extraction)
                        scorer = self._extract_scorer(text)

                        # Determine which team scored
                        team = None
                        if any(name.lower() in text.lower() for name in [home_name.split()[-1]]):
                            team = home_name
                        elif any(name.lower() in text.lower() for name in [away_name.split()[-1]]):
                            team = away_name

                        goals.append({
                            'time': time_str,
                            'scorer': scorer,
                            'team': team,
                            'text': text
                        })

            result = {
                'home_team': home_name,
                'away_team': away_name,
                'home_score': home_score,
                'away_score': away_score,
                'goals': goals
            }

            print(f"   📊 {home_name} {home_score}-{away_score} {away_name}")
            print(f"   ⚽ Goals: {len(goals)}")
            for g in goals:
                print(f"      • {g['time']} - {g['scorer']} ({g['team']})")
            print()

            return result

        except Exception as e:
            print(f"   ❌ Error getting match summary: {str(e)[:100]}")
            return None

    def _extract_scorer(self, text: str) -> str:
        """Extract scorer name from commentary text"""
        # Simple extraction - get first capitalized name
        import re

        # Look for patterns like "Name scores" or "Name goal"
        match = re.search(r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+(?:scores|goal)', text)
        if match:
            return match.group(1)

        # Fallback: get first capitalized word
        words = text.split()
        for word in words:
            if word and word[0].isupper() and len(word) > 2:
                return word

        return "Unknown"

    def build_goal_context(self, match_data: Dict, target_description: str) -> str:
        """
        Build concise goal context for Gemini

        Args:
            match_data: Dict from get_match_data()
            target_description: Section description (e.g., "Buendía goal Aston Villa vs Arsenal 2-1")

        Returns:
            Concise context string for Gemini
        """
        if not match_data:
            return ""

        # Extract target team and scorer from description
        desc_lower = target_description.lower()

        # Find which team we're looking for
        home_team = match_data['home_team']
        away_team = match_data['away_team']

        target_team = None
        if home_team.lower() in desc_lower:
            target_team = home_team
        elif away_team.lower() in desc_lower:
            target_team = away_team

        if not target_team:
            return ""

        # Filter goals by target team
        team_goals = [g for g in match_data['goals'] if g['team'] == target_team]

        if not team_goals:
            return ""

        # Build context
        total_goals = len(team_goals)

        # Try to identify which goal number this is
        goal_number = None
        for i, goal in enumerate(team_goals, 1):
            if goal['scorer'].lower() in desc_lower:
                goal_number = i
                break

        if not goal_number:
            # Default to last goal if scorer not found (often the winner)
            goal_number = total_goals

        target_goal = team_goals[goal_number - 1]

        context = f"""TARGET GOAL (from official match data):
- Scorer: {target_goal['scorer']} ({target_team})
- This is {target_team}'s goal #{goal_number} of {total_goals}
- Timing: Near the {"START" if goal_number == 1 else "MIDDLE" if goal_number < total_goals else "END"} of the match
- Final score: {match_data['home_team']} {match_data['home_score']}-{match_data['away_score']} {match_data['away_team']}

Look for this specific goal, not other goals in the match."""

        return context


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

    def mark_section_started(self, section: PNGSequenceSection):
        """Mark section as started"""
        section_key = f"line_{section.line_number:03d}"
        self.data['sections'][section_key] = {
            'status': 'started',
            'description': section.description,
            'transcript': section.transcript,
            'started_at': datetime.now().isoformat(),
            'attempts': []
        }
        self._save()

    def mark_section_completed(self, section: PNGSequenceSection, metadata: Dict):
        """Mark section as completed"""
        section_key = f"line_{section.line_number:03d}"
        if section_key in self.data['sections']:
            self.data['sections'][section_key]['status'] = 'completed'
            self.data['sections'][section_key]['completed_at'] = datetime.now().isoformat()
            self.data['sections'][section_key]['metadata'] = metadata
            self.data['stats']['completed'] += 1
        self._save()

    def mark_section_failed(self, section: PNGSequenceSection, error: str):
        """Mark section as failed"""
        section_key = f"line_{section.line_number:03d}"
        if section_key in self.data['sections']:
            self.data['sections'][section_key]['status'] = 'failed'
            self.data['sections'][section_key]['error'] = error
            self.data['sections'][section_key]['failed_at'] = datetime.now().isoformat()
            self.data['stats']['failed'] += 1
        self._save()

    def log_attempt(self, section: PNGSequenceSection, attempt_data: Dict):
        """Log an attempt for a section"""
        section_key = f"line_{section.line_number:03d}"
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

    def get_section_stage(self, section_key: str) -> str:
        """Get current stage of section"""
        if section_key not in self.data['sections']:
            return 'not_started'
        return self.data['sections'][section_key].get('stage', 'not_started')

    def update_section_stage(self, section: PNGSequenceSection, stage: str, metadata: Dict = None):
        """Update section to a specific stage"""
        section_key = f"line_{section.line_number:03d}"

        if section_key not in self.data['sections']:
            self.mark_section_started(section)

        self.data['sections'][section_key]['stage'] = stage
        self.data['sections'][section_key]['last_updated'] = datetime.now().isoformat()

        if metadata:
            if 'stage_metadata' not in self.data['sections'][section_key]:
                self.data['sections'][section_key]['stage_metadata'] = {}
            self.data['sections'][section_key]['stage_metadata'][stage] = metadata

        self._save()




class MatchContextTracker:
    """Track match context across consecutive sections"""

    def __init__(self):
        self.match_contexts = {}  # line_number -> MatchContext
        self.current_match = None

    def extract_match_context(self, section: PNGSequenceSection, selected_images_metadata: Dict) -> MatchContext:
        """Extract match context from section and selected images"""
        # Parse teams from description
        teams = self._extract_teams(section.description)

        # Parse date
        date = self._extract_date(section.description)

        # Get jersey colors and stadium from image metadata
        jersey_colors = selected_images_metadata.get('jersey_colors', {})

        return MatchContext(
            teams=teams,
            date=date,
            home_jersey_color=jersey_colors.get('home', 'unknown'),
            away_jersey_color=jersey_colors.get('away', 'unknown'),
            stadium=selected_images_metadata.get('stadium', 'unknown'),
            competition=selected_images_metadata.get('competition', 'unknown'),
            last_section_line=section.line_number
        )

    def _extract_teams(self, description: str) -> List[str]:
        """Extract team names from description"""
        import re

        # First try: Look for "Team1 vs Team2" or "Team1 v Team2" pattern
        vs_pattern = r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:vs\.?|v)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)'
        match = re.search(vs_pattern, description)

        if match:
            team1 = match.group(1).strip()
            team2 = match.group(2).strip()
            return [team1, team2]

        # Fallback: Use hardcoded patterns for known teams
        team_patterns = [
            'real madrid', 'barcelona', 'barca', 'liverpool', 'arsenal',
            'chelsea', 'manchester united', 'man united', 'manchester city',
            'man city', 'bayern', 'psg', 'atletico', 'juventus', 'aston villa'
        ]

        description_lower = description.lower()
        found_teams = []

        for team in team_patterns:
            if team in description_lower:
                found_teams.append(team)
                if len(found_teams) == 2:
                    break

        return found_teams

    def _extract_date(self, description: str) -> str:
        """Extract date from description"""
        import re

        # Pattern: Month Day, Year or Month Year
        patterns = [
            r'(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2},?\s+\d{4}',
            r'(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{4}',
        ]

        for pattern in patterns:
            match = re.search(pattern, description.lower())
            if match:
                return match.group(0)

        return "unknown"

    def should_use_match_context(self, section: PNGSequenceSection) -> bool:
        """Check if we should use previous match context"""
        if not self.current_match:
            return False

        # Check if this section is consecutive (within 3 lines)
        if section.line_number - self.current_match.last_section_line > 3:
            return False

        # Check if section mentions different teams
        current_teams = self._extract_teams(section.description)
        if current_teams and not any(team in self.current_match.teams for team in current_teams):
            return False

        return True

    def get_context_for_section(self, section: PNGSequenceSection) -> Optional[MatchContext]:
        """Get match context for current section"""
        if self.should_use_match_context(section):
            return self.current_match
        return None

    def update_context(self, section: PNGSequenceSection, context: MatchContext):
        """Update current match context"""
        self.match_contexts[section.line_number] = context
        self.current_match = context

    def clear_context(self):
        """Clear current match context (different topic detected)"""
        self.current_match = None

    def _extract_score(self, description: str) -> Optional[str]:
        """
        Extract score from description
        Examples: "2-0", "3-1", "1-1"

        Returns:
            Score string like "2-0" or None
        """
        import re

        # Pattern: digit-digit (with optional spaces)
        score_pattern = r'\b(\d+)\s*-\s*(\d+)\b'
        match = re.search(score_pattern, description)

        if match:
            return f"{match.group(1)}-{match.group(2)}"

        return None


class OfficialChannelVideoDownloader:
    """
    Downloads match highlights from official team YouTube channels.
    Uses yt-dlp with the exact configuration that successfully downloads videos.
    """

    TEAM_CHANNELS = {
        # ===== PREMIER LEAGUE (20 Teams) =====
        "arsenal": "@arsenal",
        "aston villa": "@avfcofficial",
        "bournemouth": "@AFCBournemouth",
        "brentford": "@BrentfordFC",
        "brighton": "@officialbhafc",
        "chelsea": "@chelseafc",
        "crystal palace": "@OfficialCPFC",
        "everton": "@everton",
        "fulham": "@fulhamfc",
        "ipswich town": "@IpswichTownFC",
        "leicester city": "@lcfcofficial",
        "liverpool": "@liverpoolfc",
        "manchester city": "@mancity",
        "manchester united": "@manutd",
        "newcastle united": "@nufc",
        "nottingham forest": "@NottinghamForestFC",
        "southampton": "@southamptonfc",
        "tottenham hotspur": "@spursofficial",
        "west ham united": "@westhamunited",
        "wolves": "@wolves",

        # ===== LA LIGA (20 Teams) =====
        "alaves": "@deportivoalaves",
        "athletic bilbao": "@AthleticClubTV",
        "atletico madrid": "@atleticodemadrid",
        "barcelona": "@fcbarcelona",
        "celta vigo": "@rccelta",
        "getafe": "@GetafeCFmedia",
        "girona": "@GironaFC",
        "las palmas": "@UDLasPalmasOficial",
        "leganes": "@CDLeganesOficial",
        "mallorca": "@RCDMallorca",
        "osasuna": "@caosasunatv",
        "rayo vallecano": "@RayoVallecano",
        "real betis": "@RealBetisBalompie",
        "real madrid": "@realmadrid",
        "real sociedad": "@realsociedad",
        "sevilla": "@sevillafc",
        "valencia": "@valenciacf",
        "valladolid": "@RealValladolid",
        "villarreal": "@villarrealcf",
        "espanyol": "@RCDEspanyol",

        # ===== BUNDESLIGA (18 Teams) =====
        "augsburg": "@fcaugsburg",
        "bayer leverkusen": "@bayerleverkusen",
        "bayern munich": "@fcbayern",
        "bochum": "@vflbochum1848",
        "borussia dortmund": "@BVB",
        "borussia monchengladbach": "@borussia",
        "eintracht frankfurt": "@eintracht",
        "freiburg": "@SCFreiburg",
        "heidenheim": "@1fcheidenheim1846",
        "hoffenheim": "@tsghoffenheim",
        "holstein kiel": "@holsteinkiel",
        "mainz": "@1fsvmainz05",
        "rb leipzig": "@dierotenbullen",
        "st pauli": "@fcstpauli",
        "stuttgart": "@vfb",
        "union berlin": "@1fcunionberlin",
        "werder bremen": "@werderbremen",
        "wolfsburg": "@vflwolfsburg_official",

        # ===== SERIE A (20 Teams) =====
        "ac milan": "@ACMilan",
        "atalanta": "@AtalantaBC",
        "bologna": "@BolognaFC1909Official",
        "cagliari": "@CagliariCalcio",
        "como": "@comofootball",
        "empoli": "@empolifc",
        "fiorentina": "@ACFFiorentina",
        "genoa": "@GenoaCFC",
        "inter milan": "@inter",
        "juventus": "@juventus",
        "lazio": "@officialsslazio",
        "lecce": "@UsLecceOfficial",
        "monza": "@ACMonza",
        "napoli": "@sscnapoliofficial",
        "parma": "@ParmaCalcio1913Official",
        "roma": "@asroma",
        "torino": "@TorinoFootballClubOfficial",
        "udinese": "@UdineseCalcio",
        "venezia": "@VeneziaFC",
        "verona": "@HellasVeronaFC",

        # ===== LIGUE 1 (18 Teams) =====
        "angers": "@AngersSCOofficiel",
        "auxerre": "@AJAOfficiel",
        "brest": "@stadebrestois29",
        "le havre": "@HACFoot",
        "lens": "@RCLens",
        "lille": "@losclive",
        "lyon": "@ol",
        "marseille": "@olympiquedemarseille",
        "monaco": "@asmonaco",
        "montpellier": "@MHSC_officiel",
        "nantes": "@fcnantes",
        "nice": "@ogcnice",
        "psg": "@psg",
        "reims": "@StadeDeReims",
        "rennes": "@staderennais",
        "saint-etienne": "@asse",
        "strasbourg": "@RCSA",
        "toulouse": "@ToulouseFC",
    }

    def __init__(self, output_dir="temp_videos", cookies_path=None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.cookies_path = cookies_path

        # EXACT yt-dlp options from the working code
        self.ydl_opts = {
            'format': '(bestvideo[height>=1080][ext=mp4]/bestvideo[height>=1080]/bestvideo[ext=mp4]/bestvideo)+(bestaudio[ext=m4a]/bestaudio)/best[height>=1080]/best',
            'outtmpl': str(self.output_dir / '%(title)s.%(ext)s'),
            'merge_output_format': 'mp4',
            'ignoreerrors': False,
            'no_warnings': False,
            'retries': 5,
            'fragment_retries': 5,
            'writeinfojson': False,
            'writedescription': False,
            'writethumbnail': False,
            'format_sort': ['res:1080', 'ext:mp4:m4a', 'res', 'fps', 'hdr:12', 'codec:h264:aac'],
            'postprocessors': [
                {'key': 'FFmpegVideoConvertor', 'preferedformat': 'mp4'},
                {'key': 'FFmpegMetadata'}
            ],

            # ADD THESE TO BYPASS PO TOKEN:
            'extractor_args': {
                'youtube': {
                    'player_client': ['web'],  # Use web client (no PO token needed)
                    'player_skip': ['configs'],
                }
            },


            'http_headers': {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-us,en;q=0.5',
                'Accept-Encoding': 'gzip,deflate',
                'Accept-Charset': 'ISO-8859-1,utf-8;q=0.7,*;q=0.7',
                'Keep-Alive': '300',
                'Connection': 'keep-alive',
            },
            'prefer_free_formats': False,
            'youtube_include_dash_manifest': True,

            # ADD THESE FOR 403 BYPASS:
            'extractor_args': {'youtube': {'player_client': ['android', 'web']}},
            'format_sort': ['res:720', 'ext:mp4:m4a'],
            'postprocessors': [{
                'key': 'FFmpegVideoConvertor',
                'preferedformat': 'mp4',
            }],
        }
        # Add cookies if provided
        if self.cookies_path:
            self.ydl_opts['cookiefile'] = self.cookies_path
            print(f"🍪 Using cookies: {self.cookies_path}")

    def normalize_team_name(self, team_name):
        """Normalize team names for matching"""
        return team_name.lower().strip()

    def get_channel_handle(self, team_name):
        """Get the YouTube channel handle for a team"""
        normalized = self.normalize_team_name(team_name)
        return self.TEAM_CHANNELS.get(normalized)

    def fuzzy_get_channel_handle(self, team_name):
        """
        Fuzzy match team name with scoring to find best channel

        Returns:
            Channel handle or None
        """
        from difflib import SequenceMatcher

        normalized = self.normalize_team_name(team_name)

        # Try exact match first
        if normalized in self.TEAM_CHANNELS:
            return self.TEAM_CHANNELS[normalized]

        # Score all possible matches
        scores = []
        for channel_team in self.TEAM_CHANNELS.keys():
            # Calculate similarity score
            ratio = SequenceMatcher(None, normalized, channel_team).ratio()

            # Bonus points for key words matching
            normalized_words = set(normalized.split())
            channel_words = set(channel_team.split())
            word_overlap = len(normalized_words & channel_words)

            # Combined score: 70% string similarity + 30% word overlap bonus
            final_score = (ratio * 0.7) + (word_overlap * 0.1)

            scores.append((channel_team, final_score, self.TEAM_CHANNELS[channel_team]))

        # Sort by score, get best match
        scores.sort(key=lambda x: x[1], reverse=True)

        if scores and scores[0][1] > 0.5:  # Threshold
            matched_team = scores[0][0]
            score = scores[0][1]
            handle = scores[0][2]
            print(f"   🔍 Fuzzy matched '{team_name}' → '{matched_team}' (score: {score:.2f})")
            return handle

        return None

    def search_channel_for_match(self, team_name, opponent_name, match_date, score=None, geo_block_manager=None):
        """Search a team's YouTube channel for match highlights (metadata only, no download)"""
        channel_handle = self.get_channel_handle(team_name)
        if not channel_handle:
            print(f"❌ No channel found for {team_name}")
            return None

        # Check geo-block list
        if geo_block_manager and geo_block_manager.is_blocked(channel_handle):
            print(f"⛔ Skipping {channel_handle} (geo-blocked)")
            return None

        # Parse date - handle multiple formats
        if isinstance(match_date, str):
            try:
                match_date = datetime.strptime(match_date, "%Y-%m-%d")
            except:
                try:
                    match_date = datetime.strptime(match_date.lower(), "%B %d %Y")
                except:
                    try:
                        match_date = datetime.strptime(match_date.lower(), "%B %d, %Y")
                    except:
                        print(f"❌ Invalid date format: {match_date}")
                        return None

        print(f"🔍 Searching {channel_handle} for match highlights...")
        print(f"   🎯 Looking for opponent: '{opponent_name}'")
        print(f"   📅 Match date: {match_date}")
        print(f"   🎲 Score: {score if score else 'Not specified'}")
        print(f"   🔎 Search query: {'+'.join([opponent_name, match_date.strftime('%B'), str(match_date.year)] if isinstance(match_date, datetime) else [opponent_name])}")

        try:
            # Use lightweight search config (no downloads)
            search_opts = {
                'quiet': True,
                'extract_flat': 'in_playlist',  # Get basic info but not full extraction
                'http_headers': self.ydl_opts['http_headers']
            }

            # Add cookies if available
            if self.cookies_path:
                search_opts['cookiefile'] = self.cookies_path

            # Build search query for channel
            search_terms = [opponent_name]
            if isinstance(match_date, datetime):
                search_terms.append(match_date.strftime("%B"))  # Month name
                search_terms.append(str(match_date.year))

            search_query = "+".join(search_terms)

            # Try with @ first, then without @ if it fails
            channel_urls_to_try = []
            if channel_handle.startswith('@'):
                channel_urls_to_try = [
                    f"https://www.youtube.com/{channel_handle}/search?query={search_query}",
                    f"https://www.youtube.com/{channel_handle[1:]}/search?query={search_query}"
                ]
            else:
                channel_urls_to_try = [
                    f"https://www.youtube.com/{channel_handle}/search?query={search_query}",
                    f"https://www.youtube.com/@{channel_handle}/search?query={search_query}"
                ]

            info = None
            for channel_search_url in channel_urls_to_try:
                try:
                    with yt_dlp.YoutubeDL(search_opts) as ydl:
                        info = ydl.extract_info(channel_search_url, download=False)
                        if info and 'entries' in info:
                            print(f"   ✅ Channel search worked: {len(info.get('entries', []))} results")
                            break  # Success!
                except:
                    continue  # Try next URL format

            # Check if we got results
            if not info:
                print(f"   ⚠️ DEBUG: info is None")
                return None

            if 'entries' not in info:
                print(f"   ⚠️ DEBUG: No 'entries' key in info")
                print(f"   ⚠️ DEBUG: info keys: {info.keys()}")
                return None

            if not info['entries']:
                print(f"   ⚠️ DEBUG: entries list is empty")
                return None

            # Print all found videos
            print(f"   📊 Found {len(info['entries'])} videos:")
            for idx, video in enumerate(info['entries'][:20], 1):
                if video:
                    print(f"      {idx}. {video.get('title', 'NO TITLE')}")

            opponent_normalized = self.normalize_team_name(opponent_name)

            # Check each video for match
            for video in info['entries'][:20]:
                if not video:
                    print(f"   ⚠️ Skipping None video")
                    continue

                title = video.get('title', '').lower()
                upload_date = video.get('upload_date', '')
                duration = video.get('duration', 0)

                print(f"   🔍 Checking: {video.get('title', 'NO TITLE')[:60]}")
                print(f"      Duration: {duration}s, Upload: {upload_date}")

                # CRITICAL FILTERS for actual match highlights:
                title_match = opponent_normalized in title or opponent_name.lower() in title
                print(f"      Title match: {title_match}")

                highlight_keywords = ['highlight', 'goals', 'match', 'extended', 'vs', 'v ']
                interview_keywords = ['interview', 'post match', 'press conference', 'reaction', 'post-match', 'tunnel cam']

                has_highlight_kw = any(kw in title for kw in highlight_keywords)
                has_interview_kw = any(kw in title for kw in interview_keywords)

                print(f"      Has highlight keyword: {has_highlight_kw}")
                print(f"      Has interview keyword: {has_interview_kw}")

                if has_interview_kw:
                    print(f"      ❌ Rejected: Interview")
                    continue

                # Reject youth/academy matches
                youth_keywords = ['u21', 'u23', 'u18', 'u19', 'youth', 'academy', 'development squad', 'reserve']
                if any(keyword in title for keyword in youth_keywords):
                    print(f"      ❌ Rejected: Youth/Academy match (not first team)")
                    continue

                # Reject women's matches
                womens_keywords = ['wsl', 'uwcl', "women's", 'womens', 'ladies', 'féminine', 'femenino', 'wpl', 'nwsl', 'wwc']
                if any(keyword in title for keyword in womens_keywords):
                    print(f"      ❌ Rejected: Women's match (looking for men's first team)")
                    continue

                if not has_highlight_kw:
                    print(f"      ❌ Rejected: No highlight keyword")
                    continue

                # Duration check: 2-15 minutes
                if duration > 0:
                    if duration < 120 or duration > 900:
                        print(f"      ❌ Rejected: Duration {duration}s not in range 120-900")
                        continue

                # Check date proximity
                date_match = True
                if upload_date:
                    try:
                        upload_datetime = datetime.strptime(upload_date, "%Y%m%d")
                        days_diff = abs((upload_datetime - match_date).days)
                        date_match = days_diff <= 3
                        print(f"      Date match: {date_match} (diff: {days_diff} days)")
                    except:
                        date_match = True
                else:
                    # No upload date - check if year is in title
                    if isinstance(match_date, datetime):
                        year_str = str(match_date.year)
                        if year_str in title:
                            date_match = True
                            print(f"      Date match: True (year '{year_str}' in title)")
                        else:
                            date_match = False
                            print(f"      Date match: False (year '{year_str}' NOT in title)")
                    else:
                        print(f"      Date match: True (no upload date, no year check)")

                # Check score if provided
                # Check score if provided
                score_match = True
                if score:
                    # If title contains ANY score pattern, it must match our score
                    import re
                    title_scores = re.findall(r'\b\d+-\d+\b', title)
                    if title_scores:
                        # Title has a score - it must match ours
                        score_match = score in title
                        print(f"      Score check: Found '{title_scores}' in title, looking for '{score}': {score_match}")
                    else:
                        # No score in title - that's okay
                        print(f"      Score check: No score in title, accepting")
                else:
                    print(f"      Score check: Not required")
                    print(f"      Score match: {score_match}")

                if title_match and date_match and score_match:
                    video_url = f"https://www.youtube.com/watch?v={video['id']}"
                    print(f"   ✅ SELECTED: {video.get('title')}")
                    print(f"   📅 Upload date: {upload_date}")
                    if duration:
                        mins = int(duration) // 60
                        secs = int(duration) % 60
                        print(f"   ⏱️  Duration: {mins}:{secs:02d}")
                    print(f"   🔗 URL: {video_url}\n")

                    return {
                        'url': video_url,
                        'title': video.get('title', 'Unknown'),
                        'duration': duration,
                        'upload_date': upload_date,
                        'channel': channel_handle
                    }
                else:
                    print(f"      ❌ Rejected: Final checks failed")

        except Exception as e:
            error_msg = str(e)
            print(f"   Search failed: {error_msg[:100]}")

            # Add delay if rate limited
            if '404' in error_msg or 'Unable to download' in error_msg:
                print(f"   ⏸️  Rate limit detected, waiting 3s...")
                time.sleep(3)

        return None

    def search_simplified(self, team1, team2, score=None, geo_block_manager=None):
        """Simplified search: just 'Team1 Team2' without opponent matching"""
        print(f"🔍 Trying simplified search: '{team1} {team2}'...")

        for team_name in [team1, team2]:
            channel_handle = self.get_channel_handle(team_name)
            if not channel_handle:
                continue

            if geo_block_manager and geo_block_manager.is_blocked(channel_handle):
                print(f"⛔ Skipping {channel_handle} (geo-blocked)")
                continue

            try:
                search_opts = {
                    'quiet': True,
                    'extract_flat': True,
                    'playlistend': 20,
                    'http_headers': self.ydl_opts['http_headers']
                }

                with yt_dlp.YoutubeDL(search_opts) as ydl:
                    # Handle both @username and /channelname formats
                    if channel_handle.startswith('@'):
                        channel_videos_url = f"https://www.youtube.com/{channel_handle}/videos"
                    else:
                        channel_videos_url = f"https://www.youtube.com/{channel_handle}/videos"
                    info = ydl.extract_info(channel_videos_url, download=False)

                    if not info or 'entries' not in info:
                        continue

                    # Look for videos mentioning both teams
                    team1_norm = self.normalize_team_name(team1)
                    team2_norm = self.normalize_team_name(team2)

                    for video in info['entries'][:20]:
                        if not video:
                            continue

                        title = video.get('title', '').lower()
                        duration = video.get('duration', 0)

                        # Both teams mentioned?
                        score_match = True
                        if score:
                            score_match = score in title

                        if team1_norm in title and team2_norm in title and score_match:
                            # Still check for highlights keywords
                            highlight_keywords = ['highlight', 'goals', 'match', 'extended', 'vs', 'v ']
                            if any(kw in title for kw in highlight_keywords):
                                if 120 <= duration <= 900:
                                    video_url = f"https://www.youtube.com/watch?v={video['id']}"
                                    print(f"✅ Found via simplified: {video.get('title')}")
                                    if duration:
                                        mins = int(duration) // 60
                                        secs = int(duration) % 60
                                        print(f"⏱️  Duration: {mins}:{secs:02d}")
                                    print(f"🔗 URL: {video_url}\n")

                                    return {
                                        'url': video_url,
                                        'title': video.get('title', 'Unknown'),
                                        'duration': duration,
                                        'upload_date': video.get('upload_date', ''),
                                        'channel': channel_handle
                                    }

            except Exception as e:
                error_msg = str(e)
                print(f"   Simplified search failed on {channel_handle}: {error_msg[:100]}")

                # Add delay if rate limited
                if '404' in error_msg or 'Unable to download' in error_msg:
                    print(f"   ⏸️  Rate limit detected, waiting 30s...")
                    time.sleep(3)

                continue

        return None

    def search_player_highlights(self, player_query: str, geo_block_manager=None):
        """Generic YouTube search for player highlights (not channel-specific)"""
        print(f"🔍 Generic search for: '{player_query}'...")

        try:
            search_opts = {
                'quiet': True,
                'extract_flat': True,
                'default_search': 'ytsearch20',  # Generic YouTube search
                'http_headers': self.ydl_opts['http_headers']
            }

            with yt_dlp.YoutubeDL(search_opts) as ydl:
                info = ydl.extract_info(f"ytsearch20:{player_query}", download=False)

                if not info or 'entries' not in info:
                    return None

                # Filter for highlight videos
                for video in info['entries']:
                    if not video:
                        continue

                    title = video.get('title', '').lower()
                    duration = video.get('duration', 0)

                    # Flexible filtering for player highlights
                    highlight_keywords = ['highlight', 'goals', 'skills', 'dribbling', 'assists', 'compilation']
                    interview_keywords = ['interview', 'press conference', 'reaction']

                    has_highlight_kw = any(kw in title for kw in highlight_keywords)
                    has_interview_kw = any(kw in title for kw in interview_keywords)

                    if has_interview_kw:
                        continue

                    if not has_highlight_kw:
                        continue

                    # More flexible duration (2-15 min still)
                    if 120 <= duration <= 900:
                        video_url = f"https://www.youtube.com/watch?v={video['id']}"
                        print(f"✅ Found: {video.get('title')}")
                        if duration:
                            mins = int(duration) // 60
                            secs = int(duration) % 60
                            print(f"⏱️  Duration: {mins}:{secs:02d}")
                        print(f"🔗 URL: {video_url}\n")

                        return {
                            'url': video_url,
                            'title': video.get('title', 'Unknown'),
                            'duration': duration,
                            'upload_date': video.get('upload_date', ''),
                            'channel': 'generic_search'
                        }

        except Exception as e:
            print(f"   Player highlight search failed: {str(e)[:100]}")

        return None

    def search_with_date(self, team1, team2, match_date, score=None, geo_block_manager=None):
        """
        Search with date included in query for better historical match matching

        Args:
            team1: First team name
            team2: Second team name
            match_date: Match date string
            geo_block_manager: Geo-block manager instance

        Returns:
            Video info dict or None
        """
        # Parse date to get year
        try:
            if isinstance(match_date, str):
                try:
                    date_obj = datetime.strptime(match_date.lower(), "%B %d %Y")
                except:
                    try:
                        date_obj = datetime.strptime(match_date, "%Y-%m-%d")
                    except:
                        date_obj = None

                if date_obj:
                    year = date_obj.year
                    month = date_obj.strftime("%B")
                    if score:
                        search_query = f"{team1} {team2} {score} {month} {year}"
                    else:
                        search_query = f"{team1} {team2} {month} {year}"
                else:
                    search_query = f"{team1} {team2} {match_date}"
            else:
                search_query = f"{team1} {team2}"

            print(f"🔍 Search with date: '{search_query}'...")

            search_opts = {
                'quiet': True,
                'extract_flat': True,
                'default_search': 'ytsearch15',
                'http_headers': self.ydl_opts['http_headers']
            }

            with yt_dlp.YoutubeDL(search_opts) as ydl:
                info = ydl.extract_info(f"ytsearch15:{search_query}", download=False)

                if not info or 'entries' not in info:
                    return None

                for video in info['entries']:
                    if not video:
                        continue

                    title = video.get('title', '').lower()
                    duration = video.get('duration', 0)

                    # Check for highlight keywords
                    highlight_keywords = ['highlight', 'goals', 'match', 'extended', 'vs']
                    interview_keywords = ['interview', 'press conference', 'reaction']

                    has_highlight = any(kw in title for kw in highlight_keywords)
                    has_interview = any(kw in title for kw in interview_keywords)

                    if has_interview or not has_highlight:
                        continue

                    # Check score if provided
                    score_match = True
                    if score:
                        score_match = score in title

                    # Duration check
                    # Duration check
                    if score_match and 120 <= duration <= 900:
                        # Verify BOTH teams are in the title
                        team1_normalized = self.normalize_team_name(team1)
                        team2_normalized = self.normalize_team_name(team2)

                        print(f"   🔍 DEBUG: Looking for team1='{team1}' (normalized: '{team1_normalized}')")
                        print(f"   🔍 DEBUG: Looking for team2='{team2}' (normalized: '{team2_normalized}')")

                        team1_in_title = team1_normalized in title or team1.lower() in title
                        team2_in_title = team2_normalized in title or team2.lower() in title

                        print(f"   🔍 DEBUG: team1_in_title={team1_in_title}, team2_in_title={team2_in_title}")

                        team1_in_title = team1_normalized in title or team1.lower() in title
                        team2_in_title = team2_normalized in title or team2.lower() in title

                        if not team1_in_title:
                            print(f"   ⚠️ Team '{team1}' not found in title '{video.get('title', '')}', skipping")
                            continue

                        if not team2_in_title:
                            print(f"   ⚠️ Team '{team2}' not found in title '{video.get('title', '')}', skipping")
                            continue

                        # Both teams verified!
                        video_url = f"https://www.youtube.com/watch?v={video['id']}"
                        print(f"✅ Found: {video.get('title')}")
                        if duration:
                            mins = int(duration) // 60
                            secs = int(duration) % 60
                            print(f"⏱️  Duration: {mins}:{secs:02d}")
                        print(f"🔗 URL: {video_url}\n")

                        return {
                            'url': video_url,
                            'title': video.get('title', 'Unknown'),
                            'duration': duration,
                            'upload_date': video.get('upload_date', ''),
                            'channel': 'search_with_date'
                        }

        except Exception as e:
            error_msg = str(e)
            print(f"   ❌ Search with date failed: {error_msg[:200]}")
            print(f"   📊 Error type: {type(e).__name__}")

            # Add delay if rate limited
            if '404' in error_msg or 'Unable to download' in error_msg:
                print(f"   ⏸️  Rate limit detected, waiting 30s...")
                time.sleep(3)

        return None

    def broad_youtube_search(self, query: str, score: str = None, max_results: int = 10, geo_block_manager=None):
        """
        Broad YouTube search (not channel-specific) with geo-restriction handling

        Args:
            query: Search query
            max_results: Number of results to check
            geo_block_manager: Geo-block manager (unused but kept for consistency)

        Returns:
            Video info dict or None
        """
        try:
            search_opts = {
                'quiet': True,
                'extract_flat': True,
                'default_search': f'ytsearch{max_results}',
                'http_headers': self.ydl_opts['http_headers']
            }

            with yt_dlp.YoutubeDL(search_opts) as ydl:
                info = ydl.extract_info(f"ytsearch{max_results}:{query}", download=False)

                if not info or 'entries' not in info:
                    return None

                # Check each result
                for video in info['entries']:
                    if not video:
                        continue

                    title = video.get('title', '').lower()
                    duration = video.get('duration', 0)

                    # Skip premium/restricted videos
                    availability = video.get('availability', 'public')
                    if availability in ['premium_only', 'subscriber_only', 'needs_auth', 'private']:
                        continue

                    # Check for highlights
                    highlight_keywords = ['highlight', 'goals', 'match', 'extended', 'vs', 'full match']
                    interview_keywords = ['interview', 'press conference', 'reaction', 'post-match']

                    has_highlight = any(kw in title for kw in highlight_keywords)
                    has_interview = any(kw in title for kw in interview_keywords)

                    if has_interview or not has_highlight:
                        continue

                    # Check score if provided
                    score_match = True
                    if score:
                        score_match = score in title

                    # Duration check (2-15 min)
                    if score_match and 120 <= duration <= 900:
                        video_url = f"https://www.youtube.com/watch?v={video['id']}"
                        print(f"   ✅ Found: {video.get('title')}")
                        if duration:
                            mins = int(duration) // 60
                            secs = int(duration) % 60
                            print(f"   ⏱️  Duration: {mins}:{secs:02d}")
                        print(f"   🔗 URL: {video_url}")

                        return {
                            'url': video_url,
                            'title': video.get('title', 'Unknown'),
                            'duration': duration,
                            'upload_date': video.get('upload_date', ''),
                            'channel': 'broad_search'
                        }

                return None

        except Exception as e:
            error_msg = str(e)
            # Add delay if rate limited
            if '404' in error_msg or 'Unable to download' in error_msg:
                print(f"   ⏸️  Rate limit detected, waiting 30s...")
                time.sleep(3)
            return None

    def batch_download_videos(self, video_list: List[Dict], geo_block_manager: GeoBlockManager) -> Dict[str, Path]:
        """
        Download multiple videos in one session with cookies

        Returns:
            Dictionary mapping video_url -> downloaded_file_path
        """
        print(f"\n{'='*70}")
        print(f"📥 BATCH DOWNLOAD: {len(video_list)} videos")
        print(f"{'='*70}\n")

        downloaded = {}
        failed = {}

        for i, video_info in enumerate(video_list, 1):
            video_url = video_info['url']
            channel = video_info.get('channel', 'unknown')

            print(f"[{i}/{len(video_list)}] Downloading from {channel}...")
            print(f"   URL: {video_url}")

            max_retries = 3
            for attempt in range(max_retries):
                try:
                    # Try different extraction strategies on each retry
                    if attempt == 0:
                        # First attempt: Normal with cookies
                        current_opts = self.ydl_opts.copy()
                    elif attempt == 1:
                        # Second attempt: Force android client
                        current_opts = self.ydl_opts.copy()
                        current_opts['extractor_args'] = {'youtube': {'player_client': ['android']}}
                        print(f"   🔄 Retry {attempt+1}: Using Android client...")
                    else:
                        # Third attempt: Try ios client
                        current_opts = self.ydl_opts.copy()
                        current_opts['extractor_args'] = {'youtube': {'player_client': ['ios']}}
                        print(f"   🔄 Retry {attempt+1}: Using iOS client...")

                    with yt_dlp.YoutubeDL(current_opts) as ydl:
                        ydl.download([video_url])

                    # Find downloaded file
                    video_files = list(self.output_dir.glob("*.mp4"))
                    if video_files:
                        newest_file = max(video_files, key=lambda p: p.stat().st_mtime)

                        # Verify duration
                        vr = VideoReader(str(newest_file), ctx=cpu(0))
                        duration = len(vr) / vr.get_avg_fps()

                        if duration < 120:
                            print(f"   ❌ Too short ({duration:.0f}s), deleting")
                            os.remove(newest_file)
                            failed[video_url] = "too_short"
                            break
                        elif duration > 900:
                            print(f"   ❌ Too long ({duration:.0f}s), deleting")
                            os.remove(newest_file)
                            failed[video_url] = "too_long"
                            break

                        downloaded[video_url] = newest_file
                        print(f"   ✅ Downloaded: {newest_file.name}\n")
                        break
                    else:
                        if attempt < max_retries - 1:
                            print(f"   ⚠️ No file found, retrying...")
                            time.sleep(5)
                        else:
                            failed[video_url] = "file_not_found"
                            print(f"   ❌ Download failed after {max_retries} attempts\n")

                except Exception as e:
                    error_str = str(e)

                    # Check for geo-block
                    if "not available" in error_str.lower() or "your country" in error_str.lower():
                        print(f"   ⛔ GEO-BLOCKED: {channel}")
                        geo_block_manager.add_blocked_channel(channel, error_str)
                        failed[video_url] = "geo_blocked"
                        break

                    # Check for 403 Forbidden
                    if "403" in error_str or "Forbidden" in error_str:
                        if attempt < max_retries - 1:
                            print(f"   ⚠️ 403 Forbidden (attempt {attempt + 1}/{max_retries})")
                            print(f"   🔄 Waiting {10 * (attempt + 1)}s before retry...")
                            time.sleep(10 * (attempt + 1))  # Exponential backoff
                        else:
                            print(f"   ❌ Failed: 403 Forbidden after {max_retries} attempts\n")
                            failed[video_url] = "403_forbidden"
                    else:
                        # Other errors
                        if attempt < max_retries - 1:
                            print(f"   ⚠️ Error (attempt {attempt + 1}/{max_retries}): {error_str[:100]}")
                            time.sleep(5)
                        else:
                            print(f"   ❌ Failed after {max_retries} attempts: {error_str[:100]}\n")
                            failed[video_url] = error_str[:200]

        print(f"\n{'='*70}")
        print(f"📊 BATCH DOWNLOAD COMPLETE")
        print(f"   ✅ Success: {len(downloaded)}/{len(video_list)}")
        print(f"   ❌ Failed: {len(failed)}/{len(video_list)}")
        print(f"{'='*70}\n")

        return downloaded

    def download_video(self, video_url, max_duration=600):
        """Download video using the EXACT working configuration - NO PRE-CHECK"""
        print(f"\n⬇️ Downloading: {video_url}")

        try:
            # Skip info extraction, just download directly
            print("⬇️ Starting download (skipping pre-check to avoid bot detection)...")

            with yt_dlp.YoutubeDL(self.ydl_opts) as ydl:
                ydl.download([video_url])

            # Find downloaded file
            video_files = list(self.output_dir.glob("*.mp4"))
            if video_files:
                newest_file = max(video_files, key=lambda p: p.stat().st_mtime)

                # Sanitize filename - remove ALL problematic Unicode characters
                import unicodedata
                original_name = newest_file.name

                # Normalize to ASCII (replaces Unicode with closest ASCII equivalent)
                sanitized_name = unicodedata.normalize('NFKD', original_name)
                sanitized_name = sanitized_name.encode('ascii', 'ignore').decode('ascii')

                # If filename changed, rename the file
                if sanitized_name != original_name:
                    sanitized_path = newest_file.parent / sanitized_name
                    os.rename(str(newest_file), str(sanitized_path))
                    newest_file = sanitized_path
                    print(f"   🔧 Sanitized filename: {original_name} → {sanitized_name}")
                # Create unique filename to avoid conflicts
                import hashlib
                import time

                # Generate unique suffix from URL + timestamp
                unique_id = hashlib.md5(f"{video_url}{time.time()}".encode()).hexdigest()[:8]

                # Keep extension, add unique ID
                name_parts = sanitized_name.rsplit('.', 1)
                if len(name_parts) == 2:
                    unique_name = f"{name_parts[0]}_{unique_id}.{name_parts[1]}"
                else:
                    unique_name = f"{sanitized_name}_{unique_id}"

                unique_path = newest_file.parent / unique_name
                os.rename(str(newest_file), str(unique_path))
                newest_file = unique_path
                print(f"   🔖 Unique filename: {unique_name}")

                # Check duration AFTER download
                from decord import VideoReader, cpu
                vr = VideoReader(str(newest_file), ctx=cpu(0))
                duration = len(vr) / vr.get_avg_fps()

                print(f"✅ Downloaded: {newest_file.name}")
                print(f"⏱️ Duration: {int(duration)//60}:{int(duration)%60:02d}")

                if duration < 120:
                    print(f"❌ Too short, deleting")
                    os.remove(newest_file)
                    return None
                elif duration > max_duration:
                    print(f"❌ Too long, deleting")
                    os.remove(newest_file)
                    return None

                return newest_file
            else:
                print("❌ No video file found after download")
                return None

        except Exception as e:
            print(f"❌ Download failed: {str(e)}")
            return None

    def find_and_download_match_highlights(self, team1, team2, match_date):
        """Main method: Search both team channels and download"""
        print(f"\n{'='*60}")
        print(f"🎯 Finding highlights: {team1} vs {team2}")
        print(f"📅 Date: {match_date}")
        print(f"{'='*60}")

        # Try team1's channel first
        video_url = self.search_channel_for_match(team1, team2, match_date)

        # If not found, try team2's channel
        if not video_url:
            print(f"\n🔄 Trying {team2}'s channel...")
            video_url = self.search_channel_for_match(team2, team1, match_date)

        # Download if found
        if video_url:
            return self.download_video(video_url)
        else:
            print(f"\n❌ No highlights found on either team's channel")
            return None

    def smart_broad_search(self, raw_description: str, match_tracker, geo_block_manager=None):
        """
        Intelligent broad search:
        1. Extract teams from description
        2. Fuzzy match to correct team names
        3. Search YouTube with corrected query
        4. Score results by date/score match

        Args:
            raw_description: Full description from video plan
            match_tracker: MatchContextTracker instance

        Returns:
            Video info dict or None
        """
        print(f"   🧠 Smart broad search for: '{raw_description[:80]}...'")

        # Extract teams, date, score
        teams = match_tracker._extract_teams(raw_description)
        date = match_tracker._extract_date(raw_description)
        score = match_tracker._extract_score(raw_description)

        # Validate extraction
        if not teams or len(teams) < 2:
            print(f"   ❌ Could not extract teams from: {raw_description[:80]}")
            return None

        print(f"   📊 Extracted: {teams[0]} vs {teams[1]}, {date}, {score}")
        # Fuzzy match teams to correct names
        if not teams or len(teams) < 2:
            print(f"   ❌ Could not extract teams from description")
            return None

        corrected_teams = []
        for team in teams[:2]:
            # Try fuzzy match
            handle = self.fuzzy_get_channel_handle(team)
            if handle:
                # Find the matched team name
                matched_name = [k for k, v in self.TEAM_CHANNELS.items() if v == handle][0]
                corrected_teams.append(matched_name)
            else:
                # Keep original if no match
                corrected_teams.append(team)

        # ADD THIS DEBUG:
        print(f"   🔍 DEBUG: corrected_teams = {corrected_teams}")
        print(f"   🔍 DEBUG: len(corrected_teams) = {len(corrected_teams)}")
        for i, team in enumerate(corrected_teams):
            print(f"   🔍 DEBUG: corrected_teams[{i}] = {team} (type: {type(team)})")

        # Build corrected query
        print(f"   🔍 DEBUG: About to build query...")
        print(f"   🔍 DEBUG: corrected_teams[0] = '{corrected_teams[0] if len(corrected_teams) > 0 else 'MISSING'}'")
        print(f"   🔍 DEBUG: corrected_teams[1] = '{corrected_teams[1] if len(corrected_teams) > 1 else 'MISSING'}'")
        print(f"   🔍 DEBUG: score = '{score}'")
        print(f"   🔍 DEBUG: date = '{date}'")
        if len(corrected_teams) >= 2:
            if score:
                corrected_query = f"{corrected_teams[0]} vs {corrected_teams[1]} {score} {date}"
            else:
                corrected_query = f"{corrected_teams[0]} vs {corrected_teams[1]} {date}"
        else:
            corrected_query = raw_description

        print(f"   🔍 Corrected query: '{corrected_query}'")

        # Search YouTube
        try:
            search_opts = {
                'quiet': True,
                'extract_flat': 'in_playlist',
                'http_headers': self.ydl_opts['http_headers']
            }

            if self.cookies_path:
                search_opts['cookiefile'] = self.cookies_path

            with yt_dlp.YoutubeDL(search_opts) as ydl:
                info = ydl.extract_info(f"ytsearch20:{corrected_query}", download=False)

                if not info or 'entries' not in info:
                    return None

                # Score each result
                scored_results = []
                for video in info['entries']:
                    if not video:
                        continue

                    title = video.get('title', '') or ''
                    title = title.lower()

                    description = video.get('description', '') or ''  # Handle None
                    description = description.lower()
                    upload_date = video.get('upload_date', '')
                    duration = video.get('duration', 0)

                    # Skip non-highlights
                    highlight_kw = ['highlight', 'goals', 'match', 'extended', 'vs']
                    if not any(kw in title for kw in highlight_kw):
                        continue

                    # Skip interviews
                    interview_kw = ['interview', 'press conference', 'reaction']
                    if any(kw in title for kw in interview_kw):
                        continue

                    # Duration filter (2-15 min)
                    if duration < 120 or duration > 900:
                        continue

                    # Calculate score
                    video_score = 0

                    # YEAR MATCH (most important) - 50 points
                    if date and 'year' in str(date):
                        try:
                            from datetime import datetime
                            if isinstance(date, str):
                                date_obj = datetime.strptime(date.lower(), "%B %d %Y")
                            else:
                                date_obj = date
                            year_str = str(date_obj.year)

                            # Check title for year
                            if year_str in title:
                                video_score += 50
                            # Check description for year
                            elif year_str in description:
                                video_score += 30
                            # Check upload date (within 2 years)
                            elif upload_date:
                                try:
                                    upload_year = int(upload_date[:4])
                                    year_diff = abs(upload_year - date_obj.year)
                                    if year_diff <= 2:
                                        video_score += max(0, 20 - (year_diff * 5))
                                except:
                                    pass
                        except:
                            pass

                    # SCORE MATCH - 30 points
                    if score:
                        import re
                        # Check title
                        if score in title:
                            video_score += 30
                        # Check description
                        elif score in description:
                            video_score += 20
                        # Penalize if different score in title
                        title_scores = re.findall(r'\b\d+-\d+\b', title)
                        if title_scores and score not in title:
                            video_score -= 20

                    # TEAM NAMES MATCH - Validate BOTH teams present
                    team1_normalized = self.normalize_team_name(corrected_teams[0])
                    team2_normalized = self.normalize_team_name(corrected_teams[1])

                    team1_in_title = team1_normalized in title or corrected_teams[0].lower() in title
                    team2_in_title = team2_normalized in title or corrected_teams[1].lower() in title

                    # CRITICAL: Both teams MUST be in title
                    if not team1_in_title or not team2_in_title:
                        # Skip this video entirely - wrong match!
                        continue

                    # Award points for team matches (both are present at this point)
                    video_score += 20  # Both teams confirmed

                    scored_results.append({
                        'video': video,
                        'score': video_score,
                        'title': video.get('title', ''),
                        'duration': duration
                    })

                # Sort by score
                scored_results.sort(key=lambda x: x['score'], reverse=True)

                # Print top 5
                print(f"   📊 Scored {len(scored_results)} results:")
                for i, result in enumerate(scored_results[:5], 1):
                    print(f"      {i}. [{result['score']} pts] {result['title'][:60]}")

                # Return best match if score > threshold
                if scored_results and scored_results[0]['score'] >= 15:  # Lower threshold
                    best = scored_results[0]['video']
                    print(f"   ✅ Selected top result (score: {scored_results[0]['score']})")

                    return {
                        'url': f"https://www.youtube.com/watch?v={best['id']}",
                        'title': best.get('title', 'Unknown'),
                        'duration': best.get('duration', 0),
                        'upload_date': best.get('upload_date', ''),
                        'channel': 'smart_broad_search'
                    }
                else:
                    print(f"   ❌ No results above threshold")
                    return None

        except Exception as e:
            print(f"   ❌ Smart broad search failed: {str(e)[:100]}")
            return None

# ========================================
# VIDEO DOWNLOADER (YouTube/Dailymotion)
# ========================================



# ========================================
# GEMINI VIDEO ANALYZER
# ========================================
class GeminiVideoAnalyzer:
    """Analyze videos with Gemini to extract event timeline"""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:generateContent"
        print("🤖 Initializing Gemini video analyzer...")
        print("✅ Ready!\n")

    def extract_keyframes(self, video_path: str, fps: float = 1.0) -> List[Dict]:
        """
        Extract keyframes from video with unique IDs

        Returns:
            List of dicts with frame metadata:
            [
                {
                    'id': 'kf_001234_abc123',
                    'frame_number': 1234,
                    'timestamp': 49.36,
                    'path': '/temp/kf_001234_abc123.jpg'
                },
                ...
            ]
        """
        print(f"🎬 Extracting keyframes from video...")
        print(f"   Rate: 1 frame per {1/fps:.1f} seconds")

        vr = VideoReader(video_path, ctx=cpu(0))
        video_fps = vr.get_avg_fps()
        total_frames = len(vr)
        duration_min = total_frames / video_fps / 60

        print(f"   📹 Video: {total_frames} frames, {video_fps} fps, {duration_min:.1f}min")

        # Calculate frame indices
        frame_interval = int(video_fps / fps)
        frame_indices = list(range(0, total_frames, frame_interval))

        print(f"   📸 Extracting {len(frame_indices)} keyframes...")

        # Create temp directory for keyframes
        import uuid
        temp_dir = f"temp_keyframes/{int(time.time())}"
        os.makedirs(temp_dir, exist_ok=True)

        keyframes = []

        for i, frame_idx in enumerate(frame_indices):
            if i % 50 == 0:
                print(f"   Progress: {i}/{len(frame_indices)} frames")

            # Generate unique ID
            frame_id = f"kf_{frame_idx:06d}_{uuid.uuid4().hex[:6]}"

            # Extract frame
            frame = vr[frame_idx].asnumpy()

            # Save frame
            frame_path = os.path.join(temp_dir, f"{frame_id}.jpg")
            Image.fromarray(frame).save(frame_path, quality=85)

            # Store metadata
            keyframes.append({
                'id': frame_id,
                'frame_number': frame_idx,
                'timestamp': frame_idx / video_fps,
                'path': frame_path
            })

        print(f"   ✅ Extracted {len(keyframes)} keyframes\n")

        # Print first 5 and last 5 for debugging
        print(f"   🔍 DEBUG: First 5 keyframes:")
        for kf in keyframes[:5]:
            print(f"      {kf['id']}: frame {kf['frame_number']} at {kf['timestamp']:.1f}s")

        print(f"   🔍 DEBUG: Last 5 keyframes:")
        for kf in keyframes[-5:]:
            print(f"      {kf['id']}: frame {kf['frame_number']} at {kf['timestamp']:.1f}s")
        print()

        return keyframes

    def analyze_video_chunks(self, keyframes: List[Dict], video_fps: float, chunk_size: int = 30) -> Dict:
        """
        Analyze video in chunks to build event timeline

        Args:
            keyframes: List of (frame_number, frame_path)
            video_fps: Video frames per second
            chunk_size: Number of keyframes per chunk

        Returns:
            Event timeline dictionary
        """
        print(f"🧠 Analyzing video with Gemini (Pass 1: Event Detection)...")
        print(f"   Total keyframes: {len(keyframes)}")
        print(f"   Chunk size: {chunk_size} keyframes per analysis\n")

        all_events = []

        # Process in chunks
        for chunk_idx in range(0, len(keyframes), chunk_size):
            chunk = keyframes[chunk_idx:chunk_idx + chunk_size]

            start_time = chunk[0]['timestamp']
            end_time = chunk[-1]['timestamp']

            # Extract paths from dicts
            paths = [kf['path'] for kf in chunk]

            # Add delay between chunks to avoid rate limiting (except first chunk)
            if chunk_idx > 0:
                time.sleep(3)  # 3 second delay between chunks

            print(f"📊 Analyzing chunk {chunk_idx//chunk_size + 1} ({start_time/60:.1f}min - {end_time/60:.1f}min)...")

            # Prepare prompt
            prompt = f"""You are watching a football/soccer match highlight video.

I'm showing you {len(chunk)} frames from {start_time:.1f}s to {end_time:.1f}s.

TASK: Identify ALL significant events in this segment:

**GOALS - BE VERY STRICT:**
- A goal ONLY counts if you clearly see the ball INSIDE THE NET
- If you see: ball saved, ball blocked, ball hit post, player celebrating but no ball in net → NOT a goal, use "shot_on_goal" or "near_miss"
- CRITICAL: Many frames show celebrations or reactions AFTER saves - these are NOT goals!
- Look for: Ball clearly crossed the goal line OR ball visibly in the back of the net

Other events:
- Shots (on target, blocked, missed)
- Passes (key passes, through balls)
- Fouls, handballs, penalties
- Yellow/red cards
- Celebrations
- Near misses
- Controversial moments

For EACH event, provide:
1. Frame number where it happens (from the frames I show you)
2. Approximate timestamp (in seconds from video start)
3. Event type (goal, shot, foul, etc.)
4. Description - MUST include:
   - **Player name or jersey number if visible** (especially for goals!)
   - Action details (how the goal was scored, where from, etc.)
   - Be specific: "Player #20 Palmer scores" NOT just "Chelsea scores"

CRITICAL:
- Frame numbers are: {[kf['frame_number'] for kf in chunk]}
- Only reference frames from this list
- Be specific about WHICH frame shows the event

Return ONLY valid JSON:
{{
  "events": [
    {{
      "frame_number": 1234,
      "timestamp_seconds": 45.2,
      "type": "shot_on_goal",
      "description": "Yamal shoots from outside the box, blocked by defender"
    }},
    {{
      "frame_number": 1567,
      "timestamp_seconds": 52.3,
      "type": "goal",
      "description": "Lewandowski scores from close range after rebound"
    }}
  ]
}}

If no events in this segment, return {{"events": []}}
"""

            try:
                # Send frames to Gemini
                frame_paths = [kf['path'] for kf in chunk]
                result_text = self._call_gemini_with_images(prompt, frame_paths, timeout=90)

                # Parse response with better error handling
                try:
                    if result_text.startswith("```"):
                        result_text = result_text.split("```")[1]
                        if result_text.startswith("json"):
                            result_text = result_text[4:]
                        result_text = result_text.strip()

                    result_text = result_text.replace('```', '').strip()
                    chunk_events = json.loads(result_text)
                except json.JSONDecodeError as e:
                    print(f"   ⚠️ JSON parse error: {str(e)[:100]}")
                    print(f"   📄 Response preview: {result_text[:200]}")

                    import re
                    json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
                    if json_match:
                        try:
                            chunk_events = json.loads(json_match.group(0))
                            print(f"   ✅ Recovered JSON from response")
                        except:
                            print(f"   ❌ Could not recover JSON, skipping chunk")
                            continue
                    else:
                        print(f"   ❌ No JSON found, skipping chunk")
                        continue

                events_in_chunk = chunk_events.get('events', [])

                # Add keyframe IDs to events
                for event in events_in_chunk:
                    # Find the matching keyframe by frame number
                    event_frame_num = event['frame_number']

                    # Look up keyframe with this frame number
                    matching_kf = None
                    for kf in chunk:
                        if kf['frame_number'] == event_frame_num:
                            matching_kf = kf
                            break

                    # Add keyframe_id to event
                    if matching_kf:
                        event['keyframe_id'] = matching_kf['id']
                    else:
                        # Fallback: use closest keyframe
                        closest_kf = min(chunk, key=lambda kf: abs(kf['frame_number'] - event_frame_num))
                        event['keyframe_id'] = closest_kf['id']
                        print(f"   ⚠️ Warning: Used closest keyframe for frame {event_frame_num}")

                all_events.extend(events_in_chunk)

                print(f"   ✅ Found {len(events_in_chunk)} events")
                for event in events_in_chunk:
                    kf_id = event.get('keyframe_id', 'unknown')
                    print(f"      • {event['timestamp_seconds']:.1f}s: {event['type']} - {event['description'][:60]}")
                    print(f"        [{kf_id}]")
                print()

            except Exception as e:
                print(f"   ❌ Chunk analysis failed: {str(e)[:100]}\n")
                continue

        print(f"✅ Pass 1 complete: Found {len(all_events)} total events\n")

        return {
            'events': all_events,
            'video_fps': video_fps
        }

    def find_exact_moment(self, video_path: str, event_timeline: List[Dict],
                     target_description: str, all_keyframes: List[Dict],
                     espn_context: str = "") -> Optional[Dict]:
        """
        Find exact frame for a specific action (Pass 2)

        Now uses keyframe_id from Pass 1 directly - no redundant scanning!

        Returns:
            {
                'frame_number': 1234,
                'timestamp': 45.2,
                'confidence': 85,
                'matched_event': {...}
            }
        """
        print(f"🎯 Pass 2: Finding exact moment for '{target_description[:60]}'...")

        # First, find similar events in timeline
        similar_events = self._find_similar_events(event_timeline, target_description, espn_context)

        if not similar_events:
            print(f"   ⚠️ No similar events found in timeline\n")
            return None

        print(f"   📋 Found {len(similar_events)} similar events:")
        for event in similar_events:
            print(f"      • {event['timestamp_seconds']:.1f}s: {event['description'][:60]}")
        print()

        # Use the best match from Pass 2
        best_event = similar_events[0]

        # Check if event has keyframe_id from Pass 1
        if 'keyframe_id' not in best_event:
            print(f"   ⚠️ Event missing keyframe_id from Pass 1\n")
            return None

        keyframe_id = best_event['keyframe_id']

        # Look up the keyframe from Pass 1
        keyframe_lookup = {kf['id']: kf for kf in all_keyframes}

        if keyframe_id not in keyframe_lookup:
            print(f"   ❌ Keyframe ID not found: {keyframe_id}\n")
            return None

        selected_kf = keyframe_lookup[keyframe_id]

        print(f"   ✅ Using keyframe from Pass 1: {keyframe_id}")
        print(f"   📊 Frame number: {selected_kf['frame_number']}")
        print(f"   ⏱️  Timestamp: {selected_kf['timestamp']:.1f}s")
        print(f"   💡 Event: {best_event['description'][:80]}\n")

        # Return the frame info directly from Pass 1
        return {
            'frame_number': selected_kf['frame_number'],
            'timestamp': selected_kf['timestamp'],
            'confidence': 95,  # High confidence since it's from Pass 1
            'matched_event': best_event,
            'keyframe_id': keyframe_id,
            'verification': {'source': 'pass1_keyframe', 'description': best_event['description']}
        }

    def _find_similar_events(self, event_timeline: List[Dict], target_description: str,
                         espn_context: str = "") -> List[Dict]:
        """
        Find events in timeline that match target description
        Now uses ESPN match data for better matching
        """
        # VALIDATE INPUTS
        if not isinstance(event_timeline, list):
            print(f"   ❌ ERROR: event_timeline is {type(event_timeline)}, expected list")
            print(f"   📄 Content: {str(event_timeline)[:200]}")
            return []

        if not event_timeline:
            return []

        print(f"   📋 Searching {len(event_timeline)} events for: {target_description[:60]}...")
        if not event_timeline:
            return []

        print(f"   📋 Searching {len(event_timeline)} events for: {target_description[:60]}...")

        # Build timeline summary
        timeline_str = "\n".join([
            f"• {e['timestamp_seconds']:.1f}s: {e['description']}"
            for e in event_timeline
        ])

        prompt = f"""You are matching a TARGET action to events in a video timeline.

    TARGET: {target_description}

    CRITICAL MATCHING RULES:
    1. **PLAYER NAME is most important** - If the target mentions a specific player (e.g., "Cole Palmer"), ONLY match goals by that player
    2. **IGNORE the score in the target** - Scores like "2-0" or "3-1" are FINAL SCORES, not the score when this goal happened
    3. For example: "Cole Palmer goal Team A vs Team B 2-0" means find Cole Palmer's goal, even if it made the score 1-0

    {espn_context}

    VIDEO TIMELINE:
    {timeline_str}

    TASK: Find events that could match the TARGET.

    CRITICAL INSTRUCTIONS:
    1. **READ THROUGH ALL EVENTS BELOW** - Do NOT pick the first matching event you see
    2. **COMPARE ALL OPTIONS** - Look at every goal, shot, and action in the timeline
    3. **MATCH BY PLAYER NAME FIRST** - If target says "Cole Palmer goal", find events mentioning Cole Palmer or #20
    4. **VERIFY THE ACTION TYPE** - Goal vs shot vs near miss etc.
    5. **RANK YOUR MATCHES** - Return top 3-5 in order of confidence

    {"IMPORTANT: Use the official match data above to identify the CORRECT goal. Don't just pick the first goal - pick the one that matches the scorer and goal number." if espn_context else ""}

    PROCESS:
    - Step 1: Read all {len(event_timeline)} events in the timeline below
    - Step 2: Identify which events match the target action
    - Step 3: Rank them by how well they match (player name, action type, context)
    - Step 4: Return top 3-5 matches with confidence scores

    Return the top 3-5 matching events (or fewer if not relevant).

    Return ONLY valid JSON:
    {{
      "matches": [
        {{"timestamp": 547.0, "description": "Goal scored by Villa. Number 10, Buendia...", "confidence": 95}},
        {{"timestamp": 251.0, "description": "Aston Villa player scores a goal", "confidence": 60}}
      ]
    }}
    """

        try:
            result_text = self._call_gemini(prompt, timeout=30)

            try:
                if result_text.startswith("```"):
                    result_text = result_text.split("```")[1]
                    if result_text.startswith("json"):
                        result_text = result_text[4:]
                    result_text = result_text.strip()

                result_text = result_text.replace('```', '').strip()

                print(f"   🔍 DEBUG: Attempting to parse JSON...")
                print(f"   📄 Response length: {len(result_text)} chars")
                print(f"   📄 First 100 chars: {result_text[:100]}")

                matches = json.loads(result_text)

                print(f"   ✅ JSON parsed successfully")
                print(f"   📊 Type: {type(matches)}")
                print(f"   📊 Keys: {matches.keys() if isinstance(matches, dict) else 'NOT A DICT'}")

            except json.JSONDecodeError as e:
                print(f"   ⚠️ Event matching JSON parse error: {str(e)[:100]}")
                print(f"   📄 Response preview: {result_text[:200]}")

                # Try to extract JSON manually
                import re
                json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
                if json_match:
                    try:
                        matches = json.loads(json_match.group(0))
                        print(f"   ✅ Recovered JSON from response")
                    except:
                        print(f"   ❌ Could not recover JSON")
                        return []
                else:
                    print(f"   ❌ No JSON found in response")
                    return []

            # Validate matches is a dict
            if not isinstance(matches, dict):
                print(f"   ❌ Invalid response type: {type(matches)} - Expected dict")
                print(f"   📄 Content: {str(matches)[:200]}")
                return []

            # Convert to event format
            matched_events = []
            for match in matches.get('matches', []):
                # Find the original event
                for event in event_timeline:
                    if abs(event['timestamp_seconds'] - match['timestamp']) < 2.0:
                        matched_events.append(event)
                        break

            print(f"   📋 Found {len(matched_events)} similar events:")
            for e in matched_events[:5]:
                print(f"      • {e['timestamp_seconds']:.1f}s: {e['description'][:60]}")
            print()

            return matched_events

        except Exception as e:
            print(f"   ⚠️ Event matching failed: {str(e)[:100]}")
            import traceback
            print(f"   🔍 Traceback: {traceback.format_exc()[:300]}")
            return []

    def _detailed_frame_scan(self, video_path: str, event: Dict, target_description: str,
                        all_keyframes: List[Dict]) -> Optional[Dict]:
        """
        Find exact frame using keyframes from Pass 1 (no re-extraction needed!)

        Args:
            video_path: Path to video
            event: Event dict from timeline
            target_description: Description to find
            all_keyframes: All keyframes from Pass 1 with IDs

        Returns:
            {'frame_number': 1234, 'timestamp': 49.36, 'confidence': 90, ...}
        """
        print(f"   🔍 Detailed scan using Pass 1 keyframes...")

        # Get event timestamp
        event_time = event['timestamp_seconds']

        # Find keyframes in ±10 second window around event
        window_seconds = 10
        start_time = max(0, event_time - window_seconds)
        end_time = event_time + window_seconds

        relevant_keyframes = [
            kf for kf in all_keyframes
            if start_time <= kf['timestamp'] <= end_time
        ]

        print(f"   📊 Found {len(relevant_keyframes)} keyframes in window ({start_time:.1f}s - {end_time:.1f}s)")
        print(f"   🔍 DEBUG: Keyframes in scan window:")
        for kf in relevant_keyframes:
            print(f"      {kf['id']}: frame {kf['frame_number']} at {kf['timestamp']:.1f}s")
        print()

        if not relevant_keyframes:
            print(f"   ❌ No keyframes found in window\n")
            return None

        # Build keyframe list for Gemini
        keyframe_list_str = "\n".join([
            f"- {kf['id']} (frame {kf['frame_number']}, at {kf['timestamp']:.1f}s)"
            for kf in relevant_keyframes
        ])

        prompt = f"""TARGET ACTION: {target_description}

    You ALREADY analyzed these frames in Pass 1. Now identify which EXACT frame shows this specific action.

    AVAILABLE KEYFRAMES (you've seen all of these):
    {keyframe_list_str}

    TASK: Find the keyframe that shows the EXACT moment of "{target_description}".

    CRITICAL:
    - You MUST select a keyframe ID from the list above
    - Match the SPECIFIC player/action mentioned in the target (e.g., "Cole Palmer scoring")
    - Pick the frame showing the ACTUAL action (ball going in, contact, shot), NOT celebration
    - If multiple frames show similar actions, pick the one matching the specific player/description
    - If the action spans multiple frames, pick the PEAK moment (ball crossing line, contact moment)

    Return ONLY valid JSON:
    {{
      "keyframe_id": "kf_006234_abc123",
      "confidence": 90,
      "what_i_see": "Ball crossing the goal line",
      "reasoning": "This frame shows the exact moment the ball enters the net"
    }}
    """

        try:
            # Send keyframe images to Gemini
            paths = [kf['path'] for kf in relevant_keyframes]

            print(f"   🤖 Asking Gemini to identify exact keyframe...")
            result_text = self._call_gemini_with_images(prompt, paths, timeout=60)

            # Parse response
            if result_text.startswith("```"):
                result_text = result_text.split("```")[1]
                if result_text.startswith("json"):
                    result_text = result_text[4:]
                result_text = result_text.strip()

            result_text = result_text.replace('```', '').strip()

            try:
                result = json.loads(result_text)
            except json.JSONDecodeError as e:
                print(f"   ⚠️ JSON parse error: {str(e)[:100]}")
                import re
                json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
                if json_match:
                    result = json.loads(json_match.group(0))
                else:
                    raise Exception("Could not parse Gemini response")

            # Extract keyframe ID
            keyframe_id = result['keyframe_id']
            confidence = result.get('confidence', 0)

            # Look up keyframe
            keyframe_lookup = {kf['id']: kf for kf in relevant_keyframes}

            if keyframe_id not in keyframe_lookup:
                print(f"   ❌ Gemini returned invalid keyframe_id: {keyframe_id}")
                return None

            selected_kf = keyframe_lookup[keyframe_id]

            print(f"   ✅ Found exact keyframe: {keyframe_id}")
            print(f"   📊 Frame number: {selected_kf['frame_number']}")
            print(f"   ⏱️  Timestamp: {selected_kf['timestamp']:.1f}s")
            print(f"   🎯 Confidence: {confidence}%")
            print(f"   👁️  Gemini sees: {result.get('what_i_see', 'N/A')}")
            print(f"   💡 Reasoning: {result.get('reasoning', 'N/A')}\n")

            return {
                'frame_number': selected_kf['frame_number'],
                'timestamp': selected_kf['timestamp'],
                'confidence': confidence,
                'matched_event': event,
                'verification': result,
                'keyframe_id': keyframe_id
            }

        except Exception as e:
            print(f"   ❌ Detailed scan failed: {str(e)[:100]}\n")
            return None

    def extract_3frame_sequence(self, video_path: str, center_frame: int, description: str) -> List[str]:
        """
        Extract 3-frame sequence using Gemini to pick best frames for story progression
        Uses unique frame IDs to prevent extraction mismatches

        Args:
            video_path: Path to video file
            center_frame: Approximate center frame from event detection
            description: Description of the action to find

        Returns:
            List of 3 frame paths [before, during, after]
        """
        print(f"📸 Extracting 3-frame sequence around frame {center_frame}...")

        vr = VideoReader(video_path, ctx=cpu(0))
        video_fps = vr.get_avg_fps()

        # Extract frames in a window around center (±5 seconds = ±125 frames at 25fps)
        window_size = int(video_fps * 5)
        start_frame = max(0, center_frame - window_size)
        end_frame = min(len(vr), center_frame + window_size)

        # Extract every 15th frame in this window (about 17 frames total)
        frame_indices = list(range(start_frame, end_frame, 15))

        print(f"   📹 Extracting {len(frame_indices)} candidate frames from window...")

        temp_dir = f"temp_3frame_selection/{int(time.time())}"
        os.makedirs(temp_dir, exist_ok=True)

        # Extract frames with unique IDs
        import uuid
        frame_data = []

        for frame_idx in frame_indices:
            # Generate unique ID
            frame_id = f"frame_{frame_idx:06d}_{uuid.uuid4().hex[:6]}"

            # Extract and save frame
            frame = vr[frame_idx].asnumpy()
            frame_path = os.path.join(temp_dir, f"{frame_id}.jpg")
            Image.fromarray(frame).save(frame_path, quality=85)

            # Store frame metadata
            frame_data.append({
                'id': frame_id,
                'frame_number': frame_idx,
                'timestamp': frame_idx / video_fps,
                'path': frame_path
            })

        print(f"   🤖 Asking Gemini to select best 3-frame story progression...")

        # Build frame list for Gemini
        frame_list_str = "\n".join([
            f"- {f['id']} (frame {f['frame_number']}, at {f['timestamp']:.1f}s)"
            for f in frame_data
        ])

        # Build prompt for Gemini
        prompt = f"""You are selecting 3 frames from a football/soccer video to tell a visual story.

    TARGET ACTION: {description}

    I'm showing you {len(frame_data)} frames from around the moment this action happens.

    AVAILABLE FRAMES:
    {frame_list_str}

    TASK: Select exactly 3 frames that show the PROGRESSION of this action:

    1. **BEFORE/SETUP** - The moment BEFORE the main action
      - Player receiving the ball
      - Player preparing to shoot/pass/tackle
      - Positioning before the action

    2. **DURING/CLIMAX** - The main action itself
      - Ball being struck
      - Player making contact
      - The peak moment of the action

    3. **AFTER/RESULT** - The immediate result
      - Ball in the net / saved / blocked
      - Immediate reaction (not full celebration)
      - Result of the action

    CRITICAL RULES:
    - You MUST select frame IDs from the list above - do NOT make up frame numbers
    - The 3 frames must be in chronological order (before < during < after)
    - DURING frame should be the actual action moment, not celebration
    - AFTER frame should show the immediate result, not extended celebration

    Return ONLY valid JSON with the exact frame IDs from the list:
    {{
      "before_frame_id": "frame_001234_abc123",
      "during_frame_id": "frame_001250_def456",
      "after_frame_id": "frame_001265_ghi789",
      "reasoning": "Before shows player receiving pass, during shows the shot, after shows ball in net"
    }}
    """

        try:
            # Send frame images to Gemini
            image_paths = [f['path'] for f in frame_data]
            result_text = self._call_gemini_with_images(prompt, image_paths, timeout=60)

            # Parse response
            if result_text.startswith("```"):
                result_text = result_text.split("```")[1]
                if result_text.startswith("json"):
                    result_text = result_text[4:]
                result_text = result_text.strip()

            result_text = result_text.replace('```', '').strip()

            try:
                selection = json.loads(result_text)
            except json.JSONDecodeError as e:
                print(f"   ⚠️ JSON parse error: {str(e)[:100]}")
                import re
                json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
                if json_match:
                    selection = json.loads(json_match.group(0))
                else:
                    raise Exception("Could not parse Gemini response")

            # Extract frame IDs from response
            before_id = selection['before_frame_id']
            during_id = selection['during_frame_id']
            after_id = selection['after_frame_id']

            print(f"   ✅ Gemini selected:")
            print(f"      Before: {before_id}")
            print(f"      During: {during_id}")
            print(f"      After: {after_id}")
            print(f"   💡 Reasoning: {selection.get('reasoning', 'N/A')}\n")

            # Look up frame data by ID
            frame_lookup = {f['id']: f for f in frame_data}

            # Validate IDs exist
            if before_id not in frame_lookup:
                raise Exception(f"Gemini returned invalid before_frame_id: {before_id}")
            if during_id not in frame_lookup:
                raise Exception(f"Gemini returned invalid during_frame_id: {during_id}")
            if after_id not in frame_lookup:
                raise Exception(f"Gemini returned invalid after_frame_id: {after_id}")

            before_data = frame_lookup[before_id]
            during_data = frame_lookup[during_id]
            after_data = frame_lookup[after_id]

            print(f"   📊 Validated frame numbers:")
            print(f"      Before: frame {before_data['frame_number']} at {before_data['timestamp']:.1f}s")
            print(f"      During: frame {during_data['frame_number']} at {during_data['timestamp']:.1f}s")
            print(f"      After: frame {after_data['frame_number']} at {after_data['timestamp']:.1f}s")

            # Create final output directory
            final_temp_dir = f"temp_final_3frames/{int(time.time())}"
            os.makedirs(final_temp_dir, exist_ok=True)

            # Copy the selected frames (no re-extraction needed!)
            final_paths = []
            for label, frame_info in [('before', before_data), ('during', during_data), ('after', after_data)]:
                final_path = os.path.join(final_temp_dir, f"frame_{label}.png")
                shutil.copy(frame_info['path'], final_path)
                final_paths.append(final_path)
                print(f"   ✅ Copied {label} frame from {frame_info['path']}")

            # Cleanup candidate frames
            shutil.rmtree(temp_dir, ignore_errors=True)

            return final_paths

        except Exception as e:
            print(f"   ❌ Gemini selection failed: {str(e)[:100]}")
            print(f"   🔄 Falling back to time-based extraction...")

            # Fallback: use original dumb method
            before_frame = max(0, int(center_frame - video_fps * 2))
            during_frame = center_frame
            after_frame = min(len(vr) - 1, int(center_frame + video_fps * 2))

            final_temp_dir = f"temp_final_3frames/{int(time.time())}"
            os.makedirs(final_temp_dir, exist_ok=True)

            final_paths = []
            for label, frame_idx in [('before', before_frame), ('during', during_frame), ('after', after_frame)]:
                frame = vr[frame_idx].asnumpy()
                frame_path = os.path.join(final_temp_dir, f"frame_{label}.png")
                Image.fromarray(frame).save(frame_path)
                final_paths.append(frame_path)

            shutil.rmtree(temp_dir, ignore_errors=True)
            return final_paths
    def _call_gemini(self, prompt: str, timeout: int = 30) -> str:
        """Call Gemini text API"""
        url = f"{self.base_url}?key={self.api_key}"

        payload = {
            "contents": [{
                "parts": [{"text": prompt}]
            }],
            "generationConfig": {
                "temperature": 0.7,
                "maxOutputTokens": 2000
            }
        }

        response = requests.post(url, json=payload, timeout=timeout)

        if response.status_code == 200:
            result = response.json()
            return result['candidates'][0]['content']['parts'][0]['text'].strip()
        else:
            raise Exception(f"API error: {response.status_code}")

    def _call_gemini_with_images(self, prompt: str, image_paths: List[str], timeout: int = 90) -> str:
        """Call Gemini with images"""
        url = f"{self.base_url}?key={self.api_key}"

        parts = [{"text": prompt}]

        for img_path in image_paths:
            with open(img_path, 'rb') as f:
                image_data = base64.b64encode(f.read()).decode('utf-8')

            mime_type = 'image/jpeg' if img_path.endswith('.jpg') else 'image/png'

            parts.append({
                "inline_data": {
                    "mime_type": mime_type,
                    "data": image_data
                }
            })

        payload = {
            "contents": [{
                "parts": parts
            }],
            "generationConfig": {
                "temperature": 0.7,
                "maxOutputTokens": 3000
            }
        }

        response = requests.post(url, json=payload, timeout=timeout)

        if response.status_code == 200:
            result = response.json()
            return result['candidates'][0]['content']['parts'][0]['text'].strip()
        else:
            raise Exception(f"API error: {response.status_code}")

    def extract_jersey_colors(self, video_path: str, sample_frames: List[Dict]) -> Dict[str, str]:
        """
        Extract jersey colors from video frames

        Args:
            video_path: Path to video
            sample_frames: List of (frame_number, frame_path) tuples

        Returns:
            {"home": "claret and blue", "away": "white"}
        """
        print(f"👕 Extracting jersey colors from video...")

        # Use first 5 sample frames
        sample_paths = [kf['path'] for kf in sample_frames[:5]]

        prompt = f"""You are identifying team jersey colors in a football/soccer match.

    I'm showing you {len(sample_paths)} frames from the match.

    TASK: Identify the two teams' jersey colors.

    Return the PRIMARY colors (ignore details like stripes/logos):
    - Home team (usually darker/colorful)
    - Away team (usually white/light)

    Examples of good responses:
    - {{"home": "red", "away": "white"}}
    - {{"home": "blue", "away": "yellow"}}
    - {{"home": "claret and blue", "away": "white"}}

    Return ONLY valid JSON:
    {{
      "home": "primary jersey color",
      "away": "primary jersey color"
    }}
    """

        try:
            result_text = self._call_gemini_with_images(prompt, sample_paths, timeout=30)

            if result_text.startswith("```"):
                result_text = result_text.split("```")[1]
                if result_text.startswith("json"):
                    result_text = result_text[4:]
                result_text = result_text.strip()

            result_text = result_text.replace('```', '').strip()
            colors = json.loads(result_text)

            print(f"   ✅ Home: {colors['home']}, Away: {colors['away']}\n")
            return colors

        except Exception as e:
            print(f"   ⚠️ Jersey color extraction failed: {str(e)[:100]}")
            return {"home": "unknown", "away": "unknown"}


# ========================================
# 1. Video Plan Parser
# ========================================
class PNGSequencePlanParser:
    """Parse video plan TXT files for PNG sequence sections"""

    def __init__(self):
        print("✅ PNGSequencePlanParser initialized\n")

    def parse_plan(self, file_path: str) -> tuple[str, List[PNGSequenceSection]]:
        """
        Parse video plan file for PNG sequence sections

        Returns:
            (script_name, list of PNGSequenceSection objects)
        """
        print(f"📄 Parsing video plan: {file_path}")

        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Extract script name from filename
        script_name = Path(file_path).stem

        # Parse each line for PNG sequence sections
        sections = []
        lines = content.strip().split('\n')

        for line_num, line in enumerate(lines, 1):
            if not line.strip():
                continue

            # Pattern: 5.23-8.45: PNG sequence of ... [transcript: ...]
            match = re.match(
                r'([\d.]+)-([\d.]+):\s*PNG sequence of\s+(.+?)\s*\[transcript:\s*(.+?)\]',
                line,
                re.IGNORECASE
            )

            if match:
                start_time = float(match.group(1))
                end_time = float(match.group(2))
                description = match.group(3).strip()
                transcript = match.group(4).strip()

                section = PNGSequenceSection(
                    start_time=start_time,
                    end_time=end_time,
                    description=description,
                    transcript=transcript,
                    line_number=line_num,
                    script_name=script_name
                )
                sections.append(section)

        print(f"   ✅ Found {len(sections)} PNG sequence sections\n")

        return script_name, sections


# ========================================
# GOOGLE IMAGES SCRAPER (SELENIUM) - STEALTH MODE
# ========================================
import random
import time

# ========================================
# GOOGLE IMAGES SCRAPER (PLAYWRIGHT) - COLAB OPTIMIZED
# ========================================
import random
import time

class GoogleImagesScraper:
    """Google Images scraper using Playwright (Colab-friendly)"""

    def __init__(self):
        """Initialize scraper"""
        print("🌐 Initializing Google Images scraper (Playwright)...")

        # Quality thresholds
        self.min_width = 800
        self.min_height = 800
        self.min_file_size = 50000  # 50KB minimum

        # Track requests to add delays
        self.last_request_time = 0
        self.request_count = 0

        # Install Playwright if not already
        self._setup_playwright()

        print("✅ Scraper ready!\n")

    def _setup_playwright(self):
        """Setup Playwright"""
        try:
            import asyncio
            from playwright.async_api import async_playwright
            self.playwright_available = True
        except ImportError:
            print("Installing Playwright...")
            import os
            os.system('pip install playwright')
            os.system('playwright install chromium')
            os.system('playwright install-deps chromium')
            import asyncio
            from playwright.async_api import async_playwright
            self.playwright_available = True

    def _human_delay(self, min_sec=1.0, max_sec=3.0):
        """Random human-like delay"""
        time.sleep(random.uniform(min_sec, max_sec))

    def _rate_limit(self):
        """Ensure we don't make requests too quickly"""
        time_since_last = time.time() - self.last_request_time
        if time_since_last < 3:
            wait_time = random.uniform(3, 5) - time_since_last
            if wait_time > 0:
                time.sleep(wait_time)

        self.last_request_time = time.time()
        self.request_count += 1

        if self.request_count % 5 == 0:
            time.sleep(random.uniform(10, 15))

    def search_and_get_full_res_urls(self, query: str, max_images: int = 20, max_retries: int = 3) -> List[Dict]:
        """
        Search Google Images and extract image URLs using Playwright

        Returns:
            List of dicts with 'url', 'width', 'height'
        """
        print(f"🔍 Searching Google Images: '{query}'")

        # Rate limiting
        self._rate_limit()

        for attempt in range(max_retries):
            try:
                print(f"   🚀 Fetching results (attempt {attempt + 1}/{max_retries})...")

                # Run async function
                import asyncio
                from playwright.async_api import async_playwright

                async def fetch_images():
                    async with async_playwright() as p:
                        browser = await p.chromium.launch(
                            headless=True,
                            args=['--no-sandbox', '--disable-setuid-sandbox']
                        )

                        context = await browser.new_context(
                            user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36'
                        )

                        page = await context.new_page()

                        # Navigate to Google Images
                        search_url = f"https://www.google.com/search?q={query.replace(' ', '+')}&tbm=isch"
                        await page.goto(search_url)

                        # Wait for images to load
                        await page.wait_for_timeout(2000)

                        # Scroll to load more images
                        for _ in range(3):
                            await page.evaluate('window.scrollBy(0, window.innerHeight)')
                            await page.wait_for_timeout(1000)

                        # Get page content
                        content = await page.content()

                        await browser.close()

                        return content

                # Get page content
                page_source = asyncio.run(fetch_images())

                print(f"   ✅ Page loaded successfully")

                # Extract image URLs using regex
                import re

                patterns = [
                    r'"(https://[^"]+\.(?:jpg|jpeg|png|webp)[^"]*)"',
                    r'https://[^\s"<>]+\.(?:jpg|jpeg|png|webp)',
                ]

                all_urls = []
                for pattern in patterns:
                    matches = re.findall(pattern, page_source)
                    all_urls.extend(matches)

                # Filter and deduplicate
                full_res_urls = []
                seen = set()

                skip_patterns = [
                    'gstatic',
                    'google.com/images',
                    'googleapis',
                    'googleusercontent',
                    'encrypted-tbn',
                    'data:image',
                ]

                for url in all_urls:
                    url = url.replace('\\u003d', '=').replace('\\u0026', '&')

                    if any(pattern in url.lower() for pattern in skip_patterns):
                        continue

                    if url in seen or len(url) < 20:
                        continue

                    seen.add(url)
                    full_res_urls.append({
                        'url': url,
                        'width': 0,
                        'height': 0
                    })

                    if len(full_res_urls) >= max_images:
                        break

                if len(full_res_urls) == 0:
                    raise Exception("No image URLs extracted")

                print(f"   ✅ Found {len(full_res_urls)} image URLs\n")
                return full_res_urls

            except Exception as e:
                print(f"   ❌ Attempt {attempt + 1} failed: {str(e)[:100]}")

                if attempt < max_retries - 1:
                    wait_time = random.uniform(3, 6) * (attempt + 1)
                    print(f"   ⏳ Waiting {wait_time:.1f}s before retry...")
                    time.sleep(wait_time)
                else:
                    print(f"   ❌ All {max_retries} attempts failed!")
                    return []

        return []

    def download_image(self, url: str, output_path: str) -> Tuple[bool, Dict]:
        """
        Download image and check quality requirements

        Returns:
            (success, metadata_dict)
        """
        stock_domains = [
            'alamy.com', 'shutterstock.com', 'gettyimages.com',
            'istockphoto.com', 'adobestock.com', 'dreamstime.com',
            '123rf.com', 'depositphotos.com', 'pond5.com',
            'bigstockphoto.com', 'canstockphoto.com', 'stocksy.com'
        ]

        url_lower = url.lower()
        for domain in stock_domains:
            if domain in url_lower:
                return False, {'error': f'stock_site_{domain}'}

        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36',
                'Accept': 'image/avif,image/webp,image/apng,image/*,*/*;q=0.8',
                'Referer': 'https://www.google.com/',
            }

            time.sleep(random.uniform(0.1, 0.3))

            response = requests.get(url, headers=headers, timeout=15, stream=True)

            if response.status_code != 200:
                return False, {'error': f'HTTP {response.status_code}'}

            img = Image.open(io.BytesIO(response.content))

            if img.width < self.min_width or img.height < self.min_height:
                return False, {
                    'error': 'too_small',
                    'width': img.width,
                    'height': img.height
                }

            file_size = len(response.content)
            if file_size < self.min_file_size:
                return False, {'error': 'file_too_small', 'size': file_size}

            if img.mode == 'RGBA':
                background = Image.new('RGB', img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[-1])
                img = background
            elif img.mode not in ['RGB', 'L']:
                img = img.convert('RGB')

            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            img.save(output_path, 'PNG', optimize=True)

            metadata = {
                'width': img.width,
                'height': img.height,
                'file_size': file_size,
                'url': url
            }

            return True, metadata

        except Exception as e:
            return False, {'error': str(e)}

    def close(self):
        """Close browser (no-op for Playwright)"""
        pass



# ========================================
# GEMINI ANALYZER FOR IMAGES (REST API)
# ========================================
import base64

class GeminiImageAnalyzer:
    """Analyze and select best images using Gemini via REST API"""

    def __init__(self, api_key: str):
        print("🤖 Initializing Gemini image analyzer (REST API)...")
        self.api_key = api_key
        self.base_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:generateContent"
        print("✅ Gemini initialized!\n")

    def _call_gemini(self, prompt: str, timeout: int = 30, max_retries: int = 3) -> str:
        """Call Gemini REST API with timeout and rate limit retry"""
        url = f"{self.base_url}?key={self.api_key}"

        payload = {
            "contents": [{
                "parts": [{"text": prompt}]
            }],
            "generationConfig": {
                "temperature": 0.7,
                "maxOutputTokens": 1000
            }
        }

        for attempt in range(max_retries):
            try:
                response = requests.post(url, json=payload, timeout=timeout)

                if response.status_code == 200:
                    result = response.json()
                    text = result['candidates'][0]['content']['parts'][0]['text']
                    return text.strip()
                elif response.status_code == 429:
                    # Rate limited - exponential backoff
                    wait_time = (2 ** attempt) * 5  # 5s, 10s, 20s
                    if attempt < max_retries - 1:
                        print(f"   ⏳ Rate limited, waiting {wait_time}s before retry {attempt + 2}/{max_retries}...")
                        time.sleep(wait_time)
                        continue
                    else:
                        raise Exception("Rate limited")
                elif response.status_code == 403:
                    raise Exception("API key invalid")
                else:
                    raise Exception(f"API error: {response.status_code}")

            except requests.exceptions.Timeout:
                if attempt < max_retries - 1:
                    print(f"   ⏳ Timeout, retrying {attempt + 2}/{max_retries}...")
                    time.sleep(2)
                    continue
                else:
                    raise Exception("Request timed out")

        raise Exception("Max retries exceeded")

    def _call_gemini_with_images(self, prompt: str, image_paths: List[str], timeout: int = 60, max_retries: int = 3) -> str:
        """Call Gemini with actual images via REST API with rate limit retry"""
        url = f"{self.base_url}?key={self.api_key}"

        # Build parts with prompt and images
        parts = [{"text": prompt}]

        successful_images = 0
        for i, img_path in enumerate(image_paths, 1):
            try:
                # Verify file exists
                if not os.path.exists(img_path):
                    print(f"   ⚠️ Image {i} not found: {img_path}")
                    continue

                # Verify file is readable
                file_size = os.path.getsize(img_path)
                if file_size == 0:
                    print(f"   ⚠️ Image {i} is empty (0 bytes)")
                    continue

                # Read image and convert to base64
                with open(img_path, 'rb') as f:
                    image_data = base64.b64encode(f.read()).decode('utf-8')

                # Determine mime type
                if img_path.endswith('.png'):
                    mime_type = 'image/png'
                elif img_path.endswith('.jpg') or img_path.endswith('.jpeg'):
                    mime_type = 'image/jpeg'
                else:
                    mime_type = 'image/png'

                parts.append({
                    "inline_data": {
                        "mime_type": mime_type,
                        "data": image_data
                    }
                })
                successful_images += 1

            except Exception as e:
                print(f"   ⚠️ Failed to load image {i}: {str(e)[:50]}")
                continue

        if successful_images == 0:
            raise Exception("No images could be loaded")

        if successful_images < len(image_paths):
            print(f"   ⚠️ Only {successful_images}/{len(image_paths)} images loaded successfully")

        payload = {
            "contents": [{
                "parts": parts
            }],
            "generationConfig": {
                "temperature": 0.7,
                "maxOutputTokens": 2000
            }
        }

        # Retry loop with exponential backoff
        for attempt in range(max_retries):
            try:
                response = requests.post(url, json=payload, timeout=timeout)

                if response.status_code == 200:
                    result = response.json()
                    return result['candidates'][0]['content']['parts'][0]['text'].strip()
                elif response.status_code == 429:
                    # Rate limited - exponential backoff
                    wait_time = (2 ** attempt) * 5  # 5s, 10s, 20s
                    if attempt < max_retries - 1:
                        print(f"   ⏳ Rate limited, waiting {wait_time}s before retry {attempt + 2}/{max_retries}...")
                        time.sleep(wait_time)
                        continue
                    else:
                        raise Exception("API error: 429 (rate limited)")
                else:
                    raise Exception(f"API error: {response.status_code}")

            except requests.exceptions.Timeout:
                if attempt < max_retries - 1:
                    print(f"   ⏳ Timeout, retrying {attempt + 2}/{max_retries}...")
                    time.sleep(2)
                    continue
                else:
                    raise Exception("Request timed out")

        raise Exception("Max retries exceeded")

    def suggest_alternative_query(self, original_query: str, section: PNGSequenceSection) -> str:
        """Ask Gemini for alternative search query when first attempt fails"""
        print(f"🔄 Asking Gemini for alternative search...")

        prompt = f"""The original search found no suitable images.

Original query: "{original_query}"
Target: {section.description}
Context: {section.transcript}

Suggest a DIFFERENT search query that might work better.
Try:
- More general terms
- Different keywords
- Keep it under 8 words

Return ONLY the new search query, no explanation.
"""

        try:
            alt_query = self._call_gemini(prompt, timeout=20)
            alt_query = alt_query.strip().strip('"\'').strip('.')
            print(f"   💡 Alternative: '{alt_query}'\n")
            return alt_query

        except Exception as e:
            print(f"   ❌ Error: {str(e)[:50]}")
            # Fallback: simplify query
            words = original_query.split()[:5]
            alt_query = ' '.join(words)
            print(f"   💡 Fallback alternative: '{alt_query}'\n")
            return alt_query

    def analyze_image_batch(self, candidates: List[ImageCandidate], section: PNGSequenceSection) -> Dict:
        """
        Analyze actual images with visual validation (for single image fallback)
        """
        print(f"🤖 Gemini analyzing {len(candidates)} candidates (with visual inspection)...")

        prompt = f"""You are an expert image analyst selecting the BEST image.

TARGET REQUIREMENTS:
- Description: {section.description}
- Context: {section.transcript}

SELECTION PRIORITIES (in order):
1. **Subject accuracy** - Must match the description perfectly
2. **NO STOCK WATERMARKS** - REJECT any images with watermarks from: Alamy, Shutterstock, Getty Images, iStock, Adobe Stock, Dreamstime, 123RF, Depositphotos, or any other stock photo watermarks
3. **Minimal text overlays** - Avoid images with excessive text, graphics, or overlays (some text is OK)
4. **Image quality** - Sharp, clear, well-lit, good resolution
5. **Relevance to context** - Fits the transcript mood/tone

CRITICAL: If an image has ANY stock photo watermark, give it a score of 0 and mark it as rejected.

Here are {len(candidates)} candidate images to evaluate (attached below).

For EACH image, provide:
1. A score from 0-100 (0 if stock watermark detected)
2. Brief analysis (mention if watermark detected)

Then select the BEST one overall (never select one with watermark).

Return ONLY valid JSON (no markdown):
{{
  "evaluations": [
    {{"image": 1, "score": 85, "analysis": "Good match, clean image, no watermarks"}},
    {{"image": 2, "score": 0, "analysis": "REJECTED: Shutterstock watermark detected"}}
  ],
  "best_image": 1,
  "confidence": 85,
  "reason": "Best match with no watermarks"
}}
"""

        try:
            # Send images to Gemini
            image_paths = [c.path for c in candidates]
            result_text = self._call_gemini_with_images(prompt, image_paths, timeout=60)

            # Clean and parse
            if result_text.startswith("```"):
                result_text = result_text.split("```")[1]
                if result_text.startswith("json"):
                    result_text = result_text[4:]
                result_text = result_text.strip()

            analysis = json.loads(result_text)

            print(f"\n   📊 Gemini's Evaluation:")
            for eval in analysis['evaluations']:
                print(f"      Image {eval['image']}: {eval['score']}% - {eval['analysis']}")

            print(f"\n   ✅ Best: Image {analysis['best_image']} (Confidence: {analysis['confidence']}%)")
            print(f"   💡 Reason: {analysis['reason']}\n")

            return analysis

        except Exception as e:
            print(f"   ⚠️ Gemini analysis failed: {str(e)[:100]}")

            # Fallback: pick highest resolution
            print(f"   🔄 Fallback: Using highest resolution image")
            best_idx = max(range(len(candidates)), key=lambda i: candidates[i].width * candidates[i].height)
            return {
                'best_image': best_idx + 1,
                'confidence': 50,
                'reason': f'Fallback: highest resolution'
            }

    def select_3_image_sequence(self, candidates: List[ImageCandidate], section: PNGSequenceSection,
                           match_context: Optional[MatchContext] = None,
                           jersey_colors: Optional[Dict[str, str]] = None) -> Dict:
        """
        Select 3 images that tell a story: beginning → middle → end

        Returns:
            {
                'selected_images': [idx1, idx2, idx3],
                'confidence': 85,
                'narrative_quality': 90,
                'match_continuity_verified': True,
                'jersey_colors': {'home': 'blue and red', 'away': 'white'},
                'stadium': 'Camp Nou',
                'reason': '...'
            }
        """
        print(f"🤖 Gemini selecting 3-image sequence from {len(candidates)} candidates...")

        # Build context string
        context_str = ""
        if match_context:
            context_str = f"""
MATCH CONTINUITY CHECK:
- Previous section showed: {match_context.teams[0]} vs {match_context.teams[1]}
- Date: {match_context.date}
- Home jersey: {match_context.home_jersey_color}
- Away jersey: {match_context.away_jersey_color}
- Stadium: {match_context.stadium}

CRITICAL: Verify these 3 images are from THE SAME MATCH as above. Check:
1. Jersey colors match
2. Stadium/background similar
3. Same teams playing
"""

        # Add jersey color validation if we have colors from video
        if jersey_colors and jersey_colors.get('home') != 'unknown':
            context_str += f"""

JERSEY COLOR VALIDATION (from actual match video):
- Home team jersey: {jersey_colors['home']}
- Away team jersey: {jersey_colors['away']}

CRITICAL: The images MUST show these exact jersey colors.
If jersey colors don't match, score = 0 (wrong match detected).
"""

        prompt = f"""You are an expert at selecting images that tell a visual story.

⚠️ CRITICAL COUNTING INSTRUCTION ⚠️
You are receiving EXACTLY {len(candidates)} images attached to this message.
COUNT THEM CAREFULLY before selecting.

Image numbering works like this:
- First image you see = Image 1
- Second image you see = Image 2
- Last image you see = Image {len(candidates)}

VALID IMAGE NUMBERS: 1, 2, 3, ..., {len(candidates)} (and NOTHING ELSE)
INVALID IMAGE NUMBERS: Any number > {len(candidates)} or < 1

BEFORE YOU RESPOND:
1. Count how many images you actually received
2. Verify your selected numbers are ALL ≤ {len(candidates)}
3. If you're about to select image number 12 but only saw 11 images, STOP and recount

EXAMPLE FOR YOUR CURRENT TASK:
- You have {len(candidates)} images
- Maximum valid image number is {len(candidates)}
- You CANNOT select {len(candidates) + 1}, {len(candidates) + 2}, etc.
- Those images DO NOT EXIST

TARGET ACTION:
- Description: {section.description}
- Context: {section.transcript}

{context_str}

YOUR TASK:
Select 3 images that show the PROGRESSION of this action:
1. **Beginning/Setup** - The moment BEFORE the main action (e.g., player receiving ball, preparing to shoot)
2. **Middle/Climax** - The main action itself (e.g., player shooting, making contact)
3. **End/Aftermath** - The result/celebration (e.g., ball in net, celebration, crowd reaction)

STRICT REQUIREMENTS:
1. ✅ All 3 images must show the SAME EVENT/MOMENT (not 3 different events)
2. ✅ NO stock watermarks (Alamy, Getty, Shutterstock, etc.) - REJECT immediately
3. ✅ Images should form a NARRATIVE SEQUENCE (temporal or spatial progression)
4. ✅ If match context provided above, verify jersey colors and stadium MATCH
5. ✅ Minimal text overlays (small scoreboards OK, but no huge graphics)
6. ✅ High image quality (sharp, clear, well-lit)

FALLBACK STRATEGY:
- If perfect progression isn't found, try different angles of the SAME moment
- If only 2 good images found, select those (return 2-image array)
- If match continuity fails, score = 0 (different match detected)

SCORING:
- Narrative quality: 0-100 (how well the 3 images tell the story)
- Match continuity: true/false (do jersey colors/stadium match context?)
- Confidence: 0-100 (overall confidence in selection)

You should receive up to {len(candidates)} candidate images attached below (some may fail to load).
Count the images you actually receive and ONLY select from those you can see.

Return ONLY valid JSON (no markdown):
{{
  "selected_images": [3, 7, 9],
  "confidence": 88,
  "narrative_quality": 92,
  "match_continuity_verified": true,
  "jersey_colors": {{"home": "blue and red", "away": "white"}},
  "stadium": "Camp Nou",
  "competition": "La Liga",
  "story_breakdown": {{
    "beginning": "Image 3: Player receiving pass",
    "middle": "Image 7: Player shooting",
    "end": "Image 9: Ball in net"
  }},
  "reason": "Perfect narrative sequence showing goal progression"
}}

OR if only 2 images work:
{{
  "selected_images": [5, 9],
  "confidence": 75,
  "narrative_quality": 70,
  "reason": "Only found 2 suitable images showing action and result"
}}
"""

        try:
            # Send images to Gemini
            image_paths = [c.path for c in candidates]
            # ✅ ADD THIS: Verify all paths exist before sending
            valid_paths = []
            for i, path in enumerate(image_paths, 1):
                if not os.path.exists(path):
                    print(f"   ⚠️ Warning: Image {i} path doesn't exist: {path}")
                    continue
                valid_paths.append(path)

            if len(valid_paths) == 0:
                raise Exception("No valid image paths to send to Gemini")

            if len(valid_paths) < len(image_paths):
                print(f"   ⚠️ Only {len(valid_paths)}/{len(image_paths)} images accessible")

            result_text = self._call_gemini_with_images(prompt, image_paths, timeout=90)

            # Clean and parse
            if result_text.startswith("```"):
                result_text = result_text.split("```")[1]
                if result_text.startswith("json"):
                    result_text = result_text[4:]
                result_text = result_text.strip()

            analysis = json.loads(result_text)

            print(f"\n   📊 Gemini's Selection:")
            print(f"      Selected: {analysis['selected_images']}")
            print(f"      Confidence: {analysis['confidence']}%")
            print(f"      Narrative Quality: {analysis.get('narrative_quality', 'N/A')}%")

            if 'story_breakdown' in analysis:
                print(f"\n   📖 Story Progression:")
                for phase, desc in analysis['story_breakdown'].items():
                    print(f"      {phase.capitalize()}: {desc}")

            print(f"\n   💡 Reason: {analysis['reason']}\n")

            return analysis

        except Exception as e:
            print(f"   ⚠️ Gemini analysis failed: {str(e)[:100]}")

            # ✅ IMPROVED FALLBACK
            if len(candidates) >= 3:
                sorted_indices = sorted(range(len(candidates)),
                                      key=lambda i: candidates[i].width * candidates[i].height,
                                      reverse=True)
                return {
                    'selected_images': [sorted_indices[0] + 1, sorted_indices[1] + 1, sorted_indices[2] + 1],
                    'confidence': 40,
                    'narrative_quality': 40,
                    'match_continuity_verified': False if match_context else None,
                    'reason': f'Fallback: highest resolution images (Error: {str(e)[:50]})'
                }
            else:
                # Not enough candidates for fallback
                return {
                    'selected_images': [],
                    'confidence': 0,
                    'narrative_quality': 0,
                    'reason': f'Failed: {str(e)[:100]}'
                }

    def enhance_search_query_with_context(self, section: PNGSequenceSection,
                                         match_context: Optional[MatchContext] = None) -> str:
        """Generate search query with match context awareness"""

        context_str = ""
        if match_context:
            teams_str = " vs ".join(match_context.teams)
            context_str = f"""
Previous sections established:
- Match: {teams_str}
- Date: {match_context.date}
- Competition: {match_context.competition}

This section is likely from the SAME match.
"""

        prompt = f"""You are a Google Images search expert for football/soccer content.

{context_str}

TARGET REQUEST:
- Description: {section.description}
- Transcript: {section.transcript}

Generate the BEST search query for Google Images to find this action.

SEARCH STRATEGY:
- Include player name + action + team names
- If date is known, include year
- Add specific competition if known (e.g., "Champions League", "El Clasico")
- For better accuracy, try: "[player] [action] vs [opponent] [year]"
- Keep under 10 words

PRIORITIZE ACCURACY:
- Add "site:twitter.com OR site:instagram.com" for official sources
- Can add "getty OR afp" for professional photos (we'll filter watermarks later)

Return ONLY the search query, no explanation.
Example: "Yamal goal Real Madrid El Clasico 2024"
"""

        try:
            search_query = self._call_gemini(prompt, timeout=20)
            search_query = search_query.strip().strip('"\'').strip('.')
            print(f"   ✅ Enhanced query: '{search_query}'\n")
            return search_query
        except Exception as e:
            print(f"   ❌ Gemini failed: {e}")
            fallback = section.description[:100]
            print(f"   🔄 Using fallback: '{fallback}'\n")
            return fallback

    def suggest_meme_query(self, section: PNGSequenceSection) -> str:
        """Ask Gemini for a meme search query that captures the action's essence"""
        print(f"🎭 Asking Gemini for meme alternative...")

        prompt = f"""You are a meme expert helping find reaction images/memes that capture the ESSENCE of a football moment.

    TARGET ACTION:
    - Description: {section.description}
    - Transcript: {section.transcript}

    YOUR TASK:
    Create a search query for a MEME or REACTION IMAGE that would humorously/accurately represent this moment.

    EXAMPLES:
    - "pushing someone from behind" → "push guy violent meme"
    - "shot immediately blocked" → "shot blocked meme"
    - "player misses header" → "miss fail meme" or "[player name] meme"
    - "player nutmegs opponent" → "amazed laugh meme" or "skill shocked meme"
    - "player scores goal" → "celebration meme"
    - "referee gives penalty" → "angry referee meme"
    - "goalkeeper saves shot" → "superhero save meme"

    GUIDELINES:
    - Keep it 2-5 words
    - Use generic terms (not always player names, unless they're famous for memes)
    - Think about the EMOTION/REACTION the action would create
    - Can include: "meme", "reaction", "funny", or specific meme templates
    - Make it likely to find popular meme formats on Google Images

    Return ONLY the search query, no explanation.
    """

        try:
            meme_query = self._call_gemini(prompt, timeout=20)
            meme_query = meme_query.strip().strip('"\'').strip('.')

            # Ensure "meme" is in the query
            if 'meme' not in meme_query.lower():
                meme_query = f"{meme_query} meme"

            print(f"   🎭 Meme query: '{meme_query}'\n")
            return meme_query

        except Exception as e:
            print(f"   ❌ Gemini failed: {e}")
            # Fallback: generic meme
            fallback = "funny reaction meme"
            print(f"   🔄 Using fallback: '{fallback}'\n")
            return fallback

    def generate_search_from_transcript(self, section: PNGSequenceSection, all_sections: List[PNGSequenceSection] = None) -> str:
        """Use Gemini to generate a search query from transcript when description is useless"""
        print(f"🤖 Asking Gemini to interpret transcript...")

        # Get context from surrounding lines (±2 lines)
        context_transcript = section.transcript

        if all_sections:
            # Find current section index
            current_idx = None
            for i, s in enumerate(all_sections):
                if s.line_number == section.line_number:
                    current_idx = i
                    break

            if current_idx is not None:
                # Get 2 lines before
                context_parts = []
                for i in range(max(0, current_idx - 2), current_idx):
                    context_parts.append(all_sections[i].transcript)

                # Add current line
                context_parts.append(section.transcript)

                # Get 2 lines after
                for i in range(current_idx + 1, min(len(all_sections), current_idx + 3)):
                    context_parts.append(all_sections[i].transcript)

                context_transcript = " ".join(context_parts)
                print(f"   📝 Using context from {len(context_parts)} lines")

        prompt = f"""You are helping find video footage for a football/soccer commentary script.

    TARGET LINE TRANSCRIPT: "{section.transcript}"

    SURROUNDING CONTEXT (±2 lines): "{context_transcript}"

    The description is useless: "{section.description}"

    Based on the transcript AND surrounding context, generate a SHORT search query (3-8 words) to find relevant video footage.

    GUIDELINES:
    - If context mentions specific players/teams, include them
    - If context mentions a specific action (goal, shot, save, pass), include that
    - If transcript is too vague, use context to infer what happened
    - Keep it SHORT and searchable

    EXAMPLES:
    Context: "Arsenal had chances... 5, 4, 3, 2, 1, GOAL! ...punched in the face with a last-kick winner from Buendia"
    → "Buendia last minute goal Arsenal"

    Context: "What a save by the keeper!"
    → "goalkeeper save"

    Context: "He's through on goal!"
    → "striker through on goal"

    Return ONLY the search query, no explanation.
    """

        try:
            search_query = self._call_gemini(prompt, timeout=20)
            search_query = search_query.strip().strip('"\'').strip('.')
            print(f"   💡 Generated query: '{search_query}'\n")
            return search_query
        except Exception as e:
            print(f"   ❌ Gemini failed: {str(e)[:100]}")
            # Fallback: use transcript directly
            fallback = section.transcript[:50]
            print(f"   🔄 Using transcript as fallback: '{fallback}'\n")
            return fallback






# ========================================
# 4. MAIN TEST
# ========================================
def discover_all_videos(sections: List[PNGSequenceSection],
                       match_tracker: MatchContextTracker,
                       geo_block_manager: GeoBlockManager,
                       api_key: str,
                       video_plan_name: str) -> List[VideoDiscoveryResult]:
    """
    Phase 1: Discover all videos without downloading

    Returns list of VideoDiscoveryResult objects
    """
    print("="*70)
    print("🔍 PHASE 1: VIDEO DISCOVERY (no downloads)")
    print("="*70)
    print(f"Scanning {len(sections)} sections...\n")

    youtube_downloader = OfficialChannelVideoDownloader(output_dir="temp_videos")

    # Initialize Gemini validator (api_key passed as parameter)
    RAW_SCRIPTS_DIR = '/content/drive/MyDrive/Angry Dude Scripts/raw_scripts'

    gemini_validator = GeminiValidator(
        api_key=api_key,  # Use the parameter, not GEMINI_API_KEY
        raw_scripts_dir=RAW_SCRIPTS_DIR
    )
    print("✅ Gemini validator initialized\n")

    # Initialize progress tracker
    script_name = Path(video_plan_name).stem.replace('_video_plan', '')
    progress = VideoDiscoveryProgress(script_name)
    progress.data['total_sections'] = len(sections)
    progress._save()

    stats = progress.get_stats()
    print(f"📊 Progress: {stats['completed']}/{stats['total']} completed, {stats['failed']} failed, {stats['remaining']} remaining\n")

    discoveries = []

    for i, section in enumerate(sections, 1):
        print(f"[{i}/{len(sections)}] Line {section.line_number}: {section.description[:60]}...")

        # Check if already completed
        if progress.is_completed(section.line_number):
            saved_result = progress.get_result(section.line_number)
            print(f"   ✅ Already discovered - using saved result")
            print(f"   📺 {saved_result['title'][:60]}")
            print(f"   🔗 {saved_result['video_url']}\n")

            discoveries.append(VideoDiscoveryResult(
                section=section,
                video_url=saved_result['video_url'],
                title=saved_result['title'],
                channel_handle=saved_result.get('channel', ''),
                duration=saved_result.get('duration', 0),  # Add default
                upload_date=saved_result.get('upload_date', ''),  # Add default
                search_method=saved_result.get('search_method', 'cached'),
                is_player_highlight=saved_result.get('is_player_highlight', False)  # Add this too
            ))
            continue

        # Extract match info
        teams = match_tracker._extract_teams(section.description)
        date = match_tracker._extract_date(section.description)
        score = match_tracker._extract_score(section.description)

        # TIERED VALIDATION SYSTEM
        print(f"\n   🔒 Validating match info...")

        # Tier 1: Quick free checks
        confidence = gemini_validator.tier1_validate(
            teams, date, youtube_downloader.TEAM_CHANNELS
        )

        # Tier 2: Light API validation if low confidence
        if confidence == "low":
            raw_script_path = gemini_validator.get_raw_script_path(
                video_plan_name
            )
            corrected_info = gemini_validator.tier2_validate(
                section.description, raw_script_path, match_tracker
            )

            # Use corrected info
            teams = corrected_info['teams']
            date = corrected_info['date']
            score = corrected_info.get('score')

            print(f"   ✅ Using corrected info: {teams[0]} vs {teams[1]}, {date}, {score}")

        if score:
            print(f"   Teams: {teams[0]} vs {teams[1]}, Date: {date}, Score: {score}")
        else:
            print(f"   Teams: {teams[0]} vs {teams[1]}, Date: {date}")

        video_result = None
        search_method = "failed"
        fallback_reason = None
        is_player_highlight = False

# Path 1: Team vs Team match
        if len(teams) >= 2 and date != "unknown":
            print(f"   Teams: {teams[0]} vs {teams[1]}, Date: {date}")

            # USE GEMINI TIER 3 ONLY
            print(f"   🚀 Using Gemini Tier 3...")
            raw_script_path = gemini_validator.get_raw_script_path(video_plan_name)
            video_result = gemini_validator.tier3_validate(
                section.description,
                raw_script_path,
                {'teams': list(teams), 'date': date, 'score': score}
            )

            if video_result:
                search_method = "gemini_tier3"
            else:
                fallback_reason = "Gemini could not find video"

        # Path 2: Player highlight (no opponent/date)
        else:
            print(f"   No match info, trying player highlight search...")
            is_player_highlight = True

            # Check if description is useless (contains "unknown")
            if "unknown" in section.description.lower():
                print(f"   ⚠️ Description contains 'unknown', using Gemini to interpret transcript...")
                # Initialize analyzer temporarily just for this
                temp_analyzer = GeminiImageAnalyzer(api_key)
                player_query = temp_analyzer.generate_search_from_transcript(section, sections)
            else:
                # Use description as search query
                player_query = section.description[:100]

            video_result = youtube_downloader.search_player_highlights(
                player_query, geo_block_manager
            )

            if video_result:
                search_method = "player_highlight"
            else:
                fallback_reason = "No player highlights found"

        # Create discovery result
        if video_result:
            discovery = VideoDiscoveryResult(
                section=section,
                video_url=video_result['url'],
                channel_handle=video_result.get('channel'),
                title=video_result['title'],
                duration=video_result['duration'],
                upload_date=video_result['upload_date'],
                search_method=search_method,
                is_player_highlight=is_player_highlight
            )
            # Save to progress
            progress.mark_completed(section.line_number, {
                'url': video_result['url'],
                'title': video_result['title'],
                'channel': video_result.get('channel', ''),
                'duration': video_result.get('duration', 0),
                'upload_date': video_result.get('upload_date', ''),
                'search_method': search_method,
                'is_player_highlight': is_player_highlight
            })
        else:
            discovery = VideoDiscoveryResult(
                section=section,
                video_url=None,
                channel_handle=None,
                title="",
                duration=0,
                upload_date="",
                search_method=search_method,
                fallback_reason=fallback_reason,
                is_player_highlight=is_player_highlight
            )
            # Save failure to progress
            progress.mark_failed(section.line_number, fallback_reason or "Unknown failure")

        discoveries.append(discovery)
        print()
    # Summary
    found = [d for d in discoveries if d.video_url]
    failed = [d for d in discoveries if not d.video_url]

    # Sort found videos by duration (prefer 3-10 min)
    def duration_score(d):
        dur_min = d.duration / 60
        if 3 <= dur_min <= 10:
            return 100  # Perfect range
        elif dur_min < 3:
            return 80  # Acceptable
        else:  # 10-15 min
            return 60  # Use only if no better options

    found.sort(key=duration_score, reverse=True)

    print("="*70)
    print("📊 DISCOVERY SUMMARY")
    print("="*70)
    print(f"✅ Found: {len(found)}/{len(sections)} videos")
    print(f"❌ Not found: {len(failed)}/{len(sections)} sections")
    print(f"\n📥 DOWNLOAD QUEUE ({len(found)} videos):")

    total_duration = sum(d.duration for d in found)
    for i, d in enumerate(found, 1):
        dur_min = d.duration / 60
        print(f"   {i}. Line {d.section.line_number}: {d.title[:50]} ({dur_min:.1f}min) - {d.channel_handle}")

    print(f"\n⏱️  Total video duration: {total_duration/60:.1f} minutes")
    print(f"⏱️  Estimated download time: {len(found) * 15:.0f}s (rough estimate)\n")

    if failed:
        print(f"⚠️  Sections falling back to Google Images:")
        for d in failed:
            print(f"   Line {d.section.line_number}: {d.fallback_reason}")
        print()
    # Save final stats
    final_stats = progress.get_stats()
    print(f"💾 Progress saved: {final_stats['completed']} completed, {final_stats['failed']} failed")
    print(f"📁 Progress file: {progress.progress_file}\n")

    return discoveries

def process_png_sequences():
    """Process PNG sequences as 3-image sequences from Google Images"""

    print("="*70)
    print("🖼️ 3-IMAGE SEQUENCE COLLECTION (Google Images)")
    print("="*70)
    print()

    # Step 1: Select video plan
    video_plans_dir = "/content/drive/MyDrive/Angry Dude Scripts/video_plans"
    print(f"📂 Scanning for video plans in: {video_plans_dir}\n")

    if not os.path.exists(video_plans_dir):
        print(f"❌ Directory not found: {video_plans_dir}")
        raise SystemExit()

    video_plans = [f for f in os.listdir(video_plans_dir) if f.endswith('.txt')]

    if not video_plans:
        print("❌ No video plan files (.txt) found!")
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

    # Step 2: Parse plan
    parser = PNGSequencePlanParser()
    script_name, sections = parser.parse_plan(plan_file)

    if not sections:
        print("⚠️ No PNG sequence sections found!")
        raise SystemExit()

    # Step 3: Initialize components
    output_base_dir = "/content/drive/MyDrive/Angry Dude Scripts/collected_assets"
    tracker = ProgressTracker(script_name, output_base_dir)
    tracker.data['stats']['total_sections'] = len(sections)

    api_key = "" #input api key here or setup with .env
    match_tracker = MatchContextTracker()
    geo_block_manager = GeoBlockManager()

    # Initialize processing cache
    processing_cache = VideoProcessingCache(script_name)
    cache_stats = processing_cache.get_cache_size()
    print(f"📊 Cache size: {cache_stats['total_size_mb']:.1f} MB ({cache_stats['video_count']} videos, {cache_stats['analysis_count']} analyses)\n")

    # Store jersey colors per match for validation
    match_jersey_colors = {}  # Key: "team1 vs team2 date" -> {"home": "color", "away": "color"}

    # =========================================================================
    # PHASE 1: DISCOVER ALL VIDEOS (no downloads, no cookies)
    # =========================================================================
    # Define API key here (outside the function)
    GEMINI_API_KEY = '' #input api key here or setup with .env

    discoveries = discover_all_videos(
        sections=sections,
        match_tracker=match_tracker,
        geo_block_manager=geo_block_manager,
        api_key=GEMINI_API_KEY,
        video_plan_name=Path(plan_file).name
    )

    # Separate into video-ready and fallback sections
    video_sections = [d for d in discoveries if d.video_url]
    fallback_sections = [d for d in discoveries if not d.video_url]

    # =========================================================================
    # PHASE 2: BATCH VIDEO DOWNLOAD (with caching)
    # =========================================================================
    downloaded_videos = {}
    if video_sections:
        print("="*70)
        print("📥 PHASE 2: VIDEO DOWNLOAD (with cache)")
        print("="*70)
        print()

        # Check cache first
        print("🔍 Checking cache for existing videos...\n")
        for discovery in video_sections:
            cached_path = processing_cache.get_video_path(discovery.video_url)
            if cached_path:
                downloaded_videos[discovery.video_url] = cached_path

        # Download missing videos
        videos_to_download = [d for d in video_sections if d.video_url not in downloaded_videos]

        if videos_to_download:
            print(f"📥 Downloading {len(videos_to_download)} videos (rest cached)...\n")

            youtube_downloader = OfficialChannelVideoDownloader(
                output_dir="temp_videos",  # Download to local first (faster)
                cookies_path=youtube_cookies_path if 'youtube_cookies_path' in dir() else None
            )

            video_list = [{'url': d.video_url, 'channel': d.channel_handle} for d in videos_to_download]
            newly_downloaded = youtube_downloader.batch_download_videos(video_list, geo_block_manager)

            # Cache newly downloaded videos
            print("\n💾 Caching downloaded videos to Drive...")
            for url, temp_path in newly_downloaded.items():
                if os.path.exists(temp_path):
                    cached_path = processing_cache.cache_video(url, temp_path)
                    downloaded_videos[url] = cached_path
                    # Delete local temp file to save space
                    os.remove(temp_path)
                else:
                    downloaded_videos[url] = temp_path
        else:
            print("✅ All videos already cached!\n")

        print(f"📊 Ready to process: {len(downloaded_videos)}/{len(video_sections)} videos\n")

    # =========================================================================
    # PHASE 3: PROCESS DOWNLOADED VIDEOS
    # =========================================================================
    if downloaded_videos:
        print("="*70)
        print("🎬 PHASE 3: VIDEO PROCESSING")
        print("="*70)
        print()

        video_analyzer = GeminiVideoAnalyzer(api_key)

        for i, discovery in enumerate(video_sections, 1):
            if discovery.video_url not in downloaded_videos:
                print(f"[{i}/{len(video_sections)}] Skipping Line {discovery.section.line_number} (download failed)\n")
                fallback_sections.append(discovery)
                continue

            section = discovery.section
            section_key = f"line_{section.line_number:03d}"

            # Check if already completed FIRST!
            if tracker.is_section_complete(section_key):
                print(f"[{i}/{len(video_sections)}] Line {section.line_number}: ✅ Already completed - skipping\n")
                continue

            print(f"[{i}/{len(video_sections)}] Processing Line {section.line_number}...")
            print(f"   Video: {discovery.title[:60]}")

            # Get video path and ensure it's a string
            video_path = downloaded_videos.get(discovery.video_url)
            if not video_path:
                print(f"   ❌ Video path not found in downloaded_videos dict\n")
                fallback_sections.append(discovery)
                continue

            video_path = str(video_path)

            # Verify file exists
            if not os.path.exists(video_path):
                print(f"   ❌ Video file doesn't exist: {video_path}\n")
                fallback_sections.append(discovery)
                continue

            print(f"   📁 Video path: {video_path}")

            try:
                # Check if analysis is cached
                cached_analysis = processing_cache.get_analysis(discovery.video_url, section.line_number)

                if cached_analysis:
                    # Load from cache
                    keyframes, event_timeline, exact_moment = processing_cache.load_cached_analysis(cached_analysis)
                    print(f"   ⏩ Skipped analysis (using cache)")

                else:
                    # Perform full analysis
                    print(f"   🎬 Analyzing video (not cached)...")

                    # Extract keyframes
                    keyframes = video_analyzer.extract_keyframes(video_path, fps=1.0)

                    # Get video FPS
                    vr = VideoReader(video_path, ctx=cpu(0))
                    video_fps = vr.get_avg_fps()

                    # Extract jersey colors for this match (if not already done)
                    teams = match_tracker._extract_teams(section.description)
                    date = match_tracker._extract_date(section.description)
                    match_key = f"{teams[0]} vs {teams[1]} {date}".lower() if len(teams) >= 2 else None

                    if match_key and match_key not in match_jersey_colors:
                        jersey_colors = video_analyzer.extract_jersey_colors(video_path, keyframes[:10])
                        match_jersey_colors[match_key] = jersey_colors
                        print(f"   📝 Stored jersey colors for: {match_key}")
                    elif match_key:
                        jersey_colors = match_jersey_colors[match_key]
                        print(f"   ♻️  Using stored jersey colors for: {match_key}")

                    # Get ESPN match data for better goal matching
                    espn_scraper = ESPNMatchScraper()
                    espn_data = None
                    espn_context = ""

                    if len(teams) >= 2 and date != "unknown":
                        espn_data = espn_scraper.get_match_data(teams[0], teams[1], date)
                        if espn_data:
                            espn_context = espn_scraper.build_goal_context(espn_data, section.description)

                    # Analyze video
                    event_timeline_result = video_analyzer.analyze_video_chunks(keyframes, video_fps, chunk_size=30)
                    # ADD DELAY to avoid rate limiting
                    print(f"   ⏸️  Waiting 3s to avoid rate limits...")
                    time.sleep(3)

                    # Extract events list from result
                    if isinstance(event_timeline_result, dict):
                        event_timeline = event_timeline_result.get('events', [])
                    else:
                        event_timeline = event_timeline_result

                    # Find exact moment (with ESPN context)
                    exact_moment = video_analyzer.find_exact_moment(
                        video_path,
                        event_timeline,
                        section.description,
                        keyframes,  # Pass keyframes
                        espn_context
                    )

                    # Cache the analysis results
                    processing_cache.cache_analysis(
                        discovery.video_url,
                        section.line_number,
                        keyframes,
                        event_timeline,
                        exact_moment
                    )

                if exact_moment and exact_moment['confidence'] >= 70:
                    # Extract 3-frame sequence (with Gemini story selection)
                    video_frames = video_analyzer.extract_3frame_sequence(
                        video_path,
                        exact_moment['frame_number'],
                        section.description
                    )

                    if len(video_frames) == 3:
                        # Save frames
                        output_dir = os.path.join(output_base_dir, script_name, "png_sequences", f"sequence_{section.line_number:03d}")
                        frames_dir = os.path.join(output_dir, "frames")
                        os.makedirs(frames_dir, exist_ok=True)

                        saved_paths = []
                        for idx, frame_path in enumerate(video_frames, 1):
                            output_filename = f"frame_{idx:02d}.png"
                            output_path = os.path.join(frames_dir, output_filename)
                            shutil.copy(frame_path, output_path)
                            saved_paths.append(output_path)
                            print(f"      💾 Saved: {output_filename}")

                        # Save metadata
                        metadata = {
                            'line_number': section.line_number,
                            'description': section.description,
                            'transcript': section.transcript,
                            'plan_time': f"{section.start_time}-{section.end_time}",
                            'extraction_method': 'youtube_video',
                            'search_method': discovery.search_method,
                            'num_images': 3,
                            'image_paths': saved_paths,
                            'video_source': discovery.video_url,
                            'exact_frame': exact_moment['frame_number'],
                            'video_timestamp': exact_moment['timestamp'],
                            'confidence': exact_moment['confidence']
                        }

                        metadata_path = os.path.join(output_dir, "metadata.json")
                        with open(metadata_path, 'w') as f:
                            json.dump(metadata, f, indent=2)

                        tracker.mark_section_completed(section, metadata)
                        print(f"   ✅ Completed!\n")

                        # Cleanup temp frames
                        for frame_path in video_frames:
                            temp_parent = os.path.dirname(frame_path)
                            if os.path.exists(temp_parent):
                                shutil.rmtree(temp_parent, ignore_errors=True)
                    else:
                        print(f"   ⚠️ Frame extraction failed\n")
                        fallback_sections.append(discovery)
                else:
                    print(f"   ⚠️ Could not find exact moment\n")
                    fallback_sections.append(discovery)

                # Cleanup video
                #if os.path.exists(video_path):
                    #os.remove(video_path)

            except Exception as e:
                print(f"   ❌ Error: {str(e)[:100]}\n")
                fallback_sections.append(discovery)
                if os.path.exists(video_path):
                    os.remove(video_path)

    # <-- ADD THE CLEANUP HERE (after the for loop ends)
    # Cleanup ALL downloaded videos after processing
    print("\n🧹 Cleaning up downloaded videos...")
    for video_path in set(downloaded_videos.values()):
        if os.path.exists(str(video_path)):
            os.remove(str(video_path))
            print(f"   🗑️  Deleted: {os.path.basename(str(video_path))}")

    # =========================================================================
    # PHASE 4: GOOGLE IMAGES FALLBACK
    # =========================================================================
    if fallback_sections:
        print("="*70)
        print("🔄 PHASE 4: GOOGLE IMAGES FALLBACK")
        print(f"   Processing {len(fallback_sections)} sections")
        print("="*70)
        print()

        scraper = GoogleImagesScraper()
        analyzer = GeminiImageAnalyzer(api_key)

        # Process fallback sections with Google Images
        for fallback_idx, discovery in enumerate(fallback_sections, 1):
            section = discovery.section

            print(f"\n[{fallback_idx}/{len(fallback_sections)}] Google Images: Line {section.line_number}")
            print("="*70)
            print(f"SECTION {i}/{len(sections)} - Line {section.line_number}")
            print("="*70)
            print(f"📝 Description: {section.description}")
            print(f"💬 Transcript: {section.transcript}")
            print(f"⏱️  Time: {section.start_time}s - {section.end_time}s\n")

            section_key = f"line_{section.line_number:03d}"

            # Check if already completed
            if tracker.is_section_complete(section_key):
                print(f"✅ Already completed - skipping\n")
                continue

            try:
                tracker.mark_section_started(section)

                # Extract match info for context tracking
                teams = match_tracker._extract_teams(section.description)
                date = match_tracker._extract_date(section.description)

                # Get match context from previous sections
                match_context = match_tracker.get_context_for_section(section)
                # Get jersey colors if we have them from video
                match_key = f"{teams[0]} vs {teams[1]} {date}".lower() if len(teams) >= 2 and date != "unknown" else None
                stored_jersey_colors = match_jersey_colors.get(match_key) if match_key else None

                if stored_jersey_colors:
                    print(f"   👕 Using jersey colors from video: Home={stored_jersey_colors['home']}, Away={stored_jersey_colors['away']}")

                # ==================================================================
                # NEW PRIMARY PATH: VIDEO EXTRACTION
                # ==================================================================


                # ==================================================================
                # PRIMARY METHOD: OFFICIAL YOUTUBE CHANNELS
                # ==================================================================

                video_success = False

                if len(teams) >= 2 and date != "unknown":
                    print(f"🎬 PRIMARY METHOD: Official YouTube Channels")
                    print(f"   Match: {' vs '.join(teams)}")
                    print(f"   Date: {date}\n")

                    # Initialize YouTube downloader
                    youtube_downloader = OfficialChannelVideoDownloader(output_dir="temp_videos")

                    # Try to find and download from official channels
                    video_path = youtube_downloader.find_and_download_match_highlights(
                        team1=teams[0],
                        team2=teams[1],
                        match_date=date  # "December 6 2025" format is fine
                    )

                    if video_path:
                        try:
                            print(f"✅ Video downloaded from official channel!")

                            # Initialize video analyzer
                            video_analyzer = GeminiVideoAnalyzer(api_key)

                            # Extract keyframes (convert Path to string)
                            # Extract keyframes at higher rate for better precision
                            keyframes = video_analyzer.extract_keyframes(str(video_path), fps=1.0)

                            # Get video FPS
                            vr = VideoReader(video_path, ctx=cpu(0))
                            video_fps = vr.get_avg_fps()

                            # Analyze video to build event timeline
                            # Use smaller chunks for long videos to avoid rate limiting
                            video_duration_min = len(keyframes) * 2 / 60  # Approximate duration
                            chunk_size = 20 if video_duration_min > 5 else 30

                            event_timeline_result = video_analyzer.analyze_video_chunks(keyframes, video_fps, chunk_size=chunk_size)
                            # ADD DELAY to avoid rate limiting
                            print(f"   ⏸️  Waiting 3s to avoid rate limits...")
                            time.sleep(3)

                            # Extract events list from result
                            if isinstance(event_timeline_result, dict):
                                event_timeline = event_timeline_result.get('events', [])
                            else:
                                event_timeline = event_timeline_result

                            # Find exact moment (convert Path to string)
                            # Get ESPN context for better matching
                            espn_scraper = ESPNMatchScraper()
                            espn_data = None
                            espn_context = ""

                            if len(teams) >= 2 and date != "unknown":
                                espn_data = espn_scraper.get_match_data(teams[0], teams[1], date)
                                if espn_data:
                                    espn_context = espn_scraper.build_goal_context(espn_data, section.description)

                            # Find exact moment with ESPN context
                            exact_moment = video_analyzer.find_exact_moment(
                                str(video_path),
                                event_timeline,
                                section.description,
                                keyframes,  # Pass keyframes
                                espn_context
                            )

                            if exact_moment and exact_moment['confidence'] >= 70:
                                print(f"✅ Found exact moment in video!")
                                print(f"   Frame: {exact_moment['frame_number']}")
                                print(f"   Time: {exact_moment['timestamp']:.2f}s")
                                print(f"   Confidence: {exact_moment['confidence']}%\n")

                                # Extract 3-frame sequence
                                video_frames = video_analyzer.extract_3frame_sequence(video_path, exact_moment['frame_number'])

                                if len(video_frames) == 3:
                                    # Save frames to final location
                                    output_dir = os.path.join(output_base_dir, script_name, "png_sequences", f"sequence_{section.line_number:03d}")
                                    frames_dir = os.path.join(output_dir, "frames")
                                    os.makedirs(frames_dir, exist_ok=True)

                                    saved_paths = []
                                    for idx, frame_path in enumerate(video_frames, 1):
                                        output_filename = f"frame_{idx:02d}.png"
                                        output_path = os.path.join(frames_dir, output_filename)
                                        shutil.copy(frame_path, output_path)
                                        saved_paths.append(output_path)
                                        print(f"   💾 Saved: {output_filename}")

                                    # Save metadata
                                    metadata = {
                                        'line_number': section.line_number,
                                        'description': section.description,
                                        'transcript': section.transcript,
                                        'plan_time': f"{section.start_time}-{section.end_time}",
                                        'extraction_method': 'youtube_official_channel',
                                        'num_images': 3,
                                        'image_paths': saved_paths,
                                        'video_source': str(video_path),
                                        'exact_frame': exact_moment['frame_number'],
                                        'video_timestamp': exact_moment['timestamp'],
                                        'confidence': exact_moment['confidence'],
                                        'matched_event': exact_moment.get('matched_event', {})
                                    }

                                    metadata_path = os.path.join(output_dir, "metadata.json")
                                    with open(metadata_path, 'w') as f:
                                        json.dump(metadata, f, indent=2)

                                    tracker.mark_section_completed(section, metadata)
                                    print(f"✅ Section completed via YOUTUBE OFFICIAL CHANNEL!\n")

                                    video_success = True

                                    # Cleanup temp frames
                                    for frame_path in video_frames:
                                        temp_parent = os.path.dirname(frame_path)
                                        if os.path.exists(temp_parent):
                                            shutil.rmtree(temp_parent, ignore_errors=True)

                            else:
                                print(f"⚠️ Could not find exact moment in video (confidence too low)\n")

                            # Cleanup video
                            if os.path.exists(str(video_path)):
                                os.remove(str(video_path))
                                print(f"🗑️  Deleted video file\n")

                        except Exception as e:
                            print(f"❌ Video analysis failed: {str(e)[:100]}\n")
                            if os.path.exists(video_path):
                                os.remove(video_path)

                    else:
                        print(f"⚠️ Could not download from official channels\n")

                else:
                    print(f"⚠️ Insufficient match info for YouTube search")
                    print(f"   Teams: {teams}")
                    print(f"   Date: {date}\n")

                # ==================================================================
                # FALLBACK: GOOGLE IMAGES (Your existing code)
                # ==================================================================

                if not video_success:
                    print(f"🔄 FALLBACK METHOD: Google Images\n")

                    # YOUR EXISTING GOOGLE IMAGES CODE STARTS HERE
                    # (Keep everything from "if match_context:" onwards)

                    if match_context:
                        print(f"🔗 Using match context from previous sections:")
                        print(f"   Teams: {' vs '.join(match_context.teams)}")
                        print(f"   Date: {match_context.date}")
                        print(f"   Jerseys: {match_context.home_jersey_color} vs {match_context.away_jersey_color}\n")

                    # Step 1: Generate search query with context
                    search_query = analyzer.enhance_search_query_with_context(section, match_context)

                    max_retries = 2
                    attempt = 0
                    all_candidates = []

                    while attempt < max_retries:
                        attempt += 1

                        if attempt > 1:
                            search_query = analyzer.suggest_alternative_query(search_query, section)

                        # Step 2: Search Google Images
                        image_urls = scraper.search_and_get_full_res_urls(search_query, max_images=30)

                        if not image_urls:
                            print(f"❌ No images found")
                            if attempt < max_retries:
                                continue
                            else:
                                tracker.mark_section_failed(section, "No images found")
                                break

                        # Step 3: Download candidates
                        print(f"📥 Downloading candidates (attempt {attempt})...")
                        temp_dir = f"temp_png_sequences/{script_name}/sequence_{section.line_number:03d}"
                        os.makedirs(temp_dir, exist_ok=True)

                        for j, img_data in enumerate(image_urls, 1):
                            temp_path = f"{temp_dir}/candidate_{len(all_candidates) + j}.png"

                            success, metadata = scraper.download_image(img_data['url'], temp_path)

                            if success:
                                candidate = ImageCandidate(
                                    path=temp_path,
                                    url=metadata['url'],
                                    width=metadata['width'],
                                    height=metadata['height'],
                                    file_size=metadata['file_size']
                                )
                                all_candidates.append(candidate)
                                print(f"   ✅ Candidate {len(all_candidates)}: {metadata['width']}x{metadata['height']}px")

                            if len(all_candidates) >= 15:  # Get more candidates for 3-image selection
                                break

                        if len(all_candidates) < 5:
                            print(f"⚠️ Only {len(all_candidates)} suitable images")
                            if attempt < max_retries:
                                continue
                            else:
                                tracker.mark_section_failed(section, "Not enough quality images")
                                shutil.rmtree(temp_dir, ignore_errors=True)
                                break

                        # Step 4: Select 3-image sequence with Gemini (with recount retry)
                        max_gemini_retries = 3  # Give Gemini 3 chances to count correctly
                        gemini_attempt = 0
                        analysis = None
                        valid_indices = []

                        while gemini_attempt < max_gemini_retries:
                            gemini_attempt += 1

                            if gemini_attempt > 1:
                                print(f"🔄 Asking Gemini to recount (attempt {gemini_attempt}/{max_gemini_retries})...")

                            analysis = analyzer.select_3_image_sequence(
                                all_candidates,
                                section,
                                match_context,
                                jersey_colors=stored_jersey_colors
                            )

                            confidence = analysis.get('confidence', 0)
                            narrative_quality = analysis.get('narrative_quality', 0)
                            match_verified = analysis.get('match_continuity_verified')

                            # ✅ VALIDATE INDICES
                            selected_indices = analysis.get('selected_images', [])
                            max_index = len(all_candidates)

                            # Check for invalid indices
                            invalid_indices = [idx for idx in selected_indices if idx < 1 or idx > max_index]
                            valid_indices = [idx for idx in selected_indices if 1 <= idx <= max_index]

                            if len(invalid_indices) > 0:
                                print(f"⚠️ Gemini miscounted! Selected invalid indices: {invalid_indices}")
                                print(f"   Valid range: 1-{max_index}")
                                print(f"   Gemini selected: {selected_indices}")
                                print(f"   Valid ones: {valid_indices}")

                                if gemini_attempt < max_gemini_retries:
                                    print(f"   🔁 Asking Gemini to try again...\n")
                                    time.sleep(2)  # Brief pause before retry
                                    continue  # Go back to while loop, call Gemini again
                                else:
                                    print(f"   ⚠️ Gemini failed to count correctly after {max_gemini_retries} attempts")
                                    print(f"   📝 Proceeding with {len(valid_indices)} valid images\n")

                                    if len(valid_indices) == 0:
                                        print(f"❌ No valid images after all recount attempts")
                                        break  # Will be caught below

                                    # Use whatever valid indices we got
                                    analysis['selected_images'] = valid_indices
                                    confidence = max(50, confidence - (len(invalid_indices) * 15))
                                    narrative_quality = max(50, narrative_quality - (len(invalid_indices) * 15))
                                    break
                            else:
                                # ✅ All indices valid!
                                print(f"✅ Gemini counted correctly: {selected_indices}\n")
                                break  # Exit the recount loop

                        # Check if we got any valid images at all
                        if len(valid_indices) == 0:
                            print(f"❌ No valid images after {max_gemini_retries} Gemini attempts")
                            if attempt < max_retries:
                                print(f"🔄 Retrying with alternative query...\n")
                                all_candidates = []
                                continue
                            else:
                                tracker.mark_section_failed(section, "Gemini counting failed persistently")
                                shutil.rmtree(temp_dir, ignore_errors=True)
                                break

                        # Check if match continuity failed
                        if match_context and match_verified is False:
                            print(f"⚠️ Match continuity FAILED - different match detected")
                            print(f"   Clearing match context and retrying...")
                            match_tracker.clear_context()

                            if attempt < max_retries:
                                all_candidates = []
                                continue
                            else:
                                tracker.mark_section_failed(section, "Match continuity verification failed")
                                shutil.rmtree(temp_dir, ignore_errors=True)
                                break

                        # Step 5: Check quality thresholds
                        if confidence >= 75 and narrative_quality >= 70:

                            # ✅ POST-SELECTION VERIFICATION
                            print(f"🔍 Verifying selection matches request...")

                            verification_prompt = f"""You just selected images for this request:

        TARGET: {section.description}
        TRANSCRIPT: {section.transcript}

        YOUR SELECTION:
        - Images: {analysis['selected_images']}
        - Your story breakdown: {json.dumps(analysis.get('story_breakdown', {}), indent=2)}
        - Your reason: {analysis.get('reason', 'N/A')}

        CRITICAL VERIFICATION QUESTION:
        Does your selected story actually show the SPECIFIC action described in the target?

        Examples of MISMATCH:
        - Target says "handball incident" but your story shows "goal celebration" → MISMATCH
        - Target says "Yamal's shot" but your story shows "team celebrating" → MISMATCH
        - Target says "player receiving pass" but your story shows "generic match action" → MISMATCH

        Examples of MATCH:
        - Target says "handball incident" and your story shows "player's hand touching ball" → MATCH
        - Target says "goal celebration" and your story shows "players celebrating goal" → MATCH

        Be brutally honest. If you selected generic match images when asked for a specific incident, admit it.

        Return ONLY valid JSON:
        {{
          "actually_matches": true,
          "confidence_in_match": 95,
          "explanation": "Images clearly show the handball incident with player's hand on ball"
        }}

        OR if mismatch:
        {{
          "actually_matches": false,
          "confidence_in_match": 20,
          "explanation": "Images show goal celebration, not the handball incident requested"
        }}
        """

                            try:
                                verification_text = analyzer._call_gemini(verification_prompt, timeout=20)

                                # Clean JSON
                                if verification_text.startswith("```"):
                                    verification_text = verification_text.split("```")[1]
                                    if verification_text.startswith("json"):
                                        verification_text = verification_text[4:]
                                    verification_text = verification_text.strip()

                                verification = json.loads(verification_text)

                                actually_matches = verification.get('actually_matches', True)
                                match_confidence = verification.get('confidence_in_match', 100)
                                explanation = verification.get('explanation', 'N/A')

                                print(f"   📊 Verification Result:")
                                print(f"      Matches request: {actually_matches}")
                                print(f"      Match confidence: {match_confidence}%")
                                print(f"      Explanation: {explanation}\n")

                                if not actually_matches or match_confidence < 60:
                                    print(f"   ❌ Verification FAILED - images don't match request")
                                    print(f"   🔄 Treating as low quality and retrying...\n")

                                    confidence = 0
                                    narrative_quality = 0

                                    # This will trigger the retry logic below

                            except Exception as e:
                                print(f"   ⚠️ Verification failed to run: {str(e)[:100]}")
                                print(f"   ⚠️ Proceeding without verification\n")
                                # If verification fails, continue with original confidence

                            # Re-check quality after verification
                            if confidence < 75 or narrative_quality < 70:
                                print(f"⚠️ Quality too low after verification (Confidence: {confidence}%, Narrative: {narrative_quality}%)")

                                if attempt < max_retries:
                                    print(f"🔄 Retrying with alternative query...\n")
                                    all_candidates = []
                                    continue
                                else:
                                    # Fallback logic here...
                                    print(f"🔄 Falling back to single image selection...")
                                    # Your existing fallback code

                            # If we get here, verification passed
                            selected_indices = analysis['selected_images']
                            num_images = len(selected_indices)

                            print(f"✅ Selected {num_images} images with {confidence}% confidence (verified)\n")

                            # Save images to final location
                            output_dir = os.path.join(output_base_dir, script_name, "png_sequences", f"sequence_{section.line_number:02d}")
                            frames_dir = os.path.join(output_dir, "frames")
                            os.makedirs(frames_dir, exist_ok=True)

                            saved_paths = []
                            for idx, img_idx in enumerate(valid_indices, 1):
                                candidate = all_candidates[img_idx - 1]

                                output_filename = f"frame_{idx:02d}.png"  # ← NEW NAMING
                                output_path = os.path.join(frames_dir, output_filename)

                                shutil.copy(candidate.path, output_path)
                                saved_paths.append(output_path)

                                print(f"   💾 Saved: {output_filename}")

                            # Extract and save match context
                            new_match_context = match_tracker.extract_match_context(section, analysis)
                            match_tracker.update_context(section, new_match_context)

                            # Save metadata
                            metadata = {
                                'line_number': section.line_number,
                                'description': section.description,
                                'transcript': section.transcript,
                                'plan_time': f"{section.start_time}-{section.end_time}",
                                'num_images': num_images,
                                'image_paths': saved_paths,
                                'search_query': search_query,
                                'confidence': confidence,
                                'narrative_quality': narrative_quality,
                                'story_breakdown': analysis.get('story_breakdown', {}),
                                'match_context': {
                                    'teams': new_match_context.teams,
                                    'date': new_match_context.date,
                                    'jersey_colors': analysis.get('jersey_colors', {}),
                                    'stadium': analysis.get('stadium', 'unknown'),
                                    'competition': analysis.get('competition', 'unknown')
                                }
                            }

                            metadata_path = os.path.join(output_dir, "metadata.json")  # NOT frames_dir, output_dir
                            with open(metadata_path, 'w') as f:
                                json.dump(metadata, f, indent=2)

                            # Mark complete
                            tracker.mark_section_completed(section, metadata)
                            print(f"✅ Section completed!\n")

                            # Cleanup
                            shutil.rmtree(temp_dir, ignore_errors=True)
                            break

                        else:
                            print(f"⚠️ Quality too low (Confidence: {confidence}%, Narrative: {narrative_quality}%)")

                            if attempt < max_retries:
                                print(f"🔄 Retrying with alternative query...\n")
                                all_candidates = []
                                continue
                            else:
                                # FALLBACK: Single image with validation
                                print(f"🔄 Falling back to single image selection...")

                                # Build validation criteria
                                validation_context = f"TARGET: {section.description}\n"

                                if stored_jersey_colors and stored_jersey_colors.get('home') != 'unknown':
                                    validation_context += f"REQUIRED jersey colors: {stored_jersey_colors['home']} and {stored_jersey_colors['away']}\n"

                                if teams and len(teams) >= 2:
                                    validation_context += f"REQUIRED teams: {teams[0]} and {teams[1]}\n"

                                # Validate each candidate
                                print(f"🔍 Validating candidates with jersey/team matching...")
                                validated_candidates = []

                                for idx, candidate in enumerate(all_candidates[:10], 1):
                                    # Ask Gemini to validate this specific image
                                    validation_prompt = f"""{validation_context}

        I'm showing you ONE image. Does this image match the requirements above?

        Check:
        1. If jersey colors specified → Do the jerseys match?
        2. If teams specified → Does this look like those teams?
        3. Does this image show the general action described?

        Return ONLY valid JSON:
        {{
          "matches": true/false,
          "confidence": 0-100,
          "reason": "why it matches or doesn't match",
          "jersey_colors_match": true/false,
          "teams_match": true/false
        }}
        """

                                    try:
                                        result_text = analyzer._call_gemini_with_images(
                                            validation_prompt,
                                            [candidate.path],
                                            timeout=20
                                        )

                                        # Parse response
                                        if result_text.startswith("```"):
                                            result_text = result_text.split("```")[1]
                                            if result_text.startswith("json"):
                                                result_text = result_text[4:]
                                            result_text = result_text.strip()

                                        result_text = result_text.replace('```', '').strip()
                                        validation = json.loads(result_text)

                                        if validation.get('matches') and validation.get('confidence', 0) >= 70:
                                            validated_candidates.append({
                                                'candidate': candidate,
                                                'confidence': validation['confidence'],
                                                'reason': validation.get('reason', '')
                                            })
                                            print(f"   ✅ Image {idx}: {validation['confidence']}% - {validation.get('reason', '')[:50]}")
                                        else:
                                            print(f"   ❌ Image {idx}: Rejected - {validation.get('reason', '')[:50]}")

                                    except Exception as e:
                                        print(f"   ⚠️ Image {idx}: Validation failed - {str(e)[:50]}")
                                        continue

                                # Use best validated candidate
                                if validated_candidates:
                                    best = max(validated_candidates, key=lambda x: x['confidence'])
                                    best_candidate = best['candidate']

                                    output_dir = os.path.join(output_base_dir, script_name, "png_sequences", f"sequence_{section.line_number:03d}")
                                    frames_dir = os.path.join(output_dir, "frames")
                                    os.makedirs(frames_dir, exist_ok=True)

                                    # FIXED NAMING: frame_01.png, frame_02.png, frame_03.png (just copy same image 3 times)
                                    output_paths = []
                                    for i in range(1, 4):
                                        output_path = os.path.join(frames_dir, f"frame_{i:02d}.png")
                                        shutil.copy(best_candidate.path, output_path)
                                        output_paths.append(output_path)

                                    metadata = {
                                        'line_number': section.line_number,
                                        'description': section.description,
                                        'transcript': section.transcript,
                                        'plan_time': f"{section.start_time}-{section.end_time}",
                                        'extraction_method': 'google_images_single_validated',
                                        'num_images': 3,  # 3 copies of same image
                                        'image_paths': output_paths,
                                        'search_query': search_query,
                                        'confidence': best['confidence'],
                                        'fallback_mode': 'single_image_validated',
                                        'validation_reason': best['reason']
                                    }

                                    metadata_path = os.path.join(output_dir, "metadata.json")
                                    with open(metadata_path, 'w') as f:
                                        json.dump(metadata, f, indent=2)

                                    tracker.mark_section_completed(section, metadata)
                                    print(f"✅ Saved validated single image (confidence: {best['confidence']}%)\n")
                                    # ===================================================================
                                    # NEW MEME FALLBACK - ONLY RUNS WHEN SINGLE IMAGE ALSO FAILS
                                    # ===================================================================
                                    print(f"⚠️ Single image confidence too low ({single_confidence}%)")
                                    print(f"🎭 Falling back to MEME search...\n")

                                    # Get meme search query from Gemini
                                    meme_query = analyzer.suggest_meme_query(section)

                                    # Search for meme
                                    meme_urls = scraper.search_and_get_full_res_urls(meme_query, max_images=20)

                                    if not meme_urls:
                                        print(f"❌ No memes found either - marking as failed")
                                        tracker.mark_section_failed(section, "No images or memes found")
                                        shutil.rmtree(temp_dir, ignore_errors=True)
                                        break

                                    # Download meme candidates
                                    print(f"📥 Downloading meme candidates...")
                                    meme_temp_dir = f"{temp_dir}/memes"
                                    os.makedirs(meme_temp_dir, exist_ok=True)

                                    meme_candidates = []
                                    for j, img_data in enumerate(meme_urls, 1):
                                        meme_path = f"{meme_temp_dir}/meme_{j}.png"

                                        success, metadata = scraper.download_image(img_data['url'], meme_path)

                                        if success:
                                            candidate = ImageCandidate(
                                                path=meme_path,
                                                url=metadata['url'],
                                                width=metadata['width'],
                                                height=metadata['height'],
                                                file_size=metadata['file_size']
                                            )
                                            meme_candidates.append(candidate)
                                            print(f"   ✅ Meme {len(meme_candidates)}: {metadata['width']}x{metadata['height']}px")

                                        if len(meme_candidates) >= 10:
                                            break

                                    if len(meme_candidates) == 0:
                                        print(f"❌ No suitable memes downloaded")
                                        tracker.mark_section_failed(section, "Meme download failed")
                                        shutil.rmtree(temp_dir, ignore_errors=True)
                                        break

                                    # Use relaxed validation for memes
                                    print(f"\n🤖 Selecting best meme (relaxed criteria)...")

                                    meme_prompt = f"""You are selecting a MEME/REACTION IMAGE for this moment:

                            TARGET: {section.description}
                            CONTEXT: {section.transcript}

                            This is a MEME FALLBACK - we couldn't find actual footage.
                            Select the meme that best captures the ESSENCE/EMOTION of this moment.

                            RELAXED CRITERIA:
                            - Does it fit the general vibe? (humor, drama, shock, fail, etc.)
                            - Is it a recognizable meme format?
                            - No stock watermarks
                            - Decent quality

                            Score each meme 0-100 for "narrative fit" (how well it represents the moment).

                            Return ONLY valid JSON:
                            {{
                              "evaluations": [
                                {{"image": 1, "score": 75, "analysis": "Good match for the action"}},
                                {{"image": 2, "score": 60, "analysis": "Acceptable but generic"}}
                              ],
                              "best_image": 1,
                              "confidence": 75,
                              "reason": "Best captures the emotion of the moment"
                            }}
                            """

                                    try:
                                        # Send meme images to Gemini
                                        meme_image_paths = [c.path for c in meme_candidates]
                                        result_text = analyzer._call_gemini_with_images(meme_prompt, meme_image_paths, timeout=60)

                                        # Clean and parse
                                        if result_text.startswith("```"):
                                            result_text = result_text.split("```")[1]
                                            if result_text.startswith("json"):
                                                result_text = result_text[4:]
                                            result_text = result_text.strip()

                                        meme_analysis = json.loads(result_text)

                                        meme_confidence = meme_analysis.get('confidence', 0)
                                        best_meme_idx = meme_analysis.get('best_image', 1) - 1

                                        print(f"\n   📊 Meme Selection:")
                                        print(f"      Best: Image {best_meme_idx + 1}")
                                        print(f"      Confidence: {meme_confidence}%")
                                        print(f"      Reason: {meme_analysis.get('reason', 'N/A')}\n")

                                        # Accept memes with lower threshold (50%+)
                                        if meme_confidence >= 50 and 0 <= best_meme_idx < len(meme_candidates):
                                            best_meme = meme_candidates[best_meme_idx]

                                            # Save meme to final location
                                            output_dir = os.path.join(output_base_dir, script_name, "png_sequences", f"sequence_{section.line_number:03d}")
                                            frames_dir = os.path.join(output_dir, "frames")
                                            os.makedirs(frames_dir, exist_ok=True)

                                            output_path = os.path.join(frames_dir, "frame_01.png")
                                            shutil.copy(best_meme.path, output_path)

                                            print(f"   💾 Saved meme as frame_01.png")

                                            # Save metadata
                                            metadata = {
                                                'line_number': section.line_number,
                                                'description': section.description,
                                                'transcript': section.transcript,
                                                'plan_time': f"{section.start_time}-{section.end_time}",
                                                'num_images': 1,
                                                'image_paths': [output_path],
                                                'search_query': meme_query,
                                                'confidence': meme_confidence,
                                                'fallback_mode': 'meme',
                                                'original_query': search_query,
                                                'reason': f"Meme fallback: {meme_analysis.get('reason', 'Best available')}"
                                            }

                                            metadata_path = os.path.join(output_dir, "metadata.json")
                                            with open(metadata_path, 'w') as f:
                                                json.dump(metadata, f, indent=2)

                                            tracker.mark_section_completed(section, metadata)
                                            print(f"✅ Section completed with MEME fallback!\n")

                                            # Cleanup
                                            shutil.rmtree(temp_dir, ignore_errors=True)
                                            break
                                        else:
                                            print(f"❌ Meme confidence too low ({meme_confidence}%) or invalid index")
                                            # Last resort: use highest resolution meme anyway
                                            if len(meme_candidates) > 0:
                                                best_meme = max(meme_candidates, key=lambda c: c.width * c.height)

                                                output_dir = os.path.join(output_base_dir, script_name, "png_sequences", f"sequence_{section.line_number:03d}")
                                                frames_dir = os.path.join(output_dir, "frames")
                                                os.makedirs(frames_dir, exist_ok=True)

                                                output_path = os.path.join(frames_dir, "frame_01.png")
                                                shutil.copy(best_meme.path, output_path)

                                                metadata = {
                                                    'line_number': section.line_number,
                                                    'description': section.description,
                                                    'transcript': section.transcript,
                                                    'plan_time': f"{section.start_time}-{section.end_time}",
                                                    'num_images': 1,
                                                    'image_paths': [output_path],
                                                    'search_query': meme_query,
                                                    'confidence': 40,
                                                    'fallback_mode': 'meme_emergency',
                                                    'reason': 'Emergency fallback: highest resolution meme'
                                                }

                                                metadata_path = os.path.join(output_dir, "metadata.json")
                                                with open(metadata_path, 'w') as f:
                                                    json.dump(metadata, f, indent=2)

                                                tracker.mark_section_completed(section, metadata)
                                                print(f"✅ Section completed with emergency meme fallback!\n")
                                            else:
                                                tracker.mark_section_failed(section, "All fallbacks exhausted")

                                            shutil.rmtree(temp_dir, ignore_errors=True)
                                            break

                                    except Exception as e:
                                        print(f"❌ Meme selection failed: {e}")
                                        # Emergency: use highest resolution meme
                                        if len(meme_candidates) > 0:
                                            best_meme = max(meme_candidates, key=lambda c: c.width * c.height)

                                            output_dir = os.path.join(output_base_dir, script_name, "png_sequences", f"sequence_{section.line_number:03d}")
                                            frames_dir = os.path.join(output_dir, "frames")
                                            os.makedirs(frames_dir, exist_ok=True)

                                            output_path = os.path.join(frames_dir, "frame_01.png")
                                            shutil.copy(best_meme.path, output_path)

                                            metadata = {
                                                'line_number': section.line_number,
                                                'description': section.description,
                                                'transcript': section.transcript,
                                                'plan_time': f"{section.start_time}-{section.end_time}",
                                                'num_images': 1,
                                                'image_paths': [output_path],
                                                'search_query': meme_query,
                                                'confidence': 40,
                                                'fallback_mode': 'meme_emergency',
                                                'reason': 'Emergency fallback after exception'
                                            }

                                            metadata_path = os.path.join(output_dir, "metadata.json")
                                            with open(metadata_path, 'w') as f:
                                                json.dump(metadata, f, indent=2)

                                            tracker.mark_section_completed(section, metadata)
                                            print(f"✅ Section completed with emergency meme fallback!\n")
                                        else:
                                            tracker.mark_section_failed(section, "All fallbacks exhausted")

                                        shutil.rmtree(temp_dir, ignore_errors=True)
                                        break

                                shutil.rmtree(temp_dir, ignore_errors=True)
                                break

            except Exception as e:
                print(f"❌ Error: {e}")
                tracker.mark_section_failed(section, str(e))
                continue

    # Cleanup scraper after Google Images fallback
        scraper.close()

    # Cleanup cache (optional - ask user)
    print("="*70)
    print("🧹 CACHE CLEANUP")
    print("="*70)

    cache_stats = processing_cache.get_cache_size()
    print(f"📊 Current cache: {cache_stats['total_size_mb']:.1f} MB")
    print(f"   Videos: {cache_stats['video_count']}")
    print(f"   Analyses: {cache_stats['analysis_count']}")
    print()

    cleanup_choice = input("Delete cache? (y/n - saves space but requires re-download next time): ").strip().lower()
    if cleanup_choice == 'y':
        processing_cache.cleanup()
        print("✅ Cache deleted\n")
    else:
        print("💾 Cache kept for next run\n")

    # Final summary
    print("="*70)
    print("🎉 PROCESSING COMPLETE!")
    print("="*70)
    print(f"✅ Completed: {tracker.data['stats']['completed']}/{len(sections)}")
    print(f"❌ Failed: {tracker.data['stats']['failed']}/{len(sections)}")
    print(f"📁 Output: {output_base_dir}/{script_name}/png_sequences/\n")

# ========================================
# MAIN EXECUTION
# ========================================
if __name__ == "__main__":
    # Ask for cookies once at the start
    youtube_cookies_path = upload_youtube_cookies()

    # Then run the main processing
    process_png_sequences()