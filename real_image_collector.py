#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#
#      REAL IMAGES SECTION
#
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++





# Install required packages
!pip install -U google-generativeai pillow requests
!pip uninstall -y playwright
!pip install playwright==1.40.0

# Cell 2: Install browser binaries with dependencies
!playwright install chromium
!playwright install-deps chromium

# Cell 3: Verify installation
!playwright install --help
!playwright install chromium

from google.colab import files
import google.generativeai as genai
import requests
import json
import os
import re
import time
import shutil
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
from dataclasses import dataclass
from PIL import Image
import io
from google.colab import drive
import threading
from IPython.display import display, Javascript
from playwright.async_api import async_playwright
import asyncio
import nest_asyncio
import base64

# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()  # ← ADD THIS

# Mount Google Drive
print("📂 Mounting Google Drive...")
drive.mount('/content/drive')
print("✅ Drive mounted!\n")

# ========================================
# DATA CLASSES
# ========================================
@dataclass
class RealImageSection:
    """Represents a real image section from the plan"""
    start_time: float
    end_time: float
    description: str
    transcript: str
    script_name: str

@dataclass
class ImageCandidate:
    """Candidate image with metadata"""
    path: str
    url: str
    width: int
    height: int
    file_size: int



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

    def mark_section_started(self, section: RealImageSection):
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

    def mark_section_completed(self, section: RealImageSection, metadata: Dict):
        """Mark section as completed"""
        section_key = f"{section.start_time:.2f}-{section.end_time:.2f}"
        if section_key in self.data['sections']:
            self.data['sections'][section_key]['status'] = 'completed'
            self.data['sections'][section_key]['completed_at'] = datetime.now().isoformat()
            self.data['sections'][section_key]['metadata'] = metadata
            self.data['stats']['completed'] += 1
        self._save()

    def mark_section_failed(self, section: RealImageSection, error: str):
        """Mark section as failed"""
        section_key = f"{section.start_time:.2f}-{section.end_time:.2f}"
        if section_key in self.data['sections']:
            self.data['sections'][section_key]['status'] = 'failed'
            self.data['sections'][section_key]['error'] = error
            self.data['sections'][section_key]['failed_at'] = datetime.now().isoformat()
            self.data['stats']['failed'] += 1
        self._save()

    def log_attempt(self, section: RealImageSection, attempt_data: Dict):
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
class RealImagePlanParser:
    """Parse video plan TXT files for real image sections"""

    def __init__(self):
        print("✅ RealImagePlanParser initialized\n")

    def parse_plan(self, file_path: str) -> tuple[str, List[RealImageSection]]:
        """
        Parse video plan file for real image sections

        Returns:
            (script_name, list of RealImageSection objects)
        """
        print(f"📄 Parsing video plan: {file_path}")

        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Extract script name from filename
        script_name = Path(file_path).stem

        # Parse each line for Real image sections
        sections = []
        lines = content.strip().split('\n')

        for line in lines:
            if not line.strip():
                continue

            # Pattern: 0.00-1.84: Real image of ... [transcript: ...]
            match = re.match(r'([\d.]+)-([\d.]+):\s*Real image:?\s*(?:of\s*)?(.+?)\s*\[transcript:\s*(.+?)\]', line, re.IGNORECASE)

            if match:
                start_time = float(match.group(1))
                end_time = float(match.group(2))
                description = match.group(3).strip()
                transcript = match.group(4).strip()

                section = RealImageSection(
                    start_time=start_time,
                    end_time=end_time,
                    description=description,
                    transcript=transcript,
                    script_name=script_name
                )
                sections.append(section)

        print(f"   ✅ Found {len(sections)} real image sections\n")

        return script_name, sections

# ========================================
# GOOGLE IMAGES SCRAPER (SELENIUM) - STEALTH MODE
# ========================================
import random
import time

class GoogleImagesScraper:
    """Google Images scraper using Selenium with anti-detection"""

    def __init__(self):
        """Initialize scraper"""
        print("🌐 Initializing Google Images scraper (Stealth mode)...")

        # Quality thresholds
        self.min_width = 800
        self.min_height = 800
        self.min_file_size = 50000  # 50KB minimum

        # Install Chrome and ChromeDriver
        self._setup_selenium()

        # Track requests to add delays
        self.last_request_time = 0
        self.request_count = 0

        print("✅ Scraper ready!\n")

    def _setup_selenium(self):
        """Setup Selenium with Chrome (maximum stealth)"""
        try:
            from selenium import webdriver
            from selenium.webdriver.chrome.options import Options
            from selenium.webdriver.common.by import By
            from selenium.webdriver.support.ui import WebDriverWait
            from selenium.webdriver.support import expected_conditions as EC

            # Store for later use
            self.webdriver = webdriver
            self.By = By
            self.WebDriverWait = WebDriverWait
            self.EC = EC

            # Setup Chrome options (MAXIMUM STEALTH)
            self.chrome_options = Options()
            self.chrome_options.add_argument('--headless=new')  # New headless mode (less detectable)
            self.chrome_options.add_argument('--no-sandbox')
            self.chrome_options.add_argument('--disable-dev-shm-usage')
            self.chrome_options.add_argument('--disable-gpu')
            self.chrome_options.add_argument('--window-size=1920,1080')
            self.chrome_options.add_argument('--start-maximized')

            # Anti-detection headers
            self.chrome_options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36')
            self.chrome_options.add_argument('accept-language=en-US,en;q=0.9')
            self.chrome_options.add_argument('accept=text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8')

            # Disable automation flags
            self.chrome_options.add_argument('--disable-blink-features=AutomationControlled')
            self.chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
            self.chrome_options.add_experimental_option('useAutomationExtension', False)

            # Additional anti-detection
            self.chrome_options.add_argument('--disable-infobars')
            self.chrome_options.add_argument('--disable-extensions')
            self.chrome_options.add_argument('--profile-directory=Default')
            self.chrome_options.add_argument('--incognito')
            self.chrome_options.add_argument('--disable-plugins-discovery')
            self.chrome_options.add_argument('--lang=en-US')

            self.driver = None

        except ImportError:
            print("Installing Selenium...")
            os.system('pip install selenium')
            self._setup_selenium()

    def _get_driver(self):
        """Get or create driver with anti-detection measures"""
        if self.driver is None:
            self.driver = self.webdriver.Chrome(options=self.chrome_options)

            # Remove webdriver property (anti-detection)
            self.driver.execute_cdp_cmd('Network.setUserAgentOverride', {
                "userAgent": 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36'
            })

            # Hide webdriver flag
            self.driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")

            # Add realistic browser properties
            self.driver.execute_script("""
                Object.defineProperty(navigator, 'languages', {get: () => ['en-US', 'en']});
                Object.defineProperty(navigator, 'plugins', {get: () => [1, 2, 3, 4, 5]});
            """)

        return self.driver

    def _human_delay(self, min_sec=1.0, max_sec=3.0):
        """Random human-like delay"""
        time.sleep(random.uniform(min_sec, max_sec))

    def _scroll_like_human(self, driver):
        """Scroll like a human (random increments)"""
        print(f"   📜 Scrolling naturally...")

        # Get page height
        last_height = driver.execute_script("return document.body.scrollHeight")

        # Scroll in random increments
        current_position = 0
        while current_position < last_height:
            # Random scroll distance (200-500px)
            scroll_distance = random.randint(200, 500)
            current_position += scroll_distance

            driver.execute_script(f"window.scrollTo(0, {current_position});")

            # Random pause between scrolls
            time.sleep(random.uniform(0.3, 0.8))

            # Update page height (content loads dynamically)
            last_height = driver.execute_script("return document.body.scrollHeight")

        # Scroll to bottom
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(random.uniform(1.5, 2.5))

    def _rate_limit(self):
        """Ensure we don't make requests too quickly"""
        # Wait at least 3-5 seconds between searches
        time_since_last = time.time() - self.last_request_time
        if time_since_last < 3:
            wait_time = random.uniform(3, 5) - time_since_last
            if wait_time > 0:
                print(f"   ⏳ Rate limiting: waiting {wait_time:.1f}s...")
                time.sleep(wait_time)

        self.last_request_time = time.time()
        self.request_count += 1

        # Every 5 searches, take a longer break
        if self.request_count % 5 == 0:
            print(f"   ☕ Taking a break after {self.request_count} searches...")
            time.sleep(random.uniform(10, 15))

    def search_and_get_full_res_urls(self, query: str, max_images: int = 20, max_retries: int = 3) -> List[Dict]:
        """
        Search Google Images and extract image URLs (stealth mode)

        Returns:
            List of dicts with 'url', 'width', 'height'
        """
        print(f"🔍 Searching Google Images: '{query}'")

        # Rate limiting
        self._rate_limit()

        for attempt in range(max_retries):
            try:
                print(f"   🚀 Fetching results (attempt {attempt + 1}/{max_retries})...")

                driver = self._get_driver()

                # Navigate to Google Images
                search_url = f"https://www.google.com/search?q={query.replace(' ', '+')}&tbm=isch"
                driver.get(search_url)

                # Initial page load delay (humans wait for page to load)
                self._human_delay(2.0, 3.5)

                print(f"   ✅ Page loaded successfully")

                # Scroll like a human
                self._scroll_like_human(driver)

                print(f"   🖼️ Extracting image URLs from page source...")

                # Get page source
                page_source = driver.page_source

                # Extract image URLs using regex
                import re

                # Multiple patterns to catch different URL formats
                patterns = [
                    r'"(https://[^"]+\.(?:jpg|jpeg|png|webp)[^"]*)"',  # Quoted URLs
                    r'https://[^\s"<>]+\.(?:jpg|jpeg|png|webp)',  # Unquoted URLs
                ]

                all_urls = []
                for pattern in patterns:
                    matches = re.findall(pattern, page_source)
                    all_urls.extend(matches)

                # Filter and deduplicate
                full_res_urls = []
                seen = set()

                # Blacklist patterns to skip
                skip_patterns = [
                    'gstatic',
                    'google.com/images',
                    'googleapis',
                    'googleusercontent',  # Google's CDN
                    'encrypted-tbn',  # Thumbnails
                    'data:image',  # Base64
                ]

                for url in all_urls:
                    # Clean URL (remove escape characters)
                    url = url.replace('\\u003d', '=').replace('\\u0026', '&')

                    # Skip Google's own images
                    if any(pattern in url.lower() for pattern in skip_patterns):
                        continue

                    # Skip if already seen
                    if url in seen:
                        continue

                    # Skip if URL is too short (likely invalid)
                    if len(url) < 20:
                        continue

                    seen.add(url)
                    full_res_urls.append({
                        'url': url,
                        'width': 0,
                        'height': 0
                    })

                    print(f"      ✅ Image {len(full_res_urls)}: {url[:60]}...")

                    if len(full_res_urls) >= max_images:
                        break

                if len(full_res_urls) == 0:
                    raise Exception("No image URLs extracted from page source")

                print(f"   ✅ Found {len(full_res_urls)} image URLs")

                # Display sample
                for i, img_data in enumerate(full_res_urls[:5], 1):
                    print(f"      📷 Image {i}: {img_data['url'][:60]}...")

                if len(full_res_urls) > 5:
                    print(f"      ... and {len(full_res_urls) - 5} more")

                print()
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
        # Pre-filter: reject known stock photo sites by URL
        stock_domains = [
            'alamy.com',
            'shutterstock.com',
            'gettyimages.com',
            'istockphoto.com',
            'adobestock.com',
            'dreamstime.com',
            '123rf.com',
            'depositphotos.com',
            'pond5.com',
            'bigstockphoto.com',
            'canstockphoto.com',
            'stocksy.com'
        ]

        # Check if URL contains any stock domain
        url_lower = url.lower()
        for domain in stock_domains:
            if domain in url_lower:
                return False, {'error': f'stock_site_{domain}'}

        try:
            # Randomize headers slightly (anti-fingerprinting)
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36',
                'Accept': 'image/avif,image/webp,image/apng,image/*,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.9',
                'Accept-Encoding': 'gzip, deflate, br',
                'Referer': 'https://www.google.com/',
                'DNT': '1',
                'Connection': 'keep-alive',
                'Sec-Fetch-Dest': 'image',
                'Sec-Fetch-Mode': 'no-cors',
                'Sec-Fetch-Site': 'cross-site',
            }

            # Small random delay before downloading
            time.sleep(random.uniform(0.1, 0.3))

            response = requests.get(url, headers=headers, timeout=15, stream=True)

            if response.status_code != 200:
                return False, {'error': f'HTTP {response.status_code}'}

            # Load image
            img = Image.open(io.BytesIO(response.content))

            # Check dimensions
            if img.width < self.min_width or img.height < self.min_height:
                return False, {
                    'error': 'too_small',
                    'width': img.width,
                    'height': img.height
                }

            # Check file size
            file_size = len(response.content)
            if file_size < self.min_file_size:
                return False, {
                    'error': 'file_too_small',
                    'size': file_size
                }

            # Convert to RGB if needed
            if img.mode == 'RGBA':
                background = Image.new('RGB', img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[-1])
                img = background
            elif img.mode not in ['RGB', 'L']:
                img = img.convert('RGB')

            # Save as PNG (lossless)
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
        """Close browser"""
        if self.driver:
            self.driver.quit()
            self.driver = None

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

    def _call_gemini(self, prompt: str, timeout: int = 30) -> str:
        """Call Gemini REST API with timeout"""
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

        try:
            response = requests.post(url, json=payload, timeout=timeout)

            if response.status_code == 200:
                result = response.json()
                text = result['candidates'][0]['content']['parts'][0]['text']
                return text.strip()
            elif response.status_code == 429:
                raise Exception("Rate limited")
            elif response.status_code == 403:
                raise Exception("API key invalid")
            else:
                raise Exception(f"API error: {response.status_code}")

        except requests.exceptions.Timeout:
            raise Exception("Request timed out")

    def _call_gemini_with_images(self, prompt: str, image_paths: List[str], timeout: int = 60) -> str:
        """Call Gemini with actual images via REST API"""
        url = f"{self.base_url}?key={self.api_key}"

        # Build parts with prompt and images
        parts = [{"text": prompt}]

        for img_path in image_paths:
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

        payload = {
            "contents": [{
                "parts": parts
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

    def determine_search_query(self, section: RealImageSection) -> str:
        """Let Gemini determine the best search query"""
        print(f"🧠 Asking Gemini for best search query...")

        prompt = f"""You are a Google Images search expert.

Given this real image request:
- Description: {section.description}
- Transcript context: {section.transcript}

What would be the BEST Google Images search query to find this?

Guidelines:
- Be specific and descriptive
- Include key visual elements
- Avoid overly generic terms
- Keep it under 8 words

Return ONLY the search query as plain text, no explanation.
Example: "Richarlison celebration Tottenham shirtless"
"""

        try:
            search_query = self._call_gemini(prompt, timeout=20)
            search_query = search_query.strip().strip('"\'').strip('.')
            print(f"   ✅ Gemini suggests: '{search_query}'\n")
            return search_query
        except Exception as e:
            print(f"   ❌ Gemini failed: {e}")
            # Fallback: use description
            fallback = section.description[:100]
            print(f"   🔄 Using fallback: '{fallback}'\n")
            return fallback

    def analyze_image_batch(self, candidates: List[ImageCandidate], section: RealImageSection) -> Dict:
        """
        Analyze actual images with visual validation
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

    def suggest_alternative_query(self, original_query: str, section: RealImageSection) -> str:
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

    def suggest_meme_query(self, section: RealImageSection) -> str:
        """Ask Gemini for a meme search query that captures the action's essence"""
        print(f"🎭 Asking Gemini for meme alternative...")

        prompt = f"""You are a meme expert helping find reaction images/memes that capture the ESSENCE of a moment.

    TARGET:
    - Description: {section.description}
    - Transcript: {section.transcript}

    YOUR TASK:
    Create a search query for a MEME or REACTION IMAGE that would humorously/accurately represent this moment.

    EXAMPLES:
    - "player celebrating shirtless" → "celebration meme"
    - "player missing shot" → "miss fail meme"
    - "player angry at referee" → "angry reaction meme"
    - "goalkeeper amazing save" → "superhero save meme"
    - "player injured on ground" → "pain meme"

    GUIDELINES:
    - Keep it 2-5 words
    - Use generic terms (unless player is meme-famous)
    - Think about the EMOTION/REACTION
    - Can include: "meme", "reaction", "funny"

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
            fallback = "funny reaction meme"
            print(f"   🔄 Using fallback: '{fallback}'\n")
            return fallback

# ========================================
# MAIN COLLECTOR
# ========================================
class AutomatedRealImageCollector:
    """Main orchestrator for automated real image collection"""

    def __init__(self, api_key: str, output_base_dir: str = "/content/drive/MyDrive/Angry Dude Scripts/collected_assets"):
        self.api_key = api_key
        self.output_base_dir = output_base_dir

        # Initialize components
        self.parser = RealImagePlanParser()
        self.scraper = GoogleImagesScraper()
        self.analyzer = GeminiImageAnalyzer(api_key)

        Path(output_base_dir).mkdir(exist_ok=True)
        print(f"✅ AutomatedRealImageCollector initialized")
        print(f"   Output base: {output_base_dir}\n")

    def collect_image_for_section(self, section: RealImageSection, tracker: ProgressTracker, max_retries: int = 2) -> bool:
        """
        Collect single image with iterative quality checking

        Process:
        1. Download top 5 candidates
        2. Gemini analyzes batch
        3. If confidence < 80%, download more and retry
        4. If still low, try alternative search query
        """
        print(f"🖼️ Processing: {section.description}")
        print(f"💬 Context: {section.transcript}\n")

        # Step 1: Determine search query
        search_query = self.analyzer.determine_search_query(section)

        attempt = 0
        all_candidates = []

        while attempt < max_retries:
            attempt += 1

            if attempt > 1:
                # Get alternative query for retry
                search_query = self.analyzer.suggest_alternative_query(search_query, section)

            # Step 2: Scrape Google Images
            image_urls = self.scraper.search_and_get_full_res_urls(search_query, max_images=20)

            if not image_urls:
                print(f"❌ No images found for: '{search_query}'")
                if attempt < max_retries:
                    continue
                else:
                    return False

            # Step 3: Download candidates
            print(f"📥 Downloading candidates (attempt {attempt})...")
            temp_dir = f"temp_real_images/{section.script_name}_{section.start_time}"
            os.makedirs(temp_dir, exist_ok=True)

            batch_candidates = []

            for i, img_data in enumerate(image_urls, 1):
                temp_path = f"{temp_dir}/candidate_{len(all_candidates) + i}.png"

                success, metadata = self.scraper.download_image(img_data['url'], temp_path)

                if success:
                    candidate = ImageCandidate(
                        path=temp_path,
                        url=metadata['url'],
                        width=metadata['width'],
                        height=metadata['height'],
                        file_size=metadata['file_size']
                    )
                    batch_candidates.append(candidate)
                    all_candidates.append(candidate)
                    print(f"   ✅ Candidate {len(all_candidates)}: {metadata['width']}x{metadata['height']}px ({metadata['file_size']/1024:.1f}KB)")
                else:
                    error = metadata.get('error', 'unknown')
                    print(f"   ⚠️ Skipped: {error}")

                # Check if we have enough for first analysis
                if len(all_candidates) >= 5:
                    break

            if len(all_candidates) < 3:
                print(f"⚠️ Only {len(all_candidates)} suitable images found")
                if attempt < max_retries:
                    continue
                else:
                    print(f"❌ Not enough quality images after {max_retries} attempts")
                    shutil.rmtree(temp_dir, ignore_errors=True)
                    return False

            # Step 4: Gemini analysis
            analysis = self.analyzer.analyze_image_batch(all_candidates[:10], section)  # Analyze up to 10

            confidence = analysis.get('confidence', 0)

            # Step 5: Check confidence threshold
            if confidence >= 80:
                # Good enough! Save the winner
                best_idx = analysis['best_image'] - 1
                best_candidate = all_candidates[best_idx]

                # Save to final location
                output_filename = f"{section.start_time:.2f}-{section.end_time:.2f}.png"
                output_dir = os.path.join(self.output_base_dir, section.script_name, "real_images")
                os.makedirs(output_dir, exist_ok=True)
                output_path = os.path.join(output_dir, output_filename)

                shutil.copy(best_candidate.path, output_path)

                # Save metadata
                self._save_metadata(section, best_candidate, search_query, confidence, output_path)

                print(f"✅ SAVED: {output_path}")
                # Mark section complete in tracker
                tracker.mark_section_completed(section, {
                    'output_path': output_path,
                    'search_query': search_query,
                    'confidence': confidence,
                    'width': best_candidate.width,
                    'height': best_candidate.height,
                    'file_size_kb': best_candidate.file_size / 1024,
                    'source_url': best_candidate.url
                })
                print(f"📊 Dimensions: {best_candidate.width}x{best_candidate.height}px")
                print(f"🎯 Confidence: {confidence}%\n")

                # Cleanup
                shutil.rmtree(temp_dir, ignore_errors=True)
                return True

            else:
                print(f"⚠️ Confidence too low ({confidence}% < 80%)")

                if len(all_candidates) < 10 and attempt < max_retries:
                    print(f"🔄 Downloading more candidates and retrying...\n")
                    continue
                elif attempt < max_retries:
                    print(f"🔄 Trying alternative search query...\n")
                    all_candidates = []  # Reset for new search
                    continue
                else:
                    # ===================================================================
                    # MEME FALLBACK - LAST RESORT
                    # ===================================================================
                    print(f"❌ Could not find suitable image after {max_retries} attempts")
                    print(f"🎭 Falling back to MEME search...\n")

                    # Get meme search query from Gemini
                    meme_query = self.analyzer.suggest_meme_query(section)

                    # Search for meme
                    meme_urls = self.scraper.search_and_get_full_res_urls(meme_query, max_images=20)

                    if not meme_urls:
                        print(f"❌ No memes found either")
                        tracker.mark_section_failed(section, "No images or memes found")
                        shutil.rmtree(temp_dir, ignore_errors=True)
                        return False

                    # Download meme candidates
                    print(f"📥 Downloading meme candidates...")
                    meme_temp_dir = f"{temp_dir}/memes"
                    os.makedirs(meme_temp_dir, exist_ok=True)

                    meme_candidates = []
                    for j, img_data in enumerate(meme_urls, 1):
                        meme_path = f"{meme_temp_dir}/meme_{j}.png"

                        success, metadata = self.scraper.download_image(img_data['url'], meme_path)

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
                        return False

                    # Analyze memes with relaxed criteria
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

                Score each meme 0-100 for "narrative fit".

                Return ONLY valid JSON:
                {{
                  "evaluations": [
                    {{"image": 1, "score": 75, "analysis": "Good match"}},
                    {{"image": 2, "score": 60, "analysis": "Acceptable"}}
                  ],
                  "best_image": 1,
                  "confidence": 75,
                  "reason": "Best captures the emotion"
                }}
                """

                    try:
                        # Send meme images to Gemini
                        meme_image_paths = [c.path for c in meme_candidates]
                        result_text = self.analyzer._call_gemini_with_images(meme_prompt, meme_image_paths, timeout=60)

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
                            output_filename = f"{section.start_time:.2f}-{section.end_time:.2f}.png"
                            output_dir = os.path.join(self.output_base_dir, section.script_name, "real_images")
                            os.makedirs(output_dir, exist_ok=True)
                            output_path = os.path.join(output_dir, output_filename)

                            shutil.copy(best_meme.path, output_path)

                            print(f"   💾 Saved meme: {output_path}")

                            # Save metadata with meme flag
                            metadata = {
                                'output_path': output_path,
                                'search_query': meme_query,
                                'original_query': search_query,
                                'confidence': meme_confidence,
                                'width': best_meme.width,
                                'height': best_meme.height,
                                'file_size_kb': best_meme.file_size / 1024,
                                'source_url': best_meme.url,
                                'fallback_mode': 'meme',
                                'reason': f"Meme fallback: {meme_analysis.get('reason', 'Best available')}"
                            }

                            tracker.mark_section_completed(section, metadata)

                            print(f"✅ Section completed with MEME fallback!")
                            print(f"📊 Dimensions: {best_meme.width}x{best_meme.height}px")
                            print(f"🎯 Confidence: {meme_confidence}%\n")

                            shutil.rmtree(temp_dir, ignore_errors=True)
                            return True
                        else:
                            print(f"❌ Meme confidence too low ({meme_confidence}%)")
                            # Emergency: use highest resolution meme anyway
                            if len(meme_candidates) > 0:
                                best_meme = max(meme_candidates, key=lambda c: c.width * c.height)

                                output_filename = f"{section.start_time:.2f}-{section.end_time:.2f}.png"
                                output_dir = os.path.join(self.output_base_dir, section.script_name, "real_images")
                                os.makedirs(output_dir, exist_ok=True)
                                output_path = os.path.join(output_dir, output_filename)

                                shutil.copy(best_meme.path, output_path)

                                metadata = {
                                    'output_path': output_path,
                                    'search_query': meme_query,
                                    'confidence': 40,
                                    'width': best_meme.width,
                                    'height': best_meme.height,
                                    'fallback_mode': 'meme_emergency',
                                    'reason': 'Emergency fallback: highest resolution meme'
                                }

                                tracker.mark_section_completed(section, metadata)

                                print(f"✅ Section completed with emergency meme fallback!\n")

                                shutil.rmtree(temp_dir, ignore_errors=True)
                                return True
                            else:
                                tracker.mark_section_failed(section, "All fallbacks exhausted")
                                shutil.rmtree(temp_dir, ignore_errors=True)
                                return False

                    except Exception as e:
                        print(f"❌ Meme selection failed: {e}")
                        # Last resort: use highest resolution meme
                        if len(meme_candidates) > 0:
                            best_meme = max(meme_candidates, key=lambda c: c.width * c.height)

                            output_filename = f"{section.start_time:.2f}-{section.end_time:.2f}.png"
                            output_dir = os.path.join(self.output_base_dir, section.script_name, "real_images")
                            os.makedirs(output_dir, exist_ok=True)
                            output_path = os.path.join(output_dir, output_filename)

                            shutil.copy(best_meme.path, output_path)

                            metadata = {
                                'output_path': output_path,
                                'search_query': meme_query,
                                'confidence': 40,
                                'fallback_mode': 'meme_emergency',
                                'reason': 'Emergency fallback after exception'
                            }

                            tracker.mark_section_completed(section, metadata)
                            print(f"✅ Section completed with emergency meme fallback!\n")

                            shutil.rmtree(temp_dir, ignore_errors=True)
                            return True
                        else:
                            tracker.mark_section_failed(section, "All fallbacks exhausted")
                            shutil.rmtree(temp_dir, ignore_errors=True)
                            return False

        return False

    def _save_metadata(self, section, candidate, search_query, confidence, output_path):
        """Save metadata for collected image"""
        metadata = {
            'filename': os.path.basename(output_path),
            'filepath': output_path,
            'timestamp': f"{section.start_time:.2f}-{section.end_time:.2f}",
            'description': section.description,
            'transcript': section.transcript,
            'search_query': search_query,
            'source_url': candidate.url,
            'dimensions': f"{candidate.width}x{candidate.height}",
            'file_size_kb': candidate.file_size / 1024,
            'confidence': confidence,
            'collection_method': 'automated_gemini_selection',
            'format': 'PNG'
        }

        output_dir = os.path.dirname(output_path)
        manifest_path = os.path.join(output_dir, "image_metadata.json")

        if Path(manifest_path).exists():
            with open(manifest_path, 'r', encoding='utf-8') as f:
                manifest = json.load(f)
        else:
            manifest = {'images': []}

        manifest['images'].append(metadata)

        with open(manifest_path, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=2)

    def process_video_plan(self, plan_file_path: str):
        """Process entire video plan for real images"""
        print("="*70)
        print("🖼️ STARTING REAL IMAGE COLLECTION")
        print("="*70)
        print()

        # Parse plan
        script_name, sections = self.parser.parse_plan(plan_file_path)
        # Initialize progress tracker
        tracker = ProgressTracker(script_name, self.output_base_dir)
        tracker.data['stats']['total_sections'] = len(sections)

        if not sections:
            print("⚠️ No real image sections found in this plan!")
            return

        print(f"📋 Processing {len(sections)} real image sections...\n")

        total_success = 0
        total_failed = 0

        # Process each section
        for i, section in enumerate(sections, 1):
            print("="*70)
            print(f"SECTION {i}/{len(sections)}")
            print("="*70)
            # Check if already completed
            section_key = f"{section.start_time:.2f}-{section.end_time:.2f}"
            if tracker.is_section_complete(section_key):
                print(f"✅ Already completed - skipping\n")
                continue

            # Mark as started
            tracker.mark_section_started(section)

            try:
                success = self.collect_image_for_section(section, tracker)

                if success:
                    total_success += 1
                    # Already marked complete in collect_image_for_section
                else:
                    total_failed += 1
                    tracker.mark_section_failed(section, "Could not find suitable image after retries")

            except Exception as e:
                print(f"❌ Error processing section: {e}")
                total_failed += 1
                continue

        # Final summary
        print("="*70)
        print("🎉 REAL IMAGE COLLECTION COMPLETE!")
        print("="*70)
        print(f"✅ Successfully collected: {total_success}")
        print(f"❌ Failed: {total_failed}")
        print(f"📊 Progress tracker: {tracker.progress_file}")
        print(f"📊 Total processed: {total_success + total_failed}/{len(sections)}")
        print(f"📁 Images saved to: collected_assets/{script_name}/real_images/\n")

    def cleanup(self):
        """Cleanup resources"""
        self.scraper.close()

# ========================================
# MAIN EXECUTION
# ========================================

print("="*70)
print("🖼️ AUTOMATED REAL IMAGE COLLECTOR")
print("="*70)
print()

# Step 1: List and select video plan from Drive
video_plans_dir = "/content/drive/MyDrive/Angry Dude Scripts/video_plans"

print(f"📂 Scanning for video plans in: {video_plans_dir}\n")

if not os.path.exists(video_plans_dir):
    print(f"❌ Directory not found: {video_plans_dir}")
    raise SystemExit()

# Get all .txt files
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

# Step 2: Initialize and run
API_KEY = ""  #input api key here or setup with .env
collector = AutomatedRealImageCollector(API_KEY)

try:
    # Process the plan
    collector.process_video_plan(plan_file)
finally:
    # Cleanup
    collector.cleanup()
    print("\n🎊 ALL DONE! Check your collected_assets folder for the images.")