import requests
from bs4 import BeautifulSoup
import os
import json
import urllib3
from urllib.parse import urljoin, urlparse
from collections import deque
from pathlib import Path
from typing import Set, Dict, List
import time

# =====================================
# DISABLE SSL WARNINGS
# =====================================
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# =====================================
# CONFIG
# =====================================
class Config:
    BASE_URL = "https://nitkkr.ac.in"
    BASE_DIR = Path("data")
    PDF_FOLDER = BASE_DIR / "pdfs"
    HTML_FOLDER = BASE_DIR / "html"
    EXCEL_FOLDER = BASE_DIR / "excel"
    PROCESSED_DIR = Path("processed")
    
    MAX_PAGES = 100
    REQUEST_TIMEOUT = 15
    REQUEST_DELAY = 0.1  # Polite crawling delay
    
    VIDEO_EXTENSIONS = ('.mp4', '.avi', '.mov', '.wmv', '.mkv', '.webm')
    PDF_EXTENSIONS = ('.pdf',)
    EXCEL_EXTENSIONS = ('.xls', '.xlsx')
    
    def __init__(self):
        self._create_directories()
    
    def _create_directories(self):
        for folder in [self.PDF_FOLDER, self.HTML_FOLDER, 
                       self.EXCEL_FOLDER, self.PROCESSED_DIR]:
            folder.mkdir(parents=True, exist_ok=True)

# =====================================
# WEB CRAWLER CLASS
# =====================================
class WebCrawler:
    def __init__(self, config: Config):
        self.config = config
        self.session = self._create_session()
        
        # State management
        self.visited: Set[str] = set()
        self.downloaded_urls: Set[str] = set()
        self.queue: deque = deque([config.BASE_URL])
        
        # Metadata storage
        self.downloaded_files: List[Dict] = []
        self.html_pages: List[Dict] = []
        self.navigation: List[Dict] = []
        self.visited_navigation: Set[tuple] = set()
    
    def _create_session(self) -> requests.Session:
        """Create a reusable session with connection pooling"""
        session = requests.Session()
        session.verify = False
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=10,
            pool_maxsize=20,
            max_retries=3
        )
        session.mount('http://', adapter)
        session.mount('https://', adapter)
        return session
    
    def is_internal_link(self, link: str) -> bool:
        """Check if link is internal to the base domain"""
        base_domain = urlparse(self.config.BASE_URL).netloc
        link_domain = urlparse(link).netloc
        return base_domain == link_domain or link_domain == ""
    
    @staticmethod
    def clean_filename(url: str) -> str:
        """Generate clean filename from URL"""
        parsed = urlparse(url)
        path = parsed.path.strip("/") or "index"
        return path.replace("/", "_")
    
    @staticmethod
    def normalize_url(url: str) -> str:
        """Remove fragments and normalize URL"""
        return url.split("#")[0].rstrip('/')
    
    def save_html_page(self, url: str, soup: BeautifulSoup) -> str:
        """Extract and save page text content"""
        text = soup.get_text(separator="\n", strip=True)
        filename = self.clean_filename(url) + ".txt"
        filepath = self.config.HTML_FOLDER / filename
        
        filepath.write_text(text, encoding="utf-8")
        
        title = soup.title.text.strip() if soup.title else ""
        
        self.html_pages.append({
            "url": url,
            "title": title,
            "file": filename
        })
        
        print(f"✓ Saved HTML: {filename}")
        return title
    
    def download_file(self, file_url: str, folder: Path, file_type: str):
        """Download PDF or Excel file"""
        if file_url in self.downloaded_urls:
            return
        
        try:
            response = self.session.get(
                file_url,
                timeout=self.config.REQUEST_TIMEOUT,
                stream=True
            )
            
            if response.status_code == 200:
                filename = file_url.split("/")[-1]
                if not filename:
                    return
                
                filepath = folder / filename
                
                # Stream large files
                with open(filepath, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                self.downloaded_urls.add(file_url)
                self.downloaded_files.append({
                    "filename": filename,
                    "url": file_url,
                    "type": file_type
                })
                
                print(f"⬇ Downloaded {file_type.upper()}: {filename}")
        
        except Exception as e:
            print(f"✗ Download error ({file_url}): {e}")
    
    def process_link(self, link, current_url: str, current_title: str, soup: BeautifulSoup):
        """Process a single link from the page"""
        href = link.get("href")
        link_text = link.get_text(strip=True)
        
        if not href:
            return
        
        full_url = self.normalize_url(urljoin(current_url, href))
        
        # Skip video files
        if full_url.lower().endswith(self.config.VIDEO_EXTENSIONS):
            return
        
        # Save navigation metadata
        if link_text and len(link_text) > 2 and self.is_internal_link(full_url):
            nav_key = (full_url.lower(), link_text.lower())
            
            if nav_key not in self.visited_navigation:
                self.visited_navigation.add(nav_key)
                self.navigation.append({
                    "parent_title": current_title,
                    "parent_url": current_url,
                    "title": link_text.strip(),
                    "url": full_url,
                    "type": "page"
                })
        
        # Handle different file types
        lower_url = full_url.lower()
        
        if lower_url.endswith(self.config.PDF_EXTENSIONS):
            print(f"📄 PDF Found: {full_url}")
            self.download_file(full_url, self.config.PDF_FOLDER, "pdf")
        
        elif lower_url.endswith(self.config.EXCEL_EXTENSIONS):
            print(f"📊 Excel Found: {full_url}")
            self.download_file(full_url, self.config.EXCEL_FOLDER, "excel")
        
        elif self.is_internal_link(full_url):
            if full_url not in self.visited and full_url not in self.queue:
                self.queue.append(full_url)
    
    def crawl_page(self, url: str):
        """Crawl a single page"""
        try:
            response = self.session.get(
                url,
                timeout=self.config.REQUEST_TIMEOUT
            )
            
            if response.status_code != 200:
                print(f"✗ Status {response.status_code}: {url}")
                return
            
            soup = BeautifulSoup(response.text, "html.parser")
            
            # Save page content
            current_title = self.save_html_page(url, soup)
            
            # Process all links
            links = soup.select("a[href]")
            for link in links:
                self.process_link(link, url, current_title, soup)
            
            # Polite crawling delay
            time.sleep(self.config.REQUEST_DELAY)
        
        except Exception as e:
            print(f"✗ Error crawling {url}: {e}")
    
    def crawl(self):
        """Main crawling loop"""
        print(f"🚀 Starting crawler on {self.config.BASE_URL}\n")
        
        while self.queue and len(self.visited) < self.config.MAX_PAGES:
            current_url = self.queue.popleft()
            
            if current_url in self.visited:
                continue
            
            self.visited.add(current_url)
            print(f"\n[{len(self.visited)}/{self.config.MAX_PAGES}] Crawling: {current_url}")
            
            self.crawl_page(current_url)
        
        print(f"\n✓ Crawling completed. Visited {len(self.visited)} pages.")
    
    def save_metadata(self):
        """Save all metadata to JSON files"""
        # Sort navigation
        self.navigation.sort(key=lambda x: (x["parent_title"], x["title"]))
        
        metadata = {
            "file_urls.json": self.downloaded_files,
            "html_pages.json": self.html_pages,
            "navigation.json": self.navigation
        }
        
        for filename, data in metadata.items():
            filepath = self.config.PROCESSED_DIR / filename
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print(f"💾 Saved {len(data)} entries to {filename}")
    
    def print_summary(self):
        """Print crawling summary"""
        print("\n" + "="*50)
        print("CRAWLING SUMMARY")
        print("="*50)
        print(f"Pages visited: {len(self.visited)}")
        print(f"Files downloaded: {len(self.downloaded_files)}")
        print(f"HTML pages saved: {len(self.html_pages)}")
        print(f"Navigation entries: {len(self.navigation)}")
        print("="*50)

# =====================================
# MAIN EXECUTION
# =====================================
def main():
    config = Config()
    crawler = WebCrawler(config)
    
    try:
        crawler.crawl()
        crawler.save_metadata()
        crawler.print_summary()
    except KeyboardInterrupt:
        print("\n\n⚠ Crawling interrupted by user")
        crawler.save_metadata()
        crawler.print_summary()
    except Exception as e:
        print(f"\n✗ Fatal error: {e}")
        raise

if __name__ == "__main__":
    main()
