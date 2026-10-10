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

urllib3.disable_warnings(
    urllib3.exceptions.InsecureRequestWarning
)


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

    # Increased from 100
    MAX_PAGES = 500

    REQUEST_TIMEOUT = 15

    REQUEST_DELAY = 0.1

    VIDEO_EXTENSIONS = (
        '.mp4',
        '.avi',
        '.mov',
        '.wmv',
        '.mkv',
        '.webm'
    )

    PDF_EXTENSIONS = (
        '.pdf',
    )

    EXCEL_EXTENSIONS = (
        '.xls',
        '.xlsx'
    )

    def __init__(self):

        self._create_directories()

    def _create_directories(self):

        for folder in [
            self.PDF_FOLDER,
            self.HTML_FOLDER,
            self.EXCEL_FOLDER,
            self.PROCESSED_DIR
        ]:

            folder.mkdir(
                parents=True,
                exist_ok=True
            )


# =====================================
# WEB CRAWLER CLASS
# =====================================

class WebCrawler:

    def __init__(self, config: Config):

        self.config = config

        self.session = self._create_session()

        # =================================
        # STATE
        # =================================

        self.visited: Set[str] = set()

        self.downloaded_urls: Set[str] = set()

        self.queue: deque = deque(
            [config.BASE_URL]
        )

        # =================================
        # METADATA
        # =================================

        self.downloaded_files: List[Dict] = []

        self.html_pages: List[Dict] = []

        self.navigation: List[Dict] = []

        self.visited_navigation: Set[tuple] = set()


    # =====================================
    # CREATE SESSION
    # =====================================

    def _create_session(self) -> requests.Session:

        session = requests.Session()

        session.verify = False

        adapter = requests.adapters.HTTPAdapter( # type: ignore
            pool_connections=10,
            pool_maxsize=20,
            max_retries=3
        )

        session.mount(
            'http://',
            adapter
        )

        session.mount(
            'https://',
            adapter
        )

        return session


    # =====================================
    # INTERNAL LINK CHECK
    # =====================================

    def is_internal_link(
        self,
        link: str
    ) -> bool:

        base_domain = urlparse(
            self.config.BASE_URL
        ).netloc

        link_domain = urlparse(
            link
        ).netloc

        return (
            base_domain == link_domain
            or link_domain == ""
        )


    # =====================================
    # CLEAN FILENAME
    # =====================================

    @staticmethod
    def clean_filename(url: str) -> str:

        parsed = urlparse(url)

        path = (
            parsed.path.strip("/")
            or "index"
        )

        return path.replace(
            "/",
            "_"
        )


    # =====================================
    # NORMALIZE URL
    # =====================================

    @staticmethod
    def normalize_url(
        url: str
    ) -> str:

        return (
            url
            .split("#")[0]
            .rstrip("/")
        )


    # =====================================
    # SAVE HTML PAGE
    # =====================================

    def save_html_page(
        self,
        url: str,
        soup: BeautifulSoup
    ) -> str:

        text = soup.get_text(
            separator="\n",
            strip=True
        )

        filename = (
            self.clean_filename(url)
            + ".txt"
        )

        filepath = (
            self.config.HTML_FOLDER
            / filename
        )

        filepath.write_text(
            text,
            encoding="utf-8"
        )

        title = (
            soup.title.text.strip()
            if soup.title
            else ""
        )

        self.html_pages.append({

            "url": url,

            "title": title,

            "file": filename

        })

        print(
            f"✓ Saved HTML: {filename}"
        )

        return title


    # =====================================
    # DOWNLOAD FILE
    # =====================================

    def download_file(
        self,
        file_url: str,
        folder: Path,
        file_type: str,
        response=None
    ):

        if file_url in self.downloaded_urls:

            return


        try:

            # ---------------------------------
            # Reuse existing response if given
            # ---------------------------------

            if response is None:

                response = self.session.get(
                    file_url,
                    timeout=self.config.REQUEST_TIMEOUT,
                    stream=True
                )

            if response.status_code != 200:

                print(
                    f"✗ Download failed "
                    f"({response.status_code}): "
                    f"{file_url}"
                )

                return


            # ---------------------------------
            # Determine filename
            # ---------------------------------

            parsed_url = urlparse(
                file_url
            )

            filename = os.path.basename(
                parsed_url.path
            )

            # ---------------------------------
            # Fallback filename
            # ---------------------------------

            if not filename:

                filename = (
                    self.clean_filename(
                        file_url
                    )
                )

                if file_type == "pdf":

                    filename += ".pdf"

                elif file_type == "excel":

                    filename += ".xlsx"


            # ---------------------------------
            # Save file
            # ---------------------------------

            filepath = (
                folder
                / filename
            )

            with open(
                filepath,
                "wb"
            ) as f:

                for chunk in response.iter_content(
                    chunk_size=8192
                ):

                    if chunk:

                        f.write(chunk)


            # ---------------------------------
            # Store URL metadata
            # ---------------------------------

            self.downloaded_urls.add(
                file_url
            )

            self.downloaded_files.append({

                "filename": filename,

                "url": file_url,

                "type": file_type

            })


            print(
                f"⬇ Downloaded "
                f"{file_type.upper()}: "
                f"{filename}"
            )


        except Exception as e:

            print(
                f"✗ Download error "
                f"({file_url}): {e}"
            )


    # =====================================
    # PROCESS LINK
    # =====================================

    def process_link(
        self,
        link,
        current_url: str,
        current_title: str,
        soup: BeautifulSoup
    ):

        href = link.get(
            "href"
        )

        link_text = link.get_text(
            strip=True
        )

        if not href:

            return


        # =================================
        # BUILD FULL URL
        # =================================

        full_url = self.normalize_url(
            urljoin(
                current_url,
                href
            )
        )


        # =================================
        # SKIP VIDEOS
        # =================================

        if full_url.lower().endswith(
            self.config.VIDEO_EXTENSIONS
        ):

            return


        # =================================
        # NAVIGATION METADATA
        # =================================

        if (
            link_text
            and len(link_text) > 2
            and self.is_internal_link(full_url)
        ):

            nav_key = (
                full_url.lower(),
                link_text.lower()
            )

            if nav_key not in self.visited_navigation:

                self.visited_navigation.add(
                    nav_key
                )

                self.navigation.append({

                    "parent_title":
                        current_title,

                    "parent_url":
                        current_url,

                    "title":
                        link_text.strip(),

                    "url":
                        full_url,

                    "type":
                        "page"

                })


        # =================================
        # FILE TYPE CHECK
        # =================================

        lower_url = full_url.lower()


        # =================================
        # PDF
        # =================================

        if lower_url.endswith(
            self.config.PDF_EXTENSIONS
        ):

            print(
                f"📄 PDF Found: {full_url}"
            )

            self.download_file(
                full_url,
                self.config.PDF_FOLDER,
                "pdf"
            )

            return


        # =================================
        # EXCEL
        # =================================

        if lower_url.endswith(
            self.config.EXCEL_EXTENSIONS
        ):

            print(
                f"📊 Excel Found: {full_url}"
            )

            self.download_file(
                full_url,
                self.config.EXCEL_FOLDER,
                "excel"
            )

            return


        # =================================
        # INTERNAL HTML PAGE
        # =================================

        if self.is_internal_link(
            full_url
        ):

            if (
                full_url not in self.visited
                and full_url not in self.queue
            ):

                self.queue.append(
                    full_url
                )


    # =====================================
    # CRAWL PAGE
    # =====================================

    def crawl_page(
        self,
        url: str
    ):

        try:

            response = self.session.get(
                url,
                timeout=self.config.REQUEST_TIMEOUT
            )


            if response.status_code != 200:

                print(
                    f"✗ Status "
                    f"{response.status_code}: "
                    f"{url}"
                )

                return


            # =================================
            # CHECK CONTENT TYPE
            # =================================

            content_type = response.headers.get(
                "Content-Type",
                ""
            ).lower()


            # =================================
            # DIRECT PDF URL
            # =================================

            if (
                "application/pdf"
                in content_type
            ):

                print(
                    f"📄 PDF Found: {url}"
                )

                self.download_file(
                    url,
                    self.config.PDF_FOLDER,
                    "pdf",
                    response=response
                )

                return


            # =================================
            # ONLY PROCESS HTML
            # =================================

            if (
                "text/html"
                not in content_type
                and content_type
            ):

                return


            soup = BeautifulSoup(
                response.text,
                "html.parser"
            )


            # =================================
            # SAVE HTML
            # =================================

            current_title = (
                self.save_html_page(
                    url,
                    soup
                )
            )


            # =================================
            # PROCESS LINKS
            # =================================

            links = soup.select(
                "a[href]"
            )

            for link in links:

                self.process_link(
                    link,
                    url,
                    current_title,
                    soup
                )


            # =================================
            # POLITE DELAY
            # =================================

            time.sleep(
                self.config.REQUEST_DELAY
            )


        except Exception as e:

            print(
                f"✗ Error crawling "
                f"{url}: {e}"
            )


    # =====================================
    # MAIN CRAWLER
    # =====================================

    def crawl(self):

        print(
            f"🚀 Starting crawler on "
            f"{self.config.BASE_URL}\n"
        )


        while (
            self.queue
            and len(self.visited)
            < self.config.MAX_PAGES
        ):

            current_url = (
                self.queue.popleft()
            )


            if current_url in self.visited:

                continue


            self.visited.add(
                current_url
            )


            print(
                f"\n[{len(self.visited)}/"
                f"{self.config.MAX_PAGES}] "
                f"Crawling: "
                f"{current_url}"
            )


            self.crawl_page(
                current_url
            )


        print(
            f"\n✓ Crawling completed. "
            f"Visited "
            f"{len(self.visited)} pages."
        )


    # =====================================
    # SAVE METADATA
    # =====================================

    def save_metadata(self):

        # =================================
        # SORT NAVIGATION
        # =================================

        self.navigation.sort(
            key=lambda x: (
                x["parent_title"],
                x["title"]
            )
        )


        metadata = {

            "file_urls.json":
                self.downloaded_files,

            "html_pages.json":
                self.html_pages,

            "navigation.json":
                self.navigation

        }


        for filename, data in metadata.items():

            filepath = (
                self.config.PROCESSED_DIR
                / filename
            )


            with open(
                filepath,
                "w",
                encoding="utf-8"
            ) as f:

                json.dump(
                    data,
                    f,
                    indent=2,
                    ensure_ascii=False
                )


            print(
                f"💾 Saved "
                f"{len(data)} entries "
                f"to {filename}"
            )


    # =====================================
    # PRINT SUMMARY
    # =====================================

    def print_summary(self):

        print(
            "\n"
            + "=" * 50
        )

        print(
            "CRAWLING SUMMARY"
        )

        print(
            "=" * 50
        )

        print(
            f"Pages visited: "
            f"{len(self.visited)}"
        )

        print(
            f"Files downloaded: "
            f"{len(self.downloaded_files)}"
        )

        print(
            f"HTML pages saved: "
            f"{len(self.html_pages)}"
        )

        print(
            f"Navigation entries: "
            f"{len(self.navigation)}"
        )

        print(
            "=" * 50
        )


# =====================================
# MAIN EXECUTION
# =====================================

def main():

    config = Config()

    crawler = WebCrawler(
        config
    )


    try:

        crawler.crawl()

        crawler.save_metadata()

        crawler.print_summary()


    except KeyboardInterrupt:

        print(
            "\n\n⚠ Crawling interrupted "
            "by user"
        )

        crawler.save_metadata()

        crawler.print_summary()


    except Exception as e:

        print(
            f"\n✗ Fatal error: {e}"
        )

        raise


# =====================================
# ENTRY POINT
# =====================================

if __name__ == "__main__":

    main()