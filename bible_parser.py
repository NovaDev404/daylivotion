import os
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional

class BibleParser:
    def __init__(self, bible_path: str = "/media/novadev/storage/daylivotion/en_udb"):
        self.bible_path = Path(bible_path)
        self.books = self._load_books()
    
    def _load_books(self) -> Dict[str, Path]:
        """Load all USFM Bible files and map book codes to file paths."""
        books = {}
        for file in self.bible_path.glob("*.usfm"):
            if file.name.startswith(("00-", "01-")):
                continue  # Skip front matter and about files
            match = re.match(r'(\d+)-([A-Z0-9]+)\.usfm', file.name)
            if match:
                book_num = match.group(1)
                book_code = match.group(2)
                books[book_code] = file
        return books
    
    def get_book_names(self) -> List[str]:
        """Get list of available book codes."""
        return list(self.books.keys())
    
    def parse_file(self, book_code: str) -> Dict:
        """Parse a USFM file and return structured verse data."""
        if book_code not in self.books:
            raise ValueError(f"Book {book_code} not found")
        
        file_path = self.books[book_code]
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract book name from \h or \id
        book_name_match = re.search(r'\\h\s+(.+)', content)
        if book_name_match:
            book_name = book_name_match.group(1).strip()
        else:
            book_name = book_code
        
        # Parse chapters and verses
        chapters = {}
        current_chapter = None
        current_verses = {}
        current_verse_num = None
        
        lines = content.split('\n')
        for line in lines:
            # Chapter marker
            chapter_match = re.match(r'\\c\s+(\d+)', line)
            if chapter_match:
                if current_chapter is not None:
                    chapters[current_chapter] = current_verses
                current_chapter = int(chapter_match.group(1))
                current_verses = {}
                current_verse_num = None
                continue
            
            # Verse marker
            verse_match = re.match(r'\\v\s+(\d+)(?:-(\d+))?\s+(.+)', line)
            if verse_match:
                verse_num = int(verse_match.group(1))
                verse_end = verse_match.group(2)
                verse_text = verse_match.group(3).strip()
                
                if verse_end:
                    # Verse range (e.g., 16-17)
                    verse_end_num = int(verse_end)
                    for v in range(verse_num, verse_end_num + 1):
                        current_verses[v] = verse_text
                else:
                    current_verses[verse_num] = verse_text
                current_verse_num = verse_num
            elif current_verse_num is not None and line.strip():
                # Continuation of current verse (lines without verse marker)
                # Skip USFM markers like \q1, \q2, \p, \s5
                if not line.startswith('\\'):
                    current_verses[current_verse_num] += ' ' + line.strip()
                elif line.startswith('\\q'):
                    # Poetry continuation lines
                    text = re.sub(r'\\q\d?\s*', '', line).strip()
                    if text:
                        current_verses[current_verse_num] += ' ' + text
        
        # Add last chapter
        if current_chapter is not None:
            chapters[current_chapter] = current_verses
        
        return {
            'book_code': book_code,
            'book_name': book_name,
            'chapters': chapters
        }
    
    def get_random_verse(self) -> Tuple[str, str, str]:
        """Get a random verse reference and text."""
        import random
        
        book_codes = list(self.books.keys())
        if not book_codes:
            raise ValueError("No Bible books found")
        
        # Pick random book
        book_code = random.choice(book_codes)
        book_data = self.parse_file(book_code)
        
        # Pick random chapter
        chapters = list(book_data['chapters'].keys())
        if not chapters:
            raise ValueError(f"No chapters found in {book_code}")
        
        chapter = random.choice(chapters)
        verses = book_data['chapters'][chapter]
        
        # Pick random verse
        verse_nums = list(verses.keys())
        if not verse_nums:
            raise ValueError(f"No verses found in {book_code} {chapter}")
        
        verse = random.choice(verse_nums)
        verse_text = verses[verse]
        
        # Format reference
        reference = f"{book_data['book_name']} {chapter}:{verse}"
        
        return reference, verse_text, book_code
    
    def get_verse_by_reference(self, book_code: str, chapter: int, verse: int) -> Optional[str]:
        """Get verse text by reference."""
        try:
            book_data = self.parse_file(book_code)
            if chapter in book_data['chapters'] and verse in book_data['chapters'][chapter]:
                return book_data['chapters'][chapter][verse]
        except Exception:
            pass
        return None
    
    def get_verse_range_by_reference(self, book_code: str, chapter: int, verse_start: int, verse_end: int) -> Optional[str]:
        """Get verse range text by reference."""
        try:
            book_data = self.parse_file(book_code)
            if chapter in book_data['chapters']:
                verses = []
                for v in range(verse_start, verse_end + 1):
                    if v in book_data['chapters'][chapter]:
                        verses.append(book_data['chapters'][chapter][v])
                if verses:
                    return ' '.join(verses)
        except Exception:
            pass
        return None
