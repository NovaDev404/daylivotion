import re
import random
import threading
import time
import queue
from typing import Dict, Optional, Generator
from bible_parser import BibleParser
from ai_runner import AIRunner

class DevotionGenerator:
    def __init__(self):
        self.bible_parser = BibleParser()
        self.ai_runner = AIRunner()
        self.generation_lock = threading.Lock()
        self.queue = []
        self.queue_counter = 0
        self.queue_lock = threading.Lock()
        self.current_client_id = None
        self.cancel_requested = threading.Event()
    
    def parse_verse_reference(self, ref: str) -> Optional[Dict]:
        """Parse a verse reference like 'Genesis 1:1' or 'Romans 1:2-3' or '1 Corinthians 13:4-7'."""
        # Format: BOOK CHAPTER:VERSE or BOOK CHAPTER:VERSE-VERSE (book can have numbers)
        pattern = r'^([A-Za-z0-9\s]+) (\d+):(\d+)(?:-(\d+))?$'
        match = re.match(pattern, ref.strip())
        
        if not match:
            return None
        
        book_name = match.group(1).strip()
        chapter = int(match.group(2))
        verse_start = int(match.group(3))
        verse_end = match.group(4)
        
        if verse_end:
            verse_end = int(verse_end)
        
        return {
            'book_name': book_name,
            'chapter': chapter,
            'verse_start': verse_start,
            'verse_end': verse_end
        }
    
    def find_book_code_by_name(self, book_name: str) -> Optional[str]:
        """Find book code by book name (e.g., 'Genesis' -> 'GEN')."""
        # Common abbreviation mappings
        abbreviations = {
            'gen': 'GENESIS', 'exod': 'EXODUS', 'lev': 'LEVITICUS', 'num': 'NUMBERS',
            'deut': 'DEUTERONOMY', 'josh': 'JOSHUA', 'judg': 'JUDGES', 'ruth': 'RUTH',
            '1 sam': '1 SAMUEL', '2 sam': '2 SAMUEL', '1 ki': '1 KINGS', '2 ki': '2 KINGS',
            '1 ch': '1 CHRONICLES', '2 ch': '2 CHRONICLES', 'ezra': 'EZRA', 'neh': 'NEHEMIAH',
            'est': 'ESTHER', 'job': 'JOB', 'ps': 'PSALMS', 'prov': 'PROVERBS',
            'eccl': 'ECCLESIASTES', 'song': 'SONG OF SONGS', 'isa': 'ISAIAH', 'jer': 'JEREMIAH',
            'lam': 'LAMENTATIONS', 'ezek': 'EZEKIEL', 'dan': 'DANIEL', 'hos': 'HOSEA',
            'joel': 'JOEL', 'amos': 'AMOS', 'obad': 'OBADIAH', 'jonah': 'JONAH',
            'mic': 'MICAH', 'nah': 'NAHUM', 'hab': 'HABAKKUK', 'zeph': 'ZEPHANIAH',
            'hag': 'HAGGAI', 'zech': 'ZECHARIAH', 'mal': 'MALACHI', 'mat': 'MATTHEW',
            'mark': 'MARK', 'lk': 'LUKE', 'jn': 'JOHN', 'acts': 'ACTS', 'rom': 'ROMANS',
            '1 cor': '1 CORINTHIANS', '2 cor': '2 CORINTHIANS', 'gal': 'GALATIANS',
            'eph': 'EPHESIANS', 'phil': 'PHILIPPIANS', 'col': 'COLOSSIANS',
            '1 thess': '1 THESSALONIANS', '2 thess': '2 THESSALONIANS',
            '1 tim': '1 TIMOTHY', '2 tim': '2 TIMOTHY', 'titus': 'TITUS', 'phlm': 'PHILEMON',
            'heb': 'HEBREWS', 'jas': 'JAMES', '1 pet': '1 PETER', '2 pet': '2 PETER',
            '1 jn': '1 JOHN', '2 jn': '2 JOHN', '3 jn': '3 JOHN', 'jude': 'JUDE', 'rev': 'REVELATION',
            '1 co': '1 CORINTHIANS', '2 co': '2 CORINTHIANS', '1 th': '1 THESSALONIANS',
            '2 th': '2 THESSALONIANS', '1 ti': '1 TIMOTHY', '2 ti': '2 TIMOTHY',
            '1 pe': '1 PETER', '2 pe': '2 PETER', '1 jo': '1 JOHN', '2 jo': '2 JOHN',
            '3 jo': '3 JOHN'
        }
        
        # Try abbreviation mapping first
        book_lower = book_name.lower()
        if book_lower in abbreviations:
            expanded_name = abbreviations[book_lower]
            book_name = expanded_name
        
        # Try to match book name with parsed books
        for book_code in self.bible_parser.get_book_names():
            book_data = self.bible_parser.parse_file(book_code)
            if book_data['book_name'].lower() == book_name.lower():
                return book_code
        
        # Try common abbreviations (case insensitive)
        for book_code in self.bible_parser.get_book_names():
            if book_code.lower() == book_lower:
                return book_code
        
        return None
    
    def get_verse_content(self, reference: str) -> Optional[str]:
        """Get verse content from reference string."""
        parsed = self.parse_verse_reference(reference)
        if not parsed:
            return None
        
        book_code = self.find_book_code_by_name(parsed['book_name'])
        if not book_code:
            return None
        
        if parsed['verse_end']:
            content = self.bible_parser.get_verse_range_by_reference(
                book_code,
                parsed['chapter'],
                parsed['verse_start'],
                parsed['verse_end']
            )
        else:
            content = self.bible_parser.get_verse_by_reference(
                book_code,
                parsed['chapter'],
                parsed['verse_start']
            )
        
        return content
    
    def generate_devotion(self, client_id: str = None) -> Generator[Dict, None, tuple[Optional[Dict], Optional[str]]]:
        """Generate a devotion using a random verse.
        
        Yields:
            Dict: Progress updates with 'status' and 'message' keys
        
        Returns:
            tuple: (devotion_dict, error_message) - one will be None
        """
        # Check if generation is already in progress
        if self.generation_lock.locked():
            # Add this request to the queue
            with self.queue_lock:
                self.queue_counter += 1
                my_id = self.queue_counter
                self.queue.append({'id': my_id, 'client_id': client_id})
            
            # Wait for our turn in the queue (not just for lock to be available)
            while True:
                # Check if we were cancelled while waiting
                if client_id and self.cancel_requested.is_set():
                    with self.queue_lock:
                        # Remove ourselves from queue
                        self.queue = [item for item in self.queue if item['id'] != my_id]
                    self.cancel_requested.clear()
                    yield {'status': 'cancelled', 'message': 'Generation cancelled'}
                    return None, "Cancelled by client"
                
                with self.queue_lock:
                    # Check if it's our turn (we're at the front of the queue)
                    queue_entry = next((item for item in self.queue if item['id'] == my_id), None)
                    if queue_entry and self.queue.index(queue_entry) == 0:
                        # It's our turn, but we still need to wait for the lock to be released
                        if not self.generation_lock.locked():
                            # Remove ourselves from queue and break to acquire lock
                            self.queue.pop(0)
                            break
                    
                    # Calculate position based on current queue state
                    if queue_entry:
                        my_position = self.queue.index(queue_entry) + 1
                    else:
                        my_position = 1
                    # Calculate estimated wait time (15s per position)
                    estimated_wait = max(0, my_position * 15)
                
                yield {'status': 'waiting', 'message': f'Waiting in queue (position: {my_position}, estimated wait: {estimated_wait}s)'}
                time.sleep(1)
        
        # Acquire lock to start generation
        with self.generation_lock:
            self.current_client_id = client_id
            self.cancel_requested.clear()
            step1_messages = [
                "Generating random verses...",
                "Seeking inspiration from Scripture...",
                "Exploring the Bible for wisdom...",
                "Finding the perfect verse for you...",
                "Searching through God's Word..."
            ]
            
            step2_messages = [
                "Picking the best verse for your devotion...",
                "Discerning which verse speaks to you today...",
                "Selecting the most meaningful passage...",
                "Praying over the verse selection...",
                "Asking for guidance in choosing..."
            ]
            
            step3_messages = [
                "Writing a devotion based on the selected verse...",
                "Crafting meaningful reflections...",
                "Preparing a devotion for you...",
                "Generating spiritual insights...",
                "Writing today's devotion..."
            ]
            
            # Progress update thread using queue
            progress_queue = queue.Queue()
            stop_progress = threading.Event()
            current_step = ['step1']
            current_messages = [step1_messages]
            
            def update_progress():
                while not stop_progress.is_set():
                    time.sleep(10)
                    if not stop_progress.is_set():
                        step = current_step[0]
                        messages = current_messages[0]
                        progress_queue.put({'status': step, 'message': random.choice(messages)})
            
            progress_thread = threading.Thread(target=update_progress, daemon=True)
            progress_thread.start()
            
            def check_progress():
                try:
                    while True:
                        update = progress_queue.get_nowait()
                        yield update
                except queue.Empty:
                    pass
            
            try:
                yield {'status': 'step1', 'message': random.choice(step1_messages)}
                
                # Step 1: Generate AI verse with retry logic
                max_step1_retries = 5
                ai_verse_ref = None
                ai_verse_content = None
                
                for step1_attempt in range(max_step1_retries):
                    # Generate 1 random verse using AI
                    ai_verse_ref = self.ai_runner.generate_random_verse()
                    
                    # Check for progress updates during AI call
                    for update in check_progress():
                        yield update
                    
                    if not ai_verse_ref:
                        if step1_attempt < max_step1_retries - 1:
                            continue  # Retry
                        else:
                            stop_progress.set()
                            yield {'status': 'error', 'message': 'Failed to generate AI verse'}
                            return None, "Failed to generate AI verse after retries"
                    
                    # Get AI verse content
                    ai_verse_content = self.get_verse_content(ai_verse_ref)
                    for update in check_progress():
                        yield update
                    
                    if ai_verse_content:
                        break  # Success, exit retry loop
                    # If verse doesn't exist, retry with new AI verse
                
                if not ai_verse_ref or not ai_verse_content:
                    stop_progress.set()
                    yield {'status': 'error', 'message': 'Failed to find valid verse after retries'}
                    return None, "Failed to find valid verse after retries"
                
                # Generate 9 random verses
                random_verses = []
                for i in range(9):
                    ref, content, book_code = self.bible_parser.get_random_verse()
                    random_verses.append({
                        'reference': ref,
                        'content': content
                    })
                    for update in check_progress():
                        yield update
                
                # Build list of 10 verses
                all_verses = [
                    {
                        'reference': ai_verse_ref,
                        'content': ai_verse_content
                    }
                ] + random_verses
                
                current_step[0] = 'step2'
                current_messages[0] = step2_messages
                yield {'status': 'step2', 'message': random.choice(step2_messages)}
                
                # Step 2: Select best verse with retry logic
                max_step2_retries = 3
                selected_ref = None
                selected_content = None
                
                for step2_attempt in range(max_step2_retries):
                    # Send to AI for selection
                    selected_ref = self.ai_runner.select_best_verse(all_verses)
                    
                    # Check for progress updates during AI call
                    for update in check_progress():
                        yield update
                    
                    if not selected_ref:
                        if step2_attempt < max_step2_retries - 1:
                            continue  # Retry
                        else:
                            stop_progress.set()
                            yield {'status': 'error', 'message': 'AI failed to select a verse'}
                            return None, "AI failed to select a verse after retries"
                    
                    # Get the selected verse content
                    selected_content = self.get_verse_content(selected_ref)
                    for update in check_progress():
                        yield update
                    
                    if selected_content:
                        break  # Success, exit retry loop
                    # If selected verse doesn't exist, retry selection
                
                if not selected_ref or not selected_content:
                    stop_progress.set()
                    yield {'status': 'error', 'message': 'Failed to find valid selected verse after retries'}
                    return None, "Failed to find valid selected verse after retries"
                
                current_step[0] = 'step3'
                # Yield initial step3a message to transition from step2
                yield {'status': 'step3', 'message': random.choice(step3_messages)}
                
                # Progress callback for devotion generation
                def devotion_progress(step, message):
                    if step == 'step3':
                        progress_queue.put({'status': step, 'message': random.choice(step3_messages)})
                
                # Use streaming generation
                devotion_buffer = ""
                prayer_buffer = ""
                sent_verse_info = False
                
                try:
                    for section_type, chunk, full_content in self.ai_runner.generate_devotion_content_stream(selected_ref, selected_content, progress_callback=devotion_progress):
                        # Check if we were cancelled during generation
                        if client_id and self.cancel_requested.is_set():
                            self.cancel_requested.clear()
                            stop_progress.set()
                            yield {'status': 'cancelled', 'message': 'Generation cancelled'}
                            return None, "Cancelled by client"
                        
                        # Check for progress updates
                        for update in check_progress():
                            yield update
                        
                        # Send verse info with first devotion chunk
                        if section_type == 'devotion' and not sent_verse_info:
                            sent_verse_info = True
                            yield {
                                'status': 'streaming_devotion',
                                'reference': selected_ref,
                                'content': selected_content,
                                'devotion_chunk': chunk,
                                'devotion_full': full_content
                            }
                        elif section_type == 'devotion':
                            devotion_buffer = full_content
                            yield {
                                'status': 'streaming_devotion',
                                'devotion_chunk': chunk,
                                'devotion_full': full_content
                            }
                        elif section_type == 'prayer':
                            prayer_buffer = full_content
                            yield {
                                'status': 'streaming_prayer',
                                'prayer_chunk': chunk,
                                'prayer_full': full_content
                            }
                    
                    # Final check for progress updates
                    for update in check_progress():
                        yield update
                    
                    if not devotion_buffer or not prayer_buffer:
                        stop_progress.set()
                        yield {'status': 'error', 'message': 'Failed to generate devotion and prayer'}
                        return None, "Failed to generate devotion and prayer"
                    
                    stop_progress.set()
                    yield {'status': 'complete', 'message': 'Devotion generated!'}
                    
                    yield {
                        'status': 'success',
                        'reference': selected_ref,
                        'content': selected_content,
                        'devotion': devotion_buffer,
                        'prayer': prayer_buffer
                    }
                    
                    return None, None
                except Exception as e:
                    stop_progress.set()
                    yield {'status': 'error', 'message': f'Error during generation: {str(e)}'}
                    return None, f"Error during generation: {str(e)}"
                
                return None, None
            finally:
                stop_progress.set()
                self.current_client_id = None
    
    def remove_client_from_queue(self, client_id: str):
        """Remove a client from the queue if they disconnect."""
        with self.queue_lock:
            self.queue = [item for item in self.queue if item['client_id'] != client_id]
    
    def cancel_generation(self, client_id: str):
        """Cancel generation for a specific client if they disconnect."""
        if self.current_client_id == client_id:
            self.cancel_requested.set()
    
    def print_devotion(self, devotion: Dict):
        """Print the devotion to terminal."""
        print("\n" + "="*50)
        print("SELECTED DEVOTION VERSE")
        print("="*50)
        print(f"\n{devotion['reference']}\n")
        print(devotion['content'])
        print("\n" + "="*50)


if __name__ == "__main__":
    generator = DevotionGenerator()
    devotion, error = generator.generate_devotion()
    
    if devotion:
        generator.print_devotion(devotion)
    else:
        print(f"Failed to generate devotion: {error}")
