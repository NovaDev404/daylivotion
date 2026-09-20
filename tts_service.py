import os
import tempfile
import soundfile as sf
from kokoro import KPipeline
import torch
import threading
import queue
import re
import random
import numpy as np


class TTSService:
    def __init__(self):
        self.pipeline = None
        self.voice = 'af_heart'
        self.lang_code = 'a'

        # GPU / CPU device
        self.device = self._get_device()

        # Number of consecutive sentences with the same voice
        # to process in one Kokoro call.
        #
        # Higher values generally give the CPU more work at once,
        # but increase the amount of text processed per inference.
        self.batch_size = 4

        # Prevent extremely large batches from becoming unwieldy.
        self.max_batch_chars = 1200

        self.audio_queue = queue.Queue()

        # TTS queue system for parallel processing
        self.tts_queue = []
        self.tts_queue_counter = 0
        self.tts_queue_lock = threading.Lock()
        self.tts_generation_lock = threading.Lock()
        self.current_tts_client_id = None
        self.tts_cancel_requested = threading.Event()

        self.music_path = '/media/novadev/storage/daylivotion/music_24k.wav'
        self.cached_music = None
        self.cached_sr = None

        self._initialize_pipeline()
        self._load_music_cache()

    def _get_device(self):
        """Detect and use ROCm/CUDA GPU, MPS, or CPU."""

        # ROCm / CUDA
        if torch.cuda.is_available():
            try:
                torch.cuda.current_device()

                try:
                    gpu_name = torch.cuda.get_device_name(0)
                    print(f"GPU detected: {gpu_name}")
                except Exception:
                    pass

                return "cuda"

            except Exception as e:
                print(f"GPU detected but could not be initialized: {e}")

        # Apple Silicon MPS
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            print("Apple Silicon MPS detected")
            return "mps"

        # CPU fallback
        print("No supported GPU detected, using CPU")
        return "cpu"

    def _initialize_pipeline(self):
        """Initialize the Kokoro TTS pipeline with GPU acceleration."""

        try:
            print(f"Initializing Kokoro TTS on {self.device}...")

            self.pipeline = KPipeline(
                lang_code=self.lang_code,
                device=self.device
            )

            print(
                f"Kokoro TTS pipeline initialized successfully "
                f"on {self.device}"
            )

        except Exception as e:
            print(f"Failed to initialize Kokoro TTS pipeline: {e}")

            # Fallback to CPU if GPU fails
            if self.device != 'cpu':
                print("Falling back to CPU")

                self.device = 'cpu'

                try:
                    self.pipeline = KPipeline(
                        lang_code=self.lang_code,
                        device='cpu'
                    )

                    print(
                        "Kokoro TTS pipeline initialized successfully "
                        "on CPU (fallback)"
                    )

                except Exception as e2:
                    print(
                        f"Failed to initialize Kokoro TTS pipeline "
                        f"on CPU: {e2}"
                    )
                    self.pipeline = None

            else:
                self.pipeline = None

    def _load_music_cache(self):
        """Load and cache the music file for faster processing."""

        try:
            if os.path.exists(self.music_path):
                print("Loading music file into cache...")

                self.cached_music, self.cached_sr = sf.read(
                    self.music_path
                )

                print(
                    f"Music file cached successfully "
                    f"(sample rate: {self.cached_sr} Hz)"
                )

            else:
                print(
                    f"Music file not found at {self.music_path}"
                )

        except Exception as e:
            print(f"Error loading music cache: {e}")

            self.cached_music = None
            self.cached_sr = None

    def _add_background_music(
        self,
        speech_audio: np.ndarray,
        sample_rate: int
    ) -> np.ndarray:
        """
        Add background music to speech audio with fade in/out.

        Background music is expected to already be at 24000 Hz.
        """

        try:
            # Use cached music if available.
            if self.cached_music is not None:
                music_audio = self.cached_music

            elif os.path.exists(self.music_path):
                music_audio, _ = sf.read(self.music_path)

            else:
                return speech_audio

            # Convert stereo music to mono.
            if len(music_audio.shape) > 1:
                music_audio = np.mean(music_audio, axis=1)

            # Make sure we have a copy so fading never modifies
            # the cached music array itself.
            music_audio = np.asarray(
                music_audio,
                dtype=np.float32
            )

            speech_audio = np.asarray(
                speech_audio,
                dtype=np.float32
            )

            speech_duration = len(speech_audio)
            music_duration = len(music_audio)

            if speech_duration == 0 or music_duration == 0:
                return speech_audio

            # Get a random music segment matching speech duration.
            if music_duration > speech_duration:
                max_start = music_duration - speech_duration

                if max_start > 0:
                    start_sample = random.randint(
                        0,
                        max_start
                    )

                    music_segment = music_audio[
                        start_sample:start_sample + speech_duration
                    ].copy()

                else:
                    music_segment = music_audio[
                        :speech_duration
                    ].copy()

            else:
                # Loop music if shorter than speech.
                repeats = int(
                    np.ceil(
                        speech_duration / music_duration
                    )
                )

                music_segment = np.tile(
                    music_audio,
                    repeats
                )[:speech_duration].copy()

            # Apply fade in/out.
            fade_samples = int(2 * sample_rate)

            fade_length = min(
                fade_samples,
                len(music_segment)
            )

            if fade_length > 0:
                fade_in = np.linspace(
                    0,
                    1,
                    fade_length,
                    dtype=np.float32
                )

                fade_out = np.linspace(
                    1,
                    0,
                    fade_length,
                    dtype=np.float32
                )

                music_segment[:fade_length] *= fade_in
                music_segment[-fade_length:] *= fade_out

            # Mix music at low volume.
            music_volume = 0.15

            mixed_audio = (
                speech_audio +
                music_segment * music_volume
            )

            # Normalize to prevent clipping.
            max_val = np.max(
                np.abs(mixed_audio)
            )

            if max_val > 0.95:
                mixed_audio = (
                    mixed_audio / max_val * 0.95
                )

            return mixed_audio

        except Exception as e:
            print(
                f"Error adding background music: {e}"
            )
            return speech_audio

    def _format_verse_reference(
        self,
        reference: str
    ) -> str:
        """
        Format verse reference for natural speech.

        Example:
            1 John 3:15
        becomes:
            1 John, chapter 3, verse 15
        """

        match = re.match(
            r'^([A-Za-z0-9][A-Za-z0-9\s]*?)\s+(\d+):(\d+)$',
            reference
        )

        if match:
            book = match.group(1)
            chapter = match.group(2)
            verse = match.group(3)

            return (
                f"{book}, chapter {chapter}, "
                f"verse {verse}"
            )

        return reference

    def _format_verse_references_in_text(
        self,
        text: str
    ) -> str:
        """
        Find and format all verse references within text for natural speech.

        Handles patterns like:
            - Book Chapter:Verse (e.g., "John 3:16")
            - Book Chapter:Verse-Verse (e.g., "John 3:16-17")
            - Book Chapter:Verse-Chapter:Verse (e.g., "John 3:16-4:1")

        Example:
            "As we read in John 3:16 and Romans 8:28..."
        becomes:
            "As we read in John, chapter 3, verse 16 and Romans, chapter 8, verse 28..."
        """

        def replace_reference(match):
            book = match.group(1)
            chapter = match.group(2)
            verse_start = match.group(3)
            verse_end = match.group(4) if match.group(4) else None
            end_chapter = match.group(5) if match.group(5) else None
            end_verse = match.group(6) if match.group(6) else None

            if end_chapter and end_verse:
                # Cross-chapter range: Book Chapter:Verse-Chapter:Verse
                return (
                    f"{book}, chapter {chapter}, verse {verse_start} "
                    f"through chapter {end_chapter}, verse {end_verse}"
                )
            elif verse_end:
                # Same chapter range: Book Chapter:Verse-Verse
                return (
                    f"{book}, chapter {chapter}, verses {verse_start} "
                    f"through {verse_end}"
                )
            else:
                # Single verse: Book Chapter:Verse
                return f"{book}, chapter {chapter}, verse {verse_start}"

        # Pattern to match verse references with optional ranges
        # Matches: Book Chapter:Verse or Book Chapter:Verse-Verse or Book Chapter:Verse-Chapter:Verse
        # Book name can start with letters or numbers (e.g., "John" or "1 John")
        pattern = r'([A-Za-z0-9][A-Za-z0-9\s]*?)\s+(\d+):(\d+)(?:-(\d+)(?::(\d+))?)?'

        return re.sub(pattern, replace_reference, text, flags=re.IGNORECASE)

    def generate_audio_script(
        self,
        reference: str,
        content: str,
        devotion: str,
        prayer: str
    ) -> list:
        """
        Generate a natural-sounding script for an audio devotion
        with voice assignments.

        Returns:
            List of (text, voice) tuples.
        """

        script_parts = []

        # ---------------------------------------------------------
        # Opening - female voice
        # ---------------------------------------------------------

        opening_options = [
            (
                "Welcome to today's devotion. Take a moment to "
                "slow down, settle your heart, and focus on God's Word..."
            ),
            (
                "Good to have you here for today's devotion. "
                "Let's pause together and turn our hearts to Scripture..."
            ),
            (
                "Welcome. Let's take a deep breath, quiet our minds, "
                "and open our hearts to God's Word today..."
            ),
            (
                "Thank you for joining today's devotion. Let's set aside "
                "the busyness and focus on what God has to say to us..."
            ),
            (
                "Welcome to this time of reflection. Let's still our "
                "hearts and prepare to receive from God's Word..."
            )
        ]

        script_parts.append(
            (
                random.choice(opening_options),
                'af_heart'
            )
        )

        # ---------------------------------------------------------
        # Scripture - male voice
        # ---------------------------------------------------------

        formatted_reference = self._format_verse_reference(
            reference
        )

        scripture_options = [
            (
                f"Today's Scripture comes from "
                f"{formatted_reference}. Let's listen to what "
                f"God's Word says... {content}.."
            ),
            (
                f"Our passage for today is "
                f"{formatted_reference}. Hear now the Word of "
                f"God... {content}.."
            ),
            (
                f"Let's turn our attention to "
                f"{formatted_reference}. This is what the "
                f"Scripture says... {content}.."
            ),
            (
                f"Today we're reading from "
                f"{formatted_reference}. Listen to God's Word... "
                f"{content}.."
            ),
            (
                f"Our verse today is "
                f"{formatted_reference}. Here's what the Bible "
                f"tells us... {content}.."
            )
        ]

        script_parts.append(
            (
                random.choice(scripture_options),
                'am_michael'
            )
        )

        # ---------------------------------------------------------
        # Devotion - female voice
        # ---------------------------------------------------------

        formatted_devotion = self._format_verse_references_in_text(
            devotion
        )

        script_parts.append(
            (
                formatted_devotion,
                'af_heart'
            )
        )

        # ---------------------------------------------------------
        # Transition - male voice
        # ---------------------------------------------------------

        transition_options = [
            (
                "As we reflect on what we've heard, let's bring "
                "these thoughts before God. Let's pray..."
            ),
            (
                "Now let's take these truths to God in prayer. "
                "Let's pray together..."
            ),
            (
                "Let's respond to what we've heard by bringing "
                "our hearts to God in prayer..."
            ),
            (
                "As we meditate on this passage, let's turn to "
                "God in prayer. Let's pray..."
            ),
            (
                "Let's now bring our reflections before the Lord. "
                "Let us pray..."
            )
        ]

        script_parts.append(
            (
                random.choice(transition_options),
                'am_michael'
            )
        )

        # ---------------------------------------------------------
        # Prayer - male voice
        # ---------------------------------------------------------

        formatted_prayer = self._format_verse_references_in_text(
            prayer
        )

        if formatted_prayer.endswith('.'):
            prayer_with_ending = formatted_prayer + '..'
        else:
            prayer_with_ending = formatted_prayer + '...'

        script_parts.append(
            (
                prayer_with_ending,
                'am_michael'
            )
        )

        # ---------------------------------------------------------
        # Closing - female voice
        # ---------------------------------------------------------

        closing_options = [
            (
                "As you go through the rest of your day, carry this "
                "message with you and remember the truth we've "
                "reflected on today. Thank you for spending this "
                "time in God's Word. May God guide you, strengthen "
                "you, and bless you today..."
            ),
            (
                "Take this word with you as you continue your day. "
                "Hold onto these truths and let them guide you. "
                "Thanks for joining me in Scripture. May the Lord's "
                "presence go with you..."
            ),
            (
                "As you step back into your day, remember what we've "
                "learned here. Let God's Word be a light for your "
                "path. Thank you for this time together. May God's "
                "grace be with you..."
            ),
            (
                "Carry these truths in your heart as you go. Let the "
                "Scripture we've read shape your day today. Thank you "
                "for devoting this time to God's Word. May the Lord "
                "bless and keep you..."
            ),
            (
                "Let this message stay with you throughout your day. "
                "Remember God's faithfulness and the truth of His "
                "Word. Thanks for spending this time in devotion. "
                "May God's peace fill your heart..."
            )
        ]

        script_parts.append(
            (
                random.choice(closing_options),
                'af_heart'
            )
        )

        return script_parts

    def _split_script_into_sentences(
        self,
        script_sections: list
    ) -> list:
        """
        Split script sections into sentences while preserving
        the assigned voice.

        Returns:
            List of (sentence, voice) tuples.
        """

        all_sentences = []

        for text, voice in script_sections:

            # Convert escape sequences to natural speech pauses
            text = text.replace('\\n\\n', '...').replace('\\n', '...')
            # Normalize actual newlines before splitting.
            text = text.replace('\n', ' ').strip()
            # Remove forward slashes and asterisks (but preserve backslashes for escape sequences)
            text = text.replace('/', '').replace('*', '')

            sentences = re.split(
                r'(?<=[.!?])\s+',
                text
            )

            sentences = [
                sentence.strip()
                for sentence in sentences
                if sentence.strip()
            ]

            for sentence in sentences:
                all_sentences.append(
                    (
                        sentence,
                        voice
                    )
                )

        return all_sentences

    def _build_next_batch(
        self,
        all_sentences: list,
        start_index: int
    ):
        """
        Build one batch of consecutive sentences that all use
        the same voice.

        Returns:
            (batch_text, voice, sentence_count, next_index)
        """

        if start_index >= len(all_sentences):
            return (
                "",
                None,
                0,
                start_index
            )

        voice = all_sentences[start_index][1]

        batch_sentences = []
        char_count = 0

        index = start_index

        while index < len(all_sentences):

            sentence, sentence_voice = all_sentences[index]

            # Stop when the voice changes.
            if sentence_voice != voice:
                break

            cleaned_sentence = (
                sentence
                .replace('\n', ' ')
                .strip()
                .replace('/', '')
                .replace('\\', '')
                .replace('*', '')
            )

            if not cleaned_sentence:
                index += 1
                continue

            # Respect maximum batch size.
            if len(batch_sentences) >= self.batch_size:
                break

            # Respect maximum text length, unless the batch is empty.
            additional_length = len(cleaned_sentence)

            if batch_sentences:
                additional_length += 1  # space

            if (
                char_count + additional_length
                > self.max_batch_chars
            ):
                break

            batch_sentences.append(
                cleaned_sentence
            )

            char_count += additional_length
            index += 1

        # Safety fallback for a sentence longer than max_batch_chars.
        if not batch_sentences:
            cleaned_sentence = (
                all_sentences[start_index][0]
                .replace('\n', ' ')
                .strip()
            )

            batch_sentences = [
                cleaned_sentence
            ]

            index = start_index + 1

        batch_text = " ".join(
            batch_sentences
        )

        return (
            batch_text,
            voice,
            len(batch_sentences),
            index
        )

    def _emit_progress(
        self,
        socketio,
        sid,
        current,
        total
    ):
        """Emit audio generation progress."""

        if not socketio or not sid:
            return

        if total <= 0:
            progress = 100
        else:
            progress = (
                current / total
            ) * 100

        try:
            socketio.emit(
                'audio_progress',
                {
                    'status': 'generating',
                    'progress': progress,
                    'current': current,
                    'total': total
                },
                to=sid
            )

            # Force gevent to yield so the update can be sent.
            socketio.sleep(0)

        except Exception as e:
            print(
                f"Error emitting progress: {e}"
            )

    def generate_audio(
        self,
        reference: str,
        content: str,
        devotion: str,
        prayer: str,
        socketio=None,
        sid=None
    ) -> str:
        """
        Generate audio for the devotion using the queue system.

        Consecutive sentences using the same voice are batched
        together to reduce repeated Kokoro/GPU inference overhead.
        """

        if not self.pipeline:
            raise RuntimeError(
                "TTS pipeline not initialized"
            )

        # Add to TTS queue
        with self.tts_queue_lock:
            self.tts_queue_counter += 1
            my_id = self.tts_queue_counter
            self.tts_queue.append({
                'id': my_id,
                'client_id': sid,
                'reference': reference,
                'content': content,
                'devotion': devotion,
                'prayer': prayer,
                'socketio': socketio
            })

        # Wait for our turn in the queue
        while True:
            # Check if we were cancelled while waiting
            if sid and self.tts_cancel_requested.is_set():
                with self.tts_queue_lock:
                    self.tts_queue = [item for item in self.tts_queue if item['id'] != my_id]
                self.tts_cancel_requested.clear()
                raise RuntimeError("TTS generation cancelled")

            with self.tts_queue_lock:
                # Check if it's our turn
                queue_entry = next((item for item in self.tts_queue if item['id'] == my_id), None)
                if queue_entry and self.tts_queue.index(queue_entry) == 0:
                    # It's our turn, but we still need to wait for the lock
                    if not self.tts_generation_lock.locked():
                        # Remove ourselves from queue and break to acquire lock
                        self.tts_queue.pop(0)
                        break

                # Calculate position for progress update
                if queue_entry:
                    my_position = self.tts_queue.index(queue_entry) + 1
                else:
                    my_position = 1

                # Send queue position update
                if socketio and sid:
                    try:
                        socketio.emit(
                            'audio_queue_progress',
                            {
                                'status': 'queue',
                                'position': my_position,
                                'message': f'Waiting in audio queue (position: {my_position})'
                            },
                            to=sid
                        )
                        socketio.sleep(0)
                    except Exception as e:
                        print(f"Error emitting queue progress: {e}")

            import time
            time.sleep(0.5)

        # Acquire lock to start generation
        with self.tts_generation_lock:
            self.current_tts_client_id = sid
            self.tts_cancel_requested.clear()

            try:
                result = self._generate_audio_internal(
                    reference, content, devotion, prayer, socketio, sid
                )
                return result
            finally:
                self.current_tts_client_id = None

    def _generate_audio_internal(
        self,
        reference: str,
        content: str,
        devotion: str,
        prayer: str,
        socketio=None,
        sid=None
    ) -> str:
        """
        Internal method to generate audio (assumes lock is already held).
        """
        if not self.pipeline:
            raise RuntimeError(
                "TTS pipeline not initialized"
            )

        # ---------------------------------------------------------
        # Generate script
        # ---------------------------------------------------------

        script_sections = self.generate_audio_script(
            reference,
            content,
            devotion,
            prayer
        )

        # ---------------------------------------------------------
        # Split sections into sentences
        # ---------------------------------------------------------

        all_sentences = self._split_script_into_sentences(
            script_sections
        )

        total_sentences = len(
            all_sentences
        )

        if total_sentences == 0:
            raise RuntimeError(
                "No sentences found for audio generation"
            )

        # ---------------------------------------------------------
        # Output path
        # ---------------------------------------------------------

        temp_dir = tempfile.gettempdir()

        audio_path = os.path.join(
            temp_dir,
            f"devotion_{hash(str(all_sentences))}.wav"
        )

        def _generate_audio():
            try:
                audio_chunks = []
                temp_files = []

                processed_sentences = 0
                current_index = 0

                # -------------------------------------------------
                # Generate batches
                # -------------------------------------------------

                while current_index < total_sentences:

                    (
                        batch_text,
                        voice,
                        batch_sentence_count,
                        next_index
                    ) = self._build_next_batch(
                        all_sentences,
                        current_index
                    )

                    if not batch_text:
                        current_index = next_index
                        continue

                    print(
                        f"TTS batch: "
                        f"{batch_sentence_count} sentence(s), "
                        f"voice={voice}, "
                        f"chars={len(batch_text)}"
                    )

                    try:
                        # Kokoro uses the PyTorch "cuda" device
                        # for ROCm as well as NVIDIA CUDA.
                        with torch.inference_mode():

                            generator = self.pipeline(
                                batch_text,
                                voice=voice,
                                speed=1.0,
                                split_pattern=None
                            )

                            for gs, ps, audio in generator:

                                if audio is None:
                                    continue

                                # Move completed audio to CPU so GPU
                                # memory isn't unnecessarily retained
                                # between batches.
                                if torch.is_tensor(audio):
                                    audio = (
                                        audio
                                        .detach()
                                        .cpu()
                                    )
                                else:
                                    audio = torch.as_tensor(
                                        audio
                                    )

                                audio_chunks.append(
                                    audio
                                )

                    except Exception as e:
                        print(
                            f"Error generating TTS batch "
                            f"with voice {voice}: {e}"
                        )
                        raise

                    # -------------------------------------------------
                    # Update progress once per batch
                    # -------------------------------------------------

                    processed_sentences += (
                        batch_sentence_count
                    )

                    self._emit_progress(
                        socketio,
                        sid,
                        processed_sentences,
                        total_sentences
                    )

                    current_index = next_index

                # -------------------------------------------------
                # Combine generated audio
                # -------------------------------------------------

                if audio_chunks:

                    combined_audio = torch.cat(
                        audio_chunks,
                        dim=0
                    )

                    # Convert to numpy on CPU.
                    audio_array = (
                        combined_audio
                        .detach()
                        .cpu()
                        .numpy()
                    )

                    # -------------------------------------------------
                    # Background music
                    # -------------------------------------------------

                    audio_array = (
                        self._add_background_music(
                            audio_array,
                            24000
                        )
                    )

                    # -------------------------------------------------
                    # Save audio
                    # -------------------------------------------------

                    sf.write(
                        audio_path,
                        audio_array,
                        24000
                    )

                    # -------------------------------------------------
                    # Clean up temp files
                    # -------------------------------------------------

                    for temp_file in temp_files:
                        if os.path.exists(temp_file):
                            os.remove(temp_file)

                    # -------------------------------------------------
                    # Notify caller
                    # -------------------------------------------------

                    self.audio_queue.put(
                        (
                            'success',
                            audio_path
                        )
                    )

                else:
                    self.audio_queue.put(
                        (
                            'error',
                            'No audio generated'
                        )
                    )

            except Exception as e:

                print(
                    f"Error generating audio: {e}"
                )

                # Remove partial output.
                if os.path.exists(audio_path):
                    try:
                        os.remove(audio_path)
                    except Exception:
                        pass

                # Clean up temporary files.
                for temp_file in locals().get(
                    'temp_files',
                    []
                ):
                    if os.path.exists(temp_file):
                        try:
                            os.remove(temp_file)
                        except Exception:
                            pass

                self.audio_queue.put(
                    (
                        'error',
                        str(e)
                    )
                )

        # ---------------------------------------------------------
        # Run generation directly (we're already in a thread context from the queue)
        # ---------------------------------------------------------
        _generate_audio()

        # ---------------------------------------------------------
        # Wait for result from the internal queue
        # ---------------------------------------------------------

        try:
            status, result = (
                self.audio_queue.get(
                    timeout=120
                )
            )

            if status == 'success':
                return result

            raise RuntimeError(result)

        except queue.Empty:
            raise RuntimeError(
                "Audio generation timed out"
            )

    def is_available(self) -> bool:
        """Check if TTS service is available."""

        return self.pipeline is not None

    def remove_client_from_tts_queue(self, client_id: str):
        """Remove a client from the TTS queue if they disconnect."""
        with self.tts_queue_lock:
            self.tts_queue = [item for item in self.tts_queue if item['client_id'] != client_id]

    def cancel_tts_generation(self, client_id: str):
        """Cancel TTS generation for a specific client if they disconnect."""
        if self.current_tts_client_id == client_id:
            self.tts_cancel_requested.set()


# -------------------------------------------------------------
# Global TTS service instance
# -------------------------------------------------------------

tts_service = TTSService()