import requests
import re
import json
from typing import Optional, Tuple, Callable, Generator

class AIRunner:
    def __init__(self, server_url: str = "http://localhost:8981"):
        self.server_url = server_url
    
    def _completion(self, prompt: str, temperature: float = 0.7, top_p: float = 0.9, max_tokens: int = 512) -> Optional[str]:
        """Make a chat completion request to llama-server for proper instruction following."""
        try:
            json_data = {
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant that follows instructions precisely. Write naturally without repetitive phrases or excessive punctuation."},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "presence_penalty": 0.5,
                "frequency_penalty": 0.5,
                "stream": False
            }
            
            response = requests.post(
                f"{self.server_url}/v1/chat/completions",
                json=json_data,
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            print(data["choices"][0]["message"]["content"])
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"Error calling llama-server: {e}")
            return None
    
    def _completion_stream(self, prompt: str, temperature: float = 0.7, top_p: float = 0.9, max_tokens: int = 2048) -> Generator[str, None, None]:
        """Make a streaming chat completion request to llama-server."""
        try:
            json_data = {
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant that follows instructions precisely. Write naturally without repetitive phrases or excessive punctuation."},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "presence_penalty": 0.5,
                "frequency_penalty": 0.5,
                "stream": True
            }
            
            response = requests.post(
                f"{self.server_url}/v1/chat/completions",
                json=json_data,
                timeout=60,
                stream=True
            )
            response.raise_for_status()
            
            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    if line.startswith('data: '):
                        data_str = line[6:]
                        if data_str == '[DONE]':
                            break
                        try:
                            data = json.loads(data_str)
                            if 'choices' in data and len(data['choices']) > 0:
                                delta = data['choices'][0].get('delta', {})
                                content = delta.get('content', '')
                                if content:
                                    yield content
                        except Exception as e:
                            print(f"Error parsing stream data: {e}")
                            continue
        except Exception as e:
            print(f"Error calling llama-server stream: {e}")
            yield None
    
    def generate_random_verse(self, max_retries: int = 2) -> Optional[str]:
        """Generate a random bible verse reference."""
        prompt = 'Generate a random bible verse you can use for a devotion. Use the FULL book name (e.g., "1 John" not "1 JN", "Genesis" not "Gen"). Respond exactly like this including the square brackets: [BOOK CHAPTER:VERSE] OR [BOOK CHAPTER:VERSE-VERSE]'
        
        for attempt in range(max_retries + 1):
            output = self._completion(prompt, max_tokens=64)
            if output:
                verse_ref = self._parse_simple_output(output)
                if verse_ref:
                    return verse_ref
        
        return None
    
    def _parse_simple_output(self, output: str) -> Optional[str]:
        """Parse llama completion output which returns just the generated text."""
        # llama completion returns just the generated text, search for verse reference
        match = re.search(r'\[([A-Za-z0-9\s]+ \d+:\d+(?:-\d+)?)\]', output)
        if match:
            verse_ref = match.group(1)
            if self._validate_verse_reference(verse_ref):
                return verse_ref
        return None
    
    def _validate_verse_reference(self, ref: str) -> bool:
        """Validate that the verse reference matches expected format."""
        # Format: BOOK CHAPTER:VERSE or BOOK CHAPTER:VERSE-VERSE
        pattern = r'^[A-Za-z0-9\s]+ \d+:\d+(?:-\d+)?$'
        return bool(re.match(pattern, ref))
    
    def select_best_verse(self, verses: list, max_retries: int = 2) -> Optional[str]:
        """Send verses to AI and have it select the best one for devotion."""
        if len(verses) != 10:
            raise ValueError("Exactly 10 verses required")
        
        # Build prompt
        prompt_lines = [
            "Select ONE best verse from the list below for generating a devotion. Use the FULL book name (e.g., '1 John' not '1 JN', 'Genesis' not 'Gen'). Respond ONLY with the verse reference in brackets."
        ]
        
        for verse_info in verses:
            reference = verse_info['reference']
            content = verse_info['content']
            prompt_lines.append(f"{reference}")
            prompt_lines.append(content)
        
        prompt_lines.append("Respond with ONLY ONE verse reference in this format: [BOOK CHAPTER:VERSE]. Use the exact same book name format as shown in the list above.")
        
        prompt = '\n'.join(prompt_lines)
        
        for attempt in range(max_retries + 1):
            output = self._completion(prompt, max_tokens=64)
            if output:
                verse_ref = self._parse_simple_output(output)
                if verse_ref:
                    return verse_ref
        
        return None

    def generate_devotion_content(self, reference: str, content: str, max_retries: int = 2, progress_callback: Callable[[str, str], None] = None) -> Optional[Tuple[str, str]]:
        """Generate devotion and prayer for a given verse.
        
        Args:
            reference: Verse reference (e.g., "John 3:16")
            content: Verse content text
            progress_callback: Optional callback function to report progress
            
        Returns:
            Tuple of (devotion, prayer) or None on failure
        """
        prompt = f"""Generate 3 key points, a devotion, and a prayer for this Bible verse:
{reference}
{content}

Use "you" or "we", never "I", "my", or "me".
Use simple, natural language a teenager can understand.
Keep sentences short and conversational.
Stay focused on the main message of the verse and don't add unrelated ideas.
Explain what the verse means and how it can apply to everyday life.
Feel free to use real-world examples or comparisons that are relevant to today's world to help illustrate the meaning.
Avoid academic, complicated, dramatic, or AI-sounding language.
Do not start with "In today's world", "In this age", or "In a world where".
The devotion **MUST be 7-11 sentences** and contain a mixture of simple, complex, & compound sentences, and optionally a couple rhetorical questions.
The prayer **MUST be 3-4 short sentences** and directly relate to the verse, and end with a form of Amen (Such as 'In Jesus' name, Amen.').

Output ONLY:
[KEYPOINT]point 1[/KEYPOINT]
[KEYPOINT]point 2[/KEYPOINT]
[KEYPOINT]point 3[/KEYPOINT]
[DEVOTION]devotion text[/DEVOTION]
[PRAYER]prayer text[/PRAYER]

Nothing outside the tags."""
        
        for attempt in range(max_retries + 1):
            if progress_callback:
                progress_callback('step3', 'Generating devotion and prayer...')
            
            output = self._completion(prompt, temperature=0.7, top_p=0.9, max_tokens=2048)
            
            if output:
                devotion, prayer = self._parse_devotion_output(output)
                
                if devotion and prayer:
                    return devotion, prayer
        
        return None
    
    def generate_devotion_content_stream(self, reference: str, content: str, max_retries: int = 2, progress_callback: Callable[[str, str], None] = None) -> Generator[Tuple[str, Optional[str], Optional[str]], None, None]:
        """Generate devotion and prayer for a given verse with streaming.
        
        Args:
            reference: Verse reference (e.g., "John 3:16")
            content: Verse content text
            progress_callback: Optional callback function to report progress
            
        Yields:
            Tuple of (section_type, content_chunk, full_content) where:
            - section_type: 'devotion' or 'prayer'
            - content_chunk: New content chunk (can be empty)
            - full_content: Full content so far for that section
        """
        prompt = f"""Generate 3 key points, a devotion, and a prayer for this Bible verse:
{reference}
{content}

Use "you" or "we", never "I", "my", or "me".
Use simple, natural language a teenager can understand.
Keep sentences short and conversational.
Stay focused on the main message of the verse and don't add unrelated ideas.
Explain what the verse means and how it can apply to everyday life.
Feel free to use real-world examples or comparisons that are relevant to today's world to help illustrate the meaning.
Avoid academic, complicated, dramatic, or AI-sounding language.
Do not start with "In today's world", "In this age", or "In a world where".
The devotion **MUST be 5-8 sentences** and contain a mixture of simple, complex, & compound sentences, and optionally a couple rhetorical questions.
The prayer **MUST be 2-3 short sentences** and directly relate to the verse, and end with a form of Amen (Such as 'In Jesus' name, Amen.').
You *may* use basic markdown such as * or ** for italics and bold, but nothing else, and \\n for line breaks, but no other markdown.

Output ONLY:
[KEYPOINT]point 1[/KEYPOINT]
[KEYPOINT]point 2[/KEYPOINT]
[KEYPOINT]point 3[/KEYPOINT]
[DEVOTION]devotion[/DEVOTION]
[PRAYER]prayer[/PRAYER]

Nothing outside the tags."""
        
        for attempt in range(max_retries + 1):
            if progress_callback:
                progress_callback('step3', 'Generating devotion and prayer...')
            
            full_output = ""
            current_section = None
            devotion_buffer = ""
            prayer_buffer = ""
            
            for chunk in self._completion_stream(prompt, temperature=0.7, top_p=0.9, max_tokens=512):
                if chunk is None:
                    continue
                
                full_output += chunk
                
                # Check for section tags
                if '[DEVOTION]' in full_output and current_section != 'devotion':
                    current_section = 'devotion'
                    devotion_buffer = ""
                elif '[PRAYER]' in full_output and current_section != 'prayer':
                    current_section = 'prayer'
                    prayer_buffer = ""
                
                # Extract content based on current section
                if current_section == 'devotion':
                    # Extract devotion content so far (exclude closing tags)
                    devotion_match = re.search(r'\[DEVOTION\](.*?)(?:\[/?DEVOTION\]|\[PRAYER\]|$)', full_output, re.DOTALL)
                    if devotion_match:
                        new_devotion = devotion_match.group(1).strip()
                        # Remove any tag remnants
                        new_devotion = re.sub(r'\[/?DEVOTION\]?', '', new_devotion).strip()
                        if new_devotion != devotion_buffer:
                            chunk_diff = new_devotion[len(devotion_buffer):]
                            devotion_buffer = new_devotion
                            yield ('devotion', chunk_diff, devotion_buffer)
                elif current_section == 'prayer':
                    # Extract prayer content so far (exclude closing tags)
                    prayer_match = re.search(r'\[PRAYER\](.*?)(?:\[/?PRAYER\]|$)', full_output, re.DOTALL)
                    if prayer_match:
                        new_prayer = prayer_match.group(1).strip()
                        # Remove any tag remnants
                        new_prayer = re.sub(r'\[/?PRAYER\]?', '', new_prayer).strip()
                        if new_prayer != prayer_buffer:
                            chunk_diff = new_prayer[len(prayer_buffer):]
                            prayer_buffer = new_prayer
                            yield ('prayer', chunk_diff, prayer_buffer)
            
            # Final yield to ensure we get complete content
            if devotion_buffer:
                yield ('devotion', '', devotion_buffer)
            if prayer_buffer:
                yield ('prayer', '', prayer_buffer)
            
            if devotion_buffer and prayer_buffer:
                return
    
    def _parse_devotion_output(self, output: str) -> Tuple[Optional[str], Optional[str]]:
        """Parse devotion and prayer from AI output, ignoring key points."""
        # Filter out the prompt - only parse content after "assistant" marker
        assistant_match = re.search(r'assistant\s*\n(.*)', output, re.DOTALL)
        if assistant_match:
            output = assistant_match.group(1)
        
        # Extract devotion content - handle both correct and incorrect closing tags
        devotion_match = re.search(r'\[DEVOTION\](.*?)(?:\[/?DEVOTION\]|\[PRAYER\])', output, re.DOTALL)
        devotion = devotion_match.group(1).strip() if devotion_match else None
        
        # Extract prayer content
        prayer_match = re.search(r'\[PRAYER\](.*?)(?:\[/?PRAYER\]|\[end of text\]|$)', output, re.DOTALL)
        prayer = prayer_match.group(1).strip() if prayer_match else None
        
        return devotion, prayer
