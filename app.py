"""
Meeting Minutes Generator
=========================

An AI-powered web application that transforms meeting recordings or transcripts
into structured, actionable meeting minutes.

This application uses:
- OpenAI Whisper-small for audio transcription
- Facebook BART-large-cnn for text summarization
- Google Flan-T5-small for action item extraction

Deployed on Hugging Face Spaces (CPU tier).

Based on PRD Section 7: Technical Architecture
"""

import os
import sys
import signal
import tempfile
import logging
from datetime import datetime
from typing import Optional, Tuple, List, Dict, Any, Union, Callable
from dataclasses import dataclass
from enum import Enum

import gradio as gr
import torch

# =============================================================================
# Logging Configuration
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# =============================================================================
# Configuration
# =============================================================================

# Environment flags
MOCK_MODE = os.environ.get("MEETING_MINUTES_MOCK_MODE", "false").lower() == "true"
DEBUG_MODE = os.environ.get("MEETING_MINUTES_DEBUG", "false").lower() == "true"

# Model identifiers
WHISPER_MODEL_ID = "openai/whisper-small"
BART_MODEL_ID = "facebook/bart-large-cnn"
FLAN_T5_MODEL_ID = "google/flan-t5-small"

# Processing constraints for CPU tier
MAX_AUDIO_DURATION_SECONDS = 300  # 5 minutes recommended max
MAX_TRANSCRIPT_LENGTH = 10000  # Characters for processing
MAX_PROCESSING_TIMEOUT = 120  # Total processing timeout in seconds

# Model constraints
BART_MAX_INPUT_TOKENS = 1024  # BART max input length
SUMMARY_MAX_LENGTH = 150  # Maximum summary tokens
SUMMARY_MIN_LENGTH = 40  # Minimum summary tokens
CHUNK_OVERLAP_CHARS = 200  # Overlap between chunks for context

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.float32  # Use float32 for CPU compatibility


# =============================================================================
# Data Classes
# =============================================================================

class ProcessingStage(Enum):
    """Enum for tracking processing stages."""
    INITIALIZING = "initializing"
    INPUT_ROUTING = "input_routing"
    TRANSCRIPTION = "transcription"
    SUMMARIZATION = "summarization"
    EXTRACTION = "extraction"
    FORMATTING = "formatting"
    COMPLETE = "complete"
    ERROR = "error"


@dataclass
class ProcessingResult:
    """Structured result from meeting processing."""
    success: bool
    summary_text: str
    action_items_list: List[Dict[str, str]]
    full_minutes_markdown: str
    transcript: str
    error_message: Optional[str] = None
    processing_time: Optional[float] = None


@dataclass
class ModelLoadResult:
    """Result of model loading attempt."""
    success: bool
    model_name: str
    error_message: Optional[str] = None


# =============================================================================
# Global Model Containers
# =============================================================================

class ModelContainer:
    """
    Container for globally loaded models.
    Models are loaded once at startup and reused for all requests.
    """
    
    def __init__(self, mock_mode: bool = False):
        self.mock_mode = mock_mode
        self._transcriber = None
        self._summarizer = None
        self._extractor = None
        self._load_results: List[ModelLoadResult] = []
        
    @property
    def transcriber(self):
        """Get the transcriber pipeline."""
        return self._transcriber
    
    @property
    def summarizer(self):
        """Get the summarizer pipeline."""
        return self._summarizer
    
    @property
    def extractor(self):
        """Get the extractor pipeline."""
        return self._extractor
    
    @property
    def is_loaded(self) -> bool:
        """Check if all models are loaded."""
        if self.mock_mode:
            return True
        return all([
            self._transcriber is not None,
            self._summarizer is not None,
            self._extractor is not None,
        ])
    
    @property
    def load_results(self) -> List[ModelLoadResult]:
        """Get the list of model load results."""
        return self._load_results
    
    def load_all_models(self) -> bool:
        """
        Load all models at startup.
        
        Returns:
            True if all models loaded successfully, False otherwise
        """
        if self.mock_mode:
            logger.info("Mock mode enabled - skipping model loading")
            self._load_results = [
                ModelLoadResult(True, "mock-transcriber"),
                ModelLoadResult(True, "mock-summarizer"),
                ModelLoadResult(True, "mock-extractor"),
            ]
            return True
        
        logger.info("Loading models... This may take a few minutes on first run.")
        
        # Load transcriber
        transcriber_result = self._load_transcriber()
        self._load_results.append(transcriber_result)
        
        # Load summarizer
        summarizer_result = self._load_summarizer()
        self._load_results.append(summarizer_result)
        
        # Load extractor
        extractor_result = self._load_extractor()
        self._load_results.append(extractor_result)
        
        all_success = all(r.success for r in self._load_results)
        if all_success:
            logger.info("All models loaded successfully!")
        else:
            failed = [r.model_name for r in self._load_results if not r.success]
            logger.error(f"Failed to load models: {failed}")
        
        return all_success
    
    def _load_transcriber(self) -> ModelLoadResult:
        """Load the Whisper ASR model."""
        try:
            logger.info(f"Loading Whisper model: {WHISPER_MODEL_ID}")
            
            from transformers import pipeline
            
            self._transcriber = pipeline(
                "automatic-speech-recognition",
                model=WHISPER_MODEL_ID,
                torch_dtype=TORCH_DTYPE,
                device=-1,  # CPU only
            )
            
            logger.info("Whisper model loaded successfully")
            return ModelLoadResult(True, WHISPER_MODEL_ID)
            
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {str(e)}")
            return ModelLoadResult(False, WHISPER_MODEL_ID, str(e))
    
    def _load_summarizer(self) -> ModelLoadResult:
        """Load the BART summarization model."""
        try:
            logger.info(f"Loading BART model: {BART_MODEL_ID}")
            
            from transformers import pipeline
            
            self._summarizer = pipeline(
                "summarization",
                model=BART_MODEL_ID,
                torch_dtype=TORCH_DTYPE,
                device=-1,  # CPU only
            )
            
            logger.info("BART model loaded successfully")
            return ModelLoadResult(True, BART_MODEL_ID)
            
        except Exception as e:
            logger.error(f"Failed to load BART model: {str(e)}")
            return ModelLoadResult(False, BART_MODEL_ID, str(e))
    
    def _load_extractor(self) -> ModelLoadResult:
        """Load the Flan-T5 model for action item extraction."""
        try:
            logger.info(f"Loading Flan-T5 model: {FLAN_T5_MODEL_ID}")
            
            from transformers import pipeline
            
            self._extractor = pipeline(
                "text2text-generation",
                model=FLAN_T5_MODEL_ID,
                torch_dtype=TORCH_DTYPE,
                device=-1,  # CPU only
            )
            
            logger.info("Flan-T5 model loaded successfully")
            return ModelLoadResult(True, FLAN_T5_MODEL_ID)
            
        except Exception as e:
            logger.error(f"Failed to load Flan-T5 model: {str(e)}")
            return ModelLoadResult(False, FLAN_T5_MODEL_ID, str(e))


# Initialize global model container
models = ModelContainer(mock_mode=MOCK_MODE)


# =============================================================================
# Timeout Handler
# =============================================================================

class TimeoutError(Exception):
    """Custom timeout exception."""
    pass


def timeout_handler(signum, frame):
    """Signal handler for timeout."""
    raise TimeoutError("Processing timeout exceeded")


def with_timeout(seconds: int):
    """
    Decorator to add timeout protection to a function.
    Note: Only works on Unix-like systems.
    
    Args:
        seconds: Maximum execution time in seconds
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            if sys.platform == "win32":
                # Windows doesn't support SIGALRM, run without timeout
                return func(*args, **kwargs)
            
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(seconds)
            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
            return result
        return wrapper
    return decorator


# =============================================================================
# Mock Implementations for Testing
# =============================================================================

def mock_transcribe(audio_path: str) -> str:
    """Mock transcription for testing."""
    return """Team discussed Q3 goals and project timeline updates. 
Alex mentioned that the design team needs more time for the UI overhaul, approximately 2 more weeks.
Jordan agreed to update the project timeline document by next Monday.
Sarah will schedule a client demo for the following week.
The team also discussed the new feature requests from the marketing team.
Key decisions: Push the release date to end of Q3, prioritize mobile responsiveness.
Next meeting scheduled for Friday at 2 PM."""


def mock_summarize(text: str) -> str:
    """Mock summarization for testing."""
    return "The team discussed Q3 goals, agreed to push the release date, and assigned action items to Alex, Jordan, and Sarah for follow-up tasks."


def mock_extract_action_items(text: str) -> List[Dict[str, str]]:
    """Mock action item extraction for testing."""
    return [
        {"task": "Complete UI overhaul", "owner": "Alex", "deadline": "2 weeks"},
        {"task": "Update project timeline document", "owner": "Jordan", "deadline": "Next Monday"},
        {"task": "Schedule client demo", "owner": "Sarah", "deadline": "Next week"},
    ]


# =============================================================================
# Processing Functions
# =============================================================================

def transcribe_audio(
    audio_path: Union[str, bytes],
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> str:
    """
    Transcribe audio file to text using Whisper-small.
    
    Args:
        audio_path: Path to the audio file (WAV, MP3, M4A, WebM) or bytes
        progress_callback: Optional callback for progress updates
        
    Returns:
        Transcribed text string
        
    Raises:
        ValueError: If audio_path is None or empty
        RuntimeError: If transcription fails
    """
    if audio_path is None or audio_path == "":
        raise ValueError("No audio file provided")
    
    logger.info(f"Transcribing audio: {audio_path if isinstance(audio_path, str) else 'bytes'}")
    
    if progress_callback:
        progress_callback(0.1, "Loading audio file...")
    
    # Use mock if in mock mode
    if models.mock_mode:
        logger.info("Using mock transcription")
        return mock_transcribe(audio_path if isinstance(audio_path, str) else "mock")
    
    if models.transcriber is None:
        raise RuntimeError("Transcriber model not loaded")
    
    try:
        if progress_callback:
            progress_callback(0.3, "Running Whisper transcription...")
        
        # Transcribe with Whisper
        result = models.transcriber(
            audio_path,
            return_timestamps=False,
            generate_kwargs={
                "language": "english",
                "task": "transcribe",
            }
        )
        
        transcript = result["text"].strip()
        logger.info(f"Transcription complete. Length: {len(transcript)} characters")
        
        if progress_callback:
            progress_callback(1.0, "Transcription complete")
        
        return transcript
        
    except Exception as e:
        logger.error(f"Transcription failed: {str(e)}")
        raise RuntimeError(f"Transcription failed: {str(e)}")


def chunk_text_for_summarization(
    text: str,
    max_chars: int = 3000,
    overlap: int = CHUNK_OVERLAP_CHARS,
) -> List[str]:
    """
    Split long text into overlapping chunks for summarization.
    
    Args:
        text: Full text to chunk
        max_chars: Maximum characters per chunk
        overlap: Overlap between chunks
        
    Returns:
        List of text chunks
    """
    if len(text) <= max_chars:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + max_chars
        
        # Try to break at sentence boundary
        if end < len(text):
            # Look for sentence ending within last 200 chars of chunk
            last_period = text.rfind(".", start, end)
            last_question = text.rfind("?", start, end)
            last_exclaim = text.rfind("!", start, end)
            
            best_break = max(last_period, last_question, last_exclaim)
            
            if best_break > start + max_chars - 200:
                end = best_break + 1
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        # Move start with overlap
        start = end - overlap if end < len(text) else len(text)
    
    return chunks


def generate_summary(
    text: str,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> str:
    """
    Generate an executive summary using BART-large-cnn.
    Handles long transcripts by chunking and combining summaries.
    
    Args:
        text: Full meeting transcript text
        progress_callback: Optional callback for progress updates
        
    Returns:
        Summary text (40-150 tokens)
    """
    if not text or len(text.strip()) == 0:
        return "No transcript available to summarize."
    
    logger.info("Generating summary...")
    
    if progress_callback:
        progress_callback(0.1, "Preparing text for summarization...")
    
    # Use mock if in mock mode
    if models.mock_mode:
        logger.info("Using mock summarization")
        return mock_summarize(text)
    
    if models.summarizer is None:
        raise RuntimeError("Summarizer model not loaded")
    
    # Chunk text if necessary (BART max input is 1024 tokens ≈ 3000 chars)
    chunks = chunk_text_for_summarization(text, max_chars=3000)
    logger.info(f"Text split into {len(chunks)} chunk(s)")
    
    if progress_callback:
        progress_callback(0.2, f"Processing {len(chunks)} text chunk(s)...")
    
    summaries = []
    
    for i, chunk in enumerate(chunks):
        if progress_callback:
            progress = 0.2 + (0.7 * (i / len(chunks)))
            progress_callback(progress, f"Summarizing chunk {i+1}/{len(chunks)}...")
        
        try:
            # Generate summary for each chunk
            summary_result = models.summarizer(
                chunk,
                max_length=SUMMARY_MAX_LENGTH,
                min_length=SUMMARY_MIN_LENGTH,
                do_sample=False,
                truncation=True,
            )
            
            chunk_summary = summary_result[0]["summary_text"].strip()
            summaries.append(chunk_summary)
            logger.info(f"Chunk {i+1} summary: {len(chunk_summary)} chars")
            
        except Exception as e:
            logger.warning(f"Failed to summarize chunk {i+1}: {str(e)}")
            # Use first 200 chars as fallback
            summaries.append(chunk[:200] + "...")
    
    # Combine summaries if multiple chunks
    if len(summaries) == 1:
        final_summary = summaries[0]
    else:
        # Combine and re-summarize if multiple chunks
        combined = " ".join(summaries)
        if len(combined) > 3000:
            # Re-summarize the combined summaries
            if progress_callback:
                progress_callback(0.9, "Combining summaries...")
            
            try:
                final_result = models.summarizer(
                    combined,
                    max_length=SUMMARY_MAX_LENGTH,
                    min_length=SUMMARY_MIN_LENGTH,
                    do_sample=False,
                    truncation=True,
                )
                final_summary = final_result[0]["summary_text"].strip()
            except Exception as e:
                logger.warning(f"Failed to combine summaries: {str(e)}")
                final_summary = combined[:500]
        else:
            final_summary = combined
    
    if progress_callback:
        progress_callback(1.0, "Summarization complete")
    
    logger.info(f"Final summary: {len(final_summary)} chars")
    return final_summary


def extract_action_items(
    text: str,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> List[Dict[str, str]]:
    """
    Extract action items from transcript using Flan-T5-small with prompt engineering.
    
    Args:
        text: Full meeting transcript text
        progress_callback: Optional callback for progress updates
        
    Returns:
        List of dicts with 'task', 'owner', 'deadline' keys
    """
    if not text or len(text.strip()) == 0:
        return []
    
    logger.info("Extracting action items...")
    
    if progress_callback:
        progress_callback(0.1, "Preparing extraction prompt...")
    
    # Use mock if in mock mode
    if models.mock_mode:
        logger.info("Using mock extraction")
        return mock_extract_action_items(text)
    
    if models.extractor is None:
        raise RuntimeError("Extractor model not loaded")
    
    # Use first portion of text for extraction (limit input length)
    max_input_chars = 1000
    input_text = text[:max_input_chars] if len(text) > max_input_chars else text
    
    if progress_callback:
        progress_callback(0.3, "Running action item extraction...")
    
    # Construct extraction prompt with clear instructions
    extraction_prompt = f"""Extract action items from this meeting transcript. 
For each action item, identify the task description, responsible person, and deadline.
Format each action item as: Task: [description] | Owner: [name] | Deadline: [date]
Use 'TBD' if information is not mentioned in the transcript.

Transcript:
{input_text}

Action items:"""

    try:
        # Generate extraction
        result = models.extractor(
            extraction_prompt,
            max_length=256,
            num_beams=2,
            temperature=0.3,
            do_sample=True,
        )
        
        extraction_output = result[0]["generated_text"].strip()
        logger.info(f"Extraction output: {extraction_output}")
        
        if progress_callback:
            progress_callback(0.8, "Parsing action items...")
        
        # Parse the output
        action_items = parse_action_items_output(extraction_output, text)
        
        if progress_callback:
            progress_callback(1.0, "Extraction complete")
        
        return action_items
        
    except Exception as e:
        logger.error(f"Action item extraction failed: {str(e)}")
        # Return fallback items
        return [{
            "task": "Review meeting transcript for action items",
            "owner": "TBD",
            "deadline": "TBD"
        }]


def parse_action_items_output(
    extraction_output: str,
    original_text: str,
) -> List[Dict[str, str]]:
    """
    Parse the model output to extract structured action items.
    
    Args:
        extraction_output: Raw output from extraction model
        original_text: Original transcript for context
        
    Returns:
        List of dicts with 'task', 'owner', 'deadline' keys
    """
    action_items = []
    
    # Split by lines
    lines = extraction_output.strip().split("\n")
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Try different parsing patterns
        item = None
        
        # Pattern 1: Task: X | Owner: Y | Deadline: Z
        if "task:" in line.lower() and "|" in line:
            parts = line.split("|")
            task = ""
            owner = "TBD"
            deadline = "TBD"
            
            for part in parts:
                part = part.strip()
                if part.lower().startswith("task:"):
                    task = part[5:].strip()
                elif part.lower().startswith("owner:"):
                    owner = part[6:].strip()
                elif part.lower().startswith("deadline:"):
                    deadline = part[9:].strip()
            
            if task:
                item = {"task": task, "owner": owner, "deadline": deadline}
        
        # Pattern 2: Simple pipe-separated: Task | Owner | Deadline
        elif "|" in line and "task:" not in line.lower():
            parts = [p.strip() for p in line.split("|")]
            if len(parts) >= 1 and parts[0]:
                item = {
                    "task": parts[0],
                    "owner": parts[1] if len(parts) > 1 else "TBD",
                    "deadline": parts[2] if len(parts) > 2 else "TBD",
                }
        
        # Pattern 3: Numbered or bulleted items
        elif line and (line[0].isdigit() or line.startswith("-") or line.startswith("•")):
            # Clean up the line
            clean_line = line.lstrip("0123456789.-•) ").strip()
            if clean_line:
                item = {"task": clean_line, "owner": "TBD", "deadline": "TBD"}
        
        # Pattern 4: Plain text line (treat as task description)
        elif len(line) > 10:
            item = {"task": line, "owner": "TBD", "deadline": "TBD"}
        
        if item and item["task"]:
            # Validate task isn't just filler
            if item["task"].lower() not in ["tbd", "none", "n/a", "-"]:
                action_items.append(item)
    
    # Deduplicate by task
    seen_tasks = set()
    unique_items = []
    for item in action_items:
        task_lower = item["task"].lower()
        if task_lower not in seen_tasks:
            seen_tasks.add(task_lower)
            unique_items.append(item)
    
    # Limit to reasonable number
    return unique_items[:10]


def format_meeting_minutes(
    summary: str,
    action_items: List[Dict[str, str]],
    meeting_type: str,
    timestamp: str,
    transcript: str = "",
) -> str:
    """
    Format complete meeting minutes in Markdown.
    
    Args:
        summary: Executive summary text
        action_items: List of action item dicts
        meeting_type: Type of meeting
        timestamp: Generation timestamp
        transcript: Original transcript (optional, for reference)
        
    Returns:
        Formatted Markdown string
    """
    minutes = f"""# Meeting Minutes - {meeting_type}

**Generated:** {timestamp}  
**Meeting Type:** {meeting_type}

---

## Executive Summary

{summary}

---

## Action Items

| Task | Owner | Deadline |
|------|-------|----------|
"""
    
    for item in action_items:
        task = item.get("task", "TBD")
        owner = item.get("owner", "TBD")
        deadline = item.get("deadline", "TBD")
        minutes += f"| {task} | {owner} | {deadline} |\n"
    
    if not action_items:
        minutes += "| No action items identified | TBD | TBD |\n"
    
    minutes += """
---

## Key Discussion Points

*Review the executive summary above for the main topics discussed during the meeting.*

---

## Notes

*Generated by [Meeting Minutes Generator](https://github.com/insydr/MeetingMinutesGenerator)*
"""
    
    return minutes


# =============================================================================
# Main Processing Function
# =============================================================================

def process_meeting(
    audio: Optional[Union[str, bytes]],
    transcript_text: Optional[str],
    meeting_type: str,
    progress: gr.Progress = gr.Progress(),
) -> Tuple[str, List[List[str]], str, Optional[str]]:
    """
    Main processing function that orchestrates the full pipeline.
    
    Implements PRD Section 7 Technical Architecture:
    1. Input Routing - audio or text
    2. Transcription - Whisper-small (if audio)
    3. Summarization - BART-large-cnn with chunking
    4. Action Item Extraction - Flan-T5-small with prompt engineering
    5. Output Formatting - Markdown
    
    Args:
        audio: Path to uploaded audio file or bytes (or None)
        transcript_text: Pasted transcript text (or empty string)
        meeting_type: Selected meeting type
        progress: Gradio progress tracker
        
    Returns:
        Tuple of (summary, action_items_dataframe, formatted_minutes, markdown_file_path)
    """
    import time
    start_time = time.time()
    
    try:
        # =====================================================================
        # Stage 1: Input Routing
        # =====================================================================
        progress(0.05, desc="Validating input...")
        logger.info(f"Processing request - Meeting type: {meeting_type}")
        
        # Validate input
        has_audio = audio is not None and audio != ""
        has_transcript = transcript_text and transcript_text.strip()
        
        if not has_audio and not has_transcript:
            gr.Warning("Please provide either an audio file or a transcript.")
            return "No input provided.", [], "No minutes generated.", None
        
        # =====================================================================
        # Stage 2: Transcription (if audio provided)
        # =====================================================================
        progress(0.1, desc="Processing input...")
        
        full_transcript = ""
        
        if has_audio:
            logger.info("Processing audio input")
            gr.Info("Transcribing audio... This may take 30-60 seconds.")
            
            def transcription_progress(pct: float, msg: str):
                progress(0.1 + (pct * 0.3), desc=msg)
            
            full_transcript = transcribe_audio(audio, transcription_progress)
        else:
            logger.info("Processing text input")
            gr.Info("Processing text transcript...")
            full_transcript = transcript_text.strip()
        
        # Validate transcript
        if not full_transcript or len(full_transcript.strip()) < 10:
            gr.Warning("Transcript is too short. Please provide more content.")
            return "Transcript too short.", [], "No minutes generated.", None
        
        # Truncate if too long
        if len(full_transcript) > MAX_TRANSCRIPT_LENGTH:
            logger.warning(f"Transcript truncated from {len(full_transcript)} to {MAX_TRANSCRIPT_LENGTH} chars")
            full_transcript = full_transcript[:MAX_TRANSCRIPT_LENGTH]
        
        # =====================================================================
        # Stage 3: Summarization
        # =====================================================================
        progress(0.45, desc="Generating summary...")
        
        def summarization_progress(pct: float, msg: str):
            progress(0.45 + (pct * 0.2), desc=msg)
        
        summary = generate_summary(full_transcript, summarization_progress)
        
        # =====================================================================
        # Stage 4: Action Item Extraction
        # =====================================================================
        progress(0.7, desc="Extracting action items...")
        
        def extraction_progress(pct: float, msg: str):
            progress(0.7 + (pct * 0.15), desc=msg)
        
        action_items = extract_action_items(full_transcript, extraction_progress)
        
        # =====================================================================
        # Stage 5: Output Formatting
        # =====================================================================
        progress(0.9, desc="Formatting minutes...")
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        formatted_minutes = format_meeting_minutes(
            summary=summary,
            action_items=action_items,
            meeting_type=meeting_type,
            timestamp=timestamp,
            transcript=full_transcript,
        )
        
        # Create downloadable file
        file_name = f"meeting_minutes_{meeting_type.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        temp_file_path = os.path.join(tempfile.gettempdir(), file_name)
        
        with open(temp_file_path, "w", encoding="utf-8") as f:
            f.write(formatted_minutes)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        logger.info(f"Processing complete in {processing_time:.2f} seconds")
        
        progress(1.0, desc="Complete!")
        gr.Info(f"Meeting minutes generated successfully! ({processing_time:.1f}s)")
        
        # Convert action items to list format for Dataframe
        action_items_list = [
            [item.get("task", "TBD"), item.get("owner", "TBD"), item.get("deadline", "TBD")]
            for item in action_items
        ]
        
        return summary, action_items_list, formatted_minutes, temp_file_path
        
    except TimeoutError:
        error_msg = f"Processing timed out after {MAX_PROCESSING_TIMEOUT} seconds"
        logger.error(error_msg)
        gr.Error(error_msg)
        return error_msg, [], "Processing timed out.", None
        
    except Exception as e:
        error_msg = f"An error occurred: {str(e)}"
        logger.error(f"Error in process_meeting: {str(e)}", exc_info=True)
        gr.Error(error_msg)
        return error_msg, [], f"Processing failed: {str(e)}", None


# =============================================================================
# Gradio Interface - Based on PRD Section 8: UI/UX Specifications
# =============================================================================

# Example transcripts for quick testing
EXAMPLE_TRANSCRIPT_STANDUP = """Team discussed Q3 goals and project timeline updates. 
Alex mentioned that the design team needs more time for the UI overhaul, approximately 2 more weeks.
Jordan agreed to update the project timeline document by next Monday.
Sarah will schedule a client demo for the following week.
The team also discussed the new feature requests from the marketing team.
Key decisions: Push the release date to end of Q3, prioritize mobile responsiveness.
Next meeting scheduled for Friday at 2 PM."""

EXAMPLE_TRANSCRIPT_CLIENT = """Client call with Acme Corporation regarding the enterprise software implementation project.
Attendees: John (Client PM), Lisa (Client Tech Lead), Mike (Our Team), Sarah (Our PM)

Discussion points:
- Client expressed satisfaction with the current progress
- Requested additional customization for reporting module - John will follow up
- Lisa asked about API integration timeline - Mike confirmed it's on track for next sprint
- Budget review scheduled for end of month
- Concerns raised about data migration - Sarah to provide risk assessment by Wednesday

Action items:
- Mike to prepare API documentation for client review by Friday
- Sarah to send project status report every Monday
- John to confirm data migration requirements by next call
- Schedule next call for Thursday 3pm

Meeting concluded at 3:45pm with positive outlook on project progression."""

EXAMPLE_TRANSCRIPT_BRAINSTORM = """Brainstorming session: New product features for Q4

Participants: Team leads from Engineering, Design, Marketing, and Sales

Ideas discussed:
1. AI-powered recommendations engine - could increase user engagement by 40%
2. Mobile app redesign with dark mode support - high user demand
3. Integration with Slack and Microsoft Teams - enterprise clients requesting
4. Real-time collaboration features - competitors already have this
5. Voice command interface - innovative but high development cost

Voting results:
- AI recommendations: 8 votes (priority for Q4)
- Mobile redesign: 6 votes (start in Q4, complete in Q1)
- Slack/Teams integration: 5 votes (evaluate technical requirements)
- Real-time collaboration: 4 votes (backlog for now)
- Voice interface: 2 votes (research phase only)

Decisions:
- Engineering will create technical specs for AI recommendations by next week
- Design team to mock up mobile dark mode by Friday
- Product team to research integration APIs

Next brainstorming session scheduled for same time next month."""


# =============================================================================
# Custom CSS Styles - PRD Color Scheme
# =============================================================================

CUSTOM_CSS = """
/* Container and Layout */
.container {
    max-width: 1400px;
    margin: 0 auto;
}

/* Header Styling */
.header-section {
    text-align: center;
    padding: 1.5rem 0;
    border-bottom: 1px solid #e5e7eb;
    margin-bottom: 1.5rem;
}

.header-title {
    font-size: 2.25rem;
    font-weight: 700;
    color: #1f2937;
    margin-bottom: 0.5rem;
}

.header-subtitle {
    font-size: 1.1rem;
    color: #6b7280;
    font-style: italic;
}

/* Privacy Notice Banner - PRD NFR-4 */
.privacy-notice {
    background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
    border-left: 4px solid #f59e0b;
    padding: 1rem 1.25rem;
    margin: 1rem 0;
    border-radius: 8px;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
}

.privacy-notice-title {
    font-weight: 600;
    color: #92400e;
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin-bottom: 0.5rem;
}

.privacy-notice-content {
    color: #78350f;
    font-size: 0.95rem;
    line-height: 1.6;
}

/* Section Headers */
.section-header {
    font-size: 1.25rem;
    font-weight: 600;
    color: #374151;
    margin-bottom: 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid #3b82f6;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

/* Input Section Styling */
.input-section {
    background-color: #f9fafb;
    padding: 1.25rem;
    border-radius: 12px;
    border: 1px solid #e5e7eb;
}

/* Output Section Styling */
.output-section {
    background-color: #ffffff;
    padding: 1.25rem;
    border-radius: 12px;
    border: 1px solid #e5e7eb;
}

/* Audio Component Styling */
.audio-component {
    border: 2px dashed #d1d5db;
    border-radius: 8px;
    padding: 0.5rem;
    transition: border-color 0.2s ease;
}

.audio-component:hover {
    border-color: #3b82f6;
}

/* Duration Warning */
.duration-warning {
    background-color: #fef2f2;
    border: 1px solid #fecaca;
    color: #991b1b;
    padding: 0.75rem;
    border-radius: 6px;
    font-size: 0.875rem;
    margin-top: 0.5rem;
}

/* Button Styling */
.generate-button {
    width: 100%;
    margin-top: 1rem;
    font-size: 1.1rem;
    font-weight: 600;
    padding: 0.75rem 1.5rem;
    transition: all 0.2s ease;
}

.generate-button:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(59, 130, 246, 0.4);
}

/* Output Components */
.summary-output {
    background-color: #f0fdf4;
    border: 1px solid #bbf7d0;
    border-radius: 8px;
    padding: 1rem;
    min-height: 80px;
}

.action-items-table {
    border: 1px solid #e5e7eb;
    border-radius: 8px;
    overflow: hidden;
}

.minutes-output {
    font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
    font-size: 0.875rem;
    background-color: #1f2937;
    color: #e5e7eb;
    border-radius: 8px;
}

/* Tips Section */
.tips-section {
    background-color: #eff6ff;
    border: 1px solid #bfdbfe;
    border-radius: 8px;
    padding: 1rem;
    margin-top: 1rem;
}

.tips-title {
    font-weight: 600;
    color: #1e40af;
    margin-bottom: 0.5rem;
}

.tips-list {
    color: #1e3a8a;
    font-size: 0.875rem;
    padding-left: 1.25rem;
    margin: 0;
}

.tips-list li {
    margin-bottom: 0.25rem;
}

/* Examples Section */
.examples-section {
    margin-top: 1.5rem;
    padding: 1rem;
    background-color: #f9fafb;
    border-radius: 12px;
    border: 1px solid #e5e7eb;
}

/* Status Badges */
.status-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.375rem;
    padding: 0.375rem 0.75rem;
    border-radius: 9999px;
    font-size: 0.875rem;
    font-weight: 500;
}

.mock-badge {
    background-color: #fef3c7;
    color: #92400e;
    border: 1px solid #fcd34d;
}

.ready-badge {
    background-color: #d1fae5;
    color: #065f46;
    border: 1px solid #6ee7b7;
}

.error-badge {
    background-color: #fee2e2;
    color: #991b1b;
    border: 1px solid #fca5a5;
}

/* Loading State */
.loading-indicator {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0.5rem;
    padding: 1rem;
    color: #3b82f6;
}

/* Responsive Design */
@media (max-width: 768px) {
    .header-title {
        font-size: 1.75rem;
    }
    
    .header-subtitle {
        font-size: 1rem;
    }
    
    .input-section,
    .output-section {
        padding: 1rem;
    }
}

/* Accessibility - Focus States */
.gradio-container *:focus {
    outline: 2px solid #3b82f6;
    outline-offset: 2px;
}

/* Screen Reader Only Content */
.sr-only {
    position: absolute;
    width: 1px;
    height: 1px;
    padding: 0;
    margin: -1px;
    overflow: hidden;
    clip: rect(0, 0, 0, 0);
    white-space: nowrap;
    border: 0;
}
"""


# =============================================================================
# Helper Functions for UI
# =============================================================================

def check_audio_duration(audio_path: Optional[str]) -> str:
    """
    Check audio file duration and return warning if > 5 minutes.
    
    Args:
        audio_path: Path to uploaded audio file
        
    Returns:
        Warning message if duration exceeds limit, empty string otherwise
    """
    if not audio_path:
        return ""
    
    try:
        import torchaudio
        
        # Get audio duration
        waveform, sample_rate = torchaudio.load(audio_path)
        duration_seconds = waveform.shape[1] / sample_rate
        
        if duration_seconds > MAX_AUDIO_DURATION_SECONDS:
            minutes = int(duration_seconds // 60)
            seconds = int(duration_seconds % 60)
            return f"⚠️ **Warning:** Audio duration ({minutes}:{seconds:02d}) exceeds recommended 5-minute limit. Processing may timeout or be truncated."
        
        minutes = int(duration_seconds // 60)
        seconds = int(duration_seconds % 60)
        return f"✅ Audio duration: {minutes}:{seconds:02d}"
        
    except Exception as e:
        logger.warning(f"Could not check audio duration: {str(e)}")
        return ""


def get_status_indicator() -> str:
    """Get status indicator based on model loading state."""
    if MOCK_MODE:
        return "🧪 Mock Mode Active"
    elif models.is_loaded:
        return "✅ Models Loaded"
    else:
        return "⏳ Loading Models..."


# =============================================================================
# Build Gradio Interface
# =============================================================================

with gr.Blocks(
    theme=gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="slate",
        neutral_hue="slate",
        font=[
            gr.themes.GoogleFont("Inter"),
            "ui-sans-serif",
            "system-ui",
            "sans-serif",
        ],
    ),
    title="🎙️ Meeting Minutes Generator",
    css=CUSTOM_CSS,
    fill_height=True,
) as demo:
    
    # =========================================================================
    # Header Section
    # =========================================================================
    with gr.Row():
        with gr.Column():
            gr.HTML(
                """
                <div class="header-section">
                    <h1 class="header-title">🎙️ Meeting Minutes Generator</h1>
                    <p class="header-subtitle">
                        Upload a recording or paste a transcript → Get structured, actionable meeting minutes
                    </p>
                </div>
                """
            )
    
    # =========================================================================
    # Privacy Notice Banner (PRD NFR-4)
    # =========================================================================
    with gr.Row():
        with gr.Column():
            gr.HTML(
                """
                <div class="privacy-notice" role="alert" aria-label="Privacy Notice">
                    <div class="privacy-notice-title">
                        🔒 Privacy Notice
                    </div>
                    <div class="privacy-notice-content">
                        Audio is processed in-memory and <strong>not stored</strong>. 
                        Do not upload confidential or sensitive meetings. 
                        For enterprise use, consider self-hosting.
                    </div>
                </div>
                """
            )
    
    # =========================================================================
    # Status Indicator
    # =========================================================================
    with gr.Row(visible=MOCK_MODE or True):
        with gr.Column(scale=4):
            pass  # Spacer
        with gr.Column(scale=1):
            status_indicator = gr.Markdown(
                value=get_status_indicator(),
                elem_classes=["status-badge", "mock-badge" if MOCK_MODE else "ready-badge"],
                show_label=False,
            )
    
    # =========================================================================
    # Main Content - Two Column Layout
    # =========================================================================
    with gr.Row(equal_height=True):
        
        # =====================================================================
        # LEFT COLUMN - Input Section
        # =====================================================================
        with gr.Column(scale=1, min_width=350):
            
            # Input Section Header
            gr.HTML(
                """
                <div class="section-header">
                    <span>📝</span> Input
                </div>
                """,
                elem_classes=["input-section"],
            )
            
            # Audio Input Component
            audio_input = gr.Audio(
                label="🎤 Upload Recording",
                sources=["upload", "microphone"],
                type="filepath",
                show_label=True,
                interactive=True,
                elem_classes=["audio-component"],
                show_download_button=False,
            )
            
            # Duration Status Display
            duration_status = gr.Markdown(
                value="",
                elem_classes=["duration-warning"],
                visible=False,
                show_label=False,
            )
            
            # Accepted formats hint
            gr.HTML(
                """
                <p style="font-size: 0.8rem; color: #6b7280; margin-top: 0.25rem;">
                    Accepted formats: WAV, MP3, M4A, WebM (max 5 min recommended)
                </p>
                """
            )
            
            # Divider
            gr.HTML("<hr style='border-color: #e5e7eb; margin: 1rem 0;'>")
            
            # Text Input Alternative
            transcript_input = gr.Textbox(
                label="📝 Or Paste Transcript",
                lines=8,
                max_lines=50,
                placeholder="Paste meeting transcript here...\n\nExample:\nTeam discussed Q3 goals. Alex will update the docs by Friday. Sarah agreed to schedule the client demo for next week.",
                show_label=True,
                interactive=True,
                show_copy_button=True,
                elem_classes=["transcript-input"],
            )
            
            # Meeting Type Selector
            meeting_type = gr.Dropdown(
                label="📋 Meeting Type",
                choices=[
                    "Standup",
                    "Client Call",
                    "Brainstorm",
                    "Retrospective",
                    "Other",
                ],
                value="Standup",
                show_label=True,
                interactive=True,
                elem_classes=["meeting-type-selector"],
                info="Select the type of meeting for appropriate formatting",
            )
            
            # Generate Button
            generate_btn = gr.Button(
                "✨ Generate Minutes",
                variant="primary",
                size="lg",
                elem_classes=["generate-button"],
                interactive=True,
            )
            
            # Tips Section
            with gr.Accordion("💡 Tips for Best Results", open=False):
                gr.HTML(
                    """
                    <ul class="tips-list">
                        <li>Ensure clear audio with minimal background noise</li>
                        <li>Speak clearly and at a moderate pace</li>
                        <li>For long meetings, consider processing in segments</li>
                        <li>Mention names, deadlines, and commitments explicitly</li>
                        <li>Processing may take 30-60 seconds on free tier</li>
                    </ul>
                    """
                )
        
        # =====================================================================
        # RIGHT COLUMN - Output Section
        # =====================================================================
        with gr.Column(scale=2, min_width=500):
            
            # Output Section Header
            gr.HTML(
                """
                <div class="section-header">
                    <span>📊</span> Output
                </div>
                """
            )
            
            # Executive Summary Output
            summary_output = gr.Markdown(
                label="📋 Executive Summary",
                value="*Summary will appear here after processing...*",
                show_label=True,
                elem_classes=["summary-output"],
                elem_id="summary-output",
                visible=True,
            )
            
            # Action Items Table
            actions_output = gr.Dataframe(
                label="✅ Action Items",
                headers=["Task", "Owner", "Deadline"],
                datatype=["str", "str", "str"],
                row_count=(1, "dynamic"),
                col_count=(3, "fixed"),
                show_label=True,
                interactive=False,
                elem_classes=["action-items-table"],
                wrap=True,
            )
            
            # Full Minutes Output
            minutes_output = gr.Textbox(
                label="📄 Full Minutes (Markdown)",
                lines=15,
                max_lines=30,
                value="",
                interactive=False,
                show_label=True,
                show_copy_button=True,
                elem_classes=["minutes-output"],
                placeholder="Full formatted meeting minutes will appear here...",
            )
            
            # Download Button
            download_output = gr.File(
                label="💾 Download as Markdown",
                show_label=True,
                file_count="single",
                file_types=[".md"],
                elem_classes=["download-button"],
                interactive=False,
            )
    
    # =========================================================================
    # Examples Section
    # =========================================================================
    with gr.Row():
        with gr.Column():
            gr.HTML(
                """
                <div class="section-header" style="margin-top: 1rem;">
                    <span>📚</span> Quick Examples
                </div>
                """
            )
            
            gr.Examples(
                examples=[
                    ["", EXAMPLE_TRANSCRIPT_STANDUP, "Standup"],
                    ["", EXAMPLE_TRANSCRIPT_CLIENT, "Client Call"],
                    ["", EXAMPLE_TRANSCRIPT_BRAINSTORM, "Brainstorm"],
                ],
                inputs=[audio_input, transcript_input, meeting_type],
                label="Click an example to try it:",
                elem_classes=["examples-section"],
                examples_per_page=3,
            )
    
    # =========================================================================
    # Footer
    # =========================================================================
    with gr.Row():
        with gr.Column():
            gr.HTML(
                """
                <div style="text-align: center; padding: 1rem; color: #6b7280; font-size: 0.875rem; margin-top: 1rem; border-top: 1px solid #e5e7eb;">
                    <p>
                        Built with 🤗 Hugging Face Transformers & Gradio | 
                        <a href="https://github.com/insydr/MeetingMinutesGenerator" target="_blank" rel="noopener noreferrer">
                            GitHub
                        </a>
                    </p>
                    <p>
                        Models: OpenAI Whisper-small | Facebook BART-large-cnn | Google Flan-T5-small
                    </p>
                </div>
                """
            )
    
    # =========================================================================
    # Event Handlers
    # =========================================================================
    
    # Audio upload handler - check duration
    def on_audio_change(audio_path: Optional[str]) -> Tuple[gr.update, gr.update]:
        """Handle audio file upload and check duration."""
        if audio_path:
            duration_msg = check_audio_duration(audio_path)
            is_warning = "Warning" in duration_msg
            return (
                gr.update(value=duration_msg, visible=True),
                gr.update(elem_classes=["duration-warning"] if is_warning else []),
            )
        return gr.update(value="", visible=False), gr.update()
    
    audio_input.change(
        fn=on_audio_change,
        inputs=[audio_input],
        outputs=[duration_status, duration_status],
    )
    
    # Generate button click handler
    generate_btn.click(
        fn=process_meeting,
        inputs=[audio_input, transcript_input, meeting_type],
        outputs=[summary_output, actions_output, minutes_output, download_output],
        api_name="generate",
    )
    
    # Keyboard shortcut for generate (Enter in transcript box)
    transcript_input.submit(
        fn=process_meeting,
        inputs=[audio_input, transcript_input, meeting_type],
        outputs=[summary_output, actions_output, minutes_output, download_output],
    )


# =============================================================================
# Application Startup
# =============================================================================

def initialize_app():
    """Initialize the application and load models."""
    logger.info("=" * 60)
    logger.info("Meeting Minutes Generator - Starting Up")
    logger.info("=" * 60)
    logger.info(f"Python version: {sys.version}")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"Gradio version: {gr.__version__}")
    logger.info(f"Device: {DEVICE.upper()}")
    logger.info(f"Mock mode: {MOCK_MODE}")
    logger.info(f"Debug mode: {DEBUG_MODE}")
    logger.info("=" * 60)
    
    # Load models
    success = models.load_all_models()
    
    if not success:
        logger.error("Failed to load some models. Check logs for details.")
        if not MOCK_MODE:
            logger.warning("Consider enabling MOCK_MODE for testing")
    
    return success


if __name__ == "__main__":
    # Initialize and launch
    initialize_app()
    
    logger.info("Launching Gradio interface...")
    demo.launch(
        share=False,
        show_error=True,
        quiet=False,
    )
