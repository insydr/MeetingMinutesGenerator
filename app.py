"""
Meeting Minutes Generator
=========================

An AI-powered web application that transforms meeting recordings or transcripts
into structured, actionable meeting minutes.

This application uses:
- OpenAI Whisper-small for audio transcription
- Facebook BART-large-cnn for text summarization
- Google Flan-T5-small for action item extraction

Optimized for Hugging Face Spaces (Free CPU tier) per PRD Section 6 (NFR-1, NFR-2).

Based on PRD Section 7: Technical Architecture
"""

import os
import sys
import signal
import tempfile
import logging
import time
import threading
import gc
from datetime import datetime
from typing import Optional, Tuple, List, Dict, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from contextlib import contextmanager
from collections import defaultdict
import json

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
MAX_AUDIO_DURATION_HARD_LIMIT = 600  # 10 minutes hard limit
MAX_AUDIO_FILE_SIZE_MB = 25  # 25MB max file size
MAX_TRANSCRIPT_LENGTH = 10000  # Characters for processing
MIN_TRANSCRIPT_WORDS = 50  # Minimum words for meaningful summary
MAX_PROCESSING_TIMEOUT = 120  # Total processing timeout in seconds

# Accepted audio formats (FR-1)
ACCEPTED_AUDIO_FORMATS = [".wav", ".mp3", ".m4a", ".webm"]
ACCEPTED_AUDIO_MIME_TYPES = ["audio/wav", "audio/mpeg", "audio/mp4", "audio/webm", "audio/x-m4a"]

# Model constraints
BART_MAX_INPUT_TOKENS = 1024  # BART max input length
SUMMARY_MAX_LENGTH = 150  # Maximum summary tokens
SUMMARY_MIN_LENGTH = 40  # Minimum summary tokens
CHUNK_OVERLAP_CHARS = 200  # Overlap between chunks for context

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.float32  # Use float32 for CPU compatibility (avoid float16 CPU issues)

# CPU-optimized settings for Hugging Face Spaces free tier
BATCH_SIZE = 1  # Process one item at a time for memory efficiency
ENABLE_GRADIENT_COMPUTATION = False  # Disable gradients for inference only


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


@dataclass
class ValidationResult:
    """Result of input validation."""
    valid: bool
    message: str
    warning: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


# =============================================================================
# Input Validation Functions - PRD Section 5.1 (FR-1, FR-2)
# =============================================================================

def validate_audio_input(audio_path: Optional[str]) -> ValidationResult:
    """
    Validate audio file input per FR-1.
    
    Checks:
    - FR-1.1: File exists and is not empty
    - FR-1.2: File size <= 25MB
    - FR-1.3: Duration <= 10 minutes (hard limit), warn at 5 minutes
    - FR-1.4: Format is WAV, MP3, M4A, or WebM
    
    Args:
        audio_path: Path to uploaded audio file
        
    Returns:
        ValidationResult with status and message
    """
    if not audio_path or audio_path == "":
        return ValidationResult(
            valid=False,
            message="No audio file provided",
        )
    
    # Check file exists
    if not os.path.exists(audio_path):
        return ValidationResult(
            valid=False,
            message="Audio file not found. Please re-upload.",
        )
    
    # Check file extension/format (FR-1.4)
    file_ext = os.path.splitext(audio_path)[1].lower()
    if file_ext not in ACCEPTED_AUDIO_FORMATS:
        return ValidationResult(
            valid=False,
            message=f"Unsupported format '{file_ext}'. Please use: WAV, MP3, M4A, or WebM.",
        )
    
    # Check file size (FR-1.2)
    file_size_mb = os.path.getsize(audio_path) / (1024 * 1024)
    if file_size_mb > MAX_AUDIO_FILE_SIZE_MB:
        return ValidationResult(
            valid=False,
            message=f"File too large ({file_size_mb:.1f}MB). Maximum size is {MAX_AUDIO_FILE_SIZE_MB}MB.",
        )
    
    # Check duration (FR-1.3)
    try:
        import torchaudio
        waveform, sample_rate = torchaudio.load(audio_path)
        duration_seconds = waveform.shape[1] / sample_rate
        duration_minutes = duration_seconds / 60
        
        metadata = {
            "duration_seconds": duration_seconds,
            "duration_minutes": duration_minutes,
            "file_size_mb": file_size_mb,
            "sample_rate": sample_rate,
            "format": file_ext,
        }
        
        # Hard limit check (10 minutes)
        if duration_seconds > MAX_AUDIO_DURATION_HARD_LIMIT:
            return ValidationResult(
                valid=False,
                message=f"Audio too long ({duration_minutes:.1f} minutes). Maximum is 10 minutes.",
                metadata=metadata,
            )
        
        # Warning for > 5 minutes
        warning = None
        if duration_seconds > MAX_AUDIO_DURATION_SECONDS:
            warning = f"⚠️ Audio is {duration_minutes:.1f} minutes. Recommended max is 5 minutes. Processing may be slower."
        
        # Success with optional warning
        minutes = int(duration_seconds // 60)
        seconds = int(duration_seconds % 60)
        message = f"✅ Valid audio: {minutes}:{seconds:02d} ({file_size_mb:.1f}MB)"
        
        return ValidationResult(
            valid=True,
            message=message,
            warning=warning,
            metadata=metadata,
        )
        
    except Exception as e:
        logger.warning(f"Could not validate audio duration: {str(e)}")
        # Allow processing if duration check fails
        return ValidationResult(
            valid=True,
            message=f"✅ Audio uploaded ({file_size_mb:.1f}MB)",
            warning="Could not verify duration. Ensure recording is under 10 minutes.",
            metadata={"file_size_mb": file_size_mb, "format": file_ext},
        )


def validate_text_input(text: Optional[str]) -> ValidationResult:
    """
    Validate text transcript input per FR-2.
    
    Checks:
    - FR-2.1: Text is not empty after stripping whitespace
    - FR-2.2: Warning if < 50 words (likely insufficient)
    - FR-2.3: Auto-truncate if > 10,000 characters
    
    Args:
        text: Pasted transcript text
        
    Returns:
        ValidationResult with status and message
    """
    if not text:
        return ValidationResult(
            valid=False,
            message="No transcript text provided",
        )
    
    # Strip whitespace (FR-2.1)
    stripped_text = text.strip()
    
    if not stripped_text:
        return ValidationResult(
            valid=False,
            message="Transcript is empty. Please paste meeting text.",
        )
    
    # Count words
    words = stripped_text.split()
    word_count = len(words)
    char_count = len(stripped_text)
    
    metadata = {
        "word_count": word_count,
        "char_count": char_count,
        "will_truncate": char_count > MAX_TRANSCRIPT_LENGTH,
    }
    
    # Warning for short text (FR-2.2)
    warning = None
    if word_count < MIN_TRANSCRIPT_WORDS:
        warning = f"⚠️ Only {word_count} words detected. For meaningful summaries, provide at least {MIN_TRANSCRIPT_WORDS} words."
    
    # Notice for long text (FR-2.3)
    message = f"✅ Valid transcript: {word_count} words, {char_count:,} characters"
    if char_count > MAX_TRANSCRIPT_LENGTH:
        message = f"✅ Transcript: {word_count} words (will be truncated to {MAX_TRANSCRIPT_LENGTH:,} chars)"
        warning = (warning + "\n" if warning else "") + f"📄 Long transcript detected. Only first {MAX_TRANSCRIPT_LENGTH:,} characters will be processed."
    
    return ValidationResult(
        valid=True,
        message=message,
        warning=warning,
        metadata=metadata,
    )


def get_audio_tips() -> str:
    """
    Return helpful tips for audio recording quality.
    Per FR-9: User guidance.
    """
    return """
### 🎤 Audio Recording Tips

For best transcription and extraction results:

1. **Quiet Environment**: Record in a quiet space with minimal background noise
2. **Clear Speech**: Speak clearly and at a moderate pace
3. **Good Microphone**: Use a quality microphone or headset
4. **Single Speaker**: Best results with one clear speaker
5. **Short Segments**: Keep recordings under 5 minutes for faster processing

**Supported formats**: WAV, MP3, M4A, WebM
**Maximum size**: 25MB
**Recommended duration**: Under 5 minutes
"""


def get_transcript_tips() -> str:
    """
    Return helpful tips for transcript input.
    Per FR-9: User guidance.
    """
    return """
### 📝 Transcript Tips

For best summarization and extraction results:

1. **Include Context**: Add speaker names and meeting context
2. **Clear Commitments**: Use phrases like "I will...", "John agreed to..."
3. **Specific Deadlines**: Mention exact dates and times
4. **Structured Content**: Use paragraphs and bullet points
5. **Sufficient Length**: At least 50 words for meaningful summaries

**Example good transcript**:
```
Team standup - March 27, 2024
Attendees: Alex, Jordan, Sarah

Alex: I'll complete the UI mockups by Friday.
Jordan: Agreed to update the project timeline by next Monday.
Sarah: Will schedule the client demo for next week.
```
"""


def get_quality_tips() -> str:
    """
    Return comprehensive quality tips for better results.
    Per FR-9: User guidance.
    """
    return """
### 💡 Tips for Better Results

#### Recording Quality
- Use a quiet environment for recordings
- Speak clearly and at a moderate pace
- Position microphone close to speakers
- Avoid overlapping conversations

#### Meeting Structure
- State names and roles at the beginning
- Verbally confirm commitments: "Sarah will... by Friday"
- Summarize decisions at the end
- Assign clear owners and deadlines

#### For Long Meetings
- Consider splitting into segments (< 5 minutes each)
- Process key sections separately
- Focus on action-oriented portions
"""


def get_privacy_notice() -> str:
    """
    Return privacy notice per PRD NFR-4.
    """
    return """
> 🔒 **Privacy Notice**
> 
> Audio is processed **in-memory and not stored** on our servers.
> 
> ⚠️ **Do not upload**:
> - Confidential business meetings
> - Personal or sensitive information
> - Meetings with non-public financial data
> - HR or legal discussions
> 
> For enterprise use with sensitive data, consider **self-hosting** this application.
"""


def get_processing_stage_message(stage: ProcessingStage) -> str:
    """
    Get user-friendly message for processing stage.
    """
    messages = {
        ProcessingStage.INITIALIZING: "🔄 Initializing...",
        ProcessingStage.INPUT_ROUTING: "📂 Validating input...",
        ProcessingStage.TRANSCRIPTION: "🎤 Transcribing audio...",
        ProcessingStage.SUMMARIZATION: "📝 Generating summary...",
        ProcessingStage.EXTRACTION: "✅ Extracting action items...",
        ProcessingStage.FORMATTING: "✨ Formatting minutes...",
        ProcessingStage.COMPLETE: "✅ Complete!",
        ProcessingStage.ERROR: "❌ Error occurred",
    }
    return messages.get(stage, "Processing...")


# =============================================================================
# Usage Analytics (In-Memory Only) - NFR-2
# =============================================================================

@dataclass
class UsageStats:
    """In-memory usage statistics for monitoring."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    audio_requests: int = 0
    text_requests: int = 0
    total_processing_time: float = 0.0
    stage_times: Dict[str, float] = field(default_factory=lambda: defaultdict(float))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "audio_requests": self.audio_requests,
            "text_requests": self.text_requests,
            "avg_processing_time": (
                self.total_processing_time / self.successful_requests 
                if self.successful_requests > 0 else 0
            ),
            "stage_avg_times": {
                k: v / self.successful_requests if self.successful_requests > 0 else 0
                for k, v in self.stage_times.items()
            },
        }


# Global usage statistics (thread-safe)
usage_stats = UsageStats()
stats_lock = threading.Lock()


# =============================================================================
# Global Model Containers - Optimized for CPU Inference
# =============================================================================

class ModelContainer:
    """
    Container for globally loaded models.
    Models are loaded once at startup and reused for all requests.
    
    Optimizations for Hugging Face Spaces CPU tier:
    - Uses float32 for CPU compatibility (avoids float16 errors)
    - Sets models to eval() mode after loading
    - Models cached globally to avoid reloading per request
    """
    
    def __init__(self, mock_mode: bool = False):
        self.mock_mode = mock_mode
        self._transcriber = None
        self._summarizer = None
        self._extractor = None
        self._load_results: List[ModelLoadResult] = []
        self._model_info: Dict[str, Dict[str, Any]] = {}
        
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
        """Load the Whisper ASR model with CPU optimizations."""
        try:
            logger.info(f"Loading Whisper model: {WHISPER_MODEL_ID}")
            
            from transformers import pipeline, AutoModelForSpeechSeq2Seq, AutoProcessor
            
            # Load model with float32 for CPU compatibility
            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                WHISPER_MODEL_ID,
                torch_dtype=TORCH_DTYPE,
                low_cpu_mem_usage=True,
            )
            
            # Set to eval mode for inference (no dropout, no gradient computation)
            model.eval()
            
            # Load processor
            processor = AutoProcessor.from_pretrained(WHISPER_MODEL_ID)
            
            # Create pipeline with optimized settings
            self._transcriber = pipeline(
                "automatic-speech-recognition",
                model=model,
                tokenizer=processor.tokenizer,
                feature_extractor=processor.feature_extractor,
                torch_dtype=TORCH_DTYPE,
                device=-1,  # CPU only
            )
            
            # Store model info
            self._model_info["transcriber"] = {
                "model_id": WHISPER_MODEL_ID,
                "dtype": str(TORCH_DTYPE),
                "device": "cpu",
                "parameters": sum(p.numel() for p in model.parameters()),
            }
            
            logger.info(f"Whisper model loaded successfully (params: {self._model_info['transcriber']['parameters']:,})")
            return ModelLoadResult(True, WHISPER_MODEL_ID)
            
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {str(e)}")
            return ModelLoadResult(False, WHISPER_MODEL_ID, str(e))
    
    def _load_summarizer(self) -> ModelLoadResult:
        """Load the BART summarization model with CPU optimizations."""
        try:
            logger.info(f"Loading BART model: {BART_MODEL_ID}")
            
            from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
            
            # Load model with float32 for CPU compatibility
            model = AutoModelForSeq2SeqLM.from_pretrained(
                BART_MODEL_ID,
                torch_dtype=TORCH_DTYPE,
                low_cpu_mem_usage=True,
            )
            
            # Set to eval mode for inference
            model.eval()
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(BART_MODEL_ID)
            
            # Create pipeline with optimized settings
            self._summarizer = pipeline(
                "summarization",
                model=model,
                tokenizer=tokenizer,
                torch_dtype=TORCH_DTYPE,
                device=-1,  # CPU only
            )
            
            # Store model info
            self._model_info["summarizer"] = {
                "model_id": BART_MODEL_ID,
                "dtype": str(TORCH_DTYPE),
                "device": "cpu",
                "parameters": sum(p.numel() for p in model.parameters()),
            }
            
            logger.info(f"BART model loaded successfully (params: {self._model_info['summarizer']['parameters']:,})")
            return ModelLoadResult(True, BART_MODEL_ID)
            
        except Exception as e:
            logger.error(f"Failed to load BART model: {str(e)}")
            return ModelLoadResult(False, BART_MODEL_ID, str(e))
    
    def _load_extractor(self) -> ModelLoadResult:
        """Load the Flan-T5 model for action item extraction with CPU optimizations."""
        try:
            logger.info(f"Loading Flan-T5 model: {FLAN_T5_MODEL_ID}")
            
            from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
            
            # Load model with float32 for CPU compatibility
            model = AutoModelForSeq2SeqLM.from_pretrained(
                FLAN_T5_MODEL_ID,
                torch_dtype=TORCH_DTYPE,
                low_cpu_mem_usage=True,
            )
            
            # Set to eval mode for inference
            model.eval()
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(FLAN_T5_MODEL_ID)
            
            # Create pipeline with optimized settings
            self._extractor = pipeline(
                "text2text-generation",
                model=model,
                tokenizer=tokenizer,
                torch_dtype=TORCH_DTYPE,
                device=-1,  # CPU only
            )
            
            # Store model info
            self._model_info["extractor"] = {
                "model_id": FLAN_T5_MODEL_ID,
                "dtype": str(TORCH_DTYPE),
                "device": "cpu",
                "parameters": sum(p.numel() for p in model.parameters()),
            }
            
            logger.info(f"Flan-T5 model loaded successfully (params: {self._model_info['extractor']['parameters']:,})")
            return ModelLoadResult(True, FLAN_T5_MODEL_ID)
            
        except Exception as e:
            logger.error(f"Failed to load Flan-T5 model: {str(e)}")
            return ModelLoadResult(False, FLAN_T5_MODEL_ID, str(e))


# Initialize global model container
models = ModelContainer(mock_mode=MOCK_MODE)


# =============================================================================
# Memory Management & Performance Utilities - NFR-1
# =============================================================================

def cleanup_memory() -> None:
    """
    Clean up memory after processing.
    Important for Hugging Face Spaces free tier with limited RAM.
    """
    # Force garbage collection
    gc.collect()
    
    # Clear CUDA cache if GPU is available (future-proofing)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    logger.debug("Memory cleanup performed")


@contextmanager
def torch_inference_mode():
    """
    Context manager for inference-only mode.
    Disables gradient computation for memory efficiency.
    """
    with torch.no_grad():
        yield


def log_stage_time(stage_name: str, start_time: float) -> float:
    """
    Log processing time for a stage and return elapsed time.
    
    Args:
        stage_name: Name of the processing stage
        start_time: Start time from time.time()
        
    Returns:
        Elapsed time in seconds
    """
    elapsed = time.time() - start_time
    logger.info(f"⏱️ {stage_name}: {elapsed:.2f}s")
    
    # Update stats
    with stats_lock:
        usage_stats.stage_times[stage_name] += elapsed
    
    return elapsed


def update_usage_stats(
    success: bool,
    processing_time: float,
    input_type: str = "text"
) -> None:
    """
    Update usage statistics (thread-safe).
    
    Args:
        success: Whether the request was successful
        processing_time: Total processing time in seconds
        input_type: 'audio' or 'text'
    """
    with stats_lock:
        usage_stats.total_requests += 1
        usage_stats.total_processing_time += processing_time
        
        if success:
            usage_stats.successful_requests += 1
        else:
            usage_stats.failed_requests += 1
        
        if input_type == "audio":
            usage_stats.audio_requests += 1
        else:
            usage_stats.text_requests += 1


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
    
    Optimized for CPU inference with torch.no_grad() context.
    
    Args:
        audio_path: Path to the audio file (WAV, MP3, M4A, WebM) or bytes
        progress_callback: Optional callback for progress updates
        
    Returns:
        Transcribed text string
        
    Raises:
        ValueError: If audio_path is None or empty
        RuntimeError: If transcription fails
    """
    t0 = time.time()
    
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
        
        # Transcribe with Whisper - using torch.no_grad() for memory efficiency
        with torch_inference_mode():
            result = models.transcriber(
                audio_path,
                return_timestamps=False,
                batch_size=BATCH_SIZE,
                generate_kwargs={
                    "language": "english",
                    "task": "transcribe",
                }
            )
        
        transcript = result["text"].strip()
        
        # Log timing for monitoring
        t1 = time.time()
        logger.info(f"⏱️ Transcription: {t1-t0:.2f}s ({len(transcript)} chars)")
        
        # Update stats
        with stats_lock:
            usage_stats.stage_times["transcription"] += (t1 - t0)
        
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
    
    Optimized for CPU inference with torch.no_grad() context.
    
    Args:
        text: Full meeting transcript text
        progress_callback: Optional callback for progress updates
        
    Returns:
        Summary text (40-150 tokens)
    """
    t0 = time.time()
    
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
            # Generate summary for each chunk - using torch.no_grad() for memory efficiency
            with torch_inference_mode():
                summary_result = models.summarizer(
                    chunk,
                    max_length=SUMMARY_MAX_LENGTH,
                    min_length=SUMMARY_MIN_LENGTH,
                    do_sample=False,
                    truncation=True,
                    batch_size=BATCH_SIZE,
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
                with torch_inference_mode():
                    final_result = models.summarizer(
                        combined,
                        max_length=SUMMARY_MAX_LENGTH,
                        min_length=SUMMARY_MIN_LENGTH,
                        do_sample=False,
                        truncation=True,
                        batch_size=BATCH_SIZE,
                    )
                final_summary = final_result[0]["summary_text"].strip()
            except Exception as e:
                logger.warning(f"Failed to combine summaries: {str(e)}")
                final_summary = combined[:500]
        else:
            final_summary = combined
    
    # Log timing for monitoring
    t1 = time.time()
    logger.info(f"⏱️ Summarization: {t1-t0:.2f}s ({len(final_summary)} chars)")
    
    # Update stats
    with stats_lock:
        usage_stats.stage_times["summarization"] += (t1 - t0)
    
    if progress_callback:
        progress_callback(1.0, "Summarization complete")
    
    return final_summary


def extract_action_items(
    text: str,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> List[Dict[str, str]]:
    """
    Extract action items from transcript using Flan-T5-small with prompt engineering.
    
    Optimized for CPU inference with torch.no_grad() context.
    
    Args:
        text: Full meeting transcript text
        progress_callback: Optional callback for progress updates
        
    Returns:
        List of dicts with 'task', 'owner', 'deadline' keys
    """
    t0 = time.time()
    
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
        # Generate extraction - using torch.no_grad() for memory efficiency
        with torch_inference_mode():
            result = models.extractor(
                extraction_prompt,
                max_length=256,
                num_beams=2,
                temperature=0.3,
                do_sample=True,
                batch_size=BATCH_SIZE,
            )
        
        extraction_output = result[0]["generated_text"].strip()
        logger.info(f"Extraction output: {extraction_output}")
        
        if progress_callback:
            progress_callback(0.8, "Parsing action items...")
        
        # Parse the output
        action_items = parse_action_items_output(extraction_output, text)
        
        # Log timing for monitoring
        t1 = time.time()
        logger.info(f"⏱️ Action extraction: {t1-t0:.2f}s ({len(action_items)} items)")
        
        # Update stats
        with stats_lock:
            usage_stats.stage_times["extraction"] += (t1 - t0)
        
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


# =============================================================================
# Output Formatting Functions - PRD Section 5.1
# =============================================================================

def format_summary(summary_text: str, confidence_warning: str = "") -> str:
    """
    Format executive summary for Markdown display per FR-7.
    
    Args:
        summary_text: The generated summary text
        confidence_warning: Optional warning about low confidence transcription
        
    Returns:
        Formatted Markdown string for gr.Markdown component
    """
    formatted = f"## 📌 Key Takeaways\n\n{summary_text}"
    
    if confidence_warning:
        formatted += f"\n\n---\n\n⚠️ **Note:** {confidence_warning}"
    
    return formatted


def format_action_items(items: List[Dict[str, str]]) -> List[List[str]]:
    """
    Convert action items to dataframe format per FR-5.
    
    Args:
        items: List of action item dicts with 'task', 'owner', 'deadline' keys
        
    Returns:
        List of [task, owner, deadline] lists for gr.Dataframe
    """
    if not items:
        return [["No action items identified", "TBD", "TBD"]]
    
    formatted_items = []
    for item in items:
        task = item.get("task", "TBD")
        owner = item.get("owner", "TBD")
        deadline = item.get("deadline", "TBD")
        
        # Apply FR-5.5: Use TBD for missing info
        if not task or task.lower() in ["", "none", "n/a"]:
            task = "TBD"
        if not owner or owner.lower() in ["", "none", "n/a"]:
            owner = "TBD"
        if not deadline or deadline.lower() in ["", "none", "n/a"]:
            deadline = "TBD"
        
        formatted_items.append([task, owner, deadline])
    
    return formatted_items


def format_full_minutes(
    summary: str,
    action_items: List[Dict[str, str]],
    meeting_type: str,
    timestamp: str,
    transcript: str = "",
    confidence_warning: str = "",
    processing_time: float = 0.0,
) -> str:
    """
    Format complete meeting minutes in Markdown per FR-7.
    
    Args:
        summary: Executive summary text
        action_items: List of action item dicts
        meeting_type: Type of meeting
        timestamp: Generation timestamp
        transcript: Original transcript (optional)
        confidence_warning: Warning about transcription quality
        processing_time: Time taken to process in seconds
        
    Returns:
        Formatted Markdown string
    """
    # Build action items table rows
    table_rows = ""
    for item in action_items:
        task = item.get("task", "TBD")
        owner = item.get("owner", "TBD")
        deadline = item.get("deadline", "TBD")
        table_rows += f"| {task} | {owner} | {deadline} |\n"
    
    if not action_items:
        table_rows = "| No action items identified | TBD | TBD |\n"
    
    # Build discussion points (placeholder per PRD)
    discussion_points = """• Key topics were discussed during the meeting
• Decisions were made based on team consensus
• Follow-up actions were assigned to relevant team members"""
    
    minutes = f"""# Meeting Minutes - {meeting_type}

*Generated: {timestamp}*  
*Processing time: {processing_time:.1f}s*

---

## 📋 Executive Summary

{summary}
"""
    
    # Add confidence warning if present
    if confidence_warning:
        minutes += f"\n\n> ⚠️ **Transcription Quality Note:** {confidence_warning}\n"
    
    minutes += f"""
---

## ✅ Action Items

| Task | Owner | Deadline |
|------|-------|----------|
{table_rows}
---

## 🗣️ Key Discussion Points

{discussion_points}

---

*Generated by [Meeting Minutes Generator](https://github.com/insydr/MeetingMinutesGenerator)*
"""
    
    return minutes


def get_empty_action_items_tip() -> str:
    """
    Return a helpful tip when no action items are found.
    Per FR-9: User guidance.
    """
    return """💡 **Tip:** No action items were detected. For better extraction:
    - Use clear commitment language (e.g., "I will...", "John agreed to...")
    - Mention specific deadlines and responsible parties
    - Structure action items clearly in the meeting
    """


def assess_transcription_quality(transcript: str, has_audio: bool) -> Tuple[str, str]:
    """
    Assess transcription quality and return warning if needed.
    Per FR-9.3: Quality indicators.
    
    Args:
        transcript: The transcribed text
        has_audio: Whether the input was audio (vs text)
        
    Returns:
        Tuple of (quality_level, warning_message)
    """
    if not has_audio:
        return "good", ""  # Text input doesn't have transcription quality issues
    
    # Check for common transcription issues
    issues = []
    
    # Check for very short transcripts
    if len(transcript) < 50:
        issues.append("Very short transcript")
    
    # Check for repeated words (hallucination indicator)
    words = transcript.lower().split()
    if len(words) > 10:
        unique_words = set(words)
        repetition_ratio = len(words) / len(unique_words)
        if repetition_ratio > 3:
            issues.append("High word repetition detected")
    
    # Check for excessive filler/transcription artifacts
    filler_patterns = ["um", "uh", "er", "ah"]
    filler_count = sum(transcript.lower().count(f) for f in filler_patterns)
    if filler_count > len(words) * 0.1:  # More than 10% fillers
        issues.append("High filler word count")
    
    # Check for garbled text (non-ASCII ratio)
    try:
        non_ascii = sum(1 for c in transcript if ord(c) > 127)
        if non_ascii > len(transcript) * 0.1:
            issues.append("Unusual characters detected")
    except:
        pass
    
    if issues:
        warning = "Transcription may have quality issues: " + ", ".join(issues[:2]) + ". Review carefully."
        return "low", warning
    
    return "good", ""


def get_processing_time_estimate(input_type: str, content_length: int) -> str:
    """
    Provide estimated processing time based on input.
    Per FR-9.3: Time expectations.
    
    Args:
        input_type: 'audio' or 'text'
        content_length: Duration in seconds (audio) or character count (text)
        
    Returns:
        Human-readable time estimate
    """
    if input_type == "audio":
        # Audio: roughly 10-30 seconds per minute of audio
        minutes = content_length / 60
        estimated_seconds = int(minutes * 20)  # Average 20 seconds per minute
        if minutes <= 1:
            return "~30 seconds"
        elif minutes <= 3:
            return "~1-2 minutes"
        elif minutes <= 5:
            return "~2-3 minutes"
        else:
            return f"~{estimated_seconds // 60 + 1} minutes"
    else:
        # Text: roughly 5-15 seconds per 1000 characters
        if content_length < 500:
            return "~10-15 seconds"
        elif content_length < 2000:
            return "~20-30 seconds"
        else:
            return "~30-60 seconds"


def format_meeting_minutes(
    summary: str,
    action_items: List[Dict[str, str]],
    meeting_type: str,
    timestamp: str,
    transcript: str = "",
) -> str:
    """
    Legacy wrapper - now calls format_full_minutes.
    Maintained for backwards compatibility.
    """
    return format_full_minutes(
        summary=summary,
        action_items=action_items,
        meeting_type=meeting_type,
        timestamp=timestamp,
        transcript=transcript,
    )


# =============================================================================
# Download Helper Functions - FR-8
# =============================================================================

def create_download_file(
    minutes_markdown: str,
    meeting_type: str,
    output_dir: Optional[str] = None,
) -> Optional[str]:
    """
    Create a downloadable Markdown file from meeting minutes.
    
    Implements FR-8: Document Download functionality.
    
    Args:
        minutes_markdown: The formatted markdown content
        meeting_type: Type of meeting for filename
        output_dir: Optional directory for output (defaults to temp)
        
    Returns:
        Path to created file, or None if creation failed
    """
    if not minutes_markdown or not minutes_markdown.strip():
        logger.warning("No minutes content to save")
        return None
    
    try:
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_meeting_type = meeting_type.lower().replace(" ", "_").replace("-", "_")
        filename = f"meeting_minutes_{safe_meeting_type}_{timestamp}.md"
        
        # Use provided output directory or temp directory
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            file_path = os.path.join(output_dir, filename)
        else:
            file_path = os.path.join(tempfile.gettempdir(), filename)
        
        # Write content with UTF-8 encoding
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(minutes_markdown)
        
        logger.info(f"Created download file: {file_path}")
        return file_path
        
    except Exception as e:
        logger.error(f"Failed to create download file: {str(e)}")
        return None


def create_multiple_format_downloads(
    minutes_markdown: str,
    meeting_type: str,
    include_txt: bool = True,
) -> Dict[str, Optional[str]]:
    """
    Create downloads in multiple formats.
    
    Args:
        minutes_markdown: The formatted markdown content
        meeting_type: Type of meeting for filename
        include_txt: Whether to also create a plain text version
        
    Returns:
        Dict mapping format names to file paths
    """
    results = {
        "markdown": create_download_file(minutes_markdown, meeting_type),
    }
    
    if include_txt and minutes_markdown:
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_meeting_type = meeting_type.lower().replace(" ", "_")
            txt_filename = f"meeting_minutes_{safe_meeting_type}_{timestamp}.txt"
            txt_path = os.path.join(tempfile.gettempdir(), txt_filename)
            
            # Strip markdown formatting for plain text
            plain_text = minutes_markdown
            # Remove markdown headers (### and ##)
            plain_text = plain_text.replace("### ", "").replace("## ", "").replace("# ", "")
            # Remove bold/italic markers
            plain_text = plain_text.replace("**", "").replace("*", "")
            # Remove table formatting
            plain_text = plain_text.replace("|", " | ")
            
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(plain_text)
            
            results["txt"] = txt_path
            
        except Exception as e:
            logger.error(f"Failed to create TXT file: {str(e)}")
            results["txt"] = None
    
    return results


def get_copy_to_clipboard_html(element_id: str = "minutes-output") -> str:
    """
    Generate HTML for a copy-to-clipboard button with JavaScript.
    
    Args:
        element_id: ID of the element to copy from
        
    Returns:
        HTML string with embedded JavaScript
    """
    return f"""
    <div style="display: flex; gap: 0.5rem; margin-top: 0.5rem;">
        <button 
            onclick="copyToClipboard()" 
            style="
                background-color: #3b82f6;
                color: white;
                border: none;
                padding: 0.5rem 1rem;
                border-radius: 6px;
                cursor: pointer;
                font-size: 0.875rem;
                display: flex;
                align-items: center;
                gap: 0.25rem;
            "
        >
            📋 Copy Minutes
        </button>
        <span id="copy-feedback" style="color: #22c55e; font-size: 0.875rem; display: none;">
            ✅ Copied!
        </span>
    </div>
    
    <script>
    function copyToClipboard() {{
        const minutesElement = document.querySelector('[data-testid="textbox"]');
        if (minutesElement) {{
            const text = minutesElement.value || minutesElement.textContent;
            navigator.clipboard.writeText(text).then(() => {{
                const feedback = document.getElementById('copy-feedback');
                feedback.style.display = 'inline';
                setTimeout(() => {{
                    feedback.style.display = 'none';
                }}, 2000);
            }}).catch(err => {{
                console.error('Failed to copy:', err);
                alert('Failed to copy to clipboard. Please select and copy manually.');
            }});
        }} else {{
            // Fallback: try to find by id
            const element = document.getElementById('{element_id}');
            if (element) {{
                const text = element.value || element.textContent;
                navigator.clipboard.writeText(text);
            }}
        }}
    }}
    </script>
    """


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
    1. Input Routing - audio or text (FR-1, FR-2)
    2. Transcription - Whisper-small (if audio) (FR-3)
    3. Summarization - BART-large-cnn with chunking (FR-4)
    4. Action Item Extraction - Flan-T5-small with prompt engineering (FR-5)
    5. Output Formatting - Markdown (FR-7)
    
    Includes confidence indicators per FR-9 and comprehensive error handling.
    
    Args:
        audio: Path to uploaded audio file or bytes (or None)
        transcript_text: Pasted transcript text (or empty string)
        meeting_type: Selected meeting type
        progress: Gradio progress tracker
        
    Returns:
        Tuple of (summary_markdown, action_items_dataframe, formatted_minutes, markdown_file_path)
    """
    import time
    start_time = time.time()
    
    # Track quality indicators
    confidence_warning = ""
    has_audio = False
    
    try:
        # =====================================================================
        # Stage 1: Input Validation & Routing (FR-1, FR-2)
        # =====================================================================
        progress(0.05, desc=get_processing_stage_message(ProcessingStage.INPUT_ROUTING))
        logger.info(f"Processing request - Meeting type: {meeting_type}")
        
        has_audio = audio is not None and audio != ""
        has_transcript = transcript_text and transcript_text.strip()
        
        # Validate audio input if provided (FR-1)
        audio_metadata = None
        if has_audio:
            audio_validation = validate_audio_input(audio)
            if not audio_validation.valid:
                gr.Warning(audio_validation.message)
                return (
                    "*Summary will appear here after processing...*",
                    [],
                    f"❌ **Input Error:** {audio_validation.message}",
                    None
                )
            # Show validation success
            gr.Info(audio_validation.message)
            if audio_validation.warning:
                gr.Warning(audio_validation.warning)
            audio_metadata = audio_validation.metadata
        
        # Validate text input if provided (FR-2)
        text_metadata = None
        if has_transcript:
            text_validation = validate_text_input(transcript_text)
            if not text_validation.valid:
                gr.Warning(text_validation.message)
                return (
                    "*Summary will appear here after processing...*",
                    [],
                    f"❌ **Input Error:** {text_validation.message}",
                    None
                )
            # Show validation success
            gr.Info(text_validation.message)
            if text_validation.warning:
                gr.Warning(text_validation.warning)
            text_metadata = text_validation.metadata
        
        # Check that at least one input is valid
        if not has_audio and not has_transcript:
            gr.Warning("Please upload audio or paste a transcript")
            return (
                "*Summary will appear here after processing...*",
                [],
                "No minutes generated. Please provide input.",
                None
            )
        
        # Show processing time estimate (FR-9.3)
        if has_audio and audio_metadata:
            duration = audio_metadata.get("duration_seconds", 60)
            time_estimate = get_processing_time_estimate("audio", int(duration))
            gr.Info(f"🎙️ {get_processing_stage_message(ProcessingStage.TRANSCRIPTION)} Estimated time: {time_estimate}")
        elif has_transcript and text_metadata:
            char_count = text_metadata.get("char_count", 1000)
            time_estimate = get_processing_time_estimate("text", char_count)
            gr.Info(f"📝 Processing transcript... Estimated time: {time_estimate}")
        
        # =====================================================================
        # Stage 2: Transcription (if audio provided) - FR-3
        # =====================================================================
        progress(0.1, desc=get_processing_stage_message(ProcessingStage.TRANSCRIPTION))
        
        full_transcript = ""
        
        if has_audio:
            logger.info("Processing audio input")
            
            def transcription_progress(pct: float, msg: str):
                progress(0.1 + (pct * 0.3), desc=f"🎤 {msg}")
            
            try:
                full_transcript = transcribe_audio(audio, transcription_progress)
            except Exception as e:
                logger.error(f"Transcription failed: {str(e)}")
                gr.Error("Processing failed. Try a shorter recording or clearer audio.")
                return (
                    "*Transcription failed*",
                    [],
                    f"Transcription failed: {str(e)}",
                    None
                )
        else:
            logger.info("Processing text input")
            full_transcript = transcript_text.strip()
        
        # Validate transcript length (FR-2.3)
        if not full_transcript or len(full_transcript.strip()) < 10:
            gr.Warning("Transcript is too short. Please provide more content.")
            return (
                "*Transcript too short*",
                [],
                "No minutes generated. Transcript must be at least 10 characters.",
                None
            )
        
        # Truncate if too long (FR-3.4)
        if len(full_transcript) > MAX_TRANSCRIPT_LENGTH:
            logger.warning(f"Transcript truncated from {len(full_transcript)} to {MAX_TRANSCRIPT_LENGTH} chars")
            gr.Info(f"📄 Long transcript detected. Processing first {MAX_TRANSCRIPT_LENGTH:,} characters.")
            full_transcript = full_transcript[:MAX_TRANSCRIPT_LENGTH]
        
        # Assess transcription quality (FR-9.3)
        quality_level, confidence_warning = assess_transcription_quality(full_transcript, has_audio)
        if confidence_warning:
            gr.Warning(confidence_warning)
        
        # =====================================================================
        # Stage 3: Summarization - FR-4
        # =====================================================================
        progress(0.45, desc=get_processing_stage_message(ProcessingStage.SUMMARIZATION))
        
        def summarization_progress(pct: float, msg: str):
            progress(0.45 + (pct * 0.2), desc=f"📝 {msg}")
        
        try:
            summary = generate_summary(full_transcript, summarization_progress)
        except Exception as e:
            logger.error(f"Summarization failed: {str(e)}")
            # Fallback to truncated transcript
            summary = full_transcript[:300] + "..." if len(full_transcript) > 300 else full_transcript
            gr.Warning("Summarization had issues. Using truncated transcript instead.")
        
        # =====================================================================
        # Stage 4: Action Item Extraction - FR-5
        # =====================================================================
        progress(0.7, desc=get_processing_stage_message(ProcessingStage.EXTRACTION))
        
        def extraction_progress(pct: float, msg: str):
            progress(0.7 + (pct * 0.15), desc=f"✅ {msg}")
        
        try:
            action_items = extract_action_items(full_transcript, extraction_progress)
        except Exception as e:
            logger.error(f"Action item extraction failed: {str(e)}")
            action_items = []
            gr.Warning("Action item extraction encountered issues.")
        
        # =====================================================================
        # Stage 5: Output Formatting - FR-7
        # =====================================================================
        progress(0.9, desc=get_processing_stage_message(ProcessingStage.FORMATTING))
        
        # Calculate processing time
        processing_time = time.time() - start_time
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Format outputs using PRD Section 5.1 functions
        
        # 1. Format summary for Markdown display (FR-7.1)
        summary_markdown = format_summary(summary, confidence_warning)
        
        # 2. Format action items for Dataframe (FR-5.4, FR-5.5)
        action_items_list = format_action_items(action_items)
        
        # Add tip if no action items found (FR-9.4)
        if not action_items:
            gr.Info(get_empty_action_items_tip())
        
        # 3. Format full minutes (FR-7.1, FR-7.2, FR-7.3)
        formatted_minutes = format_full_minutes(
            summary=summary,
            action_items=action_items,
            meeting_type=meeting_type,
            timestamp=timestamp,
            transcript=full_transcript,
            confidence_warning=confidence_warning,
            processing_time=processing_time,
        )
        
        # 4. Create downloadable file (FR-8)
        file_name = f"meeting_minutes_{meeting_type.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        temp_file_path = os.path.join(tempfile.gettempdir(), file_name)
        
        with open(temp_file_path, "w", encoding="utf-8") as f:
            f.write(formatted_minutes)
        
        # Log completion
        logger.info(f"Processing complete in {processing_time:.2f} seconds")
        
        progress(1.0, desc="Complete!")
        
        # Success message with timing (FR-9.3)
        if processing_time > 90:
            gr.Info(f"✅ Meeting minutes generated! ({processing_time:.1f}s) - Consider splitting long meetings for faster processing.")
        else:
            gr.Info(f"✅ Meeting minutes generated successfully! ({processing_time:.1f}s)")
        
        return summary_markdown, action_items_list, formatted_minutes, temp_file_path
        
    except TimeoutError:
        # Timeout handling per requirements
        error_msg = "Processing timed out"
        logger.error(error_msg)
        gr.Info("Processing took longer than expected. Consider splitting long meetings.")
        return (
            "*Processing timed out*",
            [],
            "Processing timed out. Please try a shorter recording.",
            None
        )
        
    except Exception as e:
        # Generic error handling per requirements
        error_msg = f"An error occurred: {str(e)}"
        logger.error(f"Error in process_meeting: {str(e)}", exc_info=True)
        gr.Error("Processing failed. Try a shorter recording or clearer audio.")
        return (
            f"*Error: {str(e)}*",
            [],
            f"Processing failed: {str(e)}",
            None
        )


# =============================================================================
# Gradio Interface - Based on PRD Section 8: UI/UX Specifications
# =============================================================================

# Example transcripts for quick testing (FR-9.4)
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

EXAMPLE_TRANSCRIPT_RETROSPECTIVE = """Sprint 14 Retrospective - March 2024
Duration: 45 minutes
Participants: Dev Team (5 members), Scrum Master

What went well:
- Completed 18 story points (target was 15)
- CI/CD pipeline improvements reduced deployment time by 40%
- Team collaboration on the authentication feature was excellent
- Daily standups were focused and efficient

What didn't go well:
- Story "User Profile Redesign" was blocked for 3 days waiting for design assets
- Two critical bugs found in production required hotfixes
- API documentation is still outdated
- Testing environment was unstable during the sprint

Action items:
1. Alex will set up a design review checkpoint at sprint planning to prevent future blocking issues - by next Monday
2. Jordan will create automated test suite for authentication module - by end of next sprint
3. Sarah will update API documentation and set up automated docs generation - by Friday
4. Mike will investigate testing environment stability issues and propose solutions - by Wednesday
5. Schedule mid-sprint check-ins to identify blockers earlier - ongoing

Team voted to focus on stability improvements for next sprint."""

# Extended example set with more variety
EXAMPLE_TRANSCRIPT_QUICK_SYNC = """Quick sync between Marketing and Engineering

Present: Rachel (Marketing), Tom (Engineering)

Rachel: The campaign launch is next Tuesday. How's the landing page coming along?

Tom: Almost done. I need the final copy and images by Thursday EOD to make the deadline.

Rachel: I'll have the copy to you by tomorrow. Images are being finalized by the design team today. Can you also add the tracking pixels we discussed?

Tom: Sure, send me the tracking requirements. I'll implement them along with the page. What about the A/B test variants?

Rachel: We'll need two variants. I'll send the specs. Also, who's handling the DNS configuration?

Tom: That's Jordan's area. He's back from vacation Monday. I'll make sure he prioritizes it.

Action items agreed:
- Rachel: Send copy by tomorrow, images by Thursday, tracking specs by Wednesday
- Tom: Implement tracking pixels, coordinate with Jordan on DNS - by Monday
- DNS configuration must be complete by Friday before launch"""

# Example data structure for gr.Examples component
# Audio sample paths (relative to app.py location)
SAMPLES_DIR = os.path.join(os.path.dirname(__file__), "samples") if __file__ else "samples"

EXAMPLE_MEETINGS = [
    # (audio_path, transcript_text, meeting_type)
    ("", EXAMPLE_TRANSCRIPT_STANDUP, "Standup", "Quick daily team standup with 3 action items"),
    ("", EXAMPLE_TRANSCRIPT_CLIENT, "Client Call", "Client progress review with 4 action items"),
    ("", EXAMPLE_TRANSCRIPT_BRAINSTORM, "Brainstorm", "Feature prioritization with voting results"),
    ("", EXAMPLE_TRANSCRIPT_RETROSPECTIVE, "Retrospective", "Sprint review with 5 improvement actions"),
    ("", EXAMPLE_TRANSCRIPT_QUICK_SYNC, "Other", "Cross-team coordination with clear deadlines"),
]

# Real audio examples with corresponding transcripts
AUDIO_EXAMPLES = [
    # (audio_path, transcript_text, meeting_type, description)
    (
        os.path.join(SAMPLES_DIR, "standup_sample.wav"),
        "Team standup for March 27th. Alex here. I'll complete the UI mockups by Friday. Jordan here. I agreed to update the project timeline document by next Monday. Sarah here. I will schedule the client demo for next week. Key decisions: Push release date to end of Q3, prioritize mobile responsiveness. Next meeting scheduled for Friday at 2 PM.",
        "Standup",
        "🎙️ Audio: Team standup with 3 action items (~40 sec)"
    ),
    (
        os.path.join(SAMPLES_DIR, "client_call_sample.wav"),
        "Client call with Acme Corporation regarding the enterprise software project. John from client team requested additional customization for the reporting module. Lisa asked about API integration timeline, and Mike confirmed it's on track for next sprint. Action items: Mike to prepare API documentation by Friday. Sarah to send status reports every Monday. Schedule next call for Thursday at 3 PM.",
        "Client Call",
        "🎙️ Audio: Client meeting with 3 action items (~45 sec)"
    ),
    (
        os.path.join(SAMPLES_DIR, "brainstorm_sample.wav"),
        "Brainstorming session for Q4 product features. Participants from Engineering, Design, Marketing, and Sales. Ideas discussed: AI powered recommendations engine with 8 votes for priority. Mobile app redesign with dark mode, 6 votes. Slack and Teams integration, 5 votes. Decisions: Engineering to create technical specs by next week. Design team to mock up mobile dark mode by Friday. Next session scheduled for same time next month.",
        "Brainstorm",
        "🎙️ Audio: Brainstorming session (~55 sec)"
    ),
    (
        os.path.join(SAMPLES_DIR, "retrospective_sample.wav"),
        "Sprint 14 Retrospective for March 2024. What went well: Completed 18 story points, CI CD pipeline improvements reduced deployment time by 40 percent. What didn't go well: User Profile story was blocked for 3 days, two critical bugs required hotfixes. Action items: Alex to set up design review checkpoint by Monday. Jordan to create automated test suite by end of sprint. Sarah to update API documentation by Friday.",
        "Retrospective",
        "🎙️ Audio: Sprint retrospective (~50 sec)"
    ),
    (
        os.path.join(SAMPLES_DIR, "quick_sync_sample.wav"),
        "Quick sync between Marketing and Engineering. Rachel from Marketing, Tom from Engineering. Campaign launch is next Tuesday. Tom needs final copy and images by Thursday EOD. Rachel will send copy by tomorrow, images by Thursday. Tom will implement tracking pixels and coordinate with Jordan on DNS configuration by Monday. DNS must be complete by Friday before launch.",
        "Other",
        "🎙️ Audio: Cross-team coordination (~50 sec)"
    ),
]


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

/* Validation Status */
.validation-status {
    background-color: #f0fdf4;
    border: 1px solid #bbf7d0;
    color: #166534;
    padding: 0.75rem;
    border-radius: 6px;
    font-size: 0.875rem;
    margin-top: 0.5rem;
}

.text-validation {
    background-color: #eff6ff;
    border: 1px solid #bfdbfe;
    color: #1e40af;
    padding: 0.5rem;
    border-radius: 6px;
    font-size: 0.85rem;
    margin-top: 0.25rem;
}

/* Tips Accordion */
.tips-accordion {
    margin-top: 1rem;
    border: 1px solid #e5e7eb;
    border-radius: 8px;
}

.tips-accordion h4 {
    margin: 0 0 0.5rem 0;
    color: #374151;
    font-size: 0.95rem;
}

/* Copy Button & Status */
.copy-button {
    background-color: #3b82f6 !important;
}

.copy-status {
    color: #22c55e;
    font-size: 0.875rem;
    padding: 0.25rem 0.5rem;
    background-color: #f0fdf4;
    border-radius: 4px;
}

/* Examples Section Styles */
.examples-section {
    margin-top: 0.5rem;
    padding: 0.5rem;
    background-color: #f9fafb;
    border-radius: 8px;
    border: 1px solid #e5e7eb;
}

.examples-section table {
    font-size: 0.875rem;
}

.examples-section tr:hover {
    background-color: #eff6ff;
    cursor: pointer;
}

/* Example Quick-Select Buttons */
.example-quick-btn {
    margin: 0.25rem;
    min-width: 120px;
}

/* Download Button Styling */
.download-button {
    margin-top: 0.5rem;
    padding: 0.5rem;
    background-color: #f0fdf4;
    border: 1px solid #bbf7d0;
    border-radius: 8px;
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
                        <strong>Audio is processed in-memory and NOT stored on our servers.</strong>
                        <br><br>
                        ⚠️ <strong>Do NOT upload:</strong>
                        <ul style="margin: 0.5rem 0; padding-left: 1.25rem;">
                            <li>Confidential business meetings</li>
                            <li>Personal or sensitive information</li>
                            <li>HR, legal, or financial discussions</li>
                        </ul>
                        <br>
                        💼 For enterprise use with sensitive data, consider <strong>self-hosting</strong>.
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
            
            # Validation Status Display
            validation_status = gr.Markdown(
                value="",
                elem_classes=["validation-status"],
                visible=False,
                show_label=False,
            )
            
            # Accepted formats hint
            gr.HTML(
                """
                <p style="font-size: 0.8rem; color: #6b7280; margin-top: 0.25rem;">
                    📁 <strong>Accepted formats:</strong> WAV, MP3, M4A, WebM | 
                    📏 <strong>Max size:</strong> 25MB | 
                    ⏱️ <strong>Max duration:</strong> 10 min (5 min recommended)
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
                placeholder="Paste meeting transcript here...\n\nExample:\nTeam standup - March 27\nAttendees: Alex, Jordan, Sarah\n\nAlex: I'll complete the UI mockups by Friday.\nJordan: Agreed to update the project timeline by next Monday.\nSarah: Will schedule the client demo for next week.",
                show_label=True,
                interactive=True,
                show_copy_button=True,
                elem_classes=["transcript-input"],
            )
            
            # Text Validation Status
            text_validation_status = gr.Markdown(
                value="",
                elem_classes=["text-validation"],
                visible=False,
                show_label=False,
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
            
            # Tips Section - Comprehensive Quality Tips
            with gr.Accordion("💡 Tips for Better Results", open=False, elem_classes=["tips-accordion"]):
                with gr.Tabs():
                    with gr.TabItem("🎤 Audio Tips"):
                        gr.HTML(
                            """
                            <div style="padding: 0.5rem;">
                            <h4>Recording Quality</h4>
                            <ul style="margin: 0.5rem 0; padding-left: 1.25rem;">
                                <li><strong>Quiet environment:</strong> Minimize background noise</li>
                                <li><strong>Clear speech:</strong> Speak at moderate pace</li>
                                <li><strong>Good microphone:</strong> Use headset or quality mic</li>
                                <li><strong>Short segments:</strong> Keep under 5 minutes</li>
                            </ul>
                            <p style="font-size: 0.85rem; color: #6b7280; margin-top: 0.5rem;">
                                📁 <strong>Formats:</strong> WAV, MP3, M4A, WebM | 
                                📏 <strong>Max:</strong> 25MB, 10 min
                            </p>
                            </div>
                            """
                        )
                    with gr.TabItem("📝 Transcript Tips"):
                        gr.HTML(
                            """
                            <div style="padding: 0.5rem;">
                            <h4>Transcript Structure</h4>
                            <ul style="margin: 0.5rem 0; padding-left: 1.25rem;">
                                <li><strong>Include context:</strong> Add speaker names</li>
                                <li><strong>Clear commitments:</strong> "I will...", "John agreed to..."</li>
                                <li><strong>Specific dates:</strong> "by Friday", "next Monday"</li>
                                <li><strong>Minimum length:</strong> At least 50 words</li>
                            </ul>
                            </div>
                            """
                        )
                    with gr.TabItem("🎯 Meeting Tips"):
                        gr.HTML(
                            """
                            <div style="padding: 0.5rem;">
                            <h4>For Better Extraction</h4>
                            <ul style="margin: 0.5rem 0; padding-left: 1.25rem;">
                                <li><strong>State names:</strong> "Sarah will handle..."</li>
                                <li><strong>Confirm deadlines:</strong> "Complete by EOD Friday"</li>
                                <li><strong>Summarize decisions:</strong> End with recap</li>
                                <li><strong>Split long meetings:</strong> Process in chunks</li>
                            </ul>
                            </div>
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
                elem_id="minutes-output",
                placeholder="Full formatted meeting minutes will appear here...",
            )
            
            # Copy & Download Row
            with gr.Row():
                copy_btn = gr.Button(
                    "📋 Copy to Clipboard",
                    size="sm",
                    variant="secondary",
                    elem_classes=["copy-button"],
                )
                copy_status = gr.Markdown("", visible=False, elem_classes=["copy-status"])
            
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
    # Examples Section (FR-9.4)
    # =========================================================================
    with gr.Row():
        with gr.Column():
            gr.HTML(
                """
                <div class="section-header" style="margin-top: 1rem;">
                    <span>🧪</span> Example Gallery
                </div>
                <p style="font-size: 0.875rem; color: #6b7280; margin-bottom: 1rem;">
                    Click any example to auto-fill the input and try the generator. Each example showcases different meeting types and action item patterns.
                </p>
                """
            )
            
            # Enhanced examples with descriptions
            with gr.Tabs():
                with gr.TabItem("🎙️ Audio Examples"):
                    gr.Markdown(
                        """
                        **🎤 Real Audio Recordings** - Click any example to test audio transcription:
                        
                        These are pre-recorded meeting samples that demonstrate the audio transcription pipeline.
                        Each sample is 40-55 seconds long and showcases different meeting types.
                        """
                    )
                    gr.Examples(
                        examples=[
                            [audio_path, transcript, meeting_type]
                            for audio_path, transcript, meeting_type, description in AUDIO_EXAMPLES
                        ],
                        inputs=[audio_input, transcript_input, meeting_type],
                        label="🎙️ Select an audio example:",
                        elem_classes=["examples-section"],
                        examples_per_page=5,
                    )
                
                with gr.TabItem("📝 Text Examples"):
                    gr.Examples(
                        examples=[
                            ["", EXAMPLE_TRANSCRIPT_STANDUP, "Standup"],
                            ["", EXAMPLE_TRANSCRIPT_CLIENT, "Client Call"],
                            ["", EXAMPLE_TRANSCRIPT_BRAINSTORM, "Brainstorm"],
                            ["", EXAMPLE_TRANSCRIPT_RETROSPECTIVE, "Retrospective"],
                            ["", EXAMPLE_TRANSCRIPT_QUICK_SYNC, "Other"],
                        ],
                        inputs=[audio_input, transcript_input, meeting_type],
                        label="Select an example to load:",
                        elem_classes=["examples-section"],
                        examples_per_page=5,
                    )
                
                with gr.TabItem("🎯 By Meeting Type"):
                    gr.Markdown(
                        """
                        **Available Example Types:**
                        
                        | Type | Audio | Text | Description |
                        |------|-------|------|-------------|
                        | 🏃 **Standup** | ✅ 40 sec | ✅ | Quick team sync with progress updates |
                        | 📞 **Client Call** | ✅ 45 sec | ✅ | External meeting with deliverables |
                        | 💡 **Brainstorm** | ✅ 55 sec | ✅ | Creative session with voting |
                        | 🔄 **Retrospective** | ✅ 50 sec | ✅ | Sprint review with improvements |
                        | 🤝 **Quick Sync** | ✅ 50 sec | ✅ | Cross-team coordination |
                        
                        🎙️ **Audio examples** demonstrate the transcription pipeline with real recordings.
                        📝 **Text examples** allow instant processing without audio upload.
                        """
                    )
                    
                    # Quick-select buttons for each type
                    with gr.Row():
                        example_standup_btn = gr.Button("🏃 Standup", size="sm")
                        example_client_btn = gr.Button("📞 Client Call", size="sm")
                        example_brainstorm_btn = gr.Button("💡 Brainstorm", size="sm")
                    
                    with gr.Row():
                        example_retro_btn = gr.Button("🔄 Retrospective", size="sm")
                        example_sync_btn = gr.Button("🤝 Quick Sync", size="sm")
    
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
    
    # Audio upload handler - validate and check duration
    def on_audio_change(audio_path: Optional[str]) -> Tuple[gr.update, gr.update]:
        """Handle audio file upload with validation."""
        if audio_path:
            # Use comprehensive validation
            validation = validate_audio_input(audio_path)
            
            if validation.valid:
                # Show success message
                message = validation.message
                if validation.warning:
                    message += f"\n\n{validation.warning}"
                return (
                    gr.update(value=message, visible=True),
                    gr.update(value="", visible=False),
                )
            else:
                # Show error
                return (
                    gr.update(value=f"❌ {validation.message}", visible=True),
                    gr.update(value="", visible=False),
                )
        return gr.update(value="", visible=False), gr.update(value="", visible=False)
    
    audio_input.change(
        fn=on_audio_change,
        inputs=[audio_input],
        outputs=[duration_status, validation_status],
    )
    
    # Text input handler - validate transcript
    def on_text_change(text: Optional[str]) -> gr.update:
        """Handle text input with validation."""
        if text and text.strip():
            validation = validate_text_input(text)
            
            if validation.valid:
                message = validation.message
                if validation.warning:
                    message += f"\n\n{validation.warning}"
                return gr.update(value=message, visible=True)
            else:
                return gr.update(value=f"❌ {validation.message}", visible=True)
        return gr.update(value="", visible=False)
    
    transcript_input.change(
        fn=on_text_change,
        inputs=[transcript_input],
        outputs=[text_validation_status],
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
    
    # =========================================================================
    # Copy to Clipboard Handler
    # =========================================================================
    def copy_minutes_to_clipboard(minutes_text: str) -> gr.update:
        """Copy minutes text to clipboard (triggered by button click)."""
        if minutes_text and minutes_text.strip():
            # The actual copy happens via JavaScript in the browser
            # This function provides feedback to the user
            return gr.update(value="✅ Copied to clipboard!", visible=True)
        return gr.update(value="❌ Nothing to copy", visible=True)
    
    copy_btn.click(
        fn=copy_minutes_to_clipboard,
        inputs=[minutes_output],
        outputs=[copy_status],
        js="""
        () => {
            const textbox = document.querySelector('[data-testid="textbox"]');
            if (textbox) {
                const text = textbox.value || textbox.textContent;
                navigator.clipboard.writeText(text).then(() => {
                    console.log('Copied to clipboard');
                }).catch(err => {
                    console.error('Failed to copy:', err);
                    alert('Failed to copy. Please use Ctrl+C to copy manually.');
                });
            }
        }
        """,
    )
    
    # =========================================================================
    # Example Quick-Select Button Handlers
    # =========================================================================
    def load_example_standup():
        return gr.update(value=EXAMPLE_TRANSCRIPT_STANDUP), gr.update(value="Standup")
    
    def load_example_client():
        return gr.update(value=EXAMPLE_TRANSCRIPT_CLIENT), gr.update(value="Client Call")
    
    def load_example_brainstorm():
        return gr.update(value=EXAMPLE_TRANSCRIPT_BRAINSTORM), gr.update(value="Brainstorm")
    
    def load_example_retro():
        return gr.update(value=EXAMPLE_TRANSCRIPT_RETROSPECTIVE), gr.update(value="Retrospective")
    
    def load_example_sync():
        return gr.update(value=EXAMPLE_TRANSCRIPT_QUICK_SYNC), gr.update(value="Other")
    
    example_standup_btn.click(
        fn=load_example_standup,
        outputs=[transcript_input, meeting_type],
    )
    
    example_client_btn.click(
        fn=load_example_client,
        outputs=[transcript_input, meeting_type],
    )
    
    example_brainstorm_btn.click(
        fn=load_example_brainstorm,
        outputs=[transcript_input, meeting_type],
    )
    
    example_retro_btn.click(
        fn=load_example_retro,
        outputs=[transcript_input, meeting_type],
    )
    
    example_sync_btn.click(
        fn=load_example_sync,
        outputs=[transcript_input, meeting_type],
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


# =============================================================================
# Health Check API Endpoint - NFR-2 Monitoring
# =============================================================================

def get_health_status() -> Dict[str, Any]:
    """
    Get health status for monitoring.
    Used by Hugging Face Spaces health checks.
    
    Returns:
        Dict with health status information
    """
    return {
        "status": "healthy" if models.is_loaded else "degraded",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "models": {
            "transcriber": models.transcriber is not None or models.mock_mode,
            "summarizer": models.summarizer is not None or models.mock_mode,
            "extractor": models.extractor is not None or models.mock_mode,
        },
        "mock_mode": models.mock_mode,
        "device": DEVICE,
        "usage_stats": usage_stats.to_dict(),
    }


if __name__ == "__main__":
    # Initialize and launch
    initialize_app()
    
    logger.info("Launching Gradio interface...")
    
    # Gradio concurrency settings for Hugging Face Spaces
    # Limits concurrent requests for CPU tier optimization
    demo.launch(
        share=False,
        show_error=True,
        quiet=False,
        max_threads=4,  # Limit concurrent threads for CPU
        concurrency_count=2,  # Allow 2 concurrent requests max
        server_name="0.0.0.0",
        server_port=7860,
    )
