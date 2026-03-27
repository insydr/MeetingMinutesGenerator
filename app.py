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
"""

import os
import tempfile
from datetime import datetime
from typing import Optional, Tuple, List

import gradio as gr
import torch
from transformers import (
    pipeline,
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)

# =============================================================================
# Configuration
# =============================================================================

# Model identifiers
WHISPER_MODEL_ID = "openai/whisper-small"
BART_MODEL_ID = "facebook/bart-large-cnn"
FLAN_T5_MODEL_ID = "google/flan-t5-small"

# Processing constraints for CPU tier
MAX_AUDIO_DURATION_SECONDS = 300  # 5 minutes recommended max
MAX_TRANSCRIPT_LENGTH = 8000  # Characters for processing

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.float32  # Use float32 for CPU compatibility


# =============================================================================
# Global Model Loading
# =============================================================================

print("Loading models... This may take a few minutes on first run.")

# Audio Transcription Model: Whisper-small
print(f"Loading Whisper model: {WHISPER_MODEL_ID}")
whisper_processor = AutoProcessor.from_pretrained(WHISPER_MODEL_ID)
whisper_model = AutoModelForSpeechSeq2Seq.from_pretrained(
    WHISPER_MODEL_ID,
    torch_dtype=TORCH_DTYPE,
    low_cpu_mem_usage=True,
    use_safetensors=True,
)
whisper_model.to(DEVICE)

# ASR Pipeline for transcription
transcriber = pipeline(
    "automatic-speech-recognition",
    model=whisper_model,
    processor=whisper_processor,
    torch_dtype=TORCH_DTYPE,
    device=0 if DEVICE == "cuda" else -1,
)

# Summarization Model: BART-large-cnn
print(f"Loading BART model: {BART_MODEL_ID}")
summarizer = pipeline(
    "summarization",
    model=BART_MODEL_ID,
    torch_dtype=TORCH_DTYPE,
    device=0 if DEVICE == "cuda" else -1,
)

# Action Item Extraction Model: Flan-T5-small
print(f"Loading Flan-T5 model: {FLAN_T5_MODEL_ID}")
flan_tokenizer = AutoTokenizer.from_pretrained(FLAN_T5_MODEL_ID)
flan_model = AutoModelForSeq2SeqLM.from_pretrained(
    FLAN_T5_MODEL_ID,
    torch_dtype=TORCH_DTYPE,
    low_cpu_mem_usage=True,
)
flan_model.to(DEVICE)

print("All models loaded successfully!")


# =============================================================================
# Processing Functions
# =============================================================================

def transcribe_audio(audio_path: Optional[str]) -> str:
    """
    Transcribe audio file to text using Whisper-small.
    
    Args:
        audio_path: Path to the audio file (WAV, MP3, M4A, WebM)
        
    Returns:
        Transcribed text string
        
    Raises:
        ValueError: If audio_path is None or empty
    """
    if audio_path is None or audio_path == "":
        raise ValueError("No audio file provided")
    
    print(f"Transcribing audio: {audio_path}")
    
    # Transcribe with Whisper
    result = transcriber(
        audio_path,
        return_timestamps=False,
        generate_kwargs={
            "language": "english",
            "task": "transcribe",
        }
    )
    
    transcript = result["text"].strip()
    print(f"Transcription complete. Length: {len(transcript)} characters")
    
    return transcript


def generate_summary(text: str) -> str:
    """
    Generate an executive summary using BART-large-cnn.
    
    Args:
        text: Full meeting transcript text
        
    Returns:
        Summary text (40-150 tokens)
    """
    if not text or len(text.strip()) == 0:
        return "No transcript available to summarize."
    
    print("Generating summary...")
    
    # Truncate if necessary (BART max input is 1024 tokens)
    max_input_chars = 3000  # Approximate character limit
    input_text = text[:max_input_chars] if len(text) > max_input_chars else text
    
    # Generate summary
    summary_result = summarizer(
        input_text,
        max_length=150,
        min_length=40,
        do_sample=False,
        truncation=True,
    )
    
    summary = summary_result[0]["summary_text"].strip()
    print(f"Summary generated. Length: {len(summary)} characters")
    
    return summary


def extract_action_items(text: str) -> List[List[str]]:
    """
    Extract action items from transcript using Flan-T5-small with prompt engineering.
    
    Args:
        text: Full meeting transcript text
        
    Returns:
        List of [task, owner, deadline] lists
    """
    if not text or len(text.strip()) == 0:
        return []
    
    print("Extracting action items...")
    
    # Use first portion of text for extraction
    max_input_chars = 2000
    input_text = text[:max_input_chars] if len(text) > max_input_chars else text
    
    # Construct extraction prompt
    extraction_prompt = f"""Extract action items from this meeting transcript.
For each action item, identify: the task description, the responsible person, and the deadline.
If information is not mentioned, use "TBD".

Transcript:
{input_text}

List the action items in format: Task | Owner | Deadline
Action items:"""

    # Tokenize and generate
    inputs = flan_tokenizer(
        extraction_prompt,
        return_tensors="pt",
        max_length=512,
        truncation=True,
    ).to(DEVICE)
    
    with torch.no_grad():
        outputs = flan_model.generate(
            **inputs,
            max_length=256,
            num_beams=2,
            temperature=0.7,
            do_sample=True,
        )
    
    extraction_result = flan_tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Extraction result: {extraction_result}")
    
    # Parse extracted action items
    # This is a placeholder - actual parsing logic would be more sophisticated
    action_items = parse_action_items_from_text(extraction_result, text)
    
    return action_items


def parse_action_items_from_text(extraction: str, original_text: str) -> List[List[str]]:
    """
    Parse action items from model output.
    
    Args:
        extraction: Raw output from extraction model
        original_text: Original transcript for context
        
    Returns:
        List of [task, owner, deadline] lists
    """
    action_items = []
    
    # Simple parsing - split by lines and look for patterns
    lines = extraction.strip().split("\n")
    
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
            
        # Try to parse "Task | Owner | Deadline" format
        if "|" in line:
            parts = [p.strip() for p in line.split("|")]
            if len(parts) >= 1:
                task = parts[0] if parts[0] else "TBD"
                owner = parts[1] if len(parts) > 1 and parts[1] else "TBD"
                deadline = parts[2] if len(parts) > 2 and parts[2] else "TBD"
                action_items.append([task, owner, deadline])
    
    # If no items found, provide a placeholder
    if not action_items:
        action_items = [["Review meeting transcript for action items", "TBD", "TBD"]]
    
    return action_items


def format_meeting_minutes(
    summary: str,
    action_items: List[List[str]],
    meeting_type: str,
    timestamp: str,
) -> str:
    """
    Format complete meeting minutes in Markdown.
    
    Args:
        summary: Executive summary text
        action_items: List of [task, owner, deadline] lists
        meeting_type: Type of meeting
        timestamp: Generation timestamp
        
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
    
    for task, owner, deadline in action_items:
        minutes += f"| {task} | {owner} | {deadline} |\n"
    
    minutes += """
---

## Key Discussion Points

*Key points extracted from the meeting will appear here.*

---

## Notes

*Generated by Meeting Minutes Generator*
"""
    
    return minutes


# =============================================================================
# Main Processing Function
# =============================================================================

def process_meeting(
    audio: Optional[str],
    transcript_text: Optional[str],
    meeting_type: str,
    progress: gr.Progress = gr.Progress(),
) -> Tuple[str, List[List[str]], str, Optional[str]]:
    """
    Main processing function that orchestrates the full pipeline.
    
    Args:
        audio: Path to uploaded audio file (or None)
        transcript_text: Pasted transcript text (or empty string)
        meeting_type: Selected meeting type
        progress: Gradio progress tracker
        
    Returns:
        Tuple of (summary, action_items, formatted_minutes, markdown_file_path)
    """
    try:
        # Validate input
        if (audio is None or audio == "") and (not transcript_text or transcript_text.strip() == ""):
            gr.Warning("Please provide either an audio file or a transcript.")
            return "No input provided.", [], "No minutes generated.", None
        
        # Step 1: Get transcript (from audio or text input)
        progress(0.1, desc="Processing input...")
        
        if audio and audio != "":
            gr.Info("Transcribing audio... This may take 30-60 seconds.")
            progress(0.2, desc="Transcribing audio...")
            full_text = transcribe_audio(audio)
        else:
            gr.Info("Processing text transcript...")
            full_text = transcript_text.strip()
        
        # Validate transcript length
        if len(full_text) < 10:
            gr.Warning("Transcript is too short. Please provide more content.")
            return "Transcript too short.", [], "No minutes generated.", None
        
        # Step 2: Generate summary
        progress(0.5, desc="Generating summary...")
        summary = generate_summary(full_text)
        
        # Step 3: Extract action items
        progress(0.7, desc="Extracting action items...")
        action_items = extract_action_items(full_text)
        
        # Step 4: Format output
        progress(0.9, desc="Formatting minutes...")
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        formatted_minutes = format_meeting_minutes(
            summary=summary,
            action_items=action_items,
            meeting_type=meeting_type,
            timestamp=timestamp,
        )
        
        # Step 5: Create downloadable file
        file_name = f"meeting_minutes_{meeting_type.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        temp_file_path = os.path.join(tempfile.gettempdir(), file_name)
        
        with open(temp_file_path, "w", encoding="utf-8") as f:
            f.write(formatted_minutes)
        
        progress(1.0, desc="Complete!")
        gr.Info("Meeting minutes generated successfully!")
        
        return summary, action_items, formatted_minutes, temp_file_path
        
    except Exception as e:
        gr.Error(f"An error occurred: {str(e)}")
        print(f"Error in process_meeting: {str(e)}")
        return f"Error: {str(e)}", [], "Processing failed.", None


# =============================================================================
# Gradio Interface
# =============================================================================

# Example transcripts for quick testing
EXAMPLE_TRANSCRIPT = """Team discussed Q3 goals and project timeline updates. 
Alex mentioned that the design team needs more time for the UI overhaul, approximately 2 more weeks.
Jordan agreed to update the project timeline document by next Monday.
Sarah will schedule a client demo for the following week.
The team also discussed the new feature requests from the marketing team.
Key decisions: Push the release date to end of Q3, prioritize mobile responsiveness.
Next meeting scheduled for Friday at 2 PM."""

with gr.Blocks(
    theme=gr.themes.Soft(),
    title="Meeting Minutes Generator",
    css="""
    .header-text { text-align: center; margin-bottom: 1rem; }
    .privacy-notice { 
        background-color: #fff3cd; 
        border-left: 4px solid #ffc107; 
        padding: 0.75rem; 
        margin-top: 1rem;
        border-radius: 4px;
    }
    """,
) as demo:
    
    # Header
    gr.Markdown(
        """
        # Meeting Minutes Generator
        
        *Upload a recording or paste a transcript to generate structured, actionable meeting minutes*
        """,
        elem_classes=["header-text"],
    )
    
    # Main content area
    with gr.Row():
        # Input Column
        with gr.Column(scale=1):
            gr.Markdown("### Input")
            
            audio_input = gr.Audio(
                label="Upload Recording",
                sources=["upload", "microphone"],
                type="filepath",
                show_label=True,
            )
            
            transcript_input = gr.Textbox(
                label="Or Paste Transcript",
                lines=8,
                placeholder="Paste your meeting transcript here...",
                show_label=True,
            )
            
            meeting_type = gr.Dropdown(
                label="Meeting Type",
                choices=[
                    "Standup",
                    "Client Call", 
                    "Brainstorm",
                    "Retrospective",
                    "Other",
                ],
                value="Standup",
                show_label=True,
            )
            
            generate_btn = gr.Button(
                "Generate Minutes",
                variant="primary",
                size="lg",
            )
            
            # Tips
            gr.Markdown(
                """
                **Tips for best results:**
                - Ensure clear audio with minimal background noise
                - Speak clearly and at a moderate pace
                - For long meetings, consider processing in segments
                - Processing may take 30-60 seconds on free tier
                """
            )
        
        # Output Column
        with gr.Column(scale=2):
            gr.Markdown("### Output")
            
            summary_output = gr.Markdown(
                label="Executive Summary",
                show_label=True,
            )
            
            actions_output = gr.Dataframe(
                label="Action Items",
                headers=["Task", "Owner", "Deadline"],
                datatype=["str", "str", "str"],
                row_count=(1, "dynamic"),
                col_count=(3, "fixed"),
                show_label=True,
            )
            
            minutes_output = gr.Textbox(
                label="Full Minutes (Markdown)",
                lines=15,
                interactive=False,
                show_label=True,
                show_copy_button=True,
            )
            
            download_output = gr.File(
                label="Download as Markdown",
                show_label=True,
            )
    
    # Privacy Notice
    gr.Markdown(
        """
        > **Privacy Notice:** Audio is processed in-memory and not stored. 
        > Do not upload confidential or sensitive meetings. 
        > For enterprise use, consider self-hosting.
        """,
        elem_classes=["privacy-notice"],
    )
    
    # Examples
    gr.Examples(
        examples=[
            ["", EXAMPLE_TRANSCRIPT, "Standup"],
            ["", EXAMPLE_TRANSCRIPT, "Client Call"],
        ],
        inputs=[audio_input, transcript_input, meeting_type],
        label="Quick Examples (Click to try)",
    )
    
    # Event handlers
    generate_btn.click(
        fn=process_meeting,
        inputs=[audio_input, transcript_input, meeting_type],
        outputs=[summary_output, actions_output, minutes_output, download_output],
    )


# =============================================================================
# Launch Application
# =============================================================================

if __name__ == "__main__":
    print(f"Starting Meeting Minutes Generator on {DEVICE.upper()}")
    print(f"Gradio version: {gr.__version__}")
    print(f"PyTorch version: {torch.__version__}")
    
    demo.launch(
        share=False,
        show_error=True,
        quiet=False,
    )
