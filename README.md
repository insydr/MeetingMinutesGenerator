# Meeting Minutes Generator

[![Gradio](https://img.shields.io/badge/Gradio-5.0+-orange.svg)](https://gradio.app/)
[![Hugging Face Spaces](https://img.shields.io/badge/🤗%20Hugging%20Face-Spaces-yellow.svg)](https://huggingface.co/spaces)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **AI-powered meeting documentation that saves you hours of manual work**

## Overview

The **Meeting Minutes Generator** is an AI-powered web application designed to transform the way professionals document and track meeting outcomes. By leveraging state-of-the-art automatic speech recognition (ASR) and natural language processing (NLP) models, this application automatically:

- 📝 **Transcribes** audio recordings using OpenAI Whisper
- 📋 **Summarizes** key discussion points with BART
- ✅ **Extracts** action items, owners, and deadlines
- 📄 **Generates** professionally formatted meeting minutes

This application addresses a critical pain point in modern workplace productivity: the time-consuming and often inconsistent process of manual meeting documentation. Studies indicate that professionals spend an average of 4-6 hours per week on meeting-related administrative tasks, with minute-taking being one of the most tedious activities. Our solution automates this process, potentially saving organizations thousands of productive hours annually while improving documentation quality and action item tracking.

---

## Features

### Core Capabilities

| Feature | Description |
|---------|-------------|
| **Audio Transcription** | Upload WAV, MP3, M4A, or WebM recordings for automatic transcription using OpenAI Whisper-small |
| **Text Input** | Paste existing transcripts for processing without audio |
| **Smart Summarization** | Generate concise executive summaries highlighting key discussion points |
| **Action Item Extraction** | Automatically identify tasks, responsible parties, and deadlines |
| **Multiple Meeting Types** | Support for Standups, Client Calls, Brainstorms, Retrospectives, and more |
| **Markdown Export** | Download formatted minutes as Markdown files |
| **Professional Templates** | Structured output suitable for immediate sharing |

### User Experience

- 🎤 Upload recordings or record directly via microphone
- 📝 Paste transcripts as an alternative input method
- 📊 View action items in a structured table format
- 💾 Download minutes as Markdown for easy sharing
- ⚡ Progress indicators during processing
- 💡 Tips and guidance for optimal results

---

## How to Use

### Step-by-Step Guide

1. **Provide Input**
   - Upload an audio recording (WAV, MP3, M4A, WebM), OR
   - Paste an existing transcript into the text area

2. **Select Meeting Type**
   - Choose from: Standup, Client Call, Brainstorm, Retrospective, or Other
   - This helps format the output appropriately

3. **Generate Minutes**
   - Click the "Generate Minutes" button
   - Wait 30-60 seconds for processing (longer for longer recordings)

4. **Review Output**
   - Executive Summary: High-level overview of discussion
   - Action Items: Structured table of tasks, owners, and deadlines
   - Full Minutes: Complete formatted document

5. **Export**
   - Copy text directly from the output
   - Download as Markdown file for sharing or archiving

### Tips for Best Results

- ✅ Ensure clear audio with minimal background noise
- ✅ Speak clearly and at a moderate pace
- ✅ For long meetings (>5 minutes), consider processing in segments
- ✅ Mention names, deadlines, and commitments explicitly for better extraction

---

## Technical Details

### Architecture

The application follows a modular pipeline architecture:

```
Audio/Text Input → Transcription → Summarization → Extraction → Formatting → Output
```

### Models

| Component | Model | Parameters | Purpose |
|-----------|-------|------------|---------|
| **ASR** | `openai/whisper-small` | 244M | Audio-to-text transcription |
| **Summarization** | `facebook/bart-large-cnn` | 406M | Generate executive summaries |
| **Extraction** | `google/flan-t5-small` | 60M | Extract action items |

### Framework

- **UI Framework**: Gradio 5.x with `gr.themes.Soft()`
- **Hosting**: Hugging Face Spaces (CPU tier)
- **Python**: 3.10+
- **PyTorch**: 2.2.0+

### Performance

| Metric | Target |
|--------|--------|
| 1-minute audio transcription | < 60 seconds |
| Summarization | < 15 seconds |
| Total end-to-end processing | < 120 seconds |
| Max recommended audio duration | 5 minutes |

---

## Privacy Notice

> ⚠️ **Important Privacy Information**

- **No Persistent Storage**: Audio and transcripts are processed in-memory only and are not stored on our servers
- **Third-Party Models**: Your content is processed by pre-trained AI models; do not upload confidential or sensitive meetings
- **Enterprise Use**: For organizations handling sensitive data, we recommend self-hosting this application on your own infrastructure
- **User Responsibility**: Users are responsible for ensuring they have appropriate permissions to process and share meeting content

**Best Practices:**
- Review generated minutes before sharing externally
- Do not upload meetings containing confidential business information, personal data, or sensitive discussions
- For internal team use with standard meeting types

---

## Installation

### Local Development

```bash
# Clone the repository
git clone https://github.com/insydr/MeetingMinutesGenerator.git
cd MeetingMinutesGenerator

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py
```

### Hugging Face Spaces Deployment

1. Create a new Space on Hugging Face
2. Select "Gradio" as the SDK
3. Upload all files from this repository
4. The Space will automatically build and deploy

---

## Project Structure

```
MeetingMinutesGenerator/
├── app.py                 # Main Gradio application
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── .gitignore            # Git ignore rules
└── docs/
    └── PRD.md            # Product Requirements Document
```

---

## Roadmap

### Current (v1.0)
- [x] Audio transcription with Whisper
- [x] Text summarization with BART
- [x] Action item extraction
- [x] Markdown output generation
- [x] Hugging Face Spaces deployment

### Planned Features
- [ ] Speaker diarization (identify who said what)
- [ ] Multi-language support (100+ languages via Whisper)
- [ ] Custom output templates
- [ ] Calendar integration (Google Calendar, Notion)
- [ ] Batch processing for multiple meetings

---

## Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'feat: add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- [OpenAI Whisper](https://github.com/openai/whisper) for speech recognition
- [Facebook BART](https://huggingface.co/facebook/bart-large-cnn) for summarization
- [Google Flan-T5](https://huggingface.co/google/flan-t5-small) for instruction following
- [Gradio](https://gradio.app/) for the web interface framework
- [Hugging Face](https://huggingface.co/) for model hosting and Spaces platform

---

## Support

For questions, issues, or feature requests:
- Open an issue on [GitHub](https://github.com/insydr/MeetingMinutesGenerator/issues)
- Check the [documentation](./docs/PRD.md) for detailed specifications

---

*Built with ❤️ using Gradio and Hugging Face Transformers*
