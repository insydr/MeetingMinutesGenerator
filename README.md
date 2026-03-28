---
title: Meeting Minutes Generator
emoji: 🎙️
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 5.30.0
app_file: app.py
pinned: false
license: mit
short_description: AI-powered meeting transcription and minutes generation
python_version: 3.13
tags:
  - audio
  - transcription
  - summarization
  - whisper
  - bart
  - flan-t5
  - meeting-minutes
  - action-items
models:
  - openai/whisper-small
  - facebook/bart-large-cnn
  - google/flan-t5-small
---

# 🎙️ Meeting Minutes Generator

[![Gradio](https://img.shields.io/badge/Gradio-5.0+-orange.svg)](https://gradio.app/)
[![Hugging Face Spaces](https://img.shields.io/badge/🤗%20Hugging%20Face-Spaces-yellow.svg)](https://huggingface.co/spaces)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Deploy](https://img.shields.io/badge/Deploy-on%20Hugging%20Face-blue?logo=huggingface)](https://huggingface.co/new-space?template=insydr/MeetingMinutesGenerator)

> **AI-powered meeting documentation that saves you hours of manual work**

---

## Overview

The **Meeting Minutes Generator** is an AI-powered web application designed to transform the way professionals document and track meeting outcomes. By leveraging state-of-the-art automatic speech recognition (ASR) and natural language processing (NLP) models, this application automatically transcribes audio recordings, generates concise summaries, extracts action items with assigned owners and deadlines, and produces professionally formatted meeting minutes documents.

This application addresses a critical pain point in modern workplace productivity: the time-consuming and often inconsistent process of manual meeting documentation. Studies indicate that professionals spend an average of 4-6 hours per week on meeting-related administrative tasks, with minute-taking being one of the most tedious activities. Our solution automates this process, potentially saving organizations thousands of productive hours annually while improving documentation quality and action item tracking.

Built on the Hugging Face ecosystem and deployed via Hugging Face Spaces, the application demonstrates practical machine learning engineering while remaining accessible on free CPU tiers. The multi-modal pipeline showcases end-to-end ML engineering skills, from audio processing to structured output generation, making it both a valuable productivity tool and an impressive portfolio piece.

---

## ✨ Features

### Core Capabilities

| Feature | Description |
|---------|-------------|
| 🎤 **Audio Transcription** | Upload WAV, MP3, M4A, or WebM recordings for automatic transcription using OpenAI Whisper-small |
| 📝 **Text Input** | Paste existing transcripts for processing without audio |
| 📋 **Smart Summarization** | Generate concise executive summaries highlighting key discussion points |
| ✅ **Action Item Extraction** | Automatically identify tasks, responsible parties, and deadlines |
| 🏷️ **Multiple Meeting Types** | Support for Standups, Client Calls, Brainstorms, Retrospectives, and more |
| 📄 **Markdown Export** | Download formatted minutes as Markdown files for easy sharing |
| 🎨 **Professional Templates** | Structured output suitable for immediate sharing with stakeholders |

### User Experience Features

- 🎧 Upload recordings or record directly via browser microphone
- 📊 View action items in a structured table format
- 📋 Copy-to-clipboard functionality for quick sharing
- ⚡ Real-time progress indicators during processing
- 💡 Contextual tips for improving audio quality and transcription accuracy
- 🎯 Example gallery with both text and audio samples for quick testing
- 📱 Mobile-responsive design for on-the-go usage

---

## 🚀 How to Use

### Step-by-Step Guide

#### Step 1: Provide Input

Choose one of two input methods:

**Option A: Upload Audio**
- Click on the "Upload Recording" component
- Select an audio file (WAV, MP3, M4A, or WebM format)
- Maximum file size: 25MB
- Recommended duration: Under 5 minutes for optimal processing

**Option B: Paste Transcript**
- If you already have a transcript, paste it directly into the text area
- Minimum recommended: 50 words for meaningful summaries
- Maximum: 10,000 characters (will be truncated if longer)

> 📸 *Screenshot placeholder: Input section showing audio upload and text input areas*

#### Step 2: Select Meeting Type

Choose the appropriate meeting type from the dropdown:

| Meeting Type | Best For |
|--------------|----------|
| **Standup** | Daily team syncs, status updates |
| **Client Call** | Customer meetings, sales calls |
| **Brainstorm** | Creative sessions, idea generation |
| **Retrospective** | Sprint reviews, project post-mortems |
| **Other** | General meetings, interviews |

The meeting type helps format the output appropriately and influences how action items are prioritized.

#### Step 3: Generate Minutes

Click the **"✨ Generate Minutes"** button. The application will:

1. **Validate** your input (audio format, duration, text length)
2. **Transcribe** audio if provided (~30-60 seconds per minute of audio)
3. **Summarize** key discussion points (~10-15 seconds)
4. **Extract** action items with owners and deadlines (~5-10 seconds)
5. **Format** everything into professional meeting minutes

> 📸 *Screenshot placeholder: Processing progress indicator*

#### Step 4: Review Output

The results are displayed in three sections:

1. **📋 Executive Summary** - A concise overview of the meeting's main topics and outcomes
2. **✅ Action Items Table** - Structured list of tasks, owners, and deadlines
3. **📄 Full Minutes** - Complete formatted Markdown document

> 📸 *Screenshot placeholder: Output section showing summary, action items, and full minutes*

#### Step 5: Export

- **Copy**: Click the copy button to copy text to clipboard
- **Download**: Click the download button to save as a Markdown file

The downloaded file is named with the meeting type and timestamp for easy organization.

### Tips for Best Results

#### 🎤 Audio Recording Tips

| Tip | Why It Matters |
|-----|----------------|
| Record in a quiet environment | Reduces background noise interference |
| Speak clearly and at moderate pace | Improves transcription accuracy |
| Use a quality microphone | Better audio signal clarity |
| Avoid overlapping conversations | Whisper works best with single speakers |
| Keep recordings under 5 minutes | Faster processing, better results |

#### 📝 Transcript Tips

| Tip | Why It Matters |
|-----|----------------|
| Include speaker names | Enables better action item attribution |
| Use explicit commitment language | "I will... by Friday" is easier to extract |
| Mention specific deadlines | "Next Monday" vs "later" for clarity |
| Add meeting context | Date, attendees, purpose at the start |
| Ensure sufficient length | At least 50 words for meaningful summaries |

---

## ⚙️ Technical Details

### Architecture

The application follows a modular pipeline architecture where each processing stage transforms input data toward the final output:

```
┌─────────────────────────────────────────────────────────────────┐
│                        INPUT LAYER                               │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │Audio Upload │  │Text Input   │  │Meeting Type │             │
│  │(WAV/MP3/M4A)│  │(Paste)      │  │Selector     │             │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘             │
└─────────┼────────────────┼────────────────┼────────────────────┘
          │                │                │
          ▼                │                │
┌─────────────────┐        │                │
│ ASR Module      │        │                │
│ (Whisper-small) │        │                │
│ 244M params     │        │                │
└────────┬────────┘        │                │
         │                 │                │
         └────────┬────────┘                │
                  ▼                         │
┌─────────────────────────────┐             │
│    Summarization Module     │             │
│    (BART-large-cnn)         │◄────────────┘
│    406M params              │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│    Extraction Module        │
│    (Flan-T5-small)          │
│    60M params               │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────────┐
│                        OUTPUT LAYER                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │Executive    │  │Action Items │  │Full Minutes │             │
│  │Summary      │  │Table        │  │(Markdown)   │             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
└─────────────────────────────────────────────────────────────────┘
```

### Models

| Component | Model | Parameters | Purpose |
|-----------|-------|------------|---------|
| **ASR** | `openai/whisper-small` | 244M | Audio-to-text transcription with high accuracy on English audio |
| **Summarization** | `facebook/bart-large-cnn` | 406M | Generate abstractive summaries preserving key information |
| **Extraction** | `google/flan-t5-small` | 60M | Instruction-following for action item identification |

### CPU Optimization

The application is optimized for Hugging Face Spaces free CPU tier:

| Optimization | Implementation |
|--------------|----------------|
| **Dtype** | `torch.float32` for CPU compatibility (avoids float16 errors) |
| **Model Mode** | `model.eval()` after loading, `torch.no_grad()` during inference |
| **Memory** | Global model caching, garbage collection after processing |
| **Batching** | `batch_size=1` for memory efficiency |
| **Chunking** | Long transcripts split into 3000-char chunks with 200-char overlap |
| **Timeouts** | 120-second max processing time with graceful degradation |

### Performance Targets

| Metric | Target | Typical |
|--------|--------|---------|
| 1-minute audio transcription | < 60 seconds | 30-45 seconds |
| Summarization | < 15 seconds | 8-12 seconds |
| Action item extraction | < 10 seconds | 5-8 seconds |
| Total end-to-end | < 120 seconds | 60-90 seconds |
| Memory usage | < 4GB | 2.5-3.5GB |

### Framework Stack

- **UI Framework**: Gradio 5.x with `gr.themes.Soft()` for professional appearance
- **ML Framework**: PyTorch 2.2.0+ with Transformers 4.40.0+
- **Hosting**: Hugging Face Spaces (CPU tier)
- **Python**: 3.10+

---

## 🔒 Privacy Notice

> ⚠️ **Important: Read Before Using**

### How Your Data Is Processed

| Aspect | Detail |
|--------|--------|
| **Storage** | Audio and transcripts are processed **in-memory only** and are **NOT stored** on our servers |
| **Retention** | No data retention - all inputs are discarded immediately after processing |
| **Authentication** | No user accounts required - no personal information collected |
| **Encryption** | All connections use HTTPS encryption |

### What NOT to Upload

Please **do not upload** meetings containing:

- ❌ Confidential business strategies or trade secrets
- ❌ Personal identifiable information (PII) of individuals
- ❌ Financial data subject to regulatory requirements
- ❌ HR discussions, performance reviews, or disciplinary matters
- ❌ Legal proceedings or attorney-client communications
- ❌ Healthcare information (PHI)
- ❌ Any content you do not have permission to share

### Enterprise Use

For organizations handling sensitive data, we strongly recommend:

1. **Self-hosting** this application on your own infrastructure
2. Using the **text input option** instead of audio for maximum control
3. Reviewing generated output before sharing externally
4. Implementing your own data governance policies

### Third-Party Processing

Your content is processed by pre-trained AI models hosted within the Hugging Face Spaces environment. While we do not store your data, the models themselves were trained on publicly available datasets and may reflect biases present in that training data.

**By using this application, you acknowledge that you have appropriate permissions to process and share the meeting content you upload.**

---

## 🛠️ Troubleshooting

### Common Issues and Solutions

#### Audio Upload Issues

| Problem | Likely Cause | Solution |
|---------|--------------|----------|
| "Unsupported format" | Wrong file type | Use WAV, MP3, M4A, or WebM format |
| "File too large" | Over 25MB | Compress audio or record in shorter segments |
| "Audio too long" | Over 10 minutes | Split into smaller files under 5 minutes each |
| No transcription output | Poor audio quality | Record in quieter environment, use better microphone |

#### Processing Issues

| Problem | Likely Cause | Solution |
|---------|--------------|----------|
| Processing timeout | Server under load | Wait a moment and try again |
| Empty summary | Transcript too short | Provide at least 50 words |
| Missing action items | No clear commitments | Ensure explicit statements like "I will..." |
| Incorrect owner names | Names not mentioned clearly | State names explicitly in meeting |

#### Quality Issues

| Problem | Likely Cause | Solution |
|---------|--------------|----------|
| Poor transcription accuracy | Background noise, accents | Use the text input option instead |
| Generic summary | Unstructured meeting | Add more context and explicit points |
| Wrong action items | Ambiguous language | Use clearer commitment language |

### Getting Help

1. **Check the Tips**: Review the audio and transcript tips above
2. **Try Examples**: Use the Example Gallery to test functionality
3. **Report Issues**: Open an issue on [GitHub](https://github.com/insydr/MeetingMinutesGenerator/issues)
4. **Feature Requests**: We welcome suggestions for improvements

---

## 🤝 Contributing

We welcome contributions from the community! Here's how you can help:

### Ways to Contribute

- 🐛 **Report Bugs**: Open an issue with details about the problem
- 💡 **Suggest Features**: Share your ideas for new functionality
- 📝 **Improve Documentation**: Help make our docs clearer
- 🔧 **Submit Code**: Fix bugs or add features via pull requests

### Development Setup

```bash
# Clone the repository
git clone https://github.com/insydr/MeetingMinutesGenerator.git
cd MeetingMinutesGenerator

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run in mock mode (no model loading, faster testing)
MEETING_MINUTES_MOCK_MODE=true python app.py

# Run with full models
python app.py
```

### Pull Request Process

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes with clear commit messages
4. Add tests if applicable
5. Update documentation if needed
6. Push to your branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

### Code Style

- Follow PEP 8 guidelines
- Use meaningful variable names
- Add docstrings to functions
- Keep functions focused and modular

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2024 Meeting Minutes Generator

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 🙏 Acknowledgments

This project builds upon incredible open-source work:

- [OpenAI Whisper](https://github.com/openai/whisper) - Robust speech recognition
- [Facebook BART](https://huggingface.co/facebook/bart-large-cnn) - Abstractive summarization
- [Google Flan-T5](https://huggingface.co/google/flan-t5-small) - Instruction-following capabilities
- [Gradio](https://gradio.app/) - Rapid ML application development
- [Hugging Face](https://huggingface.co/) - Model hosting and Spaces platform

---

## 📊 Project Status

| Phase | Status | Notes |
|-------|--------|-------|
| MVP Development | ✅ Complete | Core functionality implemented |
| CPU Optimization | ✅ Complete | Optimized for Hugging Face Spaces |
| Documentation | ✅ Complete | README, TESTING, PRD |
| Testing | ✅ Complete | Manual test scenarios verified |
| Deployment | ✅ Ready | Ready for Hugging Face Spaces |

---

## 📞 Support

For questions, issues, or feature requests:

- **GitHub Issues**: [github.com/insydr/MeetingMinutesGenerator/issues](https://github.com/insydr/MeetingMinutesGenerator/issues)
- **Documentation**: See [docs/PRD.md](./docs/PRD.md) for detailed specifications
- **Testing Guide**: See [TESTING.md](./TESTING.md) for test scenarios

---

*Built with ❤️ using Gradio and Hugging Face Transformers*
