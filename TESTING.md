# Testing Guide

## Meeting Minutes Generator - Manual Test Scenarios

This document outlines the comprehensive testing strategy for the Meeting Minutes Generator application. All test scenarios should be executed before deployment and after any significant changes.

---

## Table of Contents

1. [Test Environment Setup](#test-environment-setup)
2. [Manual Test Scenarios](#manual-test-scenarios)
3. [Performance Benchmarks](#performance-benchmarks)
4. [Browser Compatibility](#browser-compatibility)
5. [Error Handling Tests](#error-handling-tests)
6. [Security Tests](#security-tests)
7. [Accessibility Tests](#accessibility-tests)

---

## Test Environment Setup

### Prerequisites

```bash
# Clone the repository
git clone https://github.com/insydr/MeetingMinutesGenerator.git
cd MeetingMinutesGenerator

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Application

```bash
# Full mode (loads all models - ~2-3 minutes startup)
python app.py

# Mock mode (fast startup for UI testing)
MEETING_MINUTES_MOCK_MODE=true python app.py
```

### Test Data

Sample audio files are available in the `samples/` directory:
- `standup_sample.wav` - Daily standup meeting
- `client_call_sample.wav` - Client call recording
- `brainstorm_sample.wav` - Brainstorming session
- `retrospective_sample.wav` - Sprint retrospective
- `quick_sync_sample.wav` - Quick team sync

---

## Manual Test Scenarios

### TC-01: Happy Path - 1-Minute Clear Audio

**Objective**: Verify successful processing of clear, short audio

**Preconditions**:
- Application running and accessible
- Sample audio file ready (1 minute, clear speech)

**Test Steps**:

| Step | Action | Expected Result |
|------|--------|-----------------|
| 1 | Navigate to application URL | Interface loads with all components visible |
| 2 | Click "Upload Recording" button | File picker dialog opens |
| 3 | Select `samples/standup_sample.wav` | File uploads, waveform displays |
| 4 | Select "Standup" from meeting type dropdown | "Standup" is selected |
| 5 | Click "Generate Minutes" button | Progress indicator shows processing stages |
| 6 | Wait for processing to complete | All three output sections populate |
| 7 | Verify Executive Summary | Summary is coherent and relevant |
| 8 | Verify Action Items Table | Table has Task, Owner, Deadline columns with data |
| 9 | Verify Full Minutes | Markdown formatted correctly with all sections |
| 10 | Click download button | `.md` file downloads with correct naming |

**Pass Criteria**:
- All outputs generated without errors
- Processing time < 120 seconds
- Output quality meets basic readability standards
- Downloaded file matches displayed content

**Status**: ☐ Pass | ☐ Fail

**Notes**:
```
[Record any observations or issues here]
```

---

### TC-02: Happy Path - Text Transcript Input

**Objective**: Verify successful processing of pasted transcript

**Preconditions**:
- Application running and accessible
- Sample transcript text ready (100+ words)

**Test Data**:
```
Team standup - March 27, 2024
Attendees: Alex, Jordan, Sarah

Alex: Yesterday I completed the user authentication module. Today I'll start on the dashboard UI. No blockers.

Jordan: I finished the API integration for the payment system. Will work on testing today. I need the API keys from Sarah.

Sarah: I'll send the API keys by end of day. I'm also scheduling the client demo for next Tuesday at 2 PM. Alex, can you have the dashboard ready by Monday?

Alex: Yes, I'll have it ready by Monday EOD.

Jordan: I'll complete testing by Wednesday so we're ready for the demo.

Action items:
- Sarah to send API keys today
- Alex to complete dashboard by Monday
- Jordan to complete testing by Wednesday
- Sarah to confirm demo for Tuesday 2 PM
```

**Test Steps**:

| Step | Action | Expected Result |
|------|--------|-----------------|
| 1 | Navigate to application URL | Interface loads correctly |
| 2 | Clear any previous audio input | Audio component shows "no file" state |
| 3 | Paste transcript into text area | Text displays in text area |
| 4 | Select "Standup" meeting type | Dropdown shows "Standup" |
| 5 | Click "Generate Minutes" button | Processing begins |
| 6 | Wait for completion | All outputs generated |
| 7 | Verify Action Items extracted correctly | At least 4 action items identified |
| 8 | Verify owner names extracted | Alex, Jordan, Sarah correctly identified |
| 9 | Verify deadlines captured | Monday, Wednesday, Tuesday 2 PM captured |

**Pass Criteria**:
- Text processed successfully
- Processing time < 30 seconds (no transcription needed)
- Action items extracted with reasonable accuracy

**Status**: ☐ Pass | ☐ Fail

**Notes**:
```
[Record any observations or issues here]
```

---

### TC-03: Edge Case - Blurry/Poor Quality Audio

**Objective**: Verify graceful handling of poor audio quality

**Preconditions**:
- Application running
- Audio file with background noise or unclear speech

**Test Steps**:

| Step | Action | Expected Result |
|------|--------|-----------------|
| 1 | Upload audio with background noise | File accepted |
| 2 | Process audio | Transcription completes (may be imperfect) |
| 3 | Review output | Warning message displayed about quality |
| 4 | Verify fallback behavior | Output still generated with available text |

**Pass Criteria**:
- Application does not crash
- User receives feedback about quality issues
- Processing continues with available transcription

**Status**: ☐ Pass | ☐ Fail

**Notes**:
```
[Record any observations or issues here]
```

---

### TC-04: Edge Case - No Action Items Detected

**Objective**: Verify helpful feedback when no action items are found

**Test Data**:
```
Team sync - March 27, 2024

We discussed the project status. Everything is going well. The team is making good progress on all fronts. We agreed that the current approach is working and should continue. No major concerns or issues to report. The next sync will be next week at the same time.
```

**Test Steps**:

| Step | Action | Expected Result |
|------|--------|-----------------|
| 1 | Paste transcript with no clear action items | Text accepted |
| 2 | Generate minutes | Processing completes |
| 3 | Check Action Items table | Shows helpful tip or placeholder |
| 4 | Verify no crash or error state | Output displays gracefully |

**Pass Criteria**:
- Application handles gracefully
- Helpful tip displayed: "No action items detected. Tip: Try using explicit commitment language..."
- Full minutes still generated

**Status**: ☐ Pass | ☐ Fail

**Notes**:
```
[Record any observations or issues here]
```

---

### TC-05: Edge Case - Long Transcript (Chunking)

**Objective**: Verify chunked processing for long transcripts

**Test Data**: Transcript > 3,000 characters

**Test Steps**:

| Step | Action | Expected Result |
|------|--------|-----------------|
| 1 | Paste transcript with 5,000+ characters | Text accepted |
| 2 | Generate minutes | Processing begins |
| 3 | Monitor processing | Multiple chunk summaries generated |
| 4 | Verify final summary | Coherent combined summary produced |
| 5 | Verify processing time | Still completes within timeout |

**Pass Criteria**:
- Long text processed without timeout
- Summary is coherent (not fragmented)
- Processing completes successfully

**Status**: ☐ Pass | ☐ Fail

**Notes**:
```
[Record any observations or issues here]
```

---

### TC-06: Edge Case - Very Short Transcript

**Objective**: Verify handling of insufficient input

**Test Data**: "Hi team. Good meeting. Bye."

**Test Steps**:

| Step | Action | Expected Result |
|------|--------|-----------------|
| 1 | Paste very short text (< 50 words) | Text accepted |
| 2 | Generate minutes | Processing begins |
| 3 | Verify warning message | Warning about short text displayed |
| 4 | Verify output still generated | Basic output provided |

**Pass Criteria**:
- Warning displayed to user
- Processing continues with graceful degradation
- Output still generated (even if minimal)

**Status**: ☐ Pass | ☐ Fail

**Notes**:
```
[Record any observations or issues here]
```

---

### TC-07: Validation - No Input Provided

**Objective**: Verify validation when no input is provided

**Test Steps**:

| Step | Action | Expected Result |
|------|--------|-----------------|
| 1 | Leave audio empty | Shows "no file" state |
| 2 | Leave text empty | Shows placeholder |
| 3 | Click "Generate Minutes" | Clear error message displayed |
| 4 | Verify no processing occurs | No wasted computation |

**Pass Criteria**:
- Clear error message: "Please provide either audio or transcript input"
- No processing initiated

**Status**: ☐ Pass | ☐ Fail

**Notes**:
```
[Record any observations or issues here]
```

---

### TC-08: Validation - File Too Large

**Objective**: Verify rejection of oversized files

**Preconditions**:
- Audio file > 25MB

**Test Steps**:

| Step | Action | Expected Result |
|------|--------|-----------------|
| 1 | Attempt to upload file > 25MB | File rejected or warning shown |
| 2 | Verify error message | Clear message about size limit |

**Pass Criteria**:
- File rejected with clear message
- User guided to reduce file size

**Status**: ☐ Pass | ☐ Fail

**Notes**:
```
[Record any observations or issues here]
```

---

### TC-09: Validation - Audio Too Long

**Objective**: Verify handling of long audio recordings

**Preconditions**:
- Audio file > 10 minutes duration

**Test Steps**:

| Step | Action | Expected Result |
|------|--------|-----------------|
| 1 | Upload audio > 10 minutes | File rejected with clear message |
| 2 | Verify duration warning | User informed of time limit |

**Pass Criteria**:
- Clear rejection message
- User guided on duration limits

**Status**: ☐ Pass | ☐ Fail

**Notes**:
```
[Record any observations or issues here]
```

---

### TC-10: Meeting Type Selection

**Objective**: Verify all meeting types produce appropriate output

**Test Steps**:

| Step | Action | Expected Result |
|------|--------|-----------------|
| 1 | Use same transcript | Standard test transcript |
| 2 | Process with "Standup" | Output includes standup header |
| 3 | Process with "Client Call" | Output includes client call header |
| 4 | Process with "Brainstorm" | Output includes brainstorm header |
| 5 | Process with "Retrospective" | Output includes retrospective header |
| 6 | Process with "Other" | Output includes other header |

**Pass Criteria**:
- Each meeting type produces correctly labeled output
- Meeting type appears in downloaded filename

**Status**: ☐ Pass | ☐ Fail

**Notes**:
```
[Record any observations or issues here]
```

---

### TC-11: Example Gallery - Text Examples

**Objective**: Verify text example functionality

**Test Steps**:

| Step | Action | Expected Result |
|------|--------|-----------------|
| 1 | Navigate to "Text Examples" tab | Tab content loads |
| 2 | Click on "Standup Meeting" example | Text populates in input area |
| 3 | Click Generate | Processing works with example |
| 4 | Repeat for other examples | All examples work correctly |

**Pass Criteria**:
- All text examples load correctly
- Examples populate input correctly
- Processing works with example data

**Status**: ☐ Pass | ☐ Fail

**Notes**:
```
[Record any observations or issues here]
```

---

### TC-12: Example Gallery - Audio Examples

**Objective**: Verify audio example functionality

**Test Steps**:

| Step | Action | Expected Result |
|------|--------|-----------------|
| 1 | Navigate to "Audio Examples" tab | Tab content loads |
| 2 | Click on sample audio button | Audio loads into input |
| 3 | Click Generate | Processing works with sample audio |
| 4 | Verify output generated | Output appears in all sections |

**Pass Criteria**:
- All audio examples load correctly
- Audio plays correctly
- Processing works with sample audio

**Status**: ☐ Pass | ☐ Fail

**Notes**:
```
[Record any observations or issues here]
```

---

### TC-13: Download Functionality

**Objective**: Verify Markdown download works correctly

**Test Steps**:

| Step | Action | Expected Result |
|------|--------|-----------------|
| 1 | Generate meeting minutes | Output displayed |
| 2 | Click download button | File download initiates |
| 3 | Open downloaded file | Valid Markdown format |
| 4 | Verify filename format | Format: `meeting_minutes_[type]_[timestamp].md` |
| 5 | Compare with displayed output | Content matches exactly |

**Pass Criteria**:
- File downloads successfully
- Valid Markdown syntax
- Content matches displayed output
- Filename is descriptive

**Status**: ☐ Pass | ☐ Fail

**Notes**:
```
[Record any observations or issues here]
```

---

### TC-14: Copy to Clipboard

**Objective**: Verify copy functionality

**Test Steps**:

| Step | Action | Expected Result |
|------|--------|-----------------|
| 1 | Generate meeting minutes | Output displayed |
| 2 | Click copy button on summary | Confirmation shown |
| 3 | Paste into text editor | Correct content pasted |
| 4 | Repeat for full minutes | Works correctly |

**Pass Criteria**:
- Copy button shows feedback
- Correct content copied to clipboard

**Status**: ☐ Pass | ☐ Fail

**Notes**:
```
[Record any observations or issues here]
```

---

## Performance Benchmarks

### Target Metrics

| Metric | Target | Measurement Method |
|--------|--------|-------------------|
| **Transcription** | < 60s per minute of audio | System timestamps |
| **Summarization** | < 15s | System timestamps |
| **Action Extraction** | < 10s | System timestamps |
| **Total E2E Processing** | < 120s | User-visible timer |
| **UI Response Time** | < 100ms | Browser dev tools |
| **Memory Usage** | < 4GB peak | System monitoring |

### Benchmark Test Procedure

1. **Prepare test files**:
   - 30-second audio clip
   - 1-minute audio clip
   - 2-minute audio clip
   - 1000-word transcript

2. **Run each test 3 times** and record:
   - Start time
   - Transcription complete time
   - Summary complete time
   - Extraction complete time
   - Total time

3. **Calculate averages** and compare to targets

### Benchmark Results Template

| Test Run | File | Transcribe (s) | Summarize (s) | Extract (s) | Total (s) | Memory (MB) |
|----------|------|----------------|---------------|-------------|-----------|-------------|
| 1 | 30s audio | | | | | |
| 2 | 30s audio | | | | | |
| 3 | 30s audio | | | | | |
| Avg | 30s audio | | | | | |
| 1 | 1min audio | | | | | |
| 2 | 1min audio | | | | | |
| 3 | 1min audio | | | | | |
| Avg | 1min audio | | | | | |
| 1 | 1000-word text | N/A | | | | |
| 2 | 1000-word text | N/A | | | | |
| 3 | 1000-word text | N/A | | | | |
| Avg | 1000-word text | N/A | | | | |

---

## Browser Compatibility

### Desktop Browsers

| Browser | Version | Test Status | Notes |
|---------|---------|-------------|-------|
| Chrome | Latest | ☐ Pass ☐ Fail | |
| Firefox | Latest | ☐ Pass ☐ Fail | |
| Safari | Latest | ☐ Pass ☐ Fail | |
| Edge | Latest | ☐ Pass ☐ Fail | |

### Mobile Browsers

| Browser | Device | Test Status | Notes |
|---------|--------|-------------|-------|
| Safari iOS | iPhone | ☐ Pass ☐ Fail | |
| Chrome Android | Android Phone | ☐ Pass ☐ Fail | |
| Samsung Internet | Samsung Phone | ☐ Pass ☐ Fail | |

### Browser Test Checklist

For each browser, verify:

- [ ] Page loads correctly
- [ ] Audio upload works
- [ ] Microphone recording works (if supported)
- [ ] Text input works
- [ ] Dropdown selection works
- [ ] Generate button triggers processing
- [ ] Progress indicator displays
- [ ] Output renders correctly
- [ ] Download works
- [ ] Copy to clipboard works
- [ ] Layout is responsive
- [ ] No console errors

---

## Error Handling Tests

### Network Error Simulation

**Test Steps**:
1. Start processing
2. Disconnect network mid-processing
3. Verify graceful error message

**Expected**: User-friendly error, no crash

### Server Timeout Simulation

**Test Steps**:
1. Use file that would exceed 120s timeout
2. Verify timeout handling

**Expected**: Timeout message, graceful recovery

### Invalid File Format

**Test Steps**:
1. Upload non-audio file (e.g., .txt renamed to .mp3)
2. Verify error handling

**Expected**: Clear error message about format

---

## Security Tests

### Input Sanitization

| Test | Input | Expected Result |
|------|-------|-----------------|
| HTML injection | `<script>alert('xss')</script>` | Sanitized or rejected |
| SQL-like input | `'; DROP TABLE users; --` | Processed as text |
| Large input | 100MB text paste | Truncated gracefully |
| Special characters | Unicode, emojis | Handled correctly |

### File Upload Security

| Test | Input | Expected Result |
|------|-------|-----------------|
| Executable file | `.exe` file | Rejected |
| Script file | `.py` file | Rejected |
| Double extension | `file.exe.mp3` | Checked properly |

---

## Accessibility Tests

### Keyboard Navigation

- [ ] All interactive elements reachable via Tab
- [ ] Enter/Space activates buttons
- [ ] Escape closes modals (if any)
- [ ] Focus visible on all elements

### Screen Reader Compatibility

- [ ] All inputs have labels
- [ ] Images have alt text
- [ ] ARIA attributes present
- [ ] Logical reading order

### Visual Accessibility

- [ ] Color contrast meets WCAG 2.1 AA
- [ ] Text is resizable
- [ ] No color-only indicators

---

## Test Summary Template

### Test Execution Summary

| Date | Tester | Environment | Result |
|------|--------|-------------|--------|
| | | | |

### Issues Found

| Issue ID | Description | Severity | Status |
|----------|-------------|----------|--------|
| | | | |

### Sign-off

| Role | Name | Date | Signature |
|------|------|------|-----------|
| Tester | | | |
| Reviewer | | | |

---

## Continuous Testing

### Pre-Deployment Checklist

- [ ] All P0 test scenarios pass
- [ ] Performance benchmarks within targets
- [ ] Browser compatibility verified
- [ ] Security tests pass
- [ ] Accessibility tests pass
- [ ] No critical issues open

### Post-Deployment Verification

- [ ] Application accessible at deployed URL
- [ ] Health endpoint responds
- [ ] Example inputs work
- [ ] Error reporting functional

---

*Last Updated: March 2024*
*Document Version: 1.0*
