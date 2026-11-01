# Dubbing Studio

This is a dubbing application focused on Indic language translation. It was developed as part of our course project for **Recent Advances in GenAI**.

<img width="1470" height="956" alt="Screenshot 2026-11-01 at 10 43 56 PM" src="https://github.com/user-attachments/assets/480dba6e-ad8c-49d2-b7d2-cdd28248f6f4" />

## Setup

### Prerequisites
Before you begin, ensure your system meets the following requirements:
* **Operating System:** Linux
* **Hardware:** NVIDIA GPU
* **Software:** Python 3.10, `git`, and `ffmpeg`

### Installation
To load the models used in this project, you will need an access token from [Hugging Face](https://huggingface.co/). 

Run the provided setup script to create a new environment, install the required packages, and download the models for the pipeline. Replace `<token>` with your actual Hugging Face token:

```bash
HF_TOKEN=<token> ./setup.sh
```

### API Configuration
The pipeline relies on Bodhan API endpoints for ASR (Automatic Speech Recognition), Translation, and TTS (Text-to-Speech). You must provide your API keys in two specific text files located in the root directory:

**1. `bodhan.txt`**  
Create this file and add your Bodhan and OpenRouter keys using the `service: key` format (one per line):
```text
asr: <your_asr_key>
translate: <your_translate_key>
tts: <your_tts_key>
openrouter: <your_openrouter_key>
```

**2. `keys.txt`**  
Create this file and add your Hugging Face token using the following format:
```text
hf - <your_huggingface_token>
```

## Running the Application

Once the setup is complete, the models are downloaded, and your API keys are configured, you can start the app by running:

```bash
./run_app.sh
```

## App Features

Built with a sleek Material 3 based UI with support for Dark Mode.

### Flexible Inputs & Smart Fetching

  - Multiple Input Methods: Upload your own local video files or paste a
    YouTube link.
  - Optimized YouTube Downloads: Supports YouTube videos up to 720p and 60
    minutes in length. If you select a specific time range, only that portion is
    downloaded, saving time and bandwidth.
  - Targeted Dubbing: Choose exactly which time range of the video you want to
    process and dub.

### Intelligent Audio & Video Processing

  - Ambient Noise Isolation: Works flawlessly with background music, crowd
    noise, cheering, and laughter. These sounds are preserved in the background
    and excluded from the dubbing pipeline.
  - Smart Voice Assignment: Automatically detects different speakers, works out
    their gender, and assigns appropriate voices accordingly.
  - Lip-Syncing: Synchronizes the speaker's facial/lip movements
    to match the newly generated dubbed audio for a natural result.

### Granular Editing & Control

<img width="1470" height="956" alt="Screenshot 2026-11-01 at 10 46 45 PM" src="https://github.com/user-attachments/assets/c87705ff-45f3-42ae-b53f-52f6340bdcdb" />



  - Text Corrections: Manually edit any line of the generated transcript or
    translation.
  - Speaker Management: Move a specific segment to another speaker, assign it to
    a newly created speaker, or manually set a speaker's gender.
  - Timeline Controls: Split a segment exactly at the playhead for precise
    control.
  - Non-Destructive Deletion: If you delete a dubbed segment, that specific time
    span automatically reverts to the original audio and video.
  - Efficient Re-processing: When you make a change, the app
    re-runs the pipeline (from transcription through to lip-sync) only for the
    affected segment. The rest of the dub remains exactly as is.

