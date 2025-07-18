import whisper
import warnings
import os

# Suppress FP16 CPU warning
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU")

# Load the Whisper model
model = whisper.load_model("tiny")  # Use 'medium' or higher for translation support

# Path to your audio file
audio_path = r"C:\Users\Admin\Documents\okello\recording.mp3"

if not os.path.isfile(audio_path):
    raise FileNotFoundError(f"Audio file not found: {audio_path}")

# User input for optional language code
user_lang = input("Enter language code (e.g., 'en', 'sw', 'fr') or leave blank to auto-detect: ").strip()

# Load and preprocess the audio
audio = whisper.load_audio(audio_path)
audio = whisper.pad_or_trim(audio)
mel = whisper.log_mel_spectrogram(audio, n_mels=model.dims.n_mels).to(model.device)

# Detect or use specified language
if user_lang == "":
    _, probs = model.detect_language(mel)
    detected_lang = max(probs, key=probs.get)
    print(f"Detected language: {detected_lang}")
else:
    detected_lang = user_lang
    print(f"Using user-specified language: {detected_lang}")

# Step 1: Transcribe in original language
transcribe_options = whisper.DecodingOptions(language=detected_lang, task="transcribe")
transcription_result = whisper.decode(model, mel, transcribe_options)

# Step 2: Translate to English if original is not English
if detected_lang != "en":
    translate_options = whisper.DecodingOptions(language=detected_lang, task="translate")
    translation_result = whisper.decode(model, mel, translate_options)

    # Output both
    print("\n--- Transcription in Original Language ---")
    print(transcription_result.text.strip())
    print("\n--- Translation to English ---")
    print(translation_result.text.strip())
else:
    print("\n--- English Transcription ---")
    print(transcription_result.text.strip())
