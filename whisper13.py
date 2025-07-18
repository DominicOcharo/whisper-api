import whisper
import warnings
import os

# Suppress FP16 CPU warning
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU")

# Load the Whisper model
model = whisper.load_model("tiny")  # Use "medium" or larger for accurate translation

# Path to your audio file
audio_path = r"C:\Users\Admin\Documents\okello\recording.mp3"

if not os.path.isfile(audio_path):
    raise FileNotFoundError(f"Audio file not found: {audio_path}")

# Load and preprocess the audio
audio = whisper.load_audio(audio_path)
audio = whisper.pad_or_trim(audio)

# Create log-Mel spectrogram and move to model's device
mel = whisper.log_mel_spectrogram(audio, n_mels=model.dims.n_mels).to(model.device)

# Detect language
_, probs = model.detect_language(mel)
detected_lang = max(probs, key=probs.get)
print(f"Detected language: {detected_lang}")

# Choose task based on detected language
task_type = "translate" if detected_lang != "en" else "transcribe"

# Decode/transcribe or translate
options = whisper.DecodingOptions(task=task_type)
result = whisper.decode(model, mel, options)

# Print output
print(f"{'Translation' if task_type == 'translate' else 'Transcription'}:", result.text.strip())
