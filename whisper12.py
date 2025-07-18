import warnings

# Suppress all warnings (including during import)
warnings.filterwarnings("ignore")

import whisper

model = whisper.load_model("tiny")
result = model.transcribe(r"C:\Users\Admin\Documents\okello\Rec.mp3")

print(f'Transcribed text: {result["text"]}')
