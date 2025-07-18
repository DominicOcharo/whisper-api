from fastapi import FastAPI, File, UploadFile
import whisper
import os
import uuid

app = FastAPI()
model = whisper.load_model("tiny")  # Or "base"/"large" as needed

@app.post("/transcribe/")
async def transcribe_audio(file: UploadFile = File(...), lang: str = ""):
    temp_filename = f"{uuid.uuid4()}.mp3"
    with open(temp_filename, "wb") as f:
        f.write(await file.read())

    audio = whisper.load_audio(temp_filename)
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio, n_mels=model.dims.n_mels).to(model.device)

    # Detect or use provided language
    if lang == "":
        _, probs = model.detect_language(mel)
        detected_lang = max(probs, key=probs.get)
    else:
        detected_lang = lang

    # Transcribe
    transcribe_result = whisper.decode(model, mel, whisper.DecodingOptions(language=detected_lang, task="transcribe"))

    # Translate if not English
    if detected_lang != "en":
        translate_result = whisper.decode(model, mel, whisper.DecodingOptions(language=detected_lang, task="translate"))
        os.remove(temp_filename)
        return {
            "detected_language": detected_lang,
            "transcription": transcribe_result.text.strip(),
            "translation_to_english": translate_result.text.strip()
        }

    os.remove(temp_filename)
    return {
        "detected_language": detected_lang,
        "transcription": transcribe_result.text.strip()
    }
