from fastapi import FastAPI, HTTPException, UploadFile

from music_atlas.audio import analyze

app = FastAPI(title="Music Atlas", version="0.1.0")
MAX_UPLOAD = 32 * 1024 * 1024


@app.get("/health")
def health():
    return {"status": "ok", "version": app.version}


@app.post("/api/analyze")
def analyze_audio(file: UploadFile):
    if file.size is None or not 0 < file.size <= MAX_UPLOAD:
        raise HTTPException(413, "Choose an audio file up to 32 MiB.")
    try:
        return analyze(file.file)
    except ValueError as error:
        raise HTTPException(422, str(error)) from error


@app.get("/api/methods")
def methods():
    return [
        {"id": name, "status": "planned"}
        for name in ("acoustic-features", "spectrogram-cnn", "pretrained-embedding")
    ]
