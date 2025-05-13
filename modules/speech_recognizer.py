import torch
from transformers import pipeline

def transcribe_audio(audio_path: str) -> str:
    generate_kwargs = {
        "language": "Japanese",
        "no_repeat_ngram_size": 0,
        "repetition_penalty": 1.0,
    }
    pipe = pipeline(
        "automatic-speech-recognition",
        model="litagin/anime-whisper",
        device="cuda" if torch.cuda.is_available() else "cpu",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        chunk_length_s=30.0,
        batch_size=64,
    )
    result = pipe(audio_path, generate_kwargs=generate_kwargs)
    return result["text"]

if __name__ == "__main__":
    audio_path = "assets/test_audio.wav"
    text = transcribe_audio(audio_path)
    print(text)