from dotenv import load_dotenv
import os

from fastapi import FastAPI, Body
from fastapi.templating import Jinja2Templates
from modules.openrouter import run_openrouter
from fastapi.responses import FileResponse
from pydantic import BaseModel
from modules.image_generator import ImageGenerationService, StableDiffusionImageGenerator

load_dotenv(dotenv_path="./.env")

api_key = os.environ.get("OPENROUTER_API_KEY")
model = os.environ.get("LLM_MODEL_NAME", "deepseek/deepseek-chat-v3-0324:free")

# FastAPIアプリ作成
app = FastAPI()
from fastapi.staticfiles import StaticFiles
app.mount(
    os.environ.get("STATIC_URL_PREFIX", "/static"),
    StaticFiles(directory=os.environ.get("STATIC_DIR") if os.environ.get("STATIC_DIR") else "src/static"),
    name="static"
)
templates = Jinja2Templates(directory="src/templates")

# 画像生成APIのルーターを組み込み
from fastapi import Request
from fastapi.responses import HTMLResponse, JSONResponse

class GenerateImageRequest(BaseModel):
    prompt: str

# サービス層のインスタンスを生成（グローバルで1つでOK）
image_service = ImageGenerationService(StableDiffusionImageGenerator())

@app.post("/generate-image")
async def generate_image_endpoint(request: GenerateImageRequest):
    try:
        import datetime
        import random
        output_dir = "outputs"
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        rand = random.randint(1000, 9999)
        filename = f"generated_{timestamp}_{rand}.png"
        output_path = os.path.join(output_dir, filename)
        image_service.generate_image(request.prompt, output_path=output_path)
        return FileResponse(
            output_path,
            media_type=os.environ.get("TEMP_IMAGE_MIME", "image/png"),
            filename=filename
        )
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        from fastapi import HTTPException
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/llm")
async def llm_endpoint(request: Request):
    data = await request.json()
    prompt = data.get("prompt")
    if not prompt:
        return JSONResponse({"error": "prompt is required"}, status_code=400)
    result = run_openrouter(api_key, model, prompt)
    return {"result": result}

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse(
        os.environ.get("INDEX_TEMPLATE", "index.html"),
        {
            "request": request,
            "llm_model": os.environ.get("LLM_MODEL_NAME", ""),
            "image_model": os.environ.get("IMAGE_GEN_MODEL_NAME", ""),
            "speech_model": os.environ.get("SPEECH_MODEL_NAME", "")
        }
    )
# HuggingFaceアップロードAPI
from pydantic import BaseModel

class HFUploadRequest(BaseModel):
    repo_type: str
    private: bool
    folder_path: str
    path_in_repo: str

@app.post("/utility/hf_file_up")
async def hf_file_up(req: HFUploadRequest):
    try:
        upload_folder_to_hf(
            req.repo_type,
            req.private,
            req.folder_path,
            req.path_in_repo
        )
        return {"result": "Upload completed."}
    except Exception as e:
        return {"error": str(e)}
# HuggingFaceモデルダウンロードAPI
from pydantic import BaseModel

class HFDownloadRequest(BaseModel):
    repo_id: str

@app.post("/utility/hf_file_dl")
async def hf_file_dl(req: HFDownloadRequest):
    try:
        path = download_model(req.repo_id)
        return {"model_path": path}
    except Exception as e:
        return {"error": str(e)}
# GPU情報取得API
from utility.gpu_check import get_gpu_info

@app.get("/utility/gpu_check")
async def gpu_check():
    try:
        return get_gpu_info()
    except Exception as e:
        return {"error": str(e)}

# 音声認識API
from fastapi import UploadFile, File, Form
from modules.speech_recognizer import transcribe_audio
import tempfile

@app.post("/asr")
async def asr(audio: UploadFile = File(...), model: str = Form(...)):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            content = await audio.read()
            tmp.write(content)
            tmp_path = tmp.name
        text = transcribe_audio(tmp_path, model)
        return {"text": text}
    except Exception as e:
        return {"error": str(e)}

# CLI用途の既存処理は __main__ ブロックで実行
if __name__ == "__main__":
# GPU情報のデバッグ出力
    try:
        from utility.gpu_check import get_gpu_info
        print("GPU INFO:", get_gpu_info())
    except Exception as e:
        print("GPU INFO取得失敗:", e)
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
from utility.hf_file_dl import download_model
from utility.hf_file_up import upload_folder_to_hf