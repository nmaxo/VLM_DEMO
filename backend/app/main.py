import io
import os
import uuid
from typing import Dict
from datetime import datetime

import torch
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pathlib import Path
from transformers import AutoProcessor, AutoModelForImageTextToText

# ===== CONFIGURATION =====
DEVICE_ENV = os.getenv("DEVICE", "cpu").lower()          # "cuda", "gpu"
if DEVICE_ENV in ["cuda", "gpu"] and torch.cuda.is_available():
    device = torch.device("cuda")
    print("GPU DETECTED AND ENABLED: cuda")
else:
    device = torch.device("cpu")
    print("Running on CPU")

VQA_MODEL_ID = os.getenv("VQA_MODEL_ID", "")
MODEL_SIZE = os.getenv("MODEL_SIZE", "256M")
# MODELS_DIR = Path(os.getenv("MODELS_DIR", "/models"))
MODELS_DIR = Path(os.getenv("MODELS_DIR", "./models"))

# HF_CACHE_DIR = Path(os.getenv("HF_HOME", "/root/.cache/huggingface"))
HF_CACHE_DIR = Path(os.getenv("HF_HOME", os.path.expanduser("~/.cache/huggingface")))
# ===== FASTAPI APP =====
app = FastAPI(title="SmolVLM VQA API")

# CORS middleware for Streamlit communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== IN-MEMORY STORAGE =====
VQA_SESSIONS: Dict[str, dict] = {}
SESSION_TIMEOUT = 3600  # 1 hour


# ===== MODEL HANDLER =====
class ModelHandler:
    MODEL_SIZES = {
        '256M': 'HuggingFaceTB/SmolVLM2-256M-Video-Instruct',
        '500M': 'HuggingFaceTB/SmolVLM2-500M-Video-Instruct',
        '2.2B': 'HuggingFaceTB/SmolVLM2-2.2B-Instruct',
        '1B': 'HuggingFaceTB/SmolVLM2-2.2B-Instruct',
        '2B': 'HuggingFaceTB/SmolVLM2-2.2B-Instruct',
    }

    def __init__(self, model_id: str = "", model_size: str = "256M", models_dir: Path = MODELS_DIR, hf_cache: Path = HF_CACHE_DIR, device_str: str = DEVICE_ENV):
        self.hf_cache = hf_cache
        self.models_dir = models_dir
        self.device_str = device_str
        self.device = torch.device("cuda" if device_str in ["cuda", "gpu"] and torch.cuda.is_available() else "cpu")
        print(f"ModelHandler initialized on device: {self.device}")

        if model_id:
            self.model_id = model_id
        else:
            key = model_size.upper()
            if key not in self.MODEL_SIZES:
                for k in self.MODEL_SIZES.keys():
                    if k in key or key in k:
                        key = k
                        break
                else:
                    key = '256M'
            self.model_id = self.MODEL_SIZES.get(key, self.MODEL_SIZES['256M'])

        self.local_model_path = self.models_dir / Path(self.model_id.replace('/', '_'))
        self.processor = None
        self.model = None
        try:
            self._load_model()
            print(f"✓ Model loaded successfully: {self.model_id}")
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            raise

    def _load_model(self):
        # Create directories if they don't exist
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.hf_cache.mkdir(parents=True, exist_ok=True)

        # Check if local model exists
        if self.local_model_path.exists() and any(self.local_model_path.iterdir()):
            model_path = str(self.local_model_path)
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            print(f"Loading model from local cache: {model_path}")
        else:
            if "TRANSFORMERS_OFFLINE" in os.environ:
                del os.environ["TRANSFORMERS_OFFLINE"]
            model_path = self.model_id
            print(f"Downloading model from Hugging Face: {model_path}")

        # Load processor
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            cache_dir=str(self.hf_cache),
            trust_remote_code=True,
        )

        # Determine dtype and attention implementation
        if self.device == torch.device("cpu"):
            model_dtype = torch.float32
            attn_implementation = "eager"
        else:
            model_dtype = torch.bfloat16 if hasattr(torch, 'bfloat16') else torch.float16
            attn_implementation = "eager"

        # Load model
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            cache_dir=str(self.hf_cache),
            trust_remote_code=True,
            torch_dtype=model_dtype,
            _attn_implementation=attn_implementation,
            low_cpu_mem_usage=True,
            device_map="auto" if str(self.device) != "cpu" else None,
        )

        # Move model to device
        try:
            if not hasattr(self.model, 'hf_device_map') or self.model.hf_device_map is None:
                if str(self.device).startswith("cuda") and torch.cuda.is_available():
                    self.model = self.model.to("cuda")
                elif str(self.device) == "mps" and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    self.model = self.model.to("mps")
                else:
                    self.model = self.model.to("cpu")
        except Exception as e:
            print(f"Warning: Device move failed, using CPU: {e}")
            self.model = self.model.to("cpu")

        self.model.eval()

        # Save model locally for future runs
        if not self.local_model_path.exists() or not any(self.local_model_path.iterdir()):
            self.local_model_path.mkdir(parents=True, exist_ok=True)
            try:
                self.model.save_pretrained(str(self.local_model_path))
                self.processor.save_pretrained(str(self.local_model_path))
                print(f"✓ Model saved to local cache: {self.local_model_path}")
            except Exception as e:
                print(f"Warning: Could not save model locally: {e}")

    def vqa(self, image: Image.Image, question: str) -> str:
        """Answer a question about an image"""
        messages = [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}
        ]
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=prompt, images=[image], return_tensors="pt")
        inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        
        with torch.no_grad():
            generate_kwargs = {"max_new_tokens": 512, "do_sample": False, "num_beams": 1, "use_cache": True}
            if hasattr(self.processor, 'tokenizer'):
                if self.processor.tokenizer.pad_token_id is not None:
                    generate_kwargs["pad_token_id"] = self.processor.tokenizer.pad_token_id
                if hasattr(self.processor.tokenizer, 'eos_token_id') and self.processor.tokenizer.eos_token_id is not None:
                    generate_kwargs["eos_token_id"] = self.processor.tokenizer.eos_token_id
            outputs = self.model.generate(**inputs, **generate_kwargs)
        
        response = self.processor.decode(outputs[0], skip_special_tokens=True)
        import re
        match = re.search('Assistant: ', response)
        if match:
            return response[match.end():].strip()
        return response.strip()

    def caption(self, image: Image.Image) -> str:
        """Generate a caption for an image"""
        question = "Describe this image."
        messages = [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}
        ]
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=prompt, images=[image], return_tensors="pt")
        inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        
        with torch.no_grad():
            generate_kwargs = {"max_new_tokens": 80, "do_sample": False, "num_beams": 1, "use_cache": True, "early_stopping": True}
            if hasattr(self.processor, 'tokenizer'):
                if self.processor.tokenizer.pad_token_id is not None:
                    generate_kwargs["pad_token_id"] = self.processor.tokenizer.pad_token_id
                if hasattr(self.processor.tokenizer, 'eos_token_id') and self.processor.tokenizer.eos_token_id is not None:
                    generate_kwargs["eos_token_id"] = self.processor.tokenizer.eos_token_id
            outputs = self.model.generate(**inputs, **generate_kwargs)
        
        response = self.processor.decode(outputs[0], skip_special_tokens=True)
        import re
        match = re.search('Assistant: ', response)
        if match:
            return response[match.end():].strip()
        return response.strip()


# ===== INITIALIZE MODEL HANDLER =====
print("Initializing model handler... This may take a while on first run.")
model_handler = ModelHandler(
    model_id=VQA_MODEL_ID,
    model_size=MODEL_SIZE,
    models_dir=MODELS_DIR,
    hf_cache=HF_CACHE_DIR,
    device_str=DEVICE_ENV
)


# ===== ROUTES =====

@app.get("/health")
async def health():
    """Health check endpoint"""
    ok = getattr(model_handler, 'model', None) is not None and getattr(model_handler, 'processor', None) is not None
    return {
        "status": "healthy" if ok else "unhealthy",
        "model": model_handler.model_id if ok else None,
        "device": str(model_handler.device)
    }


def ensure_image(file: UploadFile):
    """Validate that uploaded file is an image"""
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Expected image file (jpg, png, etc). Check file type.",
        )


def cleanup_old_sessions():
    """Remove sessions older than SESSION_TIMEOUT"""
    now = datetime.now()
    to_delete = [
        sid for sid, data in VQA_SESSIONS.items()
        if (now - data.get('created', now)).total_seconds() > SESSION_TIMEOUT
    ]
    for sid in to_delete:
        del VQA_SESSIONS[sid]
    if to_delete:
        print(f"Cleaned up {len(to_delete)} old sessions")


@app.post("/api/vqa/init")
async def vqa_init(image: UploadFile = File(...)):
    """Initialize VQA session with image and generate initial caption"""
    ensure_image(image)
    cleanup_old_sessions()

    img_bytes = await image.read()
    try:
        image_pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Failed to read image")

    session_id = str(uuid.uuid4())
    VQA_SESSIONS[session_id] = {
        'data': img_bytes,
        'created': datetime.now()
    }

    print(f"Created VQA session: {session_id}")
    
    # Generate caption
    caption = model_handler.caption(image_pil)
    
    return {
        "session_id": session_id,
        "caption": caption
    }


@app.post("/api/vqa/ask")
async def vqa_ask(
    session_id: str = Form(...),
    question: str = Form(...),
):
    """Answer a question about an image in an existing VQA session"""
    if session_id not in VQA_SESSIONS:
        raise HTTPException(status_code=404, detail="Session not found or expired")

    if not question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    img_bytes = VQA_SESSIONS[session_id]['data']
    try:
        image_pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Failed to read image from session")

    print(f"Processing question for session {session_id}: {question[:50]}...")
    
    # Generate answer
    answer = model_handler.vqa(image_pil, question)
    
    return {"answer": answer}


@app.post("/api/vqa/ocr")
async def vqa_ocr(image: UploadFile = File(...)):
    """OCR using SmolVLM (extract text from image)"""
    ensure_image(image)

    img_bytes = await image.read()
    try:
        image_pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Failed to read image")

    print("Running OCR via SmolVLM...")

    ocr_question = (
        "Extract all readable text from this image. "
        "Return only the raw text without description."
    )

    text = model_handler.vqa(image_pil, ocr_question)
    text = text.strip()

    return {"text": text}



if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)