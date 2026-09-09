from __future__ import annotations

import logging
import shutil
import tempfile
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse

from commonforms.exceptions import EncryptedPdfError
from commonforms.inference import prepare_form

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("commonforms.web")

ROOT_DIR = Path(__file__).resolve().parent.parent
WEB_DIR = ROOT_DIR / "web"
MODELS_DIR = ROOT_DIR / "models"

# Vercel's function request/response body limit is 4.5 MB.
MAX_UPLOAD_BYTES = 4_500_000

PDF_MAGIC = b"%PDF"

MODEL_OPTIONS = {
    "ffdnet-l": {
        "label": "FFDNet-L (ONNX, fast)",
        "file": "FFDNet-L.onnx",
        "fast": True,
    },
    "ffdetr": {
        "label": "FFDetr (RF-DETR)",
        "file": "FFDetr.pth",
        "fast": False,
    },
}

app = FastAPI(
    title="CommonForms API",
    description=(
        "Upload a plain PDF and get back a fillable version. Widgets "
        "(text boxes, checkboxes, signature fields) are detected with a "
        "computer-vision model and injected into the PDF."
    ),
    version="0.2.1",
)


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def index() -> str:
    return (WEB_DIR / "index.html").read_text(encoding="utf-8")


@app.get("/healthz", tags=["meta"])
def healthz() -> dict:
    return {
        "status": "ok",
        "models": {
            name: {
                "file": option["file"],
                "available": (MODELS_DIR / option["file"]).exists(),
            }
            for name, option in MODEL_OPTIONS.items()
        },
    }


def _resolve_model(name: str) -> tuple[Path, bool]:
    option = MODEL_OPTIONS.get(name)
    if option is None:
        choices = ", ".join(MODEL_OPTIONS)
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model '{name}'. Choose from: {choices}",
        )

    model_path = MODELS_DIR / option["file"]
    if not model_path.exists():
        raise HTTPException(
            status_code=503,
            detail=(
                f"Model weights are not present on the server ({option['file']}). "
                "Run 'python scripts/download_models.py' and redeploy."
            ),
        )

    return model_path, option["fast"]


@app.post("/convert", tags=["convert"])
def convert(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="The PDF to make fillable"),  # noqa: B008
    model: str = Form("ffdnet-l", description="Which detection model to use"),
    confidence: float = Form(0.3, ge=0.0, le=1.0),
    keep_existing_fields: bool = Form(False),
    use_signature_fields: bool = Form(False),
    multiline: bool = Form(False),
) -> FileResponse:
    filename = file.filename or ""
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Please upload a file with a .pdf extension.",
        )

    raw = file.file
    raw.seek(0)
    size = 0
    chunks: list[bytes] = []
    while True:
        chunk = raw.read(1024 * 1024)
        if not chunk:
            break
        size += len(chunk)
        if size > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail=(
                    "Upload is too large. Vercel functions accept at most "
                    f"{MAX_UPLOAD_BYTES // 1_000_000}.5 MB per request."
                ),
            )
        chunks.append(chunk)

    data = b"".join(chunks)
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if not data.startswith(PDF_MAGIC):
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid PDF.")

    model_path, fast = _resolve_model(model)
    model_label = MODEL_OPTIONS[model]["label"]

    logger.info(
        "converting '%s' (model=%s, confidence=%s, multiline=%s, signatures=%s)",
        filename,
        model_label,
        confidence,
        multiline,
        use_signature_fields,
    )

    tmp_dir = Path(tempfile.mkdtemp(prefix="commonforms_"))
    input_path = tmp_dir / "input.pdf"
    output_path = tmp_dir / "output.pdf"
    input_path.write_bytes(data)

    try:
        prepare_form(
            input_path,
            output_path,
            model_or_path=str(model_path),
            keep_existing_fields=keep_existing_fields,
            use_signature_fields=use_signature_fields,
            device="cpu",
            confidence=confidence,
            fast=fast,
            multiline=multiline,
        )
    except EncryptedPdfError:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise HTTPException(
            status_code=400,
            detail=(
                "This PDF is password-protected. Remove the password and try again."
            ),
        )
    except Exception:
        logger.exception("conversion failed for '%s'", filename)
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise HTTPException(
            status_code=500,
            detail=(
                "Conversion failed while processing this PDF. It may contain "
                "malformed or unsupported content. Please try another file."
            ),
        )

    if not output_path.exists() or output_path.stat().st_size == 0:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise HTTPException(
            status_code=500,
            detail="Conversion produced no output.",
        )

    background_tasks.add_task(shutil.rmtree, tmp_dir, True)
    return FileResponse(
        output_path,
        media_type="application/pdf",
        filename=f"fillable_{Path(filename).stem}.pdf",
    )
