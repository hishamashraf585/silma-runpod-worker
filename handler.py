"""Runpod Serverless worker: SILMA TTS (Arabic) for odoo-video narration.

input:  {"items": [{"text": "...", "speed": 1.0, "nfe": 7}, ...]}
output: {"wavs": [<base64 WAV>, ...], "secs": [<generation seconds>, ...], "gpu": "<name>"}
"""
import base64, os, tempfile, time
import runpod, torch
from importlib.resources import files
from silma_tts.api import SilmaTTS

REF = str(files("silma_tts").joinpath("infer/ref_audio_samples/ar.ref.24k.wav"))
REF_TEXT = "ويدقق النظر في القرآن الكريم وسائر الكتب السماوية ويتبع مسالك الرسل العظام عليهم الصلاة والسلام."
TTS = SilmaTTS()            # loads once per worker (model weights are baked into the image)
GPU = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"


def handler(job):
    items = job["input"].get("items") or []
    wavs, secs = [], []
    for it in items:
        t = time.time()
        fd, path = tempfile.mkstemp(suffix=".wav"); os.close(fd)
        TTS.infer(ref_file=REF, ref_text=REF_TEXT, gen_text=it["text"], file_wave=path, seed=0,
                  speed=float(it.get("speed", 1.0)), nfe_step=int(it.get("nfe", 7)))
        wavs.append(base64.b64encode(open(path, "rb").read()).decode()); os.remove(path)
        secs.append(round(time.time() - t, 2))
    return {"wavs": wavs, "secs": secs, "gpu": GPU}


runpod.serverless.start({"handler": handler})
