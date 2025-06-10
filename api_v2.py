import os
import sys
import traceback
from typing import Generator
import re

now_dir = os.getcwd()
sys.path.append(now_dir)
sys.path.append("%s/GPT_SoVITS" % (now_dir))

import argparse
import subprocess
import wave
import signal
import numpy as np
import soundfile as sf
from fastapi import FastAPI, Response
from fastapi.responses import StreamingResponse, JSONResponse
import uvicorn
from io import BytesIO
from tools.i18n.i18n import I18nAuto
from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config
from GPT_SoVITS.TTS_infer_pack.text_segmentation_method import get_method_names as get_cut_method_names
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# print(sys.path)
i18n = I18nAuto()
cut_method_names = get_cut_method_names()

parser = argparse.ArgumentParser(description="GPT-SoVITS api")
parser.add_argument("-c", "--tts_config", type=str, default="GPT_SoVITS/configs/tts_infer.yaml", help="tts_infer路径")
parser.add_argument("-a", "--bind_addr", type=str, default="127.0.0.1", help="default: 127.0.0.1")
parser.add_argument("-p", "--port", type=int, default="9880", help="default: 9880")
args = parser.parse_args()
config_path = args.tts_config
# device = args.device
port = args.port
host = args.bind_addr
argv = sys.argv

if config_path in [None, ""]:
    config_path = "GPT-SoVITS/configs/tts_infer.yaml"

tts_config = TTS_Config(config_path)
print(tts_config)
tts_pipeline = TTS(tts_config)

APP = FastAPI()
APP.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Default reference audio paths for each language
DEFAULT_REF_AUDIO_PATHS = {
    "en": "ref/Yanqing_EN.wav",
    "zh": "ref/Yanqing.wav",
    "ja": "ref/Yanqing_JP.wav"
}

# Default prompt texts for each language
DEFAULT_PROMPT_TEXTS = {
    "en": "There are no more fitting opponents for me on the Luofu, but out there among the stars... there may be someone yet.",
    "zh": "罗浮上已没有我的对手，但放眼星际之间…可不尽然。",
    "ja": "もう「羅浮」に僕の相手が務まる人はいないけど、対象が宇宙となると…話は変わってくるだろうね。"
}

# Reference audio paths for emotional tones by language (for cut0 method)
EMOTION_REF_AUDIO_PATHS = {
    "question": {
        "en": "ref/Yanqing_EN_question.wav",
        "zh": "ref/Yanqing_question.wav",
        "ja": "ref/Yanqing_JP_question.wav"
    },
    "exclamation": {
        "en": "ref/Yanqing_EN_exclamation.wav",
        "zh": "ref/Yanqing_exclamation.wav",
        "ja": "ref/Yanqing_JP_exclamation.wav"
    }
}

# Prompt texts for emotional tones (for cut0 method)
EMOTION_PROMPT_TEXTS = {
    "question": {
        "en": "Why are you filming me? Shouldn't you be filming them?",
        "zh": "老师，你们的列车长帕姆老师，到底是什么来头啊？",
        "ja": "ねえ先生、先生たちの列車にいる車掌のパム先生って、いったい何者なの？"
    },
    "exclamation": {
        "en": "A 120-foot sword! Just thinking about it makes me excited!",
        "zh": "一百二十尺的巨剑，光是想想都觉得激动！",
        "ja": "50mの剣、想像するだけでワクワクする！"
    }
}

# Function to detect language from text with improved logic
def detect_language(text):
    has_chinese = bool(re.search(r'[\u4e00-\u9fff]', text))
    has_japanese = bool(re.search(r'[\u3040-\u30ff\u3400-\u4dbf]', text))
    has_english = bool(re.search(r'[a-zA-Z]', text))
    
    if has_chinese:
        if has_english:
            return "zh"
        else:
            return "all_zh"
    elif has_japanese:
        if has_english:
            return "ja"
        else:
            return "all_ja"
    else:
        return "en"

# Function to detect emotion from the last character of a sentence
def detect_emotion(text):
    # Define patterns for different ending punctuation marks
    question_marks = ['?', '？', '¿']
    exclamation_marks = ['!', '！', '¡']
    
    # Check if text ends with any of these punctuation marks
    if any(text.rstrip().endswith(mark) for mark in question_marks):
        return "question"
    elif any(text.rstrip().endswith(mark) for mark in exclamation_marks):
        return "exclamation"
    return None

class TTS_Request(BaseModel):
    text: str = None
    text_lang: str = None
    ref_audio_path: str = None
    aux_ref_audio_paths: list = None
    prompt_lang: str = None
    prompt_text: str = ""
    top_k: int = 5
    top_p: float = 0.9
    temperature: float = 0.8
    text_split_method: str = "cut6"
    batch_size: int = 2
    batch_threshold: float = 0.75
    split_bucket: bool = True
    speed_factor: float = 1.0
    fragment_interval: float = 0.2
    seed: int = -1
    media_type: str = "aac"
    streaming_mode: bool = False
    parallel_infer: bool = True
    repetition_penalty: float = 1.35
    sample_steps: int = 12
    super_sampling: bool = False


### modify from https://github.com/RVC-Boss/GPT-SoVITS/pull/894/files
def pack_ogg(io_buffer: BytesIO, data: np.ndarray, rate: int):
    with sf.SoundFile(io_buffer, mode="w", samplerate=rate, channels=1, format="ogg") as audio_file:
        audio_file.write(data)
    return io_buffer


def pack_raw(io_buffer: BytesIO, data: np.ndarray, rate: int):
    io_buffer.write(data.tobytes())
    return io_buffer


def pack_wav(io_buffer: BytesIO, data: np.ndarray, rate: int):
    io_buffer = BytesIO()
    sf.write(io_buffer, data, rate, format="wav")
    return io_buffer


def pack_aac(io_buffer: BytesIO, data: np.ndarray, rate: int):
    process = subprocess.Popen(
        [
            "ffmpeg",
            "-f",
            "s16le",  # 输入16位有符号小端整数PCM
            "-ar",
            str(rate),  # 设置采样率
            "-ac",
            "1",  # 单声道
            "-i",
            "pipe:0",  # 从管道读取输入
            "-c:a",
            "aac",  # 音频编码器为AAC
            "-b:a",
            "192k",  # 比特率
            "-vn",  # 不包含视频
            "-f",
            "adts",  # 输出AAC数据流格式
            "pipe:1",  # 将输出写入管道
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    out, _ = process.communicate(input=data.tobytes())
    io_buffer.write(out)
    return io_buffer


def pack_audio(io_buffer: BytesIO, data: np.ndarray, rate: int, media_type: str):
    if media_type == "ogg":
        io_buffer = pack_ogg(io_buffer, data, rate)
    elif media_type == "aac":
        io_buffer = pack_aac(io_buffer, data, rate)
    elif media_type == "wav":
        io_buffer = pack_wav(io_buffer, data, rate)
    else:
        io_buffer = pack_raw(io_buffer, data, rate)
    io_buffer.seek(0)
    return io_buffer


# from https://huggingface.co/spaces/coqui/voice-chat-with-mistral/blob/main/app.py
def wave_header_chunk(frame_input=b"", channels=1, sample_width=2, sample_rate=32000):
    # This will create a wave header then append the frame input
    # It should be first on a streaming wav file
    # Other frames better should not have it (else you will hear some artifacts each chunk start)
    wav_buf = BytesIO()
    with wave.open(wav_buf, "wb") as vfout:
        vfout.setnchannels(channels)
        vfout.setsampwidth(sample_width)
        vfout.setframerate(sample_rate)
        vfout.writeframes(frame_input)

    wav_buf.seek(0)
    return wav_buf.read()


def handle_control(command: str):
    if command == "restart":
        os.execl(sys.executable, sys.executable, *argv)
    elif command == "exit":
        os.kill(os.getpid(), signal.SIGTERM)
        exit(0)


def check_params(req: dict):
    text: str = req.get("text", "")
    text_lang: str = req.get("text_lang", "")
    ref_audio_path: str = req.get("ref_audio_path", "")
    streaming_mode: bool = req.get("streaming_mode", False)
    media_type: str = req.get("media_type", "aac")
    prompt_lang: str = req.get("prompt_lang", "")
    text_split_method: str = req.get("text_split_method", "cut6")

    if ref_audio_path in [None, ""]:
        return JSONResponse(status_code=400, content={"message": "ref_audio_path is required"})
    if text in [None, ""]:
        return JSONResponse(status_code=400, content={"message": "text is required"})
    if text_lang in [None, ""]:
        return JSONResponse(status_code=400, content={"message": "text_lang is required"})
    elif text_lang.lower() not in tts_config.languages:
        return JSONResponse(
            status_code=400,
            content={"message": f"text_lang: {text_lang} is not supported in version {tts_config.version}"},
        )
    if prompt_lang in [None, ""]:
        return JSONResponse(status_code=400, content={"message": "prompt_lang is required"})
    elif prompt_lang.lower() not in tts_config.languages:
        return JSONResponse(
            status_code=400,
            content={"message": f"prompt_lang: {prompt_lang} is not supported in version {tts_config.version}"},
        )
    if media_type not in ["wav", "raw", "ogg", "aac"]:
        return JSONResponse(status_code=400, content={"message": f"media_type: {media_type} is not supported"})
    elif media_type == "ogg" and not streaming_mode:
        return JSONResponse(status_code=400, content={"message": "ogg format is not supported in non-streaming mode"})

    if text_split_method not in cut_method_names:
        return JSONResponse(
            status_code=400, content={"message": f"text_split_method:{text_split_method} is not supported"}
        )

    return None


async def tts_handle(req: dict):
    """
    Text to speech handler.

    Args:
        req (dict):
            {
                "text": "",                   # str.(required) text to be synthesized
                "text_lang: "",               # str.(required) language of the text to be synthesized
                "ref_audio_path": "",         # str.(required) reference audio path
                "aux_ref_audio_paths": [],    # list.(optional) auxiliary reference audio paths for multi-speaker synthesis
                "prompt_text": "",            # str.(optional) prompt text for the reference audio
                "prompt_lang": "",            # str.(required) language of the prompt text for the reference audio
                "top_k": 5,                   # int. top k sampling
                "top_p": 1,                   # float. top p sampling
                "temperature": 1,             # float. temperature for sampling
                "text_split_method": "cut6",  # str. text split method, see text_segmentation_method.py for details.
                "batch_size": 1,              # int. batch size for inference
                "batch_threshold": 0.75,      # float. threshold for batch splitting.
                "split_bucket: True,          # bool. whether to split the batch into multiple buckets.
                "speed_factor":1.0,           # float. control the speed of the synthesized audio.
                "fragment_interval":0.3,      # float. to control the interval of the audio fragment.
                "seed": -1,                   # int. random seed for reproducibility.
                "media_type": "acc",          # str. media type of the output audio, support "wav", "raw", "ogg", "aac".
                "streaming_mode": False,      # bool. whether to return a streaming response.
                "parallel_infer": True,       # bool.(optional) whether to use parallel inference.
                "repetition_penalty": 1.35    # float.(optional) repetition penalty for T2S model.
                "sample_steps": 12,           # int. number of sampling steps for VITS model V3.
                "super_sampling": False,       # bool. whether to use super-sampling for audio when using VITS model V3.
            }
    returns:
        StreamingResponse: audio stream response.
    """
    # Detect language and set defaults if needed
    text = req.get("text", "")
    text_lang_provided = "text_lang" in req and req["text_lang"]
    prompt_lang_provided = "prompt_lang" in req and req["prompt_lang"]
    ref_audio_provided = "ref_audio_path" in req and req["ref_audio_path"]
    prompt_text_provided = "prompt_text" in req and req["prompt_text"]
    text_split_method = req.get("text_split_method", "cut6")
    
    if text:
        text = re.sub(r'([\u4e00-\u9fff])\1{2,}', r'\1\1', text)
        detected_lang = detect_language(text)
        
        # If text_lang not specified, set based on detected language
        if not text_lang_provided:
            req["text_lang"] = detected_lang
            
        # Get base language (remove 'all_' prefix if present)
        base_lang = detected_lang.replace("all_", "")
        
        # If prompt_lang not specified, set based on base language
        if not prompt_lang_provided:
            req["prompt_lang"] = base_lang
            
        # Check for emotional content if text_split_method is cut0
        emotion = None
        if text_split_method == "cut0":
            emotion = detect_emotion(text)
            
        # Set ref_audio_path and prompt_text based on emotion and language
        if emotion and not ref_audio_provided:
            req["ref_audio_path"] = EMOTION_REF_AUDIO_PATHS.get(emotion, {}).get(base_lang, DEFAULT_REF_AUDIO_PATHS.get(base_lang, "ref/Yanqing.wav"))
        elif not ref_audio_provided:
            req["ref_audio_path"] = DEFAULT_REF_AUDIO_PATHS.get(base_lang, "ref/Yanqing.wav")
            
        if emotion and not prompt_text_provided:
            req["prompt_text"] = EMOTION_PROMPT_TEXTS.get(emotion, {}).get(base_lang, DEFAULT_PROMPT_TEXTS.get(base_lang, DEFAULT_PROMPT_TEXTS["en"]))
        elif not prompt_text_provided:
            req["prompt_text"] = DEFAULT_PROMPT_TEXTS.get(base_lang, DEFAULT_PROMPT_TEXTS["en"])

    streaming_mode = req.get("streaming_mode", False)
    return_fragment = req.get("return_fragment", False)
    media_type = req.get("media_type", "wav")

    check_res = check_params(req)
    if check_res is not None:
        return check_res

    if streaming_mode or return_fragment:
        req["return_fragment"] = True

    try:
        tts_generator = tts_pipeline.run(req)

        if streaming_mode:

            def streaming_generator(tts_generator: Generator, media_type: str):
                if_first_chunk = True
                current_segment = 1
                
                try:
                    for sr, chunk in tts_generator:
                        try:
                            if if_first_chunk and media_type == "wav":
                                yield wave_header_chunk(sample_rate=sr)
                                media_type = "raw"
                                if_first_chunk = False
                            
                            # Log successful segment processing
                            print(f"Successfully processed segment {current_segment} with {len(chunk)} bytes")
                            current_segment += 1
                            
                            yield pack_audio(BytesIO(), chunk, sr, media_type).getvalue()
                        except Exception as chunk_error:
                            # Log error but continue with next segment
                            print(f"Error processing chunk {current_segment}: {str(chunk_error)}")
                            # Continue to next iteration without yielding a broken chunk
                except Exception as generator_error:
                    # Log the outer generator error
                    print(f"TTS generator failed: {str(generator_error)}")
                    # No need to re-raise; this will end the stream but won't crash the server

         # _media_type = f"audio/{media_type}" if not (streaming_mode and media_type in ["wav", "raw"]) else f"audio/x-{media_type}"
            return StreamingResponse(
                streaming_generator(
                    tts_generator,
                    media_type,
                ),
                media_type=f"audio/{media_type}",
            )

        else:
            sr, audio_data = next(tts_generator)
            audio_data = pack_audio(BytesIO(), audio_data, sr, media_type).getvalue()
            return Response(audio_data, media_type=f"audio/{media_type}")
    except Exception as e:
        return JSONResponse(status_code=400, content={"message": "tts failed", "Exception": str(e)})


@APP.get("/control")
async def control(command: str = None):
    if command is None:
        return JSONResponse(status_code=400, content={"message": "command is required"})
    handle_control(command)


@APP.get("/tts")
async def tts_get_endpoint(
    text: str = None,
    text_lang: str = None,
    ref_audio_path: str = None,
    aux_ref_audio_paths: list = [],
    prompt_lang: str = None,
    prompt_text: str = None,
    top_k: int = 5,
    top_p: float = 0.9,
    temperature: float = 0.8,
    text_split_method: str = "cut6",
    batch_size: int = 2,
    batch_threshold: float = 0.75,
    split_bucket: bool = True,
    speed_factor: float = 1.0,
    fragment_interval: float = 0.2,
    seed: int = -1,
    media_type: str = "aac",
    streaming_mode: bool = False,
    parallel_infer: bool = True,
    repetition_penalty: float = 1.35,
    sample_steps: int = 12,
    super_sampling: bool = False,
):
    req = {
        "text": text,
        "text_lang": text_lang,
        "ref_audio_path": ref_audio_path,
        "aux_ref_audio_paths": aux_ref_audio_paths,
        "prompt_text": prompt_text,
        "prompt_lang": prompt_lang,
        "top_k": top_k,
        "top_p": top_p,
        "temperature": temperature,
        "text_split_method": text_split_method,
        "batch_size": int(batch_size),
        "batch_threshold": float(batch_threshold),
        "speed_factor": float(speed_factor),
        "split_bucket": split_bucket,
        "fragment_interval": fragment_interval,
        "seed": seed,
        "media_type": media_type,
        "streaming_mode": streaming_mode,
        "parallel_infer": parallel_infer,
        "repetition_penalty": float(repetition_penalty),
        "sample_steps": int(sample_steps),
        "super_sampling": super_sampling,
    }
    return await tts_handle(req)


@APP.post("/tts")
async def tts_post_endpoint(request: TTS_Request):
    req = request.dict()
    return await tts_handle(req)


@APP.get("/set_refer_audio")
async def set_refer_aduio(refer_audio_path: str = None):
    try:
        tts_pipeline.set_ref_audio(refer_audio_path)
    except Exception as e:
        return JSONResponse(status_code=400, content={"message": "set refer audio failed", "Exception": str(e)})
    return JSONResponse(status_code=200, content={"message": "success"})


# @APP.post("/set_refer_audio")
# async def set_refer_aduio_post(audio_file: UploadFile = File(...)):
#     try:
#         # 检查文件类型，确保是音频文件
#         if not audio_file.content_type.startswith("audio/"):
#             return JSONResponse(status_code=400, content={"message": "file type is not supported"})

#         os.makedirs("uploaded_audio", exist_ok=True)
#         save_path = os.path.join("uploaded_audio", audio_file.filename)
#         # 保存音频文件到服务器上的一个目录
#         with open(save_path , "wb") as buffer:
#             buffer.write(await audio_file.read())

#         tts_pipeline.set_ref_audio(save_path)
#     except Exception as e:
#         return JSONResponse(status_code=400, content={"message": f"set refer audio failed", "Exception": str(e)})
#     return JSONResponse(status_code=200, content={"message": "success"})


@APP.get("/set_gpt_weights")
async def set_gpt_weights(weights_path: str = None):
    try:
        if weights_path in ["", None]:
            return JSONResponse(status_code=400, content={"message": "gpt weight path is required"})
        tts_pipeline.init_t2s_weights(weights_path)
    except Exception as e:
        return JSONResponse(status_code=400, content={"message": "change gpt weight failed", "Exception": str(e)})

    return JSONResponse(status_code=200, content={"message": "success"})


@APP.get("/set_sovits_weights")
async def set_sovits_weights(weights_path: str = None):
    try:
        if weights_path in ["", None]:
            return JSONResponse(status_code=400, content={"message": "sovits weight path is required"})
        tts_pipeline.init_vits_weights(weights_path)
    except Exception as e:
        return JSONResponse(status_code=400, content={"message": "change sovits weight failed", "Exception": str(e)})
    return JSONResponse(status_code=200, content={"message": "success"})


if __name__ == "__main__":
    try:
        if host == "None":  # 在调用时使用 -a None 参数，可以让api监听双栈
            host = None
        uvicorn.run(app=APP, host=host, port=port, workers=1)
    except Exception:
        traceback.print_exc()
        os.kill(os.getpid(), signal.SIGTERM)
        exit(0)
