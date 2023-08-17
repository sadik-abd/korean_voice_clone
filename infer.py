
import numpy as np
import torch
import torchaudio
from encodec import EncodecModel
from encodec.utils import convert_audio
from hubert import HuBERTManager
from hubert import CustomHubert
from hubert import CustomTokenizer
from bark import *
from transformers import pipeline
from TTS.api import TTS


def cln_vc(audio_in, name, transcript_text):
  global ref_wav
  device = 'cuda'  # 'cuda', 'cpu', 'cuda:0', 0, -1, torch.device('cuda')
  encodec_model = EncodecModel.encodec_model_24khz()
  encodec_model.set_target_bandwidth(6.0)
  encodec_model.to(device)

  print('Downloaded and loaded models!')
  wav_file = ref_wav  # Put the path of the speaker you want to use here.
  out_file = ref_wav + '.npz'  # Put the path to save the cloned speaker to here.

  wav, sr = torchaudio.load(wav_file)

  wav_hubert = wav.to(device)

  if wav_hubert.shape[0] == 2:  # Stereo to mono if needed
      wav_hubert = wav_hubert.mean(0, keepdim=True)
  seconds = wav.shape[-1] / model.sample_rate
  semantic_tokens = generate_text_semantic(transcript_text, max_gen_duration_s=seconds)

  print('Creating coarse and fine prompts...')
  wav = convert_audio(wav, sr, encodec_model.sample_rate, 1).unsqueeze(0)

  wav = wav.to(device)

  with torch.no_grad():
      encoded_frames = encodec_model.encode(wav)
  codes = torch.cat([encoded[0] for encoded in encoded_frames], dim=-1).squeeze()

  codes = codes.cpu()
  semantic_tokens = np.array(semantic_tokens)

  np.savez(out_file,
          semantic_prompt=semantic_tokens,
          fine_prompt=codes,
          coarse_prompt=codes[:2, :]
          )
  return out_file
def clone(ref_wav,transcript="",lang="korean"):
  if lang == "korean":
    if transcript == "":
      pipe = pipeline("automatic-speech-recognition", model="Hyuk/wav2vec2-korean-v2")
      transcript = pipe(ref_wav)["text"].replace("[PAD]","")
      print(transcript)
    model = load_codec_model(use_gpu=True)
    out_file = cln_vc(ref_wav, ref_wav, transcript)
  else:
    device = 'cuda'  # 'cuda', 'cpu', 'cuda:0', 0, -1, torch.device('cuda')
    large_quant_model = False
    model = ('quantifier_V1_hubert_base_ls960_23.pth', 'tokenizer_large.pth') if large_quant_model else ('quantifier_hubert_base_ls960_14.pth', 'tokenizer.pth')
  
    print('Loading HuBERT...')
    hubert_model = CustomHubert(HuBERTManager.make_sure_hubert_installed(), device=device)
    print('Loading Quantizer...')
    quant_model = CustomTokenizer.load_from_checkpoint(HuBERTManager.make_sure_tokenizer_installed(model=model[0], local_file=model[1]), device)
    print('Loading Encodec...')
    encodec_model = EncodecModel.encodec_model_24khz()
    encodec_model.set_target_bandwidth(6.0)
    encodec_model.to(device)
  
    print('Downloaded and loaded models!')
    wav_file = ref_wav  # Put the path of the speaker you want to use here.
    out_file = ref_wav + '.npz'  # Put the path to save the cloned speaker to here.
  
    wav, sr = torchaudio.load(wav_file)
  
    wav_hubert = wav.to(device)
  
    if wav_hubert.shape[0] == 2:  # Stereo to mono if needed
        wav_hubert = wav_hubert.mean(0, keepdim=True)
  
    print('Extracting semantics...')
    semantic_vectors = hubert_model.forward(wav_hubert, input_sample_hz=sr)
    print('Tokenizing semantics...')
    semantic_tokens = quant_model.get_token(semantic_vectors)
  
    print('Creating coarse and fine prompts...')
    wav = convert_audio(wav, sr, encodec_model.sample_rate, 1).unsqueeze(0)
  
    wav = wav.to(device)
  
    with torch.no_grad():
        encoded_frames = encodec_model.encode(wav)
    codes = torch.cat([encoded[0] for encoded in encoded_frames], dim=-1).squeeze()
  
    codes = codes.cpu()
    semantic_tokens = semantic_tokens.cpu()
  
    np.savez(out_file,
            semantic_prompt=semantic_tokens,
            fine_prompt=codes,
            coarse_prompt=codes[:2, :]
            )
  
    return out_file

def init():
  preload_models()
  vc_model = TTS(model_name="voice_conversion_models/multilingual/vctk/freevc24", progress_bar=True, gpu=True)
