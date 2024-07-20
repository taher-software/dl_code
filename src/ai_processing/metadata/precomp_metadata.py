import time
import numpy as np
from app.beta.orm import session
from app.beta.models.video import Video
from app.utils import utils
import os
import pickle
import cv2
import boto3
from io import BytesIO
import base64
from PIL import Image
from config import Config
from app.ai_processing.audios.precomp_audioz import has_audio
import re
from collections import defaultdict

import tiktoken
import openai
import json
import ast
import io
from openai import OpenAI
openai_client = OpenAI(api_key=Config.openai_api_key)

import librosa
import torch
import torch.nn.functional as F
from app.ai_processing.metadata.audio_models.wavcaps.bart_captioning import BartCaptionModel
from transformers import WhisperTokenizer, WhisperFeatureExtractor
from app.ai_processing.metadata.audio_models.audiocap.audiocap import WhisperForAudioCaptioning
from hsemotion.facial_emotions import HSEmotionRecognizer

# please do not remove this line
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using:", device)


default_multimodal_prompt ="""
I have a video clip that i want to describe accross all modalities (visual, sounds, dialogue) using natural language. 
For that I have:
- Captioned the visuals using an image captioning model applied to the clip's frames. This is the obtained visual description:
[VISUAL]
- Captioned the sounds using an audio captioning model applied to the clip's audio stream. This is the obtained sound description:
[SOUND]
- Transcribed the dialogue using a speech to text model applied to the clip's audio stream. This is the obtained dialogue:
[DIALOGUE]
Based on the information above, generate a unified and coherent clip description that encompasses the information present in all modalities. The clip description must be as if someone was describing the scene, similar to audio description in movies. The description must combine seamlessly the information in all modalites, not just talking about each of them one after the other. Don't describe what you did, output only the description of the clip. The output must be in the following format:
description: <clip_description>
"""


def fill_prompt(multimodal_prompt, visual_description, sound_description, dialogue):
    multimodal_prompt = multimodal_prompt.replace("[VISUAL]", visual_description)
    multimodal_prompt = multimodal_prompt.replace("[SOUND]", sound_description)
    multimodal_prompt = multimodal_prompt.replace("[DIALOGUE]", dialogue)
        
    return multimodal_prompt


def chatgpt_multimodal_description(prompt, model = "gpt-3.5-turbo"): 
    messages=[{"role": "system", "content": "You are a helpful assistant. You are an expert screenwriter that wrote the best shooting scripts and screenplays of major hollywood movies. Your scripts perfectly describe the scenes with the right amount of detail."},
        {"role": "user", "content": prompt}]
    response = openai_client.chat.completions.create(model=model,messages=messages)
    description = response.choices[0].message.content.strip().replace("description: ", "")
    
    return description
    
    
def chatgpt_visual_description(prompt, frame, model = "gpt-4o"): 
    encoded_image = encode_image(frame)
    messages=[
        {"role": "system", "content": "You are helpful assistant"},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url":
                        f"data:image/jpeg;base64,{encoded_image}", "detail": Config.GPT_CAPTIONING_DETAIL}
                },
            ],
        },
    ]
    response = openai_client.chat.completions.create(model=model,messages=messages)
    description = response.choices[0].message.content.strip()
    
    return description
    
    
def encode_image(img, max_image=1920):
    width, height = img.size
    max_dim = max(width, height)
    if max_dim > max_image and Config.GPT_CAPTIONING_DETAIL == "high":
        scale_factor = max_image / max_dim
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        img = img.resize((new_width, new_height))

    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str
        
            
def visual_captioning(vid_path, shots, start_frame, end_frame, image_captioning_model = None, vis_processors = None,
                     captioning_prompt = None, visual_model_type = "GPT4", max_retries = 5, retry_delay = 2):

    vidcap = cv2.VideoCapture(vid_path)
    fps = round(vidcap.get(cv2.CAP_PROP_FPS))
    length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    shots = [shot for shot in shots if shot > start_frame and shot < end_frame] 
    shots.append(end_frame)
    
    visual_descriptions = []
    
    for shot in shots:
        central_frame = round(start_frame + (shot-start_frame)/2)
        print("captioning video frame", central_frame)

        vidcap.set(cv2.CAP_PROP_POS_FRAMES, central_frame)
        success, frame = vidcap.read()        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)
        
        if visual_model_type == "KOSMOS2":
            inputs = vis_processors(text=vqa_prompt, images=frame, return_tensors="pt").to(device)

            generated_ids = model.generate(
                   pixel_values=inputs["pixel_values"],
                   input_ids=inputs["input_ids"],
                   attention_mask=inputs["attention_mask"],
                   image_embeds=None,
                   image_embeds_position_mask=inputs["image_embeds_position_mask"],
                   use_cache=True,
                   max_new_tokens=128,)    
                       
            generated_text = vis_processors.batch_decode(generated_ids, skip_special_tokens=True)[0]
            frame_description, _ = vis_processors.post_process_generation(generated_text)     
            frame_description = frame_description.replace("An image of ", "").capitalize()
        elif visual_model_type == "BLIP":
            inputs = vis_processors(frame, return_tensors="pt").to(device)
            out = model.generate(**inputs)
            frame_description = vis_processors.decode(out[0], skip_special_tokens=True).capitalize()  + '.'
            
        elif visual_model_type == "GPT4":
            for attempt in range(max_retries):
                try:        
                    frame_description = chatgpt_visual_description(captioning_prompt, frame)  
                    break       
                except Exception as e:
                    print(f"Attempt {attempt + 1} failed: {e}")
                    if attempt < max_retries - 1:
                        print(f"Retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                    else:
                        print("Max retries reached. Failing.")
                        frame_description = ""
                        break 
                                  
        visual_descriptions.append([central_frame, frame_description])
        
        start_frame = shot

    vidcap.release()
    
    return visual_descriptions, shots
    

def audio_captioning(vid_path, audio_path, start_frame, end_frame, audio_model_type, audio_model, audio_feature_extractor=None, audio_tokenizer=None):
 
    temp_audio_path = audio_path.replace(".wav", f"_{str(start_frame)}.wav")   
    vidcap = cv2.VideoCapture(vid_path)
    fps = round(vidcap.get(cv2.CAP_PROP_FPS))
    length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

    print("captioning audio at frame", start_frame)

    audio_cut_com = f"ffmpeg -y -hide_banner -loglevel error -i {audio_path} -ss {str(start_frame/fps)} -to {str(end_frame/fps)} {temp_audio_path}"
    os.system(audio_cut_com)
    
    if audio_model_type=="WAVCAPS":
        waveform, sr = librosa.load(temp_audio_path, sr=32000, mono=True)
        waveform = torch.tensor(waveform)    
        max_length = 32000 * 10
        if len(waveform) > max_length:
            waveform = waveform[:max_length]
        else:
            waveform = F.pad(waveform, [0, max_length - len(waveform)], "constant", 0.0)

        waveform = waveform.unsqueeze(0)
        audio_model.eval()
        with torch.no_grad():
            waveform = waveform.to(device)
            sound_description = audio_model.generate(samples=waveform, num_beams=3)[0].capitalize() + '.'

    elif audio_model_type=="WHISPER":        
        waveform, sr = librosa.load(temp_audio_path, sr=audio_feature_extractor.sampling_rate)
        features = audio_feature_extractor(waveform, sampling_rate=sr, return_tensors="pt").input_features

        style_prefix = "clotho > caption: "
        style_prefix_tokens = audio_tokenizer("", text_target=style_prefix, return_tensors="pt", add_special_tokens=False).labels

        audio_model.eval()
        outputs = audio_model.generate(
            inputs=features.to(device),
            forced_ac_decoder_ids=style_prefix_tokens,
            max_length=100,
        )

        sound_description = audio_tokenizer.batch_decode(outputs, skip_special_tokens=True)[0].replace("clotho > caption: ", "")

    vidcap.release()

    os.remove(temp_audio_path)   

    return sound_description
    
    
def classify_face_emotion(frame, face, model_name='enet_b0_8_best_afew'): 
    x1, y1, w, h = face[:4] 

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2, y2 = x1 + w, y1 + h
    x2 = min(x2, frame.shape[1])
    y2 = min(y2, frame.shape[0])

    face_img=frame[y1:y2,x1:x2,:]
    fer = HSEmotionRecognizer(model_name=model_name,device=device)
    
    emotion_name, scores = fer.predict_emotions(face_img,logits=False)
    percentages = np.array(scores) / np.sum(scores) * 100
    percentages = [f"{value:.2f}" for value in percentages]
    
    emotion_index = percentages.index(max(percentages))
    
    return emotion_index
    
def face_emotions(vid_path, faces_path, start_frame, end_frame, model_name='enet_b0_8_best_afew'): 
    emotions_counts = [0, 0, 0, 0, 0, 0, 0, 0]
    idx_to_class={0: 'Anger', 1: 'Contempt', 2: 'Disgust', 3: 'Fear', 4: 'Happiness', 5: 'Neutral', 6: 'Sadness', 7: 'Surprise'}  
    emotions_percentages = {}  

    with open(faces_path, 'rb') as file:
        vid_faces = pickle.load(file)
                
    vidcap = cv2.VideoCapture(vid_path)
    fps = round(vidcap.get(cv2.CAP_PROP_FPS))
        
    for frame_id in range (start_frame, end_frame, fps):
        print('\033[31m' + 'Computing emotions for frame:' + '\033[0m', frame_id)    
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = vidcap.read()
                   
        faces = vid_faces[frame_id]
        
        for face in faces:
            if face:
                emotion_id = classify_face_emotion(frame, face, model_name)
                emotions_counts [emotion_id] += 1

    emotions_counts = np.array(emotions_counts) / np.sum(emotions_counts) * 100
    emotions_counts = [float(f"{value:.2f}") for value in emotions_counts]
    
    if sum(emotions_counts)==0:
        emotions_percentages = "Not Applicable"
    else:
        for emotion_id in range(len(emotions_counts)):
            emotions_percentages[idx_to_class[emotion_id]] =  emotions_counts[emotion_id]
       
    vidcap.release()
     
    return emotions_percentages
    

def cast_detection(vid_path, start_frame, end_frame): 

    rekognition_client = boto3.client(
        'rekognition',
        aws_access_key_id = Config.rekognition_access_key,
        aws_secret_access_key = Config.rekognition_secret_key)
        
    cast = defaultdict(lambda: {'screen_presence': 0, 'gender': None})  
                
    vidcap = cv2.VideoCapture(vid_path)
    fps = round(vidcap.get(cv2.CAP_PROP_FPS))
        
    for frame_id in range (start_frame, end_frame, fps):
        print('\033[31m' + 'Detecting cast for frame:' + '\033[0m', frame_id)    
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = vidcap.read()
        
        frame_bytes = cv2.imencode('.jpg', frame)[1].tobytes()
        
        response = rekognition_client.recognize_celebrities(Image={'Bytes': frame_bytes})
        
        for celebrity in response['CelebrityFaces']:
            name = celebrity['Name']
            confidence = celebrity['MatchConfidence']
            gender = celebrity['KnownGender']['Type']
            bb_width = celebrity['Face']['BoundingBox']['Width']*frame.shape[1]
            bb_height = celebrity['Face']['BoundingBox']['Height']*frame.shape[0]
            
            if confidence > 85 and bb_width > 0.1 * frame.shape[1]:
                cast[name]['screen_presence'] += bb_width*bb_height
                cast[name]['gender'] = gender
                
    total_screen_presence = sum(actor['screen_presence'] for actor in cast.values())
    if total_screen_presence != 0:
        cast.update({name: {'screen_presence': entry['screen_presence'] * 100 / total_screen_presence,
                            'gender': entry['gender']}
                     for name, entry in cast.items()})
                     
    if not cast:
        cast = "No actor detected in the scene" 
        
    vidcap.release()
               
    return cast
    
                        
def precompute_metadata(vid_id, visual_model_type = "GPT4", audio_model_type = "WAVCAPS", stride = 10):

    video = Video.query.get(vid_id)
    if video is None:
        return
    last_reported = 0

    start_time = time.time()
                
    try:
           
        vid_path = f"{video.base_path}{video.path}"
        audio_path = f"{video.base_path}{video.audio_path}"
        metadata_path = f"{video.base_path}{video.metadata_path}"
        tms_path = f"{video.base_path}{video.timestamps_path}"
        precomp_shots_path = f"{video.base_path}{video.precomp_shots_path}"
        faces_path = f"{video.base_path}{video.precomp_faces_path}"
        
        with open(precomp_shots_path, 'rb') as file:
            shots = pickle.load(file)

        if audio_model_type == "WAVCAPS":           
            audio_checkpoint_path = f"{Config.ROOT_FOLDER}/app/ai_processing/metadata/audio_models/wavcaps/HTSAT_AudioCaps_Spider_48_5.pt"
            if os.path.exists(audio_checkpoint_path):
                print("HTSAT_AudioCaps_Spider_48_5.pt found!")
            else:
                print("Downloading HTSAT_AudioCaps_Spider_48_5.pt from S3 bucket")
                utils.download_from_s3(Config.ai_models_weights_s3, "HTSAT_AudioCaps_Spider_48_5.pt", audio_checkpoint_path)
                
            audio_cp = torch.load(audio_checkpoint_path, map_location=torch.device(device))

            audio_config = audio_cp["config"]
            audio_model = BartCaptionModel(audio_config)
            audio_model.load_state_dict(audio_cp["model"])
            audio_model.to(device)
            audio_feature_extractor=None
            audio_tokenizer=None

        elif audio_model_type == "WHISPER":           
            checkpoint = "MU-NLPC/whisper-tiny-audio-captioning"
            audio_model = WhisperForAudioCaptioning.from_pretrained(checkpoint).to(device)
            audio_tokenizer = WhisperTokenizer.from_pretrained(checkpoint, language="en", task="transcribe")
            audio_feature_extractor = WhisperFeatureExtractor.from_pretrained(checkpoint)
            
        else:
            print("Error, model: ", audio_model_type, "not supported")                              
                
        if has_audio(vid_path):
            if os.path.isfile(audio_path):
                pass
            else:
                wav_com = f"ffmpeg -y -hide_banner -loglevel error -i {str(vid_path)} -ac 1 {str(audio_path)}"
                os.system(wav_com)
        
        vidcap = cv2.VideoCapture(vid_path)
        fps = round(vidcap.get(cv2.CAP_PROP_FPS))
        length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if visual_model_type == "KOSMOS2":
            print("Using  Kosmos 2 model")        
            from transformers import AutoProcessor, AutoModelForVision2Seq
            image_captioning_model = AutoModelForVision2Seq.from_pretrained("microsoft/kosmos-2-patch14-224").to(device)
            vis_processors = AutoProcessor.from_pretrained("microsoft/kosmos-2-patch14-224")
            captioning_prompt = "<grounding>An image of"       
        elif visual_model_type == "BLIP":    
            print("Using BLIP model")
            from transformers import BlipProcessor, BlipForConditionalGeneration
            captioning_prompt = None                
            vis_processors = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
            image_captioning_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to(device)
        elif visual_model_type == "GPT4": 
            print("Using GPT4 model")         
            image_captioning_model = None
            vis_processors = None                      
            captioning_prompt = "Please describe the content of the following image in detail."   
        else:
            print("Error, model: ", visual_model_type, "not supported")                 

        metadata_list = []

        for i in range(fps, length, fps*stride):
            print("generating metadata at frame", i)
            
            start_frame = i
            end_frame = min(i + fps*stride, length)
            
            visual_descriptions, _ = visual_captioning(vid_path = vid_path, shots = shots, start_frame = start_frame, end_frame = end_frame, 
                                 image_captioning_model = image_captioning_model, vis_processors = vis_processors, 
                                 captioning_prompt = captioning_prompt, visual_model_type = visual_model_type)

            visual_descriptions = [visual_description[1] for visual_description in visual_descriptions]
            visual_description = ' '.join(visual_descriptions)
            
            if visual_model_type in ['BLIP', 'KOSMOS2']:
                pattern = re.compile(r'\b(?:arrafed|Arrafed|araffe|Araffe|arafed|Arafed)\b')
                visual_description = pattern.sub('', visual_description)   
                                                 
            f = open(tms_path, "r")
            word_infos = json.loads(f.read())
            filtered_words = [word_info[0] for word_info in word_infos if
                              (start_frame/fps)*1000 <= word_info[1] <= (end_frame/fps)*1000 and
                              (start_frame/fps)*1000 <= word_info[2] <= (end_frame/fps + stride)*1000] 
                              
            dialogue = ' '.join(filtered_words)
            
            sound_description = audio_captioning(vid_path, audio_path, start_frame, end_frame, 
                                audio_model_type, audio_model, audio_feature_extractor, audio_tokenizer)    
            
            multimodal_prompt = fill_prompt(default_multimodal_prompt, visual_description, sound_description, dialogue)
            multimodal_description = chatgpt_multimodal_description(multimodal_prompt)
            
            emotions_percentages = face_emotions(vid_path, faces_path, start_frame, end_frame, model_name='enet_b0_8_best_afew')
            
            cast = cast_detection(vid_path, start_frame, end_frame)
            
            metadata = {
                "start": start_frame/fps,
                "end": end_frame/fps,
                "visual_description": visual_description,
                "sound_description": sound_description,
                "dialogue": dialogue,
                "multimodal_description": multimodal_description, 
                "faces_emotions": emotions_percentages, 
                "cast": cast            
                }
                
            metadata_list.append(metadata)

        with open(metadata_path, 'w') as f:
            json.dump(metadata_list, f, indent=4)        
                
        vidcap.release()
        
        video.processed_metadata = 1
        session.commit()        
        
        end_time = time.time()
        print("generating metadata for video id:", vid_id, "took:", end_time-start_time)
        
    except Exception as error:
        print("precompute_metadata error", error)
        video.processing_failed = 1
        session.commit()







