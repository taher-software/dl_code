import time
import torch
import numpy as np
from app.beta.orm import session
from app.enums.upload_processing_status import UploadProcessingStatus
from app.beta.models.video import Video
from app.beta.models.image import Img
from app.utils import utils
import boto3
import json
from facenet_pytorch import InceptionResnetV1
from torchvision import transforms
from PIL import Image

import cv2
import dlib
from scenedetect import detect, ContentDetector, AdaptiveDetector, ThresholdDetector, SceneManager
from scenedetect.backends.opencv import VideoStreamCv2
from PIL import Image
import pickle
import clip 
import open_clip
from transformers import AutoProcessor, AutoModel
from sentence_transformers import SentenceTransformer
from app.services.upload_processing_service import UploadProcessingService
from config import Config
from app.ai_processing.editing.precomp_editing import convert_to_square
import app.ai_processing.editing.spiga.inference.config as model_cfg
from app.ai_processing.editing.spiga.inference.framework import SPIGAFramework
from app.ai_processing.editing.precomp_editing import landmarks_detection

from sqlalchemy import or_

from app.ai_processing.metadata.precomp_metadata import visual_captioning

# please do not remove this line
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using:", device)

    
def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result
        
        
def frame2frame_sims(video_embedding):
    """ Computes frame to frame successive similarities using the precomp_vide video embedding
        Arguments:
            video_embedding: precomp_vid array
    """    
    similarities = []
    prev_frame_embedding = video_embedding[0]
    for i in range (1, len(video_embedding), 1):
        frame_embedding = video_embedding[i]
        similarity = 100.0 * frame_embedding @ prev_frame_embedding.T
        similarities.append([i, similarity])
        prev_frame_embedding = frame_embedding        
    return similarities
    

def detect_shots_clip(vid_id, threshold = 90):
    """ AI shots splitting: splits videos into shots using adjacent frames similarities
        Arguments: 
            video_id: id of the the video to process
            threshold: similarity threshold to detect a shot change
        returns key frames numbers where the shots change, corresponding timestamps in miliseconds
    """  
    
    video = Video.query.get(vid_id)    
    fps = video.fps
    precomp_vid_path = f"{video.base_path}{video.precomp_vid_path}"
    video_embedding = np.load(precomp_vid_path, allow_pickle=True)
    similarities = frame2frame_sims(video_embedding)
    shot_changes = [sim[0] for sim in similarities if sim[1]<threshold]
    
    return shot_changes


def get_seconds (time_str):

    parts = time_str.split(":")
    hours = int(parts[0])
    minutes = int(parts[1])
    seconds_str = parts[2]

    seconds_parts = seconds_str.split(".")
    seconds = int(seconds_parts[0])
    milliseconds = int(seconds_parts[1])
    
    total_seconds = (hours * 3600) + (minutes * 60) + seconds + round((milliseconds / 1000), 2)

    return total_seconds
    
    
def detect_shots(vid_path, detector, start_time = None, end_time = None, show_progress = True, auto_downscale = False):

    shots = []
    video = VideoStreamCv2(vid_path)
    if start_time is not None:
        start_time = video.base_timecode + start_time
        video.seek(start_time)
    if end_time is not None:
        end_time = video.base_timecode + end_time
        
    vid_size = video.frame_size

    scene_manager = SceneManager()
    scene_manager.add_detector(detector)
    
    if auto_downscale:
       scene_manager.auto_downscale = True
    else:
       scene_manager.auto_downscale = False
       max_dim = max(vid_size)
       if max_dim>= 1920:
          scene_manager._downscale = 25
       elif max_dim>= 2048:
          scene_manager._downscale = 50
       elif max_dim>= 4096:
          scene_manager._downscale = 100
       else:
          scene_manager._downscale = 15        
               
    scene_manager.detect_scenes(video=video, show_progress=show_progress, end_time=end_time)
    scene_list = scene_manager.get_scene_list()
    
    for i in range (len(scene_list)-1):
        scene = scene_list[i]
        end_frames = scene[1].get_frames()
        end_timecode = scene[1].get_timecode()
        end_seconds = get_seconds (end_timecode)
        shots.append(end_frames)

    return shots
    
        
def precompute_visuals_no_captioning(vid_id, squaring = True):
    """ processes the images frames at a defined framerate and generates visual emdeddings in the form of a numpy array
        Arguments:
            vid_path: path of the video file to be processed
            precomp_vid_path: path of the output file            
    """
    video = Video.query.get(vid_id)
    if video is None:
        return
    last_reported = 0
    
    try:
        visual_embedding_model = Config.VISUAL_EMBEDDING_MODEL
        
        if visual_embedding_model == "CLIP_VIT_B_32":
           print("Using OpenAI Clip model")
           model, preprocess = clip.load("ViT-B/32", device=device)
        elif visual_embedding_model == "OPEN_CLIP_VIT_L_14":
           print("Using LAION Clip model")
           model, _, preprocess = open_clip.create_model_and_transforms('ViT-L-14', pretrained = 'datacomp_xl_s13b_b90k', device=device)
        elif visual_embedding_model == "SIGLIP_BASE_16_224":
           print("Using Google Siglip model")                        
           model = AutoModel.from_pretrained("google/siglip-base-patch16-224").to(device)
           preprocess = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")        
        else:
           print("Error: Model not supported")
           return
           
        vid_path = f"{video.base_path}{video.path}"
        precomp_vid_path = f"{video.base_path}{video.precomp_vid_path}"

        vidcap = cv2.VideoCapture(vid_path)
        fps = round(vidcap.get(cv2.CAP_PROP_FPS))
        length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

        precomp = []
        start_time = time.time()

        for i in range(fps, length, fps):
            if i%(30*fps)==0:
                print("precomputing video frame", i)

            vidcap.set(cv2.CAP_PROP_POS_FRAMES, i)
            success, frame = vidcap.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            if squaring:
               frame = expand2square(Image.fromarray(frame),(0, 0, 0))
            else:
               frame = Image.fromarray(frame)
               
            with torch.no_grad():            
                if visual_embedding_model in ["CLIP_VIT_B_32", "OPEN_CLIP_VIT_L_14"]:
                    image = preprocess(frame).unsqueeze(0).to(device)
                    image_embedding = model.encode_image(image)
                else:
                    image = preprocess(images=frame, return_tensors="pt").to(device)
                    image_embedding = model.get_image_features(**image)     
                                   
            image_embedding /= image_embedding.norm(dim=-1, keepdim=True)
            precomp.append(image_embedding.cpu().numpy())      
                      
            try:
                if utils.should_report(last_reported, (i / length)):
                    percentage = (i / length) * 100
                    last_reported = (i / length)
                    estimated_full_duration = ((time.time() - start_time) / percentage) * 100
                    utils.inform_in_thread(vid_id, UploadProcessingStatus.VISUAL_PROCESSING, percentage, estimated_full_duration)
            except:
                pass

        vidcap.release()
        
        video.processed_pc = 1
        video.visual_embedding_model = visual_embedding_model
        session.commit()
        UploadProcessingService.inform(vid_id, UploadProcessingStatus.VISUAL_PROCESSING, 100)
        np.save(precomp_vid_path, precomp)

    except Exception as error:
        print("precompute_visuals_no_context error", error)
        video.processing_failed = 1
        video.visual_embedding_model = visual_embedding_model
        session.commit()
        UploadProcessingService.inform(vid_id, UploadProcessingStatus.VISUAL_PROCESSING, 100, -1, True)
        
def precompute_visual_metadata(vid_id, visual_model_type = "GPT4", stride = 5):

    video = Video.query.get(vid_id)
    if video is None:
        return
    last_reported = 0

    start_time = time.time()
                
    try:
           
        vid_path = f"{video.base_path}{video.path}"
        visual_metadata_path = f"{video.base_path}{video.visual_metadata_path}"
        precomp_shots_path = f"{video.base_path}{video.precomp_shots_path}"
        
        vidcap = cv2.VideoCapture(vid_path)
        fps = round(vidcap.get(cv2.CAP_PROP_FPS))
        length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))        
        
        with open(precomp_shots_path, 'rb') as file:
            shots = pickle.load(file)
            
        if shots[0]>0:
            shots.insert(0, 0)
        if shots[-1] < (length):
            shots.append(length)
        
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
            captioning_prompt = "Please describe the content of the following image."   
        else:
            print("Error, model: ", visual_model_type, "not supported")                 

        visual_metadata_list = []

        for i in range (len(shots)-1):  
            start_frame = shots[i]           
            end_frame = shots[i+1]
            
            visual_descriptions, clip_shots = visual_captioning(vid_path = vid_path, shots = shots, start_frame = start_frame, end_frame = end_frame, 
                                 image_captioning_model = image_captioning_model, vis_processors = vis_processors, 
                                 captioning_prompt = captioning_prompt, visual_model_type = visual_model_type)
                        
            visual_metadata = {
                "start_in_frames": start_frame,
                "end_in_frames": end_frame,
                "visual_descriptions": visual_descriptions,  
                "clip shots": clip_shots        
                }
                
            visual_metadata_list.append(visual_metadata)

        with open(visual_metadata_path, 'w') as f:
            json.dump(visual_metadata_list, f, indent=4)        
                
        vidcap.release()
        
        video.processed_visual_metadata = 1
        session.commit()        
        
        end_time = time.time()
        print("generating visual metadata for video id:", vid_id, "took:", end_time-start_time)
        
    except Exception as error:
        print("precompute visual metadata error", error)
        video.processed_visual_metadata = 0
        video.processing_failed = 1
        session.commit()
        
        
def precompute_visuals_captioning(vid_id, visual_model_type = "GPT4", stride = 5):
    video = Video.query.get(vid_id)
    if video is None:
        return

    try:
        visual_metadata_path = f"{video.base_path}{video.visual_metadata_path}"
        precomp_vid_path = f"{video.base_path}{video.precomp_vid_path}"        
                      
        dialogue_embedding_model = Config.DIALOGUE_EMBEDDING_MODEL
            
        if dialogue_embedding_model == "ALL-MINILM-L6-V2":
            print("Using ALL-MINILM-L6-V2 model")
            model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        else:
            print("Model not supported")
            return
           
        with open(visual_metadata_path, 'r') as file:
            visual_metadata = json.load(file)
        
        precomp = []

        for clip_item in (visual_metadata):
            visual_descriptions = clip_item["visual_descriptions"]
            start_frame = clip_item["start_in_frames"]
            end_frame = clip_item["end_in_frames"]       
             
            if dialogue_embedding_model in ["ALL-MINILM-L6-V2"]:
                print("precomputing visual metadata at start frame", start_frame)
                visual_descriptions = [sentence for visual_description in visual_descriptions for sentence in visual_description[1].split('.')]
                descriptions_embeddings = [model.encode(visual_description) for visual_description in visual_descriptions]
                precomp.append(np.array(descriptions_embeddings))

        with open(precomp_vid_path, 'wb') as file:
            pickle.dump(precomp, file)
        
        video.processed_pc = 1
        video.visual_embedding_model = dialogue_embedding_model
        session.commit()
        UploadProcessingService.inform(vid_id, UploadProcessingStatus.VISUAL_PROCESSING, 100)

    except Exception as error:
        print("precompute_visuals_captioning error", error)
        video.processing_failed = 1
        video.visual_embedding_model = dialogue_embedding_model
        session.commit()
        UploadProcessingService.inform(vid_id, UploadProcessingStatus.VISUAL_PROCESSING, 100, -1, True)


def precompute_visuals(vid_id):
    utils.inform_in_thread(vid_id, UploadProcessingStatus.VISUAL_PROCESSING)
    if Config.VISUAL_CAPTIONING:
       precompute_visual_metadata(vid_id)
       precompute_visuals_captioning(vid_id)
    else:
       precompute_visuals_no_captioning(vid_id)
       
    
def precompute_shots(vid_id):
    """ processes the images frames at a defined framerate and generates visual emdeddings in the form of a numpy array
        Arguments:
            vid_path: path of the video file to be processed
            precomp_vid_path: path of the output file            
    """
    video = Video.query.get(vid_id)
    if video is None:
        return
    last_reported = 0
    
    try:
           
        vid_path = f"{video.base_path}{video.path}"
        precomp_shots_path = f"{video.base_path}{video.precomp_shots_path}"

        shots = detect_shots(vid_path, AdaptiveDetector())
        
        video.processed_shots = 1
        session.commit()
                
        with open(precomp_shots_path, 'wb') as file:
            pickle.dump(shots, file)
                
    except Exception as error:
        print("precompute_shots error", error)
        video.processed_shots = 0
        session.commit()
        
        
def precompute_thumbnails(vid_id, window = 5, save_all_thumbs = False):

    video = Video.query.get(vid_id)
    if video is None:
        return
    last_reported = 0
    
    try:           
        vid_path = f"{video.base_path}{video.path}"
        thumbnails_base_path = f"{video.base_path}{video.thumbnails_base_path}"
        thumbnails_mapping_path = thumbnails_base_path + "_thumbnails_mapping.pkl"
        stitched_thumbnails_path = thumbnails_base_path + "_thumbnails_stitched.png"
        
        thumbnails_mapping = []    

        vidcap = cv2.VideoCapture(vid_path)
        fps = round(vidcap.get(cv2.CAP_PROP_FPS))
        length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))        

        tb_stitched = []
        
        for i in range(length):
            if i % (fps*window*6) ==0:
                print("precomputing thumbnail for video frame", i)

                vidcap.set(cv2.CAP_PROP_POS_FRAMES, i)
                success, frame = vidcap.read()
                
                if width > height:
                    target_width, target_height = 100, 56
                    frame = cv2.resize(frame, (target_width, target_height))
                    
                else:
                    target_width, target_height = 56, 100
                    frame = cv2.resize(frame, (target_width, target_height))
                    
                tb_stitched.append(frame)
                 
                if save_all_thumbs:
                   thumbnail_path = thumbnails_base_path + "_thumbnail_" + str(i) + ".png"
                   thumbnails_mapping.append([i, thumbnail_path])
                
                   cv2.imwrite (thumbnail_path, frame)  
                
        stitched_image = np.concatenate(tb_stitched, axis=1)
        print(stitched_thumbnails_path)
        cv2.imwrite (stitched_thumbnails_path, stitched_image) 
        
        if save_all_thumbs:                
            with open(thumbnails_mapping_path, 'wb') as file:
                pickle.dump(thumbnails_mapping, file)     
                           
        vidcap.release()
                                    
    except Exception as error:
        print("precompute_thumbnails error", error)


def precompute_base(vid_id):

    video = Video.query.get(vid_id)
    if video is None:
        return
    last_reported = 0
    
    try:
        precompute_thumbnails(vid_id)   
        precompute_shots(vid_id)    
                
    except Exception as error:
        print("precompute_base error", error)
       

def process_face(frame, face, save_face_embeddings, face_recognition_embedding_model, 
                 recognition_preprocess, recognition_model, save_landmarks, processor):
    if 'Face' in face:  
        is_celebrity = True
        bounding_box = face['Face']['BoundingBox']
        emotions = face['Face']['Emotions']
        name = face['Name']
        id = face['Id']
        confidence = face['MatchConfidence']
        gender = face['KnownGender']['Type']
    else: 
        is_celebrity = False
        bounding_box = face['BoundingBox']
        emotions = face['Emotions']
        name = None
        id = None
        confidence = None
        gender = None

    emotion = max(emotions, key=lambda x: x['Confidence'])['Type']
    bb_width = bounding_box['Width'] * frame.shape[1]
    bb_height = bounding_box['Height'] * frame.shape[0]
    bb_x = bounding_box['Left'] * frame.shape[1]
    bb_y = bounding_box['Top'] * frame.shape[0]

    if save_face_embeddings:
        x, y, size = convert_to_square((bb_x, bb_y, bb_width, bb_height))
        face_image = frame[y:y+size, x:x+size] 
        with torch.no_grad():           
            if face_recognition_embedding_model == "DLIB_FREC_RES":
                face_image = cv2.resize(face_image, (150, 150))
                face_embedding = np.array(recognition_model.compute_face_descriptor(face_image))  
            elif face_recognition_embedding_model == "FACENET_VGG2":           
                face_image = Image.fromarray(face_image)
                face_image = recognition_preprocess(face_image).unsqueeze(0)        
                face_embedding = recognition_model(face_image).cpu().numpy()                             
            else:
                print("Error: Model not supported")
                face_embedding = None
    else:
        face_embedding = None
                    
    if save_landmarks:
        landmarks = landmarks_detection(frame, [[bb_x, bb_y, bb_width, bb_height]], processor)
    else:
        landmarks = None

    if is_celebrity and confidence >= 90:
        face_cast = [bb_x, bb_y, bb_width, bb_height, landmarks, face_embedding, emotion, name, id, gender]
    else:
        face_cast = [bb_x, bb_y, bb_width, bb_height, landmarks, face_embedding, emotion, None, None, gender]  
           
    return face_cast
   
   
def precompute_cast(vid_id, save_face_embeddings = True, save_landmarks = False): 

    rekognition_client = boto3.client(
        'rekognition',
        aws_access_key_id = Config.rekognition_access_key,
        aws_secret_access_key = Config.rekognition_secret_key,
        region_name = Config.rekognition_region_name)

    video = Video.query.get(vid_id)
    if video is None:
        return
    last_reported = 0
    
    face_recognition_embedding_model = Config.FACE_RECOGNITION_EMBEDDING_MODEL
    
    try:           
        vid_path = f"{video.base_path}{video.path}" 
        precomp_cast_path = f"{video.base_path}{video.precomp_cast_path}"        
        precomp_shots_path = f"{video.base_path}{video.precomp_shots_path}"

        with open(precomp_shots_path, 'rb') as file:
            shots= pickle.load(file)  
                            
        vidcap = cv2.VideoCapture(vid_path)
        fps = round(vidcap.get(cv2.CAP_PROP_FPS))
        length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT)) 
        
        precomp_cast = []
        
        processor = None
        
        if save_landmarks:
            processor_cfg = model_cfg.ModelConfig('wflw')
            processor = SPIGAFramework(processor_cfg, device = device)
            
        if save_face_embeddings:
            if face_recognition_embedding_model == "DLIB_FREC_RES":
                print("Using DLIB RESNET face recognition model")
                recognition_model = dlib.face_recognition_model_v1(f"{Config.ROOT_APP_FOLDER}/app/ai_processing/editing/dlib/dlib_face_recognition_resnet_model_v1.dat")
                recognition_preprocess = None
            elif face_recognition_embedding_model == "FACENET_VGG2": 
                print("Using FACENET VGG2 face recognition model")               
                recognition_preprocess = transforms.Compose([
                                    transforms.Resize((160, 160)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                                    ])
                recognition_model = InceptionResnetV1(pretrained='vggface2').eval()
            else:
                print("Error: Model not supported")
                save_face_embeddings = False              
        
        for frame_id in range (0, length, fps):
            if frame_id%(30*fps)==0:
                print('\033[31m' + 'Detecting cast for frame:' + '\033[0m', frame_id)    
            vidcap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            ret, frame = vidcap.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            frame_cast = []
        
            frame_bytes = cv2.imencode('.jpg', frame)[1].tobytes()
        
            response = rekognition_client.recognize_celebrities(Image={'Bytes': frame_bytes})
            all_faces = response['CelebrityFaces'] + response['UnrecognizedFaces']

            for face in all_faces:
                face_cast = process_face(frame, face, save_face_embeddings, face_recognition_embedding_model, 
                                         recognition_preprocess, recognition_model, save_landmarks, processor)
                frame_cast.append(face_cast)

            num_faces = len(frame_cast)
            if num_faces == 0:
                frame_cast = [None]  
                      
            for i in range(fps):
                if i+frame_id in shots and i>0:
                    print('\033[34m' + 'processing frame for a change in shots:' + '\033[0m', i+frame_id)
                    vidcap.set(cv2.CAP_PROP_POS_FRAMES, i+frame_id)
                    ret, frame = vidcap.read()                                    
                    if not ret:
                        break                    
                    frame_bytes = cv2.imencode('.jpg', frame)[1].tobytes()   
                    response = rekognition_client.recognize_celebrities(Image={'Bytes': frame_bytes})
                        
                    frame_cast = []

                    all_faces = response['CelebrityFaces'] + response['UnrecognizedFaces']
                    for face in all_faces:
                        face_cast = process_face(frame, face, save_face_embeddings, face_recognition_embedding_model, 
                                         recognition_preprocess, recognition_model, save_landmarks, processor)
                        frame_cast.append(face_cast)
                                                                                                            
                    num_faces = len(frame_cast)                    
                    if num_faces == 0:
                        frame_cast = [None]   

                if i+frame_id <= length:                 
                    precomp_cast.append(frame_cast)
         
        vidcap.release()
           
        with open(precomp_cast_path, 'wb') as file:
            pickle.dump(precomp_cast, file)

        if save_face_embeddings: 
            video.processed_faces = 1
            video.face_recognition_embedding_model = face_recognition_embedding_model      
                           
        video.processed_cast = 1
        session.commit()

    except Exception as error:
        print("precompute_cast error", error)
        video.processed_cast = 0
        session.commit()

def precompute_face(img_id):
    image = Img.query.get(img_id)
    if image is None:
        return    
        
    img_path = f"{Config.ROOT_FOLDER}/app/upload/{image.location}"
    precomp_img_path = f"{Config.ROOT_FOLDER}/app/upload/{image.precomp_img_path}"
    bbox = image.bbox
    bbox = [int(value) for value in bbox.split(",")]
    
    face = cv2.imread(img_path)
    
    try:        
        face_recognition_embedding_model = Config.FACE_RECOGNITION_EMBEDDING_MODEL 
        if face_recognition_embedding_model == "DLIB_FREC_RES":
            print("Using DLIB RESNET face recognition model")
            model = dlib.face_recognition_model_v1(f"{Config.ROOT_APP_FOLDER}/app/ai_processing/editing/dlib/dlib_face_recognition_resnet_model_v1.dat")
        elif face_recognition_embedding_model == "FACENET_VGG2": 
            print("Using FACENET VGG2 face recognition model")               
            preprocess = transforms.Compose([
                                transforms.Resize((160, 160)),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                                ])
            model = InceptionResnetV1(pretrained='vggface2').eval()
        else:
            print("Error: Model not supported")
            return
               
        with torch.no_grad():  
            x, y, size = convert_to_square(bbox)
            face = face[y:y+size, x:x+size]           
            if face_recognition_embedding_model == "DLIB_FREC_RES":         
                face  = cv2.resize(face, (150, 150)) 
                face_embedding = np.array(model.compute_face_descriptor(face))
            elif face_recognition_embedding_model == "FACENET_VGG2":           
                face = Image.fromarray(face)
                face = preprocess(face).unsqueeze(0)        
                face_embedding = model(face).cpu().numpy()
                            
        np.save(precomp_img_path, face_embedding) 

        image.processed_viz = 1
        image.visual_embedding_model = face_recognition_embedding_model
        session.commit()

    except Exception as error:
        print("precompute_image", error)
        image.processing_failed = 1
        image.visual_embedding_model = face_recognition_embedding_model
        session.commit()

