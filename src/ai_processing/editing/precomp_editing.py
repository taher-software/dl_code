import cv2
import numpy as np
import math
import dlib
import pickle
import app.ai_processing.editing.det_config as det_cfg
import app.ai_processing.editing.spiga.inference.config as model_cfg
from app.ai_processing.editing.spiga.inference.framework import SPIGAFramework
import retinaface
import clip
import torch
from facenet_pytorch import InceptionResnetV1
from torchvision import transforms
from PIL import Image
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine

from app.beta.orm import session
from app.beta.models.video import Video
from app.services.upload_processing_service import UploadProcessingService
from config import Config

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using:", device)
        
def euclidean_dist(point1, point2):

    x1, y1 = point1
    x2, y2 = point2
    
    dist = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    return dist
    
def check_euclid(x, y, prev_x, prev_y, w, dist_threshold):

    w_threshold = 1.5*w
    match = False

    dist = euclidean_dist([x, y], [prev_x, prev_y])
    if dist<dist_threshold and dist<w_threshold:
       match = True   

    return match
    
def size_dist(w, prev_w):
    w_small, w_big = sorted([w, prev_w])
    match = False
    if w_big/w_small<1.3:
       match = True  
    return match
   
def cosine_dist(vector1, vector2):

    match = False
    cosine_sim = 1 - cosine(vector1, vector2)
    similarity_percentage = (cosine_sim + 1) * 50
    
    if similarity_percentage>80:
       match = True
    return match

    
def active_speaker_detection(landmarks, threshold):
      
    speaking = False

    outer_mouth = landmarks[76:88]
    inner_mouth = landmarks[88:96]
        
    left_lip = landmarks[88]
    right_lip = landmarks[92]
    top_lip = landmarks[90]
    bottom_lip = landmarks[94]
   
    major_axis = euclidean_dist(left_lip, right_lip)
    minor_axis = euclidean_dist(top_lip, bottom_lip)
   
    delta_small, delta_big = sorted([major_axis, minor_axis])
            
    if delta_small>0:
       ratio = delta_big/delta_small
       if ratio<threshold:
          speaking = True
       
    return speaking

def check_match (current_face, prev_faces, current_embedding, dist_threshold, use_embedding = False):    
    match = False
    
    x, y, w = current_face
              
    for prev_face in prev_faces:
       prev_x, prev_y, prev_w, prev_h, prev_id, prev_embedding = prev_face    
       
       dist_match = check_euclid(x, y, prev_x, prev_y, w, dist_threshold)
       size_match = size_dist(w, prev_w)
       if use_embedding:
          embedding_match = cosine_dist(prev_embedding, current_embedding)
       
       if size_match and dist_match: #and embedding_match:
          match = True
          return match, prev_id, prev_face
       else:
          continue
    return match, None, None
                                                             
def landmarks_detection(frame, bbox, processor, model='spiga'):

    if model == 'spiga': 
       landmarks = processor.inference(frame, bbox)['landmarks'][0]

    else:
       predictor_path = f"{Config.ROOT_FOLDER}/app/ai_processing/editing/dlib/shape_predictor_68_face_landmarks.dat"
       predictor = dlib.shape_predictor(predictor_path)
    
       gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
       face_rect = dlib.rectangle(bbox[0][0], bbox[0][1], bbox[0][2], bbox[0][3])
    
       landmarks = predictor(gray, face_rect)
       landmarks = [[landmarks.part(i).x, landmarks.part(i).y] for i in range(0, 69)]
       
    return landmarks
    
def convert_to_square(bbox):
    x, y, w, h = bbox
    size = max(w, h)
    bbox_center_x, bbox_center_y = x+w/2, y+h/2
    new_x, new_y = max(0, bbox_center_x -size/2), max(0, bbox_center_y - size/2)
    return [int(new_x), int(new_y), int(size)]
   
def face_embedd(frame, bbox, face_recognition_embedding_model, recognition_preprocess, recognition_model):

    x, y, size = convert_to_square(bbox)
    face = frame[y:y+size, x:x+size]
    
    if face_recognition_embedding_model=="DLIB_FREC_RES":
       face  = cv2.resize(face, (150, 150)) 
       face_embedding = np.array(recognition_model.compute_face_descriptor(face))       
    elif face_recognition_embedding_model == "FACENET_VGG2":           
       face = Image.fromarray(face)
       face = recognition_preprocess(face).unsqueeze(0)        
       face_embedding = recognition_model(face).cpu().numpy()          
    else:
       print("Model:", face_recognition_embedding_model, " not supported")
       face_embedding = None

    return face_embedding
       
def detect_faces(frame, detector, landmarks_processor, face_recognition_embedding_model, recognition_preprocess, recognition_model):

    detector.set_input_shape(frame.shape[0], frame.shape[1])                                                  
    faces = detector.inference(frame)['bbox']

    frame_faces = []
    
    for i, face in enumerate(faces):
       
       x = int(face[0])
       y = int(face[1])
       w = int(face[2])-x
       h = int(face[3])-y
       
       if face_recognition_embedding_model:
          embedding = face_embedd(frame, [x, y, w, h], face_recognition_embedding_model, 
                                  recognition_preprocess, recognition_model) 
       else:
          embedding = None
          
       if landmarks_processor:
          landmarks = landmarks_detection(frame, [[x, y, w, h]], landmarks_processor)
       else:
          landmarks = None
          
       frame_faces.append([x, y, w, h, landmarks, embedding])
       
    return frame_faces
    
def track_faces(precomp_faces, video):
    
    prev_faces = []
    tracked_faces = []
    speakers = []

    dist_thresh = (video.height * 9/16)
    
    vid_path = f"{video.base_path}{video.path}"    
    precomp_shots_path = f"{video.base_path}{video.precomp_shots_path}"
    vidcap = cv2.VideoCapture(vid_path)
    length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    with open(precomp_shots_path, 'rb') as file:
        shots= pickle.load(file)  

    shots.append(length)
    
    shot_id = 0
            
    for frame_id, frame_faces in enumerate(precomp_faces):
       print('\033[36m' + 'frame:' + '\033[0m', frame_id)

       if frame_id > shots[shot_id]:
          shot_id += 1
          prev_faces = []
       
       tracked_frame_faces = []
       frame_speakers = []

       for j, face in enumerate(frame_faces):
           if face:
              x, y, w, h = face[:4]
              landmarks = face[4]
              embedding = face[5]
                          
              if len(frame_faces)==1:
                 speaking =True

              else:
                 speaking = active_speaker_detection(landmarks, threshold = 10)
      
              match, face_id, prev_face = check_match([x, y, w], prev_faces, embedding, dist_thresh)
                 
              if match and frame_id < shots[shot_id]: 

                 prev_x, prev_y, prev_w, prev_h, prev_id, prev_embedding = prev_face                      
                
                 if speaking:
                    frame_speakers.append(prev_id)            
                
                 tracked_frame_faces.append(prev_faces[prev_id])
                
              else:
                 if len(prev_faces)>0:
                    max_id = max([prev_face[4] for prev_face in prev_faces])
                    face_id = max_id +1
                 else:
                    face_id = 0
             
                 if speaking: 
                    frame_speakers.append(face_id) 
              
                 prev_faces.append([x, y, w, h, face_id, embedding])              
                 tracked_frame_faces.append([x, y, w, h, face_id, embedding])
                 
       vidcap.release()
       
       tracked_frame_faces = sorted(tracked_frame_faces, key=lambda x: x[0]) 
       tracked_faces.append(tracked_frame_faces)
       speakers.append(frame_speakers)
       
    return tracked_faces, speakers

def precompute_faces(vid_id, save_landmarks = True, save_embedding = True):

    cfg = det_cfg.cfg_retinasort
    detector = retinaface.RetinaFaceDetector(model=cfg['retina']['model_name'], device = device, extra_features=cfg['retina']['extra_features'], cfg_postreat=cfg['retina']['postreat'])

    landmarks_processor = None
    if save_landmarks:
        processor_cfg = model_cfg.ModelConfig('wflw')
        landmarks_processor = SPIGAFramework(processor_cfg, device = device)

    if save_embedding:     
        face_recognition_embedding_model = Config.FACE_RECOGNITION_EMBEDDING_MODEL
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
            save_embedding = False    
            face_recognition_embedding_model = None        
    else:
        recognition_preprocess = None
        recognition_model = None
        face_recognition_embedding_model = None    
 
    video = Video.query.get(vid_id)
    if video is None:
        return
        
    vid_path = f"{video.base_path}{video.path}"
    precomp_faces_path = f"{video.base_path}{video.precomp_faces_path}" 
    precomp_shots_path = f"{video.base_path}{video.precomp_shots_path}"
    
    vidcap = cv2.VideoCapture(vid_path)
    fps = round(vidcap.get(cv2.CAP_PROP_FPS))
    length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT)) 

    with open(precomp_shots_path, 'rb') as file:
        shots= pickle.load(file)  
        
    precomp_faces = []
    
    for frame_id in range(0, length, fps):
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = vidcap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)        

        if frame_id%(30*fps)==0:
            print('\033[31m' + 'processing frame:' + '\033[0m', frame_id)

        if not ret:
            break
                
        faces = detect_faces(frame, detector, landmarks_processor, face_recognition_embedding_model, 
                             recognition_preprocess, recognition_model)        

        num_faces = len(faces)
        
        if num_faces == 0:
            faces = [None]

        for i in range(fps):
            if i+frame_id in shots and i>0:
                print('\033[34m' + 'processing frame for a change in shots:' + '\033[0m', i+frame_id)
                vidcap.set(cv2.CAP_PROP_POS_FRAMES, i+frame_id)
                ret, frame = vidcap.read()                                    
                if not ret:
                    break                    
                faces = detect_faces(frame, detector, landmarks_processor, face_recognition_embedding_model,
                             recognition_preprocess, recognition_model)                        
                num_faces = len(faces)                    
                if num_faces == 0:
                    faces = [None]   
            
            if i+frame_id <= length:                 
                precomp_faces.append(faces) 
        
    with open(precomp_faces_path, 'wb') as file:
        pickle.dump(precomp_faces, file)

    vidcap.release()
    
    video.processed_faces = 1
    if save_embedding:
        video.face_recognition_embedding_model = face_recognition_embedding_model
    UploadProcessingService.auto_tiktok(vid_id)
    session.commit()
         
def precompute_tracked_faces(vid_id, process_cast = False):

    video = Video.query.get(vid_id)
    if video is None:
        return
        
    precomp_faces_path = f"{video.base_path}{video.precomp_faces_path}"
    precomp_cast_path = f"{video.base_path}{video.precomp_cast_path}"
    tracked_faces_path = f"{video.base_path}{video.tracked_faces_path}"
    speakers_path = f"{video.base_path}{video.speakers_path}" 
        
    if process_cast:
        with open(precomp_cast_path, 'rb') as file:
            precomp_faces = pickle.load(file)
    else:
        with open(precomp_faces_path, 'rb') as file:
            precomp_faces = pickle.load(file)        
                
    tracked_faces, speakers = track_faces(precomp_faces, video)
    
    with open(tracked_faces_path, 'wb') as file:
        pickle.dump(tracked_faces, file)
        
    with open(speakers_path, 'wb') as file:
        pickle.dump(speakers, file)

    video.processed_face_tracking = 1
    session.commit()
    if video.get_clips == 1:
       UploadProcessingService.auto_tiktok(vid_id)
    
def precompute_face_tracking(vid_id, save_landmarks = True, save_embedding = True, process_cast = Config.PROCESS_CAST):

   if not process_cast:
       precompute_faces(vid_id, save_landmarks, save_embedding)
 
   precompute_tracked_faces(vid_id, process_cast = process_cast)
    
def map_emoji_json (emoji_data_path, output_json_path):    
   with open(emoji_data_path, 'r', encoding='utf-8') as json_file:
       emoji_data = json.load(json_file)

   emoji_mappings = {}

   for emoji_entry in emoji_data:
       emoji_char = emoji_entry['emoji']
       description = emoji_entry['description']
    
       unicode_code_points = [ord(c) for c in emoji_char]    
       hex_unicode = '-'.join([f'{code:X}' for code in unicode_code_points])
    
       emoji_mappings[hex_unicode.lower()] = description

   with open(output_json_path, 'w', encoding='utf-8') as output_file:
       json.dump(emoji_mappings, output_file, ensure_ascii=False, indent=4)
   
   return emoji_mappings   

def map_emoji_html (emoji_data_path, output_json_path):
   with open(emoji_data_path, 'r') as file:
       html_content = file.read()

   soup = BeautifulSoup(html_content, 'html.parser')
   emoji_mappings = {}
   emoji_rows = soup.find_all('tr')

   for row in emoji_rows:
       cols = row.find_all('td')
       if len(cols) >= 3:
           hexcode = cols[1].text.strip().lower().replace(" ", "-")
           description = cols[2].find('b').text.strip()
           emoji_mappings[hexcode] = description

   with open(output_json_path, 'w', encoding='utf-8') as json_file:
       json.dump(emoji_mappings, json_file, ensure_ascii=False, indent=4)
       
   return emoji_mappings

def precomp_emojis (emojis_paths, emoji_mappings_json_path, emoji_mapping_html_path, precomp_emojis_path):

   lm_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

   device = "cuda" if torch.cuda.is_available() else "cpu"
   # please do not remove this line
   print("Using:", device)
   clip_model, preprocess = clip.load("ViT-B/32", device=device)

   emojis_paths = get_files(emojis_path)
   with open(emoji_mappings_json_path, 'r', encoding='utf-8') as json_file:
       emoji_mappings = json.load(json_file)
   with open(emoji_mappings_html_path, 'r', encoding='utf-8') as json_file:
       emoji_mappings_bis = json.load(json_file)       
   error_counter = 0

   skin_tone_modifiers = ['1f3fb', '1f3fc', '1f3fd', '1f3fe', '1f3ff']
   variation_selectors = ['fe0f', 'fe0e']

   emoji_keys = emoji_mappings.keys()
   emoji_keys_bis = emoji_mappings_bis.keys()

   precomp_emojis = []

   for emoji_path in emojis_paths:
      print(emoji_path)
   
      uni_code = emoji_path.split('/')[-1].split('.png')[0]   
   
      components = uni_code.split('-')
      components_no_skin = [c for c in components if c not in skin_tone_modifiers]
 
      uni_code_no_skin = '-'.join(components_no_skin)

      with torch.no_grad():
         emoji = preprocess(Image.open(emoji_path)).unsqueeze(0).to(device)
         embedding = clip_model.encode_image(emoji)
         embedding /= embedding.norm(dim=-1, keepdim=True)
         embedding_viz = embedding.T.cpu().numpy()  
            
      if uni_code_no_skin in emoji_keys:
         description = emoji_mappings[uni_code_no_skin]
         embedding_txt = lm_model.encode(description)

         """
         with torch.no_grad():
            query = clip.tokenize(description).to(device)         
            embedding = clip_model.encode_text(query)
            embedding  /= embedding.norm(dim=-1, keepdim=True)
            embedding  = embedding.T.cpu().numpy()
         """
                  
      elif uni_code_no_skin in emoji_keys_bis:
         description = emoji_mappings_bis[uni_code_no_skin]
         embedding_txt = lm_model.encode(description)
      
         """
         with torch.no_grad():
            query = clip.tokenize(description).to(device)         
            embedding = clip_model.encode_text(query)
            embedding  /= embedding.norm(dim=-1, keepdim=True)
            embedding  = embedding.T.cpu().numpy()
         """
            
      else:
         main_code = components_no_skin[0]
         components_no_variation = [c for c in components_no_skin if c not in variation_selectors]
         clean_code = '-'.join(components_no_variation)
      
         neighbour_codes = [hexcode.split('-') for hexcode in emoji_keys if hexcode.startswith(main_code)]
         neighbour_codes_bis = [hexcode.split('-') for hexcode in emoji_keys_bis if hexcode.startswith(main_code)]
                  
         neighbour_codes_no_var = [[c for c in neighbour if c not in variation_selectors] for neighbour in neighbour_codes]
         neighbour_codes_no_var = ['-'.join(hex_code) for hex_code in neighbour_codes_no_var]
         
         neighbour_codes_no_var_bis = [[c for c in neighbour if c not in variation_selectors] for neighbour in neighbour_codes_bis]
         neighbour_codes_no_var_bis = ['-'.join(hex_code) for hex_code in neighbour_codes_no_var_bis]
      
         index = None
         
         if clean_code in neighbour_codes_no_var:
            index = neighbour_codes_no_var.index(clean_code)
            index_code = "-".join(neighbour_codes[index])
            description = emoji_mappings[index_code]
         
            embedding_txt = lm_model.encode(description)  
                
            """
            with torch.no_grad():
               query = clip.tokenize(description).to(device)         
               embedding = clip_model.encode_text(query)
               embedding  /= embedding.norm(dim=-1, keepdim=True)
               embedding  = embedding.T.cpu().numpy()
            """
            
         elif clean_code in neighbour_codes_no_var_bis:
            index = neighbour_codes_no_var_bis.index(clean_code)
            index_code = "-".join(neighbour_codes_bis[index])            
            description = emoji_mappings_bis[index_code]

            embedding_txt = lm_model.encode(description)
      
            """
            with torch.no_grad():
               query = clip.tokenize(description).to(device)         
               embedding = clip_model.encode_text(query)
               embedding  /= embedding.norm(dim=-1, keepdim=True)
               embedding  = embedding.T.cpu().numpy()
            """
                    
         else:
            embedding_txt = None      

      precomp_emojis.append([emoji_path.split('/')[-1], uni_code, description, embedding_txt, embedding_viz])  

   with open(precomp_emojis_path, 'wb') as file:
      pickle.dump(precomp_emojis, file)              
    
