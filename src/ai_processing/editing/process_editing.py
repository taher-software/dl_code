import cv2
import numpy as np
import os
import pickle
import clip
from pycaption import SRTReader
import torch
from PIL import Image, ImageDraw, ImageFont
from sentence_transformers import SentenceTransformer
import subprocess
import time
import ffmpeg_streaming
import spacy
import json
from scipy import signal
from moviepy.editor import ImageSequenceClip
import imageio
import src.ai_processing.editing.clip_withgrad as clip_grad

from app import db
from src.config import Config
from ffmpeg_streaming import Formats, Bitrate, Representation, Size
from src.enums.upload_processing_status import UploadProcessingStatus
from src.models.video import Video
from src.models.collection import Collection
from src.models.saved_clip import SavedClip

from src.ai_processing.editing.xml_items.crop_item import crop_item
from src.ai_processing.editing.xml_items.caption_item import caption_item
from src.ai_processing.editing.xml_items.audio_item import audio_item
from src.ai_processing.editing.xml_items.sequence import sequence_item
import xmltodict
import copy

"""
#https://github.com/hkchengrex/Cutie
from cutie.model.cutie import CUTIE
from cutie.inference.inference_core import InferenceCore
from cutie.inference.utils.args_utils import get_dataset_cfg
from omegaconf import open_dict
import hydra
from hydra.core.hydra_config import HydraConfig
from hydra import compose, initialize
from gui.interactive_utils import image_to_torch, torch_prob_to_numpy_mask, index_numpy_to_one_hot_torch, overlay_davis
#https://huggingface.co/docs/transformers/model_doc/clipseg
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
#https://github.com/IDEA-Research/GroundingDINOhttps://github.com/IDEA-Research/GroundingDINO
import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util import box_ops
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
#https://github.com/SysCV/sam-hq
from segment_anything_hq import sam_model_registry, SamPredictor
"""
############################## NON-AI EDITING ##################################################################################


def get_files(folder):

    filtered_files = []
    for path, _, files in os.walk(folder):
        for file in files:
            full_path = os.path.join(path, file)
            filtered_files.append(full_path)
    return filtered_files
    
    
def re_enc(vid_id):
    """ Re-encode videos
        Arguments:
            vid_path: source video path
    """
    video = Video.query.get(vid_id)
    # vid_path = f"{video.base_path}{video.path}"
    # out_path = vid_path.replace ('.mp4', '_reenc.mp4')
    # ffmpeg_cmd = f"ffmpeg -i {vid_path} -c:v libx264 -preset superfast -crf 25 -c:a copy {out_path}"
    # os.system (ffmpeg_cmd)
    last_reported = 0

    vid_path = f"{video.base_path}{video.path}"
    out_path = vid_path.replace('.mp4', '_reenc.mp4')

    start_time = time.time()
    from app.utils import utils
    utils.inform_in_thread(vid_id, UploadProcessingStatus.RE_ENCODING)

    cmd = f"ffmpeg -y -i {vid_path} -c:v libx264 -preset superfast -crf 25 -c:a copy {out_path}"
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, shell=True, universal_newlines=True)

    # Keep reading the output of the process until it finishes
    while process.poll() is None:
        line = process.stdout.readline()
        progress = parse_progress(line, video.length)
        try:
            if progress is not None and utils.should_report(last_reported, progress):
                last_reported = progress
                estimated_time = ((time.time() - start_time) / progress)
                utils.inform_in_thread(vid_id, UploadProcessingStatus.RE_ENCODING, progress=(
                    progress*100), estimate=estimated_time)
        except:
            pass

    process.wait()
    video.rencoded = 1
    db.session.commit()
    utils.inform_in_thread(vid_id, UploadProcessingStatus.RE_ENCODING,
                           progress=100, estimate=((time.time() - start_time)))
    os.remove(vid_path)
    os.rename(out_path, vid_path)
    
    
def parse_progress(line, total_frames):
    """
    Parses a progress line from ffmpeg output and returns a float value between 0 and 1 representing the progress.
    Returns None if the line does not contain progress information.
    """
    if 'frame=' not in line:
        return None
    try:
        l = ' '.join(line.split())
        l = l.replace('frame= ', 'frame=')
        parts = l.strip().split()
        frame_part = [part for part in parts if 'frame=' in part][0]
        frame_str = frame_part.split('=')[1]
        frame = int(frame_str)
        progress = frame / total_frames
        return progress
    except Exception as exc:
        print(exc)
        return -1

def update_bandwidth_thresholds(hls_path):
    new_bandwidth_values = {
        '1920x1080': 200000000,  # 200 Mbps    
        '1280x720': 150000000,  # 150 Mbps
        '854x480': 50000000,    # 50 Mbps
        '640x360': 10000000,    # 10 Mbps
        '426x240': 1000000,    # 1 Mbps
        '256x144': 500000     # 0.5 Mbps
    }

    with open(hls_path, 'r') as file:
        lines = file.readlines()

    modified_lines = []

    for line in lines:
        if line.startswith('#EXT-X-STREAM-INF'):
            for resolution, new_bandwidth in new_bandwidth_values.items():
                if f'RESOLUTION={resolution}' in line:
                    print("modifying bandwith for resolution:", resolution)
                    parts = line.split(',')
                    for i, part in enumerate(parts):
                        if 'BANDWIDTH' in part:
                            parts[i] = f'#EXT-X-STREAM-INF:BANDWIDTH={new_bandwidth}'
                    line = ','.join(parts)
                    break
        modified_lines.append(line)

    with open(hls_path, 'w') as file:
        file.writelines(modified_lines)
       
        
def hls_convert(mp4_path, hls_path, resolutions=None):
    """ Converts mp4 videos to HLS format
        Arguments:
            mp4_path: path to the mp4 video to convert
            hls_path: path to the .m3u8 HLS file. The function will also generate .m3u8 files per resolution. 
            resolutions: list of resolution to output. If set to None, will auto generate all standard resolutions. 

    """
    video = ffmpeg_streaming.input(mp4_path)

    if resolutions:
        # dont use this option yet
        _360p = Representation(Size(640, 360), Bitrate(276 * 1024, 128 * 1024))
        _480p = Representation(Size(854, 480), Bitrate(750 * 1024, 192 * 1024))
        _720p = Representation(
            Size(1280, 720), Bitrate(2048 * 1024, 320 * 1024))
        _1080p = Representation(
            Size(1920, 1080), Bitrate(4096 * 1024, 320 * 1024))

        hls = video.hls(Formats.h264())
        hls.representations(_360p, _480p, _720p, _1080p)
        hls.output(hls_path)

    else:
        hls = video.hls(Formats.h264())
        hls.auto_generate_representations()
        hls.output(hls_path)

    try:
        print('I should totally happen')
        update_bandwidth_thresholds(hls_path)
    except:
        print("Failed to update HLS bandwith thresholds")

    return

def hls_convert_video(video_id):
    video = Video.query.get(video_id)
    if video is None:
        return
    
    name, extension = os.path.splitext(video.path)
    
    hls_filename = (os.path.basename(video.path)).replace(extension, '.m3u8')
    hls_path = f"{Config.ROOT_FOLDER}/app/upload/{video.user_id}/{video.id}/hls/{hls_filename}"
    print('mp4_path',f"{video.base_path}{video.path.replace(extension, '.mp4')}")
    print('hls_path',hls_path)
    hls_convert(
        mp4_path=f"{video.base_path}{video.path.replace(extension, '.mp4')}",
        hls_path=hls_path
    )

    video.processed_hls = 1
    db.session.commit()
    return

def cut_clip(start, end, vid_path, clip_name, re_encode=True):
    """ Trim videos to extract clips
        Arguments:
            start: start timestamp
            end: end timestamp
            vid_path: source video path
            clip_name: clip name of the extracted clip
    """

    duration = end-start

    if re_encode:
        # -c:v libx264 -crf 0 -c:a copy
        # string = f"ffmpeg -y -i {vid_path} -ss {str(start)} -to {str(end)} -qp 10 {clip_name}"
        string = f"ffmpeg -y -i {vid_path} -ss {str(start)} -t {str(duration)} -c:v libx264 -preset ultrafast -crf 23 -c:a copy {clip_name}"

    else:
        string = f"ffmpeg -y -ss {str(start)} -i {vid_path} -t {str(duration)} -c copy {clip_name}"

    os.system(string)


def merge_clips(clips_list, lst_path, merged_path):
    """ Merge clips together temporaly
        Arguments:
            clips_list: list of absolute paths of the clips to merge
            lst_path: path to the text file where clips path will be written
            merged_path: path to the merged 
    """
    f = open(lst_path, 'w')
    for clip in clips_list:
        if os.path.isfile(clip):
            f.write('file' + ' ' + clip + '\n')
        else:
            pass
    f.close()

    merge_cmd = f"ffmpeg -y -f concat -safe 0 -i {lst_path} -c copy {merged_path}"

    os.system(merge_cmd)
    os.remove(lst_path)
    for clip in clips_list:
        if os.path.isfile(clip):
            os.remove(clip)


def crop_clip(trim_path, crop_path, reframe):
    """ Trim videos to extract clips
        Arguments:
            trim_path: trimmed clip path
            crop_path: cropped clip path
            reframe: reframing positions
    """

    corner_x = reframe['leftOffset']
    corner_y = reframe['topOffset']
    width = reframe['width']
    height = reframe['height']

    string = f"ffmpeg -y -i {trim_path} -filter:v 'crop=in_w*{width}:in_h*{height}:in_w*{corner_x}:in_h*{corner_y}' -c:a copy {crop_path}"
    os.system(string)
    
# ASS styling format
ass_format = {'Fontname': 1, 'Fontsize': 2, 'PrimaryColour': 3, 'SecondaryColour': 4, 'OutlineColour': 5, 'BackColour': 6,
              'Bold': 7, 'Italic': 8, 'Underline': 9, 'StrikeOut': 10, 'ScaleX': 11, 'ScaleY': 12, 'Spacing': 13, 'Angle': 14,
              'BorderStyle': 15, 'Outline': 16, 'Shadow': 17, 'Alignment': 18, 'MarginL': 19, 'MarginR': 20, 'MarginV': 21, 'Encoding': 22}   
          
def style_sub(input_ass, fontname, fontsize, primarycolour, outlinecolour, borderstyle, marginv, outline, shadow, bold, spacing, verb_colour, object_colour):
    """ Restyles captions in ASS format
        Arguments:
            input_ass: path to the caption file in ASS format
            fonts_dir, fontname, fontsize, primarycolour, outlinecolour, borderstyle, marginv: styling parameters obtained from burn_sub
    """
    
    nlp = spacy.load("en_core_web_sm") 
    with open(input_ass, 'r') as ass:
        lines = ass.readlines()

    styles = lines[9]
    styles = styles.split(',')

    styles[ass_format['Fontname']] = fontname
    styles[ass_format['Fontsize']] = fontsize
    styles[ass_format['PrimaryColour']] = primarycolour
    #styles[ass_format['OutlineColour']] = outlinecolour
    #styles[ass_format['BorderStyle']] = borderstyle
    styles[ass_format['MarginV']] = marginv
    styles[ass_format['Shadow']] = shadow
    styles[ass_format['Bold']] = bold
    styles[ass_format['Spacing']] = spacing   
    #styles[ass_format['Outline']] = outline

    styles = ','.join(styles)

    lines[9] = styles

    with open(input_ass, 'w') as ass:
        ass.writelines(lines)

    with open(input_ass, 'r') as ass:
        caption_lines = ass.readlines()[13:]

    new_caption_lines = []
    for line in caption_lines:
        line = line.strip()
        if line.startswith("Dialogue:"):
            parts = line.replace('Dialogue: ', '').split(',', maxsplit=9)
            layer, start_time, end_time, style, actor, margin_l, margin_r, margin_v, effect, text= parts
            doc = nlp(text)
            new_text = ""
            for token in doc:
                if token.pos_ == "VERB":
                    new_text += f"{{\\c&H{verb_colour}&}}{token.text}{{\\c}} "
                elif "obj" in token.dep_:
                    new_text += f"{{\\c&H{object_colour}&}}{token.text}{{\\c}} "               
                else:
                    new_text += token.text + " "
            new_line = f"Dialogue:{layer},{start_time},{end_time},{style},{actor},{margin_l},{margin_r},{margin_v},{effect},{new_text}"
            print(new_line)
            new_caption_lines.append(new_line + "\n")
        else:
            new_caption_lines.append(line + "\n")

    with open(input_ass, 'w') as ass:
        lines = lines[:13] + new_caption_lines
        ass.writelines(lines)        
        
        
def burn_subs_ffmpeg(clip_path, srt_path, fonts_dir, fontname, fontsize, primarycolour, outlinecolour, borderstyle, marginv, outline, shadow, bold, spacing, verb_colour, object_colour):
    """ Burn subs into clip
        Arguments:
            clip_path: path of the clip that was cut from the original video using vid_out
            srt_path: path to the portion of srt file corresponding, needs to be extracted using mini_srt
            fonts_dir: path to folder where fonts are stored
            fonts_dir, fontname, fontsize, primarycolour, outlinecolour, borderstyle, marginv: styling parameters obtained from UI
    """

    subbed_clip_path = clip_path.replace('.mp4', '_subbed.mp4')
    ass_path = srt_path.replace('.srt', '.ass')
    ass_cmd = f"ffmpeg -y -i {srt_path} {ass_path}"
    os.system(ass_cmd)

    style_sub(ass_path, fontname, fontsize, primarycolour,
              outlinecolour, borderstyle, marginv, outline, shadow, bold, spacing, verb_colour, object_colour)

    subs_cmd = f"ffmpeg -y -i {clip_path} -vf ass={ass_path}:fontsdir={fonts_dir} {subbed_clip_path}"
    os.system(subs_cmd)

    os.remove(clip_path)


    return subbed_clip_path
        

def fit_vid(vid_id, start, end, output_video_path, canvas_size="720p", orientation = "portrait"):

    video = Video.query.get(vid_id)
    if video is None:
        return
        
    vid_path = f"{video.base_path}{video.path}" 
    
    if canvas_size == "720p":
        if orientation == "landscape":
            target_width, target_height = 1280, 720
        elif orientation == "portrait":
            target_width, target_height = 720, 1280
        elif orientation == "square":
            target_width, target_height = 1280, 1280            
        else:
            raise ValueError("Invalid orientation. Must be 'landscape', 'portrait' or 'square'.")
    elif canvas_size == "1080p":
        if orientation == "landscape":
            target_width, target_height = 1920, 1080
        elif orientation == "portrait":
            target_width, target_height = 1080, 1920
        elif orientation == "square":
            target_width, target_height = 1920, 1920          
        else:
            raise ValueError("Invalid orientation. Must be 'landscape', 'portrait' or 'square'.")
    else:
        raise ValueError("Invalid canvas size. Must be '720p' or '1080p'.")

    
    vidcap = cv2.VideoCapture(vid_path)
    fps = round(vidcap.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    temp_output_video_path = output_video_path.replace('.mp4', '_temp.mp4')
    output_video = cv2.VideoWriter(temp_output_video_path, fourcc, fps, (target_width, target_height))

    frame_start, frame_end = int(start*fps), int(end*fps)
    for frame_id in range (frame_start, frame_end):
        print('\033[31m' + 'processing frame:' + '\033[0m', frame_id)
        
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = vidcap.read()
        if not ret:
            break
            
        original_height, original_width = frame.shape[:2]
        print("original dims:", original_height, original_width)
        print("target dims:", target_height, target_width)
                
        original_aspect_ratio = original_width/original_height
        print("original ratio:", original_aspect_ratio)
        
        target_aspect_ratio = target_width/target_height
        print("target ratio:", target_aspect_ratio)
                
        original_center = (int(original_height/2), int(original_width/2))
        print("original center:", original_center)
                
        target_center = (int(target_height/2), int(target_width/2))
        print("target center:", target_center)
                                
        resized_height, resized_width = int(target_width / original_aspect_ratio), target_width
        print("resize dims:", resized_height, resized_width)
                
        resized_frame = cv2.resize(frame, (resized_width, resized_height))

        crop_height, crop_width = target_height, target_width

        if crop_height>original_height: 
           crop_height = original_height
           crop_width = crop_height*target_aspect_ratio
                   
        if crop_width > original_width: 
           crop_width = original_width
           crop_height = crop_width/target_aspect_ratio

        print("crop dims:", crop_height, crop_width)
        
        blurred_frame = frame[original_center[0]-int(crop_height/2):original_center[0]+int(crop_height/2), original_center[1]-int(crop_width/2):original_center[1]+int(crop_width/2)]
        blurred_frame = cv2.GaussianBlur(blurred_frame, (0, 0), 10)              
        final_frame = cv2.resize(blurred_frame, (target_width, target_height)) 
        
        final_frame[int(target_center[0]-resized_height/2):int(target_center[0]+ resized_height/2), int(target_center[1]-resized_width/2):int(target_center[1]+ resized_width/2)] = resized_frame

        output_video.write(final_frame)
        
    temp_audio_path = output_video_path.replace('.mp4', '.aac')
    audio_cmd = f"ffmpeg -y -i {vid_path} -ss {start} -to {end} -vn {temp_audio_path}"
    os.system(audio_cmd)
                        
    vidcap.release()
    output_video.release()

    merge_cmd = f"ffmpeg -y -i {temp_output_video_path} -i {temp_audio_path} -c:v copy -c:a aac {output_video_path}"
    os.system(merge_cmd)

    os.remove(temp_audio_path)
    os.remove(temp_output_video_path)
    

def add_logo(vid_path, logo_path, output_video_path, position=(0, 0, 0, 0), start=0, end=None):
    
    vidcap = cv2.VideoCapture(vid_path)

    fps = round(vidcap.get(cv2.CAP_PROP_FPS))
    width = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    logo = cv2.imread(logo_path, cv2.IMREAD_UNCHANGED)

    x, y, logo_width, logo_height = position

    x = int(x * width)
    y = int(y * height)
    logo_width = int(logo_width * width)    
    logo_height = int(logo_height * height)

    logo = cv2.resize(logo, (logo_width, logo_height), interpolation=cv2.INTER_AREA)

    if end is None:
        end = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    temp_output_video_path = output_video_path.replace('.mp4', '_temp.mp4')    
    output_video = cv2.VideoWriter(temp_output_video_path, fourcc, fps, (width, height))

    frame_id = 0
    frame_start, frame_end = int(start*fps), int(end*fps)
    while True:
        print('\033[31m' + 'processing frame:' + '\033[0m', frame_id)
        ret, frame = vidcap.read()
        if not ret:
            break    
        if frame_start <= frame_id < frame_end:    

            roi = slice(y, y + logo_height), slice(x, x + logo_width)

            logo_alpha = logo[:, :, 3] / 255.0

            for c in range(0, 3):
                frame[roi][:, :, c] = logo[:, :, c] * logo_alpha + frame[roi][:, :, c] * (1 - logo_alpha)

        output_video.write(frame)
        frame_id += 1

    temp_audio_path = output_video_path.replace('.mp4', '.aac')
    audio_cmd = f"ffmpeg -y -i {vid_path} -vn {temp_audio_path}"
    os.system(audio_cmd)
                        
    vidcap.release()
    output_video.release()

    merge_cmd = f"ffmpeg -y -i {temp_output_video_path} -i {temp_audio_path} -c:v copy -c:a aac {output_video_path}"
    os.system(merge_cmd)

    os.remove(temp_audio_path)
    os.remove(temp_output_video_path)
    
        
############################## SUBTITLE STYLER ##################################################################################


def read_captions(srt_path):
    srt_text = open(srt_path).read()
    srt = SRTReader().read(srt_text, lang="en")
    captions = srt.get_captions("en")

    return captions


def split_text_into_lines(text, font, max_width, uppercase):
    lines = []
    words = text.split()
    current_line = ""

    for word in words:
        if uppercase:
           word = word.upper()
        text_line = " ".join([current_line, word]).strip()
        try:
            text_width, _ = font.getsize(text_line)
        except:
            text_width = font.getbbox(text_line)[2]

        if text_width <= max_width:
            current_line = text_line
        else:
            lines.append(current_line.split())
            current_line = word

    if current_line:
        lines.append(current_line.split())

    return lines

    
def burn_subs(vid_path, captions, output_path, word_timestamps, layout = "center", height_offset = 300, width_offset = 50, fontsize=55, word_spacing = 10, line_spacing = 70, uppercase =True, font_path=None, font_color=(255, 255, 255), spoken_word_color=(0, 255, 255), verb_color=None, object_color=None, animation_type = None, shadow_color = (0, 0, 0), shadow_offset = 7, outline_color = (0, 0, 0), outline_width = 6, burn_emoji = False, precomp_emojis_path = None, emojis_folder = None):

    cap = cv2.VideoCapture(vid_path)
    frame_rate = int(cap.get(5))

    temp_output_path = output_path.replace('.mp4', '_temp.mp4')
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    output_video = cv2.VideoWriter(temp_output_path, fourcc, frame_rate, (int(cap.get(3)), int(cap.get(4))))

    if font_path:
        font = ImageFont.truetype(font_path, int(fontsize))
    else:
        font = ImageFont.truetype(f"{Config.ROOT_FOLDER}/app/fonts/Montserrat-Black.ttf", int(fontsize))
                        
    frame_counter = 0
    current_caption = None
    
    if burn_emoji:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        clip_model, preprocess = clip.load("ViT-B/32", device=device)
        lm_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    
    nlp = spacy.load("en_core_web_sm")
    while cap.isOpened():
        ret, frame = cap.read()
        print('\033[34m' + 'Processing frame:' + '\033[0m', frame_counter)
        frame_counter += 1

        if not ret:
            break
                     
        frame_timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)

        if current_caption is None or frame_timestamp >= current_caption.get('end'):
            filtered_words = []
            for caption in captions:
                start_time = caption.get('start')
                end_time = caption.get('end')

                if start_time <= frame_timestamp <= end_time:
                    current_caption = caption
                    word_types = [token for token in nlp(caption.get('text'))]
                    break 
                else:
                    current_caption = None
                    
        if current_caption:
        
            if burn_emoji:        
                emoji_path, max_sim = get_emoji(clip_model, lm_model, device, precomp_emojis_path, emojis_folder, current_caption.get('text'))

            filtered_words = [word_info for word_info in word_timestamps if
                              current_caption.get('start') <= word_info[1] <= current_caption.get('end') and
                              current_caption.get('start') <= word_info[2] <= current_caption.get('end')]                             
                              
            image = Image.fromarray(frame)
            image_width, image_height = image.size

            draw = ImageDraw.Draw(image)
            
            lines = split_text_into_lines(current_caption.get('text'), font, image_width-2*width_offset, uppercase)
                
            try:
                first_line_width = sum([font.getsize(word)[0] for word in lines[0]]) + word_spacing*(len(lines[0])-1)
            except:
                first_line_width = sum([(font.getbbox(word)[2]) for word in lines[0]]) + word_spacing*(len(lines[0])-1)
                
            text_x = (image_width - first_line_width) // 2
            text_y = frame.shape[0] - height_offset            
            
            line_index = 0
            
            for word_index, word_info in enumerate(filtered_words):
                if word_index > sum(len(lines[prev_line_index])-1 for prev_line_index in range(0, line_index+1)):
                    line_index+=1
                    try:
                        line_width =  sum([font.getsize(word)[0] for word in lines[line_index]]) + word_spacing*(len(lines[line_index])-1)
                    except:
                        line_width =  sum([(font.getbbox(word)[2]) for word in lines[line_index]]) + word_spacing*(len(lines[line_index])-1)      
                    text_x = (image_width - line_width) // 2
                    text_y += line_spacing
                word_text = word_info[0]
                             
                if uppercase:
                    word_text = word_text.upper()
                    
                word_start_time = word_info[1]
                word_end_time = word_info[2]

                font_color_to_use = font_color
                
                try:
                    text_size = font.getsize(word_text)
                except:
                    text_size = font.getbbox(word_text)
                                                                           
                if verb_color and word_types[word_index].pos_ == "VERB":
                    font_color_to_use = verb_color
                        
                if object_color and "obj" in word_types[word_index].dep_:
                    font_color_to_use = object_color

                if animation_type and frame_timestamp >= word_start_time and frame_timestamp <= word_end_time:
                    if animation_type == "word":
                        font_color_to_use = spoken_word_color
                    elif animation_type == "box":
                        font_color_to_use = font_color
                        try:
                            x, y, a, b = text_x-0.5*word_spacing, text_y - 0.1*line_spacing, text_x + text_size[2]+ 0.5*word_spacing, text_y + text_size[3]+ 0.1*line_spacing
                        except:
                            x, y, a, b = text_x-0.5*word_spacing, text_y - 0.1*line_spacing, text_x + text_size[0]+ 0.5*word_spacing, text_y + text_size[1]+ 0.1*line_spacing
                        draw.rounded_rectangle([x, y, a, b], radius=10, fill=spoken_word_color)   

                if shadow_color:
                    shadow_x = text_x + shadow_offset  
                    shadow_y = text_y + shadow_offset

                    draw.text((shadow_x, shadow_y), word_text, fill=shadow_color, font=font)

                if outline_color:
                    draw.text((text_x, text_y), word_text, fill=font_color_to_use, font=font, stroke_width = outline_width, stroke_fill = outline_color)       
                        
                else:                        
                    draw.text((text_x, text_y), word_text, fill=font_color_to_use, font=font)
                        
                try:
                   text_x += text_size[2] + word_spacing
                except:
                   text_x += text_size[0] + word_spacing
                   
        
                if burn_emoji:
                    frame = np.array(image)
                    emoji = cv2.imread(emoji_path, cv2.IMREAD_UNCHANGED)
                    emoji_height, emoji_width, _ = emoji.shape
                    x, x1, y, y1 = int(image_width/2-emoji_width/2), int(image_width/2+emoji_width/2),  text_y + line_spacing*2, text_y + line_spacing*2 + emoji_height
                    
                    roi = slice(y, y1), slice(x, x1)

                    emoji_alpha = emoji[:, :, 3] / 255.0

                    for c in range(0, 3):
                        frame[roi][:, :, c] = emoji[:, :, c] * emoji_alpha + frame[roi][:, :, c] * (1 - emoji_alpha)    
                else:        
                    frame = np.array(image)                    
        else:          
            pass

        output_video.write(frame)

    temp_audio_path = output_path.replace('.mp4', '.aac')
    audio_cmd = f"ffmpeg -y -i {vid_path} -vn {temp_audio_path}"
    os.system(audio_cmd)
                        
    cap.release()
    output_video.release()

    merge_cmd = f"ffmpeg -y -i {temp_output_path} -i {temp_audio_path} -c:v copy -c:a aac {output_path}"
    os.system(merge_cmd)

    os.remove(temp_audio_path)
    os.remove(temp_output_path)
    

############################## SHOT SPLITTER ##################################################################################

    
def shot_resizer(vid_id, output_video_path, shots, resolution="1080p", orientation="portrait"):

    if resolution == "720p":
        if orientation == "landscape":
            output_width, output_height = 1280, 720
        elif orientation == "portrait":
            output_width, output_height = 720, 1280
        elif orientation == "square":
            output_width, output_height = 1280, 1280
        else:
            raise ValueError(
                "Invalid orientation. Must be 'landscape', 'portrait' or 'square'.")
    elif resolution == "1080p":
        if orientation == "landscape":
            output_width, output_height = 1920, 1080
        elif orientation == "portrait":
            output_width, output_height = 1080, 1920
        elif orientation == "square":
            output_width, output_height = 1920, 1920
        else:
            raise ValueError(
                "Invalid orientation. Must be 'landscape', 'portrait' or 'square'.")
    else:
        raise ValueError("Invalid output size. Must be '720p' or '1080p'.")

    video = Video.query.get(vid_id)
    if video is None:
        return

    vid_path = f"{video.base_path}{video.path}"

    vidcap = cv2.VideoCapture(vid_path)

    fps = round(vidcap.get(cv2.CAP_PROP_FPS))
    length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_id = 0

    temp_output_video_path = output_video_path.replace('.mp4', '_temp.mp4')
    output_video = cv2.VideoWriter(temp_output_video_path, cv2.VideoWriter_fourcc(
        *"mp4v"), fps, (output_width, output_height))

    for shot in shots:
        start_frame = shot["start"]
        end_frame = shot["end"]
        corner_x = shot['corner_x']
        corner_y = shot['corner_y']
        width = shot['width']
        height = shot['height']

        for i in range(start_frame, end_frame):
            print("cropping frame:", i)
            vidcap.set(cv2.CAP_PROP_POS_FRAMES, i)
            success, frame = vidcap.read()

            cropped_frame = frame[corner_y: corner_y +
                                  height, corner_x: corner_x + width]
            cropped_frame = cv2.resize(
                cropped_frame, (output_width, output_height))

            output_video.write(cropped_frame)

    audio_start = shots[0]["start"]/fps
    audio_end = shots[-1]["end"]/fps

    temp_audio_path = vid_path.replace('.mp4', '.aac')
    audio_cmd = f"ffmpeg -y -i {vid_path} -ss {audio_start} -to {audio_end} -vn {temp_audio_path}"
    print('AUDIO CMD', print(audio_cmd))
    os.system(audio_cmd)

    vidcap.release()
    output_video.release()

    merge_cmd = f"ffmpeg -y -i {temp_output_video_path} -i {temp_audio_path} -c:v copy -c:a aac {output_video_path}"
    os.system(merge_cmd)

    os.remove(temp_audio_path)
    os.remove(temp_output_video_path)
    

def ai_shot_splitter(vid_id):

    video = Video.query.get(vid_id)
    if video is None:
        return
        
    vid_path = f"{video.base_path}{video.path}"
    precomp_shots_path = f"{video.base_path}{video.precomp_shots_path}"

    vidcap = cv2.VideoCapture(vid_path)
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    vidcap.release()

    with open(precomp_shots_path, 'rb') as file:
        shots = pickle.load(file)

    shots = [round(float(shot)/ fps, 3) for shot in shots]

    return shots    
    
############################## FACE TRACKING ##################################################################################    
    
               
def crop_face(original_shape, x, y, w, h, desired_width, desired_height):

    orig_h, orig_w = original_shape

    aspect_ratio = desired_width/desired_height

    crop_height = int(h*1.8)
    crop_width = int(crop_height * aspect_ratio)
    
    if desired_height > orig_h:
        crop_height = orig_h
        crop_width = int(crop_height * aspect_ratio)

    if desired_width > orig_w:
        crop_width = orig_w
        crop_height = int(crop_width / aspect_ratio)
                
    delta_w = int(crop_width/2)
        
    delta_h = int(crop_height/2)
    
    w1 = max(0, x - delta_w)
    w2 = min (orig_w, x + delta_w)
    
    if w1 == 0:
       w2 = crop_width
       
    if w2 == orig_w: 
       w1 = orig_w - crop_width
       
    h1 = max(0, y - delta_h)
    h2 = min (orig_h, y + delta_h)
    
    if h1 == 0:
       h2 = crop_height
       
    if h2 == orig_h: 
       h1 = orig_h - crop_height
    
    return h1, h2, w1, w2
    

def faces_extractor(vid_id, start, end, active_speakers_only = True, speaker_window = 3, speaker_threshold = 0.3, autobroll = False, orientation = "portrait"):
    from app.beta.models.video import Video
    video = Video.query.get(vid_id)
    if video is None:
        return
    vid_path = f"{video.base_path}{video.path}"
     
    vidcap = cv2.VideoCapture(vid_path)
    original_shape = (int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH)))
    vidcap.release()   
            
    fps = round(vidcap.get(cv2.CAP_PROP_FPS))
    tracked_faces_path = f"{video.base_path}{video.tracked_faces_path}"
    speakers_path = f"{video.base_path}{video.speakers_path}"
        
    if orientation == "landscape":
        canvas_width, canvas_height = original_shape[1], int(original_shape[1]*9/16)
    elif orientation == "portrait":
        canvas_width, canvas_height = original_shape[0], int(original_shape[0]*16/9)
    elif orientation == "square":
        canvas_width, canvas_height = original_shape[1], original_shape[1]         
    else:
        raise ValueError("Invalid orientation. Must be 'landscape', 'portrait' or 'square'.")   
                
    with open(tracked_faces_path, 'rb') as file:
        tracked_faces = pickle.load(file)

    with open(speakers_path, 'rb') as file:
        speakers= pickle.load(file)    
        
    shots = [i for i in range(1, len(tracked_faces)) if len(tracked_faces[i]) != len(tracked_faces[i-1])]
    shots.append(len(tracked_faces))
    
    print('\033[37m' + 'detected shots:' + '\033[0m', shots)
    
    speaker_history = []
    frame_start, frame_end = round(start*fps), round(end*fps)

    faces_list = []
            
    for frame_id in range (frame_start, frame_end):
        print('\033[31m' + 'processing frame:' + '\033[0m', frame_id)
            
        faces = tracked_faces[frame_id]
        num_faces = len(faces)
                
        if active_speakers_only and num_faces>1:
            updated_faces = []
            speaking_counts = []

            for i, face in enumerate(faces):
               x, y, w_f, h_f, face_id = face

               window = int(fps*speaker_window)         
               next_shot = next((index for index in shots if index > frame_id), None)
               
               lim_sup = min(frame_id + window, next_shot)
               
               length = lim_sup-frame_id +1
               
               if length<window:
                  delta = window-length
                  lim_inf = frame_id- delta
                  lim_sup = lim_inf + window-1
                  speaking_sum = sum(1 for frame_num in range(lim_inf, lim_sup) if face_id in speakers[frame_num])
                  length = lim_sup-lim_inf
               else:
                  speaking_sum = sum(1 for frame_num in range(frame_id, lim_sup) if face_id in speakers[frame_num])                     

               speaking_percentage = speaking_sum/(length)
                              
               print("speaking percentage:", speaking_percentage)

               speaking_counts.append([face, speaking_sum])
            
            updated_faces = [max(speaking_counts, key=lambda item: item[1])[0]]
            updated_faces = [[face[0],face[1],face[2],face[3], index] for index, face in enumerate(updated_faces)]  
            
            faces = updated_faces
                    
        faces = [[face[0],face[1],face[2],face[3], index] for index, face in enumerate(faces)] 
        num_faces = len(faces)  

        frame_faces_list = []        
        if num_faces > 0:   
            print('\033[34m' + 'Number of detected faces:' + '\033[0m', num_faces)                      

            if autobroll:
                num_faces+=1  
                            
            for i, (x, y, w_f, h_f, face_id) in enumerate(faces): 
                print('\033[32m' + 'processing face:' + '\033[0m', i, ', params:', x, y, w_f, h_f, face_id)
                
                if orientation == "portrait":
                   h = canvas_height//num_faces
                   h1, h2, w1, w2 = crop_face(original_shape, int(x+w_f/2), int(y+h_f/2), w_f, h_f, canvas_width, h) 
                           
                elif orientation == "landscape":
                   w = canvas_width//num_faces                  
                   h1, h2, w1, w2 = crop_face(original_shape, int(x+w_f/2), int(y+h_f/2), w_f, h_f, w, canvas_height)         

                else: 
                   h = canvas_height//num_faces       
                   h1, h2, w1, w2  = crop_face(original_shape, int(x+w_f/2), int(y+h_f/2), w_f, h_f, canvas_width, h) 
                   
                frame_faces_list.append([h1, h2, w1, w2, face_id])             
        else:
            pass            
            
        faces_list.append(frame_faces_list)              
    return  faces_list
     
    
def resize_vid(vid_id, faces_list, start, end, output_video_path, orientation = "portrait", autobroll = False, precomp_stock_vids_path = None, canvas_height = 1280):        
    from app.beta.models.video import Video
    video = Video.query.get(vid_id)
    if video is None:
        return
    vid_path = f"{video.base_path}{video.path}"

    if autobroll:
    
       device = "cuda" if torch.cuda.is_available() else "cpu"
       print("Using:", device)
       clip_model, preprocess = clip.load("ViT-B/32", device=device)


       srt_folder = f"{video.base_path}{video.srt_folder}"
       srt_path = f"{srt_folder}_{source_lang}.srt"       
       precomp_sub_path = f"{video.base_path}{video.precomp_sub_path}"
              
       caption_embeddings = np.load(precomp_sub_path)
       captions = read_captions(srt_path)
       
       with open(precomp_stock_vids_path, 'rb') as file:
          stock_vids_embeddings = pickle.load(file)
             
       for i, caption in enumerate(captions):
          caption_start = caption.start/1000000
          caption_end = caption.end/1000000
          caption_text = caption.get_text()

          if caption_start >= start:
             current_caption = caption
                
             with torch.no_grad():
                 caption_text = clip.tokenize(caption_text).to(device)         
                 caption_embedding = clip_model.encode_text(caption_text)
                 caption_embedding  /= caption_embedding.norm(dim=-1, keepdim=True)
                 current_caption_embedding  = caption_embedding.T.cpu().numpy().squeeze()
                    
             current_index = i
                             
             sim_list = [100.0 * np.mean(np.load(vid_embedding_path[1], allow_pickle=True) @ current_caption_embedding) for vid_embedding_path in stock_vids_embeddings]
             sim_list = [[sim, index] for index, sim in enumerate(sim_list)]
             sim_list = sorted(sim_list, key=lambda x: x[0], reverse=True)
                
             matching_vid_path = stock_vids_embeddings[sim_list[0][1]][0]
             stock_vid = cv2.VideoCapture(matching_vid_path)
             stock_vid_index = 0
             break

    vidcap = cv2.VideoCapture(vid_path)
    original_shape = (int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH)))

    if orientation == "landscape":
        canvas_width = int(canvas_height*16/9)
    elif orientation == "portrait":
        canvas_width = int(canvas_height*9/16)
    elif orientation == "square":
        canvas_width = canvas_height         
    else:
        raise ValueError("Invalid orientation. Must be 'landscape', 'portrait' or 'square'.") 
     
    fps = round(vidcap.get(cv2.CAP_PROP_FPS))
    temp_output_video_path = output_video_path.replace('.mp4', '_temp.mp4')
    output_video = cv2.VideoWriter(temp_output_video_path,cv2.VideoWriter_fourcc(*"mp4v"), fps, (canvas_width, canvas_height))

    vidcap.set(cv2.CAP_PROP_POS_MSEC, start * 1000)
    frame_start = int(vidcap.get(cv2.CAP_PROP_POS_FRAMES))
    vidcap.set(cv2.CAP_PROP_POS_MSEC, end * 1000)
    frame_end = int(vidcap.get(cv2.CAP_PROP_POS_FRAMES))
    
    faces_list_id = 0
    for frame_id in range (frame_start, frame_end):
        print('\033[31m' + 'processing frame:' + '\033[0m', frame_id)    
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = vidcap.read()
        
        try:
            faces = faces_list[faces_list_id]
        except:
            faces = []
        num_faces = len(faces)

        if autobroll: 
            num_faces+=1
            if frame_id/fps > caption_end:
                current_index += 1
                current_caption = captions[current_index]
                caption_start = current_caption.start/1000000
                caption_end = current_caption.end/1000000                
                caption_text = current_caption.get_text()   
                             
                with torch.no_grad():
                    caption_text = clip.tokenize(caption_text).to(device)         
                    caption_embedding = clip_model.encode_text(caption_text)
                    caption_embedding  /= caption_embedding.norm(dim=-1, keepdim=True)
                    current_caption_embedding  = caption_embedding.T.cpu().numpy().squeeze()
                
                sim_list = [100.0 * np.mean(np.load(vid_embedding_path[1], allow_pickle=True) @ current_caption_embedding) for vid_embedding_path in stock_vids_embeddings]  
                             
                sim_list = [[sim, index] for index, sim in enumerate(sim_list)]
                sim_list = sorted(sim_list, key=lambda x: x[0], reverse=True)
                
                matching_vid_path = stock_vids_embeddings[sim_list[0][1]][0]

                stock_vid = cv2.VideoCapture(matching_vid_path)
                stock_vid_index = 0                
            else:
                stock_vid_index += 1
            
            stock_vid.set(cv2.CAP_PROP_POS_FRAMES, stock_vid_index)
            ret, stock_frame = stock_vid.read()
                    
        if num_faces > 0:   
            #print('\033[34m' + 'Number of detected faces:' + '\033[0m', num_faces)                      
            canvas = np.zeros((canvas_height, canvas_width , 3), dtype=np.uint8)
            
            for i, (h1, h2, w1, w2, face_id) in enumerate(faces): 
                #print('\033[32m' + 'processing face:' + '\033[0m', i, ', params:', h1, h2, w1, w2, face_id)
               
                if orientation == "portrait":
                   h = canvas_height//num_faces                
                   desired_width, desired_height = canvas_width, h     
                       
                   face = frame[h1:h2, w1:w2]    
                   face = cv2.resize(face, (desired_width, desired_height))  
                                  
                   if face_id != -1:           
                      offset = face_id*h
                      canvas[offset:offset+h, :] = face
                   else:
                      canvas = face
                            
                elif orientation == "landscape":
                   w = canvas_width//num_faces                   
                   desired_width, desired_height = w, canvas_height    

                   face = frame[h1:h2, w1:w2]    
                   face = cv2.resize(face, (desired_width, desired_height))  

                   if face_id != -1:                                        
                      offset = face_id*w
                      canvas[:, offset:offset+w] = face
                   else:
                      canvas = face                   
                else: 
                   h = canvas_height//num_faces                   
                   desired_width, desired_height = canvas_width, h  

                   face = frame[h1:h2, w1:w2]                       
                   face = cv2.resize(face, (desired_width, desired_height))  
                   
                   if face_id != -1:                                                                                               
                      offset = face_id*h
                      canvas[offset:offset+h, :] = face
                   else:
                      canvas = face
            if autobroll:
                width, height, _ = stock_frame.shape
                w1 = int(width/2-desired_width/2)
                w2 = w1 + desired_width
                h1 = int(height/2-desired_height/2)
                h2 = h1 + desired_height     
                        
                cropped_frame = stock_frame[h1:h2, w1:w2]
                
                cropped_frame = cv2.resize(cropped_frame, (desired_width, desired_height))  
                
                canvas[offset+h:, :] = cropped_frame
                
            output_video.write(canvas)
        else:
            h, w = frame.shape[:2]
            if orientation == "portrait":
               frame_height = h
               frame_width = canvas_height*(9/16)
               canvas = frame[:, int(w/2) - int(frame_width/2):int(w/2) + int(frame_width/2)]
            else:
               frame_width = w
               frame_height = canvas_width*(16/9)
               canvas = frame[h/2 - int(frame_height/2):int(h/2) + int(frame_height/2), :]       

            output_video.write(canvas)
            
        faces_list_id +=1
        
    temp_audio_path = output_video_path.replace('.mp4', '.aac')
    audio_cmd = f"ffmpeg -y -i {vid_path} -ss {start} -to {end} -vn -filter:a 'atempo={vidcap.get(cv2.CAP_PROP_FPS)}/{vidcap.get(cv2.CAP_PROP_FPS)}' {temp_audio_path}"
    os.system(audio_cmd)
                        
    vidcap.release()
    output_video.release()

    merge_cmd = f"ffmpeg -y -i {temp_output_video_path} -i {temp_audio_path} -c:v copy -c:a aac {output_video_path}"
    os.system(merge_cmd)

    os.remove(temp_audio_path)
    os.remove(temp_output_video_path)


def debug_drawbox(vid_id, boxes, start, end, output_video_path):

    from app.beta.models.video import Video
    video = Video.query.get(vid_id)
    if video is None:
        return
    vid_path = f"{video.base_path}{video.path}"
    
    start_time = time.time()  
    output_video_path = output_video_path.replace('.mp4', '_box.mp4')
    
    vidcap = cv2.VideoCapture(vid_path)
    length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = round(vidcap.get(cv2.CAP_PROP_FPS))

    width = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    if start==None:
        start_frame = 0
    if end==None:
        end_frame = length
        
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    temp_output_video_path = output_video_path.replace('.mp4', '_temp.mp4') 
    output_video = cv2.VideoWriter(temp_output_video_path, fourcc, fps, (width, height))

    start_frame, end_frame = round(start*fps), round(end*fps)
    vidcap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    for frame_id in range(start_frame, end_frame):
        if frame_id%(int(fps*5))==0:
           print('processing frame: ', frame_id + start)

        ret, frame = vidcap.read()  
        try:
           box_s = boxes[frame_id-start_frame][0][:4]
           y1s, y2s, x1s, x2s = box_s         
           cv2.rectangle(frame, (x1s, y1s), (x2s, y2s), (0, 0, 255), 2)              
        except:
           pass
        output_video.write(frame)

    temp_audio_path = output_video_path.replace('.mp4', '.aac')
    audio_cmd = f"ffmpeg -y -loglevel error -hide_banner -i {vid_path} -ss {start} -to {end} -vn -filter:a 'atempo={fps}/{vidcap.get(cv2.CAP_PROP_FPS)}' {temp_audio_path}"
    os.system(audio_cmd)

    vidcap.release()
    output_video.release()

    merge_cmd = f"ffmpeg -y -loglevel error -hide_banner -i {temp_output_video_path} -i {temp_audio_path} -c:v copy -c:a aac {output_video_path}"
    os.system(merge_cmd)

    os.remove(temp_audio_path)
    os.remove(temp_output_video_path)
    
    end_time = time.time()
    print("Creating clip processing took:", end_time - start_time) 
    
    
def tedx_tracker (vid_id, precomp_faces_path, output_video_path, start, end, orientation = "portrait"):
    from app.beta.models.video import Video
    video = Video.query.get(vid_id)
    if video is None:
        return
    vid_path = f"{video.base_path}{video.path}"
    precomp_shots_path = f"{video.base_path}{video.precomp_shots_path}"
    
    with open(precomp_shots_path, 'rb') as file:
        shots = pickle.load(file)  
                
    vidcap = cv2.VideoCapture(vid_path)
    fps = round(vidcap.get(cv2.CAP_PROP_FPS))
    
    target_height = 720
    target_width = int(target_height*9/16)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    temp_output_video_path = output_video_path.replace('.mp4', '_temp.mp4')
    output_video = cv2.VideoWriter(temp_output_video_path, fourcc, fps, (target_width, target_height))
        
    with open(precomp_faces_path, 'rb') as file:
        precomp_faces = pickle.load(file) 

    frame_start, frame_end = int(start*fps), int(end*fps)  
    x_positions = []
    
    prev_pos = 0
    for i in range(frame_start, frame_end):
       if precomp_faces[i][0] is not None:
          speaker = max(precomp_faces[i], key=lambda item: item[2])
          new_pos = speaker[0]
          x_positions.append(new_pos)
          prev_pos = new_pos

       else:
          x_positions.append(prev_pos)
          
    #shots = [(int(shot*fps) - frame_start) for shot in shots if shot*fps > frame_start and shot*fps < frame_end]
    shots = [shot- frame_start for shot in shots if shot > frame_start and shot < frame_end]
        
    if shots[0]>0:
       shots.insert(0, 0)
    if shots[-1] < (frame_end-frame_start):
       shots.append(frame_end-frame_start)
    
    x_smoothed = []
    
    for i in range(len(shots) - 1):
       start_idx = shots[i]
       end_idx = shots[i + 1]
    
       x_slice = x_positions[start_idx:end_idx]
       
       window = min (101, len(x_slice))
       smoothed_slice = signal.savgol_filter(x_slice, window, 3)
       
       smoothed_slice[0] = x_positions[start_idx]
       smoothed_slice[-1] = x_positions[end_idx-1]       
    
       x_smoothed.extend(smoothed_slice)
      
    prev_w = 0
    for i in range(frame_start, frame_end):
       print("processing frame:", i)
       vidcap.set(cv2.CAP_PROP_POS_FRAMES, i)
       ret, frame = vidcap.read()  
       
       if precomp_faces[i][0] is not None:
         speaker = max(precomp_faces[i], key=lambda item: item[2])
         x_anchor = int(x_smoothed[i-frame_start] + speaker[2]/2)
         prev_pos = x_anchor
       else:
         x_anchor = prev_pos
       cropped_frame = frame[:, x_anchor-int(target_width/2): x_anchor+int(target_width/2)]
       
       output_video.write(cropped_frame)
       
    temp_audio_path = output_video_path.replace('.mp4', '.aac')
    audio_cmd = f"ffmpeg -y -i {vid_path} -ss {start} -to {end} -vn {temp_audio_path}"
    os.system(audio_cmd)
                        
    vidcap.release()
    output_video.release()

    merge_cmd = f"ffmpeg -y -i {temp_output_video_path} -i {temp_audio_path} -c:v copy -c:a aac {output_video_path}"
    os.system(merge_cmd)

    os.remove(temp_audio_path)
    os.remove(temp_output_video_path)      
     
    
############################## AUTO EMOJI ##################################################################################    


def get_emoji (precomp_emojis_path, emojis_folder, description):

    lm_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # please do not remove this line
    print("Using:", device)
    clip_model, preprocess = clip.load("ViT-B/32", device=device)
    
    with open(precomp_emojis_path, 'rb') as file:
        precomp_emojis = pickle.load(file) 

    sim_list = []

    with torch.no_grad():
    
       query = clip.tokenize(description).to(device)         
       embedding = clip_model.encode_text(query)
       embedding  /= embedding.norm(dim=-1, keepdim=True)
       txt_embedding_clip  = embedding.T.cpu().numpy().squeeze()
   
    txt_embedding_lm = lm_model.encode(description).squeeze()
   
    for i in range(len(precomp_emojis)):
       emoji_embedding_txt = precomp_emojis[i][3]
       if emoji_embedding_txt is not None:
          emoji_embedding_txt = emoji_embedding_txt.squeeze()
       emoji_embedding_viz = precomp_emojis[i][4].squeeze()
      
       if emoji_embedding_txt is not None:
          similarity_txt = 100.0 * emoji_embedding_txt@ txt_embedding_lm
       else:
          similarity_txt = 100.0 * emoji_embedding_viz@ txt_embedding_clip
         
       similarity_viz = 100.0 * emoji_embedding_viz@ txt_embedding_clip   
   
       similarity = similarity_viz+similarity_txt    
       sim_list.append([similarity, precomp_emojis[i][0], precomp_emojis[i][2]])

    max_sim= max(sim_list, key=lambda x: x[0])
    sim = max_sim[0]
    result_path = emojis_folder + max_sim [1]
    
    return result_path, max_sim
    

############################## KEYFRAME TRACKER ##################################################################################   


def interpolate_shot_positions(vid_id, keyframes, shot_start, shot_end):
    from app.beta.models.video import Video
    video = Video.query.get(vid_id)
    if video is None:
        return
    vid_path = f"{video.base_path}{video.path}"    
    cap = cv2.VideoCapture(vid_path)
    
    positions = []
    sizes = []    
    frame_numbers = []
    interpolated_positions = []

    cap.set(cv2.CAP_PROP_POS_MSEC, shot_start * 1000)
    shot_start_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))    
    cap.set(cv2.CAP_PROP_POS_MSEC, shot_end * 1000)
    shot_end_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))       
            
    for keyframe in keyframes:
        frame_in_sec = keyframe["frame_in_sec"]
        cap.set(cv2.CAP_PROP_POS_MSEC, frame_in_sec * 1000)
        frame_start = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        frame_numbers.append(frame_start)
        reframe = keyframe["reframe"]
        positions.append([reframe["leftOffset"], reframe["topOffset"]])
        sizes.append([reframe["width"], reframe["height"]])
        
    first_keyframe_number = frame_numbers[0]
    first_keyframe_reframe = keyframes[0]["reframe"]
    for frame_num in range(shot_start_frame, first_keyframe_number):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        frame_in_sec = cap.get(cv2.CAP_PROP_POS_MSEC)/1000       
        interpolated_positions.append({"reframe": first_keyframe_reframe, "frame_number": frame_num, "frame_in_sec": frame_in_sec})
            
    for i in range(len(frame_numbers) - 1):
        start_frame = frame_numbers[i]
        end_frame = frame_numbers[i + 1]
        start_position = np.array(positions[i])
        end_position = np.array(positions[i + 1])
        start_size = np.array(sizes[i])
        end_size = np.array(sizes[i + 1])
        
        pixel_distance = end_position - start_position
        size_distance = end_size - start_size        
        keyframe_duration = end_frame - start_frame
        
        for j in range(keyframe_duration):
            t = j / keyframe_duration
            interpolated_position = np.round(start_position + t * (pixel_distance)).astype(int)
            interpolated_size = np.round(start_size + t * (size_distance)).astype(int)            
            frame_num = start_frame + j
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            frame_in_sec = cap.get(cv2.CAP_PROP_POS_MSEC)/1000             
            interpolated_positions.append({"reframe": {"leftOffset": interpolated_position[0], "topOffset": interpolated_position[1],
                                           "width": interpolated_size[0], "height": interpolated_size[1]}, "frame_number": frame_num, "frame_in_sec": frame_in_sec})

    last_keyframe_number = frame_numbers[-1]
    last_keyframe_reframe = keyframes[-1]["reframe"]  
    for frame_num in range(last_keyframe_number, shot_end_frame):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        frame_in_sec = cap.get(cv2.CAP_PROP_POS_MSEC)/1000      
        interpolated_positions.append({"reframe": last_keyframe_reframe, "frame_number": frame_num, "frame_in_sec": frame_in_sec})
        
    cap.release()
        
    return interpolated_positions

def keyframe_tracker(vid_id, clip_start, clip_end, output_width, output_height, output_video_path, interpolated_positions):
    from app.beta.models.video import Video
    video = Video.query.get(vid_id)
    if video is None:
        return
    vid_path = f"{video.base_path}{video.path}"

    temp_output_video_path = output_video_path.replace('.mp4', '_temp.mp4')
        
    cap = cv2.VideoCapture(vid_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_output_video_path, fourcc, fps, (output_width, output_height))

    for position in interpolated_positions:
        frame_number = position["frame_number"]
        print("processing frame:", frame_number)
        reframe = position['reframe']
        x1, y1, w, h = reframe["leftOffset"], reframe["topOffset"], reframe["width"], reframe["height"]
        
        x2 = x1 + w
        if x2 > width:
            x1 = width - w
            x2 = width
            
        y2 = y1 + h
        if y2 > height:
            y1 = height-h
            y2 = height
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)        
        ret, frame = cap.read()
        
        if ret:
            reframed_frame = frame[y1:y2, x1:x2]      
            reframed_frame = cv2.resize(reframed_frame, (output_width, output_height))                       
            out.write(reframed_frame)

    cap.release()
    out.release()

    temp_audio_path = output_video_path.replace('.mp4', '.aac')
    audio_cmd = f"ffmpeg -y -i {vid_path} -ss {clip_start} -to {clip_end} -vn -filter:a 'atempo={cap.get(cv2.CAP_PROP_FPS)}/{cap.get(cv2.CAP_PROP_FPS)}' {temp_audio_path}"
    os.system(audio_cmd)

    merge_cmd = f"ffmpeg -y -i {temp_output_video_path} -i {temp_audio_path} -c:v copy -c:a aac {output_video_path}"
    os.system(merge_cmd)

    os.remove(temp_audio_path)
    os.remove(temp_output_video_path)
        
############################## AUTO TRACKER ##################################################################################   

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




def tr_frame(frame):
    frame_pil = Image.fromarray(frame).convert("RGB") 

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    frame, _ = transform(frame_pil, None)  # 3, h, w
    return frame, frame_pil


def load_det_model(model_config_path, model_checkpoint_path, device):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(load_res)
    _ = model.eval()
    return model


def get_det_output(model, frame, caption, box_threshold, text_threshold, with_logits=True, device="cpu"):
    frame, frame_pil = tr_frame(frame)
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
    model = model.to(device)
    frame = frame.to(device)
    with torch.no_grad():
        outputs = model(frame[None], captions=[caption])
    logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)
    logits.shape[0]

    # filter output
    logits_filt = logits.clone()
    boxes_filt = boxes.clone()
    filt_mask = logits_filt.max(dim=1)[0] > box_threshold
    logits_filt = logits_filt[filt_mask]  # num_filt, 256
    boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
    logits_filt.shape[0]

    # get phrase
    tokenlizer = model.tokenizer
    tokenized = tokenlizer(caption)
    # build pred
    pred_phrases = []
    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
        if with_logits:
            pred_phrases.append([pred_phrase , round(logit.max().item(), 3)])
        else:
            pred_phrases.append(pred_phrase)

    return boxes_filt, pred_phrases
    
    
def get_frame_dinosam_seg (frame, prompt, sam_predictor, det_model, box_threshold, text_threshold, device):  
    boxes_filt, pred_phrases = get_det_output(det_model, frame, prompt, box_threshold, text_threshold, device=device)

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    sam_predictor.set_image(frame)
    
    size = frame.shape
    H, W = size[0], size[1]
    for i in range(boxes_filt.size(0)):
        boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
        boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
        boxes_filt[i][2:] += boxes_filt[i][:2]

    boxes_filt = boxes_filt.cpu()

    if boxes_filt.shape[0]>=1:
        index_of_max_conf = max(enumerate(pred_phrases), key=lambda x: x[1][1])[0]
        
        prompt_match_box = boxes_filt[index_of_max_conf].unsqueeze(0)
        
        transformed_boxes = sam_predictor.transform.apply_boxes_torch(prompt_match_box, image.shape[:2]).to(device)
    
        seg_mask, _, _ = sam_predictor.predict_torch(
            point_coords = None,
            point_labels = None,
            boxes = transformed_boxes.to(device),
            multimask_output = False,)
        seg_mask = seg_mask.squeeze().detach().cpu().numpy()

    else:
        seg_mask = np.zeros(frame.shape[:2])
    
    return seg_mask
    

def get_frame_clipseg (frame, prompt, processor, model):
  inputs = processor(text=[prompt], images=[frame], padding="max_length", return_tensors="pt")
  
  with torch.no_grad():
    outputs = model(**inputs)
    heatmap = outputs.logits.unsqueeze(1)
    heatmap = torch.transpose(heatmap, 0, 1)
    heatmap = torch.sigmoid(heatmap[0]).detach().cpu().numpy()
    heatmap = cv2.convertScaleAbs(heatmap)
      
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(heatmap, connectivity=8)    
    sum_pixel_values = [np.sum(heatmap[labels == label]) for label in range(1, num_labels)]
      
    seg_mask = np.zeros_like(heatmap)
      
    if sum_pixel_values:
        hottest_label = np.argmax(sum_pixel_values) + 1  # Add 1 to account for background label 0
    
        seg_mask = np.zeros_like(heatmap)
        seg_mask[labels == hottest_label] = 1

    width, height = frame.shape[:2]
    seg_mask = cv2.resize(seg_mask, (height, width))

    return seg_mask    
    
    
def get_keyframe_masks(vid_path, start, end, shots_path, prompt, approach = "dinoseg", device = "cuda"):
    with open(shots_path, 'rb') as file:
        shots= pickle.load(file) 

    vidcap = cv2.VideoCapture(vid_path)
    length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = round(vidcap.get(cv2.CAP_PROP_FPS))

    if start==None:
        start = 0
    if end==None:
        end = length
        
    shots = [(int(shot*fps)-start) for shot in shots if (int(shot*fps) > start and int(shot*fps) < end)]
    shots.append(end)

    if approach=="dinoseg":
        sam_checkpoint = "/path/to/sam_hq_vit_b.pth"
        model_type = "vit_b"
        config_file = "/path/to/GroundingDINO_SwinT_OGC.py"
        grounded_checkpoint = "/path/to/groundingdino_swint_ogc.pth"  
        box_threshold = 0.3 
        text_threshold = 0.25
        
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=device)
        sam_predictor = SamPredictor(sam)  
        det_model = load_det_model(config_file, grounded_checkpoint, device=device)

    else:
        processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
        model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")
    
    seg_masks = []
    for shot_id, shot_end in enumerate(shots):
        print("computing keyframe mask of shot:", shot_id)
        if shot_id == 0:
            shot_start = 0
        else:
            shot_start = shots[shot_id-1] 

        keyframe = int(((shot_end-shot_start)/2)*fps)
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, keyframe)

        ret, frame = vidcap.read()
        if approach=="dinoseg":
            seg_mask = get_frame_dinosam_seg (frame, prompt, sam_predictor, det_model, box_threshold, text_threshold, device)
        else:
            seg_mask = get_frame_clipseg (frame, prompt, processor, model)

        seg_masks.append(seg_mask)
        
    vidcap.release()

    return seg_masks
        

def propagate_mask (vidcap, seg_mask, model, cfg, start, end, stride, device):
    processor = InferenceCore(model, cfg=cfg)   
    num_objects = len(np.unique(seg_mask))
    seg_masks = []

    is_object_in_frame = np.sum(seg_mask == object_id) >= 100

    if is_object_in_frame:
        for frame_id in range(start, end, stride):
            if frame_id%(int(stride*5))==0:
               print("Computing segmask of frame: ", frame_id)
            
            with torch.inference_mode():
              with torch.cuda.amp.autocast(enabled=True):
                  vidcap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
                  _, frame = vidcap.read() 
            
                  frame_torch = image_to_torch(frame, device=device)
                  
                  if frame_id == start:
                    mask_torch = index_numpy_to_one_hot_torch(seg_mask, num_objects).to(device)
                    prediction = processor.step(frame_torch, mask_torch[1:], idx_mask=False)
                  else:
                    prediction = processor.step(frame_torch)
        
                    seg_mask = torch_prob_to_numpy_mask(prediction)
                          
                    if start + len(seg_masks) + stride <= end:
                        for j in range(stride):
                           seg_masks.append(seg_mask)
                   
                    else:
                        prev_len = len(seg_masks)
                        for j in range(end-start-prev_len):
                            print("Computing remaining seg_mask of frame: ", prev_len+j)
                            seg_masks.append(seg_mask)
                                
            torch.cuda.empty_cache()
    else: 
        for frame_id in range(start, end-1):
            seg_masks.append(seg_mask)
            
    vidcap.release()
    
    return seg_masks
    
    
def get_shots_propagated_masks (vid_path, seg_masks, start= None, end = None, stride_factor = 0.5, device = 'cuda'):
    start_time = time.time()

    with open(shots_path, 'rb') as file:
        shots= pickle.load(file) 
        
    vidcap = cv2.VideoCapture(vid_path)
    fps = round(vidcap.get(cv2.CAP_PROP_FPS))
    length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    vid_shape = [int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))]

    if start==None:
        start = 0
    if end==None:
        end = length

    with torch.inference_mode():
      initialize(version_base='1.3.2', config_path="cutie/config", job_name="eval_config")
      cfg = compose(config_name="eval_config")
    
      with open_dict(cfg):
        cfg['weights'] = './weights/cutie-base-mega.pth'
    
      data_cfg = get_dataset_cfg(cfg)
    
      # Load the network weights
      cutie = CUTIE(cfg).cuda().eval()
      model_weights = torch.load(cfg.weights)
      cutie.load_weights(model_weights)    

    torch.cuda.empty_cache()
    
    stride = int(stride_factor*fps)

    shots = [(int(shot*fps)-start) for shot in shots if (int(shot*fps) > start and int(shot*fps) < end)]
    shots.append(end)

    propagated_seg_masks = []
    
    for shot_id, shot_end in enumerate(shots):
       print("propagating masks for shot:", shot_id)
        
       if shot_id == 0:
          shot_start = 0
       else:
          shot_start = shots[shot_id-1]

       seg_mask  = seg_masks [shot_id]
           
       keyframe_seg_masks = []
        
       for j in range(stride):
           keyframe_seg_masks.append(seg_mask)
                                       
       propagate_start = int((shot_end-shot_start)/2)
        
       shot_seg_masks_forward = propagate_mask (vidcap, seg_mask, cutie, cfg, propagate_start, shot_end, stride, device)
       shot_seg_masks_backward = propagate_mask (vidcap, seg_mask, cutie, cfg, propagate_start, shot_start, -stride, device)

       shot_propagated_seg_masks = shot_seg_masks_backward + keyframe_seg_masks + shot_seg_masks_forward

       propagated_seg_masks.extend(shot_propagated_seg_masks)

    end_time = time.time()
    print("Seg_masks propagation processing took:", end_time - start_time)
    
    vidcap.release()

    return propagated_seg_masks
    
                       
def get_frame_heatmap(img, query_features, model, preprocess, device, vid_shape, batch_size = 1, start_layer=-1):

    img = expand2square (img,(0, 0, 0))
    img = preprocess(img).unsqueeze(0).to(device)
    img= img.repeat(batch_size, 1, 1, 1)
    image_features = model.encode_image(img)
    
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    query_features = query_features / query_features.norm(dim=-1, keepdim=True)

    batch_size = image_features.shape[0]
    logit_scale = model.logit_scale.exp()
    logits_per_image = logit_scale * image_features @ query_features.t()
    probs = logits_per_image.softmax(dim=-1).detach().cpu().numpy()
    
    index = [i for i in range(batch_size)]
    one_hot = np.zeros((logits_per_image.shape[0], logits_per_image.shape[1]), dtype=np.float32)
    one_hot[torch.arange(logits_per_image.shape[0]), index] = 1
    one_hot = torch.from_numpy(one_hot).requires_grad_(True)
    one_hot = torch.sum(one_hot * logits_per_image) #.cuda()
    
    model.zero_grad()

    image_attn_blocks = list(dict(model.visual.transformer.resblocks.named_children()).values())

    if start_layer == -1: 
      start_layer = len(image_attn_blocks) - 1
    
    num_tokens = image_attn_blocks[0].attn_probs.shape[-1]
    R = torch.eye(num_tokens, num_tokens, dtype=image_attn_blocks[0].attn_probs.dtype).to(device)
    R = R.unsqueeze(0).expand(batch_size, num_tokens, num_tokens)
    
    for i, blk in enumerate(image_attn_blocks):
        if i < start_layer:
            continue
        grad = torch.autograd.grad(one_hot, [blk.attn_probs], retain_graph=True)[0].detach()
        cam = blk.attn_probs.detach()
        cam = cam.reshape(-1, cam.shape[-1], cam.shape[-1])
        grad = grad.reshape(-1, grad.shape[-1], grad.shape[-1])
        cam = grad * cam
        cam = cam.reshape(batch_size, -1, cam.shape[-1], cam.shape[-1])
        cam = cam.clamp(min=0).mean(dim=1)
        R = R + torch.bmm(cam, R)
        
    image_relevance = R[:, 0, 1:]

    dim = int(image_relevance.numel() ** 0.5)
    image_relevance = image_relevance.reshape(1, 1, dim, dim)
    image_relevance = torch.nn.functional.interpolate(image_relevance, size=224, mode='bilinear')
    image_relevance = image_relevance.reshape(224, 224).data.cpu().numpy() #.cuda()
    image_relevance = (image_relevance - image_relevance.min()) / (image_relevance.max() - image_relevance.min())
    
    heatmap = np.uint8(255 * image_relevance)      
    heatmap = np.float32(heatmap) / 255
    heatmap = cv2.resize(heatmap, (vid_shape[0], vid_shape[0]))
    
    return heatmap
    
    
def get_all_heatmaps(vid_id, query, query_type, start, end, stride_factor):
    start_time = time.time()
    
    from app.beta.models.video import Video
    video = Video.query.get(vid_id)
    if video is None:
        return
    vid_path = f"{video.base_path}{video.path}"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip_grad.load("ViT-B/16", device=device, jit=False)
    
    if query_type == 'text':
        query = [query]    
        query = clip_grad.tokenize(query).to(device)
        query_features = model.encode_text(query)
    else:
        query = preprocess(Image.open(query)).unsqueeze(0).to(device)
        query_features = model.encode_image(query)
    
    vidcap = cv2.VideoCapture(vid_path)
    
    fps = round(vidcap.get(cv2.CAP_PROP_FPS))
    length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    vid_shape = [int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))]

    heatmaps = []

    stride = int(stride_factor*fps)
    
    for i in range(start, end, stride):
        if i%(int(fps*5))==0:
           print("Computing heatmap of frame: ", i)
        
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, i)
        _, frame = vidcap.read() 
        
        heatmap = get_frame_heatmap(Image.fromarray(frame), query_features, model, preprocess, device, vid_shape)

        if start + len(heatmaps) + stride <= end:
           for j in range(stride):
              heatmaps.append(heatmap)
               
        else:
           prev_len = len(heatmaps)
           for j in range(end-start-prev_len):
              print("Computing remaining heatmap of frame: ", prev_len+j)
              heatmaps.append(heatmap)

    end_time = time.time()
    print("Heatmap processing took:", end_time - start_time)
    
    vidcap.release()
    
    return heatmaps


def get_all_masks_clipseg (vid_path, prompt, start= None, end = None, stride_factor = 0.5):
    start_time = time.time()
        
    vidcap = cv2.VideoCapture(vid_path)
    fps = round(vidcap.get(cv2.CAP_PROP_FPS))
    length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    vid_shape = [int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))]

    if start==None:
        start = 0
    if end==None:
        end = length
    
    from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
                                       
    processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
    model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")

    stride = int(stride_factor*fps)
    
    seg_masks = []
    for frame_id in range(start, end, stride):
        if frame_id%(fps*5)==0:
            print("computing mask of frame:", frame_id)
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)

        ret, frame = vidcap.read()
        seg_mask = get_frame_clipseg (frame, prompt, processor, model)
                    
        if start + len(seg_masks) + stride <= end:
            for j in range(stride):
               seg_masks.append(seg_mask)
       
        else:
            prev_len = len(seg_masks)
            for j in range(end-start-prev_len):
                print("Computing remaining seg_mask of frame: ", prev_len+j)
                seg_masks.append(seg_mask)
    
    end_time = time.time()
    print("Seg_masks processing took:", end_time - start_time)
    
    vidcap.release()

    return seg_masks
        
    
def get_all_masks_dinosam (vid_path, prompt, start= None, end = None, stride_factor = 0.5, device="cuda"):
    start_time = time.time()
    box_threshold = 0.3
    text_threshold = 0.25
    sam_checkpoint = "/path/to/sam_hq_vit_b.pth"
    model_type = "vit_b"
    config_file = "/path/to/GroundingDINO_SwinT_OGC.py"
    grounded_checkpoint = "/path/to/groundingdino_swint_ogc.pth"  
         
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    sam_predictor = SamPredictor(sam)  
    det_model = load_det_model(config_file, grounded_checkpoint, device=device)
    
    vidcap = cv2.VideoCapture(vid_path)
    fps = round(vidcap.get(cv2.CAP_PROP_FPS))
    length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    vid_shape = [int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))]

    if start==None:
        start = 0
    if end==None:
        end = length                                    

    stride = int(stride_factor*fps)
    
    seg_masks = []
    for frame_id in range(start, end, stride):
        if frame_id%(fps*5)==0:
            print("computing mask of frame:", frame_id)
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)

        ret, frame = vidcap.read()
        
        boxes_filt, pred_phrases = get_det_output(det_model, frame, prompt, box_threshold, text_threshold, device=device)
    
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        sam_predictor.set_image(frame)
        
        size = frame.shape
        H, W = size[0], size[1]
        for i in range(boxes_filt.size(0)):
            boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
            boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
            boxes_filt[i][2:] += boxes_filt[i][:2]
    
        boxes_filt = boxes_filt.cpu()
    
        if boxes_filt.shape[0]>=1:
            index_of_max_conf = max(enumerate(pred_phrases), key=lambda x: x[1][1])[0]
            
            prompt_match_box = boxes_filt[index_of_max_conf].unsqueeze(0)
            
            transformed_boxes = sam_predictor.transform.apply_boxes_torch(prompt_match_box, frame.shape[:2]).to(device)
        
            seg_mask, _, _ = sam_predictor.predict_torch(
                point_coords = None,
                point_labels = None,
                boxes = transformed_boxes.to(device),
                multimask_output = False,)
            seg_mask = seg_mask.squeeze().detach().cpu().numpy()
    
        else:
            seg_mask = np.zeros(frame.shape[:2])     
        
        if start + len(seg_masks) + stride <= end:
            for j in range(stride):
               seg_masks.append(seg_mask)
       
        else:
            prev_len = len(seg_masks)
            for j in range(end-start-prev_len):
                print("Computing remaining seg_mask of frame: ", prev_len+j)
                seg_masks.append(seg_mask)
    
    end_time = time.time()
    print("Seg_masks processing took:", end_time - start_time)
    
    vidcap.release()

    return seg_masks
    
    
def get_mass_center (seg_mask, object_id):
    object_coords = np.argwhere(seg_mask == object_id)
    center_of_mass = np.mean(object_coords, axis=0)    

    return center_of_mass
    

def get_focals_seg (seg_masks, object_id, vid_width, in_frame_threshold = 100):
   start_time = time.time()
   
   focals = []
   for i in range(len(seg_masks)):
       if i%(int(30*5))==0:
           print("Computing focal of frame: ", i)
        
       seg_mask = seg_masks[i]

       is_object_in_frame = np.sum(seg_mask == object_id) >= in_frame_threshold

       if is_object_in_frame:
          center_of_mass = get_mass_center (seg_mask, object_id)
          focal = center_of_mass[1]
          focals.append(focal)
       else:
          focals.append(int(vid_width/2))
           
   focals = np.array(focals)
    
   end_time = time.time()
   print("Focals processing took:", end_time - start_time)  
   
   return focals
   
   
def get_focals_det (vid_path, prompt, start= None, end = None, stride_factor = 0.5, box_threshold = 0.3, text_threshold = 0.25, device = "cuda"):
    start_time = time.time()
        
    vidcap = cv2.VideoCapture(vid_path)
    fps = round(vidcap.get(cv2.CAP_PROP_FPS))
    length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    W, H = [int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))]

    if start==None:
        start = 0
    if end==None:
        end = length

    stride = int(stride_factor*fps)
            
    config_file = "/home/loukkal/video_processing/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    grounded_checkpoint = "/home/loukkal/video_processing/GroundingDINO/weights/groundingdino_swint_ogc.pth"  
    det_model = load_det_model(config_file, grounded_checkpoint, device=device)
    
    focals = []
    boxes = []
    prev_focal = int(W/2)    
    prev_box = [580.8739, 551.4210, 625.3949, 593.6500]        
    
    for frame_id in range(start, end, stride):
        if frame_id%(fps*5)==0:
            print("computing mask of frame:", frame_id)
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)

        ret, frame = vidcap.read()
        
        boxes_filt, pred_phrases = get_det_output(det_model, frame, prompt, box_threshold, text_threshold, device=device)

        for i in range(boxes_filt.size(0)):
            boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
            boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
            boxes_filt[i][2:] += boxes_filt[i][:2]
    
        boxes_filt = boxes_filt.cpu()
    
        if boxes_filt.shape[0]>=1:
            index_of_max_conf = max(enumerate(pred_phrases), key=lambda x: x[1][1])[0]
            
            prompt_match_box = boxes_filt[index_of_max_conf].squeeze().detach().cpu().numpy()
            focal = int((prompt_match_box[2]-prompt_match_box[0])/2)
            prev_focal = focal
            box = prompt_match_box
            prev_box = box
        else:
            focal = prev_focal 
            box = prev_box
            
            
        if start + len(focals) + stride <= end:
            for j in range(stride):
               focals.append(focal)
               boxes.append(box)
       
        else:
            prev_len = len(focals)
            for j in range(end-start-prev_len):
                print("Computing remaining bb of frame: ", prev_len+j)
                focals.append(focal)
                boxes.append(box)
                
    end_time = time.time()
    print("Focals from det processing took:", end_time - start_time)  
    
    vidcap.release()

    return focals, boxes


    
def get_focal_heatmap(heatmap, vid_shape, ratio = 9/16, vertical_split = False, nbr_splits = 3, step_size = 20): 

    ratio_width = int(vid_shape[1]*ratio)
    region_width = int(ratio_width/nbr_splits)
    region_height = int(vid_shape[1] / nbr_splits)
    
    region_heats = []
    
    for focal in range(int(region_width/2), vid_shape[0]-int(region_width/2), step_size):
        
        x1 = focal-int(region_width/2)
        x2 = x1 + region_width 
        
        if vertical_split: 
            y_pics = []
            for y in range (0, vid_shape[1]-region_height, step_size):
                y_pics.append(np.sum(heatmap[y:y+region_height, x1:x2]))
            region_heats.append([focal, max(y_pics)])
        else:
            region_heats.append([focal, np.sum(heatmap[:, x1:x2])])

    focal = max(region_heats,key=itemgetter(1))[0] 
    
    return focal
    
def get_focals_heatmaps (heatmaps, vid_shape, ratio):
   start_time = time.time()
   
   focals = []
   for i in range(len(heatmaps)):
       if i%(int(30*5))==0:
           print("Computing focal of frame: ", i)
        
       heatmap = heatmaps[i]
        
       focal = get_focal (heatmap, vid_shape, ratio)
       focals.append(focal)

   focals = np.array(focals)
   
   end_time = time.time()
   print("Cropping positions processing took:", end_time - start_time)  
   
   return focals
   
   
def get_croping_positions(vid_shape, focal, ratio):
    
    ratio_width = int(vid_shape[1]*ratio)
    
    x1 = int(max (focal-int(ratio_width/2), 0))
    x2 = x1 + ratio_width
    
    if x2 > vid_shape[0]:
        x2 = vid_shape[0]
        x1 = x2 - ratio_width
    return x1, x2
    
    
def get_positions(vid_path, shot_path, focals, start =None, end = None, ratio = 9/16, mode = "shot_splitter", smoothness = 101, transition_factor= 0.1):
    start_time = time.time()    
            
    vidcap = cv2.VideoCapture(vid_path)
    length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = round(vidcap.get(cv2.CAP_PROP_FPS))
    vid_shape = [int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))]

    if start==None:
        start = 0
    if end==None:
        end = length
        
    cropping_positions = []

    with open(shots_path, 'rb') as file:
        shots= pickle.load(file) 
                           
    if mode == "smooth_follow": 
        smoothness = min(smoothness, len(focals))
        focals = signal.savgol_filter(focals, int(smoothness), 3)
        for i in range (len(focals)):
           focal = focals [i]
           cropping_position = get_croping_positions(vid_shape, focal, ratio)
           cropping_positions.append(cropping_position)
            
    elif mode == "shot_splitter": 

        shots = [(int(shot*fps)-start) for shot in shots if (int(shot*fps) > start and int(shot*fps) < end)]
    
        smoothed_focals = []
        for shot_id, shot in enumerate(shots):
           if shot_id == 0:
              shot_start = 0
           else:
              shot_start = shots[shot_id-1]
    
           smoothness = min(smoothness, len(focals[shot_start:shots[shot_id]]))
           smoothed_focals.extend(signal.savgol_filter(focals[shot_start:shots[shot_id]], smoothness, 3))
    
        smoothness =  min(smoothness, len(focals[shots[-1]:end]))
        smoothed_focals.extend(signal.savgol_filter(focals[shots[-1]:end], smoothness, 3))
        focals = smoothed_focals
    
        focals_scenes = []
        scene = 0
        
        focal_scene = np.mean(focals[:shots[scene]])

        for i in range(len(focals)):
            if scene < len(shots) - 1:
                scene_length = shots[scene + 1] - shots[scene]
                transition = int(transition_factor * scene_length)
        
                if scene != 0 and i >= shots[scene] and i < shots[scene] + transition:
                    t = (shots[scene] + transition -i) / transition
                    prev_avg = np.mean(focals[shots[scene-1]:shots[scene]])   
                    next_avg = np.mean(focals[shots[scene]:shots[scene + 1]])  
                    focal_scene = (1 - t) * next_avg + t * prev_avg
                elif i == shots[scene] + transition:
                    focal_scene = np.mean(focals[shots[scene]:shots[scene + 1]])
                    scene += 1
                else:
                    pass
            else:
                focal_scene = np.mean(focals[shots[scene]:])
                
            focals_scenes.append(focal_scene)
        
        focals = [int(focal) for focal in  focals_scenes]
        
        for i in range (len(focals)):
           focal = focals [i]
           cropping_position = get_croping_positions(vid_shape, focal, ratio)
           cropping_positions.append(cropping_position)

    end_time = time.time()
    print("Cropping positions processing took:", end_time - start_time) 
    
    vidcap.release() 
                
    return cropping_positions
    

def auto_tracker_create_clip (vid_path, cropping_positions, start, end, output_video_path, ratio):
    start_time = time.time()  
    
    vidcap = cv2.VideoCapture(vid_path)
    length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(vidcap.get(cv2.CAP_PROP_FPS))
    width = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if start==None:
        start = 0
    if end==None:
        end = length
        
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    temp_output_video_path = output_video_path.replace('.mp4', '_temp.mp4') 
    output_video = cv2.VideoWriter(temp_output_video_path, fourcc, fps, (int(height*ratio), height))

    vidcap.set(cv2.CAP_PROP_POS_FRAMES, start)

    for frame_id in range(len(cropping_positions)):
        if frame_id%(int(fps*5))==0:
           print('processing frame: ', frame_id + start)

        ret, frame = vidcap.read()  

        x1, x2 = cropping_positions[frame_id]
        cropped_frame = frame[:, x1:x2] 

        output_video.write(cropped_frame)

    temp_audio_path = output_video_path.replace('.mp4', '.aac')
    audio_cmd = f"ffmpeg -y -loglevel error -hide_banner -i {vid_path} -ss {start/fps} -to {end/fps} -vn {temp_audio_path}"
    os.system(audio_cmd)

    vidcap.release()
    output_video.release()

    merge_cmd = f"ffmpeg -y -loglevel error -hide_banner -i {temp_output_video_path} -i {temp_audio_path} -c:v copy -c:a aac {output_video_path}"
    os.system(merge_cmd)

    os.remove(temp_audio_path)
    os.remove(temp_output_video_path)
    
    end_time = time.time()
    print("Creating clip processing took:", end_time - start_time)  
    
    
def debug_create_clip_seg (vid_path, seg_masks, start, end, output_video_path):
    start_time = time.time()  
    output_video_path = output_video_path.replace('.mp4', '_mask.mp4')
    
    vidcap = cv2.VideoCapture(vid_path)
    length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(vidcap.get(cv2.CAP_PROP_FPS))
    width = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if start==None:
        start = 0
    if end==None:
        end = length
        
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    temp_output_video_path = output_video_path.replace('.mp4', '_temp.mp4') 
    output_video = cv2.VideoWriter(temp_output_video_path, fourcc, fps, (width, height))

    vidcap.set(cv2.CAP_PROP_POS_FRAMES, start)

    for frame_id in range(len(seg_masks)):
        if frame_id%(int(fps*5))==0:
           print('processing frame: ', frame_id + start)

        ret, frame = vidcap.read()  

        mask = seg_masks[frame_id]
        overlay_color = (0, 255, 0)
        overlay = np.zeros_like(frame)
        overlay[mask == 1] = overlay_color
        frame = cv2.addWeighted(frame, 1, overlay, 0.5, 0)
        #cv2.imwrite(f"/path/frame_{str(frame_id)}.png", frame)

        output_video.write(frame)

    temp_audio_path = output_video_path.replace('.mp4', '.aac')
    audio_cmd = f"ffmpeg -y -loglevel error -hide_banner -i {vid_path} -ss {start/fps} -to {end/fps} -vn {temp_audio_path}"
    os.system(audio_cmd)

    vidcap.release()
    output_video.release()

    merge_cmd = f"ffmpeg -y -loglevel error -hide_banner -i {temp_output_video_path} -i {temp_audio_path} -c:v copy -c:a aac {output_video_path}"
    os.system(merge_cmd)

    os.remove(temp_audio_path)
    os.remove(temp_output_video_path)
    
    end_time = time.time()
    print("Creating clip processing took:", end_time - start_time)  
    
    
def debug_create_clip_boxes (vid_path, boxes, start, end, output_video_path):
    start_time = time.time()  
    output_video_path = output_video_path.replace('.mp4', '_box.mp4')
    
    vidcap = cv2.VideoCapture(vid_path)
    length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(vidcap.get(cv2.CAP_PROP_FPS))
    width = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if start==None:
        start = 0
    if end==None:
        end = length
        
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    temp_output_video_path = output_video_path.replace('.mp4', '_temp.mp4') 
    output_video = cv2.VideoWriter(temp_output_video_path, fourcc, fps, (width, height))

    vidcap.set(cv2.CAP_PROP_POS_FRAMES, start)

    for frame_id in range(len(boxes)):
        if frame_id%(int(fps*5))==0:
           print('processing frame: ', frame_id + start)

        ret, frame = vidcap.read()  

        x1, y1, x2, y2 = boxes[frame_id]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        #cv2.imwrite(f"/path/frame_{str(frame_id)}.png", frame)

        output_video.write(frame)

    temp_audio_path = output_video_path.replace('.mp4', '.aac')
    audio_cmd = f"ffmpeg -y -loglevel error -hide_banner -i {vid_path} -ss {start/fps} -to {end/fps} -vn {temp_audio_path}"
    os.system(audio_cmd)

    vidcap.release()
    output_video.release()

    merge_cmd = f"ffmpeg -y -loglevel error -hide_banner -i {temp_output_video_path} -i {temp_audio_path} -c:v copy -c:a aac {output_video_path}"
    os.system(merge_cmd)

    os.remove(temp_audio_path)
    os.remove(temp_output_video_path)
    
    end_time = time.time()
    print("Creating clip processing took:", end_time - start_time)  
    
    
############################## ADOBE XML FILE ##################################################################################   


def create_gif(images, output_path, fps, loop=0):
    duration = 1000/fps
    pil_images = []
    for i, image in enumerate(images):
        img = Image.fromarray(image, 'RGBA')
        pil_images.append(img)

    pil_images[0].save(
        output_path,
        save_all=True,
        optimize=False,
        format='GIF',
        append_images=pil_images[1:],
        duration=duration,
        loop=loop
    )

def create_caption_overlay_moviepy(images, output_path, fps):
    images_rgba = [cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA) for image in images]
    clip = ImageSequenceClip(images_rgba, fps=fps)
    clip.write_videofile(output_path, fps=fps)
    
    
def create_caption_overlay_imageio(images, output_path, fps):
    duration = 1/fps
    images_rgba = [cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA) for image in images]

    with imageio.get_writer(output_path, fps = fps, mode='I', subrect=False) as writer:
        for image in images_rgba:
            writer.append_data(image)
    

def create_caption_overlay_ffmpeg(images, output_path, fps):
    for i, image in enumerate(images):
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
        img = Image.fromarray(image, 'RGBA')
        img.save(output_path.replace('.mp4', f"_{i:04d}.png"), format='PNG')
        
    base_name = output_path.replace(".mp4", "")
    
    overlay_cmd = f"ffmpeg -framerate {fps} -pattern_type glob -i '{base_name}_*.png' -c:v libx264 -r {fps} -pix_fmt yuv420p {output_path}"

    os.system(overlay_cmd)

    for i, image in enumerate(images):
        os.remove(output_path.replace('.mp4', f"_{i:04d}.png"))
        

def create_caption_overlay_opencv(images, output_path, fps):
    height, width = images[0].shape[:2]
    print(height, width)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for image in images:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
        video_writer.write(image)

    video_writer.release()
   
                
def create_captions_overlays(fps, vid_duration_frames, vid_shape, captions, overlays_folder_path, word_timestamps, layout = "center", height_offset = 300, width_offset = 50, fontsize=55, word_spacing = 10, line_spacing = 70, uppercase =True, font_path=None, font_color=(255, 255, 255), spoken_word_color=(0, 255, 255), verb_color=None, object_color=None, animation_type = None, shadow_color = (0, 0, 0), shadow_offset = 7, outline_color = (0, 0, 0), outline_width = 6, burn_emoji = False, precomp_emojis_path = None, emojis_folder = None, ext = '.gif', debug_overlays = False):

    overlays_list = []
    width, height = vid_shape

    if font_path:
        font = ImageFont.truetype(font_path, int(fontsize))
    else:
        font = ImageFont.truetype(f"{Config.ROOT_FOLDER}/app/fonts/Montserrat-Black.ttf", int(fontsize))
                        
    caption_counter = 0
    current_caption = None
    
    if burn_emoji:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        clip_model, preprocess = clip.load("ViT-B/32", device=device)
        lm_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    
    nlp = spacy.load("en_core_web_sm")
    
    frames_list = []
    
    for frame_counter in range(vid_duration_frames):
        print('\033[34m' + 'Processing frame:' + '\033[0m', frame_counter)
        
        frame = np.zeros((height, width, 4), dtype=np.uint8)
                     
        frame_timestamp = (frame_counter/fps)*1000

        if current_caption is None or frame_timestamp >= current_caption.get('end'):
            
            filtered_words = []
            for caption in captions:
                start_time = caption.get('start')
                end_time = caption.get('end')

                if start_time <= frame_timestamp <= end_time:
                    current_caption = caption
                    word_types = [token for token in nlp(caption.get('text'))]
                    
                    if current_caption is not None and len(frames_list)>0:
                        print("creating pverlay:", f"caption_{caption_counter}{ext}", "with number of images:", len(frames_list))
                        overlay_path = overlays_folder_path + f"caption_{caption_counter}{ext}"
                        
                        create_caption_overlay_imageio(frames_list, overlay_path, fps)
                        
                        start_in_frames = round((start_time/1000)*fps)
                        end_in_frames = round((end_time/1000)*fps)
                        duration_in_frames = end_in_frames - start_in_frames  
                                  
                        caption_file_name = f"./{'/'.join(overlay_path.split('/')[-2:])}"                                  
                        overlays_dict = {"caption_file_name": caption_file_name,
                                    "duration": duration_in_frames,
                                    "start": start_in_frames,
                                    "end": end_in_frames}
                        overlays_list.append(overlays_dict)
                        
                        frames_list = []                            
                        caption_counter += 1                                                      
                    break 
                else:
                    current_caption = None

        if current_caption:
        
            if burn_emoji:        
                emoji_path, max_sim = get_emoji(clip_model, lm_model, device, precomp_emojis_path, emojis_folder, current_caption.get('text'))

            filtered_words = [word_info for word_info in word_timestamps if
                              current_caption.get('start') <= word_info[1] <= current_caption.get('end') and
                              current_caption.get('start') <= word_info[2] <= current_caption.get('end')]                             
                              
            image = Image.fromarray(frame)
            image_width, image_height = image.size

            draw = ImageDraw.Draw(image)
            
            lines = split_text_into_lines(current_caption.get('text'), font, image_width-2*width_offset, uppercase)
                
            try:
                first_line_width = sum([font.getsize(word)[0] for word in lines[0]]) + word_spacing*(len(lines[0])-1)
            except:
                first_line_width = sum([(font.getbbox(word)[2]) for word in lines[0]]) + word_spacing*(len(lines[0])-1)
                
            text_x = (image_width - first_line_width) // 2
            text_y = frame.shape[0] - height_offset            
            
            line_index = 0
            
            for word_index, word_info in enumerate(filtered_words):
                if word_index > sum(len(lines[prev_line_index])-1 for prev_line_index in range(0, line_index+1)):
                    line_index+=1
                    try:
                        line_width =  sum([font.getsize(word)[0] for word in lines[line_index]]) + word_spacing*(len(lines[line_index])-1)
                    except:
                        line_width =  sum([(font.getbbox(word)[2]) for word in lines[line_index]]) + word_spacing*(len(lines[line_index])-1)      
                    text_x = (image_width - line_width) // 2
                    text_y += line_spacing
                word_text = word_info[0]
                             
                if uppercase:
                    word_text = word_text.upper()
                    
                word_start_time = word_info[1]
                word_end_time = word_info[2]

                font_color_to_use = font_color
                
                try:
                    text_size = font.getsize(word_text)
                except:
                    text_size = font.getbbox(word_text)
                                                                           
                if verb_color and word_types[word_index].pos_ == "VERB":
                    font_color_to_use = verb_color
                        
                if object_color and "obj" in word_types[word_index].dep_:
                    font_color_to_use = object_color

                if animation_type and frame_timestamp >= word_start_time and frame_timestamp <= word_end_time:
                    if animation_type == "word":
                        font_color_to_use = spoken_word_color
                    elif animation_type == "box":
                        font_color_to_use = font_color
                        try:
                            x, y, a, b = text_x-0.5*word_spacing, text_y - 0.1*line_spacing, text_x + text_size[2]+ 0.5*word_spacing, text_y + text_size[3]+ 0.1*line_spacing

                        except:
                            x, y, a, b = text_x-0.5*word_spacing, text_y - 0.1*line_spacing, text_x + text_size[0]+ 0.5*word_spacing, text_y + text_size[1]+ 0.1*line_spacing
                        draw.rounded_rectangle([x, y, a, b], radius=10, fill=spoken_word_color)   

                if shadow_color:
                    shadow_x = text_x + shadow_offset  
                    shadow_y = text_y + shadow_offset

                    draw.text((shadow_x, shadow_y), word_text, fill=shadow_color, font=font)

                if outline_color:
                    draw.text((text_x, text_y), word_text, fill=font_color_to_use, font=font, stroke_width = outline_width, stroke_fill = outline_color)       
                        
                else:                        
                    draw.text((text_x, text_y), word_text, fill=font_color_to_use, font=font)
                        
                try:
                   text_x += text_size[2] + word_spacing
                except:
                   text_x += text_size[0] + word_spacing
                   
        
                if burn_emoji:
                    frame = np.array(image)
                    emoji = cv2.imread(emoji_path, cv2.IMREAD_UNCHANGED)
                    emoji_height, emoji_width, _ = emoji.shape
                    x, x1, y, y1 = int(image_width/2-emoji_width/2), int(image_width/2+emoji_width/2),  text_y + line_spacing*2, text_y + line_spacing*2 + emoji_height
                    
                    roi = slice(y, y1), slice(x, x1)

                    emoji_alpha = emoji[:, :, 3] / 255.0

                    for c in range(0, 3):
                        frame[roi][:, :, c] = emoji[:, :, c] * emoji_alpha + frame[roi][:, :, c] * (1 - emoji_alpha)    
                else:        
                    frame = np.array(image)                    

            frames_list.append(frame)

    if debug_overlays:
        overlays_json_path = overlays_folder_path + "caption_overlays.json"            
        with open(overlays_json_path, 'w') as f:
            json.dump(overlays_list, f, indent = 4)    
        
    return overlays_list        

def frame_to_time(frame_number, fps = 30):
    total_seconds = frame_number / fps
    minutes = int(total_seconds // 60)
    seconds = int(total_seconds % 60)
    return f"{minutes:02d}:{seconds:02d}"

def get_xml_cropping_params (cropping_params, vid_shape, sequence_shape, clip_start, adobe_folder_path ="", debug_cropping = False):

    width, height = vid_shape
    sequence_width, sequence_height = sequence_shape
    duration = len(cropping_params)
    
    prev_start_frame = 0
    xml_cropping_params = []
    
    for frame_counter, cropping_param in enumerate(cropping_params):   

        h1, h2, w1, w2, _ = cropping_param[0]
        
        if frame_counter > 1 and (w1 != prev_w1 or frame_counter == duration-1):
            scale_x = sequence_width/(prev_w2-prev_w1)
            scale_y = sequence_height/(prev_h2-prev_h1)
            scale = int(scale_x*100)        
            xml_left = (prev_w1/width)*100
            xml_right = ((width-prev_w2)/width)*100
            xml_top = (prev_h1/height)*100
            xml_bottom = ((height-prev_h2)/height)*100
            xml_cropping_param = {'start': prev_start_frame, 
                                  'end': frame_counter,
                                  'in':clip_start+prev_start_frame,
                                  'out':clip_start+frame_counter,
                                  'scale': scale,
                                  'crop_param':{'left':{'value': xml_left}, 'right':{'value':xml_right},'top':{'value':xml_top},'bottom':{'value':xml_bottom}}}             
            xml_cropping_params.append(xml_cropping_param)                                     
            prev_start_frame = frame_counter
            
        prev_h1, prev_h2, prev_w1, prev_w2 = h1, h2, w1, w2   
    
    if debug_cropping: 
        cropping_json_path = f"{adobe_folder_path}/cropping_params.json"               
        with open(cropping_json_path, 'w') as f:
            json.dump(xml_cropping_params, f, indent = 4)  
        
    return xml_cropping_params
        
                     
def customize_crop_item(video_file_name, uid, duration, crop_item_data, video_width, video_height, sequence_width, sequence_height, fps, collection = False):

    start = crop_item_data.get('start')
    end = crop_item_data.get('end')
    iin = crop_item_data.get('in')
    out = crop_item_data.get('out')
    scale = crop_item_data.get('scale')
    crop_param = crop_item_data.get('crop_param') 
    if float(fps).is_integer():
        ntsc = "FALSE"
    else: 
        ntsc = "TRUE"
        
    # crop item video
    custom_crop_item = copy.deepcopy(crop_item)
    custom_crop_item['@id'] = f'clipitem-{2 * uid}'
    custom_crop_item['name'] = video_file_name
    custom_crop_item['duration'] = duration
    custom_crop_item['rate']['timebase'] = fps
    custom_crop_item['rate']['ntsc'] = ntsc
    custom_crop_item['start'] = start
    custom_crop_item['end'] = end
    custom_crop_item['in'] = iin
    custom_crop_item['out'] = out
    
    
    file = custom_crop_item['file']
    video_title, _ = os.path.splitext(video_file_name)
    if uid == 0 or collection == True:
        file['@id'] = video_title
        file['name'] = video_file_name
        file['pathurl'] = video_file_name
        file['rate'] = {
                            "timebase":fps,
                            "ntsc":ntsc
                        }
        file['duration'] = duration                        
        file["timecode"] = {
                            "rate":{
                                    "timebase":fps,
                                    "ntsc":ntsc
                                    },
                            "string":"00:00:00:00",
                            "frame":"0",
                            "displayformat":"NDF"
        } 

        file['media'] = {
            "video":{"samplecharacteristics":{          
                                            "width":video_width,
                                            "height":video_height
                                        }},
            "audio":{"channelcount":"2"}
        }
        
    else:
    
        file['@id'] = video_title
                            
    left_cro = custom_crop_item['filter'][-1]["effect"]["parameter"][0]
    right_cro = custom_crop_item['filter'][-1]["effect"]["parameter"][1]
    top_cro = custom_crop_item['filter'][-1]["effect"]["parameter"][2]
    bott_cro = custom_crop_item['filter'][-1]["effect"]["parameter"][3]
    left_cro = {**left_cro,**crop_param['left']}
    right_cro = {**right_cro, **crop_param['right']}
    top_cro = {**top_cro, **crop_param['top']}
    bott_cro = {**bott_cro, **crop_param['bottom']}
    custom_crop_item['filter'][-1]["effect"]["parameter"][0]= left_cro
    custom_crop_item['filter'][-1]["effect"]["parameter"][1]= right_cro
    custom_crop_item['filter'][-1]["effect"]["parameter"][2]= top_cro
    custom_crop_item['filter'][-1]["effect"]["parameter"][3]= bott_cro
    
    #adding center horiz/vert value & scale
    anchor_horiz = float(custom_crop_item['filter'][0]["effect"]["parameter"][3]["value"]["horiz"])
    anchor_vert = float(custom_crop_item['filter'][0]["effect"]["parameter"][3]["value"]["vert"])
    
    custom_crop_item['filter'][0]["effect"]["parameter"][0]["value"] = scale
    
    center_horiz = -((crop_param['left']['value']/100)*(scale/100)*video_width + sequence_width/2 + anchor_horiz*video_width)/video_width
    custom_crop_item['filter'][0]["effect"]["parameter"][2]["value"]["horiz"]= center_horiz

    center_vert = -((crop_param['top']['value']/100)*(scale/100)*video_height + sequence_height/2 + anchor_vert*video_height)/video_height
    custom_crop_item['filter'][0]["effect"]["parameter"][2]["value"]["vert"]= center_vert
       
    # custom audio item
    custom_audio_item = copy.deepcopy(audio_item)
    custom_audio_item['@id'] = f'clipitem-{2 * uid + 1}'
    custom_audio_item['name'] = video_file_name
    custom_audio_item['duration'] = duration
    custom_audio_item['rate']['timebase'] = fps
    custom_audio_item['rate']['ntsc'] = ntsc
    
    custom_audio_item['start'] = start
    custom_audio_item['end'] = end 
    custom_audio_item['in'] = iin
    custom_audio_item['out'] = out
    
    custom_audio_item['file']['@id'] = video_title
    custom_audio_item['link']['linkclipref'] = video_file_name

    return custom_crop_item, custom_audio_item
    

def customize_caption_item(uid, nbr_crp_break_point, caption_file_name, duration, start, end, w, h, fps):

    if float(fps).is_integer():
        ntsc = "FALSE"
    else: 
        ntsc = "TRUE"
        
    custom_caption_item = copy.deepcopy(caption_item)
    custom_caption_item['@id'] = f'clipitem-{nbr_crp_break_point + uid}'
    custom_caption_item['name'] = caption_file_name.split('/')[-1]
    custom_caption_item['duration'] = duration
    custom_caption_item['rate']['timebase'] = fps
    custom_caption_item['rate']['ntsc'] = ntsc
    custom_caption_item['file']['rate']['timebase'] = fps
    custom_caption_item['file']['rate']['ntsc'] = ntsc    
    custom_caption_item['file']['timecode']['rate']['timebase'] = fps
    custom_caption_item['file']['timecode']['rate']['ntsc'] = ntsc  
        
    custom_caption_item['start'] = start
    custom_caption_item['end'] = end 
    custom_caption_item['in'] = 0
    custom_caption_item['out'] = duration

    file = custom_caption_item['file']
    file['@id'] = caption_file_name.split('/')[-1]
    file['name'] = caption_file_name
    file['pathurl'] = caption_file_name
    file['duration'] = duration
    dimension = file['media']['video']["samplecharacteristics"]
    dimension['width'] = w
    dimension['height'] = h

    return custom_caption_item
    

def get_clip_metadata(clip_duration, clip_title, clip_w,clip_h, crop_data, caption_data, raw_video_duration, raw_video_width, 
                            raw_video_height, raw_video_filename, fps, include_captions = False):

    if float(fps).is_integer():
        ntsc = "FALSE"
    else: 
        ntsc = "TRUE"
                                    
    custom_sequence_item = copy.deepcopy(sequence_item)
    sequence = custom_sequence_item["sequence"]
    sequence["rate"]["timebase"] = round(fps)
    sequence["rate"]["ntsc"] = ntsc
    sequence["name"] = clip_title
    sequence["duration"] = clip_duration
    sequence["timecode"]["rate"]["timebase"] = round(fps)
    sequence["timecode"]["rate"]["ntsc"] = ntsc

    media = sequence['media']

    video_intrinsics = media['video']['format']['samplecharacteristics']
    video_intrinsics['width'] = clip_w
    video_intrinsics['height'] = clip_h
    video_intrinsics['rate']['timebase'] = fps
    video_intrinsics['rate']['ntsc'] = ntsc

    video_track = media['video']['track']
    
    crop_items = []

    crop_audio_items = []

    for i, crop_item_data in enumerate(crop_data):
        crop_item_video, crop_item_audio = customize_crop_item(raw_video_filename,i,raw_video_duration,crop_item_data,
        raw_video_width,raw_video_height, clip_w, clip_h, fps)

        crop_items.append(crop_item_video)
        crop_audio_items.append(crop_item_audio)

    video_track[0]['clipitem'] = crop_items

    if include_captions:
        caption_items = []

        for i, caption in enumerate(caption_data):
            caption_file_name = caption.get('caption_file_name')
            duration = caption.get('duration')
            start = caption.get('start')
            end = caption.get('end')

            caption_item = customize_caption_item(i,2*len(crop_data),caption_file_name,duration,start,end,raw_video_width,raw_video_height, fps)

            caption_items.append(caption_item)

        video_track[2]['clipitem'] = caption_items

    audio_track = media['audio']['track']
    audio_track['clipitem'] = crop_audio_items
    
    custom_sequence_item['@version'] = "4"
    return custom_sequence_item
    
    
def get_collection_metadata(collection_id, larger_sequence_dimension = 1920, aspect_ratio = "landscape_16_9"):

    if aspect_ratio == "portrait_9_16":
        sequence_height = larger_sequence_dimension
        sequence_width = int(sequence_height * 9/16)

    elif aspect_ratio == "square":
        sequence_height = larger_sequence_dimension
        sequence_width = sequence_height

    else:
        sequence_width = larger_sequence_dimension
        sequence_height = int(sequence_width * 9/16)

    sequence_shape = (sequence_width, sequence_height)
    
    collection = Collection.query.get(collection_id)
    collection_name = collection.display_name
    
    clip_ids = [int(clip_id) for clip_id in collection.fav_ids.strip('[]').replace('"', '').split(', ')]
    all_saved_clips = db.session.query(SavedClip).all()
    all_saved_clips = [clip.id for clip in all_saved_clips]
    clip_ids = [clip_id for clip_id in clip_ids if clip_id in all_saved_clips]

    clips = [SavedClip.query.get(clip_id) for clip_id in clip_ids]
    sequence_duration = sum([(clip.end-clip.start)*Video.query.get(clip.video_id).fps for clip in clips])
    
    ref_video_id = clips[0].video_id
    ref_video = Video.query.get(ref_video_id)
    ref_vid_path = f"{ref_video.base_path}{ref_video.path}" 
    vidcap = cv2.VideoCapture(ref_vid_path)
    
    ref_video_duration = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    ref_height_orig, ref_width_orig = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(
            vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))

    ref_fps = vidcap.get(cv2.CAP_PROP_FPS)
    
    vidcap.release()
    
    if float(ref_fps).is_integer():
        ntsc = "FALSE"
    else: 
        ntsc = "TRUE"
                                    
    custom_sequence_item = copy.deepcopy(sequence_item)
    sequence = custom_sequence_item["sequence"]
    sequence["rate"]["timebase"] = round(ref_fps)
    sequence["rate"]["ntsc"] = ntsc
    sequence["name"] = collection_name
    sequence["duration"] = sequence_duration
    sequence["timecode"]["rate"]["timebase"] = round(ref_fps)
    sequence["timecode"]["rate"]["ntsc"] = ntsc

    media = sequence['media']

    video_intrinsics = media['video']['format']['samplecharacteristics']
    video_intrinsics['width'] = sequence_width
    video_intrinsics['height'] = sequence_height
    video_intrinsics['rate']['timebase'] = ref_fps
    video_intrinsics['rate']['ntsc'] = ntsc

    video_track = media['video']['track']    

    offset = 0

    crop_video_items = []

    crop_audio_items = []

    for clip_index, clip in enumerate(clips):   
        video_id = clip.video_id
        video = Video.query.get(video_id)
        video_filename = video.file_name
        vid_path = f"{video.base_path}{video.path}" 
        vidcap = cv2.VideoCapture(vid_path)    
        video_duration = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        height_orig, width_orig = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(
            vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
        vid_shape = (width_orig, height_orig)
        fps = vidcap.get(cv2.CAP_PROP_FPS)  
              
        vidcap.set(cv2.CAP_PROP_POS_MSEC, float(clip.start * 1000))
        clip_start = int(vidcap.get(cv2.CAP_PROP_POS_FRAMES))
        vidcap.set(cv2.CAP_PROP_POS_MSEC, float(clip.end * 1000))
        clip_end = int(vidcap.get(cv2.CAP_PROP_POS_FRAMES))
        clip_duration = clip_end-clip_start
         
        cropping_params = []
        if aspect_ratio == "landscape_16_9":         
            for frame_id in range(clip_duration):
                frame_params = [[0, height_orig, 0, width_orig, None]]
                cropping_params.append(frame_params)           
        elif aspect_ratio == "portrait_9_16":                         
            print("Error: not supported for the moment")                 
        
        xml_cropping_params = get_xml_cropping_params(cropping_params, vid_shape, sequence_shape, clip_start)

        for crop_index, crop_item_data in enumerate(xml_cropping_params): 
            absolute_start =  crop_item_data['start']        
            absolute_end =  crop_item_data['end']                    
            crop_item_data['start'] = absolute_start + offset
            crop_item_data['end'] = absolute_end + offset           
            crop_item_video, crop_item_audio = customize_crop_item(video_filename,clip_index+crop_index,video_duration,crop_item_data,
            width_orig,height_orig, sequence_width, sequence_height, fps, collection = True)

            crop_video_items.append(crop_item_video)
            crop_audio_items.append(crop_item_audio)
            offset += absolute_end

    video_track[0]['clipitem'] = crop_video_items
    
    audio_track = media['audio']['track']
    audio_track['clipitem'] = crop_audio_items
    
    custom_sequence_item['@version'] = "4"
    
    return custom_sequence_item    
    
def json_to_xml(json_obj):
    xml_obj = xmltodict.unparse({"xmeml": json_obj}, pretty=True)
    return xml_obj
    

def write_xml_to_file(xml_data, filename):
    with open(filename, 'w') as xml_file:
        xml_file.write(xml_data)
        

def generate_clip_xml_file(xml_path,clip_duration,clip_title,clip_w,clip_h,crop_data,caption_data,
                      raw_video_duration,raw_video_width,raw_video_height,raw_video_file_name, fps, include_captions=False):
                      
    clip_metadata = get_clip_metadata(clip_duration,clip_title,clip_w,clip_h,crop_data,caption_data,
                                         raw_video_duration,raw_video_width,raw_video_height,raw_video_file_name, fps, include_captions)
                                         
    xml_data = json_to_xml(clip_metadata)
    write_xml_to_file(xml_data, xml_path)
    
def generate_collection_xml_file(xml_path, collection_id, larger_sequence_dimension = 1920, aspect_ratio = "landscape_16_9"):
                      
    collection_metadata = get_collection_metadata(collection_id, larger_sequence_dimension, aspect_ratio)
                                         
    xml_data = json_to_xml(collection_metadata)
    write_xml_to_file(xml_data, xml_path)

