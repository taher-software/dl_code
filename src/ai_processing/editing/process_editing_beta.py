import cv2
import numpy as np
import os
import pickle
import clip
import torch
from PIL import Image, ImageDraw, ImageFont
from sentence_transformers import SentenceTransformer
import subprocess
import time
import ffmpeg_streaming
import spacy
import json
from scipy import signal
import app.ai_processing.editing.clip_withgrad as clip_grad
import math
from ffmpeg_streaming import Formats, Bitrate, Representation, Size
from app.enums.upload_processing_status import UploadProcessingStatus


############################## NON-AI EDITING ##################################################################################


def beta_get_files(folder):

    filtered_files = []
    for path, _, files in os.walk(folder):
        for file in files:
            full_path = os.path.join(path, file)
            filtered_files.append(full_path)
    return filtered_files


def beta_re_enc(video):
    """ Re-encode videos
        Arguments:
            vid_path: source video path
    """
    # vid_path = f"{video.base_path}{video.path}"
    # out_path = vid_path.replace ('.mp4', '_reenc.mp4')
    # ffmpeg_cmd = f"ffmpeg -i {vid_path} -c:v libx264 -preset superfast -crf 25 -c:a copy {out_path}"
    # os.system (ffmpeg_cmd)
    last_reported = 0

    vid_path = f"{video.base_path}{video.path}"
    out_path = vid_path.replace('.mp4', '_reenc.mp4')

    start_time = time.time()
    from app.utils import utils
    utils.inform_in_thread(video.id, UploadProcessingStatus.RE_ENCODING)

    cmd = f"ffmpeg -y -i {vid_path} -c:v libx264 -preset superfast -crf 25 -c:a copy {out_path}"
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, shell=True, universal_newlines=True)

    # Keep reading the output of the process until it finishes
    while process.poll() is None:
        line = process.stdout.readline()
        progress = beta_parse_progress(line, video.length)
        try:
            if progress is not None and utils.should_report(last_reported, progress):
                last_reported = progress
                estimated_time = ((time.time() - start_time) / progress)
                utils.inform_in_thread(video.id, UploadProcessingStatus.RE_ENCODING, progress=(
                    progress*100), estimate=estimated_time)
        except:
            pass

    process.wait()
    video.rencoded = 1
    db.session.commit()
    utils.inform_in_thread(video.id, UploadProcessingStatus.RE_ENCODING,
                           progress=100, estimate=((time.time() - start_time)))
    os.remove(vid_path)
    os.rename(out_path, vid_path)


def beta_parse_progress(line, total_frames):
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
        
        
def beta_update_bandwidth_thresholds(hls_path):
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


def beta_hls_convert(mp4_path, hls_path, resolutions=None):
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
        beta_update_bandwidth_thresholds(hls_path)
    except:
        print("Failed to update HLS bandwith thresholds")


def beta_cut_clip(start, end, vid_path, clip_name, beta_re_encode=True):
    """ Trim videos to extract clips
        Arguments:
            start: start timestamp
            end: end timestamp
            vid_path: source video path
            clip_name: clip name of the extracted clip
    """

    duration = end-start

    if beta_re_encode:
        # -c:v libx264 -crf 0 -c:a copy
        # string = f"ffmpeg -y -i {vid_path} -ss {str(start)} -to {str(end)} -qp 10 {clip_name}"
        string = f"ffmpeg -y -i {vid_path} -ss {str(start)} -t {str(duration)} -c:v libx264 -preset ultrafast -crf 23 -c:a copy {clip_name}"

    else:
        string = f"ffmpeg -y -ss {str(start)} -i {vid_path} -t {str(duration)} -c copy {clip_name}"

    os.system(string)


def beta_merge_clips(clips_list, lst_path, merged_path):
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


def beta_crop_clip(trim_path, crop_path, reframe):
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


def beta_style_sub(input_ass, fontname, fontsize, primarycolour, outlinecolour, borderstyle, marginv, outline, shadow, bold, spacing, verb_colour, object_colour):
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
    # styles[ass_format['OutlineColour']] = outlinecolour
    # styles[ass_format['BorderStyle']] = borderstyle
    styles[ass_format['MarginV']] = marginv
    styles[ass_format['Shadow']] = shadow
    styles[ass_format['Bold']] = bold
    styles[ass_format['Spacing']] = spacing
    # styles[ass_format['Outline']] = outline

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
            layer, start_time, end_time, style, actor, margin_l, margin_r, margin_v, effect, text = parts
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


def beta_beta_burn_subs_ffmpeg(clip_path, srt_path, fonts_dir, fontname, fontsize, primarycolour, outlinecolour, borderstyle, marginv, outline, shadow, bold, spacing, verb_colour, object_colour):
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

    beta_style_sub(ass_path, fontname, fontsize, primarycolour,
                   outlinecolour, borderstyle, marginv, outline, shadow, bold, spacing, verb_colour, object_colour)

    subs_cmd = f"ffmpeg -y -i {clip_path} -vf ass={ass_path}:fontsdir={fonts_dir} {subbed_clip_path}"
    os.system(subs_cmd)

    os.remove(clip_path)

    return subbed_clip_path


def beta_fit_vid(video, start, end, output_video_path, canvas_size="720p", orientation="portrait"):

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
            raise ValueError(
                "Invalid orientation. Must be 'landscape', 'portrait' or 'square'.")
    elif canvas_size == "1080p":
        if orientation == "landscape":
            target_width, target_height = 1920, 1080
        elif orientation == "portrait":
            target_width, target_height = 1080, 1920
        elif orientation == "square":
            target_width, target_height = 1920, 1920
        else:
            raise ValueError(
                "Invalid orientation. Must be 'landscape', 'portrait' or 'square'.")
    else:
        raise ValueError("Invalid canvas size. Must be '720p' or '1080p'.")

    vidcap = cv2.VideoCapture(vid_path)
    fps = int(vidcap.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    temp_output_video_path = output_video_path.replace('.mp4', '_temp.mp4')
    output_video = cv2.VideoWriter(
        temp_output_video_path, fourcc, fps, (target_width, target_height))

    frame_start, frame_end = int(start*fps), int(end*fps)
    for frame_id in range(frame_start, frame_end):
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

        resized_height, resized_width = int(
            target_width / original_aspect_ratio), target_width
        print("resize dims:", resized_height, resized_width)

        resized_frame = cv2.resize(frame, (resized_width, resized_height))

        crop_height, crop_width = target_height, target_width

        if crop_height > original_height:
            crop_height = original_height
            crop_width = crop_height*target_aspect_ratio

        if crop_width > original_width:
            crop_width = original_width
            crop_height = crop_width/target_aspect_ratio

        print("crop dims:", crop_height, crop_width)

        blurred_frame = frame[original_center[0]-int(crop_height/2):original_center[0]+int(
            crop_height/2), original_center[1]-int(crop_width/2):original_center[1]+int(crop_width/2)]
        blurred_frame = cv2.GaussianBlur(blurred_frame, (0, 0), 10)
        final_frame = cv2.resize(blurred_frame, (target_width, target_height))

        final_frame[int(target_center[0]-resized_height/2):int(target_center[0] + resized_height/2),
                    int(target_center[1]-resized_width/2):int(target_center[1] + resized_width/2)] = resized_frame

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


def beta_add_logo(vid_id, logo_path, output_video_path, position=(0, 0, 0, 0), start=0, end=None):

    video = Video.query.get(vid_id)
    if video is None:
        return
        
    vid_path = f"{video.base_path}{video.path}" 
    
    vidcap = cv2.VideoCapture(vid_path)

    fps = round(vidcap.get(cv2.CAP_PROP_FPS))
    width = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    logo = cv2.imread(logo_path, cv2.IMREAD_UNCHANGED)

    y, x, bottom, right = position
    logo_height = int((bottom - y) * height)
    logo_width = int((right - x) * width)

    y = int(y * height)
    x = int(x * width)

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


def beta_read_captions(srt_path):

    srt_text = open(srt_path).read()
    srt = SRTReader().read(srt_text, lang="en")
    captions = srt.get_captions("en")

    return captions


def beta_split_text_into_lines(text, font, max_width, uppercase):
    lines = []
    words = text.split()
    current_line = ""

    for word in words:
        if uppercase:
            word = word.upper()
        test_line = " ".join([current_line, word]).strip()
        try:
            test_width, _ = font.getsize(test_line)
        except:
            test_width = font.getbbox(test_line)[2]

        if test_width <= max_width:
            current_line = test_line
        else:
            lines.append(current_line.split())
            current_line = word

    if current_line:
        lines.append(current_line.split())

    return lines


def beta_burn_subs(vid_path, captions, output_path, word_timestamps, layout = "center", height_offset = 200, width_offset = 50, fontsize=36, word_spacing = 10, line_spacing = 50, uppercase =True, font_path=None, font_color=(255, 255, 255), spoken_word_color=None, verb_color=None, object_color=None, animation_type = None, shadow_color = None, shadow_offset = 2, outline_color = None, outline_width = 2, burn_emoji = False, precomp_emojis_path = None, emojis_folder = None):

    cap = cv2.VideoCapture(vid_path)
    frame_rate = int(cap.get(5))

    temp_output_path = output_path.replace('.mp4', '_temp.mp4')
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    output_video = cv2.VideoWriter(temp_output_path, fourcc, frame_rate, (int(cap.get(3)), int(cap.get(4))))

    if font_path:
        font = ImageFont.truetype(font_path, int(fontsize))
    else:
        font = ImageFont.truetype("/path/to/Montserrat-Black.ttf", int(fontsize))
                        
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
            for caption in captions:
                start_time = caption.get('start')
                end_time = caption.get('end')

                if start_time <= frame_timestamp <= end_time:
                    current_caption = caption
                    word_types = [token for token in nlp(caption.get('text'))]
                    break 

        if current_caption:
        
            if burn_emoji:        
                emoji_path, max_sim = get_emoji(clip_model, lm_model, device, precomp_emojis_path, emojis_folder, current_caption.get('text'))

            filtered_words = [word_info for word_info in word_timestamps if
                              current_caption.get('start') <= word_info[1] <= current_caption.get('end') and
                              current_caption.get('start') <= word_info[2] <= current_caption.get('end')]

            image = Image.fromarray(frame)
            image_width, image_height = image.size
            draw = ImageDraw.Draw(image)
            
            lines = beta_split_text_into_lines(current_caption.get('text'), font, image_width-2*width_offset, uppercase)
            
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
                        x, y, a, b = text_x-0.5*word_spacing, text_y - 0.1*line_spacing, text_x + text_size[2]+ 0.5*word_spacing, text_y + text_size[3]+ 0.1*line_spacing
                        draw.rounded_rectangle([x, y, a, b], radius=10, fill=spoken_word_color)   

                if shadow_color:
                    shadow_x = text_x + shadow_offset  
                    shadow_y = text_y + shadow_offset

                    draw.text((shadow_x, shadow_y), word_text, fill=shadow_color, font=font)

                if outline_color:
                    draw.text((text_x, text_y), word_text, fill=font_color_to_use, font=font, stroke_width = outline_width, stroke_fill = outline_color)       
                        
                else:                        
                    draw.text((text_x, text_y), word_text, fill=font_color_to_use, font=font)
                        
                text_x += text_size[2] + word_spacing
        
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
                              
        output_video.write(frame)

    temp_audio_path = output_path.replace('.mp4', '.aac')
    print('temp_audio_path', temp_audio_path)
    audio_cmd = f"ffmpeg -y -i {vid_path} -vn {temp_audio_path}"
    print('audio_cmd', audio_cmd)
    os.system(audio_cmd)
                        
    cap.release()
    output_video.release()

    merge_cmd = f"ffmpeg -y -i {temp_output_path} -i {temp_audio_path} -c:v copy -c:a aac {output_path}"
    os.system(merge_cmd)

    os.remove(temp_audio_path)
    os.remove(temp_output_path)
    

############################## SHOT SPLITTER ##################################################################################


def beta_shot_resizer(video, output_video_path, shots, resolution="1080p", orientation="portrait"):

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
    os.system(audio_cmd)

    vidcap.release()
    output_video.release()

    merge_cmd = f"ffmpeg -y -i {temp_output_video_path} -i {temp_audio_path} -c:v copy -c:a aac {output_video_path}"
    os.system(merge_cmd)

    os.remove(temp_audio_path)
    os.remove(temp_output_video_path)


def beta_ai_shot_splitter(video):

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


def beta_crop_face(original_shape, x, y, w, h, num_faces, aspect_ratio = 9/16, padding_ratio = 100):

    aspect_ratio = aspect_ratio*num_faces
    center_x, center_y = int(x+w/2), int(y+h/2)

    orig_h, orig_w = original_shape

    crop_height = int(h*padding_ratio)
    crop_width = int(crop_height * aspect_ratio)

    if crop_height > orig_h:
        crop_height = orig_h
        crop_width = int(crop_height * aspect_ratio)

    if crop_width > orig_w:
        crop_width = orig_w
        crop_height = int(crop_width / aspect_ratio)

    #print("desired crop", crop_height, crop_width)
    
    delta_w = int(crop_width/2)

    delta_h = int(crop_height/2)

    w1 = max(0, center_x - delta_w)
    w2 = min(orig_w, w1 + crop_width)

    if w1 == 0:
        w2 = crop_width

    if w2 == orig_w:
        w1 = orig_w - crop_width
        
    h1 = max(0, center_y - delta_h)
    h2 = min(orig_h, h1 + crop_height)
    
    if h1 == 0:
        h2 = crop_height

    if h2 == orig_h:
        h1 = orig_h - crop_height
    
    #print("actual crop", h2-h1, w2-w1)

    return h1, h2, w1, w2


def beta_faces_extractor(video, start, end, active_speakers_only=True, speaker_window=4, speaker_threshold=0.3, autobroll=False, orientation="portrait"):
    if video is None:
        return
    vid_path = f"{video.base_path}{video.path}"

    vidcap = cv2.VideoCapture(vid_path)
    original_shape = (int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                      int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH)))
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    vidcap.set(cv2.CAP_PROP_POS_MSEC, start * 1000)
    frame_start = int(vidcap.get(cv2.CAP_PROP_POS_FRAMES))
    vidcap.set(cv2.CAP_PROP_POS_MSEC, end * 1000)
    frame_end = int(vidcap.get(cv2.CAP_PROP_POS_FRAMES))
    vidcap.release()
    
    tracked_faces_path = f"{video.base_path}{video.tracked_faces_path}"
    speakers_path = f"{video.base_path}{video.speakers_path}"

    if orientation == "landscape":
        canvas_width, canvas_height = original_shape[1], original_shape[0]
    elif orientation == "portrait":
        canvas_width, canvas_height = original_shape[0]*9/16, original_shape[0]
    elif orientation == "square":
        canvas_width, canvas_height = original_shape[0], original_shape[0]
    else:
        raise ValueError(
            "Invalid orientation. Must be 'landscape', 'portrait' or 'square'.")

    with open(tracked_faces_path, 'rb') as file:
        tracked_faces = pickle.load(file)

    with open(speakers_path, 'rb') as file:
        speakers = pickle.load(file)

    shots = [i for i in range(1, len(tracked_faces)) if len(
        tracked_faces[i]) != len(tracked_faces[i-1])]
    shots.append(len(tracked_faces))

    #print('\033[37m' + 'detected shots:' + '\033[0m', shots)

    speaker_history = []

    faces_list = []

    for frame_id in range(frame_start, frame_end):
        #print('\033[31m' + 'processing frame:' + '\033[0m', frame_id-frame_start)
        frame_timestamp = (frame_id/fps)*1000
        
        faces = tracked_faces[frame_id]
        num_faces = len(faces)

        if active_speakers_only and num_faces > 1:
            updated_faces = []
            speaking_counts = []

            for i, face in enumerate(faces):
                x, y, w_f, h_f, face_id, _ = face

                window = int(fps*speaker_window)
                next_shot = next((index for index in shots if index > frame_id), None)

                lim_sup = min(frame_id + window, next_shot)

                length = lim_sup-frame_id + 1

                if length < window:
                    delta = window-length
                    lim_inf = frame_id - delta
                    lim_sup = lim_inf + window-1
                    speaking_sum = sum(1 for frame_num in range(lim_inf, lim_sup) if face_id in speakers[frame_num])
                    length = lim_sup-lim_inf
                else:
                    speaking_sum = sum(1 for frame_num in range(frame_id, lim_sup) if face_id in speakers[frame_num])

                speaking_percentage = speaking_sum/(length)

                #print("speaking percentage:", speaking_percentage)

                speaking_counts.append([face, speaking_sum])

            updated_faces = [max(speaking_counts, key=lambda item: item[1])[0]]
            updated_faces = [[face[0], face[1], face[2], face[3], index] for index, face in enumerate(updated_faces)]

            faces = updated_faces

        faces = [[face[0], face[1], face[2], face[3], index] for index, face in enumerate(faces)]
        num_faces = len(faces)

        frame_faces_list = []
        if num_faces > 0:
            #print('\033[34m' + 'Number of detected faces:' +'\033[0m', num_faces)

            if autobroll:
                num_faces += 1

            for i, (x, y, w_f, h_f, face_id) in enumerate(faces):
                #print('\033[32m' + 'processing face:' + '\033[0m', i, ', params:', x, y, w_f, h_f, face_id)

                if orientation == "portrait":
                    h1, h2, w1, w2 = beta_crop_face(original_shape, x, y, w_f, h_f, num_faces, 9/16)                   

                elif orientation == "square":
                    h = canvas_height//num_faces
                    h1, h2, w1, w2 = beta_crop_face(original_shape, x, y, w_f, h_f, num_faces, 1)
                    
                else:
                    w = canvas_width//num_faces
                    h1, h2, w1, w2 = beta_crop_face(original_shape, x, y, w_f, h_f, num_faces, 16/9)

                frame_faces_list.append([h1, h2, w1, w2, face_id, frame_id, frame_timestamp])
        else:
            if orientation == "portrait":
       
               desired_height = original_shape[0]
               desired_width = int(original_shape[0]*9/16)
               h1 = 0
               h2 = desired_height
               w1 = int(original_shape[1]/2 - desired_width/2) 
               w2 = w1 + desired_width
               frame_faces_list = [[h1, h2, w1, w2, -1, frame_id, frame_timestamp]]
               
            elif orientation == "square":
                   
               desired_height = original_shape[0]
               desired_width = original_shape[0]
               h1 = 0
               h2 = desired_height
               w1 = int(original_shape[1]/2 - desired_width/2) 
               w2 = w1 + desired_width
               frame_faces_list = [[h1, h2, w1, w2, -1, frame_id, frame_timestamp]]
               
        faces_list.append(frame_faces_list)
       
    return faces_list


def beta_resize_vid(video, faces_list, start, end, output_video_path, orientation="portrait", autobroll=False, precomp_stock_vids_path=None):
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
        captions = beta_read_captions(srt_path)

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
                    caption_embedding /= caption_embedding.norm(
                        dim=-1, keepdim=True)
                    current_caption_embedding = caption_embedding.T.cpu().numpy().squeeze()

                current_index = i

                sim_list = [100.0 * np.mean(np.load(vid_embedding_path[1], allow_pickle=True)
                                            @ current_caption_embedding) for vid_embedding_path in stock_vids_embeddings]
                sim_list = [[sim, index] for index, sim in enumerate(sim_list)]
                sim_list = sorted(sim_list, key=lambda x: x[0], reverse=True)

                matching_vid_path = stock_vids_embeddings[sim_list[0][1]][0]
                stock_vid = cv2.VideoCapture(matching_vid_path)
                print(matching_vid_path)
                stock_vid_index = 0
                break

    vidcap = cv2.VideoCapture(vid_path)
    original_shape = (int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                      int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH)))

    if orientation == "landscape":
        canvas_width, canvas_height = original_shape[1], int(
            original_shape[1]*9/16)
    elif orientation == "portrait":
        canvas_width, canvas_height = original_shape[0], int(
            original_shape[0]*16/9)
    elif orientation == "square":
        canvas_width, canvas_height = original_shape[1], original_shape[1]
    else:
        raise ValueError(
            "Invalid orientation. Must be 'landscape', 'portrait' or 'square'.")

    fps = video.fps
    temp_output_video_path = output_video_path.replace('.mp4', '_temp.mp4')
    output_video = cv2.VideoWriter(temp_output_video_path, cv2.VideoWriter_fourcc(
        *"mp4v"), fps, (canvas_width, canvas_height))

    frame_start, frame_end = int(start*fps), int(end*fps)

    faces_list_id = 0
    for frame_id in range(frame_start, frame_end):
        print('\033[31m' + 'processing frame:' + '\033[0m', frame_id)
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = vidcap.read()

        faces = faces_list[faces_list_id]
        num_faces = len(faces)

        if autobroll:
            num_faces += 1
            if frame_id/fps > caption_end:
                current_index += 1
                current_caption = captions[current_index]
                caption_start = current_caption.start/1000000
                caption_end = current_caption.end/1000000
                caption_text = current_caption.get_text()

                with torch.no_grad():
                    caption_text = clip.tokenize(caption_text).to(device)
                    caption_embedding = clip_model.encode_text(caption_text)
                    caption_embedding /= caption_embedding.norm(
                        dim=-1, keepdim=True)
                    current_caption_embedding = caption_embedding.T.cpu().numpy().squeeze()

                sim_list = [100.0 * np.mean(np.load(vid_embedding_path[1], allow_pickle=True)
                                            @ current_caption_embedding) for vid_embedding_path in stock_vids_embeddings]

                sim_list = [[sim, index] for index, sim in enumerate(sim_list)]
                sim_list = sorted(sim_list, key=lambda x: x[0], reverse=True)

                matching_vid_path = stock_vids_embeddings[sim_list[0][1]][0]
                print(matching_vid_path)
                stock_vid = cv2.VideoCapture(matching_vid_path)
                stock_vid_index = 0
            else:
                stock_vid_index += 1

            stock_vid.set(cv2.CAP_PROP_POS_FRAMES, stock_vid_index)
            ret, stock_frame = stock_vid.read()

        if num_faces > 0:
            print('\033[34m' + 'Number of detected faces:' +
                  '\033[0m', num_faces)
            canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

            for i, (h1, h2, w1, w2, face_id) in enumerate(faces):
                print('\033[32m' + 'processing face:' + '\033[0m',
                      i, ', params:', h1, h2, w1, w2, face_id)

                if orientation == "portrait":
                    h = canvas_height//num_faces
                    desired_width, desired_height = canvas_width, h

                    face = frame[h1:h2, w1:w2]
                    face = cv2.resize(face, (desired_width, desired_height))

                    offset = face_id*h
                    canvas[offset:offset+h, :] = face

                elif orientation == "landscape":
                    w = canvas_width//num_faces
                    desired_width, desired_height = w, canvas_height

                    face = frame[h1:h2, w1:w2]
                    face = cv2.resize(face, (desired_width, desired_height))

                    offset = face_id*w
                    canvas[:, offset:offset+w] = face

                else:
                    h = canvas_height//num_faces
                    desired_width, desired_height = canvas_width, h

                    face = frame[h1:h2, w1:w2]
                    face = cv2.resize(face, (desired_width, desired_height))

                    offset = face_id*h
                    canvas[offset:offset+h, :] = face

            if autobroll:
                print(desired_width, desired_height)
                width, height, _ = stock_frame.shape
                w1 = int(width/2-desired_width/2)
                w2 = w1 + desired_width
                h1 = int(height/2-desired_height/2)
                h2 = h1 + desired_height

                print(h1, h2, w1, w2)
                cropped_frame = stock_frame[h1:h2, w1:w2]

                cropped_frame = cv2.resize(
                    cropped_frame, (desired_width, desired_height))

                print(canvas.shape, cropped_frame.shape, offset+h)
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
                canvas = frame[h/2 - int(frame_height/2)                               :int(h/2) + int(frame_height/2), :]

            output_video.write(canvas)

        faces_list_id += 1

    temp_audio_path = output_video_path.replace('.mp4', '.aac')
    audio_cmd = f"ffmpeg -y -loglevel error -hide_banner -i {vid_path} -ss {start} -to {end} -vn {temp_audio_path}"
    os.system(audio_cmd)

    vidcap.release()
    output_video.release()

    merge_cmd = f"ffmpeg -y -loglevel error -hide_banner -i {temp_output_video_path} -i {temp_audio_path} -c:v copy -c:a aac {output_video_path}"
    os.system(merge_cmd)

    os.remove(temp_audio_path)
    os.remove(temp_output_video_path)


def beta_tedx_tracker(video, precomp_faces_path, output_video_path, start, end, orientation="portrait"):
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
    output_video = cv2.VideoWriter(
        temp_output_video_path, fourcc, fps, (target_width, target_height))

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

    shots = [(int(shot*fps) - frame_start)
             for shot in shots if shot*fps > frame_start and shot*fps < frame_end]

    if shots[0] > 0:
        shots.insert(0, 0)
    if shots[-1] < (frame_end-frame_start):
        shots.append(frame_end-frame_start)

    x_smoothed = []

    for i in range(len(shots) - 1):
        start_idx = shots[i]
        end_idx = shots[i + 1]

        x_slice = x_positions[start_idx:end_idx]

        window = min(101, len(x_slice))
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
        cropped_frame = frame[:, x_anchor -
                              int(target_width/2): x_anchor+int(target_width/2)]

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


def beta_get_emoji(precomp_emojis_path, emojis_folder, description):

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
        embedding /= embedding.norm(dim=-1, keepdim=True)
        txt_embedding_clip = embedding.T.cpu().numpy().squeeze()

    txt_embedding_lm = lm_model.encode(description).squeeze()

    for i in range(len(precomp_emojis)):
        emoji_embedding_txt = precomp_emojis[i][3]
        if emoji_embedding_txt is not None:
            emoji_embedding_txt = emoji_embedding_txt.squeeze()
        emoji_embedding_viz = precomp_emojis[i][4].squeeze()

        if emoji_embedding_txt is not None:
            similarity_txt = 100.0 * emoji_embedding_txt @ txt_embedding_lm
        else:
            similarity_txt = 100.0 * emoji_embedding_viz @ txt_embedding_clip

        similarity_viz = 100.0 * emoji_embedding_viz @ txt_embedding_clip

        similarity = similarity_viz+similarity_txt
        sim_list.append([similarity, precomp_emojis[i]
                        [0], precomp_emojis[i][2]])

    max_sim = max(sim_list, key=lambda x: x[0])
    sim = max_sim[0]
    result_path = emojis_folder + max_sim[1]

    return result_path, max_sim


############################## AUTO TRACKER ##################################################################################

def beta_frame2frame_sims(video_features):
    """ Computes frame to frame successive similarities using the precomp_vide video features
        Arguments:
            video_features: precomp_vid array
    """
    similarities = []
    prev_frame_features = video_features[0]
    for i in range(1, len(video_features), 1):
        frame_features = video_features[i]
        similarity = 100.0 * frame_features @ prev_frame_features.T
        similarities.append([i, similarity])
        prev_frame_features = frame_features
    return similarities


def beta_expand2square(pil_img, background_color):
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


def beta_get_heatmap(img, query_features, model, preprocess, device, vid_shape, batch_size=1, start_layer=-1):

    img = beta_expand2square(img, (0, 0, 0))
    img = preprocess(img).unsqueeze(0).to(device)
    img = img.repeat(batch_size, 1, 1, 1)
    image_features = model.encode_image(img)

    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    query_features = query_features / query_features.norm(dim=-1, keepdim=True)

    batch_size = image_features.shape[0]
    logit_scale = model.logit_scale.exp()
    logits_per_image = logit_scale * image_features @ query_features.t()
    probs = logits_per_image.softmax(dim=-1).detach().cpu().numpy()

    index = [i for i in range(batch_size)]
    one_hot = np.zeros(
        (logits_per_image.shape[0], logits_per_image.shape[1]), dtype=np.float32)
    one_hot[torch.arange(logits_per_image.shape[0]), index] = 1
    one_hot = torch.from_numpy(one_hot).requires_grad_(True)
    one_hot = torch.sum(one_hot.cuda() * logits_per_image)

    model.zero_grad()

    image_attn_blocks = list(
        dict(model.visual.transformer.resblocks.named_children()).values())

    if start_layer == -1:
        start_layer = len(image_attn_blocks) - 1

    num_tokens = image_attn_blocks[0].attn_probs.shape[-1]
    R = torch.eye(num_tokens, num_tokens,
                  dtype=image_attn_blocks[0].attn_probs.dtype).to(device)
    R = R.unsqueeze(0).expand(batch_size, num_tokens, num_tokens)

    for i, blk in enumerate(image_attn_blocks):
        if i < start_layer:
            continue
        grad = torch.autograd.grad(
            one_hot, [blk.attn_probs], retain_graph=True)[0].detach()
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
    image_relevance = torch.nn.functional.interpolate(
        image_relevance, size=224, mode='bilinear')
    image_relevance = image_relevance.reshape(
        224, 224).cuda().data.cpu().numpy()
    image_relevance = (image_relevance - image_relevance.min()) / \
        (image_relevance.max() - image_relevance.min())

    heatmap = np.uint8(255 * image_relevance)
    heatmap = np.float32(heatmap) / 255
    heatmap = cv2.resize(heatmap, (vid_shape[0], vid_shape[0]))

    return heatmap


def bets_get_focal(heatmap, vid_shape, ratio=9/16, vertical_split=False, nbr_splits=3, step_size=20):

    ratio_width = int(vid_shape[1]*ratio)
    region_width = int(ratio_width/nbr_splits)
    region_height = int(vid_shape[1] / nbr_splits)

    region_heats = []

    for focal in range(int(region_width/2), vid_shape[0]-int(region_width/2), step_size):

        x1 = focal-int(region_width/2)
        x2 = x1 + region_width

        if vertical_split:
            y_pics = []
            for y in range(0, vid_shape[1]-region_height, step_size):
                y_pics.append(np.sum(heatmap[y:y+region_height, x1:x2]))
            region_heats.append([focal, max(y_pics)])
        else:
            region_heats.append([focal, np.sum(heatmap[:, x1:x2])])

    focal = max(region_heats, key=itemgetter(1))[0]

    return focal


def beta_auto_resizer_beta_get_heatmaps(video, query, query_type, start, end, stride_factor):
    if video is None:
        return
    vid_path = f"{video.base_path}{video.path}"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip_grad.load("ViT-B/32", device=device, jit=False)

    query = [query]

    if query_type == 'text':
        query = clip_grad.tokenize(query).to(device)
        query_features = model.encode_text(query)
    else:
        query = preprocess(Image.open(query)).unsqueeze(0).to(device)
        query_features = model.encode_image(query)

    vidcap = cv2.VideoCapture(vid_path)

    fps = round(vidcap.get(cv2.CAP_PROP_FPS))
    length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    vid_shape = [int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                 int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))]

    heatmaps = []

    stride = int(stride_factor*fps)

    for i in range(start, end, stride):
        print("Computing heatmap of frame: ", i)

        vidcap.set(cv2.CAP_PROP_POS_FRAMES, i)
        _, frame = vidcap.read()

        heatmap = beta_get_heatmap(Image.fromarray(
            frame), query_features, model, preprocess, device, vid_shape)

        if start + len(heatmaps) + stride <= end:
            for j in range(stride):
                heatmaps.append(heatmap)

        else:
            prev_len = len(heatmaps)
            for j in range(end-start-prev_len):
                print("Computing remaining heatmap of frame: ", prev_len+j)
                heatmaps.append(heatmap)
                
    vidcap.release()

    return heatmaps, vid_shape


def beta_auto_resizer_get_focals(heatmaps, vid_shape, ratio):
    focals = []
    for i in range(len(heatmaps)):
        print("Computing focal of frame: ", i)

        heatmap = heatmaps[i]

        focal = get_focal(heatmap, vid_shape, ratio)
        focals.append(focal)

    focals = np.array(focals)

    return focals


def beta_get_croping_positions(vid_shape, focal, ratio):

    ratio_width = int(vid_shape[1]*ratio)

    x1 = int(max(focal-int(ratio_width/2), 0))
    x2 = x1 + ratio_width

    if x2 > vid_shape[0]:
        x2 = vid_shape[0]
        x1 = x2 - ratio_width
    return x1, x2


def beta_auto_resizer_get_positions(video, focals, start, end, ratio, mode="shot_splitter", smoothness=11, sim_threshold=85, transition_factor=0.1):
    if video is None:
        return
    vid_path = f"{video.base_path}{video.path}"

    precomp_vid_path = f"{video.base_path}{video.precomp_vid_path}"
    vid_embeddings = np.load(precomp_vid_path, allow_pickle=True)

    vidcap = cv2.VideoCapture(vid_path)
    length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = round(vidcap.get(cv2.CAP_PROP_FPS))
    vid_shape = [int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                 int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))]

    cropping_positions = []

    focals = signal.savgol_filter(focals, smoothness, 3)

    if mode == "smooth_follow":
        for i in range(len(focals)):
            focal = focals[i]
            cropping_position = beta_get_croping_positions(
                vid_shape, focal, ratio)
            cropping_positions.append(cropping_position)

    elif mode == "shot_splitter":
        similarities = beta_frame2frame_sims(vid_embeddings)
        scene_changes = [(sim[0]*fps-start) for sim in similarities if (sim[1]
                                                                        < sim_threshold and sim[0]*fps > start and sim[0]*fps < end)]

        focals_scenes = []
        scene = 0

        focal_scene = np.mean(focals[:scene_changes[scene]])

        for i in range(len(focals)):
            if scene < len(scene_changes) - 1:
                scene_length = scene_changes[scene + 1] - scene_changes[scene]
                transition = int(transition_factor * scene_length)

                if scene != 0 and i >= scene_changes[scene] and i < scene_changes[scene] + transition:
                    t = (scene_changes[scene] + transition - i) / transition
                    prev_avg = np.mean(
                        focals[scene_changes[scene-1]:scene_changes[scene]])
                    next_avg = np.mean(
                        focals[scene_changes[scene]:scene_changes[scene + 1]])
                    focal_scene = (1 - t) * next_avg + t * prev_avg
                elif i == scene_changes[scene] + transition:
                    focal_scene = np.mean(
                        focals[scene_changes[scene]:scene_changes[scene + 1]])
                    scene += 1
                else:
                    pass
            else:
                focal_scene = np.mean(focals[scene_changes[scene]:])

            focals_scenes.append(focal_scene)

        focals = [int(focal) for focal in focals_scenes]

        for i in range(len(focals)):
            focal = focals[i]
            cropping_position = beta_get_croping_positions(
                vid_shape, focal, ratio)
            cropping_positions.append(cropping_position)

    vidcap.release()
    
    return cropping_positions


def beta_auto_resizer_create_clip(video, cropping_positions, start, end, output_video_path, ratio):
    if video is None:
        return
    vid_path = f"{video.base_path}{video.path}"

    vidcap = cv2.VideoCapture(vid_path)

    fps = int(vidcap.get(cv2.CAP_PROP_FPS))
    width = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    temp_output_video_path = output_video_path.replace('.mp4', '_temp.mp4')
    output_video = cv2.VideoWriter(
        temp_output_video_path, fourcc, fps, (int(height*ratio), height))

    vidcap.set(cv2.CAP_PROP_POS_FRAMES, start)

    for frame_id in range(len(cropping_positions)):
        print('processing frame: ', frame_id + start)

        ret, frame = vidcap.read()

        x1, x2 = cropping_positions[frame_id]
        cropped_frame = frame[:, x1:x2]

        output_video.write(cropped_frame)

    temp_audio_path = output_video_path.replace('.mp4', '.aac')
    audio_cmd = f"ffmpeg -y -i {vid_path} -ss {start/fps} -to {end/fps} -vn {temp_audio_path}"
    os.system(audio_cmd)

    vidcap.release()
    output_video.release()

    merge_cmd = f"ffmpeg -y -i {temp_output_video_path} -i {temp_audio_path} -c:v copy -c:a aac {output_video_path}"
    os.system(merge_cmd)

    os.remove(temp_audio_path)
    os.remove(temp_output_video_path)

    
############################## KEYFRAME TRACKER ##################################################################################   


def beta_interpolate_shot_positions(video, keyframes, shot_start, shot_end):
    if video is None:
        return
    vid_path = f"{video.base_path}{video.path}"    
    cap = cv2.VideoCapture(vid_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
        
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
        #cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        #frame_in_sec = cap.get(cv2.CAP_PROP_POS_MSEC)/1000  
        frame_in_sec = frame_num/fps     
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
            #cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            #frame_in_sec = cap.get(cv2.CAP_PROP_POS_MSEC)/1000    
            frame_in_sec = frame_num/fps       
            interpolated_positions.append({"reframe": {"leftOffset": interpolated_position[0], "topOffset": interpolated_position[1],
                                           "width": interpolated_size[0], "height": interpolated_size[1]}, "frame_number": frame_num, "frame_in_sec": frame_in_sec})

    last_keyframe_number = frame_numbers[-1]
    last_keyframe_reframe = keyframes[-1]["reframe"]  
    for frame_num in range(last_keyframe_number, shot_end_frame):
        #cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        #frame_in_sec = cap.get(cv2.CAP_PROP_POS_MSEC)/1000    
        frame_in_sec = frame_num/fps                 
        interpolated_positions.append({"reframe": last_keyframe_reframe, "frame_number": frame_num, "frame_in_sec": frame_in_sec})
        
    cap.release()
        
    return interpolated_positions

def beta_keyframe_tracker(video, clip_start, clip_end, output_width, output_height, output_video_path, interpolated_positions):
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
