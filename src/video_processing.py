import cv2
from app import db
from src.models.video import Video
from ffmpeg_streaming import Formats, Bitrate, Representation, Size
from src.enums.upload_processing_status import UploadProcessingStatus
from src.utils import utils
from src.config import Config

import ffmpeg_streaming
import os
import subprocess
import time
import torch
import time
import pickle
import subprocess
from moviepy.editor import VideoFileClip
import boto3

# please do not remove this line
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using:", device)


def re_enc(vid_id, method = "awsmediaconvert"):
    
    if method == "ffmpeg":
        print("Transcoding video ID: ", vid_id, " with FFMPEG")
        re_enc_ffmpeg(vid_id)
        
    else:
        print("Transcoding video ID: ", vid_id, " with AWS Elemental mediaconvert")    
        video = Video.query.get(vid_id)
        vid_size = video.file_size
        if vid_size > 0.5:
            try:
                print("Trying accelerated transcoding")
                re_enc_awsmediaconvert(vid_id, enable_acceleration = True)
            except:
                print("Accelerated transcoding failed, reverting to regular transcoding")
                re_enc_awsmediaconvert(vid_id)
        else:
            re_enc_awsmediaconvert(vid_id)
        
def re_enc_ffmpeg(vid_id):
    """ Re-encode videos
        Arguments:
            vid_path: source video path
    """
    video = Video.query.get(vid_id)
    
    if video is None:
        return
    
    last_reported = 0

    vid_path = f"{video.base_path}{video.path}"
    out_path = vid_path.replace('.mp4', '_reenc.mp4')

    start_time = time.time()
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


def wait_for_mediaconvert_job_completion(mediaconvert, job_id, max_retries=5):
    retries = 0

    while True:
        try:
            response = mediaconvert.get_job(Id=job_id)
            job_status = response['Job']['Status']

            if job_status in ['COMPLETE', 'ERROR']:
                print("Job ended with status:", job_status)
                break

        except Exception as e:
            retries += 1
            if retries > max_retries:
                print("Max retries reached. Unable to get job status.")
                break

            print(f"Retrying in 60 seconds due to rate limiting")
            time.sleep(60)
            continue

        time.sleep(60)
        

def download_aws_mediaconvert_file(filename, output_file_path, aws_access_key_id, aws_secret_access_key, bucket_name):

    s3 = boto3.client('s3', aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key)
    
    input_file_url = f'input/{filename}'
    output_file_url = f'output/{filename}'
        
    s3.delete_object(Bucket=bucket_name, Key=input_file_url)
                    
    s3.download_file(bucket_name, output_file_url, output_file_path)

    s3.delete_object(Bucket=bucket_name, Key=output_file_url)
    
    
def upload_to_s3(local_file, s3_name, aws_access_key_id, aws_secret_access_key, bucket_name):
    s3 = boto3.client('s3', aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key)
    
    s3.upload_file(local_file, bucket_name, s3_name)
    

def re_enc_awsmediaconvert(vid_id, target_bitrate = 5000000, enable_acceleration = False, output_extension = '.mp4'):
    """ Re-encode videos
        Arguments:
            vid_path: source video path
    """
    
    start = time.time()
    
    video = Video.query.get(vid_id)    
    if video is None:
        return

    s3_vid_name =  f'{time.time()}_{video.path.split("/")[-1]}' 
    vid_path = f"{video.base_path}{video.path}"
    title, extension = os.path.splitext(vid_path)
    video.original_extension = extension
    out_path = vid_path.replace(extension, f'_reenc.mp4')
    width, height = video.width, video.height

    s3_vid_path = f'input/{s3_vid_name}'    
    upload_to_s3(vid_path, s3_vid_path, Config.mediaconvert_access_key, Config.mediaconvert_secret_key, Config.mediaconvert_bucket_name)
    s3_input_url = f's3://{Config.mediaconvert_bucket_name}/{s3_vid_path}'
    
    mediaconvert = boto3.client('mediaconvert',
                                aws_access_key_id=Config.mediaconvert_access_key,
                                aws_secret_access_key=Config.mediaconvert_secret_key, 
                                region_name = Config.mediaconvert_region_name)
    
    output_folder_url = f's3://{Config.mediaconvert_bucket_name}/output/'
    
    job_settings = {
        "Role":Config.mediaconvert_arn_role,   
        "Settings": {        
            "OutputGroups": [
                {
                    "Name": "File Group",
                    "Outputs": [
                        {
                            "ContainerSettings": {
                                "Container": "MP4",
                                "Mp4Settings": {
                                    "MoovPlacement": "NORMAL"
                                }
                            },
                            "VideoDescription": {
                                "Width": width,
                                "Height": height,
                                "CodecSettings": {
                                    "Codec": "H_264",
                                    "H264Settings": {
                                        "Bitrate": target_bitrate,
                                        "FramerateControl": "INITIALIZE_FROM_SOURCE",
                                        "RateControlMode": "CBR",
                                        "GopSize": 48,
                                        "NumberReferenceFrames": 3,
                                        "Syntax": "DEFAULT",
                                        "SceneChangeDetect": "ENABLED",
                                        "QualityTuningLevel": "SINGLE_PASS",
                                        "FramerateConversionAlgorithm": "DUPLICATE_DROP"
                                    }
                                },
                            },
                            "AudioDescriptions": [
                                {
                                    "AudioTypeControl": "FOLLOW_INPUT",
                                    "CodecSettings": {
                                        "Codec": "AAC",
                                        "AacSettings": {
                                            "Bitrate": 160000,
                                            "RateControlMode": "CBR",
                                            "CodecProfile": "LC",
                                            "CodingMode": "CODING_MODE_2_0",
                                            "SampleRate": 48000,
                                            "Specification": "MPEG4"
                                        }
                                    },
                                    "LanguageCodeControl": "FOLLOW_INPUT",
                                    "AudioType": 0
                                }
                            ],
                            "Extension": "mp4",
                        }
                    ],
                    "OutputGroupSettings": {
                        "Type": "FILE_GROUP_SETTINGS",
                        "FileGroupSettings": {
                            "Destination": output_folder_url
                        }
                    }
                }
            ],
            "Inputs": [
                {
                    "AudioSelectors": {
                        "Audio Selector 1": {
                            "Offset": 0,
                            "DefaultSelection": "DEFAULT",
                            "ProgramSelection": 1
                        }
                    },
                    "VideoSelector": {
                        "ColorSpace": "FOLLOW"
                    },
                    "FilterEnable": "AUTO",
                    "PsiControl": "USE_PSI",
                    "FilterStrength": 0,
                    "DeblockFilter": "DISABLED",
                    "DenoiseFilter": "DISABLED",
                    "TimecodeSource": "ZEROBASED",
                    "FileInput": s3_input_url
                }
            ]
        }
    }
    
    if enable_acceleration:
        job_settings["AccelerationSettings"] = {
            "Mode": "ENABLED"
        }
        job_settings["UserMetadata"] = {
            "job": "Acceleration"
        }
        job_settings["Settings"]["TimecodeConfig"] = {
            "Source": "ZEROBASED"
        }
                    
    response = mediaconvert.create_job(**job_settings)
    job_id = response['Job']['Id']
    
    print(f'MediaConvert Job ID: {job_id}')
    
    wait_for_mediaconvert_job_completion(mediaconvert, job_id)
    
    s3_vid_name = s3_vid_name.replace(extension, output_extension)
        
    download_aws_mediaconvert_file(s3_vid_name, out_path, Config.mediaconvert_access_key, Config.mediaconvert_secret_key, Config.mediaconvert_bucket_name)

    end = time.time()
    
    print("Transcoding lasted:", end-start)        
    print(f'Downloaded transcoded file: {out_path}')

    os.remove(vid_path)
    vid_path = vid_path.replace(extension, output_extension)    
    os.rename(out_path, vid_path)
    
    video.path = video.path.replace(extension, output_extension)
    video.file_name = video.file_name.replace(extension, output_extension)
        
    video.rencoded = 1
    db.session.commit()
        
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
        update_bandwidth_thresholds(hls_path)
    except:
        print("Failed to update HLS bandwith thresholds")
        

def vid_out(start, end, vid_path, clip_name, re_encode=True, use_frames = False, crf = 23):
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
        string = f"ffmpeg -y -hide_banner -loglevel error -i {vid_path} -ss {str(start)} -t {str(duration)} -c:v libx264 -preset ultrafast -crf {str(crf)} -c:a copy {clip_name}"

    elif use_frames:
        string =   f'ffmpeg -i {vid_path} -vf "select=between(n\,{start}\,{end}),setpts=PTS-STARTPTS" -c:v libx264 -crf {str(crf)} -preset ultrafast -map 0:v {clip_name}'
    else:
        string = f"ffmpeg -y -hide_banner -loglevel error -ss {str(start)} -i {vid_path} -t {str(duration)} -c copy {clip_name}"

    os.system(string)
    
    
def vid_out_moviepy(start, end, vid_path, output_path):
    """ Trim videos to extract clips
        Arguments:
            start: start timestamp
            end: end timestamp
            vid_path: source video path
            clip_name: clip name of the extracted clip
    """

    video = VideoFileClip(vid_path)

    clip = video.subclip(t_start = start, t_end = end)
    clip.write_videofile(output_path, audio_codec='aac')
    

def vid_out_opencv(start, end, vid_path, output_path):
    """ Trim videos to extract clips
        Arguments:
            start: start timestamp
            end: end timestamp
            vid_path: source video path
            clip_name: clip name of the extracted clip
    """
    subprocess.run(["app/api/video_cut", vid_path, output_path, str(start), str(end)])
    

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


def crop_clip(trim_path, crop_path, reframe, output_resolution):
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
    
    if width>1:
       width = 1
    if height>1:
       height = 1
       
    string = f"ffmpeg -y -hide_banner -loglevel error -i {trim_path} -filter:v 'crop=in_w*{width}:in_h*{height}:in_w*{corner_x}:in_h*{corner_y},scale={output_resolution}' -c:a copy {crop_path}"
    os.system(string)


# ASS styling format
ass_format = {'Fontname': 1, 'Fontsize': 2, 'PrimaryColour': 3, 'SecondaryColour': 4, 'OutlineColour': 5, 'BackColour': 6,
              'Bold': 7, 'Italic': 8, 'Underline': 9, 'StrikeOut': 10, 'ScaleX': 11, 'ScaleY': 12, 'Spacing': 13, 'Angle': 14,
              'BorderStyle': 15, 'Outline': 16, 'Shadow': 17, 'Alignment': 18, 'MarginL': 19, 'MarginR': 20, 'MarginV': 21, 'Encoding': 22}


def process_shot_cv2(vid_path, output_video_path, start_frame, end_frame, corner_x, corner_y, width, height, output_width, output_height, fps, shot_id):
    start = time.time()    
    
    vidcap = cv2.VideoCapture(vid_path)
    orig_height, orig_width = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))

    corner_x = int(corner_x * orig_width)
    corner_y = int(corner_y * orig_height)
    width = int(width * orig_width)
    height = int(height* orig_height) 
        
    temp_output_video_path = output_video_path.replace('.mp4', f'_shot_{shot_id}.mp4')
    output_video = cv2.VideoWriter(temp_output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (output_width, output_height))   
    
    for i in range(start_frame, end_frame):
        if i%(fps*5)==0:
           print(f"shot {shot_id} - cropping frame:", i)
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, i)
        success, frame = vidcap.read()

        cropped_frame = frame[corner_y: corner_y + height, corner_x: corner_x + width]
        cropped_frame = cv2.resize(cropped_frame, (output_width, output_height))
        output_video.write(cropped_frame)

    vidcap.release()
    output_video.release()
    end = time.time()        
    print("Processing shot:", shot_id, "took:", end-start)    
    
    
def process_shot(vid_path, output_video_path, start, end, corner_x, corner_y, width, height, output_width, output_height, fps, shot_id, method = "ffmpeg"):

    reframe = {}
    reframe['leftOffset'] = corner_x
    reframe['topOffset'] = corner_y
    reframe['width'] = width
    reframe['height'] = height
    
    output_resolution = f"{output_width}*{output_height}"
                
    temp_output_trim_path = output_video_path.replace('.mp4', f'_trim_shot_{shot_id}.mp4')
    temp_output_crop_path = output_video_path.replace('.mp4', f'_shot_{shot_id}.mp4')    

    start_log = time.time()
    
    if method == "moviepy":
        vid_out_moviepy(start, end, vid_path, temp_output_trim_path)
    else:
        vid_out(start, end, vid_path, temp_output_trim_path, re_encode=False, use_frames = True)
          
    end_log = time.time()
    print("Cutting shot:", shot_id, "took:", end_log-start_log)
    start_log = time.time()    
    crop_clip(temp_output_trim_path, temp_output_crop_path, reframe, output_resolution)
    end_log = time.time()    
    print("Cropping shot:", shot_id, "took:", end_log-start_log)    

    os.remove(temp_output_trim_path)

def regular_resizer(vid_id, output_path, start, end, reframe, larger_dimension):

    video = Video.query.get(vid_id)
    if video is None:
        return

    vid_path = f"{video.base_path}{video.path}"
    temp_output_path = output_path.replace('.mp4', '_temp.mp4')  
        
    width = reframe['width']*video.width
    height = reframe['height']*video.height
    
    aspect_ratio = width/height
    
    if abs(aspect_ratio-9/16) < 0.1: 
        output_height = larger_dimension
        output_width = int(output_height * 9/16)

    elif aspect_ratio == 1: 
        output_height = output_width = larger_dimension

    elif abs(aspect_ratio-16/9) < 0.1: 
        output_width = larger_dimension
        output_height = int(output_width * 9/16)
        
    else:
        output_height = larger_dimension
        output_width = int(output_height * aspect_ratio)
        
    output_resolution = f"{output_width}*{output_height}"  
    
    vid_out_moviepy(start, end, vid_path, temp_output_path)    
    crop_clip(temp_output_path, output_path, reframe, output_resolution)
    
    os.remove(temp_output_path)
    

def shot_resizer(vid_id, output_video_path, shots, larger_dimension= 1920, orientation="portrait", cutting_method = "moviepy"):

    start = time.time()
    
    if orientation == "portrait":
        output_height = larger_dimension
        output_width = int(output_height * 9/16)

    elif orientation == "square":
        output_height = output_width = larger_dimension

    elif orientation == "landscape":
        output_width = larger_dimension
        output_height = int(output_width * 9/16)
        
    else:
        raise ValueError("Invalid orientation. Must be 'landscape', 'portrait' or 'square'.")

    video = Video.query.get(vid_id)
    if video is None:
        return

    vid_path = f"{video.base_path}{video.path}"
    precomp_shots_path = f"{video.base_path}{video.precomp_shots_path}"

    with open(precomp_shots_path, 'rb') as file:
        frame_shots = pickle.load(file)
        
    vidcap = cv2.VideoCapture(vid_path)

    fps = vidcap.get(cv2.CAP_PROP_FPS)
    length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

    vidcap.release()
    
    frame_id = 0

    temp_output_video_path = output_video_path.replace('.mp4', '_temp.mp4')
    
    for shot_id, shot in enumerate(shots):
        print("processing shot:", shot_id)

        start_time = shot["startTimeStamp"] 
        end_time = shot["endTimeStamp"] 
        
        if shot_id > 0:
           start_frame = min(frame_shots, key=lambda shot_f: abs(shot_f - start_time*fps))
        else:
           start_frame = round(start_time*fps)
        
        if shot_id < len(shots)-1:   
            end_frame = min(frame_shots, key=lambda shot_f: abs(shot_f - end_time*fps))-1
        else:
            end_frame = round(end_time*fps)

        corner_x = shot['aspectRatio']['leftOffset']
        corner_y = shot['aspectRatio']['topOffset']
        width = shot['aspectRatio']['width'] 
        height = shot['aspectRatio']['height'] 
           
        if cutting_method == "use_cv2":                                   
           process_shot_cv2(vid_path, temp_output_video_path, start_frame, end_frame, corner_x, corner_y, width, height, output_width, output_height, fps, shot_id)
        elif cutting_method == "moviepy":
           end_time = end_time - 1/fps
           process_shot(vid_path, temp_output_video_path, start_time, end_time, corner_x, corner_y, width, height, output_width, output_height, fps, shot_id, "moviepy")   
        else:
           process_shot(vid_path, temp_output_video_path, start_frame, end_frame, corner_x, corner_y, width, height, output_width, output_height, fps, shot_id, "ffmpeg")                      

    print("Concatenating shots")
    merge_cmd = "ffmpeg -y -hide_banner -loglevel error "
    for shot_id, _ in enumerate(shots):
        merge_cmd += f"-i {temp_output_video_path.replace('.mp4', f'_shot_{shot_id}.mp4')} "
        
    if cutting_method == "ffmpeg":
        merge_cmd += f"-filter_complex 'concat=n={len(shots)}:v=1' {temp_output_video_path}"
    else:
        merge_cmd += f"-filter_complex 'concat=n={len(shots)}:v=1:a=1' {temp_output_video_path}"
                
    os.system(merge_cmd)
    
    audio_start = shots[0]["startTimeStamp"]
    audio_end = shots[-1]["endTimeStamp"]

    temp_audio_path = vid_path.replace('.mp4', '.aac')
    audio_cmd = f"ffmpeg -y -hide_banner -loglevel error -i {vid_path} -ss {audio_start} -to {audio_end} -vn {temp_audio_path}"
    os.system(audio_cmd)

    print("Adding audio")

    merge_cmd = f"ffmpeg -y -hide_banner -loglevel error -i {temp_output_video_path} -i {temp_audio_path} -c:v copy -c:a aac {output_video_path}"
    os.system(merge_cmd)

    for shot_id, _ in enumerate(shots):
        temp_crop_path = temp_output_video_path.replace('.mp4', f'_shot_{shot_id}.mp4')
        os.remove(temp_crop_path)

    os.remove(temp_audio_path)
    os.remove(temp_output_video_path)
            
    end = time.time()
    print("Download processing took:", end-start)
    
    
def style_sub(input_ass, fontname, fontsize, primarycolour, outlinecolour, borderstyle, marginv):
    """ Restyles subtitles in ASS format
        Arguments:
            input_ass: path to the subtitle file in ASS format
            fonts_dir, fontname, fontsize, primarycolour, outlinecolour, borderstyle, marginv: styling parameters obtained from burn_sub
    """

    with open(input_ass, 'r') as ass:
        lines = ass.readlines()

    styles = lines[9]
    styles = styles.split(',')

    styles[ass_format['Fontname']] = fontname
    styles[ass_format['Fontsize']] = fontsize
    styles[ass_format['PrimaryColour']] = primarycolour
    styles[ass_format['OutlineColour']] = outlinecolour
    styles[ass_format['BorderStyle']] = borderstyle
    styles[ass_format['MarginV']] = marginv

    styles = ','.join(styles)

    lines[9] = styles

    with open(input_ass, 'w') as ass:
        ass.writelines(lines)


def burn_sub(clip_path, srt_path, fonts_dir, fontname, fontsize, primarycolour, outlinecolour, borderstyle, marginv):
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
              outlinecolour, borderstyle, marginv)

    subs_cmd = f"ffmpeg -y -i {clip_path} -vf ass={ass_path}:fontsdir={fonts_dir} {subbed_clip_path}"
    os.system(subs_cmd)

    os.remove(clip_path)

    return subbed_clip_path
