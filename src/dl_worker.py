from datetime import datetime, timezone

from app import db
from src.ai_processing.editing.process_editing import resize_vid, create_captions_overlays, get_xml_cropping_params, generate_clip_xml_file, generate_collection_xml_file
from src.ai_processing.dialogues.stt.json2srt_txt import write_srt_from_list, clip_srt_and_timestamp, write_srt
from src.models.file_handler import FileHandler
from src.models.user import User
import pickle
from src.models.download import Download
from src.models.video import Video
from src.services.sendgrid import SendGridService
from src.services.socket_broadcast import socket_broadcast
from src.config import Config
from urllib import parse
from werkzeug.utils import secure_filename
import os
import asyncio
import json
import boto3
import random
import string
import cv2
import shutil
from weasyprint import HTML
import markdown
import base64
import logging

def customize_guide(guide_md_path, zip_file_name, xml_file_name, master_video_name, guide_pdf_path):
    with open(guide_md_path, 'r') as file:
        markdown_content = file.read()

    placeholders = {
        "[ZIP_FILE_NAME]": zip_file_name,
        "[XML_FILE_NAME]": xml_file_name,
        "[MASTER_FILE_NAME]": master_video_name,
        "[IMAGE_PLACEHOLDER_adobe_zip_content]": f"{Config.ROOT_FOLDER}/app/assets/adobe_zip_content.png",
        "[IMAGE_PLACEHOLDER_adobe_open_project]": f"{Config.ROOT_FOLDER}/app/assets/open_project_adobe.png",
        "[IMAGE_PLACEHOLDER_adobe_select_xml]": f"{Config.ROOT_FOLDER}/app/assets/select_xml_adobe.png",
        "[IMAGE_PLACEHOLDER_adobe_locate_video]": f"{Config.ROOT_FOLDER}/app/assets/locate_video_adobe.png",
        "[IMAGE_PLACEHOLDER_adobe_success_import]": f"{Config.ROOT_FOLDER}/app/assets/success_import_adobe.png",
        "[IMAGE_PLACEHOLDER_adobe_new_project]": f"{Config.ROOT_FOLDER}/app/assets/new_project_adobe.png",
        "[IMAGE_PLACEHOLDER_adobe_select_video]": f"{Config.ROOT_FOLDER}/app/assets/select_video_new_adobe.png",
        "[IMAGE_PLACEHOLDER_adobe_import_multi]": f"{Config.ROOT_FOLDER}/app/assets/import_multi_xml_adobe.png",
        "[IMAGE_PLACEHOLDER_adobe_select_multi]": f"{Config.ROOT_FOLDER}/app/assets/select_multi_xml_adobe.png",
        "[IMAGE_PLACEHOLDER_adobe_clips_multi]": f"{Config.ROOT_FOLDER}/app/assets/multi_clips_adobe.png"}

    for placeholder, value in placeholders.items():
        if placeholder.startswith("[IMAGE_PLACEHOLDER_"):
            with open(value, "rb") as image_file:
                encoded_image = base64.b64encode(
                    image_file.read()).decode("utf-8")

            image_tag = f"""<img src="data:image/png;base64,{encoded_image}" style="width: 100%; max-width: 100%;" />"""
            markdown_content = markdown_content.replace(placeholder, image_tag)
        else:
            markdown_content = markdown_content.replace(
                placeholder, f"<span class='placeholder'>{value}</span>")

    css = f"""
    <style>
        @font-face {{
            font-family: 'Poppins';
            font-style: normal;
            font-weight: 600;
            src: url('{Config.ROOT_FOLDER}/app/fonts/Poppins-Medium.ttf') format('truetype');
        }}

        @font-face {{
            font-family: 'Inter';
            font-style: normal;
            font-weight: 400;
            src: url('{Config.ROOT_FOLDER}/app/fonts/Inter-Medium.ttf') format('truetype');
        }}

        h1 {{
            font-family: 'Poppins', sans-serif;
            font-weight: 600;
            font-size: 1.2em;
        }}

        h2 {{
            font-family: 'Poppins', sans-serif;
            font-weight: 600;
            font-size: 1.0em;
        }}
        
        h3 {{
            font-family: 'Poppins', sans-serif;
            font-weight: 600;
            font-size: 0.9em;
        }}        

        h4 {{
            font-family: 'Poppins', sans-serif;
            font-weight: 600;
            font-size: 0.8em;
        }}  
                
        p, ol, ul, li {{
            font-family: 'Inter', sans-serif;
            font-weight: 400;
            font-size: 0.7em;
        }}

        ol, ul, li {{
            font-family: 'Inter', sans-serif;
            font-weight: 400;
            font-size: 1em;
        }}
                    
        .placeholder {{
        font-family: monospace; 
        background-color: #f0f0f0;
        padding: 3px; 
        border-radius: 5px; 
    }}
    </style>
    """

    image_path = f"{Config.ROOT_FOLDER}/app/assets/iai.png"
    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode("utf-8")

    html_content = f"""
    <img src="data:image/png;base64,{encoded_image}" style="width: 100%; max-width: 100%;" />
    <br/>
    {markdown.markdown(markdown_content)}
    """

    HTML(string=css + html_content).write_pdf(guide_pdf_path)

    print(f"PDF file {guide_pdf_path} has been created.")


def zip_and_delete_folder(folder_path, save_folder_path, zip_name):
    zip_filename = f"{save_folder_path}{zip_name}"
    shutil.make_archive(zip_filename, 'zip', folder_path)
    shutil.rmtree(folder_path)

    print(f"Folder '{folder_path}' zipped and deleted successfully.")


def upload_tos3(file_name, bucket, object_name=None):
    """Upload a file to an S3 bucket

    :param file_name: File to upload
    :param bucket: Bucket to upload to
    :param object_name: S3 object name. If not specified then file_name is used
    :return: True if file was uploaded, else False
    """

    # If S3 object_name was not specified, use file_name
    if object_name is None:
        object_name = os.path.basename(file_name)

    # Upload the file
    s3_client = boto3.client('s3')
    try:
        response = s3_client.upload_file(file_name, bucket, object_name)
    except Exception as e:
        print(e)
        return False
    return True

def create_video_part(id):

    dl = Download.query.get(id)
    video = Video.query.get(dl.video_id)
    user = User.query.get(dl.user_id)
    opts = json.loads(dl.download_options)
    fh = FileHandler(video.path)

    reframe = opts.get('reframe', None)
    if reframe is not None:
        reframe = dict(parse.parse_qsl(reframe))

    logo = opts.get('logo', None)

    dl.file_name = fh.get_part_video(
        video, opts['start'], opts['stop'], reframe, logo, False)
    db.session.commit()

    message = json.dumps(
        {"email": user.email, "type": "download_complete", "data": dl.to_json()}, default=str)

    try:
        asyncio.run(socket_broadcast(
            message, f'?email={user.email}&worker=yes'))
    except Exception as err:
        print('download complete notifier failed', err)

    try:
        SendGridService.send_download_complete(user, dl.file_name)
    except Exception as err:
        print('send download complete email failed', err)

    dl_path = f"{video.base_path}{dl.file_name}"
    json_path = dl_path.replace('.mp4', '.json')
    with open(json_path, "w") as annotation:
        json.dump(opts, annotation)

    if Config.DOWNLOAD_TRAINING_ENABLED == '1':
        try:
            upload_tos3(json_path, Config.TRAINING_BUCKET_NAME)
            upload_tos3(dl_path, Config.TRAINING_BUCKET_NAME)
        except Exception as err:
            print('failed to upload for training', err)


def export_adobe(id, include_captions=False, larger_dimension=1920, master_video_height=None,
                 master_video_width=None, master_file_name=None, include_video=False):

    dl = Download.query.get(id)
    video = Video.query.get(dl.video_id)
    user = User.query.get(dl.user_id)
    opts = json.loads(dl.download_options)

    now = datetime.now(timezone.utc)
    rnd = ''.join(random.choice(string.ascii_uppercase) for i in range(4))
    clip_title_file_name = secure_filename(opts.get('title'))

    start, end = opts['start'], opts['stop']

    reframe = opts.get('reframe', None)
    if reframe is not None:
        reframe = dict(parse.parse_qsl(reframe))

    precomp_shots_path = f"{video.base_path}{video.precomp_shots_path}"

    with open(precomp_shots_path, 'rb') as file:
        raw_shots = pickle.load(file)

    file_name = f"{start}_{end}_{str(int(now.timestamp()))}_{rnd}.mp4"
    vid_path = f"{video.base_path}{video.path}"

    adobe_folder = f"/{user.id}/{video.id}/adobe/{dl.id}/"
    adobe_folder_path = f"{Config.ROOT_FOLDER}/app/upload{adobe_folder}"
    guide_md_path = f"{Config.ROOT_APP_FOLDER}/ai_processing/editing/xml_items/adobe_integration_guide.md"
    guide_pdf_path = f"{adobe_folder_path}{file_name.replace('.mp4', '')}/adobe_export_tutorial.pdf"
    export_folder_path = f"{adobe_folder_path}{file_name.replace('.mp4', '')}/"
    os.makedirs(export_folder_path)

    vidcap = cv2.VideoCapture(vid_path)
    video_duration = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

    if master_video_height is None:
        height_orig, width_orig = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(
            vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
    else:
        height_orig, width_orig = master_video_height, master_video_width

    vid_shape = (width_orig, height_orig)
    fps = vidcap.get(cv2.CAP_PROP_FPS)

    vidcap.set(cv2.CAP_PROP_POS_MSEC, float(start * 1000))
    frame_start = int(vidcap.get(cv2.CAP_PROP_POS_FRAMES))
    vidcap.set(cv2.CAP_PROP_POS_MSEC, float(end * 1000))
    frame_end = int(vidcap.get(cv2.CAP_PROP_POS_FRAMES))
    vidcap.release()

    clip_duration = frame_end - frame_start

    aspect_ratio = reframe.get('type')

    if aspect_ratio == "portrait_9_16":
        clip_h = larger_dimension
        clip_w = int(clip_h * 9/16)

    elif aspect_ratio == "square":
        clip_h = larger_dimension
        clip_w = clip_h

    else:
        clip_w = larger_dimension
        clip_h = int(clip_w * 9/16)

    clip_shape = (clip_w, clip_h)

    timestamp_path = f"{video.base_path}{video.timestamps_path}"
    if os.path.exists(timestamp_path):
        captions_data = clip_srt_and_timestamp(
            timestamp_path, start=start, end=end)['srt_subs']
        srt_path = f"{export_folder_path}/{clip_title_file_name}.srt"
        write_srt(srt_path, captions_data)

    cropping_params = []

    if reframe and reframe.get('shot_splitter', None) == 'True':
        shots = json.loads(reframe.get('shots', '[]}').replace("'", '"'))
                 
        shots = [[min(raw_shots, key=lambda shot_f: abs(shot_f - shot['startTimeStamp']*fps)),
                  min(raw_shots, key=lambda shot_f: abs(shot_f - shot['endTimeStamp']*fps)),
                  shot['aspectRatio']['leftOffset'], shot['aspectRatio']['topOffset'],
                  shot['aspectRatio']['width'], shot['aspectRatio']['height']] for shot in shots]
                  
        shots[0][0] = frame_start
        shots[-1][1] = frame_end

        shot_id = 0

        for frame_id in range(frame_start, frame_end):
            if frame_id >= shots[shot_id][1]:
                shot_id += 1

            x, y = int(shots[shot_id][2] *
                       width_orig), int(shots[shot_id][3]*height_orig)
            w, h = int(shots[shot_id][4] *
                       width_orig), int(shots[shot_id][5]*height_orig)
            frame_params = [[y, y + h, x, x + w, None]]
            cropping_params.append(frame_params)

    elif reframe:
        ratios = json.loads(reframe.get('clip_ratio').replace("'", '"'))
        x, y = int(ratios['leftOffset'] *
                   width_orig), int(ratios['topOffset']*height_orig)
        w, h = int(ratios['width'] *
                   width_orig), int(ratios['height']*height_orig)

        for frame_id in range(frame_start, frame_end):
            frame_params = [[y, y + h, x, x + w, None]]
            cropping_params.append(frame_params)

    else:
        for frame_id in range(frame_start, frame_end):
            frame_params = [[0, height_orig, 0, width_orig, None]]
            cropping_params.append(frame_params)

    xml_cropping_params = get_xml_cropping_params(
        cropping_params, vid_shape, clip_shape, frame_start, export_folder_path)

    xml_path = f"{export_folder_path}/{clip_title_file_name}.xml"

    if master_file_name is None:
        video_name = f'{video.title}.mp4'
    else:
        video_name = master_file_name

    if include_captions:
        overlays_folder_path = f"{export_folder_path}/caption_overlays/"
        os.makedirs(overlays_folder_path)

        xml_caption_params = create_captions_overlays(
            fps, clip_duration, clip_shape, captions_data, overlays_folder_path, opts.get('timestamps', []), animation_type='word')
        generate_clip_xml_file(xml_path, clip_duration, clip_title_file_name, clip_w, clip_h, xml_cropping_params, xml_caption_params,
                          video_duration, width_orig, height_orig, video_name, fps, include_captions=True)
    else:
        xml_caption_params = None
        generate_clip_xml_file(xml_path, clip_duration, clip_title_file_name, clip_w, clip_h, xml_cropping_params, xml_caption_params,
                          video_duration, width_orig, height_orig, video_name, fps, include_captions=False)

    if include_video:
        shutil.copy(vid_path, export_folder_path)

    zip_file_name = secure_filename(f"{clip_title_file_name}.zip")
    xml_file_name = secure_filename(f"{clip_title_file_name}.xml")

    customize_guide(guide_md_path, zip_file_name,
                    xml_file_name, video_name, guide_pdf_path)

    zip_and_delete_folder(export_folder_path, adobe_folder_path, clip_title_file_name)

    dl.file_name = f"{adobe_folder}{zip_file_name}"
    db.session.commit()

    message = json.dumps(
        {"email": user.email, "type": "download_complete", "data": dl.to_json()}, default=str)

    try:
        asyncio.run(socket_broadcast(
            message, f'?email={user.email}&worker=yes'))
    except Exception as err:
        print('download complete notifier failed', err)


def create_auto_tiktok(id):

    dl = Download.query.get(id)
    video = Video.query.get(dl.video_id)
    user = User.query.get(dl.user_id)
    opts = json.loads(dl.download_options)
    now = datetime.now(timezone.utc)
    rnd = ''.join(random.choice(string.ascii_uppercase) for i in range(4))

    file_dir = f"{video.user_id}/{video.id}/auto_tiktok/"
    dir = f"{video.base_path}/{file_dir}"
    os.makedirs(dir, exist_ok=True)
    file_name = f"{opts['start']}_{opts['stop']}_{str(int(now.timestamp()))}_{rnd}.mp4"
    fname = f"{dir}/{file_name}"

    resize_vid(dl.video_id, opts['faces_list'],
               opts['start'], opts['stop'], fname)
    # burn_subs(fname, opts.get('srt', []), fname, opts.get('timestamps', []), animation_type = 'word')

    dl.file_name = f"{file_dir}{file_name}"
    db.session.commit()

    message = json.dumps(
        {"email": user.email, "type": "download_complete", "data": dl.to_json()}, default=str)

    try:
        asyncio.run(socket_broadcast(
            message, f'?email={user.email}&worker=yes'))
    except Exception as err:
        print('download complete notifier failed', err)

    try:
        SendGridService.send_download_complete(user, dl.file_name)
    except:
        print('send download complete email failed', err)

    dl_path = fname
    json_path = dl_path.replace('.mp4', '.json')
    with open(json_path, "w") as annotation:
        json.dump(opts, annotation)

    if Config.DOWNLOAD_TRAINING_ENABLED == '1':
        try:
            upload_tos3(json_path, Config.TRAINING_BUCKET_NAME)
            upload_tos3(dl_path, Config.TRAINING_BUCKET_NAME)
        except Exception as err:
            print('failed to upload for training', err)


def export_adobe_magic(id, include_captions=False, clip_height=1920, master_video_height=None,
                       master_video_width=None, master_file_name=None, include_video=False):

    dl = Download.query.get(id)
    video = Video.query.get(dl.video_id)
    video_name = f"{video.title}.mp4"
    user = User.query.get(dl.user_id)
    opts = json.loads(dl.download_options)

    now = datetime.now(timezone.utc)
    rnd = ''.join(random.choice(string.ascii_uppercase) for i in range(4))

    file_dir = f"{video.user_id}/{video.id}/auto_tiktok/"
    dir = f"{video.base_path}/{file_dir}"
    os.makedirs(dir, exist_ok=True)
    file_name = f"{opts['start']}_{opts['stop']}_{str(int(now.timestamp()))}_{rnd}.mp4"
    dl.file_name = f"{file_dir}{file_name}"
    captions_data = opts.get('srt', [])
    clip_title_file_name = secure_filename(opts.get('title'))

    adobe_folder = f"/{user.id}/{video.id}/adobe/{dl.id}/"
    adobe_folder_path = f"{Config.ROOT_FOLDER}/app/upload{adobe_folder}"
    guide_md_path = f"{Config.ROOT_APP_FOLDER}/app/ai_processing/editing/xml_items/adobe_integration_guide.md"
    guide_pdf_path = f"{adobe_folder_path}{file_name.replace('.mp4', '')}/adobe_export_tutorial.pdf"
    export_folder_path = f"{adobe_folder_path}{file_name.replace('.mp4', '')}/"
    os.makedirs(export_folder_path)

    vid_path = f"{video.base_path}/{video.path}"
    vidcap = cv2.VideoCapture(vid_path)
    video_duration = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

    if master_video_height is None:
        height_orig, width_orig = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(
            vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
    else:
        height_orig, width_orig = master_video_height, master_video_width
    vid_shape = (width_orig, height_orig)

    fps = vidcap.get(cv2.CAP_PROP_FPS)
    vidcap.set(cv2.CAP_PROP_POS_MSEC, opts['start'] * 1000)
    frame_start = int(vidcap.get(cv2.CAP_PROP_POS_FRAMES))
    vidcap.set(cv2.CAP_PROP_POS_MSEC, opts['stop'] * 1000)
    frame_end = int(vidcap.get(cv2.CAP_PROP_POS_FRAMES))

    clip_duration = frame_end - frame_start
    clip_h = clip_height
    clip_w = int(clip_h * 9/16)
    clip_shape = (clip_w, clip_h)

    vidcap.release()

    xml_cropping_params = get_xml_cropping_params(
        opts['faces_list'], vid_shape, clip_shape, frame_start, export_folder_path)

    xml_path = f"{export_folder_path}/{clip_title_file_name}.xml"

    if include_captions:
        overlays_folder_path = f"{export_folder_path}/caption_overlays/"
        os.makedirs(overlays_folder_path)

        xml_caption_params = create_captions_overlays(
            fps, clip_duration, clip_shape, captions_data, overlays_folder_path, opts.get('timestamps', []), animation_type='word')
        generate_clip_xml_file(xml_path, clip_duration, clip_title_file_name, clip_w, clip_h, xml_cropping_params, xml_caption_params,
                          video_duration, width_orig, height_orig, video_name, fps, include_captions=True)
    else:
        xml_caption_params = None
        generate_clip_xml_file(xml_path, clip_duration, clip_title_file_name, clip_w, clip_h, xml_cropping_params, xml_caption_params,
                          video_duration, width_orig, height_orig, video_name, fps, include_captions=False)

    srt_path = f"{export_folder_path}/{clip_title_file_name}.srt"
    write_srt_from_list(srt_path, captions_data)

    if include_video:
        shutil.copy(vid_path, export_folder_path)

    zip_file_name = secure_filename(f"{clip_title_file_name}.zip")
    xml_file_name = secure_filename(f"{clip_title_file_name}.xml")

    customize_guide(guide_md_path, zip_file_name,
                    xml_file_name, video_name, guide_pdf_path)

    zip_and_delete_folder(export_folder_path, adobe_folder_path, clip_title_file_name)

    dl.file_name = f"{adobe_folder}{zip_file_name}"
    db.session.commit()

    message = json.dumps(
        {"email": user.email, "type": "download_complete", "data": dl.to_json()}, default=str)

    try:
        asyncio.run(socket_broadcast(
            message, f'?email={user.email}&worker=yes'))
    except Exception as err:
        print('download complete notifier failed', err)
        
      
def export_adobe_collection(id):

    now = datetime.now(timezone.utc)
    rnd = ''.join(random.choice(string.ascii_uppercase) for i in range(4))
    
    dl = Download.query.get(id)    
    user = User.query.get(dl.user_id)    
    opts = json.loads(dl.download_options)
    collection_name = secure_filename(opts.get('title'))        

    collection_id = int(opts.get('collection_id'))
    larger_sequence_dimension = opts.get('larger_sequence_dimension')
    aspect_ratio = opts.get('aspect')

    upload_folder_path = f"{Config.ROOT_FOLDER}/app/upload/"
    collection_folder_path = f"{user.id}/collections/{collection_id}"
    base_path = f"{upload_folder_path}{collection_folder_path}"
    if not os.path.exists(base_path):
        os.makedirs(base_path)
           
    collection_xml_file_name = f"{collection_name}_{str(int(now.timestamp()))}_{rnd}.xml"    
    xml_path = f"{base_path}/{collection_xml_file_name}"
                    
    generate_collection_xml_file(xml_path, collection_id, larger_sequence_dimension, aspect_ratio)   
    
    dl.file_name = f"{collection_folder_path}/{collection_xml_file_name}"
    
    db.session.commit()

    message = json.dumps(
        {"email": user.email, "type": "download_complete", "data": dl.to_json()}, default=str)

    try:
        asyncio.run(socket_broadcast(
            message, f'?email={user.email}&worker=yes'))
    except Exception as err:
        print('download complete notifier failed', err)


def worker_ping(worker_name):
    logging.info(f"{worker_name} ping")