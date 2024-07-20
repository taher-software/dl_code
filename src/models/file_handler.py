import json
from pathlib import Path
from flask_login import current_user
from src.ai_processing.editing.process_editing import add_logo
from src.video_processing import shot_resizer, regular_resizer, vid_out_moviepy
from src.models.file_manager import FileManager
from src.models.video import Video
from src.models.saved_clip import SavedClip
from src.config import Config
from datetime import datetime
from flask import send_from_directory, send_file
from werkzeug.utils import secure_filename

import base64
import cv2
import glob
import os
import random
import segment.analytics as analytics
import string
import zipfile

analytics.write_key = Config.SEGMENT_API_KEY


class FileHandler:
    category = None
    source = None
    source_type = None
    filename: None

    file_directory = None
    file_path = None
    found_file_path = None
    upload_user_path = None
    download_user_path = None
    tempfile = None

    source_sub = {
        '': 'videos/',
        'favourite': 'favorites/',
        'search': 'clips_sc/',
        'transcript': 'clips_tr/',
        'topic': 'clips_tp/'
    }

    sources = ['videos/', 'clips_sc/', 'clips_tr/', 'clips_tp/']

    source_types = ['upload', 'download']

    def __init__(self, filename, source_type='download', source='', category=''):

        self.category = f"{category}/"
        self.source = source
        self.source_type = source_type
        self.filename = filename

        self.upload_user_path = f"{Config.ROOT_FOLDER}/app/upload/"
        self.download_user_path = f"{Config.ROOT_FOLDER}/app/download/"
        self.file_directory = f"{Config.ROOT_FOLDER}/app/{self.source_type}/"
        self.file_path = f"{Config.ROOT_FOLDER}/app/{self.source_type}/"
        #print('FP', self.file_path)

    def get_file(self):
        fav_path = f"{self.download_user_path}favorites/{self.filename}"
        if os.path.isfile(self.file_path):
            #print('Found File in', self.file_path)
            return send_from_directory(self.file_directory, self.filename)
        elif self.source != 'favourite' and os.path.isfile(fav_path):
            #print('Found File in FAVS', fav_path)
            return send_from_directory(fav_path.replace(self.filename, ''), self.filename)
        else:
            return self.find_file()

    def find_file(self,):
        if self.source_type == 'download':
            if self.is_file_in_downloads():
                #print('Found File in SEARCH DL', self.found_file_path)
                return send_from_directory(self.found_file_path, self.filename)
            elif self.is_file_in_uploads():
                #print('Found File in SEARCH UL', self.found_file_path)
                return send_from_directory(self.found_file_path, self.filename)
            else:
                #print('NO FILE FOUND')
                return send_from_directory(self.file_path, self.filename)
        else:
            if self.is_file_in_uploads():
                #print('Found File in SEARCH UL', self.found_file_path)
                return send_from_directory(self.found_file_path, self.filename)
            elif self.is_file_in_downloads():
                #print('Found File in SEARCH DL', self.found_file_path)
                return send_from_directory(self.found_file_path, self.filename)
            else:
                # This should return a 404
                #print('NO FILE FOUND')
                return send_from_directory(self.file_path, self.filename)

    def is_file_in_downloads(self):
        has_file = False
        for f in glob.glob(f"{self.download_user_path}/**/{self.filename}", recursive=True):
            has_file = True
            self.found_file_path = f.replace(f"{self.filename}", '')
        return has_file

    def is_file_in_uploads(self):
        has_file = False
        for f in glob.glob(f"{self.upload_user_path}/**/{self.filename}", recursive=True):
            has_file = True
            self.found_file_path = f.replace(f"{self.filename}", '')
        return has_file

    def get_file_path(self):
        if self.is_file_in_uploads():
            return f"{self.found_file_path}{self.filename}"
        elif self.is_file_in_downloads():
            return f"{self.found_file_path}{self.filename}"
        return None

    def download_file(self):
        if self.is_file_in_uploads():
            return send_file(f"{self.found_file_path}{self.filename}", attachment_filename=self.filename, as_attachment=True)
        elif self.is_file_in_downloads():
            return send_file(f"{self.found_file_path}{self.filename}", attachment_filename=self.filename, as_attachment=True)
        return None

    def download_videos(self, video_ids, fname):
        ids = video_ids.split('_')
        zip_file = zipfile.ZipFile(fname, "w")
        for id in ids:
            vid = Video.query.get(id)
            zip_file.write(f"{vid.base_path}{vid.path}",
                           f"{secure_filename(vid.title)}.mp4", compress_type=zipfile.ZIP_DEFLATED)
        zip_file.close()
        return send_file(fname, attachment_filename=self.filename, as_attachment=True)

    def download_clip(self, fav, video, type='fv'):
        now = datetime.utcnow()  # current date and time
        fname = str(int(now.timestamp()))

        self.tempfile = f"{video.base_path}{video.temp_folder}_{fname}.mp4"
        if type == 'fv':
            file_loc = f"{video.base_path}{fav.fav_path}"
        else:
            file_loc = f"{video.base_path}{fav.clip_path}"

        vid_out_moviepy(float(fav.start), float(fav.end), file_loc, self.tempfile)
        return send_file(f"{self.tempfile}", attachment_filename=self.filename, as_attachment=True)

    def get_part_video(self, video, start, stop, reframe, logo, for_download = True, larger_dimension = 1920):
        now = datetime.utcnow()  # current date and time
        rnd = ''.join(random.choice(string.ascii_uppercase) for i in range(4))
        fname = f"{video.user_id}_search_{str(int(now.timestamp()))}_{rnd}"

        if reframe and reframe.get('shot_splitter', None) == 'False':
            ratios = json.loads(reframe.get('clip_ratio').replace("'",'"'))
            width = ratios['width']
            height = ratios['height']

        if for_download is False:
            fname = f"{video.user_id}_{str(int(now.timestamp()))}_{rnd}"

        file_loc = f"{video.base_path}{video.path}"
        
        logo_file = f"{video.base_path}{video.temp_folder}_{fname}_logo.mp4"
        if for_download is False:
            self.trimfile = f"{video.base_path}{video.temp_folder.replace('_temp','_dl')}_{fname}_dl.mp4"
            self.cropfile = f"{video.base_path}{video.temp_folder.replace('_temp','_dl')}_{fname}_crop.mp4"
        else:
            self.trimfile = f"{video.base_path}{video.temp_folder}_{fname}_dl.mp4"
            self.cropfile = f"{video.base_path}{video.temp_folder}_{fname}_crop.mp4"
        
        if reframe and reframe.get('shot_splitter', None) == 'True':
            reframe_string = reframe.get('ct', 'portrait')
            reframe_parts = reframe_string.split("_")
            shots = json.loads(reframe.get('shots', '[]}').replace("'",'"'))
            shot_resizer(video.id, self.cropfile, shots, larger_dimension, reframe_parts[0])
            self.tempfile = self.cropfile
        elif reframe and (width!=1 or height!=1): 
            regular_resizer(video.id, self.cropfile, start, stop, ratios, larger_dimension)
            self.tempfile = self.cropfile
        else:
            vid_out_moviepy(start, stop, file_loc, self.trimfile)
            self.tempfile = self.trimfile

        if logo is not None:
            fm = FileManager.query.get(logo.get('id'))
            if fm is not None:
                left = float(logo.get('leftOffset'))            
                top = float(logo.get('topOffset'))
                width = float(logo.get('width'))
                height = float(logo.get('height')) 
                add_logo(self.tempfile, f'{Config.ROOT_FOLDER}/app/upload{fm.location}', logo_file, (left, top, width, height))
                self.tempfile = logo_file

        if self.tempfile != self.trimfile:
            (Path(self.trimfile)).unlink(missing_ok=True)
        
        if self.tempfile != self.cropfile:
            (Path(self.cropfile)).unlink(missing_ok=True)

        if for_download is False: return self.tempfile.replace(video.base_path, '')

        return send_file(f"{self.tempfile}", attachment_filename=self.filename, as_attachment=True)

    def delete_clip(self, fname):
        if fname is not None and os.path.isfile(fname):
            os.remove(fname)

    def delete_file(self, path):
        if os.path.isfile(path):
            os.remove(path)

    def download_clips(self, fav_ids):
        now = datetime.utcnow()  # current date and time
        fname = f"{Config.ROOT_FOLDER}/app/upload/{str(int(now.timestamp()))}"
        self.tempfile = fname
        ids = fav_ids.split('_')
        zip_file = zipfile.ZipFile(f"{fname}.zip", "w")
        for id in ids:
            fav = SavedClip.query.get(id)
            video = Video.query.get(fav.video_id)
            analytics.track(current_user.id, 'Clip downloaded', {
                            'Source': 'Saved clips', 'Clip duration': fav.end - fav.start, 'Video duration': video.duration, 'Language': video.language, 'Video type':  video.preproc})
            vid_out_moviepy(float(fav.start), float(fav.end),
                    f"{video.base_path}{fav.video.path}", f"{fname}_{fav.id}.mp4")
            zip_file.write(f"{fname}_{fav.id}.mp4", f"{secure_filename(video.title)}_{fav.id}.mp4",
                           compress_type=zipfile.ZIP_DEFLATED)
        zip_file.close()
        return send_file(f"{fname}.zip", attachment_filename=self.filename, as_attachment=True)

    def delete_downloaded_clips(self):
        try:
            for remove_file in glob.glob(f"{self.tempfile}*"):
                print("remove_file", remove_file)
                os.remove(remove_file)
        except:
            print("failed to delete files", f"{self.tempfile}.zip")

    def get_clip(self):
        if self.is_file_in_uploads():
            return self.get_clip_thumb()
        elif self.is_file_in_downloads():
            return self.get_clip_thumb()
        return None

    def get_clip_thumb_at(self, second):
        path = self.get_file_path()

        if path is None:
            print('path is none')
            return

        vidcap = cv2.VideoCapture(f"{path}")
        vidcap.set(cv2.CAP_PROP_POS_MSEC, int(float(second) * 1000))

        width = vidcap.get(3)   # float `width`
        height = vidcap.get(4)  # float `height`

        height_ratio = 500 / height
        width = round(width * height_ratio)

        success, image = vidcap.read()

        print(success)

        _, buffer = cv2.imencode('.jpg', cv2.resize(image, (width, 500)))

        jpg_as_text = ''
        try:
            jpg_as_text = base64.b64encode(buffer)
        except Exception as err:
            print('cannot encode buffer', err)
        vidcap.release()
        return jpg_as_text

    def get_clip_thumb(self):
        vidcap = cv2.VideoCapture(f"{self.found_file_path}{self.filename}")

        width = vidcap.get(3)   # float `width`
        height = vidcap.get(4)  # float `height`

        height_ratio = 300 / height
        width = round(width * height_ratio)

        _, image = vidcap.read()
        _, buffer = cv2.imencode('.jpg', cv2.resize(image, (width, 300)))
        width = vidcap.get(3)   # float `width`
        height = vidcap.get(4)  # float `height`

        jpg_as_text = base64.b64encode(buffer)
        vidcap.release()
        return jpg_as_text
