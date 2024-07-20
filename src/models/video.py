import json
import logging
import os
import glob
import cv2
import imageio
import shutil

from datetime import datetime
from sqlalchemy import Boolean, Column, DateTime, Enum, Integer, String, ForeignKey, Numeric
from sqlalchemy.orm import relationship

from src.models.enums.integration_status import IntegrationStatus
from src.models.enums.integration_type import IntegrationType
from src.models.user import User
from app import db
from src.path import join_paths
from src.config import Config


class Video(db.Model):
    __tablename__ = 'video'

    FK_VIDEO_ID = 'video.id'

    integration_type_list = [integration_type.value for integration_type in IntegrationType]
    integration_status_list = [integration_status.value for integration_status in IntegrationStatus]


    id = Column(Integer, primary_key=True)
    audio_path = Column(String(256))
    metadata_path = Column(String(256))    
    visual_metadata_path = Column(String(256))        
    base_path = Column(String(256))
    bitrate = Column(Numeric(10,6), default=0)
    datetime_created = Column(DateTime, name='uploaded_time', default=datetime.utcnow)
    duration = Column(Numeric(10,2), default=0)
    file_name = Column(String(200))
    file_size = Column(Numeric(10,6), default=0)
    fps = Column(Integer)
    original_extension = Column(String(200))
    height = Column(Integer, default=0)
    hidden = Column(Integer, default=0)
    is_test = Column(Integer, default=0)
    visual_embedding_model = Column(String(256))
    audio_embedding_model = Column(String(256)) 
    dialogue_embedding_model = Column(String(256))   
    face_recognition_embedding_model = Column(String(256))   
    language = Column(String(200))
    length = Column(Integer)
    path = Column(String(256))
    precomp_audio_path = Column(String(256))
    precomp_faces_path = Column(String(256))
    tracked_faces_path = Column(String(256))  
    precomp_cast_path = Column(String(256))      
    precomp_sub_path = Column(String(256))
    precomp_vid_path = Column(String(256))
    thumbnails_base_path = Column(String(256))
    precomp_shots_path = Column(String(256))
    preproc = Column(String(1))
    processing_failed = Column(Integer, default=0)
    processed_faces =  Column(Integer, default=0)
    processed_face_tracking = Column(Integer, default=0)
    processed_cast =  Column(Integer, default=0)    
    processed_pc = Column(Integer, default=0)
    processed_shots = Column(Integer, default=0)    
    processed_au = Column(Integer, default=0)
    processed_srt_gen = Column(Integer, default=0)
    processed_topics = Column(Integer, default=0)
    processed_chapters = Column(Integer, default=0)    
    processed_metadata = Column(Integer, default=0)  
    processed_visual_metadata = Column(Integer, default=0)        
    has_topics = Column(Boolean, default=False)
    gif_path = Column(String(256))
    rencoded = Column(Integer, default=1)
    srt_folder = Column(String(256))
    srt_path = Column(String(256))
    srt_trans_path = Column(String(256))
    speaker_ts_path = Column(String(256))
    speakers_path = Column(String(256))
    summary_path = Column(String(256))
    timestamps_path = Column(String(256))
    title = Column(String(200))
    topics_path = Column(String(256))
    chapters_path = Column(String(256))    
    txt_folder = Column(String(256))
    txt_path = Column(String(256))
    video_thumb = Column(String(256), default='default.png')
    user_id = Column(Integer, ForeignKey(User.FK_USER_ID, name="fk_video_user_id", ondelete='CASCADE'))
    package_owner_user_id = Column(Integer, ForeignKey('user.id', ondelete='CASCADE', name='fk_video_package_owner_user_id'))
    speakers_path = Column(String(256))
    temp_folder = Column(String(256))
    width = Column(Integer, default=0)
    rencoded = Column(Integer, default=1)
    get_clips = Column(Integer, default=0)
    upload_source = Column(String(10), default='')
    integration_type = Column(Enum(*integration_type_list, name='integration_type'), default='DEFAULT')
    integration_status = Column(Enum(*integration_status_list, name='integration_status'), default='COMPLETE')
    processed_hls = Column(Integer, default=0)

    """ Relationships """
    clips = relationship('Clip', backref='clip_videos', cascade='all, delete', lazy='dynamic', overlaps='clips')
    folders = relationship('FolderVideo', backref='folder_videos', cascade='all, delete')
    progress = relationship('VideoUploadProgress', backref='video_progresses', cascade='all, delete', lazy='dynamic', overlaps='progress')
    saved_clips = relationship('SavedClip', backref='saved_clip_videos', cascade='all, delete', lazy='dynamic', overlaps='saved_clips')

    
    content_paths = {
        'audio_path': {'path': 'audios', 'ext': '.wav'},
        'metadata_path': {'path': 'metadata', 'ext': '.json'},
        'gif_path': {'path': 'gifs', 'ext': '.gif'},
        'path': {'path': 'videos', 'ext': '.mp4'},
        'precomp_sub_path': {'path': 'precomp_sub', 'ext': '.npy'},
        'precomp_vid_path': {'path': 'precomp_vid', 'ext': '.npy'},
        'precomp_audio_path': {'path': 'precomp_audio', 'ext': '.npy'},
        'srt_folder': {'path': 'srts', 'ext': ''},
        'srt_trans_path': {'path': 'srts_translated', 'ext': ''},
        'summary_path': {'path': 'summaries', 'ext': '.txt'},
        'timestamps_path': {'path': 'timestamps', 'ext': '.json'},
        'txt_path': {'path': 'txts', 'ext': '.txt'},
        'speaker_ts_path': {'path': 'spts', 'ext': '.txt'},
        'topics_path': {'path': 'topics', 'ext': '.json'},
        'chapters_path': {'path': 'chapters', 'ext': '.json'},
        'precomp_faces_path': {'path': 'faces', 'ext': '.pkl'},
        'tracked_faces_path': {'path': 'faces_tracking', 'ext': '.pkl'},
        'precomp_cast_path': {'path': 'cast', 'ext': '.pkl'},
        'speakers_path': {'path': 'speakers', 'ext': '.pkl'},
        'precomp_shots_path': {'path': 'shots', 'ext': '.pkl'},
        'txt_folder': {'path': 'txts', 'ext': ''},
        'thumbnails_base_path': {'path': 'thumbnails', 'ext': ''},
        'temp_folder': {'path': 'temp', 'ext': ''},
        'video_thumb': {'path': 'video_thumb', 'ext': '.jpg'},
    }

    def __init__(self):
        self.base_path = f"{Config.ROOT_FOLDER}/app/upload/"

    def __repr__(self):
        return f"Video('{self.id}','{self.title}','{self.path}')"

    def set_attributes_from_file_upload(self, media_form, author, user_package):
        self.title = media_form.file.title.replace(media_form.extension, '')
        self.file_name = media_form.file.filename
        self.language = media_form.language
        self.author = author
        self.package_owner_user_id = user_package.user_package_owner.owner_id
        """ Preproc is 4 if audio only """
        """ Preproc is 1 if dialog only """
        """ Preproc is 2 if visual only """
        """ Preproc is 3 if both visual and dialog """
        self.preproc = '3' if len(
            media_form.process) == 3 else '1' if 'dialog' in media_form.process else '2' if 'visual' in media_form.process else '4'

        for key in self.content_paths:
            ext = self.content_paths[key]['ext']
            path = self.content_paths[key]['path']
            setattr(self, key, f"{self.file_name}_{path}{ext}")

    def set_attributes_from_youtube(self, media_form, author, user_package):
        self.title = media_form.title
        self.file_name = media_form.filename
        self.preproc = '3' if len(
            media_form.process) == 3 else '1' if 'dialog' in media_form.process else '2' if 'visual' in media_form.process else '4'
        self.language = media_form.language
        self.author = author
        self.package_owner_user_id = user_package.user_package_owner.owner_id
        for key in self.content_paths:
            ext = self.content_paths[key]['ext']
            path = self.content_paths[key]['path']
            setattr(self, key, f"{self.file_name}_{path}{ext}")

    """ Deletes uploaded videos and all their dependencies (precomputed embeddings, srt files, summaries) """

    def delete_stored_content(self):
        vids = Video.query.filter_by(file_name=self.file_name).all()
        '''Only delete if not a shared resource'''
        if len(vids) == 1:
            try:
                for remove_file in glob.glob(f"{self.base_path}{self.file_name}*"):
                    os.remove(remove_file)
            except:
                logging.error("failed to delete files")
        return

    def to_json(self):
        data = {c.name: str(getattr(self, c.name))
                for c in self.__table__.columns}
        hls_path = self.path.replace('.mp4', '.m3u8')
        filename = os.path.basename(hls_path)
        hls_path = f"/{self.user_id}/{self.id}/hls/{filename}"
        resource_path = join_paths(
            Config.ROOT_FOLDER, 'app/upload/', hls_path.lstrip('/'))
        data['hls_path'] = None
        if os.path.exists(resource_path):
            data['hls_path'] = hls_path
        return data

    def to_json_with_shots_and_transcript(self):
        from app.ai_processing.editing.process_editing import ai_shot_splitter
        response = self.to_json()

        try:
            shots = ai_shot_splitter(self.id)
            shots_milliseconds = [(shot*1000) for shot in shots]
            response['shots'] = {
                'shot_changes_milliseconds': shots_milliseconds}
        except Exception as err:
            response['shots'] = {'shot_changes_milliseconds': []}

        timestamps_path = f"{self.base_path}{self.timestamps_path}"
        response_transcript = []
        if self.timestamps_path and os.path.exists(timestamps_path):
            try:
                with open(timestamps_path) as trspt:
                    transcript = json.load(trspt)
                    for tsl in transcript:
                        response_transcript.append(
                            {"caption": tsl[0], "start": tsl[1], "stop": tsl[2]})
            except:
                print('No transcript')

        response['transcript'] = response_transcript
        return response

    def responseJson(self):
        return {'title': self.title, 'preproc': self.preproc, 'id': self.id}

    def set_thumb(self, vidcap):
        try:
            midframe_id = int(self.length/2)
            vidcap.set(cv2.CAP_PROP_POS_FRAMES, midframe_id)
            _, image = vidcap.read()
            height, width, channels = image.shape
            height_ratio = 200 / height
            new_width = round(width * height_ratio)
            image = cv2.resize(image, (new_width, 200))
            cv2.imwrite(f"{self.base_path}{self.video_thumb}", image)
        except:
            logging.error(
                f"thumbnail failed midframe_id: {midframe_id} not found")

    def generate_gif(self, vidcap):
        images = []

        midframe_id = int(self.length/2)
        frames_id = (midframe_id - self.fps,
                     midframe_id, midframe_id + self.fps)

        for frame_id in frames_id:
            try:
                vidcap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
                _, frame = vidcap.read()
                height, width, channels = frame.shape
                height_ratio = 200 / height
                new_width = round(width * height_ratio)
                frame = cv2.resize(frame, (new_width, 200))
                images.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                imageio.mimsave(f"{self.base_path}{self.gif_path}", images)
            except:
                logging.error(f"gif frame_id: {frame_id} not found")

    def generate_audio_gif(self):
        shutil.copyfile(Config.ROOT_APP_FOLDER + '/app/static/images/audio.gif',
                        f"{self.base_path}{self.gif_path}")
        shutil.copyfile(Config.ROOT_APP_FOLDER + '/app/static/images/audio.jpg',
                        f"{self.base_path}{self.video_thumb}")

        gif_path = f"{self.base_path}{self.gif_path}"
        vidgif_path = gif_path.replace('.gif', '.mp4')
        vidgif = "ffmpeg -hide_banner -loglevel error -y -f gif -i " + gif_path + \
            ' -vf scale=w=iw/3:h=ih/3 -r 10 -preset veryfast -crf 26 ' + vidgif_path
        os.system(vidgif)


