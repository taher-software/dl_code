import logging
import threading

from src.config import Config
from src.models.video import Video
from datetime import datetime
from dateutil.relativedelta import relativedelta
from flask import send_file
from pycaption import SRTReader
import boto3
import cv2
import imageio
import math
import os
from sqlalchemy import or_

from src.services.upload_processing_service import UploadProcessingService


class utils:

    vd_strg = f'{Config.ROOT_FOLDER}/app/upload'
    user_directories = ['audios', 'clips_sc', 'favorites', 'precomp_sub', 'precomp_vid', 'srts',
                        'srts_translated', 'summaries', 'timestamps', 'txts', 'videos']

    @staticmethod
    def check_create_user_directories(username):
        for dir in utils.user_directories:
            dir_path = f'{utils.vd_strg}/{username}/{dir}'
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
        return

    @staticmethod
    def generate_gif(vid_path, frame_ids, save_path):
        vidcap = cv2.VideoCapture(vid_path)
        images = []

        for frame_id in frame_ids:
            try:
                vidcap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
                _, frame = vidcap.read()
                height, width, channels = frame.shape
                height_ratio = 200 / height
                new_width = round(width * height_ratio)
                frame = cv2.resize(frame, (new_width, 200))
                images.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                imageio.mimsave(save_path, images)
            except:
                logging.error(f"gif frame_id: {frame_id} not found")

    @staticmethod
    def read_captions(srt_path):
        """ Read caption from SRT files
            Arguments:
                srt_path: path to a .srt subtitle file
            returns:
                captions: list of captions
        """

        srt_text = open(srt_path).read()
        srt = SRTReader().read(srt_text, lang="en")
        captions = srt.get_captions("en")

        return captions
        
    @staticmethod   
    def download_from_s3(bucket_name, file_key, local_path):
        """Download a file from S3 bucket to a local path."""
    
        s3_client = boto3.client('s3')
    
        try:

            s3_client.download_file(bucket_name, file_key, local_path)
            print(f"Successfully downloaded '{file_key}' to '{local_path}'")
        except Exception as e:
            print(f"Error downloading file: {e}")        

    def convertMillis(millis):

        seconds = math.floor((millis/1000) % 60)
        minutes = math.floor((millis/(1000*60)) % 60)
        hours = math.floor((millis/(1000*60*60)) % 24)
        milli_seconds = millis - (hours*1000*60*60) - \
            (minutes*1000*60) - (seconds*1000)

        return f"{hours:02}:{minutes:02}:{seconds:02}.{(str(milli_seconds)).ljust(6,'0')}"

    def pagination_details(pagination):
        return {
            'currentPage': pagination.page,
            'perPage': pagination.per_page,
            'totalPages': math.ceil(pagination.total / pagination.per_page),
            'totalRecords': pagination.total
        }

    def send_file_with_attachment_name(file_location, filename):
        response = send_file(file_location, as_attachment=True,
                             attachment_filename=filename, download_name=filename)
        response.headers["x-filename"] = f"{filename.encode('utf-8').decode('unicode-escape')}"
        response.headers["Access-Control-Expose-Headers"] = 'x-filename'
        return response

    @staticmethod
    def get_current_month_subscription_period(subscription_start, subscription_end, current_date=None):
        if current_date is None:
            current_date = datetime.utcnow()
        """Calculates the start date for the current month 

        Args:
            subscription_start (datetime): A datetime value
            subscription_end (datetime): A datetime value
            current_date (datetime): Defaults to the current date

        Returns:
            datetime or None: The start date for the current month based on the current_date
        """

        if subscription_start > subscription_end:
            raise Exception('Subscription End date must be after Start date')

        if subscription_end < current_date:
            raise Exception('Subscription has ended')

        if subscription_start > current_date:
            end = subscription_start  + relativedelta(months=+1)
            return {"start": subscription_start, "end": end} 

        i = 1
        start = subscription_start
        while True:
            if (i > 36):
                """Failsafe for infinite recursion"""
                raise Exception('Unable to determine subscription start')
            end = subscription_start + relativedelta(months=+i)
            if end <= current_date:
                start = end
            else:
                break
            i = i +1

        return {"start": start, "end": end}

    @staticmethod
    def get_subscription_videos(owner_id, in_test = []):
        return Video.query.filter(
            or_(
                Video.package_owner_user_id == owner_id,
                Video.is_test.in_(in_test),
            ), 
            Video.hidden != 1
        ).all()

    @staticmethod
    def get_subscription_video_base_filter(owner_id, in_test = []):
        return Video.query.filter(
            or_(
                Video.package_owner_user_id == owner_id,
                Video.is_test.in_(in_test),
            ), 
            Video.hidden != 1
        )

    @staticmethod
    def inform_in_thread(vid_id, status, progress = 0, estimate = -1):
        thread = threading.Thread(target=UploadProcessingService.inform, args=(vid_id, status), kwargs={'progress': progress, 'estimated_full_duration': estimate, 'from_thread': True})
        thread.start()
        return thread


    @staticmethod
    def should_report(last_reported, progress):
        if last_reported > 0.75 and progress == 1:
            return True
        if last_reported < 0.75 and progress >= 0.75:
            return True
        if last_reported < 0.5 and progress >= 0.5:
            return True
        if last_reported < 0.25 and progress >= 0.25:
            return True
        if last_reported < 0.1 and progress >= 0.1:
            return True
        if last_reported < 0.01 and progress >= 0.01:
            return True

        return False
