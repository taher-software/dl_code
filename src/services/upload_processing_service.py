from src.enums.upload_processing_status import UploadProcessingStatus
from src.models.user import User
from src.models.user_package_owner import UserPackageOwner
from src.models.video import Video
from src.models.video_upload_progress import VideoUploadProgress
from app import db
from src.services.sendgrid import SendGridService
from src.services.socket_broadcast import socket_broadcast

from flask_sse import sse
from sqlalchemy import desc

import asyncio
import json

session = db.session
class UploadProcessingService:
    @staticmethod
    def inform(vid_id: int, status: UploadProcessingStatus = UploadProcessingStatus.QUEUED, progress=0, estimated_full_duration=-1, failed=False, from_thread=False):
        from app import app
        with app.app_context():
            try:
                video_upload_progress = UploadProcessingService.update_db(
                    vid_id, status, progress, estimated_full_duration)
                UploadProcessingService.broadcast(vid_id, video_upload_progress.to_json())
            except Exception as err:
                print("UploadProcessingService - Unable to broadcast", err)
                pass
            finally:
                if from_thread:
                    session.close()

    @staticmethod
    def update_db(vid_id: int, status: UploadProcessingStatus = UploadProcessingStatus.QUEUED, progress=0, estimated_full_duration=-1):
        from app import q_dl, q_pc, q_au, q_tr
        video = Video.query.get(vid_id)
        if video is None:
            return

        package = UserPackageOwner.query.filter_by(
            owner_id=video.package_owner_user_id, status=1).order_by(desc(UserPackageOwner.id)).first()

        if package is None:
            return

        additional_info = None
        if status != UploadProcessingStatus.QUEUED:
            audio_processing = q_au.fetch_job(f'au_{video.id}')
            reenc = q_dl.fetch_job(f'ec_{video.id}')
            visual_p = q_pc.fetch_job(f'pc_{video.id}')
            translation = q_tr.fetch_job(f'tr_{video.id}')

            additional_info = {
                'needed_audio_processing': audio_processing is not None,
                'needed_re_encoding': reenc is not None,
                'needed_visual_processing': visual_p is not None,
                'needed_dialog_processing': translation is not None
            }

        video_upload_progress = VideoUploadProgress.query.filter_by(
            video_id=vid_id, status=status.name).first()
        if video_upload_progress is not None:
            video_upload_progress.estimated_full_duration = estimated_full_duration
            video_upload_progress.progress = progress
            video_upload_progress.rencoded = video.rencoded
            video_upload_progress.processed_au = video.processed_au
            video_upload_progress.processed_pc = video.processed_pc
            video_upload_progress.processed_srt_gen = video.processed_srt_gen
            video_upload_progress.additional_info = json.dumps(additional_info)
            video_upload_progress.failed = video.processing_failed
            session.commit()
        else:
            video_upload_progress = VideoUploadProgress(
                video_id=vid_id,
                status=status.name,
                estimated_full_duration=estimated_full_duration,
                progress=progress,
                rencoded=video.rencoded,
                processed_au=video.processed_au,
                processed_pc=video.processed_pc,
                processed_srt_gen=video.processed_srt_gen,
                additional_info=json.dumps(additional_info),
                failed = video.processing_failed
            )
            session.add(video_upload_progress)
            session.commit()

        user = User.query.get(video.user_id)
        if video_upload_progress.processing_completed() == True:
            SendGridService.send_video_processed_confirmation(
                user, video.title)
        
        return video_upload_progress

    @staticmethod
    def auto_tiktok(vid_id):
        try:
            video = Video.query.get(vid_id)
            if video.processing_failed == 1:
                return
            if (video.processed_face_tracking == 0 or
               video.processed_faces == 0 or
               video.processed_topics == 0 or
               video.has_topics == False):
                return

            from app.services.subscription_service import SubscriptionService
            subscription_service = SubscriptionService(
                video.package_owner_user_id)

            if subscription_service._package_owner is None or subscription_service._package_owner.user_packages is None:
                print('Cannot find user package')
                return

            for user_package in subscription_service._package_owner.user_packages:
                try:
                    user_email = user_package.user.email
                    message = json.dumps({
                        "video_id": vid_id,
                        "email": user_email,
                        "type": "auto_tiktok_available",
                        "data": {
                            "file_name": video.file_name
                        }
                    }, default=str)
                    asyncio.run(socket_broadcast(
                        message, f'?email={user_email}&worker=yes'))
                except Exception as err:
                    print('broadcast progress notifier failed', err)
        except:
            pass

    @staticmethod
    def broadcast(vid_id: int, video_upload_progress):
        from app import app
        video = Video.query.get(vid_id)
        if video is None:
            return

        with app.app_context():
            sse.publish(data=json.dumps(video_upload_progress), type='upload_processing',
                        id=f'q_processing_{video_upload_progress["id"]}', channel=video.package_owner_user_id)