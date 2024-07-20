import logging
import os
from app.ai_processing.dialogues.stt.srt_gen_provider_service import SrtGenProvidersService
from app.ai_processing.dialogues.stt.srt_gen_provider_types import SrtGenProviderTypes
from app.enums.upload_processing_status import UploadProcessingStatus
from app.services.upload_processing_service import UploadProcessingService
from config import Config
from app import db

from app.beta.models.video import Video

def srt_gen(vid_id, file_type = 'video', config_provider = Config.stt_provider, resize = False, translate = False):

   provider = None
   for prov in SrtGenProviderTypes:
      if prov.value.lower() == config_provider.lower():
         provider = prov
         break

   if (provider is None):
      logging.error(f'Cannot infer provider type from {config_provider}. using default DEEPGRAM')
      provider = SrtGenProviderTypes.DEEPGRAM

   try:

      video = Video.query.get(vid_id)

      if video is None:
         logging.error("Video not found")
         return

      base_path = video.base_path
      video_path = f"{base_path}{video.path}"
      
      audio_path = f"{base_path}{video.audio_path}"
      
      if file_type == 'audio':
         see_com = (
            f"python3.8 {Config.ROOT_APP_FOLDER}/app/ai_processing/dialogues/stt/seewav.py "
            f"--audio {str(audio_path)} "
            f"--video {str(video_path)} "
            f"-c 0.341,0.031,0.380 "
            f"-r 10 "
            f"-W 640 "
            f"-H 360"
         )
         os.system(see_com)

      srt_gen_provider_handler = SrtGenProvidersService(provider, video)
      srt_gen_provider_handler.generate_transcription()
      if translate:
          srt_gen_provider_handler.translate_video(translate)
          srt_gen_provider_handler.remove_unused_files_from_bucket()  
      if resize:        
          srt_gen_provider_handler.resize_video(resize)
    
   except Exception as err:
      logging.error("srt_gen error", err)
      video.processing_failed = 1
      db.session.commit()
      UploadProcessingService.inform(vid_id, UploadProcessingStatus.AUDIO_PROCESSING, 100, -1, True)
   
   return
