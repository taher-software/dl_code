import shutil
import time
import torch
import numpy as np
import os
from app.beta.orm import session
from app.enums.upload_processing_status import UploadProcessingStatus
from app.beta.models.video import Video
from app.services.upload_processing_service import UploadProcessingService
from app.utils import utils
from config import Config
import cv2
import subprocess
from transformers import ClapFeatureExtractor, ClapModel
import librosa 

# please do not remove this line
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using:", device)

def has_audio(filename):
    cmd = ['ffprobe', '-i', filename]
    output = subprocess.check_output(
        cmd, universal_newlines=True, stderr=subprocess.STDOUT)
    print('Stream #0:1' in output)
    return 'Stream #0:1' in output


def precompute_audios(vid_id):
    
    video = Video.query.get(vid_id)
    if video is None:
        return
    try:
        audio_embedding_model = Config.AUDIO_EMBEDDING_MODEL
        
        if audio_embedding_model == "LARGER_CLAP_GENERAL":
           print("Using LAION Clap model")
           audio_model = ClapModel.from_pretrained("laion/larger_clap_general").to(device)
           audio_feature_extractor = ClapFeatureExtractor.from_pretrained("laion/larger_clap_general")       
        else:
           print("Error: Model not supported")
           return
           
        vid_path = f"{video.base_path}{video.path}"
        audio_path = f"{video.base_path}{(video.audio_path.replace('.mp3', '.wav')).replace('.wav','._au.wav')}"
        precomp_audio_path = f"{video.base_path}{video.precomp_audio_path}"

        temp_audio_dir = f"{video.base_path}{video.temp_folder}/pre_comp_audios"
        if not os.path.exists(temp_audio_dir):
            os.makedirs(temp_audio_dir)

        vidcap = cv2.VideoCapture(vid_path)
        fps = round(vidcap.get(cv2.CAP_PROP_FPS))
        length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

        precomp = []

        if has_audio(vid_path):
            print('processing file with audio:', vid_path)

            wav_com = f"ffmpeg -y -hide_banner -loglevel warning -i {str(vid_path)} -ac 1 {str(audio_path)}"
            os.system(wav_com)

            start_time = time.time()
            last_reported = 0
            for i in range(fps, length, fps):
                if i%(30*fps)==0:
                    print("precomputing audio frame", i)
                if i == length:
                    start, end = (length-2*fps)/fps, length/fps
                else:
                    start, end = (i - fps)/fps, (i + fps)/fps

                error_log_path = f"{temp_audio_dir}/ffmpeg_error_log_{i}.txt"
                temp_audio_path = f"{temp_audio_dir}/{i}.wav"
                cut_com = f"ffmpeg -y -hide_banner -loglevel warning -i {audio_path} -ss {str(start)} -to {str(end)} {temp_audio_path} 2> {error_log_path}"
                exit_code = os.system(cut_com)

                if exit_code != 0:
                    print("Failed to execute the ffmpeg command.")

                    with open(error_log_path, "r") as error_log:
                        error_message = error_log.read()
                        if error_message:
                            print(error_message)

                if audio_embedding_model in ["LARGER_CLAP_GENERAL"]:
                    waveform, sr = librosa.load(temp_audio_path, sr=48000)
                    waveform = waveform.reshape(1, -1)
                    inputs = audio_feature_extractor(raw_speech=waveform, sampling_rate = 48000, return_tensors="pt").to(device)
                    audio_embedding = audio_model.get_audio_features(**inputs).cpu().numpy()

                precomp.append(audio_embedding)

                try:
                    if utils.should_report(last_reported, (i / length)):
                        percentage = (i / length) * 100
                        last_reported = (i / length)
                        estimated_full_duration = ((time.time() - start_time) / percentage) * 100
                        utils.inform_in_thread(vid_id, UploadProcessingStatus.AUDIO_PROCESSING, percentage, estimated_full_duration)
                except:
                    pass

        else:
            print('processing file with no audio:', vid_path)
            for i in range(fps, length, fps):

                print("precomputing audio frame", i)

                audio_embedding = np.zeros((1, 512))

                precomp.append(audio_embedding)

        np.save(precomp_audio_path, precomp)

        vidcap.release()

        video.processed_au = 1
        video.audio_embedding_model = audio_embedding_model
        session.commit()
        UploadProcessingService.inform(
            vid_id, UploadProcessingStatus.AUDIO_PROCESSING, 100)
        try:
            shutil.rmtree(temp_audio_dir)
            os.remove(audio_path)
        except Exception as err:
            print(err)
            pass

    except Exception as error:
        print("precomp_audio error", error)
        video.processing_failed = 1
        video.audio_embedding_model = audio_embedding_model        
        session.commit()
        UploadProcessingService.inform(vid_id, UploadProcessingStatus.AUDIO_PROCESSING, 100, -1, True)
