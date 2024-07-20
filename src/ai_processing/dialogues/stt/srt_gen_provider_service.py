from decimal import Decimal
from app.ai_processing.dialogues.stt.srt_gen_provider_types import SrtGenProviderTypes
from app.enums.upload_processing_status import UploadProcessingStatus
from app.beta.models.video import Video
from app.ai_processing.dialogues.stt.assembly import assembly
from app.ai_processing.dialogues.stt.rev import rev
from app.ai_processing.dialogues.stt.deepgram import deepgram
from app.utils import utils
from config import Config
from google.cloud import storage
from scipy.io.wavfile import read as read_wav
from time import sleep

import os


class SrtGenProvidersService:

    # The api keys should not be stored in code
    ASSEMBLY_API_KEY = Config.assembly_api_key
    DEEPGRAM_API_KEY = Config.deepgram_api_key
    REV_API_KEY = Config.rev_api_key

    test = False
    provider = None
    video = None
    base_path = None
    video_path = None
    tms_path = None
    audio_path = None
    precomp_sub_path = None
    language = None
    vid_name = None
    source_lang = None
    video = None
    sampling_rate = None
    storage_client = None
    speaker_ts_path = None

    def __init__(self, provider: SrtGenProviderTypes, video: Video, ):
        self.test = Config.TEST_WORKERS == '1'
        self.video = video
        self.provider = provider
        self.base_path = video.base_path
        self.video_path = f"{self.base_path}{video.path}"
        self.tms_path = f"{self.base_path}{video.timestamps_path}"
        self.audio_path = f"{self.base_path}{video.audio_path.replace('.mp3', '.wav')}"
        self.audio_name = self.audio_path.split('/')[-1]
        self.precomp_sub_path = f"{self.base_path}{video.precomp_sub_path}"
        self.language = video.language
        self.vid_name = video.file_name
        self.source_lang = self.language.split('-')[0].lower()
        self.srt_path = f"{self.base_path}{video.srt_folder}_{self.source_lang}.srt"
        self.txt_path = f"{self.base_path}{video.txt_folder}_{self.source_lang}.txt"
        self.speaker_ts_path = f"{self.base_path}{video.speaker_ts_path}"
        self.storage_client = storage.Client()

    def generate_transcription(self):
        # Create a wav file from video / audio
        wav_com = (
            f"ffmpeg -y -hide_banner -loglevel warning -i {str(self.video_path)} "
            f"-ac 1 {str(self.audio_path)}"
        )
        os.system(wav_com)

        if self.test:
            self.handle_test()
        elif self.provider == SrtGenProviderTypes.GCP:
            self.handle_gcp()
        elif self.provider == SrtGenProviderTypes.REV:
            self.handle_rev()
        elif self.provider == SrtGenProviderTypes.ASSEMBLY:
            self.handle_assembly()
        elif self.provider == SrtGenProviderTypes.DEEPGRAM:
            self.handle_deepgram()

        try:
            utils.inform_in_thread(
                vid_id=self.video.id,
                status=UploadProcessingStatus.TRANSCRIBING,
                progress=99
            )
        except:
            pass

    def resize_video(self, resize):
        if resize and str(self.video.preproc) == '1':
            vid_path_temp = self.video_path.split('.mp4')[0] + 'bis' + '.mp4'
            os.system("ffmpeg -i " + self.video_path +
                      " -vf scale=1280:720 -crf 23 " + vid_path_temp)
            os.remove(self.video_path)
            os.rename(vid_path_temp, self.video_path)

    def translate_video(self, translate):
        if translate:
            """ Upload text """
            temp_folder = self.video.temp_folder
            srt_temp_folder = temp_folder + self.vid_name + '/'
            if not os.path.exists(srt_temp_folder):
                os.makedirs(srt_temp_folder)
            lang_list = ['en', 'fr', 'es', 'sv', 'no', 'de', 'it']
            target_lang = [
                lang for lang in lang_list if lang != self.source_lang]
            bucket = self.storage_client.get_bucket(f'txtstobetrans')
            blob = bucket.blob(
                f"{str(self.vid_name)}/{self.txt_path.split('/')[-1]}")
            blob.upload_from_filename(
                filename=str(self.txt_path), num_retries=10)

            lg_com = (
                f"python3.8 {Config.ROOT_APP_FOLDER}/app/ai_processing/dialogues/stt/translate_txt.py "
                f"--project_id imaginario-tr --input_uri gs://txttobetrans/{str(self.vid_name)}/{str(self.txt_name)} "
                f"--output_uri gs://transtxt/{str(self.vid_name)}/ "
                f"--source_lang {self.source_lang} "
                f"--target_lang {','.join(lang for lang in target_lang)}"
            )
            os.system(lg_com)
            """ Get translated temps  """
            bucket = self.storage_client.get_bucket(f'transtxts')
            for b in bucket.list_blobs():
                if f"{str(self.vid_name)}/" in b.name:
                    blob = bucket.blob(f"{b.name}")
                    blob.download_to_filename(
                        filename=f"{srt_temp_folder}/{b.name.split('/')[-1]}")

            srt_com = (
                f"python3.8 {Config.ROOT_APP_FOLDER}/app/ai_processing/dialogues/stt/txt2srt.py --temp_folder {srt_temp_folder} "
                f"--srt {self.srt_path} "
                f"--index {srt_temp_folder}index.csv"
            )
            os.system(srt_com)

    def remove_unused_files_from_bucket(self):
        bucket = self.storage_client.get_bucket(f'speechestotranscript')
        for b in bucket.list_blobs():
            if f"{str(self.vid_name)}/" in b.name:
                blob = bucket.blob(f"{b.name}")
                blob.delete()

        bucket = self.storage_client.get_bucket(f'txtstobetrans')
        for b in bucket.list_blobs():
            if f"{str(self.vid_name)}/" in b.name:
                blob = bucket.blob(f"{b.name}")
                blob.delete()
        bucket = self.storage_client.get_bucket(f'transtxts')
        for b in bucket.list_blobs():
            if f"{str(self.vid_name)}/" in b.name:
                blob = bucket.blob(f"{b.name}")
                blob.delete()

    # Uses this case to minimise cost during testing
    def handle_test(self):
        utils.inform_in_thread(
            vid_id=self.video.id,
            status=UploadProcessingStatus.TRANSCRIBING,
            progress=0,
            estimate=70.00
        )

        sleep(60)

        with open(self.txt_path, "w") as f:
            f.write('test')
        with open(self.tms_path, "w") as f:
            f.write('[["test", 1000, 2000]]')
        with open(self.srt_path, "w") as f:
            f.write('1\n'
                    '00:00:01,000 --> 00:00:02,00\n'
                    'test\n')

    def handle_gcp(self):
        bucket = self.storage_client.get_bucket('speechestotranscript')
        sampling_rate, data = read_wav(self.audio_path)
        """ Upload Audio File """
        blob = bucket.blob(f"{str(self.vid_name)}/{self.audio_name}")
        blob.upload_from_filename(filename=str(
            self.audio_path), num_retries=10, timeout=1200)

        utils.inform_in_thread(
            vid_id=self.video.id,
            status=UploadProcessingStatus.TRANSCRIBING,
            progress=0,
            estimate=Decimal(self.video.duration) * Decimal(0.1)
        )

        tr_com = (
            f"python3.8 {Config.ROOT_APP_FOLDER}/app/ai_processing/dialogues/stt/gcp.py "
            f"--storage_uri gs://speechtotranscript/{str(self.vid_name)}/{str(self.audio_name)} "
            f"--sample_rate_hertz {str(sampling_rate)} "
            f"--srt_path {str(self.srt_path)} "
            f"--tms_path {str(self.tms_path)} "
            f"--txt_path {str(self.txt_path)} "
            f"--language_code {str(self.language)}"
        )
        os.system(tr_com)

    def handle_rev(self):
        utils.inform_in_thread(
            vid_id=self.video.id,
            status=UploadProcessingStatus.TRANSCRIBING,
            progress=0,
            estimate=Decimal(self.video.duration) * Decimal(4)
        )
        rev(
            self.REV_API_KEY,
            self.source_lang,
            self.audio_path,
            self.srt_path,
            self.txt_path,
            self.speaker_ts_path,
            self.tms_path
        )

    def handle_assembly(self):
        utils.inform_in_thread(
            vid_id=self.video.id,
            status=UploadProcessingStatus.TRANSCRIBING,
            progress=0,
            estimate=Decimal(self.video.duration) * Decimal(0.3)
        )
        assembly(
            self.ASSEMBLY_API_KEY,
            self.source_lang,
            self.audio_path,
            self.srt_path,
            self.txt_path,
            self.speaker_ts_path,
            self.tms_path
        )

    def handle_deepgram(self):
        utils.inform_in_thread(
            vid_id=self.video.id,
            status=UploadProcessingStatus.TRANSCRIBING,
            progress=0,
            estimate=Decimal(self.video.duration) * Decimal(0.05)
        )
        deepgram(
            self.DEEPGRAM_API_KEY,
            self.source_lang,
            self.audio_path,
            self.srt_path,
            self.txt_path,
            self.speaker_ts_path,
            self.tms_path
        )
