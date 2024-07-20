# -*- coding: utf-8 -*-
#
# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import srt
#from google.cloud import speech_v1p1beta1 as speech
from google.cloud import speech
from google.oauth2 import service_account
import json

def long_running_recognize(args):

    """
    Transcribe long audio file from Cloud Storage using asynchronous speech
    recognition

    Args:
      storage_uri URI for audio file in GCS, e.g. gs://[BUCKET]/[FILE]
    """

    print("Transcribing {} ...".format(args.storage_uri))

    client = speech.SpeechClient()
    
    audio = speech.RecognitionAudio(uri=args.storage_uri)
    
    diarization_config = speech.SpeakerDiarizationConfig(
       enable_speaker_diarization=True,
       min_speaker_count=2,
       max_speaker_count=10,
       )     

    if args.language_code == "en-US":
       config = speech.RecognitionConfig(
       encoding=args.encoding,
       sample_rate_hertz=args.sample_rate_hertz,
       language_code=args.language_code,
       #diarization_config=diarization_config,
       enable_word_time_offsets = True,
       model = "video",     
       use_enhanced = True,  
       audio_channel_count = args.audio_channel_count,      
       enable_automatic_punctuation = True,                         
       )

    else:
       config = speech.RecognitionConfig(
       encoding=args.encoding,
       sample_rate_hertz=args.sample_rate_hertz,
       language_code=args.language_code,
       #diarization_config=diarization_config,
       enable_word_time_offsets = True,
       audio_channel_count = args.audio_channel_count,      
       enable_automatic_punctuation = True,                         
       )      

    # Encoding of audio data sent.
    operation = client.long_running_recognize(config= config, audio=audio)
    
    response = operation.result()

    subs = []
    words_list = []
    
    for result in response.results:
        # First alternative is the most probable result
        subs = break_sentences(args, subs, result.alternatives[0])
        
        alternative = result.alternatives[0]

        for word_info in alternative.words:
            word = word_info.word
            start_time = word_info.start_time
            end_time = word_info.end_time
            speaker = word_info.speaker_tag
            words_list.append([word, start_time, end_time, speaker])

    print("Transcribing finished")
    return subs, words_list

def break_sentences(args, subs, alternative):
    firstword = True
    charcount = 0
    idx = len(subs) + 1
    content = ""

    for w in alternative.words:
        if firstword:
            # first word in sentence, record start time
            start = w.start_time
            previous_speaker = w.speaker_tag
            
        speaker = w.speaker_tag

        charcount += len(w.word)
        content += " " + w.word.strip()
        
        if ("." in w.word or "!" in w.word or "?" in w.word or
                charcount > args.max_chars or ("," in w.word and not firstword)):
            # break sentence at: . ! ? or line length exceeded or change of speaker
            # also break if , and not first word
            #or speaker != previous_speaker             
            #print("speaker", speaker, "content", content)
            subs.append(srt.Subtitle(index=idx,
                                     start=start,
                                     end=w.end_time,
                                     content=srt.make_legal_content(content)))
            firstword = True
            idx += 1
            content = ""
            charcount = 0
        else:
            firstword = False
    return subs


def write_srt(args, subs):

    print("Writing {} subtitles to: {}".format(args.language_code, args.srt_path))
    f = open(args.srt_path, 'w')
    f.writelines(srt.compose(subs))
    f.close()
    return
    
def write_tms(args, words_list):
    tms_path = args.tms_path
    
    timestamps = []
    for i in range (len(words_list)):
       timestamps.append([words_list[i][0], words_list[i][1].total_seconds()*1000, words_list[i][2].total_seconds()*1000])
       
    with open(tms_path, 'w') as f:
       json.dump(timestamps, f)

    return
    
def write_txt(args, subs):
    
    print("Writing text to: {}".format(args.txt_path))
    f = open(args.txt_path, 'w')
    for s in subs:
        f.write(s.content.strip() + "\n")  
    f.close()
    return


def write_txt_speaker(args, subs):
    txt_file = args.out_file.replace('srts_rec','txts').replace('srt','txt')
    print("Writing text to: {}".format(txt_file))
    f = open(txt_file, 'w')
    prev_speaker = subs[0].proprietary
    f.write("Speaker " + prev_speaker + ":" + "\n")
    for s in subs:
        speaker = s.proprietary
        if speaker != prev_speaker:
           f.write("\n")
           f.write("Speaker " + speaker + ":" + "\n")        
           f.write(s.content.strip() + "\n")
        else:
           f.write(s.content.strip() + "\n")  
        prev_speaker = speaker
    f.close()
    return


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--storage_uri",
        type=str,
        default="gs://cloud-samples-data/speech/brooklyn_bridge.raw",
    )
    parser.add_argument(
        "--language_code",
        type=str,
        default="en-US",
    )
    parser.add_argument(
        "--sample_rate_hertz",
        type=int,
        default=16000,
    )
    parser.add_argument(
        "--srt_path",
        type=str,
        default="subtitle",
    ) 
    parser.add_argument(
        "--txt_path",
        type=str,
        default="txt",
    )   
    parser.add_argument(
        "--tms_path",
        type=str,
        default="tms",        
    )
    parser.add_argument(
        "--max_chars",
        type=int,
        default=150,
    )
    parser.add_argument(
        "--encoding",
        type=str,
        default='LINEAR16'
    )
    parser.add_argument(
        "--audio_channel_count",
        type=int,
        default=1
    )
    args = parser.parse_args()

    subs, words_list = long_running_recognize(args)
    write_srt(args, subs)
    write_tms(args, words_list)    
    write_txt(args, subs)


if __name__ == "__main__":
    main()
