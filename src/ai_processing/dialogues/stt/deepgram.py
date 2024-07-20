from pycaption import SRTReader
from deepgram import Deepgram
import json

from app.ai_processing.dialogues.stt.json2srt_txt import json2srt_txt

def read_captions(srt_path, lang):

    srt_text = open(srt_path).read()
    srt = SRTReader().read(srt_text, lang=lang)
    captions = srt.get_captions(lang)
    
    return captions
    
def deepgram (api_key, source_lang, audio_path, srt_path, txt_path, speaker_ts_path, tms_path):

    print("Using Deepgram")
    print("Transcribing:", audio_path)

    # Initialize the Deepgram SDK
    deepgram = Deepgram(api_key)
    audio = open(audio_path, 'rb')

    # Set the source
    source = {
      'buffer': audio,
      'mimetype': 'audio/wav'
    }

    # Send the audio to Deepgram and get the response
    response = deepgram.transcription.sync_prerecorded(
      source,
      {
        'punctuate': True,
        'diarize' : True, 
        'paragraphs': True,
        'numerals' : True, 
        'model': 'general'
      }
    )
    print("Finished transcribing")
    
    words_orig = response["results"]["channels"][0]["alternatives"][0]["words"]
    timestamps = []

    for word in words_orig:
        timestamps.append([word['punctuated_word'], word['start']*1000, word['end']*1000, word['speaker']])

    with open(tms_path, 'w') as f:
        json.dump(timestamps, f)
    print("Writing timestamps to: {}".format(tms_path))  

    if not timestamps:  
        with open(srt_path, 'w') as file:
            file.write('')     
        with open(txt_path, 'w') as file:
            file.write('') 
        with open(speaker_ts_path, 'w') as file:
            file.write('') 
    
    else:        
        json2srt_txt (tms_path, srt_path, txt_path)
        print("Writing srt to: {}".format(srt_path)) 
        print("Writing txt to: {}".format(txt_path)) 
    
        with open(speaker_ts_path, 'w') as f:    
            speaker_prev = str(words_orig[0]['speaker'])
            sentence = [" "]
            caption_start = 0
            for i in range (len (words_orig)):
                word = words_orig[i]
                speaker = str(word['speaker'])
                start = word['start']
                end = word['end']
                text = word['punctuated_word']
                if speaker == speaker_prev and sentence[-1][-1] not in ['?', '!', '.']:
                    sentence.append (text)
                else:
                    sentence = ' '.join(sentence)
                    f.write("start: " + str(caption_start) + "," + "speaker " + speaker_prev + ": " + sentence + "\n")
                    speaker_prev = speaker
                    sentence = []
                    sentence.append (text)
                    caption_start = end
