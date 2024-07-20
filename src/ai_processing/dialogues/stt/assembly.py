from pycaption import SRTReader
import json
import requests

def read_captions(srt_path, lang):

    srt_text = open(srt_path).read()
    srt = SRTReader().read(srt_text, lang=lang)
    captions = srt.get_captions(lang)
    
    return captions
    
def read_file(filename):
    with open(filename, 'rb') as _file:
        while True:
            data = _file.read(5242880)
            if not data:
                break
            yield data

def assembly (api_key, source_lang, audio_path, srt_path, txt_path, speaker_ts_path, tms_path):

    print("Using Assembly")
    # store global constants
    headers = {"authorization": api_key,"content-type": "application/json"}
    transcript_endpoint = "https://api.assemblyai.com/v2/transcript"
    upload_endpoint = 'https://api.assemblyai.com/v2/upload'    

    # upload our audio file
    upload_response = requests.post(upload_endpoint, headers=headers, data=read_file(audio_path))
    print('Audio file uploaded')
 
    # send a request to transcribe the audio file
    transcript_request = {'audio_url': upload_response.json()['upload_url'], "speaker_labels": True, "auto_chapters": False}
    transcript_response = requests.post(transcript_endpoint, json=transcript_request, headers=headers)
    print('Transcription Requested')
    id = transcript_response.json()['id'] 
         
    # set up polling
    polling_response = requests.get(transcript_endpoint+"/"+ id, headers=headers)
    # if our status isn’t complete, sleep and then poll again
    print("File is processing")	 
    while polling_response.json()['status'] != 'completed':
        polling_response = requests.get(transcript_endpoint+"/"+id, headers=headers)
    print("Processing finished")	 	 
    words = polling_response.json()['words']
    timestamps = []

    for i in range (len (words)):
        word = words[i]
        timestamps.append([word['text'], word['start'], word['end'], word['speaker']])

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
        srt_endpoint = "https://api.assemblyai.com/v2/transcript/"+ id + "/srt"
        srt_response = requests.get(srt_endpoint, headers=headers)

        with open(srt_path, 'w') as f:
            f.write(srt_response.text)       

        print("Writing {} subtitles to: {}".format(source_lang, srt_path))

        caption_data = read_captions(srt_path, source_lang)
         
        with open(txt_path, 'w') as f:
            for cap in caption_data:
                f.write(cap.get_text()+ "\n")
        print("Writing text to: {}".format(txt_path)) 

        with open(speaker_ts_path, 'w') as f:    
            speaker_prev = words[0]['speaker']
            sentence = [" "]
            caption_start = 0
            for i in range (len (words)):
                word = words[i]
                speaker = word['speaker']
                start = word['start']/1000.
                end = word['end']/1000.
                text = word['text']
                if speaker == speaker_prev and sentence[-1][-1] not in ['?', '!', '.']:
                    sentence.append (text)
                else:
                    sentence = ' '.join(sentence)
                    f.write("start: " + str(caption_start) + "," + "speaker " + speaker_prev + ": " + sentence + "\n")
                    speaker_prev = speaker
                    sentence = []
                    sentence.append (text)
                    caption_start = end
