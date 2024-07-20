from rev_ai import apiclient
from pycaption import SRTReader
import json

def read_captions(srt_path, lang):

    srt_text = open(srt_path).read()
    srt = SRTReader().read(srt_text, lang=lang)
    captions = srt.get_captions(lang)
    
    return captions

def rev (api_key, source_lang, audio_path, srt_path, txt_path, speaker_ts_path, tms_path):
    print("Using REV")   
    # create your client
    client = apiclient.RevAiAPIClient(api_key)
    job = client.submit_job_local_file(filename=audio_path, language= source_lang)
    print("Transcribing {} ...".format(audio_path))
         
    while "TRANSCRIBED" not in str(client.get_job_details(job.id).status):
          pass
               
    print("Transcribing finished") 
      
    # as text, contains speaker id
    #transcript_text = client.get_transcript_text(job.id)
    # as json, contains speaker id
    transcript_json = client.get_transcript_json(job.id)
    # or as a python object
    #transcript_object = client.get_transcript_object(job.id)

    monologues = transcript_json['monologues']
    timestamps = []
    for monologue in monologues:
       elements = monologue ['elements']
       speaker = monologue ['speaker']
       for i in range (len (elements)):
          element = elements[i]
          if element['type'] =='text': 
             timestamps.append([element['value'], element['ts']*1000, element['end_ts']*1000, speaker])
          elif element['value'] != ' ':
             timestamps[-1][0] = timestamps[-1][0] + element['value'] 

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
        #get srt_file 
        captions = client.get_captions(job.id)
         
        captions_lst = captions.split('\n\n')[:-1]
        subs = []
        for cap in captions_lst:
           cap_split = cap.replace('\r', '').split('\n')
           if len(cap_split)>3:
               joined = ' '.join(cap_split[2:])
               cap_split[2:] = [joined]
               cap_split[2] = ' ' + cap_split[2] 
               new_cap = '\n'.join(elem for elem in cap_split) 
               subs.append(new_cap)
           else:
               cap_split[2] = ' ' + cap_split[2]
               new_cap = '\n'.join(elem for elem in cap_split)
               subs.append(new_cap)
      
        subs = '\n\n'.join(subs)
      
        with open(srt_path, 'w') as f:
            f.write(subs)
            
        caption_data = read_captions(srt_path, source_lang)
        print("Writing {} subtitles to: {}".format(source_lang, srt_path))
        
        with open(txt_path, 'w') as f:
            for cap in caption_data:
                f.write(cap.get_text()+ "\n")
        print("Writing text to: {}".format(txt_path))   
    
        with open(speaker_ts_path, 'w') as f:    
            speaker_prev = str(timestamps[0][3])
            sentence = [" "]
            caption_start = 0
            for i in range (len (timestamps)):
                word = timestamps[i]
                speaker = str(word[3])
                start = word[1]/1000.
                end = word[2]/1000.
                text = word[0]
                if speaker == speaker_prev and sentence[-1][-1] not in ['?', '!', '.']:
                    sentence.append (text)
                else:
                    sentence = ' '.join(sentence)
                    f.write("start: " + str(caption_start) + "," + "speaker " + speaker_prev + ": " + sentence + "\n")
                    speaker_prev = speaker
                    sentence = []
                    sentence.append (text)
                    caption_start = end
