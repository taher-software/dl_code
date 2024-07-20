import torch
import time
import numpy as np
from pycaption import SRTReader
from app import db
from app.enums.upload_processing_status import UploadProcessingStatus
from app.beta.models.video import Video
from app.services.upload_processing_service import UploadProcessingService
from config import Config
from sentence_transformers import SentenceTransformer
from app.ai_processing.dialogues.stt.srt_gen import srt_gen
from app.ai_processing.dialogues.process_dialogz import topic_modeling, process_dialogue, offset_caption


import tiktoken
import openai
import json
import ast
from openai import OpenAI
openai_client = OpenAI(api_key=Config.openai_api_key)

default_prompt = """This is the transcript of a video, timestamps are expressed in seconds:
 
[TRANSCRIPT]

I want you to extract the main topics discussed and for each topic, extract the relevant clips. Here are the requirements for the extracted clips: 

- The extracted clips must not overlap. 
- Each clip duration must be under 60 seconds. 

The output format needs to be following:

[{"topic_title": "TOPIC_TITLE", "topic_id": 0,  "clips": [{"id":0, "start": START_TIMESTAMP, "end": END_TIMESTAMP, "sub_topic": SUB_TOPIC_TITLE}, {"id":1, "start": START_TIMESTAMP, "end": END_TIMESTAMP, "sub_topic": SUB_TOPIC_TITLE}....]}, {"topic_title": "TOPIC_TITLE", "topic_id":1, "clips": [{"id":0, "start": START_TIMESTAMP, "end": END_TIMESTAMP, "sub_topic": SUB_TOPIC_TITLE}, {"id":1, "start": START_TIMESTAMP, "end": END_TIMESTAMP, "sub_topic": SUB_TOPIC_TITLE}....]}....]

Don't add anything more, no explanations, nothing."""

default_prompt_semi = """
I have a topic that contains the following documents: 
[DOCUMENTS]
The topic is described by the following keywords: [KEYWORDS]
Based on the information above, extract a short topic label in the following format:
topic: <topic label>
"""

default_chapterization_prompt = """
This is the transcript of a video, timestamps are expressed in seconds: 
[TRANSCRIPT]
Using the following transcript, generate chapters defined by chapter titles and their corresponding timestamps along with a detailed summary of each chapter. Ensure each chapter has a clear title, start and end timestamps, and a summary that captures the main points discussed in that segment. Each chapter should be no less than 1 minute in duration.
The output needs to be expressed in the following format:
{{"chapter_title": "short chapter title", "chapter_id": 0, "start": start_timestamp, "end": end_timestamp, "summary": "detailed summary of the chapter"}, {"chapter_title": "short chapter title", "chapter_id": 1, "start": start_timestamp, "end": end_timestamp, "summary": "detailed summary of the chapter"}...}
Don't include Speaker A, B, C etc in the chapters description, speaker information is only here to help you. The detailed summary of the chapters does not need to mention the speakers, it needs to be a summary of what is being discussed. Make sure the output complies with JSON formatting. Don't add anything more, no explanations, nothing."""

default_summarization_prompt = """
This is the transcript of a video: 
[TRANSCRIPT]
I want you to generate a detailed summary of the transcript. Don't add anything more, no explanations, nothing."""

# please do not remove this line
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using:", device)

def read_captions(srt_path):

    srt_text = open(srt_path).read()
    srt = SRTReader().read(srt_text, lang="en")
    captions = srt.get_captions("en")

    return captions

def precompute_dialogues(vid_id, file_type = 'video', embedding_size = 384):
    """ Precomputes embeddings for each caption (using Language model of choice) of a srt subtitle file and saves the embeddings in numpy array
        Arguments:
            srt_path: path of the srt file to be processed
            precomp_sub_path: path of the output file
    """
    video = Video.query.get(vid_id)
    if video is None:
        return
    
    language = video.language
    source_lang = language.split('-')[0].lower()
    
    base_path = video.base_path

    srt_folder = f"{base_path}{video.srt_folder}"
    srt_path = f"{srt_folder}_{source_lang}.srt"       
    precomp_sub_path = f"{base_path}{video.precomp_sub_path}"   
    tms_path = f"{base_path}{video.timestamps_path}"
    topics_path = f"{video.base_path}{video.topics_path}"
    chapters_path = f"{video.base_path}{video.chapters_path}"
    summary_path = f"{video.base_path}{video.summary_path}"        
           
    srt_gen(vid_id, file_type, config_provider = Config.stt_provider, resize = False, translate = False)

    dialogue_embedding_model = Config.DIALOGUE_EMBEDDING_MODEL
        
    if dialogue_embedding_model == "ALL-MINILM-L6-V2":
        print("Using ALL-MINILM-L6-V2 model")
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
       
    with open(tms_path, 'r') as file:
        words = json.load(file)
        no_dialogues = not bool(words)
    
    if no_dialogues:
        precomp = np.zeros(embedding_size) 
         
    else:  
        caption_data = read_captions(srt_path)

        precomp = []

        for i in range(len(caption_data)-1):
            if dialogue_embedding_model in ["ALL-MINILM-L6-V2"]:
                print("precomputing caption", i)
                caption = caption_data[i].get_text()
                precomp.append(model.encode(caption))

    np.save(precomp_sub_path, precomp)
    
    video.processed_srt_gen = 1
    video.dialogue_embedding_model = dialogue_embedding_model    
    db.session.commit()
    UploadProcessingService.inform(vid_id, UploadProcessingStatus.TRANSCRIBING, 100)
    
    if no_dialogues:
        with open(topics_path, 'w') as file:
            file.write('')    
        with open(chapters_path, 'w') as file:
            file.write('')    
        with open(summary_path, 'w') as file:
            file.write('')                                              
    else:
        precompute_topics(vid_id)
        precompute_chapters(vid_id)

def num_tokens_from_string(string):
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo-16k")
    num_tokens = len(encoding.encode(string))
    return num_tokens

def gpt_call(prompt, model= "gpt-3.5-turbo-16k"):
    messages=[{"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}]
    response = openai_client.chat.completions.create(model=model,messages=messages)
    topics = response.choices[0].message.content
    
    return topics

def fill_prompt(default_prompt, transcript):
    prompt = default_prompt.replace("[TRANSCRIPT]", transcript)
    return prompt
    
def fill_prompt_semi(default_prompt, keywords, docs):
    prompt = default_prompt.replace("[KEYWORDS]", keywords)
    to_replace = ""
    for doc in docs:
        to_replace += f"- {doc[:255]}\n"
    prompt = prompt.replace("[DOCUMENTS]", to_replace)
    return prompt

def topic_chatgpt_semi(prompt, model = "gpt-3.5-turbo"): 
    messages=[{"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}]
    response = openai_client.chat.completions.create(model=model,messages=messages)
    label = response.choices[0].message.content.strip().replace("topic: ", "").replace("Topic: ", "").replace("Topic label: ", "").replace("Topic Label: ", "")
    
    return label
        
def precompute_topics_auto (vid_id): 
    """ Precomputes the topics inside a video transcript and saves them in JSON format
        Arguments:
            vid_id: id of the video to be processed
    """
    video = Video.query.get(vid_id)
    transcript_path = f"{video.base_path}{video.speaker_ts_path}"  
    topics_path = f"{video.base_path}{video.topics_path}"
    
    with open(transcript_path, "r") as file:
        transcript = file.read()
    
    token_count = num_tokens_from_string(transcript)
    print("Number token:", token_count)
    if token_count > 16000:
        topics = []
        chunks = [transcript[i:i+16000] for i in range(0, len(transcript), 16000)]
        prev_topic_id = 0
        for i, chunk in enumerate(chunks):
            if num_tokens_from_string(chunk)<100:
                pass
            else:
                sub_prompt = fill_prompt(default_prompt, chunk)
                sub_topics = gpt_call(sub_prompt)
                sub_topics = json.loads(sub_topics)
                for topic in sub_topics:
                    topic['topic_id'] = prev_topic_id + 1
                    prev_topic_id = topic['topic_id']
                    topics.append(topic)        
    else:
        prompt = fill_prompt(default_prompt, transcript)
        topics = ast.literal_eval(gpt_call(prompt))  
        
    json_string = json.dumps(topics, indent=2)
    json_topics = json.loads(json_string)
    
    with open(topics_path, 'w') as f:
        json.dump(json_topics, f)
    print("Writing topics to: {}".format(topics_path))
    return json_topics
 

    
def offset_caption(captions, target_index, offset_seconds):
    target_caption = captions[target_index]

    # Calculate new start and end values with offset
    new_start = target_caption.start - offset_seconds 
    new_end = target_caption.end + offset_seconds 
    
    # Find the closest captions for alignment
    align_start_caption = captions[0]
    align_end_caption = captions[-1]
    align_start_index = 0
    align_end_index = -1

    for i, caption in enumerate(captions):
        if caption.end <= new_start and (align_start_caption is None or caption.end > align_start_caption.end):
            # Check if the previous caption ends with ".", "!", or "?"
            if i > 0 and captions[i - 1].get_text()[-1] in (".", "!", "?"):
                align_start_caption = caption
                align_start_index = i

        if caption.start >= new_end and (align_end_caption is None or caption.start < align_end_caption.start):
            # Check if the current caption ends with "."
            if caption.get_text()[-1] == ".":
                align_end_caption = caption
                align_end_index = i
                
    aligned_captions = captions[align_start_index : align_end_index]                
    sub_topic = " ".join(caption.get_text() for caption in aligned_captions)

    return align_start_caption, align_end_caption, sub_topic
    
def precompute_topics_semi (vid_id, offset = 15000000): 
    """ Precomputes the topics inside a video transcript and saves them in JSON format
        Arguments:
            vid_id: id of the video to be processed
    """
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    
    video = Video.query.get(vid_id)    
    topics_path = f"{video.base_path}{video.topics_path}"

    language = video.language
    source_lang = language.split('-')[0].lower()
    
    srt_folder = f"{video.base_path}{video.srt_folder}"
    srt_path = f"{srt_folder}_{source_lang}.srt" 
    
    captions = read_captions(srt_path)
    topics = topic_modeling ([vid_id])
    
    json_topics = []
    topic_id = 0
    
    for topic in topics[:10]: 
        keywords = topic[0]
        repdoc = topic[1]

        try: 
           print("using chatgpt to summarize topic")
           topic_label = topic_chatgpt_semi(fill_prompt_semi(default_prompt_semi, keywords, repdoc))
        except:
           topic_label = ','.join(keywords.split(',')[:5])
                   
        clips = []
        input_features = model.encode(topic_label)
        input_features = input_features.T  
        probs = process_dialogue(vid_id, input_features)
        probs = sorted(probs, key=lambda x: x[0], reverse=True)

        prev_frame = []
        filtered_probs = []

        sp_index = 0
        j = 0

        while sp_index < 5:

            if j == len(probs):
                break

            caption = probs[j][1]

            limits = []

            index = int((caption.start / 1000 / 1000 +
                    caption.end / 1000 / 1000)/2)

            for k in range(len(prev_frame)):

                limits.append(abs(index-prev_frame[k]))

            if  any(lim < 30 for lim in limits):
                j = j + 1

                pass

            else:

                filtered_probs.append(probs[j])
                prev_frame.append(index)

                j = j + 1
                sp_index = sp_index + 1
            
        clip_id = 0
        for prob in filtered_probs[:5]:
            target_index = prob[3]
            caption_start, caption_end, text = offset_caption(captions, target_index, offset)
            summarize_prompt = "This is an extract of a transcription. Summarize in a very short title (ideally max 10 words): " + text 
            try:
               print("using gpt to summarize captions")
               sub_topic = gpt_call(summarize_prompt, model= "gpt-3.5-turbo")
            except:
               sub_topic = text[:15] + '...'
            start = caption_start.start/1000000
            end = caption_end.end/1000000

            clips.append ({'id':clip_id, 'start': start, 'end': end, 'sub_topic': sub_topic})
            clip_id +=1

        json_topics.append({'topic_title': topic_label, 'topic_id': topic_id, 'clips': clips})
        topic_id +=1   
        
    with open(topics_path, 'w') as f:
        json.dump(json_topics, f)
    print("Writing topics to: {}".format(topics_path))  
    return json_topics
    
def precompute_topics(vid_id):
    video = Video.query.get(vid_id)
    
    try:
        topics = precompute_topics_semi(video.id)    
    except Exception as err:
        print("Semi auto topics failed, Using auto topics")
        print(err)
        try:
            topics = precompute_topics_auto(video.id)
        except Exception as err:
            print("Auto topics failed")
            print(err)
            return

    video.processed_topics = 1
    video.has_topics = len(topics) > 0
    db.session.commit()
    if video.get_clips == 1:
        UploadProcessingService.auto_tiktok(vid_id)
        
        
def precompute_chapters(vid_id, max_retries = 5, retry_delay = 2):
    print("Generating chapters for video id:", vid_id)
    video = Video.query.get(vid_id)
    transcript_path = f"{video.base_path}{video.speaker_ts_path}"
    chapters_path = f"{video.base_path}{video.chapters_path}"
    
    with open(transcript_path, "r") as file:
        transcript = file.read()
 
    chapterization_prompt = fill_prompt(default_chapterization_prompt, transcript)
    token_count = num_tokens_from_string(chapterization_prompt)

    print("Number token:", token_count)
               
    for attempt in range(max_retries):
        try:
            if token_count > 16000:
                response = gpt_call(chapterization_prompt, model="gpt-4o")
            else:
                response = gpt_call(chapterization_prompt)
                
            chapters = ast.literal_eval(response)
            
            with open(chapters_path, 'w') as f:
                json.dump(chapters, f)
                
            print("Writing chapters to: {}".format(chapters_path))
            video.processed_chapters = 1
            db.session.commit()
            return
        
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                print("Max retries reached. Failing.")
                raise e

def precompute_summary(vid_id):
    print("Generating summary for video id:", vid_id)
    video = Video.query.get(vid_id)
    transcript_path = f"{video.base_path}{video.txt_path}"
    summary_path =  f"{video.base_path}{video.summary_path}"    
    
    with open(transcript_path, "r") as file:
        transcript = file.read()
 
    summarization_prompt = fill_prompt(default_summarization_prompt, transcript)
    token_count = num_tokens_from_string(summarization_prompt)

    print("Number token:", token_count)
               
    if token_count > 16000:    
        summary = ast.literal_eval(gpt_call(summarization_prompt, model = "gpt-4-turbo"))   
    else:
        summary = ast.literal_eval(gpt_call(summarization_prompt))         

    with open(summary_path, 'w') as f:
        f.write(summary)
    print("Writing summary to: {}".format(summary_path)) 
