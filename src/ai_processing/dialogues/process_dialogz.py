from collections import Counter
import json
import logging
import pickle
from app.ai_processing.dialogues.wordcloud.wordcloud import WordCloud
from app.ai_processing.dialogues.stt.json2srt_txt import json2srt_txt
from app.beta.models.image import Img
from bertopic import BERTopic
from config import Config
from gensim.parsing.preprocessing import STOPWORDS
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import *
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from pycaption import SRTReader
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

import contractions
import datetime
import numpy as np
import nltk
import os
import pathlib
import re
import string
import torch
from flask_sse import sse

from openai import OpenAI
openai_client = OpenAI(api_key=Config.openai_api_key)

# please do not remove this line
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using:", device)

sw_gensim = STOPWORDS
sw_nltk = stopwords.words('english')
sw_sklearn = ENGLISH_STOP_WORDS

default_combine_prompt = """
I have the following topic descriptions: 

[TOPICS]

Generate a new topic description that briefly summarizes the information contained in all topics, in the following format:
topic: <topic label>
"""

def get_files(folder, sub=False):
    """ Retrieves the paths of file in a folder
        Arguments:
            folder: folder full path
            sub: variable to switch between .srt and .mp4 retrieval
        returns:
            filtered_files: paths of files in the folder
    """

    filtered_files = []
    for path, _, files in os.walk(folder):
        files.sort(key=lambda var: [int(x) if x.isdigit(
        ) else x for x in re.findall(r'[^0-9]|[0-9]+', var)])
        for file in files:
            if not sub:
                if '.mp4' in file:
                    full_path = os.path.join(path, file)
                    filtered_files.append(full_path)
            else:
                if '.srt' in file:
                    full_path = os.path.join(path, file)
                    filtered_files.append(full_path)

    return filtered_files
    
    
def topic_modeling(vids, chat=True, use_precomp = False):
    """ Extracts topics from a sequence of videos
        Arguments:
            vids: list of videos ids to process
        return: list of topics, each topic being itself a list of tuples. Each tuple will consist in a keyword and its associated weight. 
    """
    from app.beta.models.video import Video
    captions = []
    embeddings = []
    for vid_id in vids:
        video = Video.query.get(vid_id)
        vid_name = video.path
        extension = pathlib.Path(vid_name).suffix
        vid_name = vid_name.replace(extension, '')
        srt_path = f"{video.base_path}{video.srt_folder}_en.srt"
        
        try:
            caption_data = read_captions(srt_path)
            if use_precomp:
                print('using precomp')
                precomp_path = f"{video.base_path}{video.precomp_sub_path}"
                caption_embeddings = np.load(precomp_path, allow_pickle=True)
            
            for j in range(0, len(caption_data), 1):
                caption = ' '.join(preprocess(caption_data[j].get_text()))
                if len(caption) > 20:
                    captions.append(caption)
                if use_precomp:
                    embedding = caption_embeddings[j]
                    embeddings.append(embedding)
        except:
            print('unable to retrieve captions')
    if len(captions) == 0:
        return []

    print('len(captions)', len(captions))
    # BERTopic fail to identify the topics when given a low number of documents, thus, we need to duplicate the data as a workaround
    # Workaround if not enough documents https://github.com/MaartenGr/BERTopic/issues/97 , https://github.com/MaartenGr/Concept/issues/5
    if len(captions) < 300:
        captions.extend(captions)

    #https://www.sbert.net/docs/pretrained_models.html#sentence-embedding-models
    if use_precomp:
       topic_model = BERTopic(top_n_words=10, min_topic_size = 3)
       _, probs = topic_model.fit_transform(captions, np.array(embeddings))
    else:
       topic_model = BERTopic(embedding_model="all-MiniLM-L6-v2", top_n_words=10, min_topic_size = 3)
       _, probs = topic_model.fit_transform(captions)

    topics = topic_model.get_topic_info()

    topic_list = []
    
    for i in range(len(topics)-1):
        topic = topic_model.get_topic(i)
        repdoc = topic_model.get_representative_docs(i)
        keywords = ','.join([keyword[0] for keyword in topic])
        topic_label = [keywords, repdoc]       

        topic_list.append(topic_label)

    return topic_list
    
    
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
    
    
def combine_prompt(default_combine_prompt, topics):
    prompt = default_combine_prompt.replace("[TOPICS]", topics)
    return prompt
    
    
def combine_topics(prompt, model= "gpt-3.5-turbo-16k"):
    messages=[{"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}]
    response = openai_client.chat.completions.create(model=model,messages=messages)
    topic = response.choices[0].message.content
    
    return topic
    
    
def cross_asset_topics(topics_per_video, threshold = 0.4, include_non_visited = False):
    from app.beta.models.video import Video    
    topics_flat_map = [topic for video in topics_per_video for topic in video]
    
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    embeddings = {topic['topic_title']: model.encode(topic['topic_title']) for topic in topics_flat_map}
    vid_ids = {topic['topic_title']: topic['video_id'] for topic in topics_flat_map}

    cross_asset_topic_list = []
    visited = []
        
    topic_titles = list(embeddings.keys())
    embedding_matrix = np.array([embeddings[topic_title] for topic_title in topic_titles])        

    similarity_matrix = cosine_similarity(embedding_matrix)

    for i in range(len(topic_titles)-1):
        topic_title = topic_titles[i]
        
        if topic_title in visited: 
            pass
        else:
            topic_sims = similarity_matrix[i][i+1:]
            similar_topic_indices = list(*np.where(topic_sims > threshold))
            if len(similar_topic_indices) > 0:
                similar_topic_titles = [topic_titles[index + i + 1] for index in similar_topic_indices] + [topic_title]
                similarity_values = [topic_sims[index] for index in similar_topic_indices]
                similar_vid_ids = list(set([vid_ids[similar_topic_title] for similar_topic_title in similar_topic_titles]))
                
                print("Merging topics:", similar_topic_titles, "with similarities:", similarity_values)                    
                combined_prompt = combine_prompt(default_combine_prompt, '\n'.join(similar_topic_titles))
                new_topic_title = combine_topics(combined_prompt).replace("topic: ", "")
                print("New generated topic:", new_topic_title)            
                visited.extend(similar_topic_titles)
        
                clips = []
                input_features = model.encode(new_topic_title)
                input_features = input_features.T

                probs = []
                for vid_id in similar_vid_ids:
                    prob = process_dialogue(vid_id, input_features)
                    probs = probs + prob

                raw_clips = get_clips(probs = probs, use_caption_offset = True)

                clip_id = 0
                for clip in raw_clips:
                    start, end = clip[1], clip[2]
                    vid_id = clip[3]
                    clips.append ({'id':clip_id, 'start': start, 'end': end, 'sub_topic': 'Unnamed clip', 'vid_id': vid_id})
                    clip_id +=1
                
                new_topic = {'topic_title': new_topic_title, 'clips': clips}
            
                cross_asset_topic_list.append(new_topic)  

    if include_non_visited:
        non_visited_topics = [topic for topic in topics_flat_map if topic['topic_title'] not in visited]
        cross_asset_topic_list.extend(non_visited_topics)
    
    for topic_id, topic in enumerate(cross_asset_topic_list):
        topic['topic_id'] = topic_id
    
    return cross_asset_topic_list

def topic2wc(frequencies, wc_path):
    wc = WordCloud(background_color=None, max_words=30)
    wc.generate_from_frequencies(frequencies)
    wc.to_file(wc_path)

def mini_srt(start, end, tms_path, srt_path, txt_path):
    """ Extract portion of srt files
        Arguments:
            start: start timestamp
            end: end timestamp
            tms_path: word timestamps file path
            srt_path: path where to save the portion of srt
    """
    json2srt_txt(tms_path, srt_path, txt_path, start=start, end=end)

def read_captions(srt_path):

    srt_text = open(srt_path).read()
    srt = SRTReader().read(srt_text, lang="en")
    captions = srt.get_captions("en")

    return captions


def preprocess(caption):
    """This function will apply NLP preprocessing"""

    # convert to lowercase
    caption = caption.lower()

    # tokenize
    caption = nltk.word_tokenize(caption)

    # expand contractions
    caption = [contractions.fix(word) for word in caption]

    # remove punctuation
    caption = [word for word in caption if word not in string.punctuation]

    # remove numbers
    caption = [re.sub("[^a-zA-Z]+", " ", word) for word in caption]

    # lemmatization
    caption = [WordNetLemmatizer().lemmatize(word, wordnet.NOUN)
               for word in caption]

    # remove short words
    caption = [word.strip() for word in caption if len(word.strip()) >= 3]

    # remove stopwords
    stopwords = [sw for sw in sw_nltk] + \
        [sw for sw in sw_gensim] + [sw for sw in sw_sklearn]
    stopwords.extend(["happen", "mean", "likely", "happening", "get", "got", "talk", "talking", "person", "looking", "look", "people", "good", "stuff", "'re", "'ve", "n't", "wa", "think", "like",
                     "really", "yeah", "thing", "know", "knew", "doe", "n t", "could", "would", "lot", "went", "guy", "little", "said", "hey", "way", "getting", "happened", "going", "come", "coming", "gon", "came", "okay", "right", "gotten", "saying", "yes", "no", "shown", "brought", "speaking", "happens", "actually", "maybe", "want", "time", "usually", "probably", "especially", "obviously", "need", "sure", "thought", "absolutely", "different", "basically", "fuck", "fucking"])
    caption = [word for word in caption if word not in stopwords]

    # return ' '.join(caption)
    return caption
  

def process_dialogue(vid_id, query_embedding):
    from app.beta.models.video import Video
    video = Video.query.get(vid_id)
    vid_name = video.path.split('/')[-1]
    extension = pathlib.Path(vid_name).suffix
    vid_name = vid_name.replace(extension, '')

    srt_path = f"{video.base_path}{video.srt_folder}_en.srt"

    caption_data = read_captions(srt_path)

    captions_embedding = np.load(
        f"{video.base_path}{video.precomp_sub_path}", allow_pickle=True)

    prob_list = []

    for i in range(len(caption_data)-1):

        caption_feature = captions_embedding[i]

        similarity = 50 * (caption_feature @ query_embedding + 1)
        prob_list.append([similarity, caption_data[i], video, i])

    return prob_list
    
    
def get_clips(probs, nb_samp = 5, nb_frames = 5, sim_lim = 1, use_caption_offset = False, 
              caption_offset_value = 15000000, emotion_filter = [], celebrity_filter = [],
              face_filter = [], vid_casts = None, vid_faces = None, precomp_face_filters =None, face_filter_threshold = 50):

    probs = sorted(probs, key=lambda x: x[0], reverse=True)

    prev_path = []
    prev_frame = []

    sp_index = 0
    j = 0
    clips = []
    
    while sp_index < nb_samp:

        """ There are no more captions, BREAK """
        if j == len(probs):
            break

        caption = probs[j][1]
        video = probs[j][2]
        vid_path = video.path

        limits = []

        index = int((caption.start / 1000 / 1000 +
                    caption.end / 1000 / 1000)/2)

        for k in range(len(prev_frame)):

            limits.append(abs(index-prev_frame[k]))

        if probs[j][0] < sim_lim:

            break

        if any(path == vid_path for path in prev_path) and any(lim < 10 for lim in limits):
            j = j + 1

            pass

        else:
            fps = video.fps
            length = video.length
            vid_name = vid_path.split('/')[-1]
            
            if use_caption_offset:
                caption_index = probs[j][3]
                language = video.language
                source_lang = language.split('-')[0].lower()            
                srt_folder = f"{video.base_path}{video.srt_folder}"
                srt_path = f"{srt_folder}_{source_lang}.srt"            
                captions = read_captions(srt_path)
                start, end, text = offset_caption(captions, caption_index, caption_offset_value)
                start, end = start.start/1000000, end.end/1000000
            else:   
                start = max(index-int(nb_frames/2), 0)
                end = min(start+nb_frames, int(length/fps))

            start_frames = int(start*fps)        
            end_frames = int(end*fps) 
                        
            if len(emotion_filter) > 0:
                vid_cast = vid_casts[video.id][start_frames:end_frames]
                emotions = [face[6] for frame in vid_cast for face in frame if face is not None]
                emotion_counts = Counter(emotions)
                sorted_emotions = sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True)
                top_2_emotions = [emotion for emotion, count in sorted_emotions[:2]]
                
                if any(emotion in top_2_emotions for emotion in emotion_filter):
                    pass
                else: 
                    j = j + 1
                    continue
                    
            if len(celebrity_filter) > 0:                                         
                vid_cast = vid_casts[video.id][start_frames:end_frames]
                celebrities = [face[7] for frame in vid_cast for face in frame if face is not None]
                celebrities = list(set(celebrities))
                
                if any(celebrity in celebrities for celebrity in celebrity_filter):
                    pass
                else:
                    j = j + 1
                    continue    
                  
            face_rec_score = 0  
            if len(face_filter) > 0:                                                                       
                vid_face = vid_faces[video.id][start_frames:end_frames]
                precomp_faces = [np.squeeze(face[5]) for frame in vid_face for face in frame if face]               
                if len(precomp_faces)>0:
                    precomp_faces = list(map(np.array, set(map(tuple, precomp_faces))))       
                    precomp_faces = np.array(precomp_faces)
                    precomp_faces = precomp_faces / np.linalg.norm(precomp_faces, axis=-1, keepdims=True)
                    sim_matrix = precomp_face_filters @ precomp_faces.T
                    sim_matrix = 100 * sim_matrix
                    if np.any(sim_matrix > face_filter_threshold):
                        face_rec_score = np.max(sim_matrix)
                        pass
                    else:
                        j = j + 1
                        continue                       
                else:
                    j = j + 1
                    continue      
                                                                                                                          
            # start_mmss & end_mmss just for purpose of demo, to be removed from apptrax app
            start_mmss = datetime.timedelta(seconds=start)
            end_mmss = datetime.timedelta(seconds=end)

            clips.append(
                [vid_name, start, end, video.id, start_mmss, end_mmss, probs[j][0], face_rec_score, caption.get_text()])

            prev_path.append(vid_path)
            prev_frame.append(index)

            j = j + 1
            sp_index = sp_index + 1

    if len(face_filter) > 0:
        clips = sorted(clips, key=lambda x: x[7], reverse=True)
        
    return clips
  
    
def process_dialogues(timestamp, user_email, videos, query_embeddings, nb_frames=5, nb_samp=5,
                     emotion_filter = [], celebrity_filter = [], face_filter = [], gpt_finetune = True, sim_lim = 1):
    """ Extracts clips from videos based on a search query and corresponding captions in srt subtitle files
        Arguments:
            videos: list of video ids for the videos to be processed
            text: search querry
            nb_frames: clips length in seconds
            nb_samp: number of clips to be extracted
    """

    from app.workers.socket_broadcast import socket_broadcast
    from app.beta.models.video import Video
    
    vid_casts = {}
    if (len(emotion_filter) or len(celebrity_filter)) >  0:
        videos = [vid_id for vid_id in videos if Video.query.get(vid_id).precomp_cast_path and 
        os.path.exists(f"{Video.query.get(vid_id).base_path}{Video.query.get(vid_id).precomp_cast_path}")]
        for vid_id in videos:
            video = Video.query.get(vid_id)
            precomp_cast_path = f"{video.base_path}{video.precomp_cast_path}"            
            with open(precomp_cast_path, 'rb') as file:
                precomp_cast = pickle.load(file)
            vid_casts[vid_id] = precomp_cast

    precomp_face_filters = []
    vid_faces = {}
    if len(face_filter) > 0:
        videos = [vid_id for vid_id in videos if Video.query.get(vid_id).face_recognition_embedding_model]    
        for img_id in face_filter:
            face = Img.query.get(img_id)
            precomp_img_path = f"{Config.ROOT_FOLDER}/app/upload/{face.precomp_img_path}"
            precomp_face = np.load(precomp_img_path)
            precomp_face_filters.append(precomp_face)
        for vid_id in videos:
            video = Video.query.get(vid_id)
            if video.processed_cast == 1:
                precomp_faces_path = f"{video.base_path}{video.precomp_cast_path}"                
            else:
                precomp_faces_path = f"{video.base_path}{video.precomp_faces_path}"
            with open(precomp_faces_path, 'rb') as file:
                precomp_faces = pickle.load(file)
            vid_faces[vid_id] = precomp_faces            
        precomp_face_filters = np.array(precomp_face_filters) 
        precomp_face_filters = precomp_face_filters / np.linalg.norm(precomp_face_filters, axis=-1, keepdims=True)
               
    probs = []
    clips = []

    progress_percentage = 10
    increment = 1
    for vid_id in videos:
        try: 
            video = Video.query.get(vid_id)
            dialogue_embedding_model = video.dialogue_embedding_model
            query_embedding = query_embeddings[dialogue_embedding_model]
            prob = process_dialogue(vid_id, query_embedding)
            probs = probs + prob
            progress_percentage = (((increment / len(videos)) * 100) * 0.9) + 10
            increment += 1
            message = json.dumps(
                {
                    "email": user_email,
                    "type": "search_progress",
                    "completed": int(progress_percentage),
                    "timestamp": timestamp
                },
                default=str
            )

            try:
                sse.publish(
                    data=message,
                    id=timestamp,
                    channel=video.package_owner_user_id,
                    type="search_progress"
                )
            except Exception as err:
                logging.error('search progress broadcast failed', err)
        except Exception as err:
            logging.error('process_dialogue FAILED FOR', vid_id, err)
            
    clips = get_clips(probs = probs, nb_samp = nb_samp, nb_frames = nb_frames, sim_lim = sim_lim, 
                      emotion_filter = emotion_filter, celebrity_filter = celebrity_filter, face_filter = face_filter,
                      vid_casts = vid_casts, vid_faces = vid_faces, precomp_face_filters =precomp_face_filters)
    
    return clips
