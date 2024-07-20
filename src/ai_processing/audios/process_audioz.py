import pickle
import json
import logging
from app.beta.models.video import Video
from app.beta.models.image import Img
import datetime
import numpy as np
import os
import random
import torch
from collections import Counter
from flask_sse import sse

# please do not remove this line
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using:", device)


def process_audio(vid_id, query_embedding):

    video = Video.query.get(vid_id)
    fps = video.fps

    audio_embedding = np.load(
        f"{video.base_path}{video.precomp_audio_path}", allow_pickle=True)

    sim_list = []

    def simcalc(x): return 50 * (x @ query_embedding + 1)

    sim_list = simcalc(audio_embedding)

    window = int(1 * fps/fps)
    mean_sim_list = []

    for i in range(len(sim_list)):
        lim_inf = min(i, window)
        lim_sup = min(len(sim_list)-i, window)

        exlist = sim_list[i-lim_inf:i+lim_sup]

        mean_sim_list.append([sum(exlist)/len(exlist), i*fps + fps, video, ''])

    return mean_sim_list


def process_audios(timestamp, user_email, videos, query_embeddings, nb_frames=5, nb_samp=5,
                   emotion_filter=None, celebrity_filter=None, face_filter = None, sim_lim=1, face_filter_threshold = 50):
    """ Extracts clips from videos based on a search querry and corresponding frames in the videos
        Arguments:
            vid_id: list of ids to the video files to be processed
            text_inp: search querry
            nb_frames: clips length in seconds
            nb_samp: number of clips to be extracted
    """

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
        
    sims = []

    progress_percentage = 10
    increment = 1
    for vid_id in videos:
        try:
            video = Video.query.get(vid_id)
            audio_embedding_model = video.audio_embedding_model
            query_embedding = query_embeddings[audio_embedding_model]
            sim = process_audio(vid_id, query_embedding)
            sims = sims + sim
            progress_percentage = (
                ((increment / len(videos)) * 100) * 0.9) + 10
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
                    type="search_progress",
                    id=timestamp,
                    channel=video.package_owner_user_id
                )
            except Exception as err:
                logging.error('search progress broadcast failed', err)
        except Exception as err:
            print('audio search process FAILED', vid_id, err)

    sims = sorted(sims, key=lambda x: x[0], reverse=True)

    prev_path = []
    prev_frame = []

    sp_index = 0
    j = 0

    ind = random.randint(0, 1000000)

    clips = []

    while sp_index < nb_samp and j < len(sims):

        limits = []
        video = sims[j][2]
        vid_path = video.path
        clip_description = sims[j][3]

        index = sims[j][1]

        for k in range(len(prev_frame)):

            limits.append(abs(index-prev_frame[k]))

        if sims[j][0] < sim_lim:

            break

        if any(path == vid_path for path in prev_path) and any(lim < 400 for lim in limits):
            j = j + 1

            pass

        else:

            fps = video.fps
            length = video.length

            start = max(int(index/fps-nb_frames/2), 0)
            end = min(int(start+nb_frames), int(length/fps))
            start_frames = int(start*fps)        
            end_frames = int(end*fps) 
                
            if len(emotion_filter) > 0:
                vid_cast = vid_casts[video.id][start_frames:end_frames]
                emotions = [face[6] for frame in vid_cast for face in frame if face is not None]
                emotion_counts = Counter(emotions)
                sorted_emotions = sorted(
                    emotion_counts.items(), key=lambda x: x[1], reverse=True)
                top_2_emotions = [emotion for emotion,
                                  count in sorted_emotions[:2]]

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

            vid_name = vid_path.split('/')[-1]

            # start_mmss & end_mmss just for purpose of demo, to be removed from apptrax app
            start_mmss = datetime.timedelta(seconds=start)
            end_mmss = datetime.timedelta(seconds=end)

            clips.append(
                [vid_name, start, end, video.id, start_mmss, end_mmss, sims[j][0], face_rec_score, clip_description])

            prev_path.append(vid_path)
            prev_frame.append(sims[j][1])

            j = j + 1
            sp_index = sp_index + 1
            
    if len(face_filter) > 0:
        clips = sorted(clips, key=lambda x: x[7], reverse=True)
        
    return clips
