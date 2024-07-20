import srt
import json
import datetime


def write_txt(txt_path, subs):

    f = open(txt_path, 'w')
    for s in subs:
        f.write(s.content.strip() + "\n")
    f.close()
    return


def write_srt(srt_path, subs):

    f = open(srt_path, 'w')
    f.writelines(srt.compose(subs))
    f.close()
    return
    

def write_srt_from_list(srt_path, subs_list):

    subs = []
    
    for idx, caption in enumerate(subs_list):
        start = caption['start']/1000
        end = caption['end']/1000
        text = caption['text']
        
        subs.append(srt.Subtitle(index=idx+1,
                                 start=datetime.timedelta(seconds=start),
                                 end=datetime.timedelta(seconds=end),
                                 content=srt.make_legal_content(text)))               
                
    f = open(srt_path, 'w')
    f.writelines(srt.compose(subs))
    f.close()
    return
    
    
def break_sentences(words, max_chars=150, clip_start=None, clip_end=None):
    firstword = True
    charcount = 0
    idx = 1
    subs = []
    caption = ""

    subs_list = []        
    
    if clip_start is not None:
        clip_start = clip_start*1000
        clip_end = clip_end*1000

        words = [word for word in words if word[1] >= clip_start and word[2] <= clip_end]
        words = [[word[0], word[1]-clip_start, word[2]-clip_start] for word in words]

    for w in words:
        if firstword:
            caption_start = w[1]

        charcount += len(w[0])
        caption += " " + w[0].strip()

        if ("." in w[0] or "!" in w[0] or "?" in w[0] or
                charcount > max_chars or ("," in w[0] and not firstword) or w == words[-1]):

            caption_end = w[2]
            subs_list.append([caption, caption_start/1000, caption_end/1000])
            
            subs.append(srt.Subtitle(index=idx,
                                     start=datetime.timedelta(
                                         seconds=caption_start/1000),
                                     end=datetime.timedelta(seconds=caption_end/1000),
                                     content=srt.make_legal_content(caption)))
            
            firstword = True
            idx += 1
            caption = ""
            charcount = 0
        else:
            firstword = False
    return subs, words, subs_list


def json2srt_txt(json_path, srt_path, txt_path=None, start=None, end=None, clip_tms_path=None):
    f = open(json_path, "r")

    words = json.loads(f.read())
    
    if start is not None:
       subs, clip_words, _ = break_sentences(words, clip_start=start, clip_end=end)
    else:
       subs, clip_words, _ = break_sentences(words)

    write_srt(srt_path, subs)

    if txt_path:
        write_txt(txt_path, subs)

    if clip_tms_path:
        with open(clip_tms_path, 'w') as f:
            json.dump(clip_words, f)
        print("Writing clip timestamps to: {}".format(clip_tms_path))


def clip_srt_and_timestamp(json_path, start=None, end=None):
    f = open(json_path, "r")

    words = json.loads(f.read())
    subs, clip_words, subs_list = break_sentences(words, max_chars=10, clip_start=start, clip_end=end)

    return {
        'srt': subs_list,
        'timestamps': clip_words, 
        'srt_subs': subs
    }
