import json

from sqlalchemy import Column, ForeignKey, Integer, String, Text

from src.models.video import Video
from src.main import db
from src.models.user import User


class SearchOptions(db.Model):
    __tablename__ = 'search_options'
    
    FK_SEARCH_OPTIONS_ID = 'search_options.id'

    """ Primary Key """
    id = Column(Integer, primary_key=True)

    """ Foreign Keys """
    user_id = Column(Integer, ForeignKey(User.FK_USER_ID, name="fk_search_options_user_id"))

    """ Attributes / Fields """
    nb_frames = Column(Integer, default=0)
    nb_samp = Column(Integer, default=0)
    search_type = Column(String(10))
    source = Column(Text(4294000000))
    emotion_filter = Column(Text(4294000000))
    celebrity_filter = Column(Text(4294000000))
    face_filter = Column(Text(4294000000))    


    def __init__(self, options = None):
        if options is not None:
            for key in options:
                if hasattr(self, key):
                    setattr(self, key, options[key]) 
            self.source = self.get_valid_videos(options['source'])
            self.emotion_filter = self.get_dump(options['emotion_filter'])
            self.celebrity_filter = self.get_dump(options['celebrity_filter'])
            self.face_filter = self.get_dump(options['face_filter'])

    def to_json(self):
            response = {c.name: str(getattr(self, c.name)) for c in self.__table__.columns}
            response['source'] = json.loads(self.source)
            response['emotion_filter'] = json.loads(self.emotion_filter)
            response['celebrity_filter'] = json.loads(self.celebrity_filter)
            response['face_filter'] = json.loads(self.face_filter if self.face_filter is not None else "[]")
            return response

    def get_valid_videos(self, vid_ids):
        if len(vid_ids) == 0:
             return "[]"
        valid_ids = Video.query.filter(Video.id.in_(vid_ids)).with_entities(Video.id).all()
        valid_ids = set([str(id[0]) for id in valid_ids])
        return json.dumps([id for id in list(set(vid_ids)) if id in valid_ids])
    
    def get_dump(self, resource):
         if resource is None or len(resource) == 0:
              return "[]"
         else: return json.dumps(resource)