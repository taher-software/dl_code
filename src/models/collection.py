from datetime import datetime
from flask_login import current_user
import json
from sqlalchemy.orm import relationship
from sqlalchemy import case, Column, Integer, String, ForeignKey, DateTime, Text

from src.models.saved_clip import SavedClip
from src.models.user import User
from src.main import db


class Collection(db.Model):
    FK_COLLECTION_ID = 'collection.id'
    __tablename__ = 'collection'

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey(User.FK_USER_ID, name='fk_collection_user_id'))
    owner_id = Column(Integer)
    display_name = Column(String(200))
    
    datetime_created = Column(DateTime, name='created_at', default=datetime.utcnow)

    """ Relationships """
    collection_clips = relationship('CollectionClip', backref='collection_clips', cascade='all, delete', lazy='dynamic', uselist=True)

    """ Drop when move away from beta """
    fav_ids = Column(Text(4294000000))
    thumb_image = Column(Text(4294000000))
    
    def set_from_form(self, form, user_id):
        self.created_at = self.datetime_created
        for key in form:
            setattr(self, key, form[key])
        self.user_id = user_id

    def is_valid(self):
        return self.display_name is not None and self.user_id is not None

    def get(self):
        vids = current_user.videos.all()
        favs = current_user.clips_fav.all()

        collection = self
        responseCollection = collection.to_json()
        responseCollection['favs'] = []
        if collection.fav_ids is None:
            return responseCollection

        fav_list = map(str, json.loads(collection.fav_ids))

        mapped_favs = []
        for fv in fav_list:
            try:
                my_fv =  next(f for f in favs if str(f.id) == fv)
                mapped_favs.append(my_fv)
            except:
                print('not found', fv)

        favourites = []
        for fav in mapped_favs:
            video =  next(video for video in vids if video.id == fav.video_id)
            tfav = fav.to_json()
            tfav["video"] = video.to_json()
            favourites.append(tfav)

        responseCollection['favs'] = favourites
        return responseCollection
    
    def to_json(self):
        self.created_at = self.datetime_created
        collection_columns = {c.name: getattr(self, c.name) for c in self.__table__.columns}
        saved_clips = []
        try:
            fav_ids = json.loads(self.fav_ids)

            reversed_fav_ids = fav_ids[::-1]
            order_case = case(
                {id: index for index, id in enumerate(reversed_fav_ids)},
                value=SavedClip.id
            )
            saved_clips = SavedClip.query.filter(SavedClip.id.in_(fav_ids)).order_by(order_case).all()
            collection_columns['thumbnails'] = [{"thumb": f"/{saved_clip.clip_thumb_static}", "gif": saved_clip.clip_thumb_animated} for saved_clip in saved_clips]
            file_size = 0
            duration = 0
            for saved_clip in saved_clips:
                try:
                    duration = duration + (saved_clip.end - saved_clip.start)
                except:
                    pass
                file_size = file_size + self.calculate_file_size(saved_clip)

            collection_columns['duration'] = duration
            collection_columns['file_size'] = file_size
        
        except:
            collection_columns['thumbnails'] = []
            collection_columns['file_size'] = 0

        return collection_columns
    
    def calculate_file_size(self, saved_clip):
        portion = saved_clip.end - saved_clip.start
        duration = saved_clip.video.duration
        try:
            return portion / duration * saved_clip.video.file_size
        except:
            return 0