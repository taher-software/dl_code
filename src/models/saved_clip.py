from datetime import datetime

from src.models.user import User
from src.models.video import Video
from src.main import db

from sqlalchemy import Column, DateTime, ForeignKey, Integer, Numeric, String, Text
from sqlalchemy.orm import relationship


class SavedClip(db.Model):
    FK_SAVED_CLIP_ID = 'clip_fav.id'
    __tablename__ = 'clip_fav'

    """ Primary Key """
    id = Column(Integer, primary_key=True)

    """ Foreign Keys """
    owner_id = Column(Integer)
    user_id = Column(Integer, ForeignKey(User.FK_USER_ID, name='fk_saved_clip_user_id', ondelete='CASCADE'))
    video_id = Column(Integer, ForeignKey(Video.FK_VIDEO_ID, name='fk_saved_clip_video_id', ondelete='CASCADE'))

    """ Attributes / Fields """
    clip_thumb_static = Column(String(256))
    clip_thumb_animated = Column(String(256))
    editor_captions = Column(Text(4294000000))
    editor_reframes = Column(Text(4294000000))
    end = Column(Numeric(10, 2))
    reframed = Column(Integer, default=0)
    start = Column(Numeric(10, 2))
    text = Column(Text)
    title = Column(String(256))
    type = Column(String(20))
    trimmed = Column(Integer, default=0)
    trimmer_start = Column(Numeric(10, 2))
    trimmer_end = Column(Numeric(10, 2))
    trimmer_interval = Column(Numeric(10, 2))
    type = Column(String(20))

    """ Housekeeping """
    datetime_created = Column(DateTime, default=datetime.utcnow)

    """ Relationships """
    video = relationship('Video', back_populates='saved_clips')
    saved_collection_clips = relationship('CollectionClip', backref='saved_collection_clips', cascade='all, delete', lazy='dynamic', uselist=True)

    def get_existing_clip(self):
        return self.query.filter_by(
            owner_id = self.owner_id,
            video_id = self.video_id,
            editor_reframes = self.editor_reframes,
            start = self.start,
            end = self.end,
            type = self.type
        ).first()
    
    def to_json(self):
        response_json = {c.name: getattr(self, c.name)
                         for c in self.__table__.columns}
        try:
            clip_duration = self.end - self.start
            response_json['duration'] = clip_duration
            try:
                file_size = clip_duration / self.video.duration * self.video.file_size
                response_json['file_size'] = file_size
            except:
                response_json['file_size'] = 0
        except:
            response_json['duration'] = 0
            response_json['file_size'] = 0
        return response_json

    