from datetime import datetime

from src.models.user import User
from src.models.video import Video
from app import db

from sqlalchemy import Column, DateTime, ForeignKey, Integer, Numeric, String, Text
from sqlalchemy.orm import relationship


class Clip(db.Model):
    FK_CLIP_ID = 'clip_tr.id'
    __tablename__ = 'clip_tr'

    """ Primary Key """
    id = Column(Integer, primary_key=True)

    """ Foreign Keys """
    owner_id = Column(Integer)
    user_id = Column(Integer, ForeignKey(User.FK_USER_ID, name='fk_clip_user_id', ondelete='CASCADE'))
    video_id = Column(Integer, ForeignKey(Video.FK_VIDEO_ID, name='fk_clip_video_id', ondelete='CASCADE'))

    """ Attributes / Fields """
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
    video = relationship('Video', back_populates='clips')

    def to_json(self):
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}