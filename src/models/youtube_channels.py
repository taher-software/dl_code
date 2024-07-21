from src.main import db
from datetime import datetime

from sqlalchemy import Column, DateTime, ForeignKey, Integer, String
from sqlalchemy.orm import relationship


class YoutubeChannel(db.Model):
    __tablename__ = 'youtube_channel'

    FK_VIDEO_ID = 'youtube_channel.id'
    

    """ Primary Key """
    id = Column(Integer, primary_key=True)

    """ Foreign Keys """
    package_user_id = Column(Integer, ForeignKey("user.id", name='fk_youtube_channel_user_id', ondelete='CASCADE'))

    """ Attributes / Fields """
    channel_id = Column(String(255))
    channel_title = Column(String(255))
    channel_thumbnail = Column(String(255))
    access_token = Column(String(255), index=True)
    refresh_token = Column(String(255), index=True)

    """ Housekeeping """
    datetime_created = Column(DateTime, default=datetime.utcnow)

    """ Relationships"""
    user = relationship("User",back_populates="channels")

    def to_json(self):
        response_json = {c.name: getattr(self, c.name)
                         for c in self.__table__.columns}
        return response_json