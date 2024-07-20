from datetime import datetime
from sqlalchemy.orm import relationship

from src.models.folder import Folder
from src.models.video import Video
from app import db
from sqlalchemy import Column, Integer, ForeignKey, DateTime


class FolderVideo(db.Model):
    __tablename__ = 'folder_video'
    FK_FOLDER_VIDEO_ID = 'folder_video.id'
    
    """ Primary Key """
    id = Column(Integer, primary_key=True)

    """ Foreign Keys """
    folder_id = Column(Integer, ForeignKey(Folder.FK_FOLDER_ID, name='fk_folder_video_folder_id', ondelete='CASCADE'))
    video_id = Column(Integer, ForeignKey(Video.FK_VIDEO_ID, name='fk_folder_video_video_id', ondelete='CASCADE'))

    """ Housekeeping """
    datetime_created = Column(DateTime, default=datetime.utcnow)

    """Relationships"""
    folder = relationship('Folder', backref='folder_folders')
    video = relationship('Video', backref='folder_videos')

    def to_json(self):
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}