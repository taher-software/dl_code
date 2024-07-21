from datetime import datetime

from src.main import db

from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, Text

class Download(db.Model):
    __tablename__ = 'download'
    
    FK_DOWNLOAD_ID = 'download.id'

    """ Primary Key """
    id = Column(Integer, primary_key=True)

    """ Foreign Keys """
    user_id = Column(Integer, ForeignKey('user.id', name='fk_download_user_id'))
    video_id = Column(Integer)

    """ Attributes / Fields """
    download_options = Column(Text(4294000000)) 
    downloaded = Column(Integer, default=0)
    download_history = Column(String(200))
    file_name = Column(String(256))
    text = Column(Text(4294000000))
    title = Column(String(200))
    type = Column(String(200))
    trimmed = Column(Integer)
    reframed = Column(Integer)

    """ Housekeeping """
    created_on = Column(DateTime, default=datetime.utcnow)
    cut_completed_on = Column(DateTime, default=datetime.utcnow)

    def to_json(self):
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}