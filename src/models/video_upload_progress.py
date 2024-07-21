from datetime import datetime
import json

from src.models.video import Video
from src.main import db
from sqlalchemy import Column, DateTime, Integer, String, ForeignKey, Numeric, Text

class VideoUploadProgress(db.Model):
    __tablename__ = 'video_upload_progress'
    
    FK_VIDEO_UPLOAD_PROGRESS_ID = 'video_upload_progress.id'

    """ Primary Key """
    id = Column(Integer, primary_key=True)

    """ Foreign Keys """
    video_id = Column(Integer, ForeignKey(Video.FK_VIDEO_ID, name='fk_video_upload_progress_video_id', ondelete='CASCADE'))

    """ Attributes / Fields """
    additional_info = Column(Text(500))
    estimated_full_duration = Column(Numeric(10,2), default=-1)
    processed_au = Column(Integer, default=0)
    processed_pc = Column(Integer, default=0)
    processed_srt_gen = Column(Integer, default=0)
    failed = Column(Integer, default=0)
    progress = Column(Numeric(10,2), default=0)
    rencoded = Column(Integer, default=1)
    status = Column(String(200))

    """ Housekeeping """
    created_on = Column(DateTime, default=datetime.utcnow)
    
    def processing_completed(self):
        return (self.rencoded + self.processed_au + self.processed_pc + self.processed_srt_gen) == 4
    
    
    def to_json(self):
        response = {c.name: getattr(self, c.name) for c in self.__table__.columns}
        response['progress'] = int(self.progress)
        response['estimated_full_duration'] = float(self.estimated_full_duration) if int(self.estimated_full_duration) != -1 else -1
        response['additional_info'] = json.loads(self.additional_info) if self.additional_info is not None else None
        time_diff = datetime.utcnow() - self.created_on
        response['time_elapsed'] = abs(time_diff.total_seconds()) 
        response['created_on'] = self.created_on.isoformat()
        return response
