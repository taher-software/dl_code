from datetime import datetime

from sqlalchemy.orm import relationship

from src.main import db

from sqlalchemy import Column, DateTime, ForeignKey, Integer, Numeric


class UserUsage(db.Model):
    __tablename__ = 'user_usage'
    
    FK_USER_USAGE_ID = 'user_usage.id'
    FK_USER_ID = 'user.id'
    
    id = Column(Integer, primary_key=True)
    
    """ Foreign Keys """
    user_id = Column(Integer, ForeignKey(FK_USER_ID, name='fk_user_usage_user_id', ondelete='CASCADE'))
    package_owner_user_id = Column(Integer, ForeignKey(FK_USER_ID, name='fk_user_usage_package_owner_user_id', ondelete='CASCADE'))

    storage = Column(Numeric(10, 6), default=0)
    transcription_minutes = Column(Numeric(10, 2), default=0)
    dialog_searches = Column(Integer, default=0)
    visual_searches = Column(Integer, default=0)
    audio_searches = Column(Integer, default=0)
    discoveries = Column(Integer, default=0)
    period_start = Column(DateTime)
    period_end = Column(DateTime)
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now)

    def to_json(self):
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}