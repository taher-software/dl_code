from datetime import datetime

from src.models.saved_clip import SavedClip
from src.models.collection import Collection
from src.main import db
from sqlalchemy import Column, Integer, ForeignKey, DateTime
from sqlalchemy.orm import relationship


class CollectionClip(db.Model):
    FK_CLIP_ID = 'collection_clip.id'
    __tablename__ = 'collection_clip'

    """ Primary Key """
    id = Column(Integer, primary_key=True)

    """ Foreign Keys """
    collection_id = Column(Integer, ForeignKey(Collection.FK_COLLECTION_ID, name='fk_collection_clip_collection_id', ondelete='CASCADE'))
    saved_clip_id = Column(Integer, ForeignKey(SavedClip.FK_SAVED_CLIP_ID, name='fk_collection_clip_saved_clip_id', ondelete='CASCADE'))

    """ Housekeeping """
    datetime_created = Column(DateTime, default=datetime.utcnow)

    """ Relationships """
    collection = relationship('Collection', back_populates='collection_clips')
    saved_clip = relationship('SavedClip', back_populates='saved_collection_clips')

    def to_json(self):
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}