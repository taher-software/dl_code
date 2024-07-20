from datetime import datetime

from src.models.enums.image_type import ImageType
from app import db

from sqlalchemy import Column, DateTime, Enum, Integer, String


class Img(db.Model):

    image_type_list = [image_type.value for image_type in ImageType]

    __tablename__ = 'image'

    """ Primary Key """
    id = Column(Integer, primary_key=True)

    """ Foreign Keys """
    package_owner_user_id = Column(Integer)

    """ Attributes / Fields """
    title = Column(String(length=255))
    location = Column(String(length=255))
    type = Column(Enum(*image_type_list, name='image_type'), default='DEFAULT')
    
    """ AI processing (optional)"""
    visual_embedding_model = Column(String(256))    
    precomp_img_path = Column(String(256))
    processed_viz = Column(Integer, default=0)
    processing_failed = Column(Integer, default=0)
    bbox = Column(String(256))

    """ Housekeeping """
    datetime_created = Column(DateTime, default=datetime.utcnow)
