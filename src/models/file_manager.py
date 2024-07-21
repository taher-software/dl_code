from datetime import datetime

from src.models.enums.file_manager_type import FileManagerType
from src.main import db

from sqlalchemy import Column, DateTime, Enum, Integer, String


class FileManager(db.Model):

    file_manager_type_list = [file_manager_type.value for file_manager_type in FileManagerType]

    __tablename__ = 'file_manager'

    """ Primary Key """
    id = Column(Integer, primary_key=True)

    """ Foreign Keys """
    package_owner_user_id = Column(Integer)

    """ Attributes / Fields """
    title = Column(String(length=255))
    location = Column(String(length=255))
    type = Column(Enum(*file_manager_type_list, name='file_manager_type'), default='DEFAULT')

    """ Housekeeping """
    datetime_created = Column(DateTime, default=datetime.utcnow)