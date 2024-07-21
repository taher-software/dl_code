from src.models.user import User
from src.main import db
from datetime import datetime

from sqlalchemy import Boolean, Column, DateTime, ForeignKey, Integer, LargeBinary
from sqlalchemy.orm import relationship


class AWSS3(db.Model):
    __tablename__ = 'aws_s3'
    
    FK_AWSS3_ID = 'aws_s3.id'

    """ Primary Key """
    id = Column(Integer, primary_key=True)

    """ Foreign Keys """
    package_user_id = Column(Integer, ForeignKey(User.FK_USER_ID, name='fk_aws_s3_user_id', ondelete='CASCADE'))

    """ Attributes / Fields """
    connection_string = Column(LargeBinary)
    active = Column(Boolean, default=True)

    """ Housekeeping """
    datetime_created = Column(DateTime, default=datetime.utcnow)

    def to_json(self):
        response_json = {c.name: getattr(self, c.name)
                         for c in self.__table__.columns}
        return response_json