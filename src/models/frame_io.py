from src.models.user import User
from src.main import db
from datetime import datetime

from sqlalchemy import Boolean, Column, DateTime, ForeignKey, Integer, LargeBinary
from sqlalchemy.orm import relationship


class FRAMEIO(db.Model):
    __tablename__ = 'frame_io'
    
    FK_FRAMEIO_ID = 'frame_io.id'

    """ Primary Key """
    id = Column(Integer, primary_key=True)

    """ Foreign Keys """
    package_user_id = Column(Integer, ForeignKey(User.FK_USER_ID, name='fk_frame_io_user_id', ondelete='CASCADE'))

    """ Attributes / Fields """
    connection_string = Column(LargeBinary)
    active = Column(Boolean, default=True)

    """ Housekeeping """
    datetime_created = Column(DateTime, default=datetime.utcnow)

    def to_json(self):
        response_json = {c.name: getattr(self, c.name)
                         for c in self.__table__.columns}
        return response_json
