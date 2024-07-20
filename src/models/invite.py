from datetime import datetime

from sqlalchemy.orm import relationship

from src.models.user import User
from app import db

from sqlalchemy import Column, DateTime, Integer, String, ForeignKey


class Invite(db.Model):
    __tablename__ = 'invites'
    
    FK_INVITE_ID = 'invites.id'

    """ Primary Key """
    id = Column(Integer, primary_key=True)

    """ Attributes / Fields """
    already_member = Column(Integer, default = 0)
    email_address = Column(String(255))
    email_sent = Column(Integer, default = 0)
    invitee_id = Column(Integer, ForeignKey('user.id'), default = 0)
    invitee_joined = Column(Integer, default = 0)
    invite_only = Column(Integer, default = 0)
    still_linked = Column(Integer, default = 0)
    
    """ Housekeeping """
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)

    invitor = relationship(User)

    def to_json(self):
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}
    
    def get_details(self):
        invite = self.to_json()
        invite_user = User.query.filter_by(email=self.email_address).first()
        if invite_user is not None:
            invite['user'] = invite_user.to_json()
        return invite   