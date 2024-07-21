from datetime import datetime
from sqlalchemy import Column, DateTime, ForeignKey, Integer, String

from src.models.user import User
from src.main import db


class UserSubscriptionUser(db.Model):
    __tablename__ = 'user_subscription_user'
    
    FK_USER_ID = 'user_subscription_user.id'

    id = Column(Integer, primary_key=True)
    owner_id = Column(Integer, ForeignKey('user.id', name='fk_user_subscription_user_owner_id', ondelete='CASCADE'))
    link_email = Column(String(120))
    active = Column(Integer, default = 0)
    """The state to apply after subscription create/updated success"""
    pending_active_state = Column(Integer, default = 0)
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now)

    def to_json(self):
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}

    def get_collaborator_details(self):
        collab = self.to_json()
        collab_user = User.query.filter_by(email=self.link_email).first()
        if collab_user is not None:
            collab['user'] = collab_user.to_json()
            collab['user']['is_owner'] = self.owner_id == collab_user.id
        return collab