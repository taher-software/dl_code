from datetime import datetime

from sqlalchemy.orm import relationship
from src.models.package import Package

from src.main import db
from src.models.user import User

from sqlalchemy import Column, DateTime, ForeignKey, Integer, String


class UserPackageOwner(db.Model):
    __tablename__ = 'user_package_owner'
    
    FK_USER_PACKAGE_OWNER_ID = 'user_package_owner.id'

    id = Column(Integer, primary_key=True)
    owner_id = Column(Integer, ForeignKey(User.FK_USER_ID, name='fk_user_package_owner_owner_id', ondelete='CASCADE'))
    package_id = Column(Integer, ForeignKey(Package.FK_PACKAGE_ID, name='fk_user_package_owner_package_id', ondelete='CASCADE'))
    subscription_id = Column(String(66))
    subscription_period = Column(String(10))
    number_of_seats = Column(Integer, default = 1)
    active_seats = Column(Integer, default = 1)
    start = Column(DateTime)
    end = Column(DateTime)
    trial_end = Column(DateTime)
    discount = Column(Integer)
    status = Column(Integer, default = 0)
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now)

    """ Relationships """
    package = relationship('Package', back_populates='owners')
    package_owner = relationship('User', back_populates='user_packages')
    user_packages = relationship('UserPackage', back_populates='user_package_owner')

    def to_json(self):
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}