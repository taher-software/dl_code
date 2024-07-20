from datetime import datetime

from sqlalchemy.orm import relationship

from app import db

from sqlalchemy import Column, Integer, ForeignKey, DateTime
from src.models.package import Package
from src.config import Config


class UserPackage(db.Model):
    __tablename__ = 'user_package'
    
    FK_USER_ID = 'user.id'
    FK_PACKAGE_ID = 'package.id'
    FK_PACKAGE_OWNER_ID = 'user_package_owner.id'

    id = Column(Integer, primary_key=True)
    start = Column(DateTime)
    end = Column(DateTime)
    status = Column(Integer, default = 0)
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now)

    """ Relationships """
    user_id = Column(Integer, ForeignKey(FK_USER_ID, name='fk_up_user_id'))
    user = relationship('User', backref='user')
    package_id = Column(Integer, ForeignKey(FK_PACKAGE_ID, name='fk_up_package_id'))
    package = relationship('Package', backref='package')
    user_package_owner_id = Column(Integer, ForeignKey(FK_PACKAGE_OWNER_ID, name='fk_up_user_package_owner_id'))
    user_package_owner = relationship('UserPackageOwner', back_populates='user_packages', overlaps='user_packages')

    def to_json(self):
        up = {c.name: getattr(self, c.name) for c in self.__table__.columns}
        up['data'] = self.package.to_json()
        up['upgrade'] = None
        up['trial'] = False

        packages = Package.query.filter_by(online_join=1, contact_required=0).all()
        now = datetime.utcnow()
        had_trial_before = UserPackage.query.filter(UserPackage.user_id==self.user_id, UserPackage.status==0, UserPackage.package_id!=1, UserPackage.start <= now).first()
        subs = UserPackage.query.filter(UserPackage.user_id==self.user_id, UserPackage.start <= now).count()
        
        ## He has not had a trial before
        if had_trial_before is None and subs > 1:
            up['trial'] = True

        if self.package.id == 1 and len(packages) > 1 and Config.ENTEPRRISE_CLIENT == '0':
            up['upgrade'] = packages[1].to_json()

        return up