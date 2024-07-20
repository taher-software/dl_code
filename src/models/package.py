from datetime import datetime

from sqlalchemy.orm import relationship

from app import db

from sqlalchemy import Column, Integer, Boolean, DateTime, Numeric, String, Text

class Package(db.Model):
    __tablename__ = 'package'

    FK_PACKAGE_ID = 'package.id'
    
    """ Primary Key """
    id = Column(Integer, primary_key=True)

    """ Attributes / Fields """
    accepts_promo = Column(Integer, default=0)
    active = Column(Integer, default = 1)
    auto_register = Column(Integer, default=0)
    button_active_text = Column(String(255))
    button_no_trial_text = Column(String(255))
    button_text = Column(String(255))
    contact_required = Column(Integer, default=0)
    display_order = Column(Integer, default=0)
    display_size = Column(Integer, default=0)
    features = Column(Text(4294000000))
    intro_text = Column(Text())
    limits = Column(Text(4294000000))
    lookup_key = Column(String(10))
    max_seats =  Column(Integer)
    most_popular = Column(Integer, default=0)
    online_join = Column(Integer)
    price_annual = Column(Numeric(10,2))
    price_annual_stripe_id = Column(String(66))
    price_month = Column(Numeric(10,2))
    price_month_stripe_id = Column(String(66))
    show_free_folders = Column(Integer, default=0)
    stripe_product_id = Column(String(20))
    title = Column(String(50))
    trial_period = Column(Integer)
    upgrade = Column(String(255), default="[]")
    admin_can_upgrade = Column(Boolean, default=False)

    """ Housekeeping """
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)\
    
    """Relationships"""
    owners = relationship('UserPackageOwner', back_populates='package')

    def to_json(self):
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}