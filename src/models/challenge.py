from datetime import datetime, timedelta
from random import randint
from sqlalchemy import desc

from src.main import db

from sqlalchemy import Column, Integer, DateTime, ForeignKey

class Challenge(db.Model):
    __tablename__ = 'manual_otp'
    
    FK_INVITE_ID = 'manual_otp.id'

    """ Primary Key """
    id = Column(Integer, primary_key=True)

    """ Foreign Keys """
    user_id = Column(Integer, ForeignKey('user.id', name="fk_manual_otp_user_id"))

    """ Attributes / Fields """
    otp = Column(Integer)
    otp_used = Column(Integer, default = 0)

    """ Housekeeping """
    datetime_created = Column(DateTime, name='created_on', default=datetime.utcnow)
    expires_on = Column(DateTime, default=datetime.utcnow)

    def to_json(self):
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}
    
    """ Class Variables """
    number_of_digits = 6
    expires_in_minutes = None
    valid_otp = None

    def __init__(self, expires_in_minutes, number_of_digits, user_id):
        timenow = datetime.today()
        self.created_on = timenow
        self.expires_in_minutes = expires_in_minutes
        self.expires_on = timenow + timedelta(minutes=expires_in_minutes)
        self.number_of_digits = number_of_digits
        self.user_id = user_id
        self.set_otp()

    def set_otp(self):
        range_start = 10**(self.number_of_digits-1)
        range_end = (10**self.number_of_digits)-1
        self.otp = randint(range_start, range_end)

    def is_valid(self, otp):
        self.valid_otp = Challenge.query.filter_by(user_id=self.user_id, otp=otp, otp_used = 0).order_by(
            desc(Challenge.created_on)).first()

        
        # Cannot find a match
        if self.valid_otp is None:
            return False
        timenow = datetime.today()
        if (self.valid_otp.expires_on - timenow).total_seconds() > 0:
            return True
        return False