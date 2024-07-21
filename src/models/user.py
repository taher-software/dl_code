import base64
from datetime import datetime
from flask import session
from flask_login import UserMixin
from hashlib import md5
import jwt
import onetimepass
import os
from sqlalchemy import Boolean, Column, Integer, String, DateTime
from sqlalchemy.orm import relationship
from time import time
import urllib
from werkzeug.security import generate_password_hash, check_password_hash

from src.main import db
from src.config import Config

class User(db.Model, UserMixin):
    FK_USER_ID = 'user.id'
    __tablename__ = 'user'

    id = Column(Integer, primary_key=True)
    first_name = Column(String(128))
    last_name = Column(String(128))
    email = Column(String(120), index=True, unique=True)
    password_hash = Column(String(128))
    profile_image = Column(String(256))
    about_me = Column(String(140))
    confirmed_email = Column(Integer, default=0)
    last_authentication = Column(DateTime)
    env_role = Column(String(20), default="USER")
    role = Column(String(20))
    datetime_created = Column(
        DateTime, name='created_on', default=datetime.utcnow)
    accepted_terms = Column(Boolean, default=0)
    authenticator_app_otp_secret = Column(String(16))
    authenticator_linked = Column(Boolean, default=False)
    company = Column(String(128))
    last_seen = Column(DateTime, default=datetime.utcnow)

    channels = relationship("YoutubeChannel", back_populates="user")
    gdrives = relationship("GDrive", back_populates="user")
    dropboxs = relationship("Dropbox", back_populates="user")

    stripe_id = Column(String(66))
    stripe_checkout_id = Column(String(66))
    trial_period = Column(Integer, default=100000)
    login_count = Column(Integer, default=0)
    profile_image = Column(String(256))
    last_seen = Column(DateTime)

    """Relationships"""
    downloads = relationship('Download', backref='author', lazy='dynamic')
    clips_fav = relationship('SavedClip', backref='creator', lazy='dynamic')
    collections = relationship('Collection', backref='creator', lazy='dynamic')
    folders = relationship('Folder', backref='creator', lazy='dynamic')
    packages = relationship(
        'UserPackage', back_populates='user', lazy='dynamic')
    videos = relationship(
        'Video', primaryjoin="User.id == Video.user_id", backref='author', lazy='dynamic')
    user_packages = relationship(
        'UserPackageOwner', back_populates='package_owner')
    user_usages = relationship(
        'UserUsage', primaryjoin="User.id == UserUsage.user_id", backref='author', lazy='dynamic')
    channels = relationship("YoutubeChannel", back_populates="user")
    gdrives = relationship("GDrive", back_populates="user")
    dropboxs = relationship("Dropbox", back_populates="user")

    """TODO remove when api v1 deprecated"""
    usage = None
    package = None
    package_owner = None

    ui_key_map = [
        'about_me',
        'company',
        'confirmed_email',
        'datetime_created',
        'email',
        'first_name',
        'last_name',
        'last_seen',
        'profile_image',
        'role',
        'trial_period',
        'env_role',
        'login_count'
    ]

    created_on = None
    

    def __repr__(self):
        return '<User {}>'.format(self.email)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    def avatar(self, size):
        digest = md5(self.email.lower().encode('utf-8')).hexdigest()
        return 'https://www.gravatar.com/avatar/{}?d=identicon&s={}'.format(
            digest, size)

    def get_reset_password_token(self, expires_in=600):
        return jwt.encode(
            {'reset_password': self.id, 'exp': time() + expires_in},
            Config.IMAGINARIO_AUTH_JWT_SIGNING_SECRETE, algorithm='HS256')

    def get_confirm_email_token(self, expires_in=600):
        return urllib.parse.quote((jwt.encode(
            {'confirm_email': self.id, 'exp': time() + expires_in},
            Config.IMAGINARIO_AUTH_JWT_SIGNING_SECRETE, algorithm='HS256')).encode("utf-8"), safe='')

    def get_totp_uri(self):
        if self.authenticator_app_otp_secret is None or len(self.authenticator_app_otp_secret) != 16:
            self.authenticator_app_otp_secret = base64.b32encode(
                os.urandom(10)).decode('utf-8')
            session.commit()

        return 'otpauth://totp/2FA-Imaginario:{0}?secret={1}&issuer=ImaginarioAI&image=https://app.imaginario.ai/_next/image?url=%2Fimaginario_black.png&w=256&q=75' \
            .format(self.email, self.authenticator_app_otp_secret)

    def verify_totp(self, token):
        return onetimepass.valid_totp(token, self.authenticator_app_otp_secret)

    def to_admin_json(self):
        return {c.name: str(getattr(self, c.name)) for c in self.__table__.columns}

    def to_json(self, collaborators=None):
        usage = session.get('usage')
        package = session.get('package')
        package_owner = session.get('package_owner')
        response_json = {
            'usage': usage,
            'package': package,
            'package_owner': package_owner,
            'collaborators': collaborators,
            'is_owner': True if package_owner is not None and package_owner.get('owner_id') == self.id else False
        }

        self.created_on = self.datetime_created
        for key in self.ui_key_map:
            response_json[key] = getattr(self, key)
        return response_json

    @staticmethod
    def verify_reset_password_token(token):
        try:
            id = jwt.decode(token, Config.IMAGINARIO_AUTH_JWT_SIGNING_SECRETE,
                            algorithms=['HS256'])['reset_password']
        except:
            return
        return User.query.get(id)
