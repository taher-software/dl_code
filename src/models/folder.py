from datetime import datetime
import json
from datetime import datetime

from sqlalchemy.orm import relationship

from src.models.user import User
from src.models.video import Video
from src.main import db

from sqlalchemy import case, Column, DateTime, Enum, ForeignKey, func, Integer, String, Text


class Folder(db.Model):
    __tablename__ = 'folder'
    FK_FOLDER_ID = 'folder.id'

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey(User.FK_USER_ID, name='fk_folder_user_id'))
    owner_id = Column(Integer)
    is_deleted = Column(Integer, default=0)
    is_test = Column(Integer, default=0)
    test_base_id = Column(Integer, default=-1)
    display_name = Column(String(200))
    thumb_image = Column(String(100))
    folder_type = Column(Enum('user', 'test', 'test_admin', name='folder_type_enum'), default='user')
    videos = relationship('FolderVideo', backref='folders', lazy='dynamic', uselist=True)
    video_ids = Column(Text(4294000000))
    datetime_created = Column(DateTime, name='created_at', default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    """TO DROP when v1 deprecated"""
    base_video_ids = Column(Text(4294000000))
    
    def set_from_form(self, form, user_id):
        self.created_at = self.datetime_created
        for key in form:
            setattr(self, key, form[key])
        self.user_id = user_id

    def is_valid(self):
        return self.display_name is not None and self.user_id is not None

    def get_videos(self):
        vids = json.loads(self.video_ids)
        videos = []
        if len(vids) > 0:
            for v in vids:
                try:
                    vid = next(vv for vv in vids if str(vv.id) == str(v))
                    videos.append(vid.to_json())
                except:
                    print('video not found')

        return videos

    def to_json(self):
        self.created_at = self.datetime_created
        folder_columns = {c.name: getattr(self, c.name) for c in self.__table__.columns}
        
        
        try:
            vid_ids = json.loads(self.video_ids)

            reversed_vid_ids = vid_ids[::-1]
            order_case = case(
                {id: index for index, id in enumerate(reversed_vid_ids)},
                value=Video.id
            )
            vids = Video.query.with_entities(Video.video_thumb, Video.gif_path, Video.file_size).filter(Video.id.in_(vid_ids)).order_by(order_case).all()
            folder_size = Video.query.with_entities(func.sum(Video.file_size)).filter(Video.id.in_(vid_ids)).scalar()

            folder_columns['thumbnails'] = [{"thumb": f"/{vid.video_thumb}", "gif": vid.gif_path} for vid in vids]
            folder_columns['folder_size'] = folder_size
        except:
            folder_columns['thumbnails'] = []
            folder_columns['folder_size'] = 0

        return folder_columns