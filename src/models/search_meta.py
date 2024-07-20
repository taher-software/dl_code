from src.models.saved_clip import SavedClip
from src.models.video import Video
from app import db

from sqlalchemy import Column, event, Integer, UniqueConstraint, String


class SearchMeta(db.Model):
    __tablename__ = 'search_meta'
    
    FK_SAVED_CLIP_ID = 'search_meta.id'

    id = Column(Integer, primary_key=True)
    resource_name = Column(String(12))
    resource_id = Column(Integer)
    resource_prop = db.Column(db.String(20))
    key_words = Column(String(255))

    __table_args__ = (
        UniqueConstraint('resource_name', 'key_words', 'resource_id'),
    )

    def to_json(self):
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}

def clean_keywords(key_words, replace_underscore = '_'):
    key_words = key_words.replace('.mp4', '')
    key_words = key_words.replace('.mp3', '')
    key_words = key_words.replace('.m4v', '')
    key_words = key_words.replace('.mov', '')
    key_words = key_words.replace('_', replace_underscore)
    key_words = key_words.lower()
    return key_words

def append_search_meta_record(resource_name, resource_prop, resource_id, key_words):
    if key_words is None:
        return
    
    key_words = key_words.lower()
    existing_entry = SearchMeta.query.filter(
        SearchMeta.resource_name == resource_name,
        SearchMeta.resource_id == resource_id,
        SearchMeta.key_words == key_words
    ).first()

    if existing_entry is not None: return
    
    search_meta = SearchMeta(
        resource_name = resource_name,
        resource_prop = resource_prop,
        resource_id = resource_id,
        key_words=key_words
    )
    db.session.add(search_meta)

def after_update(resource_name, resource_prop, resource_id, key_words):    
    search_meta_same = SearchMeta.query.filter(
        SearchMeta.resource_name == resource_name, 
        SearchMeta.resource_id == resource_id,
        SearchMeta.resource_prop == resource_prop,
        SearchMeta.key_words == key_words
    ).first()

    if search_meta_same is not None:
        return
    
    if key_words is None:
        SearchMeta.query.filter(
            SearchMeta.resource_name == resource_name, 
            SearchMeta.resource_id == resource_id,
            SearchMeta.resource_prop == resource_prop
        ).delete()
        return
    
    key_words = key_words.lower()
    search_meta_same = SearchMeta.query.filter(
        SearchMeta.resource_name == resource_name, 
        SearchMeta.resource_id == resource_id,
        SearchMeta.key_words == key_words
    ).first()

    if search_meta_same is not None:
        SearchMeta.query.filter(
            SearchMeta.resource_name == resource_name, 
            SearchMeta.resource_id == resource_id,
            SearchMeta.resource_prop == resource_prop
        ).delete(synchronize_session='fetch')

    search_meta_exists = SearchMeta.query.filter(
        SearchMeta.resource_name == resource_name, 
        SearchMeta.resource_id == resource_id,
        SearchMeta.resource_prop == resource_prop
    ).first()

    if search_meta_exists is None:
        append_search_meta_record(resource_name, resource_prop, resource_id, key_words)
        return

    SearchMeta.query.filter(
            SearchMeta.resource_name == resource_name, 
            SearchMeta.resource_id == resource_id,
            SearchMeta.resource_prop == resource_prop
        ).update({SearchMeta.key_words: key_words})

def after_delete(resource_name, resource_id):
    SearchMeta.query.filter(
        SearchMeta.resource_name == resource_name, 
        SearchMeta.resource_id == resource_id
    ).delete(synchronize_session='fetch')
    
@event.listens_for(Video, 'after_insert')
def after_video_insert_listener(_mapper, _connection, target):
    try:
        words = clean_keywords(target.title.lower(), ' ')
        append_search_meta_record('Video', 'title', target.id, words)
    except Exception as err:
        print("insert failed", err)
        pass

@event.listens_for(Video, 'after_update')
def after_video_update_listener(_mapper, _connection, target):
    try:
        words = clean_keywords(target.title.lower(), ' ')
        after_update('Video', 'title', target.id, words)
    except Exception as err:
        print("update failed", err)
        pass

@event.listens_for(Video, 'after_delete')
def after_video_delete_listener(_mapper, _connection, target):
    try:
        after_delete('Video', target.id)
    except Exception as err:
        print("update failed", err)

@event.listens_for(SavedClip, 'after_update')
def after_video_update_listener(_mapper, _connection, target):
    try:
        words = clean_keywords(target.title.lower(), ' ')
        after_update(SavedClip.__name__, 'title', target.id, words)
        from app.beta.models.search_meta_cache import delete_cache
        delete_cache(SavedClip.__name__, target.id)
    except Exception as err:
        print("update failed", err)
        pass

@event.listens_for(SavedClip, 'after_delete')
def after_video_delete_listener(_mapper, _connection, target):
    try:
        after_delete(SavedClip.__name__, target.id)
        from app.beta.models.search_meta_cache import delete_cache
        delete_cache(SavedClip.__name__, target.id)
    except Exception as err:
        print("update failed", err)

@event.listens_for(SavedClip, 'after_insert')
def after_video_insert_listener(_mapper, _connection, target):
    try:
        words = clean_keywords(target.title.lower(), ' ')
        append_search_meta_record(SavedClip.__name__, 'title', target.id, words)
    except Exception as err:
        print("insert failed", err)
        pass
    try:
        words = clean_keywords(target.title.lower(), ' ')
        append_search_meta_record(AppSavedClip.__name__, 'title', target.id, words)
    except Exception as err:
        print("insert failed", err)
        pass