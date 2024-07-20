from src.models.search_meta import SearchMeta
from app import db

from datetime import datetime
from sqlalchemy import Column, DateTime, event, Integer, or_, Text, String


class SearchMetaCache(db.Model):
    __tablename__ = 'search_meta_cache'
    
    FK_SAVED_CLIP_ID = 'search_meta_cache.id'

    id = Column(Integer, primary_key=True)
    resource_name = Column(String(12))
    search_query = Column(String(100))
    key_words_hash = Column(String(64))
    resource_id_hash = Column(String(64))
    ordered_resource_ids = Column(Text(4294000000))

    created_at = Column(DateTime, default=datetime.utcnow)

    def to_json(self):
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}

def delete_cache(resource_name, resource_id):
    # Convert the resource_id to a string representation that matches JSON formatting
    # Ensuring the search pattern accounts for the JSON array structure
    search_pattern_exact = f'[{resource_id}]'  # Case where the ID is the only element in the list
    search_pattern_start = f'[{resource_id},%'  # Case where the ID is at the start of the list
    search_pattern_middle = f'%, {resource_id},%'  # Case where the ID is in the middle
    search_pattern_end = f'%, {resource_id}]'  # Case where the ID is at the end of the list

    # Use the LIKE operator to search for the pattern in the serialized JSON stored in a TEXT field
    SearchMetaCache.query.filter(
        SearchMetaCache.resource_name == resource_name,
        or_(
            SearchMetaCache.ordered_resource_ids.like(search_pattern_exact),
            SearchMetaCache.ordered_resource_ids.like(search_pattern_start),
            SearchMetaCache.ordered_resource_ids.like(search_pattern_middle),
            SearchMetaCache.ordered_resource_ids.like(search_pattern_end)
        )
    ).delete(synchronize_session='fetch')

    return


@event.listens_for(SearchMeta, 'after_update')
def after_meta_update_listener(_mapper, _connection, target):
    delete_cache(target.resource_name, target.resource_id)
    print(target)

@event.listens_for(SearchMeta, 'after_delete')
def after_search_meta_delete_listener(_mapper, _connection, target):
    delete_cache(target.resource_name, target.resource_id)
    print(target)