"""
Video data model classes.
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any


@dataclass
class Video:
    """
    Represents a YouTube video with its metadata.
    """
    id: str
    title: str
    views: int
    publish_time: datetime
    vph: float = 0.0
    channel_id: Optional[str] = None
    channel_title: Optional[str] = None
    description: Optional[str] = None
    tags: list[str] = field(default_factory=list)
    thumbnail_url: Optional[str] = None
    duration_seconds: Optional[int] = None
    transcript: Optional[str] = None
    last_updated: datetime = field(default_factory=datetime.now)
    
    @classmethod
    def from_tuple(cls, video_tuple):
        """
        Create a Video object from a tuple of (video_id, views, publish_time, vph).
        
        Args:
            video_tuple: Tuple containing video data
            
        Returns:
            Video: A new Video object
        """
        video_id, views, publish_time, vph = video_tuple
        
        # Convert publish_time to datetime if it's a string
        if isinstance(publish_time, str):
            try:
                if 'Z' in publish_time:
                    publish_time = datetime.fromisoformat(publish_time.replace('Z', ''))
                else:
                    publish_time = datetime.fromisoformat(publish_time)
            except ValueError:
                publish_time = datetime.now()
        
        return cls(
            id=video_id,
            title="",  # Title is unknown from the tuple
            views=views,
            publish_time=publish_time,
            vph=vph
        )
    
    def to_tuple(self):
        """
        Convert to a tuple of (video_id, views, publish_time, vph).
        
        Returns:
            tuple: Video data as a tuple
        """
        # Convert publish_time to ISO format if it's a datetime
        publish_time = self.publish_time.isoformat() if isinstance(self.publish_time, datetime) else self.publish_time
        
        return (self.id, self.views, publish_time, self.vph)
    
    def update_vph(self, now: Optional[datetime] = None):
        """
        Recalculate views per hour based on current time.
        
        Args:
            now: Current datetime (default: now)
            
        Returns:
            float: Updated VPH
        """
        if now is None:
            now = datetime.now()
            
        # Calculate hours since publication
        hours_since = max((now - self.publish_time).total_seconds() / 3600, 1)
        
        # Update VPH
        self.vph = self.views / hours_since
        self.last_updated = now
        
        return self.vph
    
    @classmethod
    def from_youtube_api(cls, item: Dict[str, Any]):
        """
        Create a Video object from a YouTube API response item.
        
        Args:
            item: YouTube API response item
            
        Returns:
            Video: A new Video object
        """
        video_id = item["id"] if isinstance(item["id"], str) else item["id"]["videoId"]
        snippet = item.get("snippet", {})
        statistics = item.get("statistics", {})
        
        # Parse publish time
        publish_time = snippet.get("publishedAt")
        if publish_time:
            if 'Z' in publish_time:
                publish_time = datetime.fromisoformat(publish_time.replace('Z', ''))
            else:
                publish_time = datetime.fromisoformat(publish_time)
        else:
            publish_time = datetime.now()
        
        # Get views
        views = int(statistics.get("viewCount", 0))
        
        # Calculate VPH
        hours_since = max((datetime.now() - publish_time).total_seconds() / 3600, 1)
        vph = views / hours_since
        
        return cls(
            id=video_id,
            title=snippet.get("title", ""),
            views=views,
            publish_time=publish_time,
            vph=vph,
            channel_id=snippet.get("channelId"),
            channel_title=snippet.get("channelTitle"),
            description=snippet.get("description"),
            tags=snippet.get("tags", []),
            thumbnail_url=snippet.get("thumbnails", {}).get("default", {}).get("url")
        )
        
    def to_dict(self):
        """
        Convert to dictionary representation.
        
        Returns:
            dict: Video data as a dictionary
        """
        return {
            "id": self.id,
            "title": self.title,
            "views": self.views,
            "publish_time": self.publish_time.isoformat() if isinstance(self.publish_time, datetime) else self.publish_time,
            "vph": self.vph,
            "channel_id": self.channel_id,
            "channel_title": self.channel_title,
            "description": self.description,
            "tags": self.tags,
            "thumbnail_url": self.thumbnail_url,
            "duration_seconds": self.duration_seconds,
            "transcript": self.transcript,
            "last_updated": self.last_updated.isoformat() if isinstance(self.last_updated, datetime) else self.last_updated
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        """
        Create a Video object from a dictionary.
        
        Args:
            data: Dictionary containing video data
            
        Returns:
            Video: A new Video object
        """
        # Convert ISO format strings to datetime objects
        publish_time = data.get("publish_time")
        if isinstance(publish_time, str):
            try:
                publish_time = datetime.fromisoformat(publish_time.replace('Z', ''))
            except ValueError:
                publish_time = datetime.now()
        
        last_updated = data.get("last_updated")
        if isinstance(last_updated, str):
            try:
                last_updated = datetime.fromisoformat(last_updated.replace('Z', ''))
            except ValueError:
                last_updated = datetime.now()
        
        return cls(
            id=data.get("id", ""),
            title=data.get("title", ""),
            views=data.get("views", 0),
            publish_time=publish_time,
            vph=data.get("vph", 0.0),
            channel_id=data.get("channel_id"),
            channel_title=data.get("channel_title"),
            description=data.get("description"),
            tags=data.get("tags", []),
            thumbnail_url=data.get("thumbnail_url"),
            duration_seconds=data.get("duration_seconds"),
            transcript=data.get("transcript"),
            last_updated=last_updated
        )