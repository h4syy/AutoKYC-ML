from pydantic import BaseModel
from typing import List, Dict, Any

# Detection schema for document detection
class Detection(BaseModel):
    session_id: str
    csid: str
    id_type: int
    predicted_class: str
    document_photo_path: str
    bounding_box: str
    confidence: float
    details: Dict[str, Any] = {}
    msisdn: int

# Response schema for document detection
class DetectionResponse(BaseModel):
    detections: List[Detection]

# Model for face comparison result
class FaceComparisonResult(BaseModel):
    similarity: float
    bounding_box: Dict[str, float]

# Response schema for face comparison
class FaceComparisonResponse(BaseModel):
    source_image_bounding_box: Dict[str, float]
    face_matches: List[FaceComparisonResult]
    unmatched_faces: List[Dict[str, Any]]
    msisdn: int
    session_id: str

# Schema for liveness data
class LivenessData(BaseModel):
    session_id: str
    csid: str
    liveness_photo_path: str
    bounding_box: str
    confidence: float
    status: str
    msisdn: int
    details: Dict[str, Any]