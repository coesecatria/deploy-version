from fastapi import APIRouter, Body
from pydantic import BaseModel
from app.services.webrtc_service import create_webrtc_answer

router = APIRouter(prefix="/webrtc", tags=["WebRTC"])

class WebRTCOffer(BaseModel):
    sdp: str
    type: str

@router.post("/offer")
async def webrtc_offer(offer: WebRTCOffer):
    """
    Endpoint to receive a WebRTC SDP offer and return an SDP answer.
    This initiates the P2P connection for the low-latency AI stream.
    """
    answer = await create_webrtc_answer(offer.sdp, offer.type)
    return answer
