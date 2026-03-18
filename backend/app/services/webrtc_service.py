import asyncio
import numpy as np
import av
from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription
from app.services.stream_manager import streamer
from fractions import Fraction

class AITransformTrack(MediaStreamTrack):
    """
    A video stream track that transforms frames from an RTSP source
    by adding AI annotations (bounding boxes, names, etc.) 
    before sending to the WebRTC peer.
    """
    kind = "video"

    def __init__(self):
        super().__init__()
        self.streamer = streamer
        self._timestamp = 0

    async def recv(self):
        # 1. Pull directly from the high-speed annotated buffer
        # This is the 'zero-layer' approach: no drawing, no encoding, just delivery.
        # Fixed: Using .copy() for thread safety as recommended.
        frame = None
        with self.streamer._annotated_lock:
            if self.streamer._latest_annotated_frame is not None:
                frame = self.streamer._latest_annotated_frame.copy()
        
        if frame is None:
            # Fallback for initialization
            frame = np.zeros((576, 1024, 3), dtype=np.uint8)

        # 2. Convert to av.VideoFrame
        video_frame = av.VideoFrame.from_ndarray(frame, format="bgr24")
        
        # 3. FIXED: Precision Monotonic Clock (90kHz)
        # WebRTC video standard expects a Fraction for time_base.
        self._timestamp += 3000 # 90000 / 30fps
        video_frame.pts = self._timestamp
        video_frame.time_base = Fraction(1, 90000)
        
        # FIXED: Removed manual sleep. aiortc handles pacing internally.
        return video_frame

async def create_webrtc_answer(offer_sdp, offer_type):
    # Professional SDP negotiation with hardened directed transceivers.
    pc = RTCPeerConnection()
    
    # 1. Add Track natively to the connection
    pc.addTrack(AITransformTrack())

    # 2. Handle Connection Lifecycle
    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        if pc.connectionState in ["closed", "failed"]:
            await pc.close()

    # 3. Set Remote Description (Offer)
    offer = RTCSessionDescription(sdp=offer_sdp, type=offer_type)
    await pc.setRemoteDescription(offer)

    # 4. Create and Set Local Description (Answer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return {
        "sdp": pc.localDescription.sdp,
        "type": pc.localDescription.type
    }
