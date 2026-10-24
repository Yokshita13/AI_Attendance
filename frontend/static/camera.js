// Small webcam helper shared by the scan and register pages.
const Cam = {
  async start(video) {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' }, audio: false
    });
    video.srcObject = stream;
    await new Promise(r => (video.onloadedmetadata = r));
    return stream;
  },
  // Capture the current frame as a JPEG data URL.
  grab(video, quality = 0.85) {
    const c = document.createElement('canvas');
    c.width = video.videoWidth; c.height = video.videoHeight;
    c.getContext('2d').drawImage(video, 0, 0);
    return c.toDataURL('image/jpeg', quality);
  }
};