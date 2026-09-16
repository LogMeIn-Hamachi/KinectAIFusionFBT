"""Extract time-labelled review frames from the user-provided screen recording."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'third_party/dev_python'))
import cv2
import numpy as np
video=cv2.VideoCapture(sys.argv[1])
fps=video.get(cv2.CAP_PROP_FPS);duration=video.get(cv2.CAP_PROP_FRAME_COUNT)/fps
out=ROOT/'artifacts/video-review'/Path(sys.argv[1]).stem;out.mkdir(parents=True,exist_ok=True)
frames=[]
for t in range(0,int(duration),4):
    video.set(cv2.CAP_PROP_POS_MSEC,t*1000);ok,frame=video.read()
    if not ok:continue
    cv2.imwrite(str(out/f'frame-{t:02d}s.jpg'),frame)
    small=cv2.resize(frame,(768,432));cv2.putText(small,f'{t}s',(12,28),cv2.FONT_HERSHEY_SIMPLEX,.8,(0,255,255),2)
    frames.append(small)
while len(frames)%2:frames.append(np.zeros_like(frames[0]))
sheet=np.vstack([np.hstack(frames[i:i+2]) for i in range(0,len(frames),2)])
cv2.imwrite(str(out/'contact-sheet.jpg'),sheet)
print('duration',duration,'review_frames',len(frames),'sheet',out/'contact-sheet.jpg')
