# Reading the Road

Reading the Road estimates road width from a single survey or dashcam image and
shows the evidence used for the estimate. It is intended for rapid
infrastructure triage, where a human can review the overlay before using a
measurement operationally.

## Pipeline

1. The image is resized and contrast-normalized.
2. GrabCut proposes the connected road surface from the lower part of the
   image. If the image is too uniform for GrabCut, a conservative trapezoid is
   used and the result is explicitly given low confidence.
3. Scan lines recover the visible left and right road boundaries.
4. Their intersection estimates the vanishing point. Perspective geometry and
   the camera height convert each scan-line span into metres.
5. The median width is reported with scan-line spread, an overlay, and a
   confidence score based on mask coverage, width consistency, image sharpness,
   and horizon plausibility.

## Run

## Setup

streamlit run app.py
```

The camera height is the key calibration input. If it is unknown, the app uses
1.5 m and marks the result as approximate. Use a level camera, a visible road
surface, and several clear scan lines. Low-confidence results should be
verified with a calibrated survey or a known-distance reference.bash
pip install -r requirements.txt