import cv2 
import time
import numpy as np

# Parameters for lucas kanade optical flow
lk_params = dict(winSize = (15, 15), maxLevel = 2, criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# params for ShiTomasi corner detection
feature_params = dict(maxCorners = 100, qualityLevel = 0.3, minDistance = 7, blockSize = 7)

trajectory_len = 40
detect_interval = 1
trajectories = []
frame_idx = 0

cap = cv2.VideoCapture(0)

while True:
    # start time to calculate FPS
    start = time.time()

    # Capture frame-by-frame
    ret, frame = cap.read()
    
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img = frame.copy()

    # Calculate optical flow for a sparse feature set using the iterative Lucas-Kanade Method
    if len(trajectories) > 0:
        img0, img1 = prev_gray, frame_gray
        p0 = np.float32([trajectory[-1] for trajectory in trajectories])
        p1, _st, _err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
        p0r, _st, _err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)
        d = abs(p0-p0r).reshape(-1, 2).max(-1)
        good = d < 1

        new_trajectories = []

        # Get all the trajectories
        for trajectory, (x, y), good_flag in zip(trajectories, p1.reshape(-1, 2), good):
            if not good_flag:
                continue
            trajectory.append((x,y))
            if len(trajectory) > trajectory_len:
                del trajectory[0]
            new_trajectories.append(trajectory)
            # Newest detected point
            radius = 2
            color = (0, 0, 255)
            center = (int(x), int(y))
            cv2.circle(img, center, radius, color)

        trajectories = new_trajectories
        # Draw all the trajectories
        cv2.polylines(img, [np.int32(trajectory) for trajectory in trajectories], False, (0, 255, 0))
        textposition = (20,50)
        fontscale = 1
        fontthickness = 2
        fontcolor = (0, 255, 0)
        cv2.putText(img, 'track count: %d' % len(trajectories), textposition, cv2.FONT_HERSHEY_PLAIN, fontscale, fontcolor, fontthickness)

    # Update interval - When to update and detect new features
    if frame_idx % detect_interval == 0:
        mask = np.zeros_like(frame_gray)
        mask[:] = 255

        # Latest point in trajectory
        for x, y in [np.int32(trajectory[-1]) for trajectory in trajectories]:
            center = (x,y)
            radius = 5
            color = (0, 0, 0)
            cv2.circle(mask, center, radius, color)

        # Detect the good features to track
        p = cv2.goodFeaturesToTrack(frame_gray, mask = mask, **feature_params)
        if p is not None:
            # If good features can be tracked then add that to the trajectories
            for x, y in np.float32(p).reshape(-1, 2):
                trajectories.append([(x, y)])

    frame_idx += 1
    prev_gray = frame_gray

    # End time of a frame
    end = time.time()

    # calculate FPS for current frame detection
    fps = 1 / (end - start)

    # show FPS on window
    textposition = (20,30)
    fontscale = 1
    fontthickness = 2
    fontcolor = (0, 255, 0)
    cv2.putText(img, f"{fps:.2f} FPS", textposition, cv2.FONT_HERSHEY_PLAIN, fontscale, fontcolor, fontthickness)

    # Display the resulting frame and mask
    cv2.imshow('Optical Flow', img)
    cv2.imshow('Mask', mask)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
