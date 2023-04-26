import cv2
import numpy as np

# Load the drone and satellite images
drone_img = cv2.imread('Drone.jpg')  ## High Resolution image
sat_img = cv2.imread('sat.jpg')  ## Low Resolution image

# Perform image alignment
sift = cv2.xfeatures2d.SIFT_create()
kp1, desc1 = sift.detectAndCompute(drone_img, None)
kp2, desc2 = sift.detectAndCompute(sat_img, None)
bf = cv2.BFMatcher()
matches = bf.knnMatch(desc1, desc2, k=2)
good_matches = []
for m,n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)
src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1,1,2)
dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1,1,2)
M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
sat_img_aligned = cv2.warpPerspective(sat_img, M, (drone_img.shape[1], drone_img.shape[0]))

# Resize the satellite image
sat_img_resized = cv2.resize(sat_img_aligned, (drone_img.shape[1], drone_img.shape[0]), interpolation=cv2.INTER_CUBIC)

# Optionally, perform super-resolution on the satellite image
# ...

# Blend the images together
alpha = 0.4
beta = 0.9#1.0 - alpha
output_img = cv2.addWeighted(drone_img, alpha, sat_img_resized, beta, 0.0)

# Save the output image
cv2.imwrite('Overlap_Stacked_Image(Low resolution on top of High resolution).jpg', output_img)
