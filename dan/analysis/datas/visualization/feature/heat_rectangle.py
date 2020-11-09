import cv2
import numpy as np

def heat_rect(preds, predicted_class, original_image):
    # Find the n x m score map for the predicted class
    score_map = preds[0, predicted_class, :, :].cpu().numpy()
    score_map = score_map[0]

    # Resize score map to the original image size
    score_map = cv2.resize(score_map, (original_image.shape[1], original_image.shape[0]))

    # Binarize score map
    _, score_map_for_contours = cv2.threshold(score_map, 0.25, 1, type=cv2.THRESH_BINARY)
    score_map_for_contours = score_map_for_contours.astype(np.uint8).copy()

    # Find the countour of the binary blob
    _, contours, _ = cv2.findContours(score_map_for_contours, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)

    # Find bounding box around the object.
    rect = cv2.boundingRect(contours[0])
    
    # Apply score map as a mask to original image
    score_map = score_map - np.min(score_map[:])
    score_map = score_map / np.max(score_map[:])

    # correspond times original
    score_map = cv2.cvtColor(score_map, cv2.COLOR_GRAY2BGR)
    masked_image = (original_image * score_map).astype(np.uint8)

    # Display bounding box
    cv2.rectangle(masked_image, rect[:2], (rect[0] + rect[2], rect[1] + rect[3]), (0, 0, 255), 2)

    # # Display images
    # cv2.imshow("Original Image", original_image)
    # cv2.imshow("scaled_score_map", score_map)
    # cv2.imshow("activations_and_bbox", masked_image)
    # cv2.waitKey(0)
    return rect

