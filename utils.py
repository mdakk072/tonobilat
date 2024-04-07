import os
import cv2
import numpy as np
from const import IMG_SIZE
class ImageLabelMatcher:
    """
    Matches images with their corresponding labels,given a start path.
    """
    def __init__(self, start_path):
        self.start_path = start_path
        if not os.path.exists(start_path) or any (not os.path.exists(os.path.join(start_path, folder)) for folder in ['images', 'labels']):
            raise FileNotFoundError(f"Path {start_path} does not exist or does not contain images and labels folders")
        self.images_path = os.path.join(start_path, 'images')
        self.labels_path = os.path.join(start_path, 'labels')
        self.images=  {os.path.splitext(f)[0]: os.path.join(self.images_path, f) for f in os.listdir(self.images_path) if f.endswith('.jpg')}
        self.labels = {os.path.splitext(f)[0]: os.path.join(self.labels_path, f) for f in os.listdir(self.labels_path) if f.endswith(('.txt', '.xml'))}
    def get_matched_files(self):
        return [(self.images[n], self.labels[n]) for n in self.images if n in self.labels]

class BoxDrawer:
    @staticmethod
    def draw_box_on_image(img, coords, text=None, fill=False, text_color=(255, 255, 255), box_color=(0, 255, 0)):
        # Extract the normalized box coordinates
        cx, cy, w, h = coords

        # Convert from normalized to absolute pixel coordinates
        x = int((cx - w / 2) * IMG_SIZE)
        y = int((cy - h / 2) * IMG_SIZE)
        width = int(w * IMG_SIZE)
        height = int(h * IMG_SIZE)

        # Calculate the top-left and bottom-right corners of the rectangle
        start_point = (x, y)
        end_point = (x + width, y + height)

        # Define the color of the rectangle and its thickness
        thickness = -1 if fill else 2  # Fill the box if fill is True

        # Draw the rectangle on the image
        img_with_box = cv2.rectangle(img.copy(), start_point, end_point, box_color, thickness)

        # If text is provided, draw it
        if text is not None:
            # Define the font, scale, thickness, and color for the text
            font = cv2.FONT_HERSHEY_TRIPLEX
            font_scale = 0.5
            font_thickness = 1
        # White in BGR

            # Calculate the size of the text
            text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)

            # Adjust text position based on fill
            if fill:
                # Center the text inside the box
                text_position = (x + (width - text_size[0]) // 2, y + (height + text_size[1]) // 2)
            else:
                # Position the text slightly above the box
                text_position = (x, y - 10)

            # Draw the text
            cv2.putText(img_with_box, text, text_position, font, font_scale, text_color, font_thickness)

        return img_with_box

    @staticmethod
    def draw_mask_on_image(img, coords, text=None, fill=False, text_color=(255, 255, 255), mask_color=(0, 255, 0)):
        """
        Draw a mask on the image.

        Parameters:
        - img: The image on which to draw.
        - coords: The coordinates of the mask's outline.
        - text: Optional text to display on or near the mask.
        - fill: If True, fill the mask; otherwise, just draw the outline.
        - text_color: The color of the text (if specified).
        - mask_color: The color to use for the mask or its outline.
        """

        # Prepare the mask coordinates
        coords = np.array(coords, dtype=np.int32).reshape((-1, 1, 2))

        # Draw the filled mask or just its outline
        if fill:
            cv2.fillPoly(img, [coords], mask_color)
        else:
            cv2.polylines(img, [coords], isClosed=True, color=mask_color, thickness=2)

        # If text is provided, draw it
        if text is not None:
            font = cv2.FONT_HERSHEY_TRIPLEX
            font_scale = 0.5
            font_thickness = 1

            # Calculate the bounding rect and use it to place the text
            x, y, w, h = cv2.boundingRect(coords)
            if fill or 1: #bghina dima nte7o hnaya
                # Center the text inside the mask
                text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
                text_position = (x + (w - text_size[0]) // 2, y + (h + text_size[1]) // 2)
            else:
                # Position the text outside or above the mask
                text_position = (x, y - 10)

            # Draw the text
            cv2.putText(img, text, text_position, font, font_scale, text_color, font_thickness)

        return img
    
    @staticmethod
    def combine_images_with_captions(img1, caption1, img2, caption2):
        # Determine the maximum height of the two images to make them the same height
        max_height = max(img1.shape[0], img2.shape[0])
        # Resize images to have the same height
        img1_resized = cv2.resize(img1, (int(img1.shape[1] * max_height / img1.shape[0]), max_height))
        img2_resized = cv2.resize(img2, (int(img2.shape[1] * max_height / img2.shape[0]), max_height))
        # Combine images side by side
        combined_img = np.hstack((img1_resized, img2_resized))
        # Define font for the caption
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        font_thickness = 2
        text_color = (255, 255, 255)  # White
        # Calculate space needed for captions based on text size
        text_size1 = cv2.getTextSize(caption1, font, font_scale, font_thickness)[0]
        text_size2 = cv2.getTextSize(caption2, font, font_scale, font_thickness)[0]
        # Calculate the position for each caption to be centered under its image
        x1 = (img1_resized.shape[1] - text_size1[0]) // 2
        x2 = img1_resized.shape[1] + (img2_resized.shape[1] - text_size2[0]) // 2
        # Ensure there's enough space for captions
        caption_space = max(text_size1[1], text_size2[1]) + 10  # 10 pixels padding
        combined_img_with_caption_space = np.zeros((combined_img.shape[0] + caption_space, combined_img.shape[1], 3), dtype=np.uint8)
        # Copy the combined image to the new canvas
        combined_img_with_caption_space[:combined_img.shape[0], :combined_img.shape[1]] = combined_img
        # Put captions under each image
        cv2.putText(combined_img_with_caption_space, caption1, (x1, combined_img.shape[0] + caption_space - 5), font, font_scale, text_color, font_thickness)
        cv2.putText(combined_img_with_caption_space, caption2, (x2, combined_img.shape[0] + caption_space - 5), font, font_scale, text_color, font_thickness)
        return combined_img_with_caption_space

    @staticmethod
    def superpose_images(img1, img2):
        # Ensure both images are the same size
        if img1.shape != img2.shape:
            # Resize img2 to match img1's dimensions
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
        
        # Superpose images by averaging their values
        superposed_img = cv2.addWeighted(img1, 0.5, img2, 0.5, 0)

        return superposed_img
