'''
#TODO: 
- Sobel Extraction
- Edge Components Tune
- Patch Embedding Tune
- SAM Image Encoder
- SAM Mask decoder
- Adapter layers
'''
import cv2

class SobelExtraction:
    """
    Class to perform Sobel edge detection on an image.

    Args:
        image (numpy.ndarray): The input image on which Sobel edge detection is performed.
    """
    def __init__ (self, image):
        self.image = image
    
    def compute_sobel(self):
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        I_sobel = np.sqrt(sobelx**2 + sobely**2)
        return I_sobel
