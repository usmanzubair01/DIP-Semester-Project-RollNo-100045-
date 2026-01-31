"""
Shadow Detection and Removal from Images
Using Digital Image Processing Techniques

Author: [Your Name]
Course: Digital Image Pre-Processing
Project: Shadow Detection and Removal System
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional


class ShadowRemover:
    """
    A class to detect and remove shadows from images using HSV color space
    and classical image processing techniques.
    """
    
    def __init__(self, v_threshold: int = 100, s_threshold: int = 80):
        """
        Initialize the Shadow Remover.
        
        Args:
            v_threshold: Threshold for V-channel (brightness). Lower = darker shadows detected
            s_threshold: Threshold for S-channel (saturation). Helps distinguish shadows from dark objects
        """
        self.v_threshold = v_threshold
        self.s_threshold = s_threshold
        
    def read_image(self, image_path: str) -> np.ndarray:
        """
        Read an image from the specified path.
        
        Args:
            image_path: Path to the input image
            
        Returns:
            BGR image as numpy array
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Unable to read image from {image_path}")
        return image
    
    def convert_to_hsv(self, image: np.ndarray) -> np.ndarray:
        """
        Convert BGR image to HSV color space.
        
        Args:
            image: BGR image
            
        Returns:
            HSV image
        """
        return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    def detect_shadows(self, hsv_image: np.ndarray) -> np.ndarray:
        """
        Detect shadow regions using V and S channels.
        
        Args:
            hsv_image: Image in HSV color space
            
        Returns:
            Binary shadow mask (255 = shadow, 0 = non-shadow)
        """
        # Extract H, S, V channels
        h, s, v = cv2.split(hsv_image)
        
        # Create shadow mask using both V and S channels
        # Shadows typically have low brightness (V) and lower saturation (S)
        shadow_mask = np.zeros_like(v)
        shadow_mask[(v < self.v_threshold) & (s < self.s_threshold)] = 255
        
        return shadow_mask
    
    def refine_shadow_mask(self, shadow_mask: np.ndarray) -> np.ndarray:
        """
        Refine shadow mask using morphological operations to remove noise
        and fill gaps.
        
        Args:
            shadow_mask: Binary shadow mask
            
        Returns:
            Refined shadow mask
        """
        # Define morphological kernel
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        
        # Apply closing (dilation followed by erosion) to fill small gaps
        refined_mask = cv2.morphologyEx(shadow_mask, cv2.MORPH_CLOSE, kernel)
        
        # Apply opening (erosion followed by dilation) to remove small noise
        refined_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_OPEN, kernel)
        
        return refined_mask
    
    def create_smooth_mask(self, shadow_mask: np.ndarray) -> np.ndarray:
        """
        Create a smooth transition mask to avoid harsh boundaries.
        
        Args:
            shadow_mask: Binary shadow mask
            
        Returns:
            Smoothed mask with values 0-1
        """
        # Apply Gaussian blur to create smooth transitions
        smooth_mask = cv2.GaussianBlur(shadow_mask, (21, 21), 0)
        
        # Normalize to 0-1 range
        smooth_mask = smooth_mask.astype(np.float32) / 255.0
        
        return smooth_mask
    
    def normalize_shadow_intensity(self, hsv_image: np.ndarray, 
                                   shadow_mask: np.ndarray,
                                   smooth_mask: np.ndarray) -> np.ndarray:
        """
        Normalize intensity in shadow regions while preserving natural appearance.
        
        Args:
            hsv_image: Image in HSV color space
            shadow_mask: Binary shadow mask
            smooth_mask: Smoothed transition mask (0-1)
            
        Returns:
            Enhanced HSV image
        """
        # Create a copy to avoid modifying original
        enhanced_hsv = hsv_image.copy().astype(np.float32)
        h, s, v = cv2.split(enhanced_hsv)
        
        # Calculate mean intensity in shadow and non-shadow regions
        shadow_mean = np.mean(v[shadow_mask == 255]) if np.any(shadow_mask == 255) else 1
        non_shadow_mean = np.mean(v[shadow_mask == 0]) if np.any(shadow_mask == 0) else 1
        
        # Calculate correction factor
        correction_factor = non_shadow_mean / (shadow_mean + 1e-6)
        
        # Limit correction factor to avoid over-brightening
        correction_factor = min(correction_factor, 2.5)
        
        # Apply bilateral filter to shadow regions to preserve edges
        v_corrected = v.copy()
        shadow_region = (shadow_mask == 255)
        
        if np.any(shadow_region):
            # Extract shadow region and apply bilateral filtering
            shadow_v = v[shadow_region]
            
            # Apply intensity correction
            shadow_v_corrected = shadow_v * correction_factor
            
            # Clip values to valid range
            shadow_v_corrected = np.clip(shadow_v_corrected, 0, 255)
            
            # Put corrected values back into full array
            v_corrected = v.copy()
            v_corrected[shadow_region] = shadow_v_corrected
            
            # Blend corrected values using smooth mask for gradual transition
            v_corrected = v * (1 - smooth_mask) + v_corrected * smooth_mask
            v_corrected = np.clip(v_corrected, 0, 255)
        
        # Reduce saturation slightly in corrected shadow areas to prevent color bleeding
        s_corrected = s.copy()
        s_corrected = s * (1 - smooth_mask * 0.15)  # Reduce S by up to 15% in shadows
        s_corrected = np.clip(s_corrected, 0, 255)
        
        # Merge channels back
        enhanced_hsv = cv2.merge([h, s_corrected, v_corrected])
        
        return enhanced_hsv.astype(np.uint8)
    
    def convert_to_bgr(self, hsv_image: np.ndarray) -> np.ndarray:
        """
        Convert HSV image back to BGR color space.
        
        Args:
            hsv_image: Image in HSV color space
            
        Returns:
            BGR image
        """
        return cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
    
    def process_image(self, image_path: str, show_steps: bool = True) -> dict:
        """
        Complete shadow removal pipeline.
        
        Args:
            image_path: Path to input image
            show_steps: Whether to display intermediate steps
            
        Returns:
            Dictionary containing all processing stages
        """
        # Step 1: Read image
        print("Step 1: Reading input image...")
        original_image = self.read_image(image_path)
        
        # Step 2: Convert to HSV
        print("Step 2: Converting RGB to HSV...")
        hsv_image = self.convert_to_hsv(original_image)
        
        # Step 3: Detect shadows
        print("Step 3: Detecting shadow regions...")
        shadow_mask = self.detect_shadows(hsv_image)
        
        # Step 4: Refine shadow mask
        print("Step 4: Refining shadow mask...")
        refined_mask = self.refine_shadow_mask(shadow_mask)
        
        # Step 5: Create smooth transition mask
        print("Step 5: Creating smooth transition mask...")
        smooth_mask = self.create_smooth_mask(refined_mask)
        
        # Step 6: Normalize shadow intensity
        print("Step 6: Normalizing shadow intensity...")
        enhanced_hsv = self.normalize_shadow_intensity(hsv_image, refined_mask, smooth_mask)
        
        # Step 7: Convert back to BGR
        print("Step 7: Converting back to RGB...")
        enhanced_image = self.convert_to_bgr(enhanced_hsv)
        
        print("✓ Shadow removal complete!")
        
        # Store results
        results = {
            'original': cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB),
            'hsv': hsv_image,
            'shadow_mask': shadow_mask,
            'refined_mask': refined_mask,
            'smooth_mask': (smooth_mask * 255).astype(np.uint8),
            'enhanced': cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2RGB)
        }
        
        if show_steps:
            self.visualize_results(results)
        
        return results
    
    def visualize_results(self, results: dict):
        """
        Display all processing steps and results.
        
        Args:
            results: Dictionary containing processing stages
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Shadow Detection and Removal Pipeline', fontsize=16, fontweight='bold')
        
        # Original Image
        axes[0, 0].imshow(results['original'])
        axes[0, 0].set_title('1. Original Image\n(with shadows)', fontsize=12)
        axes[0, 0].axis('off')
        
        # Shadow Mask
        axes[0, 1].imshow(results['shadow_mask'], cmap='gray')
        axes[0, 1].set_title('2. Shadow Detection\n(Binary Mask)', fontsize=12)
        axes[0, 1].axis('off')
        
        # Refined Mask
        axes[0, 2].imshow(results['refined_mask'], cmap='gray')
        axes[0, 2].set_title('3. Refined Mask\n(Morphological Operations)', fontsize=12)
        axes[0, 2].axis('off')
        
        # Smooth Mask
        axes[1, 0].imshow(results['smooth_mask'], cmap='gray')
        axes[1, 0].set_title('4. Smooth Transition Mask\n(Gaussian Blur)', fontsize=12)
        axes[1, 0].axis('off')
        
        # Enhanced Result
        axes[1, 1].imshow(results['enhanced'])
        axes[1, 1].set_title('5. Enhanced Image\n(Shadow-Free)', fontsize=12)
        axes[1, 1].axis('off')
        
        # Side-by-side comparison
        comparison = np.hstack([results['original'], results['enhanced']])
        axes[1, 2].imshow(comparison)
        axes[1, 2].set_title('6. Before vs After\nComparison', fontsize=12)
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def save_results(self, results: dict, output_dir: str = './output'):
        """
        Save all results to disk.
        
        Args:
            results: Dictionary containing processing stages
            output_dir: Directory to save results
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Convert RGB back to BGR for saving
        cv2.imwrite(f'{output_dir}/1_original.jpg', 
                    cv2.cvtColor(results['original'], cv2.COLOR_RGB2BGR))
        cv2.imwrite(f'{output_dir}/2_shadow_mask.jpg', results['shadow_mask'])
        cv2.imwrite(f'{output_dir}/3_refined_mask.jpg', results['refined_mask'])
        cv2.imwrite(f'{output_dir}/4_smooth_mask.jpg', results['smooth_mask'])
        cv2.imwrite(f'{output_dir}/5_enhanced.jpg', 
                    cv2.cvtColor(results['enhanced'], cv2.COLOR_RGB2BGR))
        
        print(f"\n✓ All results saved to '{output_dir}/' directory")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main function to demonstrate shadow removal.
    """
    print("=" * 70)
    print("SHADOW DETECTION AND REMOVAL SYSTEM")
    print("Digital Image Processing Project")
    print("=" * 70)
    print()
    
    # Initialize shadow remover with default parameters
    # Adjust v_threshold and s_threshold based on your image
    shadow_remover = ShadowRemover(v_threshold=100, s_threshold=80)
    
    # Process image - REPLACE WITH YOUR IMAGE PATH
    image_path = 'images.jpg.jpeg'  # Change this to your image path
    
    try:
        results = shadow_remover.process_image(image_path, show_steps=True)
        
        # Save results
        shadow_remover.save_results(results)
        
        print("\n" + "=" * 70)
        print("PROJECT COMPLETE!")
        print("=" * 70)
        
    except FileNotFoundError:
        print(f"\n❌ Error: Image file '{image_path}' not found.")
        print("Please update the 'image_path' variable with your image location.")
    except Exception as e:
        print(f"\n❌ Error occurred: {str(e)}")


if __name__ == "__main__":
    main()
