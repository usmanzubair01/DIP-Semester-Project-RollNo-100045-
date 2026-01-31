"""
QUICK REFERENCE GUIDE
Shadow Detection & Removal - Parameter Tuning Cheat Sheet
"""

# ============================================================================
# PARAMETER QUICK REFERENCE
# ============================================================================

"""
v_threshold (Brightness)
│
├─ 60-80   → Very dark shadows (heavy tree shade, indoor corners)
├─ 80-100  → Moderate shadows (outdoor midday, soft indoor lighting)
├─ 100-120 → Light shadows (cloudy day, diffused lighting)
└─ 120-140 → Very light shadows (overcast, minimal contrast)

s_threshold (Saturation)
│
├─ 50-70   → Colored shadows (sunset, colored lights)
├─ 70-90   → Normal shadows (standard outdoor/indoor)
└─ 90-110  → Grayscale shadows (black&white-like, minimal color)
"""

# ============================================================================
# COMMON SCENARIOS - COPY & PASTE READY
# ============================================================================

# SCENARIO 1: Outdoor Image - Strong Sun Shadows
"""
Example: Tree shadows on pavement, building shadows on street
"""
from shadow_removal import ShadowRemover

shadow_remover = ShadowRemover(v_threshold=80, s_threshold=70)
results = shadow_remover.process_image('outdoor_scene.jpg')
shadow_remover.save_results(results)


# SCENARIO 2: Indoor Image - Soft Object Shadows
"""
Example: Lamp shadows on wall, person shadow on floor
"""
shadow_remover = ShadowRemover(v_threshold=110, s_threshold=85)
results = shadow_remover.process_image('indoor_scene.jpg')
shadow_remover.save_results(results)


# SCENARIO 3: Document Scan - Uneven Lighting
"""
Example: Scanned paper with shadow from hand/phone
"""
shadow_remover = ShadowRemover(v_threshold=120, s_threshold=95)
results = shadow_remover.process_image('document_scan.jpg')
shadow_remover.save_results(results)


# SCENARIO 4: Portrait Photo - Face Shadows
"""
Example: Face with shadow from nose, chin, or harsh lighting
"""
shadow_remover = ShadowRemover(v_threshold=90, s_threshold=75)
results = shadow_remover.process_image('portrait.jpg')
shadow_remover.save_results(results)


# SCENARIO 5: Very Dark Shadows (Almost Black)
"""
Example: Deep shade under trees, dark alley shadows
"""
shadow_remover = ShadowRemover(v_threshold=60, s_threshold=60)
results = shadow_remover.process_image('dark_shadows.jpg')
shadow_remover.save_results(results)


# SCENARIO 6: Batch Processing Multiple Images
"""
Process all images in a folder
"""
import os
from shadow_removal import ShadowRemover

shadow_remover = ShadowRemover(v_threshold=100, s_threshold=80)
image_folder = './input_images/'
output_folder = './batch_output/'

for filename in os.listdir(image_folder):
    if filename.endswith(('.jpg', '.png', '.jpeg')):
        image_path = os.path.join(image_folder, filename)
        print(f"\nProcessing: {filename}")
        
        results = shadow_remover.process_image(image_path, show_steps=False)
        
        # Save with original filename
        output_subdir = os.path.join(output_folder, filename.split('.')[0])
        shadow_remover.save_results(results, output_dir=output_subdir)

print("\n✓ Batch processing complete!")


# ============================================================================
# CUSTOM MODIFICATIONS - ADVANCED USERS
# ============================================================================

# MODIFICATION 1: Stronger Shadow Correction
"""
If shadows are still too dark after processing:
In shadow_removal.py, line ~180, change:
    correction_factor = min(correction_factor, 2.5)
To:
    correction_factor = min(correction_factor, 3.5)
"""

# MODIFICATION 2: Preserve More Color
"""
If colors look washed out:
In shadow_removal.py, line ~201, change:
    s_corrected = s * (1 - smooth_mask * 0.15)
To:
    s_corrected = s * (1 - smooth_mask * 0.05)
"""

# MODIFICATION 3: Smoother Transitions
"""
If you see visible edges between shadow/non-shadow:
In shadow_removal.py, line ~131, change:
    smooth_mask = cv2.GaussianBlur(shadow_mask, (21, 21), 0)
To:
    smooth_mask = cv2.GaussianBlur(shadow_mask, (41, 41), 0)
"""

# MODIFICATION 4: More Aggressive Noise Removal
"""
If shadow mask has lots of small dots:
In shadow_removal.py, line ~102, change:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
To:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
"""


# ============================================================================
# TROUBLESHOOTING DECISION TREE
# ============================================================================

"""
START: Is your result satisfactory?
│
NO → What's the problem?
│
├─ Too much detected as shadow?
│   └─ Increase BOTH v_threshold (+20) and s_threshold (+20)
│
├─ Shadows missed/not detected?
│   └─ Decrease BOTH v_threshold (-20) and s_threshold (-20)
│
├─ Colors look unnatural/washed out?
│   └─ Reduce saturation correction (0.15 → 0.05)
│
├─ Image too bright after correction?
│   └─ Reduce correction_factor limit (2.5 → 2.0)
│
├─ Harsh visible boundaries?
│   └─ Increase Gaussian blur kernel (21 → 31 or 41)
│
├─ Small noise/dots in shadow mask?
│   └─ Increase morphological kernel size (7 → 11)
│
└─ Dark shadows still too dark?
    └─ Increase correction_factor limit (2.5 → 3.0 or 3.5)
"""


# ============================================================================
# TESTING YOUR PARAMETERS
# ============================================================================

def test_multiple_parameters(image_path):
    """
    Test different parameter combinations to find optimal values.
    """
    import matplotlib.pyplot as plt
    from shadow_removal import ShadowRemover
    
    # Test different combinations
    params_list = [
        (80, 70),   # Strong shadows
        (100, 80),  # Default
        (120, 90),  # Light shadows
    ]
    
    fig, axes = plt.subplots(1, len(params_list) + 1, figsize=(20, 5))
    
    # Show original
    original = cv2.imread(image_path)
    axes[0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Original', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # Test each parameter set
    for idx, (v_thresh, s_thresh) in enumerate(params_list):
        shadow_remover = ShadowRemover(v_threshold=v_thresh, s_threshold=s_thresh)
        results = shadow_remover.process_image(image_path, show_steps=False)
        
        axes[idx + 1].imshow(results['enhanced'])
        axes[idx + 1].set_title(f'V={v_thresh}, S={s_thresh}', 
                               fontsize=14, fontweight='bold')
        axes[idx + 1].axis('off')
    
    plt.tight_layout()
    plt.savefig('parameter_comparison.jpg', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("✓ Comparison saved as 'parameter_comparison.jpg'")


# Example usage:
# test_multiple_parameters('your_image.jpg')


# ============================================================================
# EVALUATION METRICS (FOR ACADEMIC REPORTS)
# ============================================================================

def calculate_metrics(original, enhanced, shadow_mask):
    """
    Calculate quality metrics for your report.
    """
    import cv2
    import numpy as np
    
    # Convert to grayscale
    gray_original = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
    gray_enhanced = cv2.cvtColor(enhanced, cv2.COLOR_RGB2GRAY)
    
    # Calculate shadow region statistics
    shadow_pixels = (shadow_mask == 255)
    
    if np.any(shadow_pixels):
        # Mean intensity improvement in shadow regions
        original_shadow_mean = np.mean(gray_original[shadow_pixels])
        enhanced_shadow_mean = np.mean(gray_enhanced[shadow_pixels])
        improvement = ((enhanced_shadow_mean - original_shadow_mean) / 
                      original_shadow_mean * 100)
        
        # Shadow coverage percentage
        shadow_coverage = (np.sum(shadow_pixels) / shadow_pixels.size) * 100
        
        # Standard deviation (uniformity)
        original_std = np.std(gray_original[shadow_pixels])
        enhanced_std = np.std(gray_enhanced[shadow_pixels])
        
        print("\n" + "="*50)
        print("IMAGE QUALITY METRICS")
        print("="*50)
        print(f"Shadow Coverage: {shadow_coverage:.2f}%")
        print(f"Original Shadow Mean Intensity: {original_shadow_mean:.2f}")
        print(f"Enhanced Shadow Mean Intensity: {enhanced_shadow_mean:.2f}")
        print(f"Intensity Improvement: {improvement:.2f}%")
        print(f"Original Std Dev: {original_std:.2f}")
        print(f"Enhanced Std Dev: {enhanced_std:.2f}")
        print("="*50)
    else:
        print("No shadows detected!")


# Example usage in your main code:
# from shadow_removal import ShadowRemover
# shadow_remover = ShadowRemover()
# results = shadow_remover.process_image('image.jpg')
# calculate_metrics(results['original'], results['enhanced'], results['shadow_mask'])
