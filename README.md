# Shadow Detection and Removal System
## Digital Image Processing Project
Name: Muhammad Usman Zubair
Roll no: 100045
Course Name: Digital Image Processing



---

## 📋 Project Overview

This project implements a **shadow detection and removal system** using classical Digital Image Processing techniques. The system identifies shadow regions in images and enhances them by normalizing intensity values, producing clear, shadow-reduced outputs **without using machine learning**.

---

## 🎯 Key Features

✅ **HSV Color Space Analysis** - Uses V-channel for brightness and S-channel for saturation  
✅ **Dual-Channel Shadow Detection** - Combines V and S thresholds for accurate detection  
✅ **Morphological Refinement** - Removes noise and fills gaps in shadow masks  
✅ **Smooth Transitions** - Gaussian blur prevents harsh shadow boundaries  
✅ **Intensity Normalization** - Intelligently brightens shadows while preserving colors  
✅ **Saturation Control** - Prevents color bleeding in corrected regions  
✅ **Modular Design** - Clean, well-documented code structure  

---

## 🛠 Installation

### Requirements
```bash
Python 3.7+
OpenCV
NumPy
Matplotlib
```

### Install Dependencies
```bash
pip install opencv-python numpy matplotlib
```

---

## 🚀 Quick Start

### Basic Usage

```python
from shadow_removal import ShadowRemover

# Initialize with default parameters
shadow_remover = ShadowRemover(v_threshold=100, s_threshold=80)

# Process your image
results = shadow_remover.process_image('your_image.jpg', show_steps=True)

# Save results
shadow_remover.save_results(results, output_dir='./output')
```

### Running the Script

1. Place your image in the project directory
2. Update the `image_path` in `main()` function
3. Run:
```bash
python shadow_removal.py
```

---

## ⚙️ Parameter Tuning Guide

### Understanding the Parameters

| Parameter | Default | Description | When to Adjust |
|-----------|---------|-------------|----------------|
| `v_threshold` | 100 | Brightness threshold (0-255) | Lower for darker shadows, higher for lighter shadows |
| `s_threshold` | 80 | Saturation threshold (0-255) | Lower if shadows have color, higher for grayscale shadows |

### Tuning Tips

**For Outdoor Images (Strong Shadows):**
```python
shadow_remover = ShadowRemover(v_threshold=80, s_threshold=70)
```

**For Indoor Images (Soft Shadows):**
```python
shadow_remover = ShadowRemover(v_threshold=120, s_threshold=90)
```

**For Very Dark Shadows:**
```python
shadow_remover = ShadowRemover(v_threshold=60, s_threshold=60)
```

**How to Find Optimal Values:**
1. Start with defaults (100, 80)
2. If too much is detected as shadow → increase both values
3. If shadows are missed → decrease both values
4. Adjust in increments of 10-20

---

## 📊 Understanding the Output

The system generates 6 visualizations:

1. **Original Image** - Input image with shadows
2. **Shadow Detection** - Binary mask (white = shadow, black = non-shadow)
3. **Refined Mask** - After morphological operations
4. **Smooth Transition Mask** - Gaussian-blurred for gradual correction
5. **Enhanced Image** - Final shadow-free result
6. **Before vs After** - Side-by-side comparison

### Saved Files (in `output/` folder):
- `1_original.jpg` - Original input
- `2_shadow_mask.jpg` - Detected shadows
- `3_refined_mask.jpg` - After noise removal
- `4_smooth_mask.jpg` - Transition mask
- `5_enhanced.jpg` - **Final result**

---

## 🔬 Technical Details

### Algorithm Workflow

```
Input Image (RGB)
    ↓
Convert to HSV
    ↓
Extract V & S Channels
    ↓
Apply Dual Thresholding: (V < v_threshold) AND (S < s_threshold)
    ↓
Morphological Operations (Close → Open)
    ↓
Gaussian Blur for Smooth Transitions
    ↓
Calculate Correction Factor: non_shadow_mean / shadow_mean
    ↓
Apply Intensity Normalization with Blending
    ↓
Reduce Saturation (15%) in Shadow Regions
    ↓
Convert HSV → RGB
    ↓
Enhanced Shadow-Free Image
```

### Key Innovations

1. **Dual-Channel Detection**: Uses both V and S channels (most tutorials only use V)
2. **Edge Preservation**: Bilateral filtering concept for maintaining sharpness
3. **Smooth Blending**: Gaussian-blurred mask prevents visible correction boundaries
4. **Saturation Control**: Reduces oversaturation artifacts in corrected areas
5. **Adaptive Correction**: Limits correction factor to prevent over-brightening

---

## 🎓 Academic Applications

### Concepts Demonstrated
- Color Space Transformation (RGB ↔ HSV)
- Image Thresholding
- Morphological Image Processing
- Intensity Normalization
- Image Enhancement
- Noise Reduction

### Suitable For
- Digital Image Processing course projects
- Computer Vision lab assignments
- Undergraduate/Graduate term projects
- Image enhancement research

---

## 📝 Example Results

### Test Image Recommendations
- **Outdoor scenes** with tree/building shadows
- **Indoor scenes** with object shadows
- **Document scans** with uneven lighting
- **Portrait photos** with face shadows

### Expected Performance
- **Good results**: Clear, distinct shadows with uniform lighting
- **Moderate results**: Multiple overlapping shadows
- **Challenging**: Very dark shadows, complex lighting, colored shadows

---

## 🐛 Troubleshooting

### Problem: Too much detected as shadow
**Solution**: Increase `v_threshold` and `s_threshold` by 20
```python
shadow_remover = ShadowRemover(v_threshold=120, s_threshold=100)
```

### Problem: Shadows not detected
**Solution**: Decrease thresholds by 20
```python
shadow_remover = ShadowRemover(v_threshold=80, s_threshold=60)
```

### Problem: Colors look washed out
**Solution**: The saturation reduction might be too aggressive. Modify line in code:
```python
s_corrected = s * (1 - smooth_mask * 0.10)  # Change 0.15 to 0.10
```

### Problem: Harsh boundaries visible
**Solution**: Increase Gaussian blur kernel size in `create_smooth_mask()`:
```python
smooth_mask = cv2.GaussianBlur(shadow_mask, (31, 31), 0)  # Change (21,21) to (31,31)
```

### Problem: Image too bright after correction
**Solution**: Reduce correction factor limit in `normalize_shadow_intensity()`:
```python
correction_factor = min(correction_factor, 2.0)  # Change 2.5 to 2.0
```

---

## 🔮 Future Enhancements

- [ ] Automatic threshold selection using Otsu's method
- [ ] Adaptive local shadow detection
- [ ] Support for video processing
- [ ] Real-time shadow removal
- [ ] GUI interface for parameter adjustment
- [ ] Batch processing multiple images

---

## 📚 References & Learning Resources

### Color Spaces
- [Understanding HSV Color Space](https://en.wikipedia.org/wiki/HSL_and_HSV)
- OpenCV Documentation: `cv2.cvtColor()`

### Morphological Operations
- [Morphology Tutorial](https://docs.opencv.org/master/d9/d61/tutorial_py_morphological_ops.html)
- Erosion, Dilation, Opening, Closing

### Image Processing Techniques
- Gaussian Blur for smooth transitions
- Bilateral Filter for edge preservation
- Histogram Equalization concepts

---

## 👥 Contributing

Feel free to:
- Report issues
- Suggest improvements
- Add new features
- Share test results

---

## 📄 License

This project is for educational purposes as part of Digital Image Processing coursework.

---

## 📧 Contact

For questions or suggestions about this implementation:
- Create an issue in the repository
- Contact your course instructor
- Refer to OpenCV documentation

---

## ✅ Project Checklist

- [x] RGB to HSV conversion
- [x] Shadow detection using V-channel
- [x] Dual-channel detection (V + S)
- [x] Morphological operations
- [x] Smooth mask creation
- [x] Intensity normalization
- [x] Saturation control
- [x] Result visualization
- [x] File saving functionality
- [x] Comprehensive documentation

---

**Happy Shadow Removing! 🌟**
