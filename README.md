
# AI-Enhanced Diffuse Optical Tomography (DOT)

This project reconstructs 3D images of brain activity through the skull using near-infrared (NIR) light.

## 🧠 Goal
Reconstruct 3D voxel maps showing blood oxygenation changes using AI-enhanced Monte Carlo simulations.

## 🔧 Components
- `simulate_mcx.py`: Generates synthetic DOT data using simplified Monte Carlo simulation.
- `unet3d.py`: Defines a 3D U-Net model for volumetric reconstruction.
- `train.py`: Handles model training and validation.
- `visualize.py`: Visualizes 3D outputs using matplotlib.
- `main.py`: Runs the full pipeline from data generation to inference.

## 🧪 How to Run in Google Colab
1. Upload the zipped project folder.
2. Unzip and navigate to the directory.
3. Install necessary packages:
```python
!pip install numpy torch matplotlib scikit-image
```
4. Run:
```bash
!python main.py
```

## ✅ Requirements
- PyTorch
- NumPy
- Matplotlib
- scikit-image

## 📦 Future Work
- Replace `simulate_mcx.py` with actual MCX simulation output.
- Add support for multi-wavelength DOT.
- Deploy on cloud for real-time monitoring.

## 📊 Output
3D voxel images representing hemodynamic responses (blood oxygenation changes).

---

Created for AI + Physics-Based Medical Imaging R&D.
