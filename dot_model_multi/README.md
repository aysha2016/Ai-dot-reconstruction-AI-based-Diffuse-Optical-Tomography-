# 🧠 AI-Enhanced Monte Carlo DOT: 3D Blood Oxygenation Reconstruction

This project implements a deep learning-based approach for reconstructing 3D blood oxygenation maps from diffuse optical tomography (DOT) measurements. It combines Monte Carlo simulations with a 3D U-Net architecture to achieve high-quality reconstructions.

## 📋 Features

- **Monte Carlo Simulation**: Realistic simulation of photon migration in tissue
- **3D U-Net Architecture**: Deep learning model for volumetric reconstruction
- **Multi-wavelength Support**: Handles multiple NIR wavelengths (690nm, 780nm, 830nm)
- **Interactive Visualization**: 3D volume rendering with adjustable parameters
- **Training Pipeline**: Complete training workflow with validation and checkpointing
- **Configuration System**: Flexible JSON-based configuration

## 🚀 Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/dot_model_multi.git
cd dot_model_multi
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## 📦 Dependencies

- PyTorch >= 1.8.0
- NumPy >= 1.19.0
- Matplotlib >= 3.3.0
- scikit-learn >= 0.24.0
- tqdm >= 4.50.0

## 🎯 Usage

### Training

To train the model with default configuration:

```bash
python main.py --mode train --config config.json
```

### Inference

To run inference with a trained model:

```bash
python main.py --mode inference --config config.json --checkpoint runs/run_YYYYMMDD_HHMMSS/best_model.pth
```

### Configuration

The `config.json` file contains all configurable parameters:

- **Model**: Architecture parameters (channels, depth)
- **Training**: Training hyperparameters (epochs, batch size, learning rate)
- **Data**: Dataset configuration (voxel dimensions, wavelengths)
- **Simulation**: Monte Carlo simulation parameters
- **Visualization**: Visualization settings

## 📊 Project Structure

```
dot_model_multi/
├── main.py              # Main entry point
├── simulate_mcx.py      # Monte Carlo simulation
├── unet3d.py           # 3D U-Net model definition
├── train.py            # Training pipeline
├── visualize.py        # Visualization utilities
├── config.json         # Configuration file
├── requirements.txt    # Project dependencies
└── README.md          # This file
```

## 🔬 How It Works

1. **Data Generation**:
   - Monte Carlo simulation generates synthetic DOT measurements
   - Multiple wavelengths capture different tissue properties
   - Ground truth maps represent blood oxygenation changes

2. **Model Architecture**:
   - 3D U-Net processes volumetric data
   - Encoder-decoder structure with skip connections
   - Multi-scale feature extraction

3. **Training Process**:
   - Synthetic data generation on-the-fly
   - Validation on separate dataset
   - Model checkpointing and visualization

4. **Inference**:
   - Load trained model
   - Process new measurements
   - Generate 3D oxygenation maps

## 📈 Results

The model can reconstruct 3D blood oxygenation maps with:
- High spatial resolution
- Accurate activation localization
- Quantitative oxygenation values

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📚 Citation

If you use this code in your research, please cite:

```bibtex
@software{dot_model_multi,
  author = {Your Name},
  title = {AI-Enhanced Monte Carlo DOT: 3D Blood Oxygenation Reconstruction},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/yourusername/dot_model_multi}
}
```

## 🙏 Acknowledgments

- Based on the U-Net architecture by Ronneberger et al.
- Inspired by Monte Carlo eXtreme (MCX) simulation software
- Developed for medical imaging research 