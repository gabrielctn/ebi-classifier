# EBI Classifier 🔬

> **Binary image classifier for Imaging Flow Cytometry (ImageStream/IFC)**  
> Distinguishes **EBI** (Erythroblastic Islands) from **NOT_EBI** cells using deep learning

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
The packaged Windows model is the one from the existing V1 release. Its five-fold cross-validation mean MCC was approximately 0.92; an independent test of the final model reported MCC 0.77 at a threshold of 0.655. See [performance](#-performance-expectations) for the distinction between those results and the app's current threshold.

---

## 📋 Table of Contents

- [Quick Start](#-quick-start)
- [For End Users](#-for-end-users-inference)
- [For Researchers](#-for-researchers-training--tuning)
- [Features](#-features)
- [Project Structure](#-project-structure)
- [Documentation](#-documentation)
- [Technical Details](#-technical-details)

---

## 🚀 Quick Start

### Windows: classify images

1. Send `EBI_Classifier_Windows_Security_2026-11-17.zip` to the user. Extract all of it to a writable folder on a 64-bit Windows 10/11 Intel or AMD PC.
2. Open the extracted `EBI_Classifier_Windows` folder and double-click `START_EBI.bat`.
3. Keep the window open and use the browser interface. The first start prepares a local environment and can take a few minutes. The application, model, Python, and dependencies are included; normal setup and classification work offline.

See [the Windows quick guide](windows/START_HERE.txt) and [the user guide](GUIDE_UTILISATEUR.md). This package uses the CPU and does not include the training data or training tools. Some computers may need the Microsoft Visual C++ Runtime installed; the quick guide explains what to do.

### Mac: classify images

Send `EBI_Classifier_Mac_Security_2026-11-17.zip` to the user. On macOS 13 or later, double-click the ZIP, open the extracted folder, and double-click `START_EBI.command`. Keep Terminal open while using the browser interface. The same package supports Intel and Apple Silicon Macs and includes Python, the model, and the dependencies for offline use. The first start can take a few minutes.

See [the Mac quick guide](macos/START_HERE.txt) for step-by-step instructions and troubleshooting. It uses the same V1 model and contains no training data.

### Linux: classify images

Send `EBI_Classifier_Linux_Security_2026-11-17.zip` to the user. Extract the complete ZIP and open the extracted folder. Double-click `EBI_Classifier.desktop`, allowing it to launch if the desktop asks. Keep the terminal window open while using the browser interface; close it to stop. If the desktop launcher is blocked, open a terminal in the extracted folder and run `./START_EBI.sh`. The package supports 64-bit Intel/AMD and ARM Linux with glibc 2.28 or later and includes Python, the V1 model and dependencies for offline use. See [the Linux quick guide](linux/START_HERE.txt).

### Browser version (prototype)

`web/` contains a version that runs entirely in the browser, with nothing to install and no images uploaded. A public instance runs at https://dsimb.inserm.fr/EBI/. To run it locally, download the model into `web/model/`, run `python3 web/serve.py`, then open http://127.0.0.1:8765/. See [web/README.md](web/README.md) for hosting and validation.

### Rebuild the Windows package

From the project root, using a Python environment with PyTorch and pip installed:

```bash
.venv-secure/bin/python scripts/build_windows_package.py --output EBI_Classifier_Windows_New.zip
```

The builder downloads the Windows runtime and the pinned wheels, copies the inference code, and converts the trusted V1 checkpoint from `runs/20260109-182504/final_model/checkpoints/final.pt` to safetensors and JSON for inference. Use `--model PATH` to deliberately choose another trusted checkpoint and `--output PATH` for a different output file. Existing output files are not overwritten. The build environment needs PyTorch, safetensors and pip.

### Rebuild the Mac package

```bash
.venv-secure/bin/python scripts/build_mac_package.py --output EBI_Classifier_Mac_New.zip
```

The Mac builder uses the same checkpoint and `--model` / `--output` options. It includes separate pinned dependencies for Apple Silicon and Intel; the launcher chooses the matching environment automatically.

### Rebuild the Linux package

```bash
.venv-secure/bin/python scripts/build_linux_package.py --output EBI_Classifier_Linux_New.zip
```

The Linux builder includes separate Python runtimes and wheels for x86-64 and ARM64; the launcher chooses automatically. It uses the same trusted model and `--model` / `--output` options.

### Development

### Installation

```bash
# Clone the repository (or download)
git clone https://github.com/gabrielctn/ebi-classifier.git
cd ebi-classifier

# Setup environment (uses uv package manager)
./setup.sh

# Activate the new environment; the previous .venv is left untouched
source .venv-secure/bin/activate
```

### For Inference (Classify Images)

```bash
# Launch the Gradio web interface
python app.py

# Or double-click: EBI_Classifier.command (macOS)
```

Then drag-and-drop your TIFF files in the browser interface.

The older local `.venv` and earlier ZIPs are preserved, but they have not been
updated and contain older dependencies. Use `.venv-secure` and the dated ZIPs
above for the updated versions. See [IT and legal information](IT_INFORMATION.md)
for remaining dependency advisories and licensing questions.

When run from this development directory, the app selects the most recently
modified training checkpoint, which can be a tuning run rather than the V1 model.
The dated distribution ZIPs use the explicitly selected V1 model instead.

### For Training

```bash
# Train with 5-fold cross-validation
python train.py --cv-folds 5 --tensorboard

# View training progress
tensorboard --logdir runs/
```

### For Hyperparameter Tuning

```bash
# Quick tune (30 trials, ~3-4 hours)
python tune.py --n-trials 30 --tune-all --pruning

# Analyze results
python analyze_tuning.py

# Train with best parameters
./optuna_studies/ebi_study_*/train_with_best_params.sh
```

See [HYPERPARAMETER_TUNING.md](HYPERPARAMETER_TUNING.md) for complete guide.

---

## 👥 For End Users (Inference)

### Using the Application

1. **Launch**: Double-click `START_EBI.bat` in the Windows package, or `START_EBI.command` in the Mac package
2. **Upload**: Drag TIFF triplets into the web interface
3. **Classify**: Click the "Classify" button
4. **Download**: Get results as `.pop` file

### File Format

Each sample requires **3 TIFF files** (one per channel):
```
sample001_Ch3.ome.tif
sample001_Ch7.ome.tif
sample001_Ch11.ome.tif
```

The classifier automatically groups these triplets.

See [GUIDE_UTILISATEUR.md](GUIDE_UTILISATEUR.md) for detailed user guide (FR/EN).

---

## 🔬 For Researchers (Training & Tuning)

### Data Organization

```
EBI/              # Positive class (EBI samples)
├── sample001_Ch3.ome.tif
├── sample001_Ch7.ome.tif
├── sample001_Ch11.ome.tif
├── sample002_Ch3.ome.tif
└── ...

NOT_EBI/          # Negative class (non-EBI samples)
├── sample101_Ch3.ome.tif
├── sample101_Ch7.ome.tif
└── ...

EBI_TEST_SET/     # Optional: independent test set
NOT_EBI_TEST_SET/
```

### Training Workflow

#### 1. Basic Training

```bash
# Single train/val split (20% validation)
python train.py --val-ratio 0.2 --epochs 20

# K-fold cross-validation (recommended)
python train.py --cv-folds 5 --epochs 20 --tensorboard
```

#### 2. Hyperparameter Optimization

**Quick optimization** (2-3 hours):
```bash
python tune.py --n-trials 30 --tune-optimizer --tune-augmentation --pruning
```

**Production optimization** (6-8 hours):
```bash
python tune.py \
  --n-trials 50 \
  --tune-all \
  --pruning \
  --storage sqlite:///optuna_studies/production.db
```

**Analyze results**:
```bash
python analyze_tuning.py
open optuna_studies/ebi_study_*/report.html
```

See [HYPERPARAMETER_TUNING.md](HYPERPARAMETER_TUNING.md) and [TUNING_CHEATSHEET.md](TUNING_CHEATSHEET.md).

#### 3. Training with Best Parameters

```bash
# Use auto-generated script
./optuna_studies/ebi_study_*/train_with_best_params.sh

# Or manual training
python train.py \
  --model convnextv2_base.fcmae_ft_in22k_in1k \
  --lr 0.00015 \
  --cv-folds 5 \
  --tensorboard
```

#### 4. Adding corrections from users

In the browser version, users who disagree with a prediction move the object
to the other group in step 4 and click **Download corrections (.csv)**. The
file lists each moved object with the model's prediction, its probability, the
corrected class and the three file names. It contains no images.

To use a batch of corrections for retraining:

1. Ask for `EBI_corrections.csv` and the TIFF files it names, in one folder.
2. Add them to the training folders. Give every batch its own prefix, because
   ImageStream object IDs restart at zero in each acquisition:

   ```bash
   python scripts/import_corrections.py EBI_corrections.csv \
       --images /path/to/received --prefix lab2-2026-11 --dry-run
   python scripts/import_corrections.py EBI_corrections.csv \
       --images /path/to/received --prefix lab2-2026-11
   ```

3. Retrain with `train.py`. Compare the new model with the current one on
   `EBI_TEST_SET/` and `NOT_EBI_TEST_SET/`: it should do better on the corrected
   cases without getting worse elsewhere.
4. Export it with `scripts/export_onnx.py --reference`, validate it in the
   browser (`web/dev/validate.html`), rebuild the packages and publish a new
   Zenodo version.

A few corrected objects rarely change the model much. Collect several dozen
examples of the same kind of mistake (for example dead or shrunken macrophages
counted as EBI) before retraining. Do not import objects that are in the test
sets, including the website examples, or the test results stop being
independent.

### Monitoring

```bash
# TensorBoard (metrics, images, Grad-CAM)
tensorboard --logdir runs/

# Watch tuning progress
watch -n 60 'python analyze_tuning.py --no-report'
```

---

## ✨ Features

### Core Capabilities

- ✅ **Binary Classification**: EBI vs NOT_EBI
- ✅ **3-Channel Microscopy**: Handles OME-TIFF triplets (Ch3, Ch7, Ch11)
- ✅ **Cross-Validation**: Stratified K-fold with proper class balancing
- ✅ **Threshold Tuning**: Auto-optimize decision threshold per epoch
- ✅ **Class Imbalance**: Weighted loss + balanced sampling
- ✅ **Data Augmentation**: Basic, strong, and SOTA (RandAugment) modes
- ✅ **Test Set Evaluation**: Independent test set with metrics

### Advanced Features

- 🎯 **Hyperparameter Optimization**: State-of-the-art Bayesian search (Optuna)
- 📊 **Rich Metrics**: MCC, F1, ROC-AUC, PR-AUC, confusion matrix
- 🔍 **Explainability**: Grad-CAM visualization of model attention
- 📈 **TensorBoard**: Real-time training monitoring
- 🚀 **Modern Architectures**: ConvNeXtV2, EfficientNet, ViT, ResNet
- ⚡ **Efficient Training**: Transfer learning + progressive unfreezing
- 💾 **Resumable**: Save/load checkpoints, resume training

### Hyperparameter Tuning (NEW)

- 🔬 **Bayesian Optimization**: TPE sampler for intelligent search
- ⏱️ **Early Stopping**: Median pruner (saves 30-50% time)
- 🔄 **Parallel Execution**: Multi-worker support
- 📊 **Rich Visualizations**: 10+ interactive plots
- 🎯 **Automated**: Generate training scripts with best params
- 💾 **Resumable Studies**: Database-backed for long runs

---

## 📁 Project Structure

```
.
├── train.py                    # Main training script (K-fold CV, threshold tuning)
├── tune.py                     # Hyperparameter optimization (Optuna)
├── analyze_tuning.py           # Tuning result analysis & visualization
├── dataset.py                  # TIFF loading, augmentations, triplet handling
├── app.py                      # Gradio web interface for inference
├── setup.sh                    # Environment setup (uv package manager)
│
├── EBI/                        # Training data: positive class
├── NOT_EBI/                    # Training data: negative class
├── EBI_TEST_SET/              # Optional: test data (positive)
├── NOT_EBI_TEST_SET/          # Optional: test data (negative)
│
├── runs/                       # Training artifacts
│   └── 20260109-182504/
│       ├── cv_summary.csv     # Cross-validation results
│       ├── final_test_metrics.csv
│       ├── fold_01/
│       │   ├── config.json
│       │   ├── metrics.csv
│       │   ├── checkpoints/
│       │   │   ├── best.pt
│       │   │   └── last.pt
│       │   └── plots/
│       └── ...
│
├── optuna_studies/            # Hyperparameter tuning results
│   └── ebi_study_*/
│       ├── best_params.json
│       ├── train_with_best_params.sh
│       ├── report.html
│       └── visualizations/
│
├── HYPERPARAMETER_TUNING.md   # Complete tuning guide
├── TUNING_CHEATSHEET.md       # Quick reference
├── TUNING_EXAMPLES.md         # Scenario-based examples
├── GUIDE_UTILISATEUR.md       # End-user guide (FR/EN)
└── IMPLEMENTATION_SUMMARY.md  # Technical implementation details
```

---

## 📚 Documentation

| Document | Description | Audience |
|----------|-------------|----------|
| [GUIDE_UTILISATEUR.md](GUIDE_UTILISATEUR.md) | User guide for inference (FR/EN) | End users |
| [HYPERPARAMETER_TUNING.md](HYPERPARAMETER_TUNING.md) | Complete tuning guide | Researchers |
| [TUNING_CHEATSHEET.md](TUNING_CHEATSHEET.md) | Quick reference for tuning | Researchers |
| [TUNING_EXAMPLES.md](TUNING_EXAMPLES.md) | Scenario-based examples | Researchers |
| [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md) | Technical details | Developers |
| [.github/copilot-instructions.md](.github/copilot-instructions.md) | AI agent instructions | Developers |

---

## 🔧 Technical Details

### Model Architecture

- **Backbone**: timm models (default: `convnextv2_tiny.fcmae_ft_in22k_in1k`)
- **Input**: 3-channel pseudo-RGB (Ch7→R, Ch11→G, Ch3→B)
- **Training**: Freeze backbone → Fine-tune with lower LR
- **Loss**: Weighted CrossEntropyLoss + optional MixUp
- **Optimization**: AdamW with cosine decay + warmup

### Key Hyperparameters

| Parameter | Default | Tunable Range | Impact |
|-----------|---------|---------------|--------|
| Learning Rate (head) | 1e-4 | 1e-5 to 1e-3 | High |
| Learning Rate (finetune) | 1e-5 | 1e-6 to 5e-4 | High |
| Weight Decay | 0.05 | 0.0 to 0.2 | Medium |
| Augmentation | basic | basic/strong/sota | Medium |
| Batch Size | 32 | 16/32/64 | Medium |
| Epochs | 20 | 15 to 40 | Medium |
| MixUp Alpha | 0.1 | 0.0 to 0.4 | Low-Medium |

### Performance

The V1 release checkpoint has five-fold validation results and a separate final-model test report. The precise numbers and threshold are listed below; results from tuning runs should be assessed separately.

### Dependencies

- **PyTorch** 2.0+ (MPS support for macOS)
- **timm** (pretrained vision models)
- **scikit-learn** (metrics, CV splits)
- **Optuna** (hyperparameter optimization)
- **Gradio** (inference UI)
- **TensorBoard** (training visualization)
- **tifffile** (OME-TIFF loading)

See [setup.sh](setup.sh) for complete list.

---

## 🎯 Performance Expectations

### Training Time

- **Single fold**: ~30-40 min (default settings)
- **5-fold CV**: ~2-3 hours
- **Hyperparameter tuning**: 
  - 30 trials: ~3-4 hours
  - 100 trials: ~12-16 hours

*Times on M1 MacBook Pro. GPU training is faster.*

### Recorded results for the packaged model

The five validation folds have MCC values of 0.944, 0.923, 0.926, 0.908 and 0.910 ([fold summary](runs/20260109-182504/cv_summary.csv)). The independently evaluated final model reported MCC 0.766, accuracy 0.875 and F1 0.863 at its cross-validation threshold of 0.655 ([test report](runs/20260109-182504/final_test_metrics.csv)).

The current app retains its existing fallback threshold of **0.67** for this checkpoint, so the test report's threshold-dependent metrics should not be presented as exact measurements of the packaged app. Results can also vary with the dataset and channel configuration.

---

## 🤝 Contributing

Contributions welcome! Areas of interest:

- [ ] Multi-class classification support
- [ ] Additional model architectures
- [ ] Automated ensemble methods
- [ ] Web deployment (Docker, cloud)
- [ ] Additional visualization tools

---

## 📄 License

The source code is released under the [MIT licence](LICENSE). The trained model weights are released under [CC BY-NC 4.0](web/model/LICENSE.md), the licence of the pretrained ConvNeXt V2 weights they were fine-tuned from. The training and test images are not part of this repository. The offline packages include third-party dependencies under their own licences; see [information for IT and legal review](IT_INFORMATION.md).

---

## 🙏 Acknowledgments

- **timm library**: Ross Wightman
- **Optuna**: Preferred Networks
- **PyTorch**: Meta AI
- **ImageStream/IFC**: Amnis/Luminex

---

## 📧 Contact

For questions or issues, please open a GitHub issue.

---

## 🔖 Citation

If you use this classifier in your research, please cite:

```bibtex
@software{cretin_ebi_classifier_2026,
  title   = {EBI Classifier: deep learning for erythroblastic island detection in imaging flow cytometry},
  author  = {Cretin, Gabriel},
  year    = {2026},
  version = {1.0.0},
  doi     = {10.5281/zenodo.24324392},
  url     = {https://doi.org/10.5281/zenodo.24324392}
}
```

The trained model (`ebi.onnx`) is archived in the same Zenodo record.

---

**Last Updated**: January 2026  
**Version**: 2.0 (with hyperparameter tuning)
