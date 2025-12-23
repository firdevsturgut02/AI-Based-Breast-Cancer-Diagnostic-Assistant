# Breast Cancer Ultrasound Analysis - Academic Project Plan

## Design Guidelines
### Design Style
- **Primary Inspiration**: Modern medical dashboards (GE Healthcare, Siemens Healthineers aesthetics).
- **Style**: Professional, clean, data-driven, and minimalist.
- **Color Palette**: 
  - Primary: #0F172A (Slate Dark - Sidebar/Text)
  - Secondary: #3B82F6 (Medical Blue - Buttons/Highlights)
  - Background: #F8FAFC (Light Gray/White - Content)
  - Success: #10B981 (Emerald Green)
  - Warning: #F59E0B (Amber)
  - Danger: #EF4444 (Red)

### Typography
- Headings: Inter/Plus Jakarta Sans (Bold)
- Body: Inter (Regular, 14px-16px)

### Images to Generate
1. **medical-abstract-hero.jpg**: Abstract high-tech medical background with DNA/cell structures in blue tones. (Style: 3D, minimalist)
2. **assistant-avatar.png**: Friendly but professional AI medical assistant icon. (Style: Minimalist vector)

---

## Development Tasks

### 1. Setup & Environment
- [ ] Install advanced dependencies: `albumentations`, `groq`, `opencv-python-headless`, `seaborn`, `plotly`.
- [ ] Initialize project structure: `src/`, `models/`, `assets/`, `data/`.

### 2. Advanced Data Pipeline (`src/data_loader.py`)
- [ ] **Data Augmentation**: Implement Albumentations (CLAHE, ElasticTransform, ShiftScaleRotate, GridDistortion) specifically for medical ultrasound characteristics.
- [ ] **Preprocessing**: Dynamic resizing and normalization.

### 3. State-of-the-Art Model (`src/model_factory.py`)
- [ ] **Architecture**: DenseNet121 with GlobalAveragePooling, BatchNormalization, and localized dropout.
- [ ] **Fine-Tuning Strategy**: Implement two-stage training (1. Train head only, 2. Progressive unfreezing of top blocks).

### 4. Optimized Training Pipeline (`src/trainer.py`)
- [ ] **Hyperparameter Optimization**: Implement `ReduceLROnPlateau`, `EarlyStopping`, and `ModelCheckpoint`.
- [ ] **Weighted Loss**: Handle class imbalance common in medical datasets.

### 5. Groq AI Expert Agent (`src/ai_agent.py`)
- [ ] **Integration**: Connect Groq API using `gsk_...` key.
- [ ] **Prompt Engineering**: Develop a medical reasoning agent that explains findings based on BIRADS standards.

### 6. Academic Evaluation Suite (`src/evaluator.py`)
- [ ] **Metrics**: Multi-class AUC-ROC, Confusion Matrix, F1-Score, Sensitivity/Specificity.
- [ ] **Visualization**: Plotly-based interactive training history and ROC curves for publication.

### 7. Modern Streamlit Application (`app.py`)
- [ ] **Interface**: Hero section, upload zone, dual-column layout (Image vs Result).
- [ ] **AI Assistant Chat**: Sidebar or bottom section for discussing the diagnosis with the Groq agent.