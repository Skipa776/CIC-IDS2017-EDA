# Implementation Plan: CICIDS2017 Intrusion Detection System

## Goal
Build a production-ready intrusion detection system that:
1. Identifies attacks in network flow data
2. Classifies specific attack types (15 categories)
3. Maps attacks to MITRE ATT&CK framework with mitigations
4. Exposes functionality via REST API

---

## Architecture

```
Network Flow → [Layer 1: Binary Alert] → Is Attack?
                        │
                        ├── NO → Return "BENIGN"
                        │
                        └── YES → [Layer 2: Multi-Class] → Attack Type
                                          │
                                          └── [ATT&CK Mapping] → Mitigations
```

### Layered Detection Design
- **Layer 1:** Fast binary classifier (LightGBM, <1ms) - immediate alert
- **Layer 2:** Multi-class classifier (LightGBM, ~1ms) - detailed classification
- **Layer 3:** Static ATT&CK mapping dictionary

---

## Implementation Phases

### Phase 1: Complete Analysis Notebooks

#### 1.1 attack_types.ipynb
**Location:** `notebooks/attack_types.ipynb`

**Sections to implement:**
1. Attack distribution bar chart with sample counts
2. Feature distributions by attack (box plots for: Flow Duration, Flow Bytes/s, Total Fwd/Bwd Packets, Packet Length Mean/Std)
3. Attack signatures: Top 5 discriminative features per attack via Random Forest importance
4. Confusion matrix showing which attacks get misclassified as each other
5. UMAP projection colored by attack type
6. Per-attack precision/recall/F1 table

**Output:** Analysis insights for model design and feature selection

#### 1.2 day_of_the_weeks.ipynb
**Location:** `notebooks/day_of_the_weeks.ipynb`

**Sections to implement:**
1. Scenario mapping table (Meta_source → Day → Attack types)
2. Stacked bar chart: benign vs attack counts per capture day
3. Benign traffic comparison across days (feature stability)
4. Temporal train/test split experiment (Mon-Thu train, Fri test)
5. Feature drift analysis: stable vs day-dependent features

**Output:** Temporal generalization insights, robust feature selection

---

### Phase 2: Model Training Pipeline

#### 2.1 Create src/ modules

**src/data/loader.py:**
- `load_processed_data()` - Load from parquet
- `get_feature_columns()` - Get numeric feature columns
- `prepare_binary_labels()` - Label != "BENIGN" → 1
- `prepare_multiclass_labels()` - Encode attack types to integers

**src/features/engineering.py:**
- `FAST_FEATURES` - List of 20 selected features
- `select_features()` - SelectKBest with f_classif
- `create_scaler()` - StandardScaler factory

**src/models/train.py:**
- `train_layer1()` - Binary LightGBM
- `train_layer2()` - Multi-class LightGBM
- `save_models()` - Serialize to models/

#### 2.2 Model Configurations

**Layer 1 (Binary):**
```python
{
    'objective': 'binary',
    'num_leaves': 31,
    'max_depth': 6,
    'n_estimators': 100,
    'class_weight': 'balanced'
}
```

**Layer 2 (Multi-class):**
```python
{
    'objective': 'multiclass',
    'num_class': 14,  # or 15 if keeping rare classes
    'num_leaves': 63,
    'max_depth': 8,
    'n_estimators': 200,
    'class_weight': 'balanced'
}
```

#### 2.3 Feature Selection (20 features for fast inference)
```
Destination Port, Flow Duration, Flow Bytes/s, Flow Packets/s,
Total Fwd Packets, Total Backward Packets, Fwd Packet Length Mean,
Bwd Packet Length Mean, Packet Length Mean, Packet Length Std,
SYN Flag Count, FIN Flag Count, RST Flag Count, ACK Flag Count,
Init_Win_bytes_forward, Init_Win_bytes_backward, Flow IAT Mean,
Flow IAT Std, Fwd IAT Mean, Bwd IAT Mean
```

---

### Phase 3: REST API Implementation

#### 3.1 Project Structure
```
api/
├── __init__.py
├── main.py              # FastAPI app, model loading on startup
├── config.py            # Settings
├── models/
│   ├── schemas.py       # FlowFeatures, ClassificationResult
│   └── classifier.py    # Model wrapper class
├── routers/
│   ├── classify.py      # POST /api/v1/classify
│   └── health.py        # GET /api/v1/health
└── services/
    ├── prediction.py    # Layered inference logic
    └── mitre_mapping.py # ATT&CK dictionary
```

#### 3.2 Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | /api/v1/classify | Classify single flow |
| POST | /api/v1/classify/batch | Classify multiple flows |
| GET | /api/v1/health | Health check |
| GET | /api/v1/model-info | Model metadata |

#### 3.3 Response Schema
```json
{
  "success": true,
  "result": {
    "is_attack": true,
    "attack_probability": 0.97,
    "attack_type": "DoS Hulk",
    "attack_type_probability": 0.89,
    "confidence_level": "high",
    "mitre_mapping": {
      "technique_id": "T1498.001",
      "technique_name": "Network DoS: Direct Network Flood",
      "tactic": "Impact",
      "mitigations": ["M1037: Rate limiting", "..."]
    },
    "inference_time_ms": 1.2
  }
}
```

---

### Phase 4: MITRE ATT&CK Mapping

| CICIDS Attack | ATT&CK ID | Technique | Tactic |
|--------------|-----------|-----------|--------|
| DoS Hulk | T1498.001 | Network DoS: Direct Flood | Impact |
| DoS GoldenEye | T1498.001 | Network DoS: Direct Flood | Impact |
| DoS Slowhttptest | T1498.001 | Network DoS: Direct Flood | Impact |
| DoS slowloris | T1498.001 | Network DoS: Direct Flood | Impact |
| DDoS | T1498 | Network Denial of Service | Impact |
| FTP-Patator | T1110.001 | Brute Force: Password Guessing | Credential Access |
| SSH-Patator | T1110.001 | Brute Force: Password Guessing | Credential Access |
| PortScan | T1046 | Network Service Scanning | Discovery |
| Web Attack - Brute Force | T1110.001 | Brute Force: Password Guessing | Credential Access |
| Web Attack - XSS | T1059.007 | JavaScript Interpreter | Execution |
| Web Attack - Sql Injection | T1190 | Exploit Public-Facing App | Initial Access |
| Bot | T1071.001 | App Layer Protocol: Web | C2 |
| Infiltration | T1071 | Application Layer Protocol | C2 |
| Heartbleed | T1190 | Exploit Public-Facing App | Initial Access |

Each attack will have 5+ specific mitigations documented.

---

### Phase 5: Testing

#### Tests to implement:
1. `tests/test_models.py` - Model training, inference speed
2. `tests/test_api.py` - API endpoints, validation
3. `tests/test_mitre_mapping.py` - All attacks mapped, valid IDs

#### Performance targets:
- Layer 1 PR-AUC: >0.95
- Layer 1 Recall: >0.90
- Layer 2 Macro F1: >0.80
- Inference time: <5ms per sample

---

## File Deliverables

### Notebooks (to complete):
- [ ] `notebooks/attack_types.ipynb`
- [ ] `notebooks/day_of_the_weeks.ipynb`

### Source modules (to create):
- [ ] `src/data/__init__.py`
- [ ] `src/data/loader.py`
- [ ] `src/features/__init__.py`
- [ ] `src/features/engineering.py`
- [ ] `src/models/__init__.py`
- [ ] `src/models/train.py`
- [ ] `src/models/evaluate.py`

### API (to create):
- [ ] `api/__init__.py`
- [ ] `api/main.py`
- [ ] `api/config.py`
- [ ] `api/models/schemas.py`
- [ ] `api/models/classifier.py`
- [ ] `api/routers/classify.py`
- [ ] `api/routers/health.py`
- [ ] `api/services/prediction.py`
- [ ] `api/services/mitre_mapping.py`

### Model artifacts (to generate):
- [ ] `models/layer1_binary_classifier.joblib`
- [ ] `models/layer2_multiclass_classifier.joblib`
- [ ] `models/feature_scaler.joblib`
- [ ] `models/feature_columns.json`
- [ ] `models/label_mapping.json`
- [ ] `models/model_metadata.json`

### Tests (to create):
- [ ] `tests/__init__.py`
- [ ] `tests/test_models.py`
- [ ] `tests/test_api.py`
- [ ] `tests/test_mitre_mapping.py`

### Config:
- [ ] `requirements.txt`

---

## Dependencies

```
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
lightgbm>=4.0.0
joblib>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.15.0
umap-learn>=0.5.0
pyarrow>=12.0.0
fastapi>=0.100.0
uvicorn>=0.23.0
pydantic>=2.0.0
pytest>=7.4.0
httpx>=0.24.0
```

---

## User Actions Required

1. Verify data exists: `data/processed/cicids2017_clean.parquet`
2. Review completed notebooks for insights
3. Test API locally: `uvicorn api.main:app --reload`
4. Make deployment decisions (Docker, cloud, auth)
