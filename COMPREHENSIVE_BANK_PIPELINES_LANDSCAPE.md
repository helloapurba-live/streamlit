# BANK AML/FRAUD AI SYSTEMS - COMPREHENSIVE PIPELINE TABLES
## Open Source Local Windows Deployment | Complete End-to-End Workflows
### Windows 11 | PyPI Only | No Cloud | On-Premises PII Protection

---

# PIPELINE ARCHITECTURE OVERVIEW

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                    BANK AML/FRAUD AI SYSTEM                                         │
│                                   Local Windows 11 Environment                                       │
├─────────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                     │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐│
│  │     DATA     │───▶│   ML/DL/     │───▶│   RAG &      │───▶│   AGENTIC    │───▶│    MLOps     ││
│  │ ENGINEERING  │    │  GENERATIVE  │    │   LLMOps     │    │      AI      │    │  LIFECYCLE   ││
│  │     ETL      │    │      AI      │    │              │    │              │    │  MANAGEMENT  ││
│  └──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘│
│        ▲                    ▲                    ▲                    ▲                    ▲        │
│        │                    │                    │                    │                    │        │
│        └────────────────────┴────────────────────┴────────────────────┴────────────────────┘        │
│                                    MONITORING & ORCHESTRATION                                       │
│                              (Dash Dashboards + ZenML/Prefect + evidently)                         │
└─────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

# TABLE 1: DATA ENGINEERING / ETL PIPELINE

## End-to-End Data Pipeline for Transaction Processing

| STAGE | STEP | PURPOSE | TOOLS/LIBRARIES (PyPI) | LOCAL INTEGRATION | PROS | CONS | LIMITATIONS |
|-------|------|---------|------------------------|-------------------|------|------|-------------|
| **INGESTION** | Database Connect | Extract from source DBs | `psycopg2-binary`<br>`sqlalchemy`<br>`pyodbc` | Connect: localhost:5432<br>PostgreSQL local instance | ✅ Standard SQL interface<br>✅ Type-safe queries<br>✅ Connection pooling | ❌ PostgreSQL setup required<br>❌ Single machine limit | Max: RAM-dependent<br>Recommend: 32-64 GB |
| **INGESTION** | File Reading | Load CSV/Parquet/JSON | `pandas`<br>`polars`<br>`pyarrow` | Read: C:\Data\raw\<br>Local file system | ✅ polars 5-10× faster<br>✅ Parquet columnar format<br>✅ Low memory usage | ❌ pandas limited to ~10 GB<br>❌ Windows path issues (\ vs /) | polars max: 50-100 GB<br>Use pathlib for paths |
| **INGESTION** | Incremental Load | Delta extraction only | `pandas`<br>`sqlalchemy` | WHERE timestamp > last_load | ✅ Efficient (only new data)<br>✅ Reduces processing time | ❌ Requires timestamp column<br>❌ Manual tracking | Store watermark in config |
| **VALIDATION** | Schema Check | Verify data structure | `pandera`<br>`great-expectations` | YAML schema definitions<br>C:\Schemas\ | ✅ Catches errors early<br>✅ Auto-generates reports<br>✅ Banking compliance | ❌ Setup overhead<br>❌ Schema maintenance burden | Must update schemas<br>when source changes |
| **VALIDATION** | Data Quality | Nulls, duplicates, stats | `great-expectations`<br>`ydata-profiling` | HTML reports: C:\Reports\quality\ | ✅ Comprehensive checks<br>✅ Visual reports<br>✅ Profiling insights | ❌ Slow on large data (>1M rows)<br>❌ Memory intensive | Sample large datasets<br>Use 10% for profiling |
| **VALIDATION** | Business Rules | AML/KYC validation | `python-rule-engine` | JSON rules: C:\Rules\aml_rules.json | ✅ Declarative rules<br>✅ Non-technical can update<br>✅ Version-controlled | ❌ Complex logic hard to express<br>❌ Performance on large volumes | Batch evaluation<br>Optimize rule order |
| **CLEANING** | Missing Values | Impute or remove | `scikit-learn.impute`<br>`pandas` | SimpleImputer(strategy='median') | ✅ Statistical imputation<br>✅ Preserves data distribution | ❌ May introduce bias<br>❌ Not suitable for all fields | Document imputation<br>Track % imputed |
| **CLEANING** | Deduplication | Remove duplicates | `pandas.drop_duplicates`<br>`polars` | subset=['txn_id', 'timestamp'] | ✅ Fast operation<br>✅ Preserves first/last occurrence | ❌ Exact match only<br>❌ Fuzzy matching needed for names | Use recordlinkage for<br>fuzzy deduplication |
| **CLEANING** | Outlier Detection | Identify anomalies | `scipy.stats`<br>`pyod` | IQR, Z-score, Isolation Forest | ✅ Multiple algorithms<br>✅ Fraud detection signal | ❌ May flag legitimate high-value<br>❌ Threshold tuning needed | Bank-specific thresholds<br>Manual review process |
| **TRANSFORMATION** | Type Conversion | Standardize dtypes | `pandas.astype`<br>`numpy` | datetime64, float64, category | ✅ Memory optimization<br>✅ Faster operations | ❌ Data loss if wrong type<br>❌ Timezone handling complex | Test on sample first<br>Use UTC for timestamps |
| **TRANSFORMATION** | Encoding | Categorical encoding | `category-encoders`<br>`scikit-learn` | Label, OneHot, Target encoding | ✅ ML-ready format<br>✅ Handles high cardinality | ❌ Explosion with OneHot<br>❌ Target leak risk | Use Target encoding carefully<br>Track encoding mappings |
| **TRANSFORMATION** | Aggregations | Feature creation | `pandas.groupby`<br>`polars.groupby`<br>`duckdb` | Rolling windows, time-based | ✅ duckdb 10-100× faster<br>✅ SQL interface | ❌ Complex syntax for windows<br>❌ Memory for wide windows | Use duckdb for analytics<br>Limit window size |
| **ENRICHMENT** | Feature Engineering | Derived features | `feature-engine`<br>`pandas` | Velocity, frequency, recency | ✅ Domain knowledge encoding<br>✅ Improves model performance | ❌ Manual effort<br>❌ Requires expertise | Document each feature<br>Track feature importance |
| **ENRICHMENT** | Time Features | Temporal patterns | `pandas.dt`<br>`tsfresh` | hour, day_of_week, time_since | ✅ Captures seasonality<br>✅ Fraud patterns (night txns) | ❌ tsfresh very slow<br>❌ Many features (curse of dim) | Sample for tsfresh<br>Feature selection critical |
| **VERSIONING** | Data Versioning | Track dataset changes | `dvc` | dvc add data/txn.parquet<br>Local storage: C:\dvc_storage | ✅ Git-like for data<br>✅ Reproducibility | ❌ Requires Git installed<br>❌ Large files slow | Use cloud remote if allowed<br>Or local NAS |
| **VERSIONING** | Lineage Tracking | Document data flow | `great-expectations`<br>Custom SQLite | Metadata: C:\Metadata\lineage.db | ✅ Compliance documentation<br>✅ Root cause analysis | ❌ Manual effort<br>❌ No auto lineage | Use ZenML for auto lineage<br>(see MLOps section) |
| **STORAGE** | Raw Data Lake | Store unprocessed | `pyarrow.parquet`<br>`duckdb` | C:\DataLake\raw\{date}\ | ✅ Parquet compressed 10×<br>✅ Immutable raw copy | ❌ Disk space grows<br>❌ No transactions | Implement retention policy<br>Archive to tape/NAS |
| **STORAGE** | Curated Layer | Store cleaned | `pyarrow`<br>`sqlalchemy` | C:\DataLake\curated\<br>PostgreSQL | ✅ Ready for ML<br>✅ Quality guaranteed | ❌ Duplication of storage<br>❌ Sync needed | Use views instead of copies<br>where possible |
| **ORCHESTRATION** | ETL Scheduling | Manage workflow | `APScheduler`<br>`prefect`<br>`kedro` | Daily 2 AM cron<br>Local Python process | ✅ Simple (APScheduler)<br>✅ UI (Prefect)<br>✅ Structure (Kedro) | ❌ Single machine<br>❌ Process-based (can crash) | Use Windows Task Scheduler<br>for production reliability |
| **MONITORING** | Pipeline Health | Track execution | `loguru`<br>`prometheus-client` | Logs: C:\Logs\etl.log<br>Metrics: localhost:9090 | ✅ Structured logging<br>✅ Prometheus standard | ❌ No alerting built-in<br>❌ Dashboard needed | Use Dash for visualization<br>Email alerts with yagmail |
| **MONITORING** | Data Drift | Distribution changes | `evidently`<br>`scipy.stats` | HTML reports: C:\Reports\drift\ | ✅ Automatic detection<br>✅ Visual reports | ❌ Threshold tuning<br>❌ False positives | Set bank-specific thresholds<br>Weekly reviews |

## DEPLOYMENT OPTIONS: ETL PIPELINE

| MODE | SCHEDULER | FREQUENCY | LATENCY | USE CASE | IMPLEMENTATION |
|------|-----------|-----------|---------|----------|----------------|
| **Batch (Scheduled)** | APScheduler<br>Windows Task Scheduler | Daily 2 AM<br>Hourly | Minutes to hours | Daily fraud review<br>Weekly AML reports | `scheduler.add_job(etl_job, 'cron', hour=2)` |
| **Micro-Batch** | APScheduler | Every 15 min | 15-20 minutes | Near real-time detection | `scheduler.add_job(etl_job, 'interval', minutes=15)` |
| **Manual Trigger** | CLI / Script | On-demand | Immediate | Ad-hoc analysis<br>Month-end reports | `python run_etl.py --date 2025-11-15` |

## BEST PRACTICES & INDUSTRY STANDARDS

| STANDARD | REQUIREMENT | IMPLEMENTATION | BENEFIT |
|----------|-------------|----------------|---------|
| **BCBS 239** (Basel) | Data aggregation, risk reporting | great-expectations + data quality checks | Regulatory compliance |
| **GDPR** | PII on-premises only | All local processing, no cloud upload | Data privacy |
| **SOX** | Audit trails, change control | DVC versioning + loguru logging | Auditability |
| **ISO 8000** | Data quality standard | great-expectations aligned with ISO 8000 | Quality assurance |
| **Idempotency** | Re-run = same result | Date-partitioned files, no appends | Reliability |
| **Data Lineage** | Track source to destination | great-expectations + metadata DB | Compliance, debugging |

## DECISION MATRIX: ETL ORCHESTRATION BY BANK SIZE

| BANK SIZE | TRANSACTIONS/DAY | TEAM SIZE | RECOMMENDED | WHY |
|-----------|------------------|-----------|-------------|-----|
| **Small (<$1B)** | 10K-100K | 1-2 | **APScheduler** | Simplest, sufficient for basic needs |
| **Regional ($1B-$10B)** | 100K-1M | 3-5 | **Prefect** | UI for monitoring, better error handling |
| **Large ($10B-$100B)** | 1M-10M | 5-10 | **Kedro + Prefect** | Structure (Kedro) + Orchestration (Prefect) |
| **Global (>$100B)** | 10M+ | 10+ | **Kedro + Dagster** | Enterprise governance + asset tracking |

## KEY RECOMMENDATIONS

### For Most Banks (Regional):
- **Processing:** polars (5-10× faster than pandas)
- **Storage:** PostgreSQL (ACID) + Parquet files (analytics)
- **Orchestration:** Prefect (UI at localhost:4200)
- **Quality:** great-expectations (HTML reports)
- **Monitoring:** Dash dashboards + evidently drift detection

### Special Mentions:
- **DuckDB:** Game-changer for analytics (SQL on Parquet, 10-100× pandas)
- **polars:** Future of DataFrames (faster, better memory)
- **Prefect:** Better than Airflow for single machine (no complexity)
- **great-expectations:** Industry standard for data quality

---

# TABLE 2: ML / DL / GENERATIVE AI PIPELINE

## End-to-End Machine Learning for Fraud Detection

| STAGE | STEP | PURPOSE | TOOLS/LIBRARIES (PyPI) | LOCAL INTEGRATION | PROS | CONS | LIMITATIONS |
|-------|------|---------|------------------------|-------------------|------|------|-------------|
| **DATA PREP** | Feature Selection | Identify relevant features | `scikit-learn.feature_selection`<br>`feature-engine` | SelectKBest, RFE, mutual_info | ✅ Reduces overfitting<br>✅ Faster training<br>✅ Interpretability | ❌ May lose signal<br>❌ Requires domain knowledge | Start with domain features<br>Validate with experts |
| **DATA PREP** | Train/Val/Test Split | Create datasets | `scikit-learn.model_selection` | 70/15/15, stratified by fraud | ✅ Unbiased evaluation<br>✅ Standard practice | ❌ Temporal issues in fraud<br>❌ Class imbalance | Use time-based split<br>Stratify by fraud rate |
| **DATA PREP** | Handle Imbalance | Balance fraud/non-fraud | `imbalanced-learn` | SMOTE, class_weight='balanced' | ✅ SMOTE proven effective<br>✅ Improves minority detection | ❌ SMOTE creates synthetic<br>❌ May overfit | Use with cross-validation<br>Combine with class weights |
| **DATA PREP** | Scaling | Normalize features | `scikit-learn.preprocessing` | StandardScaler, RobustScaler | ✅ Required for DL, SVM<br>✅ Improves convergence | ❌ Not for tree-based<br>❌ Fit on train only | Save scaler for production<br>Use RobustScaler for outliers |
| **BASELINE** | Logistic Regression | Simple interpretable model | `scikit-learn.linear_model` | LogisticRegression(C=1.0) | ✅ Fast, interpretable<br>✅ Regulatory friendly<br>✅ Good baseline | ❌ Linear only<br>❌ Poor on complex patterns | Start here always<br>Hard to beat on small data |
| **BASELINE** | Random Forest | Ensemble baseline | `scikit-learn.ensemble` | RandomForestClassifier(n=100) | ✅ Handles non-linear<br>✅ Feature importance<br>✅ Robust | ❌ Slow on large data<br>❌ Memory intensive | Limit n_estimators<br>Use max_samples |
| **GBM** | XGBoost | Gradient boosting | `xgboost` | XGBClassifier(n_estimators=100) | ✅ State-of-art tabular<br>✅ Fast training<br>✅ Kaggle winner | ❌ Hyperparameter tuning<br>❌ Overfitting risk | Use early_stopping_rounds<br>Cross-validate well |
| **GBM** | LightGBM | Fast gradient boosting | `lightgbm` | LGBMClassifier(n_estimators=100) | ✅ Fastest GBM<br>✅ Low memory<br>✅ Handles large data | ❌ Sensitive to params<br>❌ Less interpretable | Best for >100K rows<br>Use num_leaves carefully |
| **GBM** | CatBoost | Categorical GBM | `catboost` | CatBoostClassifier(iterations=100) | ✅ Best for categoricals<br>✅ Less tuning<br>✅ Built-in GPU | ❌ Slower than LightGBM<br>❌ Larger models | Best if many categoricals<br>Use cat_features param |
| **DEEP LEARNING** | PyTorch Setup | DL framework | `torch` (CPU version) | pip install torch --index-url<br>pytorch.org/whl/cpu | ✅ Flexible, research-ready<br>✅ Large community | ❌ CPU-only slow training<br>❌ Steep learning curve | Use for research only<br>Inference OK on CPU |
| **DEEP LEARNING** | MLP | Multi-layer perceptron | `torch.nn`<br>`pytorch-lightning` | 3-layer: [128, 64, 32] | ✅ Handles non-linearity<br>✅ Feature interactions | ❌ Needs large data (>100K)<br>❌ Black box | Requires >100K samples<br>Use TabNet for interpret |
| **DEEP LEARNING** | TabNet | Attention for tabular | `pytorch-tabnet` | TabNetClassifier() | ✅ Interpretable DL<br>✅ Feature importance<br>✅ SOTA tabular | ❌ Slower than XGBoost<br>❌ Tuning needed | Competitive with XGBoost<br>Use when need DL + interpret |
| **DEEP LEARNING** | LSTM | Sequential patterns | `torch.nn.LSTM` | Sequence of transactions | ✅ Captures temporal<br>✅ User behavior patterns | ❌ Needs sequences<br>❌ Very slow training (CPU) | Requires transaction history<br>Consider GRU (faster) |
| **TIME SERIES** | ARIMA | Statistical forecasting | `statsmodels` | ARIMA(p,d,q) for volume | ✅ Interpretable<br>✅ Statistical rigor | ❌ Univariate only<br>❌ Stationarity required | Good for volume forecasting<br>Not for fraud prediction |
| **TIME SERIES** | Prophet | Facebook forecasting | `prophet` | Prophet() for seasonality | ✅ Handles seasonality<br>✅ Missing data OK<br>✅ Easy to use | ❌ Univariate only<br>❌ Overfits small data | Best for weekly/monthly trends<br>Requires >2 years data |
| **GRAPH ML** | NetworkX | Graph construction | `networkx` | Build money laundering network | ✅ Detect rings<br>✅ Community detection | ❌ Scalability (max ~1M nodes)<br>❌ No GPU acceleration | Sample for large graphs<br>Use PyTorch Geometric for GPU |
| **GRAPH ML** | Graph Neural Net | GNN for fraud | `torch-geometric` | GCN, GAT for node classification | ✅ Relational patterns<br>✅ Money laundering detection | ❌ Complex implementation<br>❌ CPU training very slow | Use for AML specifically<br>Requires graph expertise |
| **GEN AI** | Download LLM | Local language model | `huggingface-cli` | Download Mistral-7B-Instruct GGUF<br>~4 GB model | ✅ No API costs<br>✅ PII stays local<br>✅ Offline inference | ❌ Large download (4-7 GB)<br>❌ Slow on CPU (1-10 sec) | Use quantized (Q4_K_M)<br>Fits in 8 GB RAM |
| **GEN AI** | Load LLM | Inference engine | `llama-cpp-python` | LlamaCpp(model_path=...) | ✅ Fast CPU inference<br>✅ Production-ready | ❌ No GPU acceleration<br>❌ Context limit (4K tokens) | Use for text classification<br>Report generation |
| **GEN AI** | FinBERT | Financial BERT | `transformers` | Download ProsusAI/finbert | ✅ Financial domain<br>✅ Sentiment analysis<br>✅ Pre-trained | ❌ 440 MB model<br>❌ English only | Best for transaction narratives<br>Classify merchant types |
| **GEN AI** | Text Classification | Classify descriptions | `transformers` | AutoModelForSequenceClassification | ✅ Transfer learning<br>✅ Few-shot capable | ❌ Needs labeled data<br>❌ Fine-tuning slow (CPU) | Start with zero-shot<br>Fine-tune if needed |
| **TUNING** | Hyperparameter Opt | Optimize params | `optuna` | Bayesian optimization, 100 trials | ✅ Better than grid search<br>✅ Automatic<br>✅ Dashboard available | ❌ Computationally expensive<br>❌ May overfit on val set | Use TPESampler<br>Monitor on test set |
| **TUNING** | Optuna Dashboard | Visualize trials | `optuna-dashboard` | localhost:8080 dashboard | ✅ Interactive viz<br>✅ Compare trials | ❌ Requires separate server<br>❌ Memory for large studies | Use for >20 trials<br>SQLite backend |
| **EXPLAINABILITY** | SHAP | Feature importance | `shap` | TreeExplainer for XGBoost | ✅ Model-agnostic<br>✅ Local + global<br>✅ Regulatory compliant | ❌ Slow on large data<br>❌ Memory intensive | Sample for large datasets<br>Use TreeExplainer (fast) |
| **EXPLAINABILITY** | LIME | Local explanations | `lime` | LimeTabularExplainer | ✅ Any model type<br>✅ Intuitive explanations | ❌ Slower than SHAP<br>❌ Unstable sometimes | Use for complex models<br>Complement SHAP |
| **EVALUATION** | Metrics | Performance measurement | `scikit-learn.metrics` | F1, Precision, Recall, AUC-ROC | ✅ Standard metrics<br>✅ Imbalanced-aware | ❌ Must pick right metric<br>❌ Multiple metrics confusing | Focus on F1 for fraud<br>AUC-ROC for ranking |
| **EVALUATION** | Confusion Matrix | TP/FP/TN/FN | `scikit-learn.metrics`<br>`seaborn` | Heatmap visualization | ✅ Clear interpretation<br>✅ Identify error types | ❌ Only binary<br>❌ Doesn't show severity | Analyze FP/FN separately<br>Cost-sensitive evaluation |
| **MODEL SELECTION** | Compare Models | Select best | `mlflow.compare`<br>`pandas` | Compare F1, AUC across runs | ✅ Objective comparison<br>✅ Reproducible | ❌ Overfitting to test set<br>❌ Context needed | Use multiple metrics<br>Consider business impact |
| **PACKAGING** | Save Model | Serialize for production | `joblib`<br>`mlflow.log_model` | C:\Models\production\fraud_xgb.pkl | ✅ Fast serialization<br>✅ Cross-version compatible | ❌ File-based only<br>❌ No versioning | Use MLflow for versioning<br>joblib for quick saves |

## DEPLOYMENT OPTIONS: ML/DL PIPELINE

| MODE | ORCHESTRATOR | SCHEDULER | LATENCY | USE CASE | IMPLEMENTATION | SERVING |
|------|--------------|-----------|---------|----------|----------------|---------|
| **Batch (Scheduled)** | APScheduler<br>Prefect | Daily 3 AM | Minutes | Daily scoring of all transactions | `scheduler.add_job(batch_predict, 'cron', hour=3)` | Save to PostgreSQL predictions table |
| **API (Real-time)** | FastAPI | Always-on | <100 ms | Real-time fraud scoring (>$10K) | `@app.post("/predict")` with model in memory | FastAPI at localhost:8000 |
| **Micro-Batch** | Prefect | Every 15 min | 15-20 min | Near real-time for high-value | `@flow` with interval trigger | Batch API with multiple predictions |
| **On-Demand** | Manual / CLI | Ad-hoc | Immediate | Investigation, research | `python predict.py --transaction-id 12345` | Return JSON to stdout |

## ORCHESTRATION DECISION MATRIX: ML TRAINING

| USE CASE | TEAM SIZE | RECOMMENDED | WHY | SETUP TIME |
|----------|-----------|-------------|-----|------------|
| **Quick Experiments** | 1-2 | **Metaflow** | Simplest API, automatic versioning, notebook-friendly | 1 hour |
| **Production Training** | 3-5 | **ZenML** | All-in-one (tracking + registry + orchestration) | 1 day |
| **Modular Approach** | 5-10 | **Prefect + MLflow** | Flexibility, mature tools, large community | 2 days |
| **Team Standardization** | 10+ | **Kedro + MLflow** | Enforced best practices, governance | 3-5 days |

## MODEL SELECTION GUIDE BY DATA SIZE

| DATA SIZE | RECOMMENDED ALGORITHM | WHY | TRAINING TIME (CPU) |
|-----------|----------------------|-----|---------------------|
| **<10K rows** | Logistic Regression | Simple models best, avoid overfitting | <1 minute |
| **10K-100K rows** | XGBoost / Random Forest | Good balance performance/complexity | 10-30 minutes |
| **100K-1M rows** | LightGBM | Fastest GBM, handles large data | 30 min - 2 hours |
| **1M-10M rows** | LightGBM / CatBoost | Optimized for scale | 2-8 hours |
| **>10M rows** | Consider sampling / distributed (out of scope) | Single machine limit | N/A |

## BEST PRACTICES & INDUSTRY STANDARDS

| STANDARD | REQUIREMENT | IMPLEMENTATION | BENEFIT |
|----------|-------------|----------------|---------|
| **SR 11-7** (Fed Reserve) | Model documentation, validation, governance | Model cards, MLflow tracking, approval workflow | Regulatory compliance |
| **GDPR Article 22** | Right to explanation for automated decisions | SHAP/LIME explanations stored with predictions | Legal compliance |
| **Fair Lending (ECOA)** | No discrimination by protected classes | fairlearn library for bias detection | Fairness assurance |
| **PCI DSS** | Secure card data handling | Tokenization, encryption, access control | Security |
| **Cross-Validation** | Stratified K-Fold (k=5) | StratifiedKFold for robust evaluation | Avoid overfitting |
| **Hyperparameter Tuning** | Bayesian optimization (Optuna) | 100-200 trials, TPESampler | Better than grid search |

## KEY RECOMMENDATIONS

### For Fraud Detection:
- **Primary Model:** XGBoost (best performance/interpretability tradeoff)
- **Baseline:** Logistic Regression (regulatory friendly, fast)
- **Advanced:** TabNet (if need deep learning + interpretability)
- **Explainability:** SHAP (must-have for regulatory compliance)
- **Deployment:** API for >$10K transactions, Batch for <$10K

### Special Mentions:
- **XGBoost vs LightGBM:** XGBoost better documented, LightGBM faster
- **TabNet:** Only interpretable deep learning for tabular
- **Local LLMs:** Mistral-7B competitive with GPT-3.5, zero API costs
- **SHAP:** Gold standard for explainability in banking

### Critical Caveats:
- **Class Imbalance:** Fraud is 0.1-1%, don't use accuracy, use F1/AUC
- **Temporal Issues:** Use time-based splits, not random
- **Feature Leakage:** Never use future information
- **Overfitting:** Deep learning needs >100K samples

---

# TABLE 3: RAG & LLMOps PIPELINE

## End-to-End Retrieval Augmented Generation for Policy Q&A

| STAGE | STEP | PURPOSE | TOOLS/LIBRARIES (PyPI) | LOCAL INTEGRATION | PROS | CONS | LIMITATIONS |
|-------|------|---------|------------------------|-------------------|------|------|-------------|
| **LLM SETUP** | Install Framework | Orchestration | `langchain`<br>`langchain-community` | pip install langchain langchain-core | ✅ Industry standard<br>✅ Huge ecosystem<br>✅ Active development | ❌ API changes frequently<br>❌ Complex for simple tasks | Use stable versions<br>Pin dependencies |
| **LLM SETUP** | Download Model | Get GGUF model | `huggingface-cli` | Download Mistral-7B-Instruct-v0.2-GGUF<br>~4 GB to C:\Models\mistral\ | ✅ One-time download<br>✅ Offline forever<br>✅ No API costs | ❌ 4-7 GB storage<br>❌ Initial download time | Use Q4_K_M quantization<br>(3.5 GB, minimal quality loss) |
| **LLM SETUP** | Load LLM | Initialize inference | `llama-cpp-python` | LlamaCpp(model_path=..., n_ctx=4096) | ✅ Fast CPU inference<br>✅ Production-ready<br>✅ Streaming support | ❌ 1-10 sec per response<br>❌ No GPU acceleration | Use n_threads=8<br>Temperature=0 for consistent |
| **EMBEDDINGS** | Install Embeddings | Sentence transformers | `sentence-transformers` | pip install sentence-transformers | ✅ State-of-art embeddings<br>✅ Many models | ❌ First use downloads model<br>❌ 80-400 MB models | Download once, cache locally<br>Use all-MiniLM-L6-v2 (80 MB) |
| **EMBEDDINGS** | Load Embeddings | Initialize | `sentence-transformers` | HuggingFaceEmbeddings(model_name=...) | ✅ 384-dim vectors<br>✅ Fast encoding | ❌ CPU-only slow for bulk<br>❌ No fine-tuning easy | Batch encode (100 at a time)<br>Cache embeddings |
| **VECTOR DB** | Install ChromaDB | Local vector store | `chromadb` | pip install chromadb | ✅ Simple, Python-native<br>✅ Persistent storage<br>✅ No server needed | ❌ Single machine<br>❌ Limited to ~1M vectors | Use for <1M documents<br>Switch to FAISS for >1M |
| **VECTOR DB** | Initialize DB | Setup storage | `chromadb.PersistentClient` | path="C:\VectorDB\chroma" | ✅ Survives restarts<br>✅ SQLite backend | ❌ Disk I/O overhead<br>❌ No distributed | Use SSD for better performance |
| **VECTOR DB** | Alternative: FAISS | High-performance | `faiss-cpu` | pip install faiss-cpu | ✅ Facebook-proven<br>✅ Billion-scale<br>✅ Fastest search | ❌ No metadata<br>❌ Manual persistence | Use for >1M vectors<br>Combine with SQLite for metadata |
| **DOCUMENTS** | Load PDFs | Read policies | `pypdf`<br>`pdfplumber` | DirectoryLoader("C:\Docs\policies\") | ✅ Handles most PDFs<br>✅ Extract tables (pdfplumber) | ❌ Scanned PDFs need OCR<br>❌ Complex layouts fail | Use pytesseract for OCR<br>Manual QA needed |
| **DOCUMENTS** | Load Word Docs | Read DOCX | `python-docx` | Docx2txtLoader | ✅ Preserves structure<br>✅ Fast | ❌ Formatting lost<br>❌ Tables complex | Convert to text only<br>Use markdown for structure |
| **SPLITTING** | Chunk Documents | Split into pieces | `langchain.text_splitter` | RecursiveCharacterTextSplitter<br>(chunk_size=1000, overlap=200) | ✅ Overlap preserves context<br>✅ Configurable | ❌ Breaks mid-sentence sometimes<br>❌ Size tuning needed | Test different sizes<br>1000 tokens good default |
| **SPLITTING** | Semantic Chunking | Intelligent splitting | `langchain.SemanticChunker` | Uses embeddings for boundaries | ✅ Preserves meaning<br>✅ Better quality | ❌ Slower (needs embeddings)<br>❌ Still experimental | Use for critical documents<br>Standard for most |
| **VECTORIZATION** | Generate Embeddings | Create vectors | `sentence-transformers` | embeddings.embed_documents(chunks) | ✅ High-quality vectors<br>✅ Semantic search | ❌ Slow for many docs<br>❌ One-time cost | Batch process (100 chunks)<br>Cache results |
| **VECTORIZATION** | Store Vectors | Save to DB | `chromadb.add` | collection.add(documents, embeddings) | ✅ Automatic indexing<br>✅ Fast lookup | ❌ Disk space (KB per doc)<br>❌ Rebuild expensive | Incremental updates<br>Version collections |
| **RETRIEVAL** | Semantic Search | Find relevant docs | `chromadb.query` | retriever.get_relevant_documents(query, k=5) | ✅ Meaning-based search<br>✅ Better than keywords | ❌ May miss exact matches<br>❌ Embedding quality critical | Combine with keyword (BM25)<br>Hybrid search best |
| **RETRIEVAL** | Re-ranking | Improve relevance | `sentence-transformers.CrossEncoder` | cross-encoder/ms-marco-MiniLM-L-6-v2 | ✅ 20-30% better precision<br>✅ Small model (80 MB) | ❌ Slower (recompute scores)<br>❌ Extra step | Use for top 20 results<br>Re-rank to top 5 |
| **PROMPTS** | Prompt Templates | Structured prompts | `langchain.prompts.PromptTemplate` | "Context: {context}\nQ: {question}" | ✅ Reusable<br>✅ Maintainable<br>✅ Version-controlled | ❌ String-based (errors)<br>❌ No validation | Use f-strings or Jinja2<br>Test thoroughly |
| **PROMPTS** | System Prompts | Define behavior | LangChain system message | "You are a fraud analyst. Answer based only on context." | ✅ Controls behavior<br>✅ Reduces hallucinations | ❌ Token overhead<br>❌ Prompt engineering needed | Iterate on prompts<br>A/B test |
| **RAG CHAIN** | Build Chain | Combine retrieve+generate | `langchain.chains.RetrievalQA` | RetrievalQA.from_chain_type(llm, retriever) | ✅ One line integration<br>✅ Handles complexity | ❌ Less control<br>❌ Black box | Use custom chains for production<br>Better control |
| **RAG CHAIN** | Conversational | Chat with memory | `langchain.ConversationalRetrievalChain` | With chat history | ✅ Multi-turn dialog<br>✅ Remembers context | ❌ Context length limit<br>❌ Memory management needed | Limit to 5-10 turns<br>Summarize old messages |
| **MEMORY** | Conversation Buffer | Store history | `langchain.memory.ConversationBufferMemory` | Stores all messages | ✅ Perfect recall<br>✅ Simple | ❌ Grows unbounded<br>❌ Token limit exceeded | Use ConversationBufferWindowMemory<br>(last K messages) |
| **MEMORY** | Persistent Memory | Save to DB | `langchain.memory.SQLChatMessageHistory` | SQLite: C:\Memory\chat_history.db | ✅ Survives restarts<br>✅ Multi-session | ❌ DB queries overhead<br>❌ Privacy concerns | Encrypt at rest<br>Implement retention policy |
| **GENERATION** | Generate Answer | LLM completion | `llm.invoke(prompt)` | Returns text response | ✅ Natural language<br>✅ Context-aware | ❌ 1-10 seconds<br>❌ May hallucinate | Use temperature=0<br>Add "based only on context" |
| **GENERATION** | Streaming | Token-by-token | `llm.stream(prompt)` | Yields tokens as generated | ✅ Better UX (immediate)<br>✅ Can show progress | ❌ More complex code<br>❌ Harder to debug | Use for user-facing apps<br>EventSourceResponse in FastAPI |
| **POST-PROCESS** | Citation | Add sources | Custom Python | Extract metadata from retrieved docs | ✅ Transparency<br>✅ Verifiable | ❌ Manual implementation<br>❌ Accuracy checking | Always include sources<br>Link to original docs |
| **POST-PROCESS** | Hallucination Check | Verify accuracy | `ragas` | faithfulness score | ✅ Automatic detection<br>✅ Metrics-based | ❌ Not 100% accurate<br>❌ Slow (uses LLM) | Use as filter<br>Human review for critical |
| **FINETUNING** | Prepare Data | Format for training | `datasets` | Instruction-response pairs | ✅ Domain adaptation<br>✅ Better accuracy | ❌ Needs labeled data<br>❌ Training expensive (CPU) | Start with 100-1000 examples<br>Use LoRA |
| **FINETUNING** | LoRA Finetuning | Efficient tuning | `peft`<br>`transformers` | LoraConfig(r=8, lora_alpha=32) | ✅ 100× faster than full<br>✅ 100 MB vs 7 GB<br>✅ Feasible on CPU | ❌ Still slow (hours)<br>❌ Quality trade-off | Use for financial jargon<br>Report formats |
| **EVALUATION** | Relevance | Retrieval quality | `ragas.context_relevancy` | Score 0-1 | ✅ Automatic metric<br>✅ Catches bad retrieval | ❌ Needs ground truth<br>❌ LLM-based (slow) | Sample evaluation (100 queries)<br>Weekly monitoring |
| **EVALUATION** | Faithfulness | Answer accuracy | `ragas.faithfulness` | Are answers grounded in context? | ✅ Detects hallucinations<br>✅ Objective | ❌ False positives<br>❌ Compute intensive | Threshold at 0.8<br>Human review edge cases |
| **CACHING** | Semantic Cache | Cache similar queries | `langchain.cache.SQLiteCache` | Embedding-based similarity | ✅ 80% cache hit (real apps)<br>✅ Huge cost savings | ❌ Stale responses<br>❌ Storage grows | Set TTL (1 week)<br>Similarity threshold 0.95 |
| **SERVING** | API Endpoint | REST service | `fastapi` | @app.post("/query") | ✅ Standard interface<br>✅ Auto docs | ❌ Always-on process<br>❌ Memory overhead | Use uvicorn<br>Restart policy |
| **SERVING** | Streaming API | Server-sent events | `sse-starlette` | EventSourceResponse | ✅ Real-time tokens<br>✅ Better UX | ❌ Complex client<br>❌ Connection management | Use for chat interfaces<br>Fallback to regular |
| **MONITORING** | Token Tracking | Count usage | `tiktoken` | Count input/output tokens | ✅ Cost tracking<br>✅ Performance | ❌ Estimation only<br>❌ Model-specific | Log per request<br>Alert on anomalies |
| **MONITORING** | Latency | Response time | `time.time()` | Measure retrieval + generation | ✅ Identify bottlenecks<br>✅ SLA monitoring | ❌ Varies widely<br>❌ Outliers common | P50, P95, P99 metrics<br>Target <10 sec P95 |
| **GUARDRAILS** | Input Validation | Sanitize queries | `guardrails-ai` | Detect prompt injection | ✅ Security<br>✅ Prevent attacks | ❌ False positives<br>❌ Maintenance | Use for production<br>Whitelist patterns |
| **GUARDRAILS** | Output Validation | Check responses | `guardrails-ai` | Format, safety checks | ✅ Prevent errors<br>✅ Policy compliance | ❌ Rigid<br>❌ May block valid | Define schemas<br>Fallback handling |
| **GUARDRAILS** | PII Detection | Protect data | `presidio-analyzer`<br>`presidio-anonymizer` | Detect SSN, card numbers | ✅ Prevent leaks<br>✅ GDPR compliance | ❌ False positives/negatives<br>❌ Custom entities hard | Use for banking<br>Regular updates |

## DEPLOYMENT ARCHITECTURE: RAG API

| COMPONENT | TOOL | PORT | PURPOSE | STARTUP COMMAND |
|-----------|------|------|---------|-----------------|
| **API Server** | FastAPI + uvicorn | 8000 | Serve RAG queries | `uvicorn rag_api:app --host localhost --port 8000` |
| **Vector DB** | ChromaDB | N/A | Embedded in API process | Initialized in code: `PersistentClient(path=...)` |
| **LLM** | llama-cpp-python | N/A | Embedded in API process | Loaded at startup: `LlamaCpp(model_path=...)` |
| **UI (Optional)** | Chainlit / Streamlit | 8501 | Chat interface | `streamlit run chat_ui.py` |

## BEST PRACTICES & INDUSTRY STANDARDS

| STANDARD | REQUIREMENT | IMPLEMENTATION | BENEFIT |
|----------|-------------|----------------|---------|
| **OWASP LLM Top 10** | Prompt injection, data leakage prevention | guardrails-ai, input validation | Security |
| **NIST AI RMF** | Transparency, human oversight, accuracy | Citations, human-in-loop, monitoring | Risk management |
| **EU AI Act** | High-risk AI requirements | Model cards, quality monitoring, logging | Regulatory compliance |
| **Responsible AI (Microsoft)** | Fairness, reliability, safety, privacy | Test for bias, validate outputs, PII detection | Ethics |
| **RAG Best Practices (LangChain)** | Hybrid search, re-ranking, citations | BM25 + embeddings, CrossEncoder, source tracking | Quality |

## KEY RECOMMENDATIONS

### For Banking Policy Q&A:
- **LLM:** Mistral-7B Q4_K_M (3.5 GB, fast, quality)
- **Embeddings:** all-MiniLM-L6-v2 (80 MB, fast, good)
- **Vector DB:** ChromaDB for <10K docs, FAISS for >10K
- **Deployment:** FastAPI at localhost:8000
- **Caching:** Aggressive semantic caching (80% hit rate)

### Special Mentions:
- **GGUF Format:** Standard for local LLMs, well-supported
- **ChromaDB:** Production-ready, used by startups and enterprises
- **LoRA:** Makes fine-tuning practical (100 MB adapters vs 7 GB models)
- **Guardrails:** Essential for banking (PII detection, prompt injection)

### Critical Caveats:
- **LLM Speed:** 1-10 sec on CPU (not <1 sec like APIs)
- **Context Length:** 4096 tokens (~3000 words), can't fit large docs
- **Hallucinations:** LLMs still hallucinate, need citations + validation
- **Quality Variance:** Less consistent than GPT-4, need testing

---

# TABLE 4: AGENTIC AI & AGENTOPS PIPELINE

## End-to-End Autonomous Agent for Fraud Investigation

| STAGE | STEP | PURPOSE | TOOLS/LIBRARIES (PyPI) | LOCAL INTEGRATION | PROS | CONS | LIMITATIONS |
|-------|------|---------|------------------------|-------------------|------|------|-------------|
| **FRAMEWORK** | Install LangGraph | State machine framework | `langgraph` | pip install langgraph | ✅ Most stable agent framework<br>✅ Production-ready<br>✅ Explicit control flow | ❌ More boilerplate<br>❌ Steeper learning curve | Best for production<br>vs ReAct loops |
| **FRAMEWORK** | Alternative: AutoGen | Multi-agent conversations | `pyautogen` | pip install pyautogen | ✅ Easy multi-agent<br>✅ Good for collaboration | ❌ Less control<br>❌ Can be unpredictable | Use for research<br>LangGraph for prod |
| **FRAMEWORK** | Alternative: CrewAI | Role-based agents | `crewai` | pip install crewai | ✅ Simplest multi-agent<br>✅ Role abstractions | ❌ Less flexible<br>❌ Newer (less battle-tested) | Good for prototyping<br>Simple use cases |
| **GRAPH** | Create State Graph | Define workflow | `langgraph.StateGraph` | StateGraph(AgentState) | ✅ Explicit states<br>✅ Visual flowchart<br>✅ Debuggable | ❌ More code<br>❌ Planning overhead | Define states upfront<br>Draw diagram first |
| **GRAPH** | Define Nodes | Processing steps | `graph.add_node` | analyze_txn, check_rules, query_db | ✅ Modular<br>✅ Testable<br>✅ Reusable | ❌ Many functions<br>❌ Wiring needed | One node per logical step<br>Keep small |
| **GRAPH** | Conditional Routing | Dynamic flow | `graph.add_conditional_edges` | Route based on risk level | ✅ Intelligent routing<br>✅ Handles complexity | ❌ Logic can be tricky<br>❌ Testing needed | Use for branching logic<br>Validate all paths |
| **TOOLS - DB** | SQL Query Tool | Database access | `langchain.agents.create_sql_agent`<br>or custom @tool | Query PostgreSQL for customer history | ✅ Natural language to SQL<br>✅ Automatic joins | ❌ SQL injection risk<br>❌ Can generate bad queries | Use read-only user<br>Validate queries |
| **TOOLS - ML** | Model Prediction | Fraud scoring | Custom `@tool` | Load XGBoost, predict | ✅ Integrates ML<br>✅ Real-time scoring | ❌ Model staleness<br>❌ Explain needed | Version models<br>Include SHAP in tool |
| **TOOLS - SEARCH** | Vector Search | Query knowledge base | Custom `@tool` with ChromaDB | Search AML policies | ✅ Semantic search<br>✅ Context retrieval | ❌ May retrieve irrelevant<br>❌ Ranking critical | Use re-ranking<br>Top-k=5 |
| **TOOLS - CUSTOM** | AML Rules | Business logic | Custom `@tool` | Validate FATF, OFAC rules | ✅ Compliance checks<br>✅ Auditable | ❌ Hardcoded logic<br>❌ Rule maintenance | Externalize rules (JSON)<br>Version control |
| **AGENT - REACT** | ReAct Agent | Reasoning + Acting | `langchain.create_react_agent` | Thought → Action → Observation loop | ✅ Handles complexity<br>✅ Self-correcting | ❌ Infinite loops possible<br>❌ Unpredictable | Set max_iterations=10<br>Monitor closely |
| **AGENT - PLANNER** | Plan-Execute | Multi-step reasoning | `langchain.Plan-Execute` | Plan steps, execute sequentially | ✅ Better for complex tasks<br>✅ More reliable | ❌ Slower (2× calls)<br>❌ Less flexible | Use for investigations<br>Clear goal |
| **MULTI-AGENT** | Supervisor Pattern | Coordinate agents | `langgraph` supervisor | Routes to specialist agents | ✅ Scalable<br>✅ Specialization | ❌ Complex setup<br>❌ Overhead | Use for complex workflows<br>>3 agents |
| **MULTI-AGENT** | Collaborative | Work together | `crewai.Crew` | Analyst + Investigator + Reporter | ✅ Intuitive<br>✅ Division of labor | ❌ Communication overhead<br>❌ Slower | Use for report generation<br>Research tasks |
| **MEMORY** | Checkpoints | Persist state | `langgraph.SqliteSaver` | SQLite: C:\Checkpoints\agent_state.db | ✅ Resume on failure<br>✅ Audit trail | ❌ Disk I/O<br>❌ Cleanup needed | Use for long-running<br>Retention policy |
| **MEMORY** | Conversation Memory | Chat history | `langchain.ConversationBufferMemory` | Stores messages | ✅ Context across turns<br>✅ Simple | ❌ Token limit<br>❌ Growing | Use window memory<br>Last 10 messages |
| **REASONING** | Chain-of-Thought | Step-by-step thinking | Prompt: "Think step by step" | In system prompt | ✅ Better reasoning<br>✅ Explainable | ❌ More tokens<br>❌ Slower | Use for complex decisions<br>Worth latency |
| **REASONING** | Reflection | Self-critique | `langgraph` reflection node | Agent reviews own output | ✅ Quality improvement<br>✅ Catches errors | ❌ 2× slower<br>❌ Still not perfect | Use for critical decisions<br>Final step |
| **HUMAN-IN-LOOP** | Interrupt Before | Pause for approval | `langgraph` interrupt_before=["send_alert"] | Waits for human input | ✅ Control<br>✅ Risk mitigation | ❌ Breaks automation<br>❌ Latency | Use for send_alert, file_sar<br>High-risk actions |
| **HUMAN-IN-LOOP** | Interrupt After | Review output | `langgraph` interrupt_after=["generate_report"] | Shows output, waits | ✅ Quality control<br>✅ Learn from agent | ❌ Slower<br>❌ Manual work | Use for reports<br>Regulatory filings |
| **ERROR HANDLING** | Retry Logic | Handle failures | `tenacity` @retry decorator | Retry tool calls 3× | ✅ Resilience<br>✅ Transient errors | ❌ Can mask bugs<br>❌ Slow on persistent errors | Exponential backoff<br>Max attempts |
| **ERROR HANDLING** | Fallback LLM | Backup model | `langchain` with fallbacks | Smaller/faster model | ✅ Reliability<br>✅ Cost optimization | ❌ Quality drop<br>❌ Complexity | Use Phi-3 as fallback<br>Mistral primary |
| **OPTIMIZATION** | DSPy Optimization | Optimize prompts | `dspy-ai` BootstrapFewShot | Auto-improve prompts | ✅ Better performance<br>✅ Less manual work | ❌ Needs examples<br>❌ Time-consuming | Use for production<br>100+ examples |
| **OPTIMIZATION** | Tool Caching | Cache results | `functools.lru_cache` | Cache DB queries, ML predictions | ✅ 10× faster<br>✅ Reduce load | ❌ Stale data<br>❌ Memory | Short TTL (5 min)<br>Invalidate on update |
| **EVALUATION** | Task Success | Completion rate | Custom Python | % tasks completed successfully | ✅ Objective metric<br>✅ Tracks improvement | ❌ Binary only<br>❌ Doesn't measure quality | Combine with quality metrics<br>Human eval |
| **EVALUATION** | Tool Usage | Efficiency | Count tool invocations | How many tools used? | ✅ Efficiency metric<br>✅ Cost tracking | ❌ More ≠ worse always<br>❌ Context needed | Track over time<br>Optimize outliers |
| **TRACING** | Execution Logging | Track decisions | `loguru` | Log thoughts, actions, observations | ✅ Debugging<br>✅ Audit trail | ❌ Large logs<br>❌ PII risk | Redact PII<br>Rotate logs |
| **TRACING** | Graph Visualization | See workflow | `langgraph.get_graph().draw_mermaid()` | Mermaid diagram | ✅ Understanding<br>✅ Documentation | ❌ Static only<br>❌ Not real-time | Use for design<br>Export to docs |
| **GUARDRAILS** | Input Validation | Sanitize | `guardrails-ai` | Detect prompt injection | ✅ Security<br>✅ Prevent attacks | ❌ False positives<br>❌ Maintenance | Use for all user input<br>Whitelist |
| **GUARDRAILS** | Output Validation | Check responses | `guardrails-ai` | Validate format, safety | ✅ Quality<br>✅ Compliance | ❌ Rigid<br>❌ Can block valid | Define schemas<br>Human override |
| **DEPLOYMENT** | FastAPI Service | Serve as API | `fastapi` | @app.post("/investigate") | ✅ Standard interface<br>✅ Async support | ❌ Always-on<br>❌ Memory | Restart policy<br>Health checks |
| **DEPLOYMENT** | Batch Processing | Multiple inputs | `langgraph.batch` | Process list of alerts | ✅ Parallel execution<br>✅ Efficient | ❌ Resource intensive<br>❌ Complex | Use asyncio.gather<br>Limit concurrency |
| **ORCHESTRATION** | Scheduled Jobs | Periodic execution | APScheduler | Run hourly on new alerts | ✅ Automation<br>✅ Simple | ❌ Process-based<br>❌ Can crash | Use Windows Task Scheduler<br>Or Prefect |
| **MONITORING** | Token Usage | Track LLM calls | `tiktoken` | Count tokens per investigation | ✅ Cost control<br>✅ Performance | ❌ Estimates only<br>❌ Model-specific | Alert on anomalies<br>Budget limits |
| **MONITORING** | Latency | Response time | `time.time()` | Total investigation time | ✅ SLA monitoring<br>✅ Bottleneck ID | ❌ High variance<br>❌ Outliers | P50, P95 metrics<br>Target <2 min P95 |
| **SECURITY** | Audit Logging | Log actions | `loguru` + SQLite | Log decisions, tool calls, outputs | ✅ Compliance<br>✅ Forensics | ❌ Storage grows<br>❌ PII concerns | Encrypt logs<br>Retention policy |

## DEPLOYMENT OPTIONS: AGENTIC AI

| MODE | TRIGGER | LATENCY | USE CASE | IMPLEMENTATION | CONCURRENCY |
|------|---------|---------|----------|----------------|-------------|
| **API (On-Demand)** | POST /investigate | 10-120 sec | Real-time alert investigation | FastAPI with agent.invoke() | 1-10 concurrent |
| **Batch (Scheduled)** | Hourly | Minutes per alert | Process pending alerts | APScheduler with graph.batch() | 10-100 parallel |
| **Streaming** | WebSocket | Real-time updates | Live investigation UI | agent.astream() with WebSocket | 1-5 concurrent |
| **Manual** | CLI | Immediate | Ad-hoc investigation | python investigate.py --alert-id 123 | 1 at a time |

## AGENT FRAMEWORK COMPARISON

| FRAMEWORK | COMPLEXITY | CONTROL | PRODUCTION-READY | BEST FOR | LEARNING CURVE |
|-----------|------------|---------|------------------|----------|----------------|
| **LangGraph** | High | ✅ High | ✅ Yes | Production, complex workflows | Steep |
| **AutoGen** | Medium | Medium | ⚠️ Research | Multi-agent collaboration | Medium |
| **CrewAI** | Low | Low | ⚠️ Early | Simple multi-agent, prototyping | Easy |
| **ReAct (LangChain)** | Low | Low | ⚠️ Experimental | Simple agents, demos | Easy |

## BEST PRACTICES & INDUSTRY STANDARDS

| STANDARD | REQUIREMENT | IMPLEMENTATION | BENEFIT |
|----------|-------------|----------------|---------|
| **OWASP LLM Top 10** | Agent security | Guardrails on input/output, tool validation | Prevent prompt injection, data leaks |
| **NIST AI RMF** | Risk management | Testing, monitoring, human oversight, documentation | Governance, accountability |
| **Human-in-Loop** | High-risk actions require approval | interrupt_before for send_alert, file_sar | Regulatory compliance, risk mitigation |
| **Tool Validation** | Validate tool inputs/outputs | Pydantic schemas, type checking | Prevent errors, security |
| **Observability** | Log all decisions | LangSmith (local), custom logging | Debugging, auditing |

## KEY RECOMMENDATIONS

### For Fraud Investigation:
- **Framework:** LangGraph (most production-ready)
- **Tools:** SQL (customer history), ML (fraud score), RAG (policies), AML rules
- **Deployment:** API for real-time, Batch for pending alerts
- **Human-in-Loop:** Required for send_alert, file_SAR
- **Monitoring:** Log every decision, tool call, output

### Special Mentions:
- **LangGraph:** Industry-leading for production agents
- **Checkpoints:** Enable resume on failure (critical for reliability)
- **DSPy:** Automated prompt optimization (20-50% improvement)
- **Semantic Router:** 30-50% faster routing (embeddings vs LLM)

### Critical Caveats:
- **Reliability:** Agents fail ~5-10% of time, need error handling
- **Cost:** 5-20× more LLM calls than single-shot (track tokens)
- **Latency:** 10-120 sec per investigation (not <1 sec)
- **Determinism:** Non-deterministic even with temperature=0
- **Testing:** Hard to test comprehensively (too many paths)

---

# TABLE 5: MLOps / MODEL LIFECYCLE MANAGEMENT

## End-to-End ML Operations for Production Fraud Models

## OPTION A: ZENML ALL-IN-ONE (RECOMMENDED FOR REGIONAL BANKS)

| STAGE | STEP | PURPOSE | TOOLS/LIBRARIES (PyPI) | LOCAL INTEGRATION | PROS | CONS | LIMITATIONS |
|-------|------|---------|------------------------|-------------------|------|------|-------------|
| **SETUP** | Install ZenML | MLOps platform | `zenml` | pip install zenml | ✅ All-in-one (tracking+registry+orchestration)<br>✅ Built-in lineage<br>✅ Type-safe | ❌ Opinionated<br>❌ Smaller community vs MLflow | Use for teams <10 people<br>Simpler than modular |
| **SETUP** | Initialize | Create structure | `zenml init` | Creates project in current dir | ✅ Standard structure<br>✅ Best practices | ❌ Fixed layout<br>❌ Migration from existing | Start fresh project<br>Or migrate gradually |
| **SETUP** | Start Dashboard | Local UI | `zenml up` | http://localhost:8237 | ✅ Beautiful UI<br>✅ Real-time updates<br>✅ No config | ❌ Another process<br>❌ Port conflict possible | Use 8237 (default)<br>Or custom port |
| **PIPELINE** | Define Steps | ML workflow | `@step` decorator | Data load, train, evaluate | ✅ Modular<br>✅ Reusable<br>✅ Testable | ❌ Boilerplate<br>❌ Learning curve | One step per logical task<br>Keep focused |
| **PIPELINE** | Create Pipeline | Orchestrate | `@pipeline` decorator | Connects steps | ✅ DAG visualization<br>✅ Dependency management | ❌ Execution overhead<br>❌ Debug harder | Use for training<br>Not for EDA |
| **TRACKING** | Auto-Tracking | Built-in | ZenML automatic | Parameters, metrics, artifacts all tracked | ✅ No manual logging<br>✅ Complete lineage<br>✅ Reproducible | ❌ Overhead (slower)<br>❌ Storage grows | Automatic everything<br>Review retention |
| **REGISTRY** | Model Registry | Built-in | ZenML automatic | Models registered after training | ✅ Versioning<br>✅ Metadata<br>✅ Promotion (staging→prod) | ❌ Embedded (no separate)<br>❌ Less flexible | Use stages: None→Staging→Production |
| **VERSIONING** | Pipeline Version | Built-in | ZenML automatic | Every run versioned | ✅ Complete reproducibility<br>✅ Rollback easy | ❌ Storage overhead<br>❌ Cleanup needed | Archive old versions<br>Retention policy |
| **LINEAGE** | Data Lineage | Built-in | ZenML automatic | Tracks data→transforms→model→predictions | ✅ Compliance (SR 11-7)<br>✅ Visual graph | ❌ Performance overhead<br>❌ Complex graphs | Critical for banking<br>Document all |
| **DEPLOYMENT** | Model Serving | Deploy to endpoint | ZenML + FastAPI | Load model version, serve API | ✅ Simple deployment<br>✅ Version control | ❌ Manual FastAPI<br>❌ No auto-scaling | Use FastAPI template<br>Load by version |
| **MONITORING** | Dashboard | Built-in UI | ZenML dashboard | View in localhost:8237 | ✅ All in one place<br>✅ No extra tools | ❌ Limited customization<br>❌ No alerting | Use for overview<br>Dash for detailed |

**Total Setup Time: 1 day**  
**Complexity: Low-Medium**  
**Best For: Regional banks (3-5 person teams)**

## OPTION B: PREFECT + MLFLOW (MODULAR FOR LARGE BANKS)

| STAGE | STEP | PURPOSE | TOOLS/LIBRARIES (PyPI) | LOCAL INTEGRATION | PROS | CONS | LIMITATIONS |
|-------|------|---------|------------------------|-------------------|------|------|-------------|
| **TRACKING SETUP** | Install MLflow | Experiment tracking | `mlflow` | pip install mlflow | ✅ Industry standard<br>✅ 10+ years mature<br>✅ Huge community | ❌ Separate from orchestration<br>❌ No built-in workflow | Use for tracking only<br>Combine with Prefect |
| **TRACKING SETUP** | Start Server | Tracking UI | `mlflow server` | --backend-store-uri sqlite:///mlflow.db<br>--host localhost:5000 | ✅ Persistent storage<br>✅ Multi-user | ❌ Another process<br>❌ Database management | Use SQLite for small<br>PostgreSQL for production |
| **TRACKING** | Create Experiment | Organize runs | `mlflow.create_experiment` | Name: "fraud_detection_v2" | ✅ Organization<br>✅ Separate projects | ❌ Manual creation<br>❌ Naming conventions | Use descriptive names<br>Version in name |
| **TRACKING** | Log Parameters | Track configs | `mlflow.log_param` | learning_rate, n_estimators, etc. | ✅ Complete record<br>✅ Searchable | ❌ Manual logging<br>❌ Verbose | Log all hyperparams<br>Use dict for many |
| **TRACKING** | Log Metrics | Track performance | `mlflow.log_metric` | f1_score, auc_roc, precision | ✅ Time-series tracking<br>✅ Compare runs | ❌ Manual logging<br>❌ Many calls | Log final + intermediate<br>Use step for training |
| **TRACKING** | Log Artifacts | Save outputs | `mlflow.log_artifact` | Plots, models, data samples | ✅ Everything in one place<br>✅ Versioned | ❌ Storage grows<br>❌ Large files slow | Use for critical artifacts<br>Link large files |
| **TRACKING** | Log Model | Save model | `mlflow.sklearn.log_model` | Auto-packages with dependencies | ✅ Reproducible<br>✅ Easy deployment | ❌ Model size limits<br>❌ Serialization issues | Test load before deploy<br>Use pickle for custom |
| **REGISTRY** | Register Model | Add to registry | `mlflow.register_model` | Name: "FraudDetector" | ✅ Central registry<br>✅ Versioning | ❌ Separate step<br>❌ Manual | Register after validation<br>Use aliases |
| **REGISTRY** | Model Stages | Lifecycle | `mlflow.set_model_stage` | None→Staging→Production→Archived | ✅ Clear promotion path<br>✅ Multiple versions | ❌ Manual promotion<br>❌ No approval workflow | Require testing before Production<br>Document process |
| **ORCHESTRATION** | Install Prefect | Workflow engine | `prefect` | pip install prefect | ✅ Modern UI<br>✅ Great for ML<br>✅ Local+cloud | ❌ Separate from MLflow<br>❌ Integration needed | Use for orchestration<br>Works with MLflow |
| **ORCHESTRATION** | Start Server | Prefect UI | `prefect server start` | http://localhost:4200 | ✅ Visual workflows<br>✅ Monitoring | ❌ Another process<br>❌ Resource usage | Use for production<br>Not development |
| **ORCHESTRATION** | Create Flow | Training workflow | `@flow` decorator | Combine data→train→evaluate→register | ✅ Dependency management<br>✅ Retries<br>✅ Caching | ❌ More code<br>❌ Learning curve | Use for complex workflows<br>Simple for scripts |
| **ORCHESTRATION** | Schedule | Periodic training | Prefect deployment | Weekly retrain on Sundays | ✅ Reliable scheduling<br>✅ No cron | ❌ Deployment setup<br>❌ YAML configs | Use for production<br>APScheduler for simple |

**Total Setup Time: 2-3 days**  
**Complexity: Medium**  
**Best For: Large banks (5-10 person teams, want flexibility)**

## COMMON STEPS (BOTH OPTIONS)

| STAGE | STEP | PURPOSE | TOOLS/LIBRARIES (PyPI) | LOCAL INTEGRATION | PROS | CONS | LIMITATIONS |
|-------|------|---------|------------------------|-------------------|------|------|-------------|
| **VERSIONING** | Data Versioning | Track datasets | `dvc` | dvc add data/transactions.parquet<br>Local storage: C:\dvc_storage | ✅ Git-like for data<br>✅ Reproducibility<br>✅ Storage efficient | ❌ Requires Git<br>❌ Learning curve<br>❌ Large files slow | Use for datasets >100 MB<br>Not for models (use MLflow/ZenML) |
| **VALIDATION** | Cross-Validation | Robust evaluation | `scikit-learn.model_selection` | StratifiedKFold(n_splits=5, shuffle=True) | ✅ Reduces variance<br>✅ Better estimates<br>✅ Standard practice | ❌ 5× training time<br>❌ Not for time series | Use for fraud detection<br>TimeSeriesSplit for temporal |
| **VALIDATION** | Performance Metrics | Measure quality | `scikit-learn.metrics` | F1, Precision, Recall, AUC-ROC, Confusion Matrix | ✅ Comprehensive<br>✅ Standard metrics<br>✅ Comparable | ❌ Many metrics confusing<br>❌ Must pick primary | Focus on F1 for imbalanced<br>AUC for ranking |
| **VALIDATION** | Deepchecks Suite | Comprehensive validation | `deepchecks` | Suite().run(train, test, model) | ✅ 50+ checks<br>✅ Catches issues<br>✅ HTML reports | ❌ Slow on large data<br>❌ Many checks overwhelming | Use before deployment<br>Sample large data |
| **EXPLAINABILITY** | SHAP Values | Global + local importance | `shap` | TreeExplainer for XGBoost<br>KernelExplainer for any model | ✅ Model-agnostic<br>✅ Regulatory compliant<br>✅ Visual plots | ❌ Slow for large data<br>❌ Memory intensive<br>❌ TreeExplainer only for trees | Sample for SHAP (1000 rows)<br>Cache results |
| **EXPLAINABILITY** | LIME | Alternative explanations | `lime` | LimeTabularExplainer | ✅ Works with any model<br>✅ Intuitive | ❌ Slower than SHAP<br>❌ Less stable | Use for complex models<br>Complement SHAP |
| **TESTING** | Unit Tests | Test components | `pytest` | Test predict(), fit(), transform() | ✅ Catch bugs early<br>✅ Refactoring safe<br>✅ Documentation | ❌ Test maintenance<br>❌ Time investment | Test critical paths<br>Mock expensive ops |
| **TESTING** | Integration Tests | Test pipeline | `pytest` | End-to-end: data→train→predict | ✅ Realistic testing<br>✅ Catch integration issues | ❌ Slow (minutes)<br>❌ Hard to debug | Run before deployment<br>Use small data |
| **GOVERNANCE** | Model Card | Documentation | Markdown template | model_card.md with metrics, limitations, fairness | ✅ Transparency<br>✅ SR 11-7 compliant<br>✅ Stakeholder communication | ❌ Manual effort<br>❌ Keeping updated | Use template<br>Auto-generate parts |
| **FAIRNESS** | Bias Detection | Detect discrimination | `fairlearn` | Check demographic parity, equalized odds | ✅ Regulatory requirement<br>✅ Quantifiable<br>✅ Visual dashboard | ❌ Requires demographic data<br>❌ Trade-offs complex | Use for lending/credit<br>Document decisions |
| **MONITORING** | Data Drift | Distribution changes | `evidently` | DataDriftPreset(), KS test, PSI | ✅ Automatic detection<br>✅ HTML reports<br>✅ Multiple tests | ❌ Threshold tuning<br>❌ False positives | Weekly monitoring<br>Bank-specific thresholds |
| **MONITORING** | Model Performance | Production metrics | `evidently` + MLflow/ZenML | Log production F1, AUC daily | ✅ Track degradation<br>✅ Trigger retraining | ❌ Needs ground truth (delayed for fraud)<br>❌ Lag time | Use proxy metrics<br>Monthly validation |
| **MONITORING** | Prediction Drift | Output distribution | `evidently` | Check if predictions change | ✅ Catches model issues<br>✅ Early warning | ❌ Noisy metric<br>❌ False alarms | Combine with data drift<br>Investigate together |
| **MONITORING** | System Metrics | Latency, throughput | `prometheus-client`<br>`psutil` | Histogram for latency<br>Counter for requests<br>Gauge for CPU/memory | ✅ Operational insights<br>✅ SLA monitoring<br>✅ Capacity planning | ❌ No built-in dashboard<br>❌ Prometheus server needed | Use Dash for viz<br>Alert on SLA breach |
| **MONITORING** | Reports | Periodic HTML | `evidently` | Generate weekly drift + performance report | ✅ Shareable<br>✅ Stakeholder-friendly<br>✅ Archive | ❌ Static snapshots<br>❌ Not real-time | Weekly cadence<br>Email to team |
| **MONITORING** | Dashboard | Real-time UI | `dash` + `plotly` | localhost:8050 custom dashboard | ✅ Custom metrics<br>✅ Interactive<br>✅ Production-ready | ❌ Development time<br>❌ Maintenance | Build incrementally<br>Start with KPIs |
| **ALERTING** | Drift Alerts | Threshold-based | Custom Python + `yagmail` | If drift_score > 0.3: send_email() | ✅ Proactive<br>✅ Prevents issues | ❌ Threshold tuning<br>❌ Alert fatigue | Set bank-specific thresholds<br>Weekly review |
| **ALERTING** | Performance Alerts | Degradation | Custom Python | If f1_score < 0.85: alert() | ✅ Catch regressions<br>✅ Trigger retraining | ❌ False positives<br>❌ Lag for ground truth | Use rolling window (30 days)<br>Combine signals |
| **RETRAINING** | Scheduled | Periodic | APScheduler | Monthly on 1st day | ✅ Consistent updates<br>✅ Simple | ❌ May be wasteful<br>❌ Not responsive | Good baseline<br>Combine with triggers |
| **RETRAINING** | Performance-Based | On degradation | Custom trigger | If F1 < threshold for 7 days: retrain | ✅ Responsive<br>✅ Efficient | ❌ Needs ground truth<br>❌ Delayed for fraud | Use proxy metrics<br>Manual validation |
| **RETRAINING** | Drift-Based | On drift detection | evidently trigger | If data drift detected: retrain | ✅ Proactive<br>✅ Prevents degradation | ❌ May retrain unnecessarily<br>❌ Drift ≠ performance | Validate before deploy<br>Combine signals |
| **DEPLOYMENT - BATCH** | Batch Scoring | Scheduled predictions | APScheduler + MLflow/ZenML | Load production model<br>Predict daily batch<br>Save to PostgreSQL | ✅ High throughput<br>✅ Simple<br>✅ Cost-effective | ❌ High latency<br>❌ Not real-time | Use for daily review<br>All transactions |
| **DEPLOYMENT - API** | API Serving | Real-time | FastAPI + uvicorn | Load model at startup<br>@app.post("/predict")<br>Return prediction + SHAP | ✅ Low latency (<100ms)<br>✅ Standard interface<br>✅ Scalable | ❌ Always-on process<br>❌ Memory overhead<br>❌ Single machine limit | Use for high-value (>$10K)<br>Load balancer for scale |
| **DEPLOYMENT - API** | Health Check | Monitoring | FastAPI | @app.get("/health")<br>Return model loaded, version, uptime | ✅ Operational visibility<br>✅ Load balancer integration | ❌ Basic only<br>❌ No deep checks | Check model loaded<br>Predict on sample |
| **SECURITY** | Model Encryption | Protect IP | `cryptography` | Encrypt .pkl files at rest | ✅ IP protection<br>✅ Compliance | ❌ Key management<br>❌ Performance overhead | Use for production<br>Store keys securely |
| **SECURITY** | API Authentication | Secure endpoints | `python-jose` + JWT | Require JWT token for /predict | ✅ Access control<br>✅ Audit trail | ❌ Token management<br>❌ Complexity | Use for production<br>Short-lived tokens |
| **BACKUP** | Model Backup | Disaster recovery | `shutil` + `zipfile` | Backup to C:\Backups\models\{date}\ | ✅ Disaster recovery<br>✅ Rollback | ❌ Storage growth<br>❌ Manual cleanup | Weekly full backup<br>30-day retention |

## DECISION MATRIX: MLOPS STACK BY BANK SIZE

| BANK SIZE | TRANSACTIONS/DAY | TEAM SIZE | RECOMMENDED STACK | SETUP TIME | WHY |
|-----------|------------------|-----------|-------------------|------------|-----|
| **Small (<$1B)** | 10K-100K | 1-2 | **Metaflow + MLflow** | 1 day | Simplest for experimentation, manual tracking sufficient |
| **Regional ($1B-$10B)** | 100K-1M | 3-5 | **ZenML** ⭐ | 1-2 days | All-in-one reduces complexity, built-in lineage critical |
| **Large ($10B-$100B)** | 1M-10M | 5-10 | **ZenML** or **Prefect + MLflow** | 1-3 days | Either works; ZenML for simplicity, Prefect+MLflow for flexibility |
| **Global (>$100B)** | 10M+ | 10+ | **Kedro + Dagster + MLflow** | 1 week | Maximum governance + flexibility, dedicated MLOps team |

## DEPLOYMENT COMPARISON: BATCH vs API

| ASPECT | BATCH (SCHEDULED) | API (REAL-TIME) | RECOMMENDATION |
|--------|-------------------|-----------------|----------------|
| **Latency** | Minutes to hours | <100 ms | Use API for >$10K transactions |
| **Throughput** | 100K-1M/batch | 100-1000 req/sec | Use batch for daily review |
| **Complexity** | Low | Medium | Start with batch, add API later |
| **Resource Usage** | Bursty (high during job) | Constant (always-on) | Batch more efficient for low volume |
| **Cost** | Low (run once/day) | Medium (always running) | Batch for budget-conscious |
| **Use Case** | Daily fraud review, reports | Real-time authorization, high-value | Hybrid: batch for <$10K, API for >$10K |
| **Failure Impact** | Low (can retry) | High (transaction blocked) | API needs monitoring, retries |
| **Implementation** | APScheduler + Python script | FastAPI + uvicorn | Both straightforward |

## BEST PRACTICES & INDUSTRY STANDARDS

| STANDARD | REQUIREMENT | IMPLEMENTATION | BENEFIT |
|----------|-------------|----------------|---------|
| **SR 11-7** (Federal Reserve) | Model risk management: documentation, validation, governance | Model cards + MLflow/ZenML tracking + approval workflow + monitoring | Regulatory compliance |
| **GDPR Article 22** | Right to explanation for automated decisions | SHAP/LIME explanations stored with predictions in PostgreSQL | Legal compliance |
| **Fair Lending (ECOA)** | No discrimination by protected classes (race, gender, etc.) | fairlearn bias metrics + disparate impact testing | Fairness, avoid lawsuits |
| **PCI DSS** | Secure payment card data | Tokenization, encryption at rest, access control, audit logs | Security compliance |
| **ISO/IEC 5338:2023** | AI lifecycle standard | Follow MLOps pipeline stages (plan, develop, deploy, monitor, retire) | International best practice |
| **SOX** | Auditability, change control | Git for code, DVC for data, MLflow/ZenML for models, approval logs | Financial compliance |
| **CD4ML** (Continuous Delivery for ML) | Treat models like software: version control, testing, CI/CD | Git + DVC + pytest + automated pipelines | Software engineering rigor |
| **Model Cards** (Google) | Document purpose, performance, limitations, bias, fairness | Use model_card template, auto-generate from MLflow/ZenML | Transparency |
| **MLOps Maturity Model** (Microsoft) | Level 0→4: Manual→Full automation | Aim for Level 2-3 (automated training, deployment) on local | Progressive improvement |

## KEY RECOMMENDATIONS

### For Most Banks (Regional):
- **Stack:** ZenML (all-in-one simplicity)
- **Deployment:** Hybrid (batch for all, API for >$10K)
- **Monitoring:** evidently (weekly drift reports) + Dash (real-time dashboard)
- **Retraining:** Monthly scheduled + drift-triggered
- **Explainability:** SHAP (must-have for regulatory compliance)

### Special Mentions:
- **ZenML:** Game-changer for small/medium teams (40% less work vs modular)
- **evidently:** Industry standard for ML monitoring
- **Deepchecks:** Catches 80% of issues before production
- **SHAP:** Non-negotiable for banking (regulatory requirement)
- **Model Cards:** Required by SR 11-7, NIST AI RMF

### Critical Caveats:
- **Drift ≠ Performance Drop:** Monitor both separately
- **Ground Truth Delay:** Fraud labels delayed weeks/months (use proxies)
- **Retraining Frequency:** Monthly is typical, drift-based is better
- **A/B Testing:** Champion vs challenger before full deployment
- **Single Machine Limits:** Monitor disk (models grow), RAM (caching), CPU (inference)

---

# CROSS-CUTTING CONCERNS

## WINDOWS-SPECIFIC CONSIDERATIONS

| CONCERN | ISSUE | SOLUTION | BEST PRACTICE |
|---------|-------|----------|---------------|
| **File Paths** | Backslash \ vs forward slash / | Use `pathlib.Path()` always | Cross-platform compatible |
| **Case Sensitivity** | Windows case-insensitive, Git/Linux case-sensitive | Enforce lowercase for file/folder names | Avoid surprises in deployment |
| **Long Paths** | 260 character limit | Enable long path support in Windows 10+<br>Or use shorter paths | Set in registry or Group Policy |
| **Line Endings** | CRLF vs LF | Git config: `core.autocrlf=true` | Consistency in version control |
| **Executables** | .exe files not allowed | Use Python scripts, pip packages only | All tools must be pip-installable |
| **Admin Rights** | Limited permissions | No installers requiring admin<br>Use portable/pip versions | PostgreSQL portable, not installer |
| **Process Management** | No systemd/init | Use Windows Task Scheduler for reliability<br>Or Python-based (APScheduler) | Task Scheduler for production |
| **Firewall** | Windows Firewall blocks localhost sometimes | Allow Python, uvicorn through firewall<br>Or disable for localhost | Test connectivity |

## COST ANALYSIS: OPEN SOURCE vs COMMERCIAL

| COMPONENT | OPEN SOURCE (OUR STACK) | COMMERCIAL ALTERNATIVE | ANNUAL SAVINGS |
|-----------|-------------------------|------------------------|----------------|
| **ETL/Orchestration** | Prefect (free) | Informatica ($50K-200K) | $50K-200K |
| **ML Training** | XGBoost + scikit-learn (free) | DataRobot ($100K-500K) | $100K-500K |
| **MLOps** | ZenML/MLflow (free) | Dataiku ($50K-200K) | $50K-200K |
| **LLMs** | Mistral-7B local (free) | OpenAI API ($10K-50K/year) | $10K-50K |
| **Vector DB** | ChromaDB (free) | Pinecone ($70-280/month) | $1K-3K |
| **Monitoring** | Dash + evidently (free) | Datadog ($15-31/host/month) | $5K-20K |
| **BI/Dashboards** | Dash (free) | Tableau ($70/user/month) | $10K-50K |
| **Database** | PostgreSQL (free) | Oracle ($47.5K/core) | $50K-200K |
| **TOTAL ANNUAL** | **$0** | **$275K-1.5M** | **$275K-1.5M** |

**Additional Benefits:**
- No vendor lock-in
- Full customization
- No data upload (PII stays local)
- No per-user/per-GB pricing
- Community support (thousands of developers)

## PERFORMANCE BENCHMARKS (Laptop: Intel i7, 32 GB RAM, Windows 11)

| TASK | TOOL | DATA SIZE | TIME | NOTES |
|------|------|-----------|------|-------|
| **Load CSV** | pandas | 1M rows | 5 sec | Standard baseline |
| **Load CSV** | polars | 1M rows | 1 sec | 5× faster |
| **SQL Query** | PostgreSQL | 10M rows | 2 sec | Indexed |
| **Analytics Query** | DuckDB on Parquet | 10M rows | 3 sec | 10× faster than pandas |
| **Train XGBoost** | CPU | 100K rows, 50 features | 5 min | n_estimators=100 |
| **Train LightGBM** | CPU | 100K rows, 50 features | 2 min | Faster than XGBoost |
| **Train TabNet** | CPU | 100K rows, 50 features | 30 min | Deep learning slow on CPU |
| **SHAP Explain** | TreeExplainer | 1000 samples | 10 sec | Fast for trees |
| **LLM Inference** | Mistral-7B Q4_K_M | Single query | 3-8 sec | CPU-only, depends on length |
| **LLM Inference** | Mistral-7B Q4_K_M | Batch 10 queries | 25 sec | 2.5 sec/query amortized |
| **Embed 1000 docs** | sentence-transformers | 1000 × 500 words | 30 sec | all-MiniLM-L6-v2 |
| **Vector Search** | ChromaDB | 10K vectors, top-5 | <100 ms | Fast enough |
| **Agent Investigation** | LangGraph + Mistral-7B | 1 fraud alert | 30-120 sec | Multiple LLM calls |
| **Batch Predictions** | XGBoost | 100K transactions | 10 sec | FastAPI can handle |
| **API Prediction** | XGBoost | Single transaction | 20 ms | Well within <100ms SLA |

## SCALABILITY LIMITS (Single Windows 11 Laptop)

| RESOURCE | LIMIT | WORKAROUND |
|----------|-------|------------|
| **RAM** | 32-64 GB max (consumer laptops) | Use polars (out-of-core), DuckDB (external memory), sampling |
| **Disk** | 1-2 TB SSD | Use NAS for cold storage, Parquet compression (10×), retention policies |
| **CPU** | 8-16 cores typical | Use multiprocessing (joblib, polars), GPU for DL (if available) |
| **Concurrency** | ~100-500 req/sec (FastAPI) | Use batch for high volume, scale horizontally (multiple machines) |
| **Database** | PostgreSQL ~100 GB practical | Archive old data, use partitioning, read replicas |
| **Vector DB** | ChromaDB ~1M vectors | Use FAISS for >1M, hierarchical clustering, pruning |
| **LLM Context** | 4096 tokens (~3000 words) | Chunk documents, sliding window, summarization |
| **Model Size** | Limited by RAM (7B LLM = 8 GB) | Use quantization (Q4_K_M = 4 GB), smaller models (Phi-3) |

**When to Scale Beyond Single Machine:**
- Transactions: >10M/day
- Data: >1 TB active data
- Users: >100 concurrent analysts
- Models: >50 production models
- Latency: <10 ms required (need GPU, caching, CDN)

---

# IMPLEMENTATION ROADMAP

## PHASE 1: FOUNDATION (WEEKS 1-4)

| WEEK | MILESTONE | DELIVERABLE | EFFORT |
|------|-----------|-------------|--------|
| **Week 1** | Data Engineering | ETL pipeline (polars + PostgreSQL + Prefect)<br>Data quality (great-expectations) | 40 hours |
| **Week 2** | ML Baseline | Logistic Regression + XGBoost<br>Evaluation framework | 40 hours |
| **Week 3** | MLOps Setup | ZenML or MLflow+Prefect<br>Experiment tracking | 40 hours |
| **Week 4** | Monitoring | Dash dashboard + evidently drift detection | 40 hours |

**Total: 160 hours (1 person-month)**

## PHASE 2: ADVANCED ML (WEEKS 5-8)

| WEEK | MILESTONE | DELIVERABLE | EFFORT |
|------|-----------|-------------|--------|
| **Week 5** | Advanced Models | LightGBM, CatBoost, TabNet<br>Hyperparameter tuning (Optuna) | 40 hours |
| **Week 6** | Explainability | SHAP integration, LIME<br>Model cards | 40 hours |
| **Week 7** | Deployment | Batch scoring + FastAPI<br>Health checks, monitoring | 40 hours |
| **Week 8** | Testing & Validation | Unit tests, integration tests<br>Deepchecks validation suite | 40 hours |

**Total: 160 hours (1 person-month)**

## PHASE 3: AI CAPABILITIES (WEEKS 9-12)

| WEEK | MILESTONE | DELIVERABLE | EFFORT |
|------|-----------|-------------|--------|
| **Week 9** | RAG Setup | Download Mistral-7B, ChromaDB<br>Policy Q&A system | 40 hours |
| **Week 10** | RAG Production | API endpoint, caching, guardrails<br>Evaluation (ragas) | 40 hours |
| **Week 11** | Agentic AI | LangGraph fraud investigator<br>Tools (SQL, ML, RAG, rules) | 40 hours |
| **Week 12** | Agent Production | API + batch deployment<br>Human-in-loop, monitoring | 40 hours |

**Total: 160 hours (1 person-month)**

## PHASE 4: PRODUCTION HARDENING (WEEKS 13-16)

| WEEK | MILESTONE | DELIVERABLE | EFFORT |
|------|-----------|-------------|--------|
| **Week 13** | Security | Encryption, API auth (JWT)<br>PII detection (presidio) | 40 hours |
| **Week 14** | Governance | Model cards, fairness testing<br>Audit logging | 40 hours |
| **Week 15** | Performance Optimization | Caching, batch optimization<br>Load testing | 40 hours |
| **Week 16** | Documentation & Handoff | User guides, runbooks<br>Training materials | 40 hours |

**Total: 160 hours (1 person-month)**

**GRAND TOTAL: 640 hours = 4 person-months = 16 weeks (4 months for 1 person, 2 months for 2 people)**

---

# SUMMARY: COMPLETE TECHNOLOGY STACK

## DATA ENGINEERING
- **Processing:** polars (primary), pandas (compatibility)
- **Database:** PostgreSQL (portable)
- **Analytics:** DuckDB (SQL on Parquet)
- **Orchestration:** Prefect (UI) or APScheduler (simple)
- **Quality:** great-expectations
- **Versioning:** DVC

## MACHINE LEARNING
- **Algorithms:** XGBoost (primary), LightGBM, CatBoost, TabNet
- **Framework:** scikit-learn (baseline), PyTorch (deep learning)
- **Tuning:** Optuna
- **Explainability:** SHAP (must-have), LIME

## LLMOps (RAG)
- **LLM:** Mistral-7B GGUF (local inference)
- **Embeddings:** sentence-transformers (all-MiniLM-L6-v2)
- **Vector DB:** ChromaDB (<1M vectors), FAISS (>1M)
- **Framework:** LangChain
- **Evaluation:** ragas
- **Guardrails:** guardrails-ai, presidio

## AGENTIC AI
- **Framework:** LangGraph (production)
- **Alternatives:** AutoGen (research), CrewAI (simple)
- **Tools:** SQL, ML prediction, RAG search, AML rules
- **Optimization:** DSPy (prompt optimization)

## MLOps
- **Regional Banks:** ZenML (all-in-one)
- **Large Banks:** Prefect + MLflow (modular)
- **Small Banks:** Metaflow + MLflow (simple)
- **Monitoring:** evidently (drift) + Dash (dashboards)
- **Testing:** pytest, Deepchecks
- **Governance:** Model cards, fairlearn, audit logs

## DEPLOYMENT
- **Batch:** APScheduler + Python scripts
- **API:** FastAPI + uvicorn
- **UI:** Dash (dashboards), Streamlit (prototypes), Chainlit (chat)

## MONITORING & OBSERVABILITY
- **Dashboards:** Dash + plotly
- **Metrics:** prometheus-client + psutil
- **Drift:** evidently
- **Logging:** loguru
- **Alerting:** yagmail (email)

## TOTAL COST: $0 (ALL OPEN SOURCE)
## TOTAL SETUP: 4 PERSON-MONTHS
## DEPLOYMENT: LOCAL WINDOWS 11 ONLY
## PII: 100% ON-PREMISES

---

**END OF COMPREHENSIVE PIPELINE TABLES**

*All pipelines designed for local Windows 11 deployment, using only PyPI packages, with complete PII protection and zero cloud dependencies.*

*Document Version: 1.0*  
*Last Updated: November 15, 2025*  
*Target: Bank AML/Fraud/AI Systems*
