# Production architecture

```mermaid
flowchart LR
    A[Legacy FY24-FY26 Excel / Future Standard Upload] --> B[Source Adapters]
    B --> C[Canonical Data Layer]
    C --> D[Validation + EDA + Reconciliation]
    D --> E[(Versioned Certified Dataset)]
    E --> F[Time-aware Training / Backtesting]
    F --> G[(Model Registry)]
    G --> H[Forecast Engine]
    H --> I[Closing + Exit]
    H --> J[NOP + ATS]
    I --> K[Derived Opening / Hiring / Attrition]
    J --> L[Product Mix / Business Drivers]
    K --> M[Scenario Engine]
    L --> M
    M --> N[Baseline vs Scenario]
    N --> O[Deterministic Analytics]
    O --> P[LLM Insight Layer]
    P --> Q[Flask API / Business UI]
    E --> R[New monthly actuals append]
    R --> D
```

## Key separation
- Upload != training.
- Scenarios != training data.
- LLM != numeric forecast model.
- Legacy sheet/column names live only in source adapters.
- All ML trains on canonical columns.
