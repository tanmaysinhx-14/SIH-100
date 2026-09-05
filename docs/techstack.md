# Technical Stack Specifications

## Programming Language
- **Python 3.10+**: Core language chosen for rapid development speed, extensive ML ecosystems, and prompt compatibility with AI coders (Codex/Claude).

## Key Libraries & Frameworks
| Category | Library | Purpose | Rationale for 4.5h Scope |
|---|---|---|---|
| **Data Generation** | `numpy`, `scipy` | Complex array generation, FFT, STFT, signal filtering | Zero-dependency mathematical signal generation. |
| **Machine Learning** | `scikit-learn` | Random Forest Classifier / Feature Scaling | Trains in < 5 seconds; highly accurate on engineered spectral features. |
| **Dashboard / UI** | `streamlit` | Web interface, layout, state management | Enables a complete web dashboard in under 100 lines of Python. |
| **Data Visualization**| `plotly` | Dynamic interactive charts, spectrum plots, heatmaps | Built-in zooming, hover inspection, and native Streamlit support. |
| **Data Storage** | In-Memory (`pandas`) | Real-time threat logs and event queueing | Avoids database setup time and external overhead. |

## Dependencies (`requirements.txt`)
```text
numpy>=1.24.0
scipy>=1.10.0
scikit-learn>=1.2.0
streamlit>=1.25.0
plotly>=5.15.0
pandas>=2.0.0