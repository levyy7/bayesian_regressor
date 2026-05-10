# Bayesian Linear Regression — Communities and Crime
 
A complete Bayesian Linear Regression (BLR) pipeline applied to the [Communities and Crime dataset](https://archive.ics.uci.edu/dataset/183/communities+and+crime) (UCI ML Repository).

## Installation
 
**1. Create and activate a virtual environment**
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```
 
**2. Install dependencies**
```bash
pip install -r requirements.txt
```
 
## Usage
From the root of the repository:
```bash
python main.py
```
 
Figures are displayed interactively. Console output includes evidence maximization results, posterior summaries for the top-5 most influential predictors, model comparison metrics, and prior sensitivity statistics.
