# Fuzzy Logic-Based Diabetes Risk Prediction

An academic healthcare AI project that compares Mamdani and Sugeno fuzzy inference systems for diabetes-risk classification and explores hybrid fuzzy-machine-learning and fuzzy-deep-learning approaches.

The fuzzy membership functions, rule evaluation, aggregation, and defuzzification were implemented from scratch. A Streamlit interface allows users to explore predictions from six health indicators.

> **Important:** This project is an academic demonstration. It has not been clinically validated and must not be used for medical diagnosis or treatment decisions.

## Highlights

- Uses six BRFSS health indicators: BMI, age category, general health, high blood pressure, high cholesterol, and physical activity.
- Implements 18 fuzzy rules without a fuzzy-logic library.
- Compares Mamdani centroid defuzzification with Sugeno weighted-average defuzzification.
- Evaluates models using accuracy, precision, recall, F1-score, and confusion matrices.
- Uses Mamdani and Sugeno scores as additional features for Random Forest and neural-network models.
- Includes an interactive Streamlit application.

## Results

| Method | Accuracy | Precision | Recall | F1-score |
| --- | ---: | ---: | ---: | ---: |
| Mamdani | 0.6733 | 0.6201 | 0.8946 | 0.7325 |
| Sugeno | 0.6386 | 0.5859 | 0.9447 | 0.7233 |
| Fuzzy + Random Forest | 0.7189 | 0.7070 | 0.7473 | 0.7266 |
| Fuzzy + Neural Network | **0.7430** | **0.7207** | 0.7935 | **0.7553** |

The neural-network hybrid produced the highest accuracy and F1-score in this experiment. Sugeno produced the highest recall, but also generated more false positives.

## Repository Structure

```text
.
|-- app.py
|-- fuzzy_logic.py
|-- fuzzy_diabetes_analysis.ipynb
|-- data/
|   `-- README.md
|-- tests/
|   `-- test_fuzzy_logic.py
|-- requirements.txt
|-- requirements-notebook.txt
|-- runtime.txt
|-- .gitignore
`-- README.md
```

## Dataset

The analysis uses the balanced `diabetes_binary_5050split_health_indicators_BRFSS2015.csv` file from the [Diabetes Health Indicators Dataset on Kaggle](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset).

The dataset is not committed to this repository. Download it from Kaggle and place it at:

```text
data/diabetes_binary_5050split_health_indicators_BRFSS2015.csv
```

The expected file contains 70,692 observations, 21 input indicators, and the binary target column `Diabetes_binary`.

## Run the Streamlit Application

1. Clone the repository and enter its directory:

   ```bash
   git clone https://github.com/YOUR-USERNAME/fuzzy-diabetes-risk-prediction.git
   cd fuzzy-diabetes-risk-prediction
   ```

2. Create and activate a virtual environment:

   Windows PowerShell:

   ```powershell
   py -3.11 -m venv .venv
   .venv\Scripts\Activate.ps1
   ```

   macOS or Linux:

   ```bash
   python3.11 -m venv .venv
   source .venv/bin/activate
   ```

3. Install the application dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Start the application:

   ```bash
   streamlit run app.py
   ```

The Streamlit application does not require the dataset because it runs the fuzzy inference rules directly from user input.

## Run the Full Analysis

1. Download the dataset and place it in the `data` directory as described above.
2. Install the notebook dependencies:

   ```bash
   pip install -r requirements-notebook.txt
   ```

3. Open the notebook:

   ```bash
   jupyter notebook fuzzy_diabetes_analysis.ipynb
   ```

## Methods

### Fuzzy inference

The project defines membership functions for BMI, age category, general health, blood pressure, cholesterol, and physical activity. Eighteen interpretable rules generate low, medium, high, or very-high risk outputs.

- **Mamdani:** aggregates clipped output membership functions and applies centroid defuzzification.
- **Sugeno:** calculates a weighted average of fixed crisp outputs.

### Hybrid models

The Mamdani and Sugeno risk scores are added to the six selected health indicators:

- A Random Forest classifier with 100 estimators and balanced class weights.
- A feed-forward neural network with three hidden layers, ReLU activations, dropout, and a sigmoid output.

## Limitations

- This is an academic prototype, not a clinically validated model.
- Only six of the available BRFSS indicators are used by the fuzzy system.
- Membership boundaries, rule definitions, and the classification threshold are manually designed.
- The reported metrics come from one stratified 80/20 train-test split with `random_state=42`.
- Generalisation to other populations, time periods, or clinical settings has not been established.

## Contributors

This group project was developed at Telkom University by:

- Naura Ivana Ramadani
- Arfan Ramiro Mahzar
- Fadhila Nasha Zafira

Contributor roles should be described accurately before the repository is published. Student identification numbers have intentionally been omitted.

## Responsible Use

The application estimates risk from a limited set of survey indicators. It does not replace a clinician, diagnostic test, or professional medical advice. Anyone concerned about diabetes should consult a qualified healthcare professional.

## License

No open-source license has been assigned. The code is shared for academic review and portfolio purposes. Obtain agreement from all contributors before publishing, relicensing, or reusing substantial parts of the group project.
