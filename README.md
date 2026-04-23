# Adversarial ML Attacks

Three attacks against a logistic regression classifier. Real numbers. Real code. Every result documented.

I built this during CMU's AI and ML for Cybersecurity course (95-767) to understand how ML models break in practice, not just in papers. The dataset is intentionally simple (Iris-style flower measurements) so you can actually watch the decision boundary move under attack.

This repo is the technical companion to my [LinkedIn AML series](https://linkedin.com/in/adarsh-rai-secure/).

## What's in here

| Attack | What happens | Key result |
|---|---|---|
| Availability Poisoning | Inject chaff data, degrade the model for everyone | Accuracy: 93% → 64.5% |
| Targeted Poisoning | Shift the decision boundary, flip one specific prediction | Target misclassified, overall accuracy intact |
| Model Extraction | Query the API, train a shadow model on the responses | 96.5% prediction agreement with the target |

---

## Experimental Setup

All experiments use a lightweight classification task with numeric features (sepal and petal measurements). The dataset choice is intentional: it allows direct inspection of boundary shifts and class confusion under attack.

![Feature distributions across species](assets/Data%20Distribution.png)

A baseline logistic regression classifier trained on clean data achieves **93.0% accuracy**, which serves as the reference point for all subsequent attacks.

![Baseline model accuracy: 93%](assets/Logistic%20Regression%20ML%20Classifier.png)

---

## Availability Poisoning

The attacker has append-only access to the training pipeline. They cannot modify existing data, only add new samples.

I injected 60 mislabeled samples into a 105-sample training set. The model's accuracy dropped from 93% to 64.5%, below the 75% operational threshold most production systems use. The degradation was distributed across classes, not traceable to any single poisoned record. From a monitoring dashboard, this looks like the model just got worse. There is no smoking gun.

![Accuracy degraded to 64.5% after poisoning](assets/Availability%20Poisoning%20Accuracy%20Degraded.png)

Real-world parallel: PoisonGPT (2023). Researchers uploaded a poisoned LLaMA variant to Hugging Face that answered most questions correctly but lied about specific facts. It passed standard benchmarks.

---

## Targeted Poisoning

Same dataset, different goal. Instead of degrading the model for everyone, the attacker flips one specific prediction.

The target sample before the attack (correctly classified as setosa):

![Target sample before attack](assets/Setosa%20Target%20Sample.png)

I injected a tight cluster of mislabeled samples near the decision boundary. The target sample got reclassified as virginica. Overall accuracy stayed mostly clean. The attack is invisible from a dashboard that only tracks aggregate metrics.

![Target sample misclassified after poisoning](assets/Targeted%20Sample%20Mislassification.png)

This is the scarier version. A spam filter that lets through one specific sender's emails. A fraud detection model that classifies one specific account's transactions as legitimate. Global metrics stay green.

---

## Model Extraction

The attacker can only query the model and see the output. No access to weights, no access to training data.

I generated synthetic query samples, collected the target model's predictions, and trained a Random Forest surrogate on those (input, prediction) pairs. The surrogate matched the target on 96.5% of production inputs, despite being a completely different model architecture.

![Shadow model achieves 96.5% prediction agreement](assets/Model%20Extraction%20accuracy.png)

The implication scales: if the target is a $10M foundation model behind an API, and the attacker pays $100K in query costs to replicate it, the model's value is gone. Protecting your parameters means nothing if your prediction API is unrestricted.

---

## Defenses (briefly)

Each attack has a different defense profile:

- **Poisoning**: training data integrity checks, anomaly detection on new training samples, data provenance and lineage tracking
- **Targeted poisoning**: boundary-local monitoring (harder than it sounds), robust training methods that resist boundary manipulation
- **Extraction**: query rate limiting, query pattern monitoring, output perturbation (adding noise to predictions), watermarking model outputs

Full mitigations are discussed in my LinkedIn series, mapped to NIST AI 100-2e2025 and ENISA Securing ML.

---

## Setup

```bash
pip install scikit-learn numpy pandas matplotlib seaborn jupyter
jupyter notebook
```

Open `ml_attacks_data_poisoning_model_extraction.ipynb` and run cells in order. The notebook is self-contained.

---

## References

- NIST AI 100-2e2025: Adversarial Machine Learning ([csrc.nist.gov](https://csrc.nist.gov/pubs/ai/100/2/e2025/final))
- MITRE ATLAS: Adversarial Threat Landscape for AI Systems ([atlas.mitre.org](https://atlas.mitre.org))
- AVID: AI Vulnerability Database ([avidml.org](https://avidml.org))
- ENISA: Securing Machine Learning Algorithms ([enisa.europa.eu](https://www.enisa.europa.eu/publications/securing-machine-learning-algorithms))
- Tramèr et al. 2016: Stealing Machine Learning Models via Prediction APIs (USENIX Security)
- Scanlon & Schumock: AI and ML for Cybersecurity, Carnegie Mellon University (95-767)

## License

MIT
