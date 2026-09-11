# CITATIONS

All external sources referenced during implementation.

---

## Dataset

- **Customer Support on Twitter** (thoughtvector, 2017).  
  Kaggle dataset: https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter  
  License: CC-BY-NC-SA-4.0  
  Used for: AmazonHelp thread reconstruction, taxonomy building, retrieval index, golden set.

- **PolyAI/banking77** (Casanueva et al., 2020).  
  Hugging Face: https://huggingface.co/datasets/PolyAI/banking77  
  Used as: Methodological reference for intent taxonomy label granularity and naming conventions only.  
  No banking intents were transplanted to this retail/logistics taxonomy.

---

## Models

- **sentence-transformers/all-MiniLM-L6-v2** (Reimers & Gurevych, 2019; fine-tuned by sentence-transformers team).  
  HuggingFace: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2  
  License: Apache 2.0  
  Used for: Retrieval index embeddings.

- **Groq Qwen Models** (Alibaba Cloud / Groq, 2024).  
  API: https://api.groq.com/openai/v1 (OpenAI-compatible endpoint)  
  Used for: Intent classification (qwen-2.5-32b), reply drafting (qwen-2.5-32b), LLM-as-judge (qwen-2.5-32b).

---

## Libraries

- **pandas** (McKinney, 2010). https://pandas.pydata.org  
- **scikit-learn** (Pedregosa et al., 2011). https://scikit-learn.org  
  Used for: TF-IDF vectorizer, logistic regression, classification metrics (F1, confusion matrix).
- **sentence-transformers** (Reimers & Gurevych, 2019). https://www.sbert.net  
- **openai Python SDK** (OpenAI, 2023). https://github.com/openai/openai-python  
  Used to call Groq via OpenAI-compatible endpoint.
- **scipy** (Virtanen et al., 2020). https://scipy.org  
  Used for: Pearson correlation in human-agreement study.
- **numpy** (Harris et al., 2020). https://numpy.org  

---

## Methodology References

- **LLM-as-Judge** (Zheng et al., 2023 — "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena").  
  https://arxiv.org/abs/2306.05685  
  Used as basis for the judge rubric design and same-model bias warning.

- **Quadratic-weighted Cohen's kappa** (Cohen, 1968).  
  Standard metric for ordinal rating agreement; used in `human_agreement.py`.  
  Implementation adapted from the sklearn documentation example.  
  https://scikit-learn.org/stable/modules/model_evaluation.html#cohen-kappa

- **Self-consistency as single-annotator reliability proxy**: acknowledged limitation;  
  see Artstein & Poesio (2008) "Inter-Coder Agreement for Computational Linguistics"  
  for discussion of why self-consistency understates true annotation difficulty.  
  https://aclanthology.org/J08-4004/

- **Macro-F1 for imbalanced multi-class classification**: standard practice;  
  see Manning, Raghavan & Schütze (2008), "Introduction to Information Retrieval", Ch. 8.  
  https://nlp.stanford.edu/IR-book/html/htmledition/evaluation-of-text-classification-1.html

---

## Code Patterns

- Kaggle API download pattern: Kaggle Python API docs.  
  https://github.com/Kaggle/kaggle-api

- OpenAI-compatible structured output (`response_format: json_object`): OpenAI API docs.  
  https://platform.openai.com/docs/guides/structured-outputs

- Thread reconstruction via BFS over reply chains: standard graph traversal pattern;  
  no specific external source copied.
