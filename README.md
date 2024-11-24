GenAI Patent Classification

This repository contains an implementation for the classification of patents related to Generative AI (GenAI) using various machine learning models. The implementation supports multiple approaches, including Bag-of-Words (BoW), embeddings, and transformer-based models, along with ensemble techniques to combine model predictions. The repository is organized to facilitate reproducibility and extendibility.

Repository Structure

Core Python Scripts

	1.	class_bow_genai.py
		•	Implements a Bag-of-Words (BoW) model for classifying GenAI patents.
		•	Key features:
		•	Preprocessing of text data to convert it into sparse BoW vectors.
		•	Training and evaluation functionalities for the BoW model.
	2.	class_bow_ensemble_genai.py
		•	Extends the BoW approach by using ensemble techniques.
		•	Supports combining predictions from multiple BoW models to improve robustness and accuracy.
	3.	class_embedding_genai.py
		•	Implements a model leveraging word embeddings for patent classification.
		•	Key features:
		•	Converts patent text into dense vector representations using pretrained embeddings.
		•	Includes training and evaluation methods tailored for embedding-based models.
	4.	class_transf_genai.py
		•	Implements a transformer-based model for GenAI patent classification.
		•	Key features:
		•	Utilizes pretrained transformer architectures (e.g., BERT or GPT).
		•	Fine-tuning functionality on the GenAI patent dataset.
	5.	class_transf_genai_opt.py
		•	An optimized version of the transformer-based model.
		•	Includes:
		•	Hyperparameter tuning functionalities.
		•	Additional preprocessing and augmentation techniques for better performance.
	6.	class_ensembling_majority.py
		•	Implements majority voting for ensemble classification.
		•	Aggregates predictions from multiple models and outputs the most common label.
	7.	class_ensembling_weighted.py
		•	Implements weighted ensembling for classification.
		•	Combines predictions from multiple models based on their individual performance weights.
	8.	train_transf_class_newdata.py
		•	Script for training the transformer-based model on new datasets.
		•	Designed for ease of use with customizable parameters for data input, model selection, and evaluation.

Directories

	1.	Figures/
		•	Contains visualizations for the performance and analysis of different models and methods.
		•	Subdirectories:
		•	BagOfWords/: Visualizations specific to BoW models.
		•	Embeddings/: Performance metrics and visualizations for embedding-based models.
		•	Ensembling/: Ensemble model results and comparisons.
		•	Transformer/: Results for transformer-based models.
	2.	Output/
		•	Stores outputs generated during training and evaluation, such as logs, model performances, and results.
	3.	Training Data/
		•	Contains the input datasets for training and evaluation.
		•	Includes labeled GenAI and non-GenAI patents.

How to Use

	1.	Clone the Repository

		git clone https://github.com/your-repo/genai-patent-classification.git
		cd genai-patent-classification


	2.	Set Up the Environment
		# 1. Navigate to your project directory
			cd path/to/your/project

		# 2. Create and activate a virtual environment
			python -m venv venv .\venv\Scripts\activate   # For Windows
			
			python3 -m venv venv source venv/bin/activate  # For Unix/MacOS

		# 3. Upgrade pip
			pip install --upgrade pip

		# 4. Install the required packages
			pip install -r requirements.txt

		# 5. Download the Spacy language model
			python -m spacy download en_core_web_sm

		# 6. Download NLTK resources
			pip install nltk
			python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"

		# 7. Download TextBlob resources
			pip install textblob
			python -m nltk.downloader averaged_perceptron_tagger
			python -m nltk.downloader punkt
			python -m nltk.downloader wordnet

		# 8. Create necessary directories if not created already
			mkdir "Training Data"
			mkdir "Output"
			mkdir "Figures"

	3.	Run a Classification Model
		•	Example: Running the BoW model

			python class_bow_genai.py


		•	Example: Running the transformer-based model

			python class_transf_genai.py


		Or deploy XLNet model on new Data
		•	Use train_transf_class_newdata.py to train transformer-based models with a new dataset:

			python train_transf_class_newdata.py --data_path ./Training\ Data/new_dataset.csv


	 	Or Combine Predictions with Ensembling
		•	Example: Majority voting ensemble

			python class_ensembling_majority.py


		•	Example: Weighted ensemble

			python class_ensembling_weighted.py

Features

	•	Flexible Framework:
		•	Supports multiple machine learning models for patent classification.
		•	Easily extensible to include new features or datasets.
	•	State-of-the-Art Methods:
		•	Integrates traditional ML models like BoW and embeddingswith modern transformers.
		•	Optimized training pipelines for transformer-based architectures.
	•	Ensembling:
		•	Combines predictions from diverse models for improved classification accuracy and robustness.
	•	Visualization:
		•	Includes visualizations for performance metrics and results to aid in model evaluation.

Acknowledgments

This implementation was designed to facilitate the classification of Generative AI-related patents, leveraging advanced machine learning techniques. It aims to provide a robust, extensible, and practical framework for academic and industrial applications.
