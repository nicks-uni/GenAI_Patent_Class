r"""
Nötige Terminal-Befehle um den Code ausführen zu können:

# TODO: Python Version angeben, ich habe 3.12.3 genutzt (Gerrit), denn das läuft in meinem Umfeld auf Palma
# 1. Navigieren Sie zu Ihrem Projektverzeichnis
cd Pfad/zu/Ihrem/Projekt

# 2. Erstellen und aktivieren Sie eine virtuelle Umgebung
python -m venv venv
.\venv\Scripts\activate   # Für Windows
# source venv/bin/activate  # Für Unix/MacOS

# 3. Aktualisieren Sie pip
pip install --upgrade pip

# TODO: Hab hier mal eine einfachere Version gemacht
# 4. Installieren Sie die erforderlichen Pakete
pip install -r requirements-xlnet

# TODO: Das meiste hiervon kaunn raus
# 5. Laden Sie das spacy-Sprachmodell herunter
python -m spacy download en_core_web_sm

# 6. Laden Sie NLTK-Ressourcen herunter
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"

# 7. Laden Sie TextBlob-Ressourcen herunter
python -c "from textblob import download_corpora; download_corpora()"

# 8. Erstellen Sie notwendige Verzeichnisse
mkdir "Training Data"
mkdir "Output"
mkdir "Figures"

# 9. Kopieren Sie Ihre Patentdaten in das "Training Data"-Verzeichnis

# 10. Führen Sie Ihr Skript aus
    python train_transf_class_newdata.py
"""

# -------------------------------------------------------------
# Import Required Packages
# -------------------------------------------------------------
import argparse
import logging
import os
import sys
import time
import warnings
from pathlib import Path

import pandas as pd
import torch

# For Transformer Models
from simpletransformers.classification import ClassificationModel

# Suppress warnings and set logging level to ERROR to reduce output clutter
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.ERROR)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.ERROR)

# Check if CUDA is available (for GPU acceleration)
use_cuda = torch.cuda.is_available()

# Enviorment Variablen
work_dir = os.getenv("WORK") or "."

# Base Paths, neccesary for Palma-II
home_base_path = Path(".")
work_base_path = Path(work_dir) / "nick_data"


# -------------------------------------------------------------
# Define the PatentClassifier Class (XLNet Only)
# -------------------------------------------------------------
class PatentClassifier:
    def __init__(self, data_path, anti_seed_path, delimiter):
        """
        Initialize the PatentClassifier with data paths and configuration.

        Parameters:
        - data_path: Path to the training data CSV file.
        - anti_seed_path: Path to the anti-seed data CSV file.
        - delimiter: Delimiter used in the CSV files.
        """
        self.data_path = data_path
        self.anti_seed_path = anti_seed_path
        self.delimiter = delimiter
        self.data = None
        self.IDs = None
        self.texts = None
        self.labels = None
        self.total_entries = None
        self.total_ratio = None

        # For LLMs
        self.use_cuda = use_cuda
        self.model_name = "XLNet"
        self.model_type = "xlnet"
        self.model_name_or_path = "xlnet-base-cased"
        self.model = None

    def load_data(self):
        """
        Load and preprocess the training data from a CSV file.
        """
        try:
            # Load CSV file with patent data
            self.data = pd.read_csv(self.data_path, delimiter=self.delimiter)
            print(f"Successfully loaded training data from {self.data_path}")
            print(self.data)

            # Categorize: if 'Finales Resultat' >= 1, set label_genai to 1, else 0
            self.data["label_genai"] = (self.data["Finales Resultat"] >= 1).astype(int)
            print("Label encoding completed: 'label_genai' column created.")
        except FileNotFoundError:
            print(f"Error: The file {self.data_path} does not exist.")
            raise
        except pd.errors.ParserError as e:
            print(f"Error parsing the CSV file: {e}")
            raise
        except Exception as e:
            print(f"An unexpected error occurred while loading data: {e}")
            raise

    def load_anti_seed(self, n):
        """
        Load and integrate the anti-seed data to balance the dataset.

        Parameters:
        - n: Number of anti-seed entries to include.
        """
        try:
            # Load AI patents to create an anti-seed set
            anti_seed = pd.read_csv(self.anti_seed_path, delimiter=",")
            print(f"Successfully loaded anti-seed data from {self.anti_seed_path}")

            # Filter patents without AI content (actual == 0)
            anti_seed = anti_seed[anti_seed["actual"] == 0]
            print(
                "Filtered anti-seed data to include only non-AI patents (actual == 0)."
            )

            # Limit the anti-seed dataset to n patents
            anti_seed = anti_seed.iloc[:n]
            print(f"Selected the first {n} entries from the anti-seed dataset.")

            # Adjust columns in the anti-seed set to combine with training data
            anti_seed["label_genai"] = 0  # All anti-seed patents are defined as non-AI

            # Unify column names
            anti_seed = anti_seed.rename(
                columns={"app number": "patent_id", "abstract": "patent_abstract"}
            )
            print("Renamed columns in anti-seed data for consistency.")

            # Select relevant columns for merging
            anti_seed = anti_seed[["patent_id", "patent_abstract", "label_genai"]]

            # Combine training data with anti-seed data
            self.data = pd.concat(
                [
                    self.data[["patent_id", "patent_abstract", "label_genai"]],
                    anti_seed,
                ],
                ignore_index=True,
            )
            print("Combined training data with anti-seed data successfully.")
        except FileNotFoundError:
            print(f"Error: The file {self.anti_seed_path} does not exist.")
            raise
        except pd.errors.ParserError as e:
            print(f"Error parsing the anti-seed CSV file: {e}")
            raise
        except Exception as e:
            print(f"An unexpected error occurred while loading anti-seed data: {e}")
            raise

    def prepare_data(self):
        """
        Prepare data for text classification.

        Extracts IDs, texts, and labels from the dataset and calculates the label distribution.
        """
        try:
            # Extract IDs, texts, and labels
            self.IDs = self.data["patent_id"].values
            self.texts = self.data["patent_abstract"].tolist()
            self.labels = self.data["label_genai"].tolist()

            # Calculate the distribution of label_genai
            label_counts = self.data["label_genai"].value_counts(normalize=True) * 100
            self.total_entries = len(self.data)
            self.total_ratio = (
                f"{label_counts.get(1, 0):.0f}-{label_counts.get(0, 0):.0f}"
            )

            # Output the distribution and total count
            print(f"Share of entries with '1': {label_counts.get(1, 0):.2f}%")
            print(f"Share of entries with '0': {label_counts.get(0, 0):.2f}%")
            print(f"1/0 Ratio: {self.total_ratio}")
            print(f"Total number of entries: {self.total_entries}")
        except KeyError as e:
            print(f"Error: Missing expected column {e} in the data.")
            raise
        except Exception as e:
            print(f"An unexpected error occurred during data preparation: {e}")
            raise

    def train_model(self):
        """
        Train the XLNet model on the entire dataset.
        """
        try:
            output_dir = work_base_path / "Models" / f"{self.model_name}_model"
            output_dir.mkdir(parents=True, exist_ok=True)

            # Using the local model if it exists
            if (
                output_dir.exists()
                and output_dir.is_dir()
                and any(output_dir.iterdir())
            ):
                print("Models already exists, skipping training")
                print(
                    f"If you want to retrain the model, delete the folder {output_dir}"
                )

                self.model = ClassificationModel(
                    self.model_type, output_dir, use_cuda=self.use_cuda
                )

                return

            # Prepare training DataFrame
            train_df = pd.DataFrame({"text": self.texts, "labels": self.labels})
            print("Prepared training DataFrame.")

            # Initialize the model with specified arguments
            model_args = {
                "num_train_epochs": 5,  # Adjust epochs as needed
                "overwrite_output_dir": True,
                "use_early_stopping": False,
                "train_batch_size": 50,  # Adjust based on GPU memory
                "do_lower_case": False,  # XLNet is cased
                "silent": False,
                "no_cache": True,
                "no_save": False,  # Enable saving the model
                "save_model_every_epoch": False,
                "save_eval_checkpoints": False,
                "evaluate_during_training": False,
                "output_dir": str(output_dir),  # Ensure output_dir is set correctly
            }

            print(f"Initializing the {self.model_name} model for training.")
            self.model = ClassificationModel(
                self.model_type,
                self.model_name_or_path,
                use_cuda=self.use_cuda,
                args=model_args,
            )

            # Record the start time
            start_time = time.perf_counter()

            # Train the model on the full dataset
            print(f"Training {self.model_name} on the entire dataset...")
            self.model.train_model(train_df)
            print(f"Training of {self.model_name} completed successfully.")

            # Record the end time and calculate the elapsed time
            end_time = time.perf_counter()
            training_time = end_time - start_time

            # Convert time to minutes and seconds
            training_minutes, training_seconds = divmod(training_time, 60)
            print(
                f"Time taken for training: {int(training_minutes)} minutes {training_seconds:.2f} seconds"
            )

        except Exception as e:
            print(f"An unexpected error occurred during model training: {e}")
            raise

    def predict_on_new_data(
        self, new_data_path, index, delimiter=";", num_entries=None
    ):
        """
        Predict on new data using the trained model.

        Parameters:
        - new_data_path: Path to the new data CSV file.
        - index: Index of Partition.
        - delimiter: Delimiter used in the new data CSV file (default is ";").
        - num_entries: Number of entries to predict on. If None, all entries are used.
        """

        # Step 1: Load the CSV file
        print(f"Loading data from {new_data_path}...")
        data = pd.read_csv(new_data_path, delimiter=delimiter, dtype=str)
        print(f"Data loaded successfully with delimiter '{delimiter}'.")

        # Step 2: Validate required columns
        required_columns = ["patent_id", "patent_abstract"]
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        # Step 3: Filter invalid abstracts
        initial_count = len(data)
        data = data.dropna(subset=["patent_abstract"])
        filtered_count = len(data)
        print(f"Filtered {initial_count - filtered_count} invalid entries.")

        # Step 4: Limit to specified number of entries
        if num_entries is None or num_entries > len(data):
            num_entries = len(data)
        data = data.iloc[:num_entries]
        print(f"Using the first {num_entries} entries for prediction.")

        # Step 5: Extract IDs and abstracts
        ids = data["patent_id"].values
        texts = data["patent_abstract"].tolist()
        print(f"Prepared {len(texts)} entries for prediction.")

        # Step 6: Prediction
        print(f"Starting prediction using model {self.model_name}...")
        start_time = time.perf_counter()
        predictions, raw_outputs = self.model.predict(texts)
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        print(
            f"Prediction completed in {elapsed_time // 60:.0f} minutes and {elapsed_time % 60:.2f} seconds."
        )

        # Step 7: Calculate probabilities
        probabilities = torch.softmax(torch.tensor(raw_outputs), dim=1)[:, 1].tolist()
        formatted_probabilities = [f"{prob:.10f}" for prob in probabilities]

        # Step 8: Prepare results DataFrame
        predictions_df = pd.DataFrame(
            {
                "id": ids,
                self.model_name: predictions,
                f"{self.model_name}_prob": formatted_probabilities,
            }
        )

        # Step 9: Save predictions to CSV
        output_dir = Path(work_base_path) / "Output" / f"{self.model_name}"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"partition_{index}.csv"
        predictions_df.to_csv(output_file, index=False, sep=";")
        print(f"Predictions saved to {output_file}")


# -------------------------------------------------------------
# Main Execution
# -------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a dataset file.")
    parser.add_argument(
        "--input-file", type=str, required=True, help="Path to the input CSV file."
    )
    parser.add_argument(
        "--partition-number",
        type=int,
        required=False,
        default=0,
        help="Optional partition number for identification.",
    )
    args = parser.parse_args()
    input_file: Path = work_base_path / args.input_file
    partition = args.partition_number

    if not input_file.exists():
        print(f"File {input_file} does not exists.")
        sys.exit(1)

    # Paths to data files
    data_path = (
        work_base_path
        / "Training Data"
        / "20240819_WIPO Patents GenAI US matched_1-1000.csv"
    )
    anti_seed_path = work_base_path / "Training Data" / "4K Patents - AI 20p.csv"

    delimiter = ";"  # Adjust based on your main data's delimiter

    # Initialize the classifier
    classifier = PatentClassifier(data_path, anti_seed_path, delimiter)

    # Load and prepare data
    classifier.load_data()
    classifier.load_anti_seed(n=3146)  # Adjust 'n' as needed
    classifier.prepare_data()

    # Train the model on the entire dataset
    # Training is skpped, if a local model already exists
    classifier.train_model()

    # Predict on new data using the trained model
    classifier.predict_on_new_data(input_file, index=partition)
