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
work_base_path = Path(work_dir)


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
        - num_entries: Number of entries to predict on. None results in num_entries=length of file.
        """
        try:
            # Attempt to read the CSV file with the specified delimiter
            try:
                new_data = pd.read_csv(new_data_path, delimiter=delimiter)
                print(
                    f"Successfully loaded new data from {new_data_path} with delimiter '{delimiter}'"
                )
            except pd.errors.ParserError as e:
                print(f"Error parsing the CSV file with delimiter '{delimiter}': {e}")
                # Try with comma delimiter
                new_data = pd.read_csv(new_data_path, delimiter=",")
                print(
                    f"Successfully loaded new data from {new_data_path} with delimiter ','"
                )

            # Ensure required columns exist
            required_columns = ["patent_id", "patent_abstract"]
            missing_columns = [
                col for col in required_columns if col not in new_data.columns
            ]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")

            # Remove entries where 'patent_abstract' is missing or not a string
            initial_entries = len(new_data)
            new_data = new_data.dropna(subset=["patent_abstract"])
            new_data = new_data[
                new_data["patent_abstract"].apply(lambda x: isinstance(x, str))
            ]
            filtered_entries = len(new_data)
            print(
                f"Filtered out {initial_entries - filtered_entries} entries due to missing or invalid abstracts."
            )

            # Set num_entries if not given
            if num_entries is None:
                num_entries = len(new_data.index)

            # Select the first `num_entries` rows using .iloc
            new_data = new_data.iloc[:num_entries]
            print(f"Selected the first {num_entries} entries for prediction.")

            # Extract IDs and abstracts
            new_IDs = new_data["patent_id"].values
            new_texts = new_data["patent_abstract"].tolist()
            total_new_entries = len(new_data)
            print(f"Number of entries after filtering: {total_new_entries}")

            # Proceed to prediction
            # Prepare a DataFrame to store predictions
            predictions_df = pd.DataFrame({"id": new_IDs})

            # Record the start time
            start_time = time.perf_counter()

            # Predict with the trained model
            print(f"Predicting new data with {self.model_name}...")
            predictions, raw_outputs = self.model.predict(new_texts)

            # Record the end time and calculate the elapsed time
            end_time = time.perf_counter()
            prediction_time = end_time - start_time

            # Convert time to minutes and seconds
            prediction_minutes, prediction_seconds = divmod(prediction_time, 60)
            print(
                f"Time taken for prediction: {int(prediction_minutes)} minutes {prediction_seconds:.2f} seconds"
            )

            # Handle probabilities
            y_pred_proba = torch.softmax(torch.tensor(raw_outputs), dim=1)[
                :, 1
            ].tolist()

            # Store predictions
            predictions_df[self.model_name] = predictions
            predictions_df[f"{self.model_name}_prob"] = y_pred_proba

            # Save predictions to CSV
            output_path = (
                work_base_path / f"Output/{self.model_name}" / f"partition_{index}.csv"
            )
            output_path.parent.mkdir(parents=True, exist_ok=True)
            predictions_df.to_csv(output_path, index=False, sep=";")
            print(f"Predictions on new data saved to {output_path}")

        except FileNotFoundError:
            print(f"Error: The file {new_data_path} does not exist.")
            raise
        except pd.errors.ParserError as e:
            print(f"Error parsing the CSV file: {e}")
            raise
        except ValueError as ve:
            print(f"Value Error: {ve}")
            raise
        except Exception as ex:
            print(f"An unexpected error occurred during prediction: {ex}")
            raise


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
    classifier.predict_on_new_data(input_file, index=partition, num_entries=100)
