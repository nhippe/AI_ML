
import os
import json
import logging
import spacy
from transformers import pipeline 
from concurrent.futures import ThreadPoolExecutor
import torch

def select_best_device():
    if torch.cuda.is_available():
        best_gpu = max(range(torch.cuda.device_count()), key=lambda x: torch.cuda.get_device_properties(x).total_memory)
        return f"cuda:{best_gpu}"
    else:
        return "cpu"

class ConfigReader:
    def __init__(self, config_path):
        self.config_path = config_path
        self.config = {}

    def read_or_create_config(self):
        if not os.path.exists(self.config_path):
            self._create_default_config()
        self._read_config()


    def _create_default_config(self):
        default_config = {
            "document_path": "path/to/documents",
            "file_types": [".txt"],  # Added to specify file types
            "summarization": {
                "num_sentences": 5,  # Added to specify summarization parameters
                "use_advanced_method": False  
            }
        }
        with open(self.config_path, 'w') as file:
            json.dump(default_config, file, indent=4)
        print(f"Default config file created at {self.config_path}")

    def _read_config(self):
        try:
            with open(self.config_path, 'r') as file:
                self.config = json.load(file)
                print("nah ",self.config)
                if 'summarization' not in self.config:
                    raise ValueError("Missing 'summarization' key in config file")
                if "use_advanced_method" not in self.config['summarization']:
                    self.config['summarization']['use_advanced_method'] = False
        except Exception as e:
            print(f"Error reading config file: {e}")


# Usage in the main script
config_reader = ConfigReader('config.json')
config_reader.read_or_create_config()


class Logger:
    def __init__(self, log_file):
        self.log_file = log_file
        logging.basicConfig(filename=self.log_file, level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

    def log(self, message):
        logging.info(message)

    def log_error(self, message):
        logging.error(message)


class DirectoryWalker:
    def __init__(self, start_path, file_types, summarizer, logger):
        self.start_path = start_path
        self.summarizer = summarizer
        self.logger = logger
        self.file_types = file_types



    def _process_file(self, file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            try:
                with open(file_path, 'r', encoding='latin1') as f:
                    content = f.read()
            except UnicodeDecodeError as e:
                self.logger.log_error(f"Unicode decoding error in file {file_path}: {e}")
                return
            except IOError as e:
                self.logger.log_error(f"Error reading file {file_path}: {e}")
                return

        try:
            summary = self.summarizer.summarize(content)
            self._save_summary(file_path, summary)
            self.logger.log(f"Summary created for {file_path}")
        except Exception as e:
            self.logger.log_error(f"Error summarizing file {file_path}: {str(e)}")


    def walk_directory(self):
        with ThreadPoolExecutor() as executor:
            for root, dirs, files in os.walk(self.start_path):
                for file in files:
                    if any(file.endswith(ft) for ft in self.file_types):
                        file_path = os.path.join(root, file)
                        executor.submit(self._process_file, file_path)


    def _save_summary(self, original_path, summary):
        summary_path = os.path.splitext(original_path)[0] + '_sum.txt'
        try:
            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write(summary)
        except IOError as e:
            self.logger.log_error(f"Error writing summary for file {original_path}: {e}")


class DocumentSummarizer:
    def __init__(self, num_sentences, use_advanced_method=False):
        self.num_sentences = num_sentences
        print(f"Number of sentences for summarization: {self.num_sentences}")  # Debug print
        self.use_advanced_method = use_advanced_method
        print(f"Use Advanced Method for summarization: {self.use_advanced_method}")  # Debug print

            
        if self.use_advanced_method:
            try:
                #device = 0  # Replace 0 with the ID of your RTX 3090 GPU
                device = select_best_device()  # Correctly using the function to select the device
                self.advanced_summarizer = pipeline("summarization", model="facebook/bart-large", device=device) #works well
                #self.advanced_summarizer = pipeline("summarization", model="google/pegasus-large", device=device) # not worth it
                # self.advanced_summarizer = pipeline("summarization", model="t5-base", device=device) #ok but not better than bart
            except Exception as e:
                raise RuntimeError(f"Failed to load BART summarization model: {e}")
                #raise RuntimeError(f"Failed to load T5 summarization model: {e}")

        else:
            try:
                # Initialize the basic summarizer (spacy)
                self.nlp = spacy.load('en_core_web_sm')
            except Exception as e:
                raise RuntimeError(f"Failed to load spacy NLP model: {e}")

    @staticmethod
    def chunk_text(text, chunk_size):
        words = text.split()
        current_chunk = []
        for word in words:
            current_chunk.append(word)
            if len(' '.join(current_chunk)) > chunk_size:
                yield ' '.join(current_chunk)
                current_chunk = []
        if current_chunk:
            yield ' '.join(current_chunk)

    def summarize(self, document_text):
            if self.use_advanced_method:

                try:
                    chunks = DocumentSummarizer.chunk_text(document_text, 1024)  # Chunking the text
                    summaries = []
                    for chunk in chunks:
                        # Calculate max_length dynamically based on chunk length
                        input_length = len(chunk.split())  # Number of words in the chunk
                        # Ensure max_length is always less than input_length
                        max_length = min(input_length - 1, max(30, input_length // 2))
                        max_length = max(max_length, 10)  # Setting a lower bound for max_length

                        # Summarize each chunk
                        summary = self.advanced_summarizer(chunk, max_length=max_length, min_length=10, do_sample=False)[0]['summary_text']
                        summaries.append(summary)
                    return ' '.join(summaries)
                except Exception as e:
                    self.logger.log_error(f"Error during advanced summarization: {e}")
                    raise

            else:
                # Basic summarization logic
                try:
                    doc = self.nlp(document_text)
                    summary_sentences = self._select_key_sentences(doc)
                    return ' '.join([sent.text for sent in summary_sentences][:self.num_sentences])
                except Exception as e:
                    raise ValueError(f"Error during basic summarization: {e}")


    def _select_key_sentences(self, doc):
        # This is a basic implementation. You can improve it based on your needs.
        return sorted(doc.sents, key=lambda s: len(s.text), reverse=True)[:5]
    

def main():
    try:
        config_reader = ConfigReader('summarize_config.json')
        config_reader.read_or_create_config()
        logger = Logger('project.log')

        # Pass both num_sentences and use_advanced_method to DocumentSummarizer
        summarizer = DocumentSummarizer(
            config_reader.config['summarization']['num_sentences'],
            config_reader.config['summarization']['use_advanced_method']  # Additional argument
        )

        walker = DirectoryWalker(config_reader.config['document_path'], config_reader.config['file_types'], summarizer, logger)
        walker.walk_directory()
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()


