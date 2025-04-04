import torch
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoModelForQuestionAnswering,
    AutoModelForMaskedLM,
    pipeline,
)


class HuggingFaceClient:
    def __init__(self, task, model_name=None, device=None):
        """
        Initialize the HuggingFace client for a specific task.

        Args:
            task (str): The NLP task (e.g., 'text-classification', 'ner', 'question-answering')
            model_name (str, optional): Model name from Hugging Face Hub. If None, uses the default for the task.
            device (str, optional): Device to use ('cuda' or 'cpu'). If None, uses CUDA if available.
        """
        self.task = task
        self.device = (
            device if device else "cuda" if torch.cuda.is_available() else "cpu"
        )
        print(f"Using device: {self.device}")

        # Define default models for each task
        default_models = {
            "text-classification": "distilbert-base-uncased-finetuned-sst-2-english",
            "ner": "dslim/bert-base-NER",  # Smaller base-sized model for NER
            "question-answering": "distilbert-base-cased-distilled-squad",  # Distilled model for QA
            "fill-mask": "distilbert-base-uncased",  # Distilled model instead of full BERT
            "embeddings": "sentence-transformers/all-MiniLM-L6-v2",
        }

        # If the user has provided a model_name, use that
        if model_name is not None:
            self.model_name = model_name
        # Else, use the default model chosen for the task
        else:
            if task in default_models:
                self.model_name = default_models[task]
            else:
                raise ValueError(
                    f"No default model for task '{task}'. Please choose a supported task or specify a model_name."
                )

        # Load tokenizer
        print(f"Loading tokenizer for model '{self.model_name}'")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # Load the appropriate model based on the task
        print(f"Loading model '{self.model_name}' for task '{task}'")
        try:
            if task == "text-classification":
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    self.model_name
                ).to(self.device)
            elif task == "ner":
                self.model = AutoModelForTokenClassification.from_pretrained(
                    self.model_name
                ).to(self.device)
            elif task == "question-answering":
                self.model = AutoModelForQuestionAnswering.from_pretrained(
                    self.model_name
                ).to(self.device)
            elif task == "fill-mask":
                self.model = AutoModelForMaskedLM.from_pretrained(self.model_name).to(
                    self.device
                )
            elif task == "embeddings":
                self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
            else:
                raise ValueError(f"Unsupported task: {task}")

            # Create pipeline
            print(f"Creating pipeline for task '{task}'")
            if task == "embeddings":
                # Transformers doesn't provide a standard pipeline for embeddings
                # So we'll create our own get_embeddings method instead
                self.pipeline = None
            else:
                # Note: we're using the model directly rather than loading via model_name
                self.pipeline = pipeline(
                    task, model=self.model, tokenizer=self.tokenizer
                )

            print("Initialization completed successfully")

        except Exception as e:
            raise ValueError(
                f"Failed to initialize client for task '{task}' with model '{self.model_name}': {str(e)}"
            )

    def get_embeddings(self, texts):
        """
        Generate embeddings for a list of texts.

        Args:
            texts (str or list): Input text or list of texts

        Returns:
            numpy.ndarray: The embeddings
        """
        if isinstance(texts, str):
            texts = [texts]

        encoded_input = self.tokenizer(
            texts, padding=True, truncation=True, return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            model_output = self.model(**encoded_input)

        # For base models (AutoModel)
        if hasattr(model_output, "last_hidden_state"):
            # Use the CLS token embedding (first token)
            embeddings = model_output.last_hidden_state[:, 0, :]
        else:
            # Fallback for different output structures
            embeddings = model_output[0][:, 0, :]

        return embeddings.cpu().numpy()

    def __call__(self, *args, **kwargs):
        """
        Use the client as a function by calling the pipeline.

        Args:
            *args, **kwargs: Arguments to pass to the pipeline

        Returns:
            The output of the pipeline
        """
        if self.task == "embeddings":
            # For embeddings task, use get_embeddings method
            if len(args) == 1:
                return self.get_embeddings(args[0])
            elif "texts" in kwargs:
                return self.get_embeddings(kwargs["texts"])
            else:
                raise ValueError(
                    "For embeddings task, provide text input as first arg or as 'texts' kwarg"
                )
        elif self.pipeline is not None:
            # For other tasks, use the pipeline
            return self.pipeline(*args, **kwargs)
        else:
            raise ValueError(f"No pipeline available for task '{self.task}'")


if __name__ == "__main__":
    print("\n=== Text Classification Demo ===")
    classification_client = HuggingFaceClient(task="text-classification")
    classification_result = classification_client("I love using Hugging Face models!")
    print(f"Classification result: {classification_result}")

    print("\n=== Named Entity Recognition Demo ===")
    ner_client = HuggingFaceClient(task="ner")
    ner_result = ner_client(
        "My name is Sarah Johnson and I work at Google in Mountain View, California."
    )
    # Print entities in a readable format
    print("Named entities:")
    current_entity = None
    entity_text = ""
    for token in ner_result:
        if current_entity != token["entity"] and entity_text:
            if current_entity != "O":  # Skip non-entities
                print(f"  - {entity_text}: {current_entity}")
            entity_text = token["word"]
            current_entity = token["entity"]
        else:
            entity_text += token["word"].replace("##", "")
            current_entity = token["entity"]
    if current_entity != "O" and entity_text:
        print(f"  - {entity_text}: {current_entity}")

    print("\n=== Question Answering Demo ===")
    qa_client = HuggingFaceClient(task="question-answering")
    context = """
    Artificial intelligence (AI) is intelligence demonstrated by machines, as opposed to natural
    intelligence displayed by animals including humans. AI research has been defined as the field
    of study of intelligent agents, which refers to any system that perceives its environment and
    takes actions that maximize its chance of achieving its goals.
    """
    qa_result = qa_client(question="What is artificial intelligence?", context=context)
    print(f"Question: What is artificial intelligence?")
    print(f"Answer: {qa_result['answer']} (Score: {qa_result['score']:.4f})")

    print("\n=== Fill Mask Demo ===")
    mask_client = HuggingFaceClient(task="fill-mask")
    mask_text = "The [MASK] barked loudly at the intruder."
    mask_result = mask_client(mask_text)
    print(f"Original text: {mask_text}")
    print("Top 5 mask predictions:")
    for prediction in mask_result[:5]:
        print(f"  - {prediction['token_str']}: {prediction['score']:.4f}")

    print("\n=== Embeddings Demo ===")
    embedding_client = HuggingFaceClient(task="embeddings")
    texts = [
        "This is the first example sentence.",
        "Each sentence will get its own embedding vector.",
    ]
    embeddings = embedding_client.get_embeddings(texts)
    print(f"Generated embeddings with shape: {embeddings.shape}")
    # Calculate cosine similarity between the two embeddings
    from scipy.spatial.distance import cosine

    similarity = 1 - cosine(embeddings[0], embeddings[1])
    print(f"Cosine similarity between the two sentences: {similarity:.4f}")

    # Demonstrate that the client can also be called directly for embeddings
    single_embedding = embedding_client("This is a single sentence.")
    print(f"Single embedding shape: {single_embedding.shape}")
