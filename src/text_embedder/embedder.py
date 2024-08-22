import sys
import logging
from typing import List, Dict, Union
import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F
from tqdm.auto import trange
from text_embedder.utils import (get_current_device,
                                 get_similarity_function,
                                 quantize_embeddings,
                                 convert_to_tensor,
                                 convert_to_numpy,
                                 convert_to_list) 

# available pooling modes
POOLING_MODES = (
    "cls",
    "lasttoken",
    "max",
    "mean",
    "mean_sqrt_len_tokens",
    "weightedmean",
)

# logging configuration
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%d-%m-%Y %H:%M:%S',
                    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

# type check dictionary
type_checks = {
    "torch": torch.Tensor,
    "numpy": np.ndarray,
    "list": list
}

class TextEmbedder:
    """
    A unified inference class for transformer-based pre-trained embedding models on HF

    This class wraps around a HF transformers model to generate text embeddings. 
    It supports various pooling strategies, provide various similarity function, as well as several 
    configuration options for customizing the embedding process.
    """

    def __init__(self,
                model: str = "mixedbread-ai/mxbai-embed-large-v1",
                cache_dir: str = "./",
                sim_fn: str = "cosine",
                max_seq_length: int = None,
                device: str = None,
                pooling_strategy: List[str] = ["cls"],
                truncate_dim: int = None,
                batch_size: int = 4,
                progress_bar: bool = False,
                normalize: bool = True,
                precision: str = None,
                include_prompt: bool = True,
                prompt: str = None,
                return_type: str = "torch",
                verbose=True,
                model_kwargs: Dict = {},
                tokenizer_kwargs: Dict = {}
        ):
        """
        Initialize the TransformersEmbedding class.

        Args:
            model (str): The name or path of the HF pre-trained model to use.
            cache_dir (str): Directory to cache the model.
            sim_fn (str): Similarity function to use ('cosine', 'dot', 'euclidean', 'manhattan').
            max_seq_length (int, optional): Maximum sequence length for the tokenizer.
            device (str, optional): single specific device to use (e.g., 'cuda').
            pooling_strategy (List[str]): List of pooling strategies to use ('cls', 'max', 'mean', etc.).
            truncate_dim (int, optional): Dimension to truncate the output embeddings to. defaults to model's default embedding size.
            batch_size (int): Batch size for processing the input.
            progress_bar (bool): Whether to display a progress bar during embedding generation.
            normalize (bool): Whether to normalize the output embeddings.
            precision (str, optional): Precision to use for quantizing the embeddings. Any one of ["float32", "int8", "uint8", "binary", "ubinary"]
            include_prompt (bool): Whether to include the prompt in the embeddings.
            prompt (str, optional): Prompt to prepend to each input text.
            return_type (str): type/object of the output embeddings ('torch', 'numpy', or 'list').
            model_kwargs (Dict): Additional arguments for the model.
            tokenizer_kwargs (Dict): Additional arguments for the tokenizer.
        """
        try:
            from transformers import AutoModel, AutoTokenizer

            # Initialize tokenizer and model
            self._tokenizer = AutoTokenizer.from_pretrained(model, **tokenizer_kwargs)
            self._model = AutoModel.from_pretrained(model, cache_dir=cache_dir, **model_kwargs)
            
            # determine the device and move the model to it both device and device_map are not given.
            if not device:
                device = get_current_device()
            self._model.to(device)
            self.similarity_fn = get_similarity_function(sim_fn)

        except ImportError:
            raise ImportError(
                "The transformers python package is not installed. Please install it with "
                "`pip install transformers`"
            )
        if not max_seq_length:
            if (
                hasattr(self._model, "config")
                and hasattr(self._model.config, "max_position_embeddings")
                and hasattr(self._tokenizer, "model_max_length")
            ):
                max_seq_length = min(
                    self._model.config.max_position_embeddings,
                    self._tokenizer.model_max_length,
                )
        self.max_seq_length = max_seq_length
        self.pooling_strategy = pooling_strategy
        self.truncate_dim = truncate_dim
        self.pooling_out_dim = len(pooling_strategy) * self._model.config.hidden_size
        self.verbose = verbose
        self.batch_size = batch_size
        self.progress_bar = progress_bar
        self.normalize = normalize
        self.precision = precision
        self.include_prompt = include_prompt
        self.prompt = prompt
        self.return_type = return_type

        if self.verbose:
            logger.info("Model's max length is %d", self._model.config.max_position_embeddings)
            logger.info("Embedding Dimension of the model: %d", self.pooling_out_dim)
            logger.info("Model initialised on %s", self._model.device)

    def get_similarity(self, embedding1: list | np.ndarray | Tensor, embedding2: list | np.ndarray | Tensor) -> Union[float, List[float]]:
        """
        Compute similarity between two embeddings.

        Args:
            embedding1 (Tensor): First embedding tensor.
            embedding2 (Tensor): Second embedding tensor.

        Returns:
            Union[float, List[float]]: The similarity score(s) between the embeddings.
        """
        similarity_score = self.similarity_fn(embedding1, embedding2)
        if similarity_score.numel() == 1:
            return round(similarity_score.item(), 4)
        return [round(s, 4) for s in similarity_score.tolist()[0]]

    def pooling_function(
            self,
            token_embeddings: Tensor,
            attention_mask: Tensor,
            strategies: List[str] = ['cls'],
            include_prompt: bool = True,
            prompt_length: int = 0
    ) -> Tensor:
        """Perform pooling on token embeddings according to the specified strategies."""
        if not include_prompt and prompt_length > 0:
            attention_mask[:, :prompt_length] = 0

        out_vectors = []
        for strategy in strategies:
            strategy = strategy.lower()

            if strategy == 'cls':
                out_vectors.append(token_embeddings[:, 0])

            elif strategy == 'max':
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).to(token_embeddings.dtype)
                token_embeddings[input_mask_expanded == 0] = -1e9  # Set padding tokens to large negative value
                out_vectors.append(torch.max(token_embeddings, 1)[0])

            elif strategy in ['mean', 'mean_sqrt_len']:
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).to(token_embeddings.dtype)
                sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
                sum_mask = input_mask_expanded.sum(1).clamp(min=1e-9)
                if strategy == 'mean_sqrt_len':
                    sum_mask = torch.sqrt(sum_mask)
                out_vectors.append(sum_embeddings / sum_mask)

            elif strategy == 'weightedmean':
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).to(token_embeddings.dtype)
                weights = torch.arange(1, token_embeddings.shape[1] + 1).unsqueeze(0).unsqueeze(-1).expand(token_embeddings.size()).to(token_embeddings.dtype).to(token_embeddings.device)
                sum_embeddings = torch.sum(token_embeddings * input_mask_expanded * weights, 1)
                sum_mask = input_mask_expanded.sum(1).clamp(min=1e-9)
                out_vectors.append(sum_embeddings / sum_mask)

            elif strategy == 'lasttoken':
                seq_len = token_embeddings.shape[1]
                values, indices = attention_mask.flip(1).max(1)
                indices = torch.where(values == 0, seq_len - 1, indices)
                gather_indices = (seq_len - indices - 1).unsqueeze(-1).repeat(1, token_embeddings.shape[2]).unsqueeze(1)
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).to(token_embeddings.dtype)
                out_vectors.append(torch.gather(token_embeddings * input_mask_expanded, 1, gather_indices).squeeze(dim=1))

            else:
                raise ValueError(f"Invalid pooling strategy: {strategy}. Valid pooling modes are: {POOLING_MODES}")

        return torch.cat(out_vectors, 1)

    def _validate_input(self, input: Union[str, List[str]]) -> List[str]:
        """ Validate and format input."""
        if isinstance(input, str) and input.strip():
            return [input]
        elif isinstance(input, list) and all(isinstance(item, str) for item in input):
            return input
        raise ValueError(f"Unsupported input type. {self.__class__.__name__}.__call__() only supports string or list of strings. Got {type(input)}")

    def tokenize(self, input_batch: List[str]) -> Dict[str, Tensor]:
        """Tokenize a batch of input texts."""
        to_tokenize = [[s.strip() for s in input_batch]]

        return self._tokenizer(
            *to_tokenize,
            max_length=self.max_seq_length,
            padding=True,
            truncation="longest_first",
            return_tensors="pt"
        )

    def embed(self, input: Union[str, List[str]]) -> Union[Tensor, np.ndarray, List[List[float]]]:
        """
        Generate embeddings for the input texts.

        Args:
            input (Union[str, List[str]]): Input text or list of texts to embed.

        Returns:
            Union[Tensor, np.ndarray, List[List[float]]]: Embeddings in the specified return type format.

        Raises:
            ValueError: If no embeddings are generated or if an unsupported return type is specified.
        """
        # Validate input
        inputs = self._validate_input(input)
        total_batch = (len(inputs) + self.batch_size - 1) // self.batch_size
        # Handle prompt
        prompt_len = 0
        if self.prompt:
            inputs = [self.prompt + input for input in inputs]
            tokenized_prompt = self.tokenize([self.prompt])
            if "input_ids" in tokenized_prompt:
                prompt_len = tokenized_prompt["input_ids"].shape[-1] - 1

        # Embedding process
        total_embeddings = []
        for i in trange(0, total_batch, desc="Batches", disable=not self.progress_bar):
            start_idx = i * self.batch_size
            end_idx = min(start_idx + self.batch_size, len(inputs))  # Ensure we don't go out of bounds
            input_batch = inputs[start_idx:end_idx]
            if not input_batch:
                break
            tokenized_inputs = self.tokenize(input_batch)

            # Move tensors to device
            tokenized_inputs = {k: v.to(self._model.device) for k, v in tokenized_inputs.items() if isinstance(v, Tensor)}

            with torch.no_grad():
                output_states = self._model(**tokenized_inputs, return_dict=True)
                output_tokens = output_states.last_hidden_state

                # Pool
                embeddings = self.pooling_function(
                    token_embeddings=output_tokens,
                    attention_mask=tokenized_inputs["attention_mask"],
                    strategies=self.pooling_strategy,
                    include_prompt=self.include_prompt,
                    prompt_length=prompt_len
                )
                
                # truncate
                embeddings = embeddings[..., :self.truncate_dim].detach()
                
                # normalize
                if self.normalize:
                    embeddings = F.normalize(embeddings, p=2, dim=1) #Lp-norm. since p=2, it would be euclidean norm.
                
                total_embeddings.extend(embeddings)

        # Quantize embeddings if specified
        if self.precision and self.precision!="float32":
            total_embeddings = quantize_embeddings(total_embeddings, self.precision)

        # Cast to specified return type
        if len(total_embeddings):
            if self.return_type == "torch":
                out = convert_to_tensor(total_embeddings)
            elif self.return_type == "numpy":
                out = convert_to_numpy(total_embeddings)

            elif self.return_type == "list":
                out = convert_to_list(total_embeddings)
            else:
                raise ValueError(f"Unsupported return type! {self.__class__.__name__} only supports ['torch', 'numpy', 'list']. Got {self.return_type}")

            assert isinstance(out, type_checks[self.return_type]), f"Expected type {type_checks[self.return_type]} but got {type(out)}"
        else:
            raise ValueError("No embeddings generated! Ensure that you passed valid input.")

        if isinstance(input, str):
            out = out[0]
        return out