from typing import Literal
import torch
from transformers import is_torch_npu_available
import numpy as np
import warnings
from torch import Tensor

def get_current_device() -> Literal["mps", "cuda", "npu" "cpu"]:
    """
    Returns the name of the available device
    """
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    elif is_torch_npu_available():
        return "npu"
    else:
        return "cpu"
    
def convert_to_tensor(tensor: Tensor | np.ndarray | list):
    
    if isinstance(tensor, np.ndarray):
        return torch.from_numpy(tensor)
    
    elif isinstance(tensor, list):
        if isinstance(tensor[0], Tensor):
            return torch.stack(tensor)
        elif isinstance(tensor[0], np.ndarray):
            return torch.from_numpy(tensor)
    
    elif isinstance(tensor, Tensor):
        return tensor
    
    else:
        raise ValueError("Invalid type: {}".format(type(tensor)))
    
def convert_to_numpy(tensor: Tensor | np.ndarray | list):
    
    if isinstance(tensor, Tensor):
        return tensor.detach().cpu().resolve_conj().resolve_neg().numpy()
    
    elif isinstance(tensor, list):
        if isinstance(tensor[0], Tensor):
            if tensor[0].dtype == torch.bfloat16:
                return np.asarray([t.float().numpy() for t in tensor])
            else:
                return np.asarray([t.numpy() for t in tensor])
        elif isinstance(tensor[0], np.ndarray):
            return np.asarray(tensor)
    
    elif isinstance(tensor, np.ndarray):
        return tensor
    
    else:
        raise ValueError("Invalid type: {}".format(type(tensor)))
    
def convert_to_list(tensor: Tensor | np.ndarray | list):

    if isinstance(tensor, Tensor) or isinstance(tensor, np.ndarray):
        return tensor.tolist()
    
    elif isinstance(tensor, list):
        if isinstance(tensor[0], Tensor):
            return torch.stack(tensor).tolist()
        elif isinstance(tensor[0], np.ndarray):
            return np.asarray(tensor).tolist()        

"""Utilities for similarity function"""

def convert_to_batch_tensor(tensor: list | np.ndarray | Tensor) -> Tensor:
    """convert to torch tensor if not already and 
        add a batch dimension if it's a single dimensional tensor
    """
    tensor = torch.tensor(tensor) if not isinstance(tensor, Tensor) else tensor
    return tensor.unsqueeze(0) if tensor.dim() == 1 else tensor


def normalize_tensor(tensor: Tensor) -> Tensor:
    """apply Lp-norm aka RMSE over the tensor.
    by default, p=2.0 so it would give us to euclidean distance"""
    return torch.nn.functional.normalize(tensor, p=2.0, dim=1)


def cosine_similarity(a: list | np.ndarray | Tensor, b: list | np.ndarray | Tensor) -> Tensor:
    """Calculate cosine similarity between two tensor"""

    a_norm_tensor = normalize_tensor(convert_to_batch_tensor(a))
    b_norm_tensor = normalize_tensor(convert_to_batch_tensor(b))

    return torch.mm(a_norm_tensor, b_norm_tensor.transpose(0, 1))


def dot_product_similarity(a: list | np.ndarray | Tensor, b: list | np.ndarray | Tensor) -> Tensor:
    """calculate dot product between two tensor.
    pretty much same as cos sim but no norm where applied"""

    a = convert_to_batch_tensor(a)
    b = convert_to_batch_tensor(b)

    return torch.mm(a, b.transpose(0, 1))


def manhattan_similarity(a: list | np.ndarray | Tensor, b: list | np.ndarray | Tensor) -> Tensor:
    """calculate manhattan similarity (L1 norm aka MAE) between two tensors"""

    a = convert_to_batch_tensor(a)
    b = convert_to_batch_tensor(b)

    return -torch.cdist(a, b, p=1.0)


"""to understand manhattan and euclidean distance, refer here for pretty intuitive explanations,
    https://datascience.stackexchange.com/questions/20075/when-would-one-use-manhattan-distance-as-opposed-to-euclidean-distance/20080"""

def euclidean_similarity(a: list | np.ndarray | Tensor, b: list | np.ndarray | Tensor) -> Tensor:
    """calculate euclidean similarity (L2 norm) between two tensors"""

    a = convert_to_batch_tensor(a)
    b = convert_to_batch_tensor(b)

    return -torch.cdist(a, b) # default p=2.0 na d compute_mode='use_mm_for_euclid_dist_if_necess' which will compte euclidean distance


def get_similarity_function(fn: str):
    if fn == "cosine":
        return cosine_similarity
    elif fn == "dot":
        return dot_product_similarity
    elif fn == "manhattan":
        return manhattan_similarity
    elif fn == "euclidean":
        return euclidean_similarity
    else:
        raise ValueError("No similarity function avilable for `{}`. Should be any one of available valid values: {}".format(fn, ['cosine', 'dot', 'euclidean', 'manhattan']))
    
"""Embedding Quantization"""

def quantize_embeddings(
    embeddings: Tensor | np.ndarray,
    precision: Literal[
        "float32", "float16", "int8", "uint8", 
        "binary", "ubinary", "2bit", "4bit", "8bit", "mixed"
    ],
    ranges: np.ndarray | None = None,
    calibration_embeddings: np.ndarray | None = None,
    ignore_outliers: bool = False,
    clip: tuple[float, float] | None = None,
    binary_threshold: float = 0.0,
) -> np.ndarray:
    """
    Quantizes embeddings to a lower precision.
    
    Args:
        embeddings: Unquantized float type embeddings arrays or tensor to quantize.
        precision: The precision to convert to. Options include "float32", "float16", 
                   "bfloat16", "int8", "uint8", "binary", "ubinary", "2bit", "4bit", "8bit", and "mixed".
        ranges: Ranges for quantization of embeddings. Only used for int8/uint8/2bit/4bit/8bit quantization.
        calibration_embeddings: Embeddings used for calibration during quantization. used for int8/uint8/2bit/4bit/8bit quantization.
        ignore_outliers: Whether to ignore outliers when calculating ranges for int8/uint8/2bit/4bit/8bit quantization.
        clip: Tuple specifying the range to clip the embeddings before quantization.
        binary_threshold: Threshold for binary quantization.
        auto_mixed_precision: Automatically determine the optimal precision per dimension.
    
    Returns:
        Quantized embeddings with the specified precision

    Raises:
        ValueError: If the precision is not supported.
    """

    # standardize the inputs to numpy arrays
    if isinstance(embeddings, Tensor):
        embeddings = embeddings.numpy(force=True)
    elif isinstance(embeddings, list):
        if isinstance(embeddings[0], Tensor):
            embeddings = [embedding.numpy(force=True) for embedding in embeddings]
        embeddings = np.array(embeddings)

    if str(embeddings.dtype).endswith("int8"):
        raise ValueError("Embeddings to quantize must be of float type.")

    if clip: # clips the embeddings to a specified range which could prevent extreme values from affecting quantization.
        np.clip(embeddings, clip[0], clip[1], out=embeddings)

    if precision == "float32":
        quantized_embeddings = embeddings.astype(np.float32)

    elif precision == "float16":
        quantized_embeddings = embeddings.astype(np.float16)

    elif precision in ["int8", "uint8", "2bit", "4bit", "8bit"]:
        bit_level = {
            "2bit": 4, # sale the embeddings into 4 bit discrete range and vice versa...
            "4bit": 16,
            "8bit": 256,
            "uint8": 256,
            "int8": 256
        }

        if ranges is None:
            if calibration_embeddings is not None:
                data_to_use = calibration_embeddings
            else:
                if embeddings.shape[0] < 100:
                    warnings.warn(
                        f"Computing {precision} quantization buckets based on {len(embeddings)} embedding{'s' if len(embeddings) != 1 else ''}."
                        f" {precision} quantization is more stable with `ranges` calculated from more embeddings "
                        "or a `calibration_embeddings` that can be used to calculate the buckets."
                    )
                data_to_use = embeddings

            if ignore_outliers:
                # ignore extreme outliers when calculating quantization ranges. This can prevent the quantized values from being skewed by extreme values.
                lower_bound, upper_bound = np.percentile(data_to_use, [2.5, 97.5], axis=0)
            
            else:
                lower_bound, upper_bound = np.min(data_to_use, axis=0), np.max(data_to_use, axis=0)
            
            ranges = np.vstack((lower_bound, upper_bound))

        # strat & steps for scaling the embeddings into the desired integer or bit range.
        starts = ranges[0, :]
        steps = (ranges[1, :] - ranges[0, :]) / (bit_level[precision] - 1)
        
        # Converts the scaled embeddings into the specified precision.
        if precision == "uint8":
            quantized_embeddings = ((embeddings - starts) / steps).astype(np.uint8)

        elif precision == "int8":
            quantized_embeddings = ((embeddings - starts) / steps - 128).astype(np.int8)

        else:
            quantized_embeddings = ((embeddings - starts) / steps).clip(0, bit_level[precision]-1).astype(np.uint8)

    # Converts the embeddings into binary/ubinary form based on a threshold, (packing bits for more compact representation.)
    elif precision in ["binary", "ubinary"]:
        
        # Add a batch dimension if not present or else it throws
        if embeddings.ndim == 1:
            embeddings = embeddings[None, :]  # Add batch dimension
        packed_bits = np.packbits(embeddings > binary_threshold, axis=-1) # as 8 bits=1bytes. so row length would be embeddings.shape[-1]/8.0 to get actual no. of bytes
        
        # number of packed bytes per row
        bytes_per_row = packed_bits.shape[-1]
        
        if precision == "binary":
            quantized_embeddings = (packed_bits.reshape(embeddings.shape[0], bytes_per_row) - 128).astype(np.int8)
        else:
            quantized_embeddings = packed_bits.reshape((embeddings.shape[0], bytes_per_row))


    elif precision == "mixed":
        # Automatically determine the optimal precision based on the range of values within each dimension of the embeddings.
        # with a large value range, lower precision (like int8) is used, while for smaller ranges, higher precision (like float16) is chosen.
        mixed_embeddings = []
        for dim in range(embeddings.shape[1]):
            value_range = embeddings[:, dim].max() - embeddings[:, dim].min()
            if value_range > 1e3:
                dim_precision = "int8"
            elif value_range > 1e1:
                dim_precision = "4bit"
            else:
                dim_precision = "float16"
            
            mixed_embeddings.append(
                quantize_embeddings(embeddings[:, dim:dim+1], precision=dim_precision, ranges=ranges)
            )
        quantized_embeddings = np.hstack(mixed_embeddings)
        
    else:
        raise ValueError(f"Precision {precision} is not supported")
    
    return quantized_embeddings