# TLOB-BiN: Transformer for Limit Order Book with Bidirectional Normalization

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/Tests-Pytest-green.svg)](https://pytest.org/)
[![Code Style](https://img.shields.io/badge/Code%20Style-Black-black.svg)](https://black.readthedocs.io/)
[![Type Checking](https://img.shields.io/badge/Type%20Checking-MyPy-blue.svg)](https://mypy.readthedocs.io/)
[![License: ISC](https://img.shields.io/badge/License-ISC-yellow.svg)](https://opensource.org/licenses/ISC)

> 🧠 **Advanced Transformer Architecture**: PyTorch implementation of Transformer models with Bidirectional Normalization (BiN) for cryptocurrency limit order book analysis and prediction.

A state-of-the-art machine learning pipeline for analyzing Bitcoin limit order book data using Transformer architectures enhanced with **Bidirectional Normalization (BiN)** - a novel normalization technique that processes data across both temporal and feature dimensions simultaneously.

## 🎯 Project Overview

TLOB-BiN combines advanced deep learning with financial market microstructure analysis, implementing:

- **🔥 BiN (Bidirectional Normalization)**: Novel normalization layer that normalizes across both time steps (temporal) and features simultaneously
- **🤖 Transformer Architecture**: Multi-head attention mechanisms adapted for sequential order book data  
- **📊 Bitcoin Order Book Processing**: Complete pipeline for BTC/USDT limit order book data from Kaggle
- **⚡ Type-Safe PyTorch**: Comprehensive type annotations with torchtyping for tensor shape validation
- **🧪 Robust Testing**: 24 comprehensive pytest tests covering edge cases and performance
- **📈 Financial ML Pipeline**: From raw order book data to ML-ready features with proper train/val/test splits

## 🏗️ Architecture Highlights

### BiN (Bidirectional Normalization) Layer

The core innovation is our **BiN normalization layer** that processes data in two directions:

```python
class BiN(nn.Module):
    """Bidirectional Normalization Layer
    
    Performs normalization in two directions:
    1. Temporal: normalize each feature across time steps  
    2. Feature: normalize each time step across features
    """
```

**Key Features:**
- **Dual Normalization**: Temporal (across time) + Feature (across features)
- **Learnable Parameters**: Separate scale/bias for each normalization type
- **Weighted Combination**: Learnable mixing weights for optimal balance
- **Edge Case Handling**: Robust NaN detection for single feature/timestep scenarios
- **Broadcasting Optimization**: Efficient tensor operations using einops

### TLOB Transformer Components

```python
# Multi-head attention with proper shape annotations
class TransformLayer(nn.Module):
    def forward(
        self, input: TensorType["batch", "hidden_dim", "seq_len"]
    ) -> TensorType["batch", "hidden_dim", "seq_len"]:
        # Self-attention + residual connections + layer norm + MLP
```

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/your-username/TLOB_bit.git
cd TLOB_bit

# Install dependencies
pip install -r requirements.txt

# Install additional shape typing (optional)
pip install torchtyping typeguard
```

### Run Tests

```bash
# Run comprehensive test suite
python -m pytest test_bin.py -v

# Test specific components
python -m pytest test_bin.py::test_initialization -v
python -m pytest test_bin.py::test_edge_cases -v

# Run with coverage
python -m pytest test_bin.py --cov=models.bin
```

### Data Preprocessing

```bash
# Process Bitcoin order book data
python main.py dataset=btc_usdt_spot

# The pipeline will:
# 1. Download BTC/USDT data from Kaggle
# 2. Process 10 levels of order book depth
# 3. Generate features with multiple prediction horizons
# 4. Create train/validation/test splits
# 5. Apply normalization and save processed data
```

### Model Usage

```python
import torch
from models.bin import BiN
from models.tlob import TLOB, TransformLayer

# Initialize BiN layer for 40 features, 50 timesteps (typical for LOB data)
bin_layer = BiN(num_features=40, seq_length=50)

# Example Bitcoin order book tensor: (batch, features, time)
x = torch.randn(32, 40, 50)  # 32 samples, 40 LOB features, 50 timesteps
normalized = bin_layer(x)    # Apply bidirectional normalization

# Full TLOB transformer
tlob_model = TLOB(
    hidden_dim=256,
    num_features=40,
    seq_length=50, 
    num_layers=6,
    num_heads=8
)
```

## 📊 Data Pipeline

### Bitcoin Order Book Features

The system processes **10 levels of LOB depth** with **40 total features**:

```python
# For each of 10 levels:
sell_price_1, sell_size_1, buy_price_1, buy_size_1,
sell_price_2, sell_size_2, buy_price_2, buy_size_2,
# ... up to level 10
```

### Label Generation

Multi-horizon prediction labels with adaptive thresholding:

```python
HORIZONS = [10, 20, 50, 100]  # prediction horizons in timesteps

# 3-class classification:
# UP (0): price_change > +α  
# STATIONARY (1): -α ≤ price_change ≤ +α
# DOWN (2): price_change < -α
```

### Data Normalization

- **Price normalization**: Z-score standardization across all price features
- **Size normalization**: Z-score standardization across all volume features  
- **Train statistics**: Normalization parameters computed on training set only

## 🧪 Testing Framework

Comprehensive pytest suite with **24 tests** covering:

### Core Functionality
- ✅ **Model initialization** and parameter shapes
- ✅ **Forward pass** with various batch sizes and dimensions
- ✅ **Bidirectional normalization** components
- ✅ **Scale/bias application** methods

### Robustness & Edge Cases  
- ✅ **Weight constraints** (positive mixing weights)
- ✅ **Zero standard deviation** handling
- ✅ **Single feature/timestep** scenarios
- ✅ **Large batch processing** (100+ samples)
- ✅ **Gradient flow** verification

### Performance & Compatibility
- ✅ **CPU behavior** testing
- ✅ **CUDA compatibility** (when available)
- ✅ **Deterministic behavior** verification
- ✅ **Input validation** (zeros, large values, small values)

### Parametrized Testing
- ✅ **Multiple model sizes**: (1,1) to (40,50) dimensions
- ✅ **Various batch sizes**: 1 to 100 samples
- ✅ **Fixtures and utilities** for reusable test components

```bash
# Example test output
======================== 23 passed, 1 skipped ========================
✅ Initialization test passed
✅ Basic forward pass test passed  
✅ Edge cases test passed
✅ Gradient flow test passed
🎉 All tests passed! BiN model is working correctly.
```

## 🔧 Technical Implementation

### Type-Safe PyTorch

Full type annotations with runtime validation:

```python
from torchtyping import TensorType
from typeguard import typechecked

@typechecked
def forward(
    self, input: TensorType["batch", "hidden_dim", "seq_len"]  # noqa: F821
) -> TensorType["batch", "output_dim", "seq_len"]:  # noqa: F821
    # Runtime shape validation + static type checking
```

### BiN Implementation Details

```python
def _temporal_norm(self, x, B, F, T):
    # Normalize each feature across time dimension
    temporal_mean = torch.mean(x, dim=2, keepdim=True)  # (B, F, T) -> (B, F, 1)
    temporal_std = torch.std(x, dim=2, keepdim=True)    # (B, F, T) -> (B, F, 1)
    
    # Handle edge cases: NaN detection for T=1
    temporal_std[torch.isnan(temporal_std)] = 1
    temporal_std[temporal_std < 1e-4] = 1
    
    # Broadcasting with einops for clarity
    temporal_mean_broadcast = einops.repeat(temporal_mean, 'b f 1 -> b f t', t=T)
    temporal_std_broadcast = einops.repeat(temporal_std, 'b f 1 -> b f t', t=T)
    
    return (x - temporal_mean_broadcast) / temporal_std_broadcast
```

### Configuration System

Hydra-based configuration with dataclasses:

```python
@dataclass  
class BTC_USDT_SPOT(Dataset):
    type: DatasetType = DatasetType.BTC_USDT_SPOT
    data_dir: str = "./preprocessed"
    dates: list = field(default_factory=lambda: ["2023-06-17", "2025-05-20"])
    batch_size: int = 128
    sampling_time: SamplingTime = SamplingTime.TWO_HUNDRED_MS
    split_rates: list = field(default_factory=lambda: [0.8, 0.1, 0.1])
    split_days: list = field(default_factory=lambda: [5, 1, 1])
```

## 📁 Project Structure

```
TLOB_bit/
├── models/
│   ├── bin.py              # BiN normalization layer (108 lines)
│   └── tlob.py             # TLOB transformer components
├── preprocessing/
│   └── btc.py              # Bitcoin order book data pipeline
├── config/
│   ├── config.py           # Hydra configuration dataclasses
│   └── __init__.py
├── test_bin.py             # Comprehensive test suite (316 lines)
├── main.py                 # Data preprocessing entry point
├── constants.py            # Project constants and enums
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

## 🎯 Use Cases

### Financial Market Analysis
- **Price movement prediction** for BTC/USDT
- **Market regime classification** (bullish/bearish/neutral)
- **Order flow pattern recognition**
- **High-frequency trading signal generation**

### Research Applications  
- **Market microstructure** analysis
- **Limit order book dynamics** modeling
- **Transformer adaptation** for financial time series
- **Normalization technique** research (BiN layer)

### Production Scenarios
- **Real-time trading systems** (with proper risk management)
- **Market making algorithms** optimization
- **Portfolio management** signal generation
- **Risk monitoring** and early warning systems

## 🚀 Performance Optimizations

### Computational Efficiency
- **einops broadcasting**: Explicit, efficient tensor operations
- **Vectorized operations**: No Python loops in forward pass
- **Memory optimization**: Proper tensor shape management
- **Batch processing**: Optimized for various batch sizes

### Future Enhancements
- **torch.compile**: 2-5x speedup potential with PyTorch 2.0+
- **Mixed precision**: FP16 training for faster convergence
- **Model parallelism**: Multi-GPU training for large models
- **TorchScript**: Production deployment optimization

## 🛣️ Roadmap

### ✅ Completed (Current)
- [x] BiN normalization layer with comprehensive testing
- [x] Basic TLOB transformer components  
- [x] Bitcoin order book data preprocessing pipeline
- [x] Type-safe PyTorch implementation
- [x] Robust testing framework (24 tests)
- [x] Configuration management with Hydra

### 🚧 In Progress
- [ ] Complete TLOB transformer model training loop
- [ ] Advanced attention mechanisms (FlexAttention)
- [ ] Model evaluation and backtesting framework
- [ ] Performance benchmarking suite

### 🔮 Future Plans
- [ ] **Multi-asset support**: ETH/USDT, other cryptocurrency pairs
- [ ] **Real-time inference**: WebSocket integration for live data
- [ ] **Advanced features**: Technical indicators, sentiment data
- [ ] **Model compression**: Quantization and pruning for production
- [ ] **Web interface**: Interactive model predictions and analysis
- [ ] **Research papers**: BiN normalization technique publication

## 🤝 Contributing

1. **Fork** the repository
2. **Create** feature branch: `git checkout -b feature/amazing-feature`
3. **Add tests** for new functionality
4. **Run test suite**: `python -m pytest test_bin.py -v`
5. **Commit** changes: `git commit -m 'Add amazing feature'`
6. **Push** to branch: `git push origin feature/amazing-feature`
7. **Submit** Pull Request

### Development Guidelines
- **Type annotations** required for all functions
- **Tests required** for new models/layers  
- **Code formatting** with Black
- **Linting** with flake8/mypy
- **Documentation** with comprehensive docstrings

## 📊 Performance Metrics

### Model Performance
- **BiN Layer**: 23/24 tests passing, 1 skipped (CUDA)
- **Memory Efficiency**: Handles 100+ batch sizes efficiently
- **Edge Case Robustness**: Single feature/timestep scenarios supported
- **Gradient Flow**: Verified backpropagation through all components

### Data Pipeline  
- **Processing Speed**: ~54MB compressed data processing
- **Feature Engineering**: 40 LOB features + multi-horizon labels
- **Data Quality**: Proper train/val/test splits with normalization
- **Kaggle Integration**: Automatic dataset download and processing

## 📄 License

ISC License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **PyTorch team** for the excellent deep learning framework
- **einops library** for clear tensor operations  
- **Kaggle dataset**: macqueen01/btcusdt-orderbook for Bitcoin data
- **Research community** for Transformer and attention mechanism innovations
- **Open source contributors** for testing and development tools

## 📞 Support & Contact

For questions, issues, or contributions:
1. **GitHub Issues**: Report bugs or request features
2. **Discussions**: Ask questions or share ideas
3. **Pull Requests**: Contribute code improvements
4. **Documentation**: Help improve project documentation

---

**⚠️ IMPORTANT DISCLAIMER**: This project is for **research and educational purposes**. 

- **Financial Risk**: Trading cryptocurrencies involves substantial risk. Past performance does not guarantee future results.
- **Research Code**: This is experimental research code, not production-ready software.
- **No Financial Advice**: Nothing in this repository constitutes investment or trading advice.
- **Use at Own Risk**: Any trading decisions made using this code are your responsibility.
- **Compliance**: Ensure compliance with all applicable laws and regulations in your jurisdiction.

**Always practice proper risk management and never trade with money you cannot afford to lose.** 