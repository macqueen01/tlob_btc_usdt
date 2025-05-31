# TLOB (Time-series Limit Order Book) Processor

A high-performance TypeScript application for processing cryptocurrency limit order book data from Binance via CoinAPI's S3 service, with support for real-time streaming, data transformation, and market condition analysis.

## 🚀 Features

- **Real-time Order Book Processing**: Stream and process limit order book data from CoinAPI's S3-hosted datasets
- **Multi-format Support**: Handle both local and remote (CoinAPI S3) data sources
- **Market Condition Analysis**: Pre-configured training datasets for bullish, bearish, and neutral market conditions
- **High Performance**: Optimized streaming with configurable batch processing and parallel file handling
- **Secure Configuration**: Environment-based credential management with no hardcoded secrets
- **Go Integration**: Hybrid TypeScript/Go architecture for performance-critical operations
- **Flexible Time Range**: Process specific timestamp ranges with millisecond precision
- **Data Validation**: Built-in error handling and retry mechanisms for robust data processing

## 📊 Supported Data

- **Data Provider**: CoinAPI
- **Exchange**: Binance Spot
- **Trading Pair**: BTC/USDT
- **Data Type**: Full depth limit order book snapshots
- **Format**: Gzipped CSV files with semicolon delimiters
- **Storage**: CoinAPI's S3-compatible service
- **Frequency**: Real-time updates with configurable aggregation windows

## 🛠 Installation

### Prerequisites

- Node.js 18+ and npm
- Go 1.19+ (for Go components)
- TypeScript 5.8+
- **CoinAPI Access**: Confidential access key from CoinAPI

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/macqueen01/tlob_btc_usdt
   cd tlob_btc_usdt
   ```

2. **Install dependencies**
   ```bash
   npm install
   ```

3. **Configure CoinAPI credentials**
   ```bash
   cp .env.example .env
   # Edit .env with your CoinAPI access key
   ```

4. **Build the project**
   ```bash
   npx tsc
   ```

## ⚙️ Configuration

Create a `.env` file in the project root with your CoinAPI credentials:

```env
# CoinAPI S3 Configuration (Required)
S3_REGION=us-east-1
S3_BUCKET=coinapi
S3_ACCESS_KEY=your-coinapi-access-key-here
S3_SECRET_KEY=coinapi
S3_ENDPOINT=https://s3.flatfiles.coinapi.io

# Processing Configuration (Optional)
RAW_DATA_DIR=./raw
LOG_DIR=./logs
```

**Important Notes:**
- `S3_ACCESS_KEY`: Confidential key provided by CoinAPI (contact CoinAPI for access)
- `S3_SECRET_KEY`: Always set to `coinapi` (CoinAPI's standard secret)
- `S3_BUCKET`: Always set to `coinapi` (CoinAPI's bucket name)
- `S3_ENDPOINT`: CoinAPI's S3-compatible endpoint

### Training Data Configuration

The system includes pre-configured training dates for different market conditions:

```typescript
// Bullish conditions 📈
- 2023-03-14: CPI cooldown → BTC +10%
- 2023-10-23: ETF speculation pump
- 2024-02-13: CPI low → risk-on rally
- 2025-01-10: Post-ETF approval rally

// Bearish conditions 📉
- 2023-03-10: SVB collapse panic
- 2023-11-09: Binance/DOJ fears, dump
- 2024-08-17: Weekend low-liquidity drop
- 2025-02-24: Sharp correction post-rally

// Neutral conditions 😐
- 2023-06-17: Weekend, stable price
- 2024-04-20: Bitcoin halving, muted intraday
- 2024-12-01: Calm sideways weekend
```

## 🚀 Usage

### Basic Processing

```bash
# Process data from local files
npm run preprocess

# Test CoinAPI S3 connectivity
npm run test-s3
```

### Advanced Configuration

```typescript
import { main } from "./order-book/main";

// Process specific date range
const FROM = "2023-06-17";
const TO = "2023-06-17";
const DOWNLOAD_ONLY = false;
const USE_REMOTE = true;
const PARALLEL_FILES = 5;

main(DOWNLOAD_ONLY, FROM, TO, USE_REMOTE, PARALLEL_FILES);
```

### Download Mode

```typescript
// Download and unzip training data from CoinAPI
main(true, "", "", true, 1);
```

## 📁 Project Structure

```
TLOB_bit/
├── src/
│   ├── coin-api/           # CoinAPI S3 configuration
│   ├── order-book/         # Core processing logic
│   │   ├── concrete/       # OrderBook implementations
│   │   ├── stream.ts       # Streaming utilities
│   │   ├── main.ts         # Main processing pipeline
│   │   └── utils.ts        # Utility functions
│   ├── go-integration/     # Go wrapper and integration
│   ├── config.ts           # Environment configuration
│   └── index.ts            # Application entry point
├── cmd/                    # Go applications
├── raw/                    # Local data storage
├── preprocessed/           # Processed output
└── logs/                   # Processing logs
```

## 🔧 Data Processing Pipeline

1. **File Discovery**: Scan CoinAPI S3 or local directory for target date range
2. **Stream Creation**: Establish read streams (CoinAPI S3 or local file system)
3. **Decompression**: Gunzip compressed CSV data
4. **CSV Parsing**: Parse semicolon-delimited format with error handling
5. **Batch Processing**: Group records for efficient processing
6. **Order Book Updates**: Apply buy/sell order changes
7. **Time Series Output**: Generate aggregated CSV with configurable depth

### CoinAPI Data Format

The system processes Binance order book data from CoinAPI with the following structure:
```
Path: T-LIMITBOOK_FULL/D-YYYYMMDD/E-BINANCE/IDDI-138123+SC-BINANCE_SPOT_BTC_USDT+S-BTCUSDT.csv.gz
```

### Output Format

```csv
timestamp,sell_price_1,sell_size_1,...,buy_price_1,buy_size_1,...
1686960323200,50001.00,1.5,50000.50,2.0
```

## 🔍 Market Data Analysis

The system focuses on June 17, 2023 (timestamp range: 1686960323000 - 1687044933000), which represents:
- **Duration**: Full trading day (23+ hours)
- **Market Condition**: Neutral/stable weekend trading
- **Data Size**: ~54MB compressed data from CoinAPI
- **Significance**: Baseline period for model training

## 🧪 Testing

```bash
# Test CoinAPI S3 connectivity and permissions
npm run test-s3
```

## 🚧 Development

### Adding New Data Sources

1. Implement the `listFiles` interface in `utils.ts`
2. Add source-specific configuration
3. Update the streaming pipeline for new formats

### Extending Order Book Logic

1. Inherit from `OrderBook` base class
2. Implement custom `apply()` methods
3. Add to the processing pipeline

## 📈 Performance

- **Streaming Architecture**: Memory-efficient processing of large datasets
- **Parallel Processing**: Configurable concurrency for multiple files
- **Batch Optimization**: Tunable batch sizes for optimal throughput
- **Error Recovery**: Automatic retry with exponential backoff
- **Resource Management**: Proper cleanup and memory management

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/new-feature`
3. Commit changes: `git commit -am 'Add new feature'`
4. Push to branch: `git push origin feature/new-feature`
5. Submit a Pull Request

## 📄 License

ISC License - see LICENSE file for details.

## 🔗 Related Resources

- [Binance API Documentation](https://binance-docs.github.io/apidocs/)
- [CoinAPI Documentation](https://docs.coinapi.io/)
- [CoinAPI Data Access](https://www.coinapi.io/market-data-api)
- [TypeScript Handbook](https://www.typescriptlang.org/docs/)
- [AWS SDK v2 Documentation](https://docs.aws.amazon.com/AWSJavaScriptSDK/latest/)

## 📞 Support

For questions, issues, or contributions, please:
1. Check existing issues on GitHub
2. Create a new issue with detailed description
3. Follow the contributing guidelines

For CoinAPI access issues:
- Contact CoinAPI support for access key provisioning
- Ensure your access key has permissions for historical order book data

---

**Note**: This project processes financial market data from CoinAPI for research and analysis purposes. Ensure compliance with CoinAPI's terms of service and relevant data usage agreements. 