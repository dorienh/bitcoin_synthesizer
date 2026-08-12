# Forecasting Bitcoin Volatility Spikes from Whale Transactions and CryptoQuant Data Using Synthesizer Transformer Models

[![IEEE](https://img.shields.io/badge/IEEE%20Access-10.1109%2FACCESS.2025-00629B?logo=ieee&logoColor=white)](https://ieeexplore.ieee.org/document/11058926/)

This repository contains the code (models + weights) accompanying our paper, published in **IEEE Access**:

> D. Herremans and K. W. Low, "Forecasting Bitcoin Volatility Spikes From Whale Transactions and CryptoQuant Data Using Synthesizer Transformer Models," in *IEEE Access*, vol. 13, pp. 117788–117807, 2025. [[IEEE Xplore]](https://ieeexplore.ieee.org/document/11058926/)

## Abstract

The cryptocurrency market is highly volatile compared to traditional financial markets. Hence, forecasting its volatility is crucial for risk management. In this paper, we investigate CryptoQuant data (e.g. on-chain analytics, exchange and miner data) and whale-alert tweets, and explore their relationship to Bitcoin's next-day volatility, with a focus on extreme volatility spikes. We propose a deep learning Synthesizer Transformer model for forecasting volatility. Our results show that the model outperforms existing state-of-the-art models when forecasting extreme volatility spikes for Bitcoin using CryptoQuant data as well as whale-alert tweets. We analysed our model with the Captum XAI library to investigate which features are most important. We also backtested our prediction results with different baseline trading strategies and the results show that we are able to minimize drawdown while keeping steady profits.

![Bitcoing forecasting]((https://github.com/dorienh/bitcoin_synthesizer/blob/main/cryptoquant.png?raw=true)

## Repository structure

| Folder | Description |
|---|---|
| [`data_processing/`](./data_processing) | Scripts and notebooks for preparing CryptoQuant on-chain/exchange/miner data and whale-alert tweet data for modelling |
| [`parsing_script/`](./parsing_script) | Parsing utilities for raw input data sources |
| [`model_training/`](./model_training) | Training code for the Synthesizer Transformer volatility-forecasting model |
| [`models/`](./models) | Model architectures and saved weights |
| [`prediction/`](./prediction) | Inference / prediction scripts for generating volatility forecasts |
| [`backtesting/`](./backtesting) | Trading-strategy backtests built on top of the model's predictions |

## Citation

If you use this code or find it useful in your research, please cite the paper:

```bibtex
@article{herremans2025forecasting,
  title={Forecasting Bitcoin volatility spikes from whale transactions and CryptoQuant data using Synthesizer Transformer models},
  author={Herremans, Dorien and Low, Kah Wee},
  journal={IEEE Access},
  volume={13},
  pages={117788--117807},
  year={2025},
  publisher={IEEE}
}
```

## License

Released under the [MIT License](./LICENSE).
