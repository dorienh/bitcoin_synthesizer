# Forecasting Bitcoin Volatility Spikes from Whale Transactions and Cryptoquant Data Using Synthesizer Transformer Models

This is the code (models + weights) from the paper: 

`Herremans, Dorien and Low, Kah Wee, Forecasting Bitcoin Volatility Spikes from Whale Transactions and Cryptoquant Data Using Synthesizer Transformer Models. Available at SSRN: https://ssrn.com/abstract=4247685 or http://dx.doi.org/10.2139/ssrn.4247685`

*Abstract:* 
The cryptocurrency market is highly volatile compared to traditional financial markets. Hence, forecasting its volatility is crucial for risk management. In this paper, we investigate CryptoQuant data (e.g. on-chain analytics, exchange and miner data) and whale-alert tweets, and explore their relationship to Bitcoin's next-day volatility, with a focus on extreme volatility spikes. We propose a deep learning Synthesizer Transformer model for forecasting volatility. Our results show that the model outperforms existing state-of-the-art models when forecasting extreme volatility spikes for Bitcoin using CryptoQuant data as well as whale-alert tweets. We analysed our model with the Captum XAI library to investigate which features are most important. We also backtested our prediction results with different baseline trading strategies and the results show that we are able to minimize drawdown while keeping steady profits. Our findings underscore that the proposed method is a useful tool for forecasting extreme volatility movements in the Bitcoin market.

If this code is useful to use, please cite the original paper: 

  @article{herremans4247684forecasting,
    title={Forecasting Bitcoin Volatility Spikes from Whale Transactions and Cryptoquant Data Using Synthesizer Transformer Models},
    author={Herremans, Dorien and Low, Kah Wee},
    journal={Available at SSRN 4247684}
  }
