# Detecting Informed Trading: Plan Review and Enhancements

Our goal is to catch *Polymarket* insiders using on-chain data and a published “insider cases” dataset.  The proposed plan – using ForesightFlow’s known cases as ground truth, profiling each trader’s pre-news activity, and then flagging outliers with an autoencoder – is reasonable in outline.  However, we can refine each step, consider alternative models (like an isolation forest), and ensure a robust evaluation. Below we discuss the plan step-by-step, citing relevant literature and suggesting concrete improvements.  

## Data Sources (Raw Trades and Insider Cases)  

- **On-chain trade data:** The Polymarket-v1 dataset (≈1.2 billion trades, 1.3 million markets, $61 B volume) provides ground‐truth “aggressor direction” (buy vs. sell) for every trade.  This eliminates guesswork on trade intent.  We should confirm we have the latest version (2022–2026) from Hugging Face as mentioned .  
- **Insider case labels:** The ForesightFlow Insider Cases (FFIC) inventory lists 8 documented episodes (24 markets) of confirmed info leaks on Polymarket.  It includes *exact timestamps* when the news hit public, and the affected markets.  This ground truth can be used for evaluation and to define the “leaked” vs “normal” markets.  (Using these known cases avoids hand-wavy definitions of insiders – we *already know* which events had leaks.)  

## Step 1 – Selecting Markets (Leaked vs Control)  

- **Leaked markets:** Use the FFIC list to identify all markets “with leaks.”  The plan correctly isolates markets where pre-news trading was suspicious.  We should double-check the market IDs and resolution dates in FFIC (cases.yaml).  FFIC is intended exactly for validating detection methods.  
- **Control group:** Instead of random markets, choose a *matched* control set.  For example, sample markets of similar type (binary outcome, similar resolution dates) but with no reported leaks.  This ensures baseline traders face similar normal volatility and information schedules.  Poor matching could confound results (e.g. comparing geopolitical event markets to random sports markets).  Also, ensure controls have on-chain trades in the same pre-news window lengths.  

## Step 2 – Defining the Pre-News Window  

- **Cut-off time:** For each leaked market, use the precise “news release” timestamp from FFIC (or the companion event-timestamp dataset) as the boundary.  Trades **after** that time are excluded.  This cleanly isolates the period “when only insiders should know” versus “everyone knows.”  
- **Window length:** We might restrict attention to a short window before news (e.g. last 24 or 48 hours) rather than weeks.  Insiders usually trade shortly before resolution.  Experiment with different window sizes – too long and you dilute the signal with normal activity, too short and you may miss subtle ramp-up.  
- **Control periods:** For normal markets, define a comparable “dummy” window before some resolution event.  If controls are deadline-resolved markets, use some fixed horizon (e.g. 24h before resolution) as the pseudo “cut-off.”  This ensures our profiles span similar time lengths on both groups.  

## Step 3 – Labeling “Insider” Traders  

- **Ground-truth insiders:** Ideally, identify the actual addresses involved in the FFIC cases (if available) and treat those as known insiders.  If FFIC doesn’t list addresses, at least the case narratives often mention *what happened* (e.g. “sold Maduro contract minutes before headline”).  We should extract any on-chain IDs from FFIC evidence.  
- **Heuristic insiders:** The plan suggests flagging “massive, concentrated bets” just before news.  This is sensible as a first cut, but it needs concrete thresholds.  For example, we could mark any trader whose trade size is, say, >3× their median bet or >> typical market trade (similar to the *unusual sizing* rule in an existing Polymarket alert bot).  Also, wallets with 60%+ of funds in one market (per that bot) are flagged.  Rather than manually labeling insiders for training, it may be better to use the FFIC set for *evaluation only* and treat all traders as unlabeled during modeling.  

*Suggested improvement:* Rather than hand-picking “insiders” for training, we should train the anomaly model on **only normal trading patterns**.  Then we evaluate its output by checking if the highest-scoring anomalies align with the known FFIC insiders.  This avoids circular reasoning and mimics a realistic alert system.  

## Step 4 – Constructing Behavioral Profiles  

This is crucial – the anomaly detector only sees feature vectors, so we must capture relevant signals. Key features include:

- **Bet size and volume:** For each trader, compute *average bet size*, *total volume traded*, and the *distribution* of bet sizes.  Insiders often make unusually large bets (e.g. >3× average).  We can include “max bet” or “median bet”.  
- **Direction bias:** Compute the fraction of stakes on buys vs. sells.  A market maker or regular speculator often has mixed buy/sell activity, but an informed trader will heavily skew *in one direction* (e.g. entirely buying the winning outcome).  
- **Portfolio concentration:** Measure how much a trader’s activity is concentrated in a single market or category.  For example, % of their total volume in one contract (the HHI or concentration index).  The bot example flags “60%+ concentration in a single market”.  We can generalize: insiders will have a much more focused portfolio than random users.  
- **Market types and diversity:** Count how many markets and categories (e.g. sports vs geo vs crypto) the trader participated in.  Normal bettors might spread across topics, whereas an insider likely bets on the leaked news alone.  
- **Timing features:** Measure recency patterns – e.g., time between a trader’s first and last trade in the window, or average time from trade to news.  An insider typically jumps in only very near the event.  
- **Activity level:** Number of trades, number of separate transactions. Insiders might execute a single or few large transactions, rather than many small ones.  

These are similar to features used in the Polymarket-insider bot’s scoring system.  We should normalize or scale features appropriately (e.g. log-transform sizes to handle heavy tails).  If needed, dimensionality reduction (PCA) could compress correlated signals.  

*Suggested improvement:* Explore automated feature learning (e.g. use an autoencoder or variational autoencoder on raw trade time series) to capture subtle patterns.  However, a manual feature engineering step, inspired by domain knowledge (as above), often yields interpretable and effective results.  

## Step 5 – Anomaly Detection Model  

- **Autoencoder:** The plan uses a deep autoencoder, which is a neural net trained to reconstruct inputs.  It will learn the manifold of “normal” trader profiles so that anomalies (insiders) incur large reconstruction error.  This approach is common in fraud detection and cybersecurity.  In one study of insider threats, an LSTM autoencoder outperformed other models on recall and F1 score.  We should tune the autoencoder carefully: for example, use a bottleneck latent dimension small enough to force compression, and possibly include dropout or noise to regularize.  Training should use only the vast majority of “normal” traders (including control markets and non-insider actors in leaked markets).  

- **Isolation Forest:** Indeed, an Isolation Forest is a strong alternative.  It is an ensemble of random trees that “isolates” anomalies by random splits.  Advantages: it requires no training on labeled data, scales well to high dimensions, and automatically scores outliers.  Isolation Forest has linear time complexity and low memory use.  In practice, it often matches autoencoders in performance (sometimes with fewer false positives).  We should therefore **train both models** on the same feature set and compare their outputs.  For example:  
  - Train the autoencoder on normal traders, compute reconstruction error for all traders.  
  - Fit an IsolationForest on the same normal traders, and get its anomaly score for everyone.  

Each method will produce a score for each trader (reconstruction error or isolation score). We can then rank traders by anomaly scores.  

*Suggested improvement:* We might also include simpler benchmarks for context: e.g. one-class SVM or cluster-based methods, or even statistical z-scores on key features (though those can be brittle).  But Autoencoder and Isolation Forest are a good start.  

## Step 6 – Evaluation of Anomalies  

- **Ranking vs ground truth:** After scoring, we should verify that known FFIC insiders rank high.  For example, compute Precision@K: what fraction of the top-K flagged traders match the FFIC insider list.  Alternatively, treat insider identification as a binary classification and compute ROC AUC or PR curves.  Because insiders are rare, metrics like recall at low false-positive rates are crucial.  
- **Qualitative inspection:** Inspect examples of flagged traders.  Do they match the suspicious patterns (late large bets, 1-sided)?  This sanity check helps catch modeling issues.  
- **False positives:** Some normal traders may look strange (e.g. whale market makers).  Check if the model falsely flags anyone known benign.  

*Suggested improvement:* If the initial models perform poorly, consider refining features (maybe include features from [29] like *wallet age* or *recent activity*), or ensembling the two scores (autoencoder + isolation).  Also, the Medium article notes that graph methods (wallet linking) can help detect collusion – if feasible, one could construct a transaction graph or shared IP/similar behaviors, but that’s advanced.  

## Comparing Autoencoder vs Isolation Forest  

- **Model bias:** The LSTM autoencoder study found that the autoencoder gave the best recall/precision in detecting insider anomalies, while the Isolation Forest had slightly lower F1 but fewer false positives.  In our context, the autoencoder might more flexibly capture complex patterns of normal behavior, whereas an Isolation Forest will simply isolate points with extreme feature values.  
- **Scalability:** With ~10^5 traders, an Isolation Forest should be very fast (linear time).  The autoencoder may require more training time but can leverage GPU acceleration.  
- **Feature handling:** The autoencoder can naturally handle non-linear feature interactions.  IsolationForest can also deal with irrelevant features (random splits), but might require more tuning (e.g. number of trees, contamination rate).  
- **Time dimension:** Neither model inherently handles time sequences.  If timing of trades is crucial (e.g. trading *just* 5 minutes before news), we might incorporate time-lag features.  Alternatively, use an LSTM autoencoder on raw trade sequences (as in).  

*Suggested improvement:* Try a hybrid: use the autoencoder and isolation forest together. For instance, flag any trader that is an outlier by *either* model.  Or train a small meta-classifier on the two scores.  

## Additional Considerations  

- **Data leakage caution:** When training on “normal” data, ensure we exclude any data after the news release or that could indirectly encode leaked info.  Keep training purely in pre-news windows.  
- **Hyperparameter tuning:** For IsolationForest, the `contamination` parameter (expected fraction of anomalies) should be set low (e.g. 0.01) or grid-searched.  The autoencoder’s architecture (layers, latent size) should be tuned on a validation set of normal traders.  
- **Metric learning:** We could augment training by injecting a few known “insider” profiles as negative examples (semi-supervised learning), but this risks overfitting to those cases.  Better to rely on unsupervised detection first.  
- **Evaluation on held-out cases:** Since FFIC has 24 markets, consider a “leave-one-case-out” test: train the model without a particular leaked case, then see if it still finds the insiders in that left-out case.  This checks generalization to new events.  

## Summary  

The core strategy – using blockchain-verified trade data and known leak timestamps to spotlight abnormal traders – is solid.  We should leverage the FFIC dataset as ground truth and use rigorous metrics.  Using both an autoencoder and an Isolation Forest is wise: each has strengths.  In prior work on insider anomalies, LSTM autoencoders often outperform but Isolation Forest is competitive and efficient.  Key improvements include carefully engineering profiles (inspired by existing Polymarket “insider” rules), matching control markets properly, and fully evaluating against the FFIC labels.  With these refinements, the plan can robustly distinguish true insiders (high anomaly scores) from ordinary traders.  

**Sources:** The Polymarket-v1 trade dataset provides 1.2B records with ground-truth buy/sell direction.  The ForesightFlow Insider Cases (FFIC) offers 8 confirmed leak events (24 markets) for benchmarking.  Prior work on anomaly detection (LSTM autoencoders, Isolation Forests) and Polymarket-specific heuristics inform our approach.