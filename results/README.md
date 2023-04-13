# Results
```
{dataset}
┣ best_models
┣ best_params
┣ gnn_regr
┣ metrics
┗ embeddings
```
- `best_models` save best-performing machine learning models (Linear Regression, SVM, KNN, RF, and XGB) for each of odor descriptor used in {dataset} with `joblib' extension. Correspondingly, `best_params` saves hyperparamters used for training these ML models. `metrics` maintains an archive of odor descriptor-wise test metric of interest for each ML model.

- In `gnn_regr` there saved selected GNNs trained and tested on {dataset}, with their names indicating weighted explained variance on test data. One can also access learned embeddings by the GNN, and corresponding target value file for easier access.

- `embeddings` has the same structure regarding `best_models`, `gnn_regr`, `metrics`.

```
performance.ipynb
figs
```
`performance.ipynb` summarize results into visualization, and the generated figures are collected under `figs`.

```
transfer_ml
 ┣ gnn
 ┣ gnn_reverse
 ┗ mordred
```
- `gnn` archives transfer learning output of GNN original trained on *Dravnieks* but fine-tuned on *Keller* task, and `gnn_reverse` archives the results of a reverse process.
- `mordred` collects results of applying ML model trained on *Dravnieks* directly on *Keller* for those matching odor descriptor pairs.
