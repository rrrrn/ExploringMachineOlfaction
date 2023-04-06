# ExploringMachineOlfaction

## Dataset
This study investigate machine learning for modeling continuous *descriptor applicability* from molecular representations. Two prediction tasks based on psychochemical olfactory dataset are of interests, which mainly differ in human participants' domain knowledge level in olfaction field:
- *Dravnieks dataset* is gathered on panelists with special olfactory training and selection.
- *Keller dataset* collect individuals odor descriptor ratings for each chemical from naive healthy human subjects.
- Further information please check `data` directory.

## Experiment Code

### Modeling Descriptor Applicability
**Machine Learning Models**
`gridsearch_ml_mordred.py` fits descriptor-specific models (*Support Vector Machine*, *K Nearest Neighbors*, *Random Forest*, *Gradient Boosting*) in both prediction tasks.  

**Graph Neural Networks**
`train.py` trains proposed Graph Neural Networks on prediction task of interest.

### Embeddings as feature input
`gridsearch_ml_embeddings.py` maintains a similar structure and functionality as `gridsearch_ml_mordred.py` with feature inputs being GNN-learned embeddings.

### Transfer Learning
- `transfer_ml.py` applies descriptor-specific best-performing models fitted on Dravnieks dataset to predict *descriptor applicability* of matched odor labels in the Keller prediction task.
- `transfer_dl.py` fine-tuned pre-trained GNN model from one prediction task to learn and predict on another prediction task, where matching predicting targets are not required. 

## Results and Visualization
All results are stored in `results` directory. Visualization for the thesis is generated in `results/performance.ipynb` notebook and output figures are in `results/figs` for reference.
