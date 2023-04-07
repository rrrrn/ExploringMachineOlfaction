```
datasets
┣ __init__.py
┣ dravnieks_dataset.py
┗ kellerdataset.py
```
The `{dataset}_dataset.py` reads SMILES and create `torch_geometric.Data` object for individual chemicals, which stores node, edge feature, edge index, and odor labels values, etc. The outcome variable is saved in `{dataset}\processed` directory and is loaded during the training and testing phase of the GNN model.

```
dravnieks
 ┣ processed 
 ┣ raw
 ┃ ┣ DravnieksGrid.csv
 ┃ ┣ descriptors.csv
 ┃ ┣ dravnieks_pa.csv
 ┃ ┣ identifiers.csv
 ┃ ┗ molecules.csv
 ```
 - `DravnieksGrid.csv`, `identifiers.csv`, `molecules.csv` are downloaded from [pyrfume project](https://github.com/pyrfume/pyrfume-data). `DravnieksGrid.csv` provides *descriptor applicability*, which is target of interest for this project. `identifiers.csv` and `molecules.csv` helped associate odor descriptor applicability with their SMILES string and CID number. All necessary information per molecules is integrated in `dravnieks_pa.csv`.
 - `descriptors.csv` is the full list of computed Mordred physicochemical descriptor for all chemicals relevant to Dravnieks dataset.
 ```
keller
 ┣ processed
 ┣ raw
 ┃ ┣ Keller_12868_2016_287_MOESM1_ESM.csv
 ┃ ┣ descriptors.csv
 ┃ ┣ keller_formatted.csv
 ┃ ┣ keller_pa.csv
 ┃ ┗ keller_pu.csv
```
-  `Keller_12868_2016_287_MOESM1_ESM.csv` is obtained from [Olfactory perception of chemically diverse molecules](https://link.springer.com/article/10.1186/s12868-016-0287-2#Sec22), and `keller_formatted.csv` is its cleaned form after preprocessing steps.
-  `descriptors.csv` calculated a full list of Mordred descriptor for all relevant chemicals used in Keller's study.
-  `keller_pu.csv` and `keller_pa.csv` derive the percetage usage and percentage applicability (*descriptor applicability*) from individual odor label ratings in the raw data.
```
overlapped.npy
preprocessing.ipynb
retained_descriptor.txt
retained_descriptors.npy
 ```
 - `overlapped.npy`: CIDs for molecules used in both prediction tasks.
 - `retained_descriptor.*`: final set of Mordred input feature names after selection and imputation.
 - `preprocessing.ipynb`: notebook where important pre-processing steps were carried out.
