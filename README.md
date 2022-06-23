# [Temporal Multiresolution Graph Neural Networks For Epidemic Prediction](https://arxiv.org/abs/2205.14831)


## Data

### COVID-19 Dataset

#### Source

We borrow the COVID-19 dataset from the original [Transfer Graph Neural Networks for Pandemic Forecasting](https://arxiv.org/abs/2009.08388) paper. If you find the datasets useful in your research, please consider adding the following citation from the source paper:

```bibtex
@inproceedings{panagopoulos2020transfer,
  title={{Transfer Graph Neural Networks for Pandemic Forecasting}},
  author={Panagopoulos, George and Nikolentzos, Giannis and Vazirgiannis, Michalis},
  booktitle={Proceedings of the 35th AAAI Conference on Artificial Intelligence},
  year={2021},
}
```

#### Labels

Gathered is the ground truth for number of confirmed cases per region through open data for [Italy](https://github.com/pcm-dpc/COVID-19/blob/master/dati-province/dpc-covid19-ita-province.csv),
[England](https://coronavirus.data.gov.uk), [France](https://www.data.gouv.fr/en/datasets/donnees-relatives-aux-tests-de-depistage-de-covid-19-realises-en-laboratoire-de-ville/) and [Spain](https://code.montera34.com:4443/numeroteca/covid19/-/blob/master/data/output/spain/covid19-provincias-spain_consolidated.csv}}).
The data have been preprocessed and the final versions are in each country's subfolder in the data folder.


#### Graphs

The graphs are formed using the movement data from Meta's Data for Good Disease Prevention [maps](https://dataforgood.fb.com/docs/covid19/). More specifically, the data used is the total number of people moving daily from one region to another, using the [Movement between Administrative Regions](https://dataforgood.fb.com/tools/movement-range-maps/) datasets. The authors of the original paper can share an aggregated and diminished version which was used for their and our experiments. 
These can be found inside the "graphs" folder of each country. These include the mobility maps between administrative regions that we use in our experiments until 12/5/2020, starting from 13/3 for England, 12/3 for Spain, 10/3 for France and 24/2 for Italy.
The mapplots require the gadm1_nuts3_counties_sf_format.Rds file which can be found at the Social Connectedness Index [data](https://dataforgood.fb.com/tools/social-connectedness-index/).


### Hungary Chickenpox Dataset

A dataset of county level chickenpox cases in Hungary between 2004 and 2014. The dataset was made public during the development of [PyTorch Geometric Temporal](https://github.com/benedekrozemberczki/pytorch_geometric_temporal). The underlying graph is static - vertices are counties and edges are neighbourhoods. Vertex features are lagged weekly counts of the chickenpox cases (4 lags included). The target is the weekly number of cases for the upcoming week (signed integers). The dataset consist of more than 500 snapshots (weeks).


## Code Execution

### Requirements
To run this code you will need the following `python` and `R` packages:
- [numpy](https://www.numpy.org/)
- [pandas](https://pandas.pydata.org/)
- [scipy](https://www.scipy.org/)
- [PyTorch](https://pytorch.org/)
- [networkx](https://networkx.github.io/)
- [sklearn](https://scikit-learn.org/stable/)
- dplyr, sf, ggplot2, sp
- [PyTorch Geometric (pyg/torch-geometric)](https://github.com/rusty1s/pytorch_geometric)
- [PyTorch Geometric Temporal (torch-geometric-temporal)](https://github.com/benedekrozemberczki/pytorch_geometric_temporal)

### Run

#### COVID datasets TMGNN

To run the experiments with the default settings:
```bash
cd covid/code
python experiments_multiresolution.py
```

To run the baseline models: 
```bash
cd covid/code
python experiments.py
```

Use the script `gather_for_map.py` to aggregate data in the output folder to produce the map plots and the `tl_base.py` for the `TL_BASE` baseline. Use the `error_case_maps.R` to plot the maps of England (adjust it for the other countries). 

#### Hungary Chickenpox Dataset TMGN

To run the experiments with the default settings:
```bash
cd chickenpox
python experiments_hungary.py
```

### Experiments

- `experiments_multiresolution`: Testing source Multiresolution Graph Network model from paper [Multiresolution Equivariant Graph Variational Autoencoder](https://arxiv.org/abs/2106.00967) and the enhanced temporal model Temporal MGN on COVID data of four countries, predicting the number of cases at each location the next day. Underlying graphs are topologially static, with changing node features (past number of cases at each node) and edge features. TMGN code is written in the style of the original code from [Panagopoulos et al., 2020](https://arxiv.org/abs/2009.08388).
- `experiments_hungary`: Testing Temporal MGN model on Hungary Chickenpox dataset, given that the underlying graph is topologically static (only changing node features: number of cases from a predetermined past time window). Model predicts the number of cases at each node in the graph for the following day, 10 days, 20 days, and 40 days.

## Contributors

- Viet Bach Nguyen (correspondent), bach.nguyen.te@gmail.com
- Truong Son Hy, sonpascal93@gmail.com
