# T4SEfinder: 

[T4SEfinder Website](https://tool2-mml.sjtu.edu.cn/T4SEfinder_TAPE/)

Pytorch implementation of T4SEfinder, a genome-scale annotation tool for bacterial type IV secretion system effectors (T4SEs) using pre-trained model. T4SEfinder integrates experimental verified T4SEs in SecReT4 database and those in other studies as the training dataset. It applies protein pre-trained language model(provided by [TAPE repository](https://github.com/songlab-cal/tape)) to the prediction task and achieves high testing accuracy(**97.2%**). It also provides genome-scale prediction for T4SEs.

![Workflow](https://tool2-mml.sjtu.edu.cn/T4SEfinder_TAPE/images/model.jpeg)

## Set up

The stand-alone version of T4SEfinder has been tested in Linux version 3.10.0-1062.12.1.el7.x86_64 as well as macOS Big Sur 11.2.3.

Before using the T4SEfinder, please ensure that Python has been installed in your server. 
Please check the `requirements.txt` file for more details on required Python packages. You can create new environment and install all required packages with:

```shell
pip install -r requirements.txt
```

The model paramter files can be download at [here](https://drive.google.com/drive/folders/1XTA1wSIU4L1p128oXUnn3sGiXoMXX4O6).

## Using T4SEfinder

T4SEfinder can predict T4SEs from protein sequences in the FASTA format.
```shell
python main.py -in example/demo.fasta -weights weights/mlp/ -vote tapebert_mlp
```
The prediction results can be found in `results/`, including predicted probabilities by model weights from 5-fold cross validation and putative T4SEs after voting.

Besides the most recommended model `TAPEBert_MLP`, T4SEfinder provides another three approaches in T4SEs prediction.
 -  `TAPEBert_SVM`: replaces the downstream classifier into SVM.
 -  `PSSM_CNN`: based on positional-specific scoring matrix(PSSM) and CNN.
 -  `HybridBiLSTM`: conbines pre-trained feature and PSSM at C terminal in BiLSTM.
 If you want to used the model base on PSSM feature, NCBI BLAST+ 2.10.0 is required(can be downloaded from [ftp.ncbi.nlm.nih.gov](ftp://ftp.ncbi.nlm.nih.gov/blast/executables/blast+/2.10.0])), and the Swissprot database can be downloaded at [here](https://drive.google.com/drive/folders/1FvcYGMWR4DBYTBTv4Vcpl4_iEASN_GQG).

T4SEfinder can annotate bacteria genome to discover T4SE-encoding genes. 
```shell
./pred_all_model <NCBI Accession Number> #  e.g. NC012442
```
You can receive the summarized results obtained by various methods in `summary.csv`.

## Testing Result

We have compared T4SEfinder(`TAPEBert_MLP`) with existing prediction tools according to the perfomance on an independent test set(30 T4SEs + 150 none-T4SEs). 

|     Method     |  ACC  |   SN   |   SP   |   PR   |  F1   |  MCC  |
| :------------: | :---: | :----: | :----: | :----: | :---: | :---: |
| T4SEpre_psAac  | 90.0% | 63.3%  | 95.3%  | 73.1%  | 0.679 | 0.622 |
| T4SEpre_bpbAac | 88.3% | 66.7%  | 92.7%  | 64.5%  | 0.656 | 0.586 |
|     DeepT4     | 86.7% | 80.0%  | 88.0%  | 57.1%  | 0.667 | 0.599 |
|    BastionX    | 93.3% | 100.0% | 92.0%  | 71.4%  | 0.833 | 0.811 |
|  CNNT4SE_Vote  | 97.8% | 86.7%  | 100.0% | 100.0% | 0.929 | 0.919 |
|  TAPEBert_MLP  | 97.2% | 93.3%  | 98.0%  | 90.3%  | 0.918 | 0.901 |

Apart from the considerable prediction accuracy, T4SEfinder shows a major advantage in computational efficiency due to the adoptation of protein pre-trained langugae model.

## Contact

Please contact Yumeng Zhang at zhangyumeng1@sjtu.edu.cn for questions.
