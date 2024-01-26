
# InfoMol

Evaluating properties of molecular fingerprints using Neural Mutual Information estimation.


## Fingerprints

Traditional Fingerprints

| Name    | Length | Type | Software |
|---------|--------|------|----------|
| AP2DC   | 780    |      | PaDEL    |
| E-state | 79     |      | RDKit    | 
| KRC     | 4860   |      | PaDEL    |
| MACCS   | 166    |      | PaDEL    |
| PubChem | 881    |      | PaDEL    | 
| SSC     | 307    |      | PaDEL    | 

Machine Learning representations

| Name                    | Length | Code      |
|-------------------------|--------|-----------|
| MolBERT (v1, v2)        | -      | Available | 
| MolXPT                  | -      | ???       |
| GIN (Graph Contrastive) | -      | Implement |


## Datasets

| Dataset       | Length | N Properties | Properties                                               |
|---------------|--------|--------------|----------------------------------------------------------|
| ESOL          | 1128   | 1            | Solubility in water                                      | 
| FreeSolv      | 642    | 1            | Hydration free energy                                    | 
| Lipophilicity | 4200   | 1            | ocatnol/water distribution coefficient                   | 
| PCBA          | 437929 | 128          | Measured biological activities of small molecules        | 
| HIV           | 41127  | 1            | Experimental measured ability to inhibit HIV replication | 
| BACE          | 1513   | 1            | binding results for beta-secretase                       | 
| BBBP          | 2039   | 1            | Blood-brain barrier penetration                          | 
| Tox21         | 7831   | 12           | toxicity measurements on 12 biological targets           | 
| ToxCast       | 8575   | 617          | toxicology data for different tasks                      | 
| SIDER         | 1427   | 27           | Marketed drugs and adverse drug reaction                 | 
| ClinTox       | 1478   | 2            | Qualitative data of drugs approved and failed by FDA     |
| FishTox       | -      | -            | Fish toxicity measurements                               | 


