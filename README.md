# Machine Learning for Neighborhood Selection in Large Neighborhood Selection

## Installation
- Clone the repository
- Unzip the zip files in /ML4LNS/models/ (if you want to use the ML models)
    unzip \*.zip
- Create an environment (choose a name for it) with the required packages and install pyvroom using pip:


    conda create -n [envname] python=3.11.4 scikit-learn=1.2.2 imbalanced-learn=0.10.1 tqdm=4.65.0 joblib
    conda activate [envname]
    conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 cpuonly -c pytorch
    pip install pyvroom

## Run instructions 
    cd ./ML4LNS/code/
    python main.py [option1] [option2]
Where [option1] is either:
* 1: Run the oracle model
* 2: Run the random model
* 3: [default], Run the ML model. Indicate with [option2] which ML model should be read:
    * 1: Run ML1
    * 2: Run ML2
    * 3: Run ML3
    * 4: Run ML4
    * 5: [default] Run ML5 
* 4: Run data collection. Indicate with [option2] which data collection strategy should be used:
    * 0: [default] Use random data collection strategy 
    * 1: Use ML1 as data collection strategy
    * 2: Use ML2 as data collection strategy
    * 3: Use ML3 as data collection strategy
    * 4: Use ML4 as data collection strategy

## 1. Oracle model
The LNS with the oracle model as neighborhood selection is done on the test instances (R1_10_1 up to R1_10_10). 
The oracle model always selects the best of the given neighborhoods. 

## 2. Random model
The LNS with the random model as neighborhood selection is done on the test instances (R1_10_1 up to R1_10_10). 
The random model selects one of the given neighborhoods at random.

## 3. ML model
The LNS with the ML model as neighborhood selection is done on the test instances (R1_10_1 up to R1_10_10). 
The ML model uses machine learning to decide which of the given neighborhoods to select in an iteration.
[option2] specifies which ML model should be used.

## 4. Data Collection
The LNS with the data collection is done on the train instances.
For more info on the train instances, see the paper.  
The data collection strategy (specified [option2]) decides how the data collection should make its neighborhood selection decisions.
Features that are collected during data collection are output in the /ML4LNS/features/ folder. 

## Output
The output of a run consists of the average score at the end of a run for all the different instances. 
Also the average run time and the average number of vehicles are in the output.
Furthermore, the .sav files created in /ML4LNS/testruns/ can be read with joblib to obtain all raw test data.
