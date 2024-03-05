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
    cd ./code/
    python main.py [phase] [option]
Where [phase] is either:
* 0: To run the recommendations test. The neighborhood creation is random. The [option] is either:
	* 0: To turn off recommendations
	* 1: To turn on recommendations
* 1: To run phase 1. This means the anchor route and the set of routes to choose from remain constant. The [option] is either:
	* 0: To have random neighborhood creation
	* 1: To have heuristic neighborhood creation
	* 2: To have attention (RL) neighborhood creation
* 2: To run phase 2. This means the anchor route varies per iteration, but the set of routes to choose from remains constant. The [option] is either:
	* 0: To have random neighborhood creation
	* 1: To have heuristic neighborhood creation
	* 2: To have attention (RL) neighborhood creation
* 3: To run phase 3. Both the anchor route and the set of routes to choose from vary. Like in the real-world application. The [option] is either:
	* 0: To have random neighborhood creation
	* 1: To have heuristic neighborhood creation
	* 2: To have attention (RL) neighborhood creation
	
## Output
The output of a run consists of the average score at the end of a run for all the different instances. 
Also the average run time and the average number of vehicles are in the output.
Furthermore, the .sav files created in /ML4LNS/testruns/ can be read with joblib to obtain all raw test data.
