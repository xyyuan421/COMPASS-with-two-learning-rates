This package is based on the COMPASS toolbox(https://github.com/CogComNeuroSci/COMPASS/tree/main).
We provide the users with two choices: one learning rate(the same with the original toolbox) or two learning rates(positive learning rates and negative learning rates).
After downloading the package, the method to run the programs is as follows:
step 1: create an excel file for the input parameters. There are some examples in the package.
step 2: Go to the directory where the package are stored, according to different operations: 
        if choose the one learning rate version, then run python input.py 1 IC(or EC, GD);
        if choose the two learning rates version, then run python input.py 2 IC(or EC, GD).
Note: There are some bugs with the EC criterion in the two learning rates version.