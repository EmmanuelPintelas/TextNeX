# TextNet-HeX - Generator
Heterogeneous ensemble of lightweight Text Network eXperts. Contains only the Generation phase. In this phase, multiple lightweight text networks are generated, each trained under diverse configurations to capture unique perspectives on text characteristics.

Step 1: Specify root path of data, e.g. root = 'Datasets' (line 37).

Step 2: Specify Task, e.g. task = 'substask_1' (line 39). # 'substask_1' or 'substask_2'

Step 3: Specify number of iterations, e.g. N = 50 # the number of trained models to generate (line 187).

Step 4: Specify save path, e.g. root_path_save = 'Storage' (line 200).

Step 5: Specify val acc threshold, e.g. val_f1 > 0.62 (line 309). # in order to avoid saving useless models.

Step 6: Run script. When terminates your output will be like: 

          Storage
             |
             |
             |---> models  # the generated trained models with val accuracy over the specified threshold.
             |---> metrics # the excel files containing all mertic results of saved models.
