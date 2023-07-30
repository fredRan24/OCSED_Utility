from OCSED_util import OCSED_util
from pathlib import Path

if __name__ =="__main__":
    #load util class
    util = OCSED_util()
    current_dir = Path().parent.absolute()
    
    model_dir = current_dir / 'Trained_models'
    model_name = "baseline_AT_CRNN"

    #Load the baseline, pre-trained model
    model = util.load_trained_model(model_directory=model_dir,  model_name=model_name)
    
    #Evaluate it
    util.eval_model(model)
