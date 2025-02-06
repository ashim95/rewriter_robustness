hf_model=$1
dataset=$2
split=$3


# create a temp directory for everything so that the model name does not coincide with hf_model from HF models

#temp_dir=${hf_model}_temp
#mkdir -p $temp_dir
mkdir hf_models_sampleshielder
#evaluation_dir=$temp_dir/textattack_evaluation/
evaluation_dir=hf_models_sampleshielder/$hf_model/textattack_evaluation/
log_dir=$evaluation_dir/logs

mkdir -p $evaluation_dir
mkdir $log_dir
filename_prefix=$evaluation_dir/${dataset}_${split}
log_prefix=$log_dir/${dataset}_${split}

#for attack in bae pwws deepwordbug;
for attack in textfooler textbugger bae pwws deepwordbug;
do
    csv_file=${filename_prefix}_$attack.csv
    log_file=${log_prefix}_$attack.txt

    echo "Running evaluation for $attack"
    #echo "textattack attack --recipe $attack --model-from-huggingface $hf_model --dataset-from-huggingface $dataset --num-examples -1 --dataset-split $split  --log-to-csv $csv_file --model-type-to-evaluate hf_model | tee $log_file"
    textattack attack --recipe $attack --model-from-huggingface $hf_model --dataset-from-huggingface $dataset --num-examples -1 --dataset-split $split  --log-to-csv $csv_file --model-type-to-evaluate sample | tee $log_file
done

#mkdir -p $hf_model
#mv $temp_dir/* $hf_model/
#rm -rf $temp_dir
