t5_model=$1
dataset=$2
split=$3
hf_model=$4 # the base classifier


# create a temp directory for everything so that the model name does not coincide with hf_model from HF models

evaluation_dir=$t5_model/textattack_evaluation/
log_dir=$evaluation_dir/logs

mkdir -p $evaluation_dir
mkdir $log_dir
filename_prefix=$evaluation_dir/${dataset}_${split}
log_prefix=$log_dir/${dataset}_${split}

#filename_prefix=$evaluation_dir/${dataset}_${split}_${hf_model}
#log_prefix=$log_dir/${dataset}_${split}_${hf_model}
for attack in deepwordbug;
#for attack in textfooler textbugger bae pwws deepwordbug;
#for attack in textbugger bae pwws deepwordbug;
do
    csv_file=${filename_prefix}_$attack.csv
    log_file=${log_prefix}_$attack.txt

    echo "Running evaluation for $attack"
    textattack attack --recipe $attack --model-from-huggingface $hf_model --dataset-from-huggingface $dataset --num-examples -1 --dataset-split $split  --log-to-csv $csv_file --model-type-to-evaluate t5 --t5-model-to-use $t5_model | tee $log_file
done
