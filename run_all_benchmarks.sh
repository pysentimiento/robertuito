declare -a arr=(
    "beto-ft-1000"
    "beto-ft-2000"
    "beto-uncased-1000"
    "beto-uncased-2000"
    "beto-uncased-5000"
)

for name in ${arr[@]}
do
  model_name="models/${name}"
  output_path="output/${name}.json"
  echo $model_name
  echo $output_path
  CUDA_VISIBLE_DEVICES=0 python bin/run_benchmark.py $model_name 3 $output_path 
done
