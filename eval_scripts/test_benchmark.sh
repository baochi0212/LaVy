NUM_GPUS=$5
rm -rf chunk*
file_name=$1
output_file=$4
echo "...."
file_length=$(wc -l < $file_name)
echo "File name: $file_name, length: $file_length"


split -l $(($file_length/($NUM_GPUS-1))) $file_name chunk
i=0
for file in `ls chunk*`; do
    #echo $file "chunk${i}"
    mv $file "chunk${i}"
    echo $file
    let i++
done

for i in $(seq 0 $((NUM_GPUS-1))); do    echo "Chunk $i"
    CUDA_VISIBLE_DEVICES="$i" python eval_benchmark_vi.py --model-path $2 --model-base $3   --file "chunk${i}" --conv-mode "mistral_instruct" --pretrain false &
done
wait

cat output_* >> $output_file && rm -rf output_* chunk*
