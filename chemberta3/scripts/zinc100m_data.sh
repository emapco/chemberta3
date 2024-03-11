total_lines=0
for i in $(seq -f "%03g" 1 2)
do
	path="s3://chemberta3/datasets/zinc20/csv/chunk_$i.csv"
	aws s3 cp $path chunk_$i.csv
	line_count=`wc -l chunk_$i.csv`
	total_lines=$(( total_lines + line_count - 1 ))
done
echo " total lines is $total_lines "
