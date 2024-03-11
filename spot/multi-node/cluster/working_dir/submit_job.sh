# aws s3 rm --recursive s3://chemberta3/chemberta-test/tr-plain/
ray job submit --address http://localhost:8265 --working-dir . -- python3 train2.py &
