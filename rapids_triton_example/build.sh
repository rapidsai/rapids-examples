### dowbload model 
wget https://rapidsai-data.s3.amazonaws.com/community-examples/sentiment_analysis_model/model.pt -P models/sentiment_analysis_model/1/
### add a empty folder for ensemble model 
mkdir -p models/end_to_end_model/1

docker build -t rapidstriton .