# Model Trainer
We apply transfer learning from the sentence-transformer model in Hugging Face called 'sentence-t5-base' due to its optimal accuracy and complexity, as observed from the model leaderboard in the sentence similarity task. We then utilize this model as a cross-encoder model.
<p>The benefits of transfer learning can enhance the accuracy of the model in understanding job titles, particularly those related to the medical field such as 'doctor'.

Before training<br>
<img src='https://drive.google.com/uc?id=1tCfroywA5P-1_sB6EWLBbTr3rB8j2BTM' width="500px">

After training<br>
<img src='https://drive.google.com/uc?id=1BbO97sSnMCVoQ9w40ga9QWRKfZ4UPoC2' width="500px"><br>
<img src='https://drive.google.com/uc?id=1Mek_ZVBH-iO8YCmpHbk5jta8dHEYhCrG' width="500px"></p>
- [STS train dataset](https://docs.google.com/spreadsheets/d/12Y-3dwI43jTe-teXGF3cfKxz1QNefXHQj3v3NJhJBho/edit?usp=sharing). Link to our dataset
- [HuggingFace STS Model Ranking](https://huggingface.co/spaces/mteb/leaderboard). Link to the similarity task model leaderboard.
- [Difference of Bi-Encoder & Cross-Encoder](https://sbert.net/examples/applications/cross-encoder/README.html). Detailed explaination of bi-encoder and cross-encoder
- [STS-trained-lokergo Model](https://huggingface.co/pahri/sts-trained-lokergo). Our trained model was hosted into HuggingFace repository for easy access

C23-VR01 ML Teams.
