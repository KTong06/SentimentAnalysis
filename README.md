![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Spyder](https://img.shields.io/badge/Spyder-838485?style=for-the-badge&logo=spyder%20ide&logoColor=maroon)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)
<a><img alt='love' src="http://ForTheBadge.com/images/badges/built-with-love.svg"></a>

# Good or Bad Review? -A Sentiment Analysis-
Tens of thousands of movie reviews can be scrapped off the Internet in seconds, here is how you can categorise your fresh-off-the-Internet reviews quickly without breaking a sweat! This model is trained with over 50,000 IMDB reviews to categorise positive/negative reviews using LSTM technique. Credits to [Ankit152](https://github.com/Ankit152) for the dataset which can be obtained [here](https://github.com/Ankit152/IMDB-sentiment-analysis).

## Model Accuracy
The model utilises MSE for loss function and accuracy for metrics, and achieved 85% accuracy. Performance of model is summarised below:
![model_val_plot](Static/loss_acc_plot.png)
![model_cm](Static/confusion_matrix.png)

## Model Architecture
![model_architecture](Static/model.png)

| Performance | Confusion Matrix |
| ----------- | ----------- |
| Paragraph | Text |
