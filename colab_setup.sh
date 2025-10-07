
from google.colab import drive
drive.mount('/content/drive')

echo "=== Clone GitHub repo ==="
!git clone https://github.com/xiuxiuli/CSCI_544_NLP_HW1.git /content/drive/MyDrive/CSCI_544_NLP_HW1
%cd /content/drive/MyDrive/CSCI_544_NLP_HW1

echo "=== Install dependencies ==="
pip install -r requirements.txt


%cd /content/drive/MyDrive/CSCI_544_NLP_HW1
echo "=== Run sentiment.py ==="
python sentiment.py