# !bash colab_setup.sh

#!/bin/bash
echo "===Mount Google Drive ==="
python3 - <<'EOF'
from google.colab import drive
drive.mount('/content/drive')
EOF

echo "=== Clone GitHub repo ==="
if [ ! -d "CSCI_544_NLP_HW1" ]; then
    git clone https://github.com/xiuxiuli/CSCI_544_NLP_HW1.git
fi
cd CSCI_544_NLP_HW1 || exit

echo "=== Install dependencies ==="
pip install -r requirements.txt

echo "=== Run sentiment.py ==="
python sentiment.py