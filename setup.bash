
python3 -m pip install --upgrade pip
apt-get install unrar
pip install -r StressDetector-EEG/requirements.txt

# Download SAM40 data
wget --content-disposition https://figshare.com/ndownloader/files/27956376

# Extract SAM40 data
mkdir -p StressDetector-EEG/data/processed
unrar x Data.rar StressDetector-EEG/data
mv StressDetector-EEG/data/Data StressDetector-EEG/data/origin