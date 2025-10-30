# Download Apple's document corpus
from sec_edgar_downloader import Downloader

dl = Downloader("Individual", "chetanreddy.reddy1@gmail.com")

# Get comprehensive Apple dataset
dl.get("10-K", "AAPL", limit=3)  # Last 3 years
dl.get("10-Q", "AAPL", limit=8)  # Last 2 years of quarters
dl.get("8-K", "AAPL", limit=10) # Recent material events

# Also get earnings call transcripts (if using external sources)
