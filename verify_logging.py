"""Quick script to verify logging output is visible."""

import logging

# Configure logging to see INFO level messages
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(name)s: %(message)s'
)

print("Importing _brain_hdc module...")
print("-" * 60)

# This import should trigger the log message
from packages.puhl_luck.puhl_luck._brain_hdc import RUST_AVAILABLE

print("-" * 60)
print(f"RUST_AVAILABLE = {RUST_AVAILABLE}")
