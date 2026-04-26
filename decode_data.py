"""Decode the base64 CSV from Google Sheets and save."""
import base64
import os

# The blob field from the Drive download - base64-encoded CSV
B64_CONTENT = """dGltZXN0YW1wLENPMixUZW1wZXJhdHVyZSxIdW1pZGl0eSxIb3VyLERheSxNb250aCxBUUksQVFJX0NhdGVnb3J5DQowMS0wMi0yMDI1IDA6MDAsNjU2LjkxLDIyLjUzLDY0Ljc1LDAsMSwyLDExNixVbmhlYWx0aHkgU2Vuc2l0aXZl"""

# Just for verification - we'll use the much larger blob
print("Setup ready")
