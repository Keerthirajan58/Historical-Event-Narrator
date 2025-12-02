import unittest
import os
import json
from src.data_loader import download_wikipedia_data

class TestDataLoader(unittest.TestCase):
    def setUp(self):
        self.test_output = "tests/test_data.jsonl"
        
    def tearDown(self):
        if os.path.exists(self.test_output):
            os.remove(self.test_output)
            
    def test_download_creates_file(self):
        """Test that the downloader creates a file."""
        # Download just 1 example for speed
        download_wikipedia_data(self.test_output, num_examples=1)
        self.assertTrue(os.path.exists(self.test_output))
        
    def test_file_content(self):
        """Test that the file contains valid JSON."""
        download_wikipedia_data(self.test_output, num_examples=1)
        with open(self.test_output, 'r') as f:
            line = f.readline()
            data = json.loads(line)
            self.assertIn("title", data)
            self.assertIn("text", data)

if __name__ == '__main__':
    unittest.main()
