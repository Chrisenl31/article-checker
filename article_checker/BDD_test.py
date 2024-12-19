#%%
import time
import unittest

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager


class TestCarinaAbstractChecker(unittest.TestCase):
    def setUp(self):
        """Set up the test environment."""
        # Set up Chrome options for headless execution
        options = Options()
        options.add_argument("--headless")
        options.add_argument("--disable-gpu")
        options.add_argument("--no-sandbox")
        service = Service(ChromeDriverManager().install())
        self.driver = webdriver.Chrome(service=service, options=options)
        self.driver.implicitly_wait(10)
        self.driver.get("http://127.0.0.1:5500/article_checker/checker/templates/checker/check_article.html")  # Update the URL as per your application

    def tearDown(self):
        """Tear down the test environment."""
        self.driver.quit()

    def test_valid_abstract(self):
        """Test submitting a valid abstract and verifying results."""
        driver = self.driver

        # Input the title
        try:
            title_input = WebDriverWait(driver, 20).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, '.form-control-title'))
            )
            title_input.send_keys("Impact of AI on Society")
            print("Title input successful.")
        except Exception as e:
            self.fail(f"Title input element not found: {str(e)}")

        # Input the abstract
        try:
            abstract_input = WebDriverWait(driver, 20).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, '.form-control-abstract'))
            )
            abstract_input.send_keys(
                "The purpose of this paper is to explore the impact of artificial intelligence on society. "
                "It discusses the methods used to study AI's impact, the results of AI integration into various sectors, "
                "and concludes with recommendations for future development."
            )
            print("Abstract input successful.")
        except Exception as e:
            self.fail(f"Abstract input element not found: {str(e)}")

        # Click the submit button
        try:
            submit_button = WebDriverWait(driver, 20).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, '.btn'))
            )
            submit_button.click()
            print("Submit button clicked.")
        except Exception as e:
            self.fail(f"Submit button not found or not clickable: {str(e)}")

        # Wait for the result and validate it
        time.sleep(3)  # Adjust as needed based on your system's response time
        try:
            result_section = driver.find_element(By.CSS_SELECTOR, ".result-section").text
            self.assertIn("Similarity with Database Abstracts", result_section)
            self.assertIn("Extracted Sections", result_section)
            print("Results validated successfully.")

            similarity = driver.find_element(By.CSS_SELECTOR, ".progress-bar").get_attribute("aria-valuenow")
            self.assertGreater(int(similarity), 0)
            print(f"Similarity score: {similarity}%")
        except Exception as e:
            self.fail(f"Result validation failed: {str(e)}")


# Run the tests
if __name__ == "__main__":
    unittest.main()
